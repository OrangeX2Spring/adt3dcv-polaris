"""
Test whether a V-JEPA2-AC predictor assigns lower future-latent error to the
recorded expert action than to zero or shuffled in-distribution actions.

This is a teacher-forced diagnostic: every prediction receives ground-truth
visual context and robot state from the packed training trajectory. It does not
use the planner, goal images, candidate actions from pi0.5, or Isaac Sim.

Run from any directory on the training machine:

  /workspace/polaris/.venv/bin/python \
    /workspace/polaris/experiments/jepa_verifier/teacher_forced_action_test.py \
    --checkpoint base=/workspace/polaris/third_party/vjepa2/checkpoints/vjepa2-ac-vitg.pt \
    --checkpoint e20=/workspace/polaris/runs/vjepa_foodbussing/e20.pt \
    --output /workspace/polaris/runs/vjepa_action_test
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml
from decord import VideoReader, cpu
from scipy.spatial.transform import Rotation
from torch.nn import functional as F


REPO_ROOT = Path(__file__).resolve().parents[2]
VJEPA_ROOT = REPO_ROOT / "third_party" / "vjepa2"
sys.path.insert(0, str(VJEPA_ROOT))

from app.vjepa_droid.transforms import make_transforms  # noqa: E402
from app.vjepa_droid.utils import init_video_model  # noqa: E402


@dataclass
class Window:
    episode_path: Path
    episode_id: str
    ic_index: int
    video_path: Path
    start: int
    indices: np.ndarray
    states: np.ndarray
    actions: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Teacher-forced recorded-action versus shuffled-action test."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=VJEPA_ROOT / "configs/train/vitg16/foodbussing-256px-8f.yaml",
    )
    parser.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="Checkpoint to evaluate. Repeat this option to compare checkpoints.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Override data.datasets[0] from the training config.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "runs/vjepa_action_test",
    )
    parser.add_argument("--windows-per-episode", type=int, default=1)
    parser.add_argument("--negatives", type=int, default=9)
    parser.add_argument("--candidate-batch-size", type=int, default=1)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=239)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    return parser.parse_args()


def parse_checkpoints(values: list[str]) -> list[tuple[str, Path]]:
    checkpoints = []
    labels = set()
    for value in values:
        if "=" in value:
            label, path = value.split("=", 1)
        else:
            path = value
            label = Path(path).stem
        if not label or label in labels:
            raise ValueError(f"Checkpoint labels must be unique and non-empty: {label!r}")
        checkpoint = Path(path)
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        labels.add(label)
        checkpoints.append((label, checkpoint))
    return checkpoints


def load_config(path: Path) -> dict:
    with path.open() as handle:
        return yaml.safe_load(handle)


def load_episode_paths(dataset_path: Path, max_episodes: int | None) -> list[Path]:
    with dataset_path.open() as handle:
        paths = [Path(line.strip().split()[0]) for line in handle if line.strip()]
    if max_episodes is not None:
        paths = paths[:max_episodes]
    if len(paths) < 2:
        raise ValueError("At least two episodes are required for cross-episode negatives.")
    return paths


def poses_to_diffs(poses: np.ndarray) -> np.ndarray:
    xyz_diff = poses[1:, :3] - poses[:-1, :3]
    rotations = Rotation.from_euler("xyz", poses[:, 3:6], degrees=False).as_matrix()
    rotation_diff = rotations[1:] @ np.swapaxes(rotations[:-1], 1, 2)
    angle_diff = Rotation.from_matrix(rotation_diff).as_euler("xyz", degrees=False)
    gripper_diff = poses[1:, 6:7] - poses[:-1, 6:7]
    return np.concatenate((xyz_diff, angle_diff, gripper_diff), axis=1).astype(np.float32)


def load_states(episode_path: Path) -> np.ndarray:
    with h5py.File(episode_path / "trajectory.h5", "r") as trajectory:
        cartesian = np.asarray(
            trajectory["observation"]["robot_state"]["cartesian_position"]
        )
        gripper = np.asarray(
            trajectory["observation"]["robot_state"]["gripper_position"]
        )[:, None]
    states = np.concatenate((cartesian, gripper), axis=1).astype(np.float32)
    if states.ndim != 2 or states.shape[1] != 7 or not np.isfinite(states).all():
        raise ValueError(f"Invalid states in {episode_path}: {states.shape}")
    return states


def select_windows(
    episode_paths: list[Path],
    data_config: dict,
    windows_per_episode: int,
    seed: int,
) -> list[Window]:
    if windows_per_episode < 1:
        raise ValueError("--windows-per-episode must be positive.")

    frames_per_clip = max(data_config["dataset_fpcs"])
    requested_fps = data_config["fps"]
    camera_view = data_config["camera_views"][0]
    rng = np.random.default_rng(seed)
    windows = []

    for episode_path in episode_paths:
        metadata_path = episode_path / "metadata.json"
        metadata = json.loads(metadata_path.read_text())
        video_path = episode_path / metadata[camera_view]
        states = load_states(episode_path)

        reader = VideoReader(str(video_path), num_threads=1, ctx=cpu(0))
        video_length = len(reader)
        video_fps = float(reader.get_avg_fps())
        del reader

        if video_length != len(states):
            raise ValueError(
                f"Video/state length mismatch in {episode_path}: "
                f"{video_length} video frames, {len(states)} states"
            )

        frame_step = math.ceil(video_fps / requested_fps)
        sampled_span = frames_per_clip * frame_step
        valid_start_count = video_length - sampled_span
        if valid_start_count < 1:
            raise ValueError(
                f"Episode {episode_path} is too short for the training sampler: "
                f"{video_length} <= {sampled_span}"
            )

        count = min(windows_per_episode, valid_start_count)
        starts = np.sort(rng.choice(valid_start_count, size=count, replace=False))
        for start in starts:
            indices = start + np.arange(0, sampled_span, frame_step, dtype=np.int64)
            window_states = states[indices]
            windows.append(
                Window(
                    episode_path=episode_path,
                    episode_id=metadata.get("source_episode", episode_path.name),
                    ic_index=int(metadata["ic_index"]),
                    video_path=video_path,
                    start=int(start),
                    indices=indices,
                    states=window_states,
                    actions=poses_to_diffs(window_states),
                )
            )

    return windows


def select_negative_windows(
    windows: list[Window], negatives: int, seed: int
) -> list[list[int]]:
    if negatives < 1:
        raise ValueError("--negatives must be positive.")

    rng = np.random.default_rng(seed + 1)
    selections = []
    for window in windows:
        pool = [
            index
            for index, candidate in enumerate(windows)
            if candidate.episode_id != window.episode_id
        ]
        if not pool:
            raise ValueError(f"No cross-episode negatives for {window.episode_id}.")
        chosen = rng.choice(pool, size=negatives, replace=len(pool) < negatives)
        selections.append([int(index) for index in chosen])
    return selections


def build_transform(config: dict):
    data = config["data"]
    augmentation = config["data_aug"]
    return make_transforms(
        random_horizontal_flip=augmentation["horizontal_flip"],
        random_resize_aspect_ratio=augmentation["random_resize_aspect_ratio"],
        random_resize_scale=augmentation["random_resize_scale"],
        reprob=augmentation["reprob"],
        auto_augment=augmentation["auto_augment"],
        motion_shift=augmentation["motion_shift"],
        crop_size=data["crop_size"],
    )


def build_models(config: dict) -> tuple[torch.nn.Module, torch.nn.Module]:
    data = config["data"]
    meta = config["meta"]
    model = config["model"]
    encoder, predictor = init_video_model(
        uniform_power=model["uniform_power"],
        device=torch.device("cpu"),
        patch_size=data["patch_size"],
        max_num_frames=64,
        tubelet_size=data["tubelet_size"],
        model_name=model["model_name"],
        crop_size=data["crop_size"],
        pred_depth=model["pred_depth"],
        pred_num_heads=model["pred_num_heads"],
        pred_embed_dim=model["pred_embed_dim"],
        action_embed_dim=7,
        pred_is_frame_causal=model["pred_is_frame_causal"],
        use_extrinsics=model["use_extrinsics"],
        use_sdpa=meta["use_sdpa"],
        use_silu=model.get("use_silu", False),
        use_pred_silu=model.get("use_pred_silu", False),
        wide_silu=model.get("wide_silu", True),
        use_rope=model["use_rope"],
        use_activation_checkpointing=False,
    )
    return encoder.eval(), predictor.eval()


def clean_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    cleaned = {}
    for original_key, value in state_dict.items():
        key = original_key
        changed = True
        while changed:
            changed = False
            for prefix in ("module.", "backbone."):
                if key.startswith(prefix):
                    key = key[len(prefix) :]
                    changed = True
        if key in cleaned:
            raise ValueError(f"Duplicate checkpoint key after prefix removal: {key}")
        cleaned[key] = value
    return cleaned


def load_checkpoint(
    checkpoint_path: Path,
    encoder: torch.nn.Module,
    predictor: torch.nn.Module,
) -> dict:
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        mmap=True,
        weights_only=True,
    )
    encoder.load_state_dict(clean_state_dict(checkpoint["encoder"]), strict=True)
    predictor.load_state_dict(clean_state_dict(checkpoint["predictor"]), strict=True)
    metadata = {
        "epoch": int(checkpoint["epoch"]),
        "training_loss": float(checkpoint["loss"]) if "loss" in checkpoint else None,
    }
    del checkpoint
    gc.collect()
    return metadata


def encode_window(
    frames: np.ndarray,
    transform,
    encoder: torch.nn.Module,
    device: torch.device,
) -> torch.Tensor:
    clip = transform(frames).unsqueeze(0).to(device, non_blocking=True)
    batch_size, _, num_frames, _, _ = clip.shape
    encoder_input = (
        clip.permute(0, 2, 1, 3, 4)
        .flatten(0, 1)
        .unsqueeze(2)
        .repeat(1, 1, 2, 1, 1)
    )
    representations = encoder(encoder_input)
    representations = representations.view(
        batch_size, num_frames, -1, representations.size(-1)
    ).flatten(1, 2)
    return F.layer_norm(representations, (representations.size(-1),))


def predict(
    predictor: torch.nn.Module,
    context: torch.Tensor,
    states: torch.Tensor,
    actions: torch.Tensor,
) -> torch.Tensor:
    batch_size = actions.size(0)
    predictions = predictor(
        context.expand(batch_size, -1, -1),
        actions,
        states.expand(batch_size, -1, -1),
    )
    return F.layer_norm(predictions, (predictions.size(-1),))


def frame_l1(
    left: torch.Tensor,
    right: torch.Tensor,
    transitions: int,
    tokens_per_frame: int,
) -> torch.Tensor:
    left = left.reshape(left.size(0), transitions, tokens_per_frame, -1)
    right = right.reshape(right.size(0), transitions, tokens_per_frame, -1)
    return (left - right).abs().float().mean(dim=(2, 3))


def evaluate_window(
    window: Window,
    negative_indices: list[int],
    windows: list[Window],
    transform,
    encoder: torch.nn.Module,
    predictor: torch.nn.Module,
    device: torch.device,
    candidate_batch_size: int,
) -> list[dict]:
    reader = VideoReader(str(window.video_path), num_threads=2, ctx=cpu(0))
    frames = reader.get_batch(window.indices).asnumpy()
    del reader

    representations = encode_window(frames, transform, encoder, device)
    num_frames = len(window.indices)
    transitions = num_frames - 1
    tokens_per_frame = representations.size(1) // num_frames
    frame_representations = representations.reshape(
        1, num_frames, tokens_per_frame, -1
    )
    context = representations[:, :-tokens_per_frame]
    target = representations[:, tokens_per_frame:]
    states = torch.from_numpy(window.states[:-1]).to(device).unsqueeze(0)
    true_actions = torch.from_numpy(window.actions).to(device).unsqueeze(0)

    true_prediction = predict(predictor, context, states, true_actions)
    true_errors = frame_l1(
        true_prediction, target, transitions, tokens_per_frame
    )[0]
    actual_change = (
        (frame_representations[:, 1:] - frame_representations[:, :-1])
        .abs()
        .float()
        .mean(dim=(2, 3))[0]
    )

    candidate_arrays = [np.zeros_like(window.actions)]
    candidate_arrays.extend(windows[index].actions for index in negative_indices)
    candidate_errors = []
    candidate_sensitivities = []

    for start in range(0, len(candidate_arrays), candidate_batch_size):
        action_batch = torch.from_numpy(
            np.stack(candidate_arrays[start : start + candidate_batch_size])
        ).to(device)
        candidate_prediction = predict(predictor, context, states, action_batch)
        candidate_errors.append(
            frame_l1(
                candidate_prediction,
                target.expand(candidate_prediction.size(0), -1, -1),
                transitions,
                tokens_per_frame,
            )
        )
        candidate_sensitivities.append(
            frame_l1(
                candidate_prediction,
                true_prediction.expand(candidate_prediction.size(0), -1, -1),
                transitions,
                tokens_per_frame,
            )
        )

    candidate_errors = torch.cat(candidate_errors).cpu().numpy()
    candidate_sensitivities = torch.cat(candidate_sensitivities).cpu().numpy()
    true_errors = true_errors.cpu().numpy()
    actual_change = actual_change.cpu().numpy()

    rows = []
    for transition in range(transitions):
        shuffled_errors = candidate_errors[1:, transition]
        shuffled_sensitivities = candidate_sensitivities[1:, transition]
        true_error = float(true_errors[transition])
        better = int(np.count_nonzero(shuffled_errors < true_error))
        tied = int(np.count_nonzero(shuffled_errors == true_error))
        rank = better + 1.0 + tied / 2.0
        rows.append(
            {
                "episode": window.episode_id,
                "ic_index": window.ic_index,
                "window_start": window.start,
                "transition": transition,
                "frame_index": int(window.indices[transition]),
                "true_error": true_error,
                "zero_error": float(candidate_errors[0, transition]),
                "mean_shuffled_error": float(shuffled_errors.mean()),
                "margin": float(shuffled_errors.mean() - true_error),
                "true_rank": rank,
                "actual_latent_change": float(actual_change[transition]),
                "zero_sensitivity": float(candidate_sensitivities[0, transition]),
                "mean_shuffled_sensitivity": float(
                    shuffled_sensitivities.mean()
                ),
                "negative_errors": shuffled_errors.tolist(),
                "negative_sensitivities": shuffled_sensitivities.tolist(),
                "negative_episodes": [
                    windows[index].episode_id for index in negative_indices
                ],
            }
        )
    return rows


def episode_bootstrap_interval(
    rows: list[dict], row_scores: np.ndarray, seed: int
) -> list[float]:
    episode_scores = {}
    for row, score in zip(rows, row_scores, strict=True):
        episode_scores.setdefault(row["episode"], []).append(float(score))
    values = np.asarray(
        [np.mean(scores) for scores in episode_scores.values()], dtype=np.float64
    )
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(2000, dtype=np.float64)
    for index in range(len(bootstrap)):
        bootstrap[index] = rng.choice(values, size=len(values), replace=True).mean()
    return np.quantile(bootstrap, (0.025, 0.975)).tolist()


def summarize(rows: list[dict], negatives: int, seed: int) -> dict:
    true_errors = np.asarray([row["true_error"] for row in rows])
    zero_errors = np.asarray([row["zero_error"] for row in rows])
    shuffled_errors = np.asarray([row["negative_errors"] for row in rows])
    shuffled_sensitivity = np.asarray(
        [row["negative_sensitivities"] for row in rows]
    )
    actual_change = np.asarray([row["actual_latent_change"] for row in rows])

    pair_scores = (true_errors[:, None] < shuffled_errors).astype(np.float64)
    pair_scores += 0.5 * (true_errors[:, None] == shuffled_errors)
    per_row_auc = pair_scores.mean(axis=1)
    better = (shuffled_errors < true_errors[:, None]).sum(axis=1)
    tied_with_true = 1 + (shuffled_errors == true_errors[:, None]).sum(axis=1)
    ranks = better + (tied_with_true + 1) / 2.0
    top1 = np.clip((1 - better) / tied_with_true, 0.0, 1.0)
    top3 = np.clip((3 - better) / tied_with_true, 0.0, 1.0)
    reciprocal_ranks = np.asarray(
        [
            np.mean(
                1.0
                / np.arange(
                    better_count + 1,
                    better_count + tied_count + 1,
                    dtype=np.float64,
                )
            )
            for better_count, tied_count in zip(
                better, tied_with_true, strict=True
            )
        ]
    )
    zero_scores = (true_errors < zero_errors).astype(np.float64)
    zero_scores += 0.5 * (true_errors == zero_errors)

    candidate_count = negatives + 1
    return {
        "rows": len(rows),
        "episodes": len({row["episode"] for row in rows}),
        "pairwise_auc": float(per_row_auc.mean()),
        "pairwise_auc_episode_bootstrap_95ci": episode_bootstrap_interval(
            rows, per_row_auc, seed
        ),
        "top1": float(top1.mean()),
        "top3": float(top3.mean()),
        "mean_rank": float(ranks.mean()),
        "mrr": float(reciprocal_ranks.mean()),
        "true_beats_zero": float(zero_scores.mean()),
        "mean_true_error": float(true_errors.mean()),
        "mean_zero_error": float(zero_errors.mean()),
        "mean_shuffled_error": float(shuffled_errors.mean()),
        "mean_margin": float((shuffled_errors.mean(axis=1) - true_errors).mean()),
        "mean_actual_latent_change": float(actual_change.mean()),
        "mean_shuffled_sensitivity": float(shuffled_sensitivity.mean()),
        "sensitivity_to_actual_change": float(
            shuffled_sensitivity.mean() / actual_change.mean()
        ),
        "random_baseline": {
            "pairwise_auc": 0.5,
            "top1": 1.0 / candidate_count,
            "top3": min(3, candidate_count) / candidate_count,
            "mean_rank": (candidate_count + 1) / 2.0,
        },
    }


def write_rows(path: Path, rows: list[dict], negatives: int) -> None:
    scalar_fields = [
        "episode",
        "ic_index",
        "window_start",
        "transition",
        "frame_index",
        "true_error",
        "zero_error",
        "mean_shuffled_error",
        "margin",
        "true_rank",
        "actual_latent_change",
        "zero_sensitivity",
        "mean_shuffled_sensitivity",
    ]
    negative_fields = []
    for index in range(negatives):
        negative_fields.extend(
            (
                f"negative_{index}_episode",
                f"negative_{index}_error",
                f"negative_{index}_sensitivity",
            )
        )

    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=scalar_fields + negative_fields)
        writer.writeheader()
        for row in rows:
            output = {field: row[field] for field in scalar_fields}
            for index in range(negatives):
                output[f"negative_{index}_episode"] = row["negative_episodes"][
                    index
                ]
                output[f"negative_{index}_error"] = row["negative_errors"][index]
                output[f"negative_{index}_sensitivity"] = row[
                    "negative_sensitivities"
                ][index]
            writer.writerow(output)


def print_summary(label: str, scope: str, summary: dict) -> None:
    low, high = summary["pairwise_auc_episode_bootstrap_95ci"]
    print(
        f"[{label} {scope}] n={summary['rows']} "
        f"AUC={summary['pairwise_auc']:.3f} [{low:.3f}, {high:.3f}] "
        f"top1={summary['top1']:.3f} mean-rank={summary['mean_rank']:.2f} "
        f"true={summary['mean_true_error']:.4f} "
        f"shuffled={summary['mean_shuffled_error']:.4f} "
        f"sensitivity/change={summary['sensitivity_to_actual_change']:.3f}"
    )


def main() -> None:
    args = parse_args()
    if args.candidate_batch_size < 1:
        raise ValueError("--candidate-batch-size must be positive.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    config = load_config(args.config)
    dataset_path = args.dataset or Path(config["data"]["datasets"][0])
    episode_paths = load_episode_paths(dataset_path, args.max_episodes)
    windows = select_windows(
        episode_paths,
        config["data"],
        args.windows_per_episode,
        args.seed,
    )
    negative_windows = select_negative_windows(windows, args.negatives, args.seed)
    checkpoints = parse_checkpoints(args.checkpoint)
    transform = build_transform(config)
    device = torch.device(args.device)
    if device.type != "cuda" and args.dtype == "bfloat16":
        raise ValueError("bfloat16 evaluation requires a CUDA device.")

    print(
        f"Selected {len(windows)} windows from {len(episode_paths)} episodes; "
        f"{args.negatives} cross-episode negatives per window."
    )
    encoder, predictor = build_models(config)

    for checkpoint_index, (label, checkpoint_path) in enumerate(checkpoints):
        checkpoint_metadata = load_checkpoint(checkpoint_path, encoder, predictor)
        if checkpoint_index == 0:
            encoder = encoder.to(device).eval()
            predictor = predictor.to(device).eval()
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)

        rows = []
        started = time.monotonic()
        autocast_enabled = device.type == "cuda" and args.dtype == "bfloat16"
        with torch.inference_mode(), torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=autocast_enabled,
        ):
            for index, window in enumerate(windows):
                rows.extend(
                    evaluate_window(
                        window,
                        negative_windows[index],
                        windows,
                        transform,
                        encoder,
                        predictor,
                        device,
                        args.candidate_batch_size,
                    )
                )
                print(
                    f"[{label}] {index + 1}/{len(windows)} "
                    f"{window.episode_id} IC={window.ic_index}"
                )

        first_transition_rows = [
            row for row in rows if row["transition"] == 0
        ]
        summaries = {
            "first_transition": summarize(
                first_transition_rows, args.negatives, args.seed
            ),
            "all_transitions": summarize(rows, args.negatives, args.seed),
        }
        elapsed = time.monotonic() - started
        peak_memory_gib = (
            torch.cuda.max_memory_allocated(device) / 1024**3
            if device.type == "cuda"
            else None
        )

        output_dir = args.output / label
        output_dir.mkdir(parents=True, exist_ok=True)
        write_rows(output_dir / "transitions.csv", rows, args.negatives)
        report = {
            "checkpoint": str(checkpoint_path),
            "checkpoint_metadata": checkpoint_metadata,
            "config": str(args.config),
            "dataset": str(dataset_path),
            "seed": args.seed,
            "windows_per_episode": args.windows_per_episode,
            "negative_count": args.negatives,
            "negative_sampling": "recorded action chunks from different episodes",
            "fixed_inputs": "ground-truth visual context and robot states",
            "preprocessing": "training config",
            "dtype": args.dtype,
            "elapsed_seconds": elapsed,
            "peak_cuda_memory_gib": peak_memory_gib,
            "summaries": summaries,
        }
        (output_dir / "summary.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )

        print_summary(label, "first", summaries["first_transition"])
        print_summary(label, "all", summaries["all_transitions"])
        print(f"[{label}] wrote {output_dir}")


if __name__ == "__main__":
    main()
