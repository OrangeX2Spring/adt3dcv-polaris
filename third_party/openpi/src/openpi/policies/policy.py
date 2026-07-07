"""
Subtask-goal V-JEPA verifier (best-of-10 re-rank), for the borderline-IC stabilization A/B.

Differences vs policy_jepa2 (the fixed single-goal verifier):
  - goal_image_path is a ROOT DIR of per-IC subtask goals: <root>/ic<k>/c<j>_<name>.png
    (produced by experiments/jepa_verifier/extract_subtask_goals.py).
  - eval sends "subtask/ic_index" (int) and "subtask/done" (6 bools, c0..c5) per request
    (scripts/eval.py --send-subtask-state). Oracle switching: goals of DONE subtasks are
    excluded; among incomplete ones the ACTIVE goal is the argmin of current-frame energy
    (order-agnostic: pi0.5 sometimes does grapes first).
  - Spread gate: only override candidate 0 when the candidates' relative energy spread
    exceeds VJEPA_SPREAD_GATE (default 0.02). The horizon sweep showed usable signal only
    ~1-2 video frames from subtask completion; everywhere else ranking is noise.
  - Shadow mode: VJEPA_SHADOW=1 -> ALWAYS execute candidate 0 (behaviorally = baseline)
    while logging everything. Run arm A with this to get baseline + spread statistics.

This file IS the active policy.py (the plain pi0.5 wrapper lives in policy_baseline.py).
Serve with:  --goal-image-path /workspace/polaris/runs/goals_subtask
"""
from collections.abc import Sequence
import json
import logging
import os
import pathlib
import time
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
from vjepa2.FK import PandaFK
from vjepa2 import rollout
import torch
from torch.nn import functional as F
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils
import torchvision.transforms as T
import cv2

BasePolicy: TypeAlias = _base_policy.BasePolicy

NUM_SUBTASKS = 6


class Policy(BasePolicy):
    """pi0.5 best-of-10 re-ranked by V-JEPA energy to the ACTIVE subtask goal."""

    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        pytorch_device: str = "cuda:0",
        is_pytorch: bool = False,
        goal_image_path: str | pathlib.Path | None = None,
    ):
        self._model = model
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._is_pytorch_model = is_pytorch
        self._pytorch_device = pytorch_device
        self._vjepa_device = torch.device(
            pytorch_device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._num_candidates = 10
        self._robot = PandaFK(device=str(self._vjepa_device))
        if self._is_pytorch_model:
            self._model = self._model.to(pytorch_device)
            self._model.eval()
            self._sample_actions = model.sample_actions
        else:
            self._sample_actions = nnx_utils.module_jit(model.sample_actions)
            self._rng = rng or jax.random.key(0)

        self._encoder, self._predictor = torch.hub.load(
            "facebookresearch/vjepa2", "vjepa2_ac_vit_giant", pretrained=False
        )
        ckpt = torch.load(
            "/workspace/polaris/third_party/vjepa2/checkpoints/vjepa2-ac-vitg.pt", map_location="cpu"
        )
        strip = lambda sd: {k.replace("module.", "", 1): v for k, v in sd.items()}
        self._encoder.load_state_dict(strip(ckpt["encoder"]))
        self._predictor.load_state_dict(strip(ckpt["predictor"]))
        self._encoder = self._encoder.to(self._vjepa_device).eval()
        self._predictor = self._predictor.to(self._vjepa_device).eval()
        crop_size = 256
        self._tokens_per_frame = int((crop_size // self._encoder.patch_size) ** 2)
        self._transform = T.Compose([
            T.ToPILImage(),
            T.Resize((256, 256)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Per-IC subtask goals: <root>/ic<k>/c<j>_*.png, encoded lazily, cached.
        self._goal_root = pathlib.Path(goal_image_path or "/workspace/polaris/runs/goals_subtask").expanduser()
        if not self._goal_root.is_dir():
            raise FileNotFoundError(
                f"goal root {self._goal_root} is not a directory. Run extract_subtask_goals.py "
                "first and pass its --out via --goal-image-path."
            )
        self._goal_cache: dict[int, list[torch.Tensor | None]] = {}

        self._spread_gate = float(os.environ.get("VJEPA_SPREAD_GATE", 0.02))
        self._shadow = os.environ.get("VJEPA_SHADOW", "0") == "1"
        self._energy_log_path = pathlib.Path(
            os.environ.get(
                "VJEPA_ENERGY_LOG", f"vjepa_subtask_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"
            )
        ).expanduser()
        self._infer_step = 0
        logging.info(
            "Subtask verifier: goals=%s shadow=%s spread_gate=%.3f log=%s",
            self._goal_root, self._shadow, self._spread_gate, self._energy_log_path.resolve(),
        )

    # ---------- goals ----------

    def _encode_bgr(self, bgr: np.ndarray) -> torch.Tensor:
        t = self._transform(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        clip = np.expand_dims(np.stack([t, t], axis=0), axis=0)  # (1,2,3,256,256)
        tensor = torch.from_numpy(clip).float().permute(0, 2, 1, 3, 4).to(self._vjepa_device)
        with torch.inference_mode():
            h = self._encoder(tensor)[:, -self._tokens_per_frame:, :]
            return F.layer_norm(h, (h.size(-1),))

    def _goals_for_ic(self, ic: int) -> list[torch.Tensor | None]:
        if ic not in self._goal_cache:
            goals: list[torch.Tensor | None] = [None] * NUM_SUBTASKS
            icdir = self._goal_root / f"ic{ic}"
            for j in range(NUM_SUBTASKS):
                matches = sorted(icdir.glob(f"c{j}_*.png"))
                if matches:
                    bgr = cv2.imread(str(matches[0]))
                    if bgr is not None:
                        goals[j] = self._encode_bgr(bgr)
            n = sum(g is not None for g in goals)
            logging.info("Loaded %d/%d subtask goals for IC %d from %s", n, NUM_SUBTASKS, ic, icdir)
            if n == 0:
                raise FileNotFoundError(f"No subtask goals found under {icdir}")
            self._goal_cache[ic] = goals
        return self._goal_cache[ic]

    # ---------- inference ----------

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:
        # Subtask state from eval (pop BEFORE transforms so pi0.5's pipeline never sees it).
        ic_index = obs.pop("subtask/ic_index", None)
        done = obs.pop("subtask/done", None)  # list of 6 bools, c0..c5

        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        if not self._is_pytorch_model:
            inputs = jax.tree.map(
                lambda x: jnp.repeat(jnp.asarray(x)[None, ...], self._num_candidates, axis=0),
                inputs,
            )
            self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
        else:
            inputs = jax.tree.map(
                lambda x: (
                    torch.from_numpy(np.asarray(x))
                    .to(self._pytorch_device)
                    .unsqueeze(0)
                    .repeat(self._num_candidates, *([1] * np.asarray(x).ndim))
                ),
                inputs,
            )
            sample_rng_or_pytorch_device = self._pytorch_device

        sample_kwargs = dict(self._sample_kwargs)
        if noise is not None:
            noise = (
                torch.from_numpy(noise).to(self._pytorch_device)
                if self._is_pytorch_model else jnp.asarray(noise)
            )
            if noise.ndim == 2:
                noise = jnp.repeat(noise[None, ...], self._num_candidates, axis=0)
            sample_kwargs["noise"] = noise

        observation = _model.Observation.from_dict(inputs)
        start_time = time.monotonic()
        outputs = {
            "state": inputs["state"],
            "actions": self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs),
        }

        executed_idx = 0
        log: dict[str, Any] = {"step": self._infer_step, "ic": ic_index, "done": done}
        if ic_index is not None:
            goals = self._goals_for_ic(int(ic_index))
            done = list(done) if done is not None else [False] * NUM_SUBTASKS
            incomplete = [j for j in range(NUM_SUBTASKS) if not done[j] and goals[j] is not None]
            if not incomplete:  # everything done (or no goals) -> just run candidate 0
                incomplete = [j for j in range(NUM_SUBTASKS) if goals[j] is not None][-1:]

            # A1 fix: encode the RAW uint8 camera frame, matching the goal path exactly.
            raw_base = np.asarray(obs["observation/exterior_image_1_left"])
            if np.issubdtype(raw_base.dtype, np.floating):
                raw_base = (255 * raw_base).astype(np.uint8)
            if raw_base.shape[0] == 3:
                raw_base = np.transpose(raw_base, (1, 2, 0))
            frame = self._transform(raw_base)
            clip = np.expand_dims(np.stack([frame, frame], axis=0), axis=0)
            frames_tensor = (
                torch.from_numpy(clip).float().permute(0, 2, 1, 3, 4).to(self._vjepa_device)
            )

            actions_joint = outputs["actions"][:, :, :8]
            # A2 fix: 4 steps of ~0.27s each, matching DROID 4fps training granularity.
            actions_joint_downsampled = actions_joint[:, [3, 7, 11, 15], :]
            curr_state = np.array(outputs["state"][:, :8])[:, np.newaxis, :]
            full_traj = np.concatenate([curr_state, np.array(actions_joint_downsampled)], axis=1)
            result = self._robot.convert_trajectory(full_traj)
            ee_actions = torch.from_numpy(result["actions"]).float().to(self._vjepa_device)
            ee_states = torch.from_numpy(result["states"]).float().to(self._vjepa_device)

            with torch.inference_mode():
                z_current = self._encoder(frames_tensor)[:, -self._tokens_per_frame:, :]
                z_current = F.layer_norm(z_current, (z_current.size(-1),))
                # Active goal = incomplete subtask nearest to the CURRENT frame (order-agnostic).
                current_e = {j: F.l1_loss(z_current, goals[j]).item() for j in incomplete}
                active = min(current_e, key=current_e.get)
                # Rank candidates by predicted energy to the active goal.
                z_hat = rollout.forward_actions(z_current, self._predictor, ee_states, ee_actions)
                losses = rollout.loss_fn(z_hat, goals[active])  # list[10]

            best_idx = int(np.argmin(losses))
            spread = (max(losses) - min(losses)) / (sum(losses) / len(losses))
            gated = spread < self._spread_gate
            executed_idx = 0 if (self._shadow or gated) else best_idx
            log.update(
                active_goal=active,
                current_energies={str(j): round(e, 5) for j, e in current_e.items()},
                energies=[round(l, 6) for l in losses],
                spread=round(spread, 5),
                gated=bool(gated),
                best_idx=best_idx,
                executed_idx=executed_idx,
                shadow=self._shadow,
            )
            logging.info(
                "subtask=%d spread=%.4f gated=%s best=%d exec=%d",
                active, spread, gated, best_idx, executed_idx,
            )

        outputs = {
            "state": inputs["state"][executed_idx][None, ...],
            "actions": outputs["actions"][executed_idx][None, ...],
        }
        model_time = time.monotonic() - start_time
        if self._is_pytorch_model:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
        else:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)

        with self._energy_log_path.open("a") as f:
            f.write(json.dumps(log) + "\n")
        self._infer_step += 1

        outputs = self._output_transform(outputs)
        outputs["policy_timing"] = {"infer_ms": model_time * 1000}
        return outputs

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy
        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)
        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")
        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1
        np.save(output_path, np.asarray(data))
        return results
