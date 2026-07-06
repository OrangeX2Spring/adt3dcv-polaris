"""
Horizon sweep on ONE episode: within how many frames ahead is V-JEPA's latent-L1-to-goal
actually usable (i.e. descends toward the goal)? Tests the tutor's "only useful within a few
dozen frames" claim ON OUR splat domain, offline, no rollout.

Goals = each checker's COMPLETION frame (the moment a subtask finishes), read from the per-step
log that scripts/eval.py writes when POLARIS_STEP_LOG=1 (episode_<k>_steps.jsonl). For each such
goal g we plot distance-to-g against "frames before completion" (g - t). If the curve only dips
in the last H frames and is flat before that, H is the usable horizon; a subtask verifier goal
must live inside it.

Also saves each subtask's completion frame as subtask_c<k>_<name>.png -> ready-made short-horizon
goals for the next experiment.

Run in the openpi venv on the eval box:
  cd /workspace/polaris/third_party/openpi
  uv run python /workspace/polaris/experiments/jepa_verifier/horizon_sweep.py \
      --videos-dir /workspace/polaris/runs/ep3_repeat --episode 0 \
      --steps /workspace/polaris/runs/ep3_repeat/episode_0_steps.jsonl \
      --ckpt /workspace/polaris/third_party/vjepa2/checkpoints/vjepa2-ac-vitg.pt \
      --out /workspace/polaris/experiments/jepa_verifier/figs/ep3_horizon
"""
import argparse
import json
import re
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.nn import functional as F
import torchvision.transforms as T


def build_transform():
    return T.Compose([
        T.ToPILImage(),
        T.Resize((256, 256)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def encode_frame(encoder, transform, frame_rgb_uint8, tokens_per_frame, device):
    t = transform(frame_rgb_uint8)
    stacked = np.expand_dims(np.stack([t, t], axis=0), axis=0)   # (1,2,3,256,256)
    tensor = torch.from_numpy(stacked).float().permute(0, 2, 1, 3, 4).to(device)
    with torch.inference_mode():
        h = encoder(tensor)[:, -tokens_per_frame:, :]
        return F.layer_norm(h, (h.size(-1),))


def read_base_frames(path):
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        frames.append(np.ascontiguousarray(rgb[:, : rgb.shape[1] // 2, :]))
    cap.release()
    return frames


def subtask_goals_from_steps(steps_path):
    """Return ordered list of (stage_key, name, completion_video_frame)."""
    records = [json.loads(l) for l in Path(steps_path).read_text().splitlines() if l.strip()]
    ever_keys = [k for k in records[0] if k.endswith("_ever")]
    # order by the c<idx> in the key name
    def cidx(k):
        m = re.search(r"c(\d+)", k)
        return int(m.group(1)) if m else 999
    ever_keys.sort(key=cidx)
    goals = []
    for k in ever_keys:
        first = next((r for r in records if r.get(k)), None)
        if first is None:
            continue  # subtask never completed in this rollout
        name = re.sub(r"^c\d+_", "", k[:-len("_ever")])
        goals.append((k, name, int(first["frame"])))
    return goals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos-dir", required=True)
    ap.add_argument("--episode", type=int, required=True)
    ap.add_argument("--steps", required=True, help="episode_<k>_steps.jsonl from POLARIS_STEP_LOG=1")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--stride", type=int, default=1, help="1 = exact frame alignment (recommended)")
    ap.add_argument("--out", default="ep3_horizon")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder, _ = torch.hub.load("facebookresearch/vjepa2", "vjepa2_ac_vit_giant", pretrained=False)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    encoder.load_state_dict({k.replace("module.", "", 1): v for k, v in ckpt["encoder"].items()})
    encoder = encoder.to(device).eval()
    tokens = int((256 // encoder.patch_size) ** 2)
    transform = build_transform()

    frames = read_base_frames(Path(args.videos_dir) / f"episode_{args.episode}.mp4")
    idxs = list(range(0, len(frames), args.stride))
    zs = [encode_frame(encoder, transform, frames[i], tokens, device) for i in idxs]
    orig2strided = {o: s for s, o in enumerate(idxs)}

    goals = subtask_goals_from_steps(args.steps)
    print(f"{len(frames)} frames; subtask completion frames: "
          f"{[(n, f) for _, n, f in goals]}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 12})
    fig, ax = plt.subplots(figsize=(9, 5.5))
    colors = plt.cm.viridis(np.linspace(0, 0.85, max(1, len(goals))))

    print("\nframes-before-completion -> distance   (per subtask)")
    for (key, name, gframe), c in zip(goals, colors):
        g = orig2strided.get(gframe) or min(range(len(idxs)), key=lambda s: abs(idxs[s] - gframe))
        z_goal = zs[g]
        d = np.array([F.l1_loss(zs[t], z_goal).item() for t in range(g + 1)])
        offsets = (np.array([idxs[t] for t in range(g + 1)]) - gframe)  # <=0, in original frames
        ax.plot(-offsets, d, marker=".", lw=2, color=c, label=f"c: {name} @f{gframe}")
        # save the goal frame for reuse as a short-horizon subtask goal
        cv2.imwrite(str(out.parent / f"subtask_{key[:-len('_ever')]}.png"),
                    cv2.cvtColor(frames[gframe], cv2.COLOR_RGB2BGR))
        tbl = "  ".join(f"{o:>2}:{d[max(0, g - o)]:.3f}" for o in [1, 2, 4, 8, 16, 32] if o <= g)
        print(f"  {name:<22} d@goal={d[-1]:.3f}  {tbl}")

    ax.invert_xaxis()  # 0 (completion) on the right, far past on the left
    ax.set_xlabel("frames before subtask completion  (0 = the moment it finishes)")
    ax.set_ylabel("latent L1 distance to that subtask's completion frame")
    ax.set_title("Usable horizon of V-JEPA latent-goal energy (per subtask)\n"
                 "descent confined to the last few frames = short effective horizon")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{args.out}.png", dpi=160)
    print(f"\nsaved {args.out}.png and subtask goal frames in {out.parent}")


if __name__ == "__main__":
    main()
