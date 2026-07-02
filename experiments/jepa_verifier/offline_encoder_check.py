"""
Offline go/no-go: does V-JEPA2-AC's latent-L1-to-goal track task progress on the
splat-rendered PolaRiS domain — using ONLY the encoder (no predictor, no candidate
sampling)?

For each eval video (`episode_N.mp4`, whose left half is the base camera, matching the
observation the policy sees), encode every frame and measure L1 distance to the goal-image
embedding, using the *exact* preprocessing/encoding as policy.py. A working goal-distance
signal must trend DOWN as a high-progress episode approaches the goal. If it stays flat or
trends up, raw latent-L1-to-goal is not a usable verifier signal on this domain and the
sample+rerank line needs a different signal (e.g. fine-tuned V-JEPA, foreground masking, or a
trained scoring head) rather than more tuning.

Run inside the openpi venv on the eval machine, e.g.:
  cd /workspace/polaris/third_party/openpi
  uv run python /workspace/polaris/experiments/jepa_verifier/offline_encoder_check.py \
      --videos-glob "/workspace/polaris/runs/<run>/episode_*.mp4" \
      --goal /workspace/polaris/<...>/episode_0003_success_external_cam.png \
      --ckpt /workspace/polaris/third_party/vjepa2/checkpoints/vjepa2-ac-vitg.pt \
      --out /workspace/polaris/experiments/jepa_verifier/offline_encoder_check
Then copy the produced .png + .csv back to the Mac.
"""
import argparse
import glob
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
    """frame_rgb_uint8: (H,W,3) uint8 RGB. Returns layer-normed (1, tokens, D)."""
    t = transform(frame_rgb_uint8)                       # (3,256,256)
    stacked = np.stack([t, t], axis=0)                   # (2,3,256,256) tubelet=2
    stacked = np.expand_dims(stacked, axis=0)            # (1,2,3,256,256)
    tensor = torch.from_numpy(stacked).float().permute(0, 2, 1, 3, 4).to(device)
    with torch.inference_mode():
        h = encoder(tensor)[:, -tokens_per_frame:, :]
        return F.layer_norm(h, (h.size(-1),))


def read_video_frames(path):
    """Return list of RGB uint8 frames (base-cam = left half of the base|wrist viz)."""
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        w = rgb.shape[1]
        base = rgb[:, : w // 2, :]                        # left half = exterior cam
        frames.append(np.ascontiguousarray(base))
    cap.release()
    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos-glob", required=True)
    ap.add_argument("--goal", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="offline_encoder_check")
    ap.add_argument("--stride", type=int, default=3, help="encode every k-th frame")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder, _ = torch.hub.load("facebookresearch/vjepa2", "vjepa2_ac_vit_giant", pretrained=False)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    encoder.load_state_dict({k.replace("module.", "", 1): v for k, v in ckpt["encoder"].items()})
    encoder = encoder.to(device).eval()
    tokens_per_frame = int((256 // encoder.patch_size) ** 2)
    transform = build_transform()

    goal_bgr = cv2.imread(args.goal)
    if goal_bgr is None:
        raise FileNotFoundError(f"goal image not found: {args.goal}")
    z_goal = encode_frame(encoder, transform, cv2.cvtColor(goal_bgr, cv2.COLOR_BGR2RGB),
                          tokens_per_frame, device)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    videos = sorted(glob.glob(args.videos_glob))
    print(f"{len(videos)} videos; goal={args.goal}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    rows = []
    for vp in videos:
        frames = read_video_frames(vp)[:: args.stride]
        zs = [encode_frame(encoder, transform, f, tokens_per_frame, device) for f in frames]
        name = Path(vp).stem

        # (1) energy to the EXTERNAL goal image (goal-render path)
        e_goal = np.array([F.l1_loss(z, z_goal).item() for z in zs])
        # (2) energy to THIS video's OWN last frame (same obs-render path as all frames).
        #     Isolates the goal-vs-obs render-path gap: if the encoder separates task states,
        #     this must fall smoothly toward ~0 near the end; last point is 0 by construction.
        z_self = zs[-1]
        e_self = np.array([F.l1_loss(z, z_self).item() for z in zs])

        xs = np.linspace(0, 1, len(zs))
        ax.plot(xs, e_goal, marker=".", ms=3, lw=1, label=name)
        ax2.plot(xs, e_self, marker=".", ms=3, lw=1, label=name)

        m = len(e_goal)
        trend = e_goal[-max(1, m // 3):].mean() - e_goal[: max(1, m // 3)].mean()
        # self-goal: how much does the first frame differ from the last (same render path)?
        self_range = e_self[0] - e_self[-1]  # = e_self[0], since e_self[-1]=0
        rows.append((name, len(zs), e_goal[0], e_goal[-1], e_goal.min(), trend, e_self[0]))
        print(f"{name}: n={len(zs)} E0={e_goal[0]:.4f} Elast={e_goal[-1]:.4f} "
              f"Emin={e_goal.min():.4f} trend={trend:+.4f} | self-goal first-frame dist={e_self[0]:.4f}")

    ax.set_xlabel("normalized episode time")
    ax.set_ylabel("encoder L1 to EXTERNAL goal image")
    ax.set_title("(1) vs external goal (goal-render path)")
    ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3)
    ax2.set_xlabel("normalized episode time")
    ax2.set_ylabel("encoder L1 to this video's OWN last frame")
    ax2.set_title("(2) vs own last frame (same obs-render path)")
    ax2.legend(fontsize=7, ncol=2); ax2.grid(alpha=0.3)
    fig.suptitle("Go/no-go: does latent-L1-to-goal track progress? "
                 "(1) flat but (2) descending ⇒ render-path gap; both flat ⇒ encoder no signal")
    fig.tight_layout()
    fig.savefig(f"{args.out}.png", dpi=160)

    with open(f"{args.out}.csv", "w") as fcsv:
        fcsv.write("episode,n_frames,E_first,E_last,E_min,trend_last_minus_first,self_goal_first_frame_dist\n")
        for r in rows:
            fcsv.write(f"{r[0]},{r[1]},{r[2]:.6f},{r[3]:.6f},{r[4]:.6f},{r[5]:.6f},{r[6]:.6f}\n")
    n_down = sum(1 for r in rows if r[5] < 0)
    print(f"\nsaved {args.out}.png / .csv")
    print(f"episodes with downward external-goal trend: {n_down}/{len(rows)}")
    print("Interpretation: if self-goal first-frame dist >> external-goal spread, the encoder DOES "
          "separate states and the flat external-goal curve is a goal/obs render-path gap "
          "(fixable). If self-goal dist is also tiny, the encoder carries no task signal here.")


if __name__ == "__main__":
    main()
