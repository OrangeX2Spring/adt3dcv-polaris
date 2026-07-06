"""
v3 plot, decluttered: does episode-3's goal image transfer to OTHER episodes?

Uses ONE shared goal (episode 3's success frame) and plots the per-frame latent-L1
distance-to-goal for a SMALL, hand-picked set of episodes so the tutor can actually read it:
  - 3 SUCCESSFUL episodes (task completed), EXCLUDING episode 3 itself   -> solid lines
  - 3 FAILED episodes                                                    -> dashed lines

Expectation if ep3's goal only works for ep3: even the *successful* episodes do NOT descend
toward ep3's goal (their bowls sit at different positions), i.e. all 6 curves stay flat/high.

Same encoder + preprocessing as offline_encoder_check.py. Run in the openpi venv on the eval box:
  cd /workspace/polaris/third_party/openpi
  uv run python /workspace/polaris/experiments/jepa_verifier/plot_goal_transfer.py \
      --videos-dir /workspace/polaris/runs/<baseline_run> \
      --goal /workspace/polaris/<...>/episode_0003_success_external_cam.png \
      --ckpt /workspace/polaris/third_party/vjepa2/checkpoints/vjepa2-ac-vitg.pt \
      --success 2,11,21 --fail 6,26,47 \
      --out /workspace/polaris/experiments/jepa_verifier/figs/v3_goal_transfer_6ep
Then copy the .png back to the Mac.
"""
import argparse
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
    t = transform(frame_rgb_uint8)                       # (3,256,256)
    stacked = np.expand_dims(np.stack([t, t], axis=0), axis=0)   # (1,2,3,256,256)
    tensor = torch.from_numpy(stacked).float().permute(0, 2, 1, 3, 4).to(device)
    with torch.inference_mode():
        h = encoder(tensor)[:, -tokens_per_frame:, :]
        return F.layer_norm(h, (h.size(-1),))


def read_base_frames(path):
    """RGB uint8 frames; base cam = left half of the base|wrist viz."""
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos-dir", required=True, help="dir with episode_<N>.mp4")
    ap.add_argument("--goal", required=True, help="episode-3 goal image")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--success", default="2,11,21", help="comma list of SUCCESS episode indices (exclude 3)")
    ap.add_argument("--fail", default="6,26,47", help="comma list of FAIL episode indices")
    ap.add_argument("--no-source", dest="include_source", action="store_false",
                    help="hide episode 3 itself (by default it is drawn bold black as the "
                         "reference that DOES descend toward its own goal)")
    ap.set_defaults(include_source=True)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--out", default="v3_goal_transfer_6ep")
    args = ap.parse_args()

    succ = [int(x) for x in args.success.split(",") if x.strip() != ""]
    fail = [int(x) for x in args.fail.split(",") if x.strip() != ""]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder, _ = torch.hub.load("facebookresearch/vjepa2", "vjepa2_ac_vit_giant", pretrained=False)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    encoder.load_state_dict({k.replace("module.", "", 1): v for k, v in ckpt["encoder"].items()})
    encoder = encoder.to(device).eval()
    tokens_per_frame = int((256 // encoder.patch_size) ** 2)
    transform = build_transform()

    gbgr = cv2.imread(str(args.goal))
    if gbgr is None:
        raise FileNotFoundError(f"goal image not found: {args.goal}")
    z_goal = encode_frame(encoder, transform, cv2.cvtColor(gbgr, cv2.COLOR_BGR2RGB),
                          tokens_per_frame, device)

    def curve(ep):
        vp = Path(args.videos_dir) / f"episode_{ep}.mp4"
        if not vp.exists():
            raise FileNotFoundError(f"missing video: {vp}")
        frames = read_base_frames(vp)[:: args.stride]
        e = np.array([F.l1_loss(encode_frame(encoder, transform, f, tokens_per_frame, device),
                                z_goal).item() for f in frames])
        return np.linspace(0, 1, len(e)), e

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 13})
    fig, ax = plt.subplots(figsize=(9, 5.5))

    greens = plt.cm.Greens(np.linspace(0.55, 0.9, len(succ)))
    reds = plt.cm.Reds(np.linspace(0.55, 0.9, len(fail)))

    for ep, c in zip(succ, greens):
        xs, e = curve(ep)
        ax.plot(xs, e, "-", lw=2.4, color=c, marker="o", ms=4, label=f"ep {ep}  (success)")
    for ep, c in zip(fail, reds):
        xs, e = curve(ep)
        ax.plot(xs, e, "--", lw=2.4, color=c, marker="s", ms=4, label=f"ep {ep}  (fail)")
    if args.include_source:
        xs, e = curve(3)
        ax.plot(xs, e, "-", lw=3.2, color="black", marker="D", ms=5, label="ep 3  (goal source)")

    ax.set_xlabel("normalized episode time  (0 = start,  1 = end)")
    ax.set_ylabel("latent L1 distance to episode-3 goal")
    ax.set_title("Episode-3's goal image does not transfer:\n"
                 "even successful episodes never descend toward it")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(f"{args.out}.png", dpi=160)
    print(f"saved {args.out}.png  (success={succ}, fail={fail}"
          f"{', +source ep3' if args.include_source else ''})")


if __name__ == "__main__":
    main()
