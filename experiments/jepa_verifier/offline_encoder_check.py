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
    ap.add_argument("--goal", default=None, help="single shared goal image (all videos)")
    ap.add_argument("--goal-dir", default=None,
                    help="per-episode goals: dir with goal_XXXX.png matched by the episode index "
                         "parsed from each video filename (episode_N.mp4 -> goal_000N.png)")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="offline_encoder_check")
    ap.add_argument("--stride", type=int, default=3, help="encode every k-th frame")
    ap.add_argument("--mask-dir", default=None,
                    help="dir with mask_XXXX.npy (grid x grid bool); if set, L1 is averaged "
                         "only over foreground tokens (bowl/food region) for that episode")
    args = ap.parse_args()
    if (args.goal is None) == (args.goal_dir is None):
        ap.error("pass exactly one of --goal or --goal-dir")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder, _ = torch.hub.load("facebookresearch/vjepa2", "vjepa2_ac_vit_giant", pretrained=False)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    encoder.load_state_dict({k.replace("module.", "", 1): v for k, v in ckpt["encoder"].items()})
    encoder = encoder.to(device).eval()
    tokens_per_frame = int((256 // encoder.patch_size) ** 2)
    transform = build_transform()

    def load_goal_z(path):
        bgr = cv2.imread(str(path))
        if bgr is None:
            raise FileNotFoundError(f"goal image not found: {path}")
        return encode_frame(encoder, transform, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
                            tokens_per_frame, device)

    def ep_index(video_path):
        # 'episode_7.mp4' -> 7
        stem = Path(video_path).stem
        digits = "".join(c for c in stem if c.isdigit())
        return int(digits) if digits else None

    z_goal_shared = load_goal_z(args.goal) if args.goal else None

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    videos = sorted(glob.glob(args.videos_glob))
    print(f"{len(videos)} videos; goal={'per-episode ' + args.goal_dir if args.goal_dir else args.goal}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    rows = []
    for vp in videos:
        frames = read_video_frames(vp)[:: args.stride]
        zs = [encode_frame(encoder, transform, f, tokens_per_frame, device) for f in frames]
        name = Path(vp).stem

        idx = ep_index(vp)
        if z_goal_shared is not None:
            z_goal = z_goal_shared
        else:
            gpath = Path(args.goal_dir) / f"goal_{idx:04d}.png"
            if not gpath.exists():
                print(f"  !! skip {name}: no goal {gpath}")
                continue
            z_goal = load_goal_z(gpath)

        # foreground-token mask (bowl/food region). Required for the order-invariant metrics.
        tok_mask = None
        pmask = None
        if args.mask_dir is not None:
            mpath = Path(args.mask_dir) / f"mask_{idx:04d}.npy"
            if not mpath.exists() or not np.load(mpath).any():
                print(f"  !! skip {name}: missing/empty mask {mpath}")
                continue
            m2d = np.load(mpath).astype(bool)                    # (grid,grid)
            tok_mask = torch.from_numpy(m2d.reshape(-1)).to(device)
            H, W = frames[0].shape[:2]
            pmask = cv2.resize(m2d.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)

        # goal pixels (for the color-distribution metric)
        goal_img = None
        if pmask is not None:
            gp = Path(args.goal_dir) / f"goal_{idx:04d}.png" if args.goal_dir else Path(args.goal)
            gbgr = cv2.imread(str(gp))
            goal_img = cv2.cvtColor(cv2.resize(gbgr, (frames[0].shape[1], frames[0].shape[0])),
                                    cv2.COLOR_BGR2RGB)

        def color_hist(img, pm, bins=16):
            px = img[pm]                                         # (N,3)
            h = np.concatenate([np.histogram(px[:, c], bins=bins, range=(0, 255))[0]
                                for c in range(3)]).astype(float)
            return h / (h.sum() + 1e-8)

        # Three metrics vs the goal, all restricted to the bowl/food tokens/pixels:
        #  posl1 = position-matched token L1 (current default; pose-sensitive via RoPE)
        #  pool  = L1 of MEAN-POOLED masked feature (order-invariant over tokens)
        #  hist  = L1 between masked-region color histograms (pixel distribution, pose-free)
        def posl1(z):
            return (z - z_goal).abs()[:, tok_mask, :].mean().item() if tok_mask is not None \
                else F.l1_loss(z, z_goal).item()

        gpool = z_goal[:, tok_mask, :].mean(dim=1) if tok_mask is not None else z_goal.mean(dim=1)
        def pool(z):
            zp = z[:, tok_mask, :].mean(dim=1) if tok_mask is not None else z.mean(dim=1)
            return (zp - gpool).abs().mean().item()

        ghist = color_hist(goal_img, pmask) if pmask is not None else None
        def hist(frame):
            return float(np.abs(color_hist(frame, pmask) - ghist).sum()) if pmask is not None else 0.0

        e_pos = np.array([posl1(z) for z in zs])
        e_pool = np.array([pool(z) for z in zs])
        e_hist = np.array([hist(f) for f in frames]) if pmask is not None else np.zeros(len(zs))

        xs = np.linspace(0, 1, len(zs))
        ax.plot(xs, e_pool, marker=".", ms=3, lw=1, label=name)
        ax2.plot(xs, e_hist, marker=".", ms=3, lw=1, label=name)

        def tr(e):
            k = max(1, len(e) // 3)
            return e[-k:].mean() - e[:k].mean()
        rows.append((name, len(zs),
                     e_pos[0], e_pos[-1], tr(e_pos),
                     e_pool[0], e_pool[-1], tr(e_pool),
                     e_hist[0], e_hist[-1], tr(e_hist)))
        print(f"{name}: n={len(zs)} | posL1 tr={tr(e_pos):+.4f} | pool tr={tr(e_pool):+.4f} "
              f"| hist tr={tr(e_hist):+.4f}")

    ax.set_xlabel("normalized episode time"); ax.set_ylabel("mean-pooled masked feature L1 to goal")
    ax.set_title("(A) order-invariant V-JEPA feature (pooled over bowl tokens)")
    ax.legend(fontsize=6, ncol=3); ax.grid(alpha=0.3)
    ax2.set_xlabel("normalized episode time"); ax2.set_ylabel("bowl-region color-histogram L1 to goal")
    ax2.set_title("(B) pixel color distribution (pose-free)")
    ax2.legend(fontsize=6, ncol=3); ax2.grid(alpha=0.3)
    fig.suptitle("Pose-invariant progress signals over the bowl region "
                 "(descending toward goal = usable verifier signal)")
    fig.tight_layout()
    fig.savefig(f"{args.out}.png", dpi=160)

    with open(f"{args.out}.csv", "w") as fcsv:
        fcsv.write("episode,n_frames,"
                   "posl1_first,posl1_last,posl1_trend,"
                   "pool_first,pool_last,pool_trend,"
                   "hist_first,hist_last,hist_trend\n")
        for r in rows:
            fcsv.write(f"{r[0]},{r[1]}," + ",".join(f"{v:.6f}" for v in r[2:]) + "\n")
    print(f"\nsaved {args.out}.png / .csv  ({len(rows)} episodes)")
    for label, base in [("posL1", 2), ("pool", 5), ("hist", 8)]:
        trends = [r[base + 2] for r in rows]
        down = sum(1 for t in trends if t < 0)
        print(f"  {label:>6}: {down}/{len(trends)} episodes trend down toward goal")


if __name__ == "__main__":
    main()
