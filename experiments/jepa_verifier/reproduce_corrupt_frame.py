"""
Minimal reproduction of the "corrupted current-frame" bug (Bug #5 in REPORT_2026-07-02.md).

It uses the SAME transform as the buggy policy (policy_jepa.py:80-88) and the SAME
normalization openpi applies to images (model.py:118), with NO model involved.

Two paths, one image:
  - GOAL path  (correct): feed uint8 [0,255]     -> transform   (what _encode_goal_image does)
  - CURR path  (buggy)  : feed float  [-1,1]      -> transform   (what infer() does on the live frame)

Run in the docker env that already has torch/torchvision/PIL:
    python reproduce_corrupt_frame.py --image /workspace/polaris/.../last_frame.jpg
Outputs:
    corrupt_frame_repro.png   (side by side: original | goal-path | curr-path)
    and prints the [-1,1] -> byte wrap-around table.
"""
import argparse
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T

# --- exact transform from policy_jepa.py:80-88 (the buggy version) ---
transform = T.Compose([
    T.ToPILImage(),
    T.Resize((256, 256)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def denorm_to_uint8(t):
    """Undo Normalize so we can *see* what the encoder actually received."""
    x = (t * STD + MEAN).clamp(0, 1)          # [0,1]
    return (x.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="any RGB frame (jpg/png)")
    ap.add_argument("--out", default="corrupt_frame_repro.png")
    args = ap.parse_args()

    img_u8 = np.array(Image.open(args.image).convert("RGB"))   # (H,W,3) uint8 [0,255]

    # GOAL path (correct): uint8 in, exactly like _encode_goal_image()
    goal_t = transform(img_u8)

    # CURR path (buggy): openpi first maps uint8 -> [-1,1] float (model.py:118),
    # then policy_jepa.py:178 feeds that float array straight into the same transform.
    img_f = img_u8.astype(np.float32) / 255.0 * 2.0 - 1.0      # [-1,1]  <-- the poison
    curr_t = transform(np.array(img_f))

    # ---- visual proof ----
    goal_vis = Image.fromarray(denorm_to_uint8(goal_t)).resize((256, 256))
    curr_vis = Image.fromarray(denorm_to_uint8(curr_t)).resize((256, 256))
    orig_vis = Image.fromarray(img_u8).resize((256, 256))
    canvas = Image.new("RGB", (256 * 3 + 20, 256), "white")
    canvas.paste(orig_vis, (0, 0))
    canvas.paste(goal_vis, (256 + 10, 0))
    canvas.paste(curr_vis, (256 * 2 + 20, 0))
    canvas.save(args.out)
    print(f"saved {args.out}  (left: original | middle: GOAL path (clean) | right: CURR path (corrupted))")

    # ---- numeric proof: what ToPILForm does to [-1,1] via .mul(255).byte() wrap ----
    ramp = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0]).view(1, 5, 1).repeat(3, 1, 1)
    wrapped = np.array(T.ToPILImage()(ramp))[0, :, 0]
    print("\n[-1,1] float  ->  byte after ToPILImage (negatives wrap around):")
    for v, b in zip([-1.0, -0.5, 0.0, 0.5, 1.0], wrapped):
        print(f"   {v:+.1f}  ->  {int(b):3d}")

    # ---- energy proof: distance is dominated by the corruption, not content ----
    # (pixel-space L1 stand-in; with the real encoder the same gap shows up in latent space)
    l1_clean_vs_corrupt = (goal_t - curr_t).abs().mean().item()
    l1_clean_vs_clean = (goal_t - transform(img_u8)).abs().mean().item()
    print(f"\nL1(goal_clean, curr_CORRUPT) = {l1_clean_vs_corrupt:.4f}")
    print(f"L1(goal_clean, curr_clean)   = {l1_clean_vs_clean:.4f}  (should be ~0: same image)")


if __name__ == "__main__":
    main()
