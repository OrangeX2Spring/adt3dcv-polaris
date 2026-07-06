#!/usr/bin/env python3
"""
De-noise the V-JEPA goal image (task 1: "goal image 尾部噪声").

Problem: the verifier's goal image is currently the LAST frame of a rollout video
(e.g. last_frame.jpg / goal_from_video.png). The tail of a rollout is the noisiest
moment: the arm/gripper still occludes the bowl, objects are mid-motion/mid-fall,
and the frame is transitional -- a poor target for latent-L1.

Fix here (no GPU, read-only on videos): instead of the last frame, pick a SETTLED
frame near task completion -- the frame at the centre of the longest low-motion run
in the latter half of the rollout, where objects are at rest. Emits the chosen clean
goal plus a candidates contact sheet (chosen + a few alternatives + the last frame)
so a human can eyeball and override.

Uses the external cam (left half of episode_<k>.mp4, matching the policy obs). This
is a single-goal de-noiser; per-IC goals + serve wiring are a separate step.

Usage:
  python denoise_goal_images.py --run ../../runs/food_bussing_goal_frames --episode 21 \
      --out out_goal_denoise
  # batch over the successful episodes of a run:
  python denoise_goal_images.py --run ../../runs/food_bussing_goal_frames --successes \
      --out out_goal_denoise
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def read_external(path):
    """External cam = left half of the external|wrist policy-view video."""
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        frames.append(bgr[:, : bgr.shape[1] // 2, :])
    cap.release()
    return frames


def motion_curve(frames):
    m = np.zeros(len(frames), np.float32)
    for i in range(1, len(frames)):
        m[i] = np.abs(frames[i].astype(np.int16) - frames[i - 1].astype(np.int16)).mean()
    return m


def pick_settled(frames, tail_frac=0.5, still_pct=30):
    """Centre frame of the longest low-motion run in the latter `tail_frac` of the clip."""
    n = len(frames)
    m = motion_curve(frames)
    start = int(n * (1 - tail_frac))
    region = m[start:]
    if len(region) == 0:
        return n - 1, m
    thresh = np.percentile(region, still_pct)
    best_len, best_mid, cur_start = 0, n - 1, None
    for i in range(start, n):
        if m[i] <= thresh:
            cur_start = i if cur_start is None else cur_start
            run_len = i - cur_start + 1
            if run_len > best_len:
                best_len, best_mid = run_len, (cur_start + i) // 2
        else:
            cur_start = None
    return best_mid, m


def candidate_sheet(frames, chosen, m, width=384):
    """Stack: chosen + 2 neighbours + last frame, each labelled, for eyeballing."""
    n = len(frames)
    picks = [("CHOSEN (settled)", chosen),
             ("chosen-4", max(0, chosen - 4)),
             ("chosen+4", min(n - 1, chosen + 4)),
             ("LAST frame (noisy)", n - 1)]
    rows = []
    for label, idx in picks:
        f = frames[idx]
        fh, fw = f.shape[:2]
        img = cv2.resize(f, (width, int(fh * width / fw)))
        cv2.putText(img, f"{label}  f{idx}  motion={m[idx]:.1f}", (6, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
        rows.append(img)
        rows.append(np.full((2, width, 3), 255, np.uint8))
    return np.vstack(rows)


def process(run, episode, out_dir):
    vp = Path(run) / f"episode_{episode}.mp4"
    if not vp.exists():
        print(f"  [skip] ep {episode}: no {vp}")
        return
    frames = read_external(vp)
    if len(frames) < 3:
        print(f"  [skip] ep {episode}: too few frames")
        return
    chosen, m = pick_settled(frames)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out / f"goal_denoised_ep{episode}.png"), frames[chosen])
    cv2.imwrite(str(out / f"candidates_ep{episode}.png"),
                candidate_sheet(frames, chosen, m))
    print(f"  ep {episode}: {len(frames)} frames, chose f{chosen} "
          f"(motion {m[chosen]:.1f} vs last {m[-1]:.1f}) "
          f"-> goal_denoised_ep{episode}.png")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--episode", type=int, default=None)
    ap.add_argument("--successes", action="store_true",
                    help="process every episode with success==True in eval_results.csv")
    ap.add_argument("--out", default="out_goal_denoise")
    args = ap.parse_args()

    if (args.episode is None) == (not args.successes):
        ap.error("pass exactly one of --episode or --successes")

    if args.successes:
        df = pd.read_csv(Path(args.run) / "eval_results.csv").drop_duplicates("episode", keep="last")
        eps = [int(e) for e in df.loc[df["success"].astype(bool), "episode"]]
        print(f"{len(eps)} successful episodes: {eps}")
    else:
        eps = [args.episode]

    for ep in eps:
        process(args.run, ep, args.out)
    print(f"\ndone -> {args.out}")


if __name__ == "__main__":
    main()
