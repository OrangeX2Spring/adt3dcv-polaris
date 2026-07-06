#!/usr/bin/env python3
"""
Export side-by-side comparison videos: for each initial condition (IC), stack the
baseline rollout (left) next to the verifier rollout (right) into one mp4, with a
banner showing the IC index and each side's SUCCESS/FAIL. This is the artefact for
manual behaviour-failure analysis ("what went wrong, and did the verifier fix it").

Pick which ICs to export either explicitly (--episodes 0,5,12) or by category from a
comparison.csv produced by compare_runs.py (--from-comparison comparison.csv
--category verifier_only) -- the latter gives exactly the "verifier succeeds where
baseline fails" clips.

READ-ONLY on the runs (reads episode_<k>.mp4 + eval_results.csv). Writes only new
mp4s into --out. Never modifies existing runs or code.

Usage:
  # the money clips: verifier fixes what baseline failed
  python export_pairs.py \
      --baseline ../../runs/food_bussing_goal_frames \
      --verifier ../../runs/food_bussing_goal_jepa \
      --from-comparison ./out_food_bussing/comparison.csv --category verifier_only \
      --out ./pairs_verifier_fixed

  # specific ICs
  python export_pairs.py --baseline ... --verifier ... --episodes 3,7,19 --out ./pairs
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

BANNER_H = 40


def load_success(run):
    df = pd.read_csv(Path(run) / "eval_results.csv").drop_duplicates("episode", keep="last")
    return {int(r.episode): bool(r.success) for r in df.itertuples()}


def read_frames(path):
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        frames.append(bgr)
    cap.release()
    return frames


def pad_to(frames, n, h):
    """Freeze last frame so both sides run to the same length; resize to height h."""
    if not frames:
        blank = np.zeros((h, h, 3), np.uint8)
        return [blank] * n
    out = []
    for f in frames:
        fh, fw = f.shape[:2]
        out.append(cv2.resize(f, (int(fw * h / fh), h)))
    while len(out) < n:
        out.append(out[-1].copy())
    return out[:n]


def banner(width, text, ok):
    strip = np.full((BANNER_H, width, 3), (40, 40, 40), np.uint8)
    color = (60, 200, 60) if ok else (60, 60, 220)  # BGR: green ok / red fail
    cv2.putText(strip, text, (8, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    return strip


def make_pair(ep, base_dir, ver_dir, base_ok, ver_ok, out_dir, height, fps,
              base_name, ver_name):
    bp = Path(base_dir) / f"episode_{ep}.mp4"
    vp = Path(ver_dir) / f"episode_{ep}.mp4"
    if not bp.exists() or not vp.exists():
        print(f"  [skip] ep {ep}: missing {'baseline' if not bp.exists() else 'verifier'} video")
        return False
    bf = read_frames(bp)
    vf = read_frames(vp)
    n = max(len(bf), len(vf))
    bf = pad_to(bf, n, height)
    vf = pad_to(vf, n, height)
    wl = bf[0].shape[1]
    wr = vf[0].shape[1]

    bb = banner(wl, f"{base_name} ep{ep}: {'SUCCESS' if base_ok else 'FAIL'}", base_ok)
    vb = banner(wr, f"{ver_name} ep{ep}: {'SUCCESS' if ver_ok else 'FAIL'}", ver_ok)
    top = np.hstack([bb, vb])

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"pair_ep{ep}_{base_name}_{'S' if base_ok else 'F'}_vs_{ver_name}_{'S' if ver_ok else 'F'}.mp4"
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (wl + wr, height + BANNER_H))
    for i in range(n):
        row = np.hstack([bf[i], vf[i]])
        writer.write(np.vstack([top, row]))
    writer.release()
    print(f"  wrote {out_path.name}")
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--verifier", required=True)
    ap.add_argument("--out", default="./pairs")
    ap.add_argument("--episodes", default=None, help="comma list, e.g. 0,5,12")
    ap.add_argument("--from-comparison", default=None, help="comparison.csv from compare_runs.py")
    ap.add_argument("--category", default="verifier_only",
                    help="category filter when using --from-comparison")
    ap.add_argument("--height", type=int, default=256)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--baseline-name", default="baseline")
    ap.add_argument("--verifier-name", default="verifier")
    args = ap.parse_args()

    if (args.episodes is None) == (args.from_comparison is None):
        ap.error("pass exactly one of --episodes or --from-comparison")

    if args.episodes:
        eps = [int(x) for x in args.episodes.split(",") if x.strip() != ""]
    else:
        cmp = pd.read_csv(args.from_comparison)
        eps = sorted(int(e) for e in cmp.loc[cmp["category"] == args.category, "episode"])
        print(f"{len(eps)} ICs in category '{args.category}': {eps}")

    base_ok = load_success(args.baseline)
    ver_ok = load_success(args.verifier)

    done = 0
    for ep in eps:
        done += make_pair(ep, args.baseline, args.verifier,
                          base_ok.get(ep, False), ver_ok.get(ep, False),
                          args.out, args.height, args.fps,
                          args.baseline_name, args.verifier_name)
    print(f"\nexported {done}/{len(eps)} pair videos to {args.out}")


if __name__ == "__main__":
    main()
