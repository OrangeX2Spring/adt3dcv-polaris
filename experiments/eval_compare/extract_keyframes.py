#!/usr/bin/env python3
"""
Turn each comparison mp4 into a single "contact sheet" PNG: N frames sampled evenly
across the clip, stacked top-to-bottom (time goes down), each labelled with its
normalized time. One PNG per video -> lets a human (or a vision model) read a whole
rollout at a glance for behaviour-failure analysis.

READ-ONLY on videos; writes only PNGs into --out.

Usage:
  python extract_keyframes.py --videos "pairs/verifier_fixed/*.mp4" --out sheets/verifier_fixed --n 6
"""
import argparse
import glob
from pathlib import Path

import cv2
import numpy as np


def sample_frames(path, n):
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = np.linspace(0, max(total - 1, 0), n).round().astype(int)
    want = set(int(i) for i in idxs)
    grabbed = {}
    i = 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        if i in want:
            grabbed[i] = bgr
        i += 1
    cap.release()
    return [grabbed[j] for j in sorted(want) if j in grabbed], sorted(want), total


def contact_sheet(path, n, width):
    frames, idxs, total = sample_frames(path, n)
    if not frames:
        return None
    rows = []
    for f, idx in zip(frames, idxs):
        fh, fw = f.shape[:2]
        img = cv2.resize(f, (width, int(fh * width / fw)))
        t = idx / max(total - 1, 1)
        cv2.putText(img, f"t={t:.2f} (f{idx})", (6, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        rows.append(img)
        rows.append(np.full((2, width, 3), 255, np.uint8))  # separator
    return np.vstack(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True, help="glob of mp4s")
    ap.add_argument("--out", default="sheets")
    ap.add_argument("--n", type=int, default=6, help="frames per sheet")
    ap.add_argument("--width", type=int, default=640)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    vids = sorted(glob.glob(args.videos))
    print(f"{len(vids)} videos -> {out}")
    for vp in vids:
        sheet = contact_sheet(vp, args.n, args.width)
        if sheet is None:
            print(f"  [skip] {vp}: no frames")
            continue
        name = Path(vp).stem + ".png"
        cv2.imwrite(str(out / name), sheet)
        print(f"  wrote {name}")


if __name__ == "__main__":
    main()
