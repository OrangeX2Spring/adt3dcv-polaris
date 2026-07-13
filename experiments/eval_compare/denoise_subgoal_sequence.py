#!/usr/bin/env python3
"""
De-noise the per-subtask goal SEQUENCE for the V-JEPA verifier.

Combines two existing pieces:
  * extract_subtask_goals.py (teammate): for each IC, reads a successful rollout's
    episode_<e>_steps.jsonl and locates the frame where each checker (c0..c5) first
    flips True -- the "completion frame".
  * denoise_goal_images.py (task 1): the last/transitional frame of a rollout is the
    noisiest (arm/gripper occludes, objects mid-motion) -- pick a SETTLED frame instead.

The gap this fills: the completion frame is itself the noisiest moment of a subtask --
it is the instant the checker trips, i.e. the arm is right on the object, mid-motion.
So for EACH subtask we don't take frame f_c; we search a window around f_c and keep the
lowest-motion (most settled) frame. Output is a clean per-IC subgoal sequence
(c0..c5) plus, per subgoal, a candidates contact sheet (completion frame vs chosen
settled frame vs neighbours, each labelled with its inter-frame motion) plus a per-IC
overview.png (the six chosen subgoals in a grid) so a human can eyeball and override.

Frames use the external/base cam (left half of the external|wrist policy-view mp4),
matching how the verifier consumes goals (droid_jointpos_client resizes the same view).

READ-ONLY: reads <repeat-root>/repeat_ic<k>/{eval_results.csv, episode_<e>.mp4,
episode_<e>_steps.jsonl}; writes only into --out. Never touches existing runs or code.

Usage:
  python denoise_subgoal_sequence.py \
      --repeat-root ../../runs/pi05 \
      --ics 2 9 10 13 20 21 24 37 42 46 \
      --out out_subgoal_denoise
  # single IC, wider forward window:
  python denoise_subgoal_sequence.py --repeat-root ../../runs/pi05 --ics 42 \
      --back 4 --fwd 10 --out out_subgoal_denoise
"""
import argparse
import csv
import json
import re
from pathlib import Path

import cv2
import numpy as np


def tb(x):
    return str(x).strip().lower() in ("true", "1", "1.0")


def read_base_cam(video_path):
    """External/base cam = left half of the external|wrist policy-view video."""
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        frames.append(bgr[:, : bgr.shape[1] // 2, :])
    cap.release()
    return frames


def motion_curve(frames):
    """Mean abs pixel diff to the previous frame; m[0] = 0."""
    m = np.zeros(len(frames), np.float32)
    for i in range(1, len(frames)):
        m[i] = np.abs(frames[i].astype(np.int16) - frames[i - 1].astype(np.int16)).mean()
    return m


def completion_frames(steps_path):
    """{checker_key_without_ever: completion_video_frame} from a steps.jsonl."""
    recs = [json.loads(l) for l in Path(steps_path).read_text().splitlines() if l.strip()]
    out = {}
    for k in recs[0]:
        if not k.endswith("_ever"):
            continue
        first = next((r for r in recs if r.get(k)), None)
        if first is not None:
            out[k[: -len("_ever")]] = int(first["frame"])
    return out


def pick_settled(m, f_c, back, fwd):
    """Lowest-motion frame in [f_c - back, f_c + fwd], clamped. Returns (idx, motion)."""
    n = len(m)
    lo = max(0, f_c - back)
    hi = min(n - 1, f_c + fwd)
    window = m[lo : hi + 1]
    idx = lo + int(np.argmin(window))
    return idx, float(m[idx])


def overview_sheet(tiles, ncol=3, cell=224, pad=6):
    """Grid of the chosen denoised subgoals (c0..c5) for one IC, each labelled.

    tiles: list of (label, frame_bgr) in checker order.
    """
    nrow = (len(tiles) + ncol - 1) // ncol
    canvas = np.full((nrow * (cell + pad) + pad, ncol * (cell + pad) + pad, 3),
                     255, np.uint8)
    for i, (label, f) in enumerate(tiles):
        r, c = divmod(i, ncol)
        img = cv2.resize(f, (cell, cell))
        cv2.putText(img, label, (4, 16), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (0, 255, 255), 1, cv2.LINE_AA)
        y, x = pad + r * (cell + pad), pad + c * (cell + pad)
        canvas[y : y + cell, x : x + cell] = img
    return canvas


def candidate_sheet(frames, m, f_c, chosen, width=320):
    """Stack: completion frame (noisy) + chosen settled + 2 neighbours, labelled."""
    n = len(frames)
    picks = [
        ("COMPLETION f%d (noisy)" % f_c, f_c),
        ("CHOSEN settled", chosen),
        ("chosen-3", max(0, chosen - 3)),
        ("chosen+3", min(n - 1, chosen + 3)),
    ]
    rows = []
    for label, idx in picks:
        f = frames[min(idx, n - 1)]
        fh, fw = f.shape[:2]
        img = cv2.resize(f, (width, int(fh * width / fw)))
        cv2.putText(img, f"{label}  f{idx}  motion={m[idx]:.1f}", (6, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1, cv2.LINE_AA)
        rows.append(img)
        rows.append(np.full((2, width, 3), 255, np.uint8))
    return np.vstack(rows)


def process_ic(repeat_root, ic, out_root, back, fwd, episode=None):
    run = Path(repeat_root) / f"repeat_ic{ic}"
    csv_path = run / "eval_results.csv"
    if not csv_path.exists():
        print(f"!! IC {ic}: no {csv_path}, skipped")
        return None

    rows = list(csv.DictReader(open(csv_path)))
    succ = [int(r["episode"]) for r in rows if tb(r["success"])]
    if episode is not None:
        ep = episode
    elif succ:
        ep = succ[0]
    else:
        print(f"!! IC {ic}: no successful episode, skipped (cannot build clean goals)")
        return None

    steps_path = run / f"episode_{ep}_steps.jsonl"
    video_path = run / f"episode_{ep}.mp4"
    if not steps_path.exists() or not video_path.exists():
        print(f"!! IC {ic}: missing {steps_path.name} or {video_path.name}, skipped")
        return None

    frames_by_checker = completion_frames(steps_path)
    if len(frames_by_checker) < 6:
        print(f"   IC {ic}: episode {ep} only {len(frames_by_checker)} completed checkers")

    frames = read_base_cam(video_path)
    if len(frames) < 3:
        print(f"!! IC {ic}: episode {ep} has too few frames, skipped")
        return None
    m = motion_curve(frames)

    outdir = Path(out_root) / f"ic{ic}"
    outdir.mkdir(parents=True, exist_ok=True)

    manifest = []
    tiles = []
    for key, f_c in sorted(frames_by_checker.items(), key=lambda kv: kv[1]):
        mm = re.match(r"c(\d+)_(.+)", key)
        cj, name = mm.group(1), mm.group(2)
        f_c = min(f_c, len(frames) - 1)
        chosen, chosen_motion = pick_settled(m, f_c, back, fwd)

        cv2.imwrite(str(outdir / f"c{cj}_{name}.png"), frames[chosen])
        cv2.imwrite(str(outdir / f"candidates_c{cj}_{name}.png"),
                    candidate_sheet(frames, m, f_c, chosen))
        tiles.append((f"c{cj}_{name} f{chosen}", frames[chosen]))
        manifest.append({
            "ic": ic, "episode": ep, "checker": key,
            "completion_frame": f_c, "completion_motion": round(float(m[f_c]), 2),
            "chosen_frame": chosen, "chosen_motion": round(chosen_motion, 2),
            "shift": chosen - f_c,
        })

    cv2.imwrite(str(outdir / "overview.png"), overview_sheet(tiles))

    with open(outdir / "manifest.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(manifest[0].keys()))
        w.writeheader()
        w.writerows(manifest)

    avg_drop = np.mean([r["completion_motion"] - r["chosen_motion"] for r in manifest])
    print(f"IC {ic}: episode {ep} -> {len(manifest)} denoised subgoals in {outdir}  "
          f"(avg motion {avg_drop:+.1f} vs completion frame)")
    return manifest


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repeat-root", required=True, help="dir containing repeat_ic<k>/")
    ap.add_argument("--ics", type=int, nargs="+", required=True)
    ap.add_argument("--out", default="out_subgoal_denoise")
    ap.add_argument("--episode", type=int, default=None,
                    help="force this episode index (default: first success in the IC)")
    ap.add_argument("--back", type=int, default=4,
                    help="frames before completion to search for a settled frame")
    ap.add_argument("--fwd", type=int, default=10,
                    help="frames after completion to search (settling usually follows completion)")
    args = ap.parse_args()

    all_rows = []
    for ic in args.ics:
        mani = process_ic(args.repeat_root, ic, args.out, args.back, args.fwd, args.episode)
        if mani:
            all_rows.extend(mani)

    if all_rows:
        out = Path(args.out) / "manifest_all.csv"
        with open(out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(all_rows[0].keys()))
            w.writeheader()
            w.writerows(all_rows)
        print(f"\ndone -> {args.out}  ({len(all_rows)} subgoals across {len(args.ics)} ICs)"
              f"\nmanifest -> {out}")
    else:
        print("\nno subgoals produced (no usable successful episodes?)")


if __name__ == "__main__":
    main()
