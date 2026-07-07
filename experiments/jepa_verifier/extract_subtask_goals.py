"""
Extract per-IC, per-subtask goal images from the repeat-screening runs.

For each requested IC, finds a successful episode in repeat_ic<k>/ (eval_results.csv),
reads its episode_<e>_steps.jsonl to locate each checker's completion frame, and saves
that frame (base-cam = left half) as <out>/ic<k>/c<j>_<name>.png.

These are the goals for the subtask verifier (policy_subtask.py): same IC -> same layout,
same render pipeline as live observations (the v3 lesson).

Run on the eval box (only needs cv2, no torch):
  python3 /workspace/polaris/experiments/jepa_verifier/extract_subtask_goals.py \
      --repeat-root /workspace/polaris/runs/pi05 \
      --ics 2 9 10 13 20 21 24 37 42 46 \
      --out /workspace/polaris/runs/goals_subtask
"""
import argparse
import csv
import json
import re
from pathlib import Path

import cv2


def tb(x):
    return str(x).strip().lower() in ("true", "1", "1.0")


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat-root", required=True, help="dir containing repeat_ic<k>/")
    ap.add_argument("--ics", type=int, nargs="+", required=True)
    ap.add_argument("--out", required=True, help="goals root; writes <out>/ic<k>/c<j>_<name>.png")
    args = ap.parse_args()

    for ic in args.ics:
        run = Path(args.repeat_root) / f"repeat_ic{ic}"
        rows = list(csv.DictReader(open(run / "eval_results.csv")))
        succ = [int(r["episode"]) for r in rows if tb(r["success"])]
        if not succ:
            print(f"!! IC {ic}: no successful episode, skipped (cannot build goals)")
            continue
        ep = succ[0]
        frames_by_checker = completion_frames(run / f"episode_{ep}_steps.jsonl")
        if len(frames_by_checker) < 6:
            print(f"!! IC {ic}: episode {ep} steps.jsonl has only {len(frames_by_checker)} completed checkers")

        cap = cv2.VideoCapture(str(run / f"episode_{ep}.mp4"))
        frames = []
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            frames.append(bgr[:, : bgr.shape[1] // 2, :])  # left half = base cam
        cap.release()

        outdir = Path(args.out) / f"ic{ic}"
        outdir.mkdir(parents=True, exist_ok=True)
        for key, f in sorted(frames_by_checker.items(), key=lambda kv: kv[1]):
            m = re.match(r"c(\d+)_(.+)", key)
            fname = f"c{m.group(1)}_{m.group(2)}.png"
            cv2.imwrite(str(outdir / fname), frames[min(f, len(frames) - 1)])
        print(f"IC {ic}: episode {ep} -> {len(frames_by_checker)} goals in {outdir}  "
              f"(completion frames: {dict(sorted(frames_by_checker.items(), key=lambda kv: kv[1]))})")


if __name__ == "__main__":
    main()
