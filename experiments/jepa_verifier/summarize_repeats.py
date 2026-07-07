"""
Summarize the repeat-IC screening runs (runs/pi05/repeat_ic<k>/eval_results.csv).

Prints per-IC: n runs, success rate, mean/min/max progress, stage where failures stall,
and a borderline verdict (0 < rate < 1 -> usable for the verifier stabilization test).

Usage (works anywhere python3 exists, no deps beyond stdlib):
  python3 summarize_repeats.py --runs-root /workspace/polaris/runs/pi05 [--prefix repeat_ic]
"""
import argparse
import csv
import re
from pathlib import Path

STAGES = [
    ("c0", "reach_ice_cream"),
    ("c1", "reach_grapes"),
    ("c2", "lift_ice_cream"),
    ("c3", "lift_grapes"),
    ("c4", "inside_ice_cream__bowl"),
    ("c5", "inside_grapes_bowl"),
]
# category each IC belonged to in the original baseline-vs-jepa comparison
CATEGORY = {2: "both", 11: "both", 21: "both",
            3: "base_only", 9: "base_only", 13: "base_only",
            15: "base_only", 43: "base_only", 56: "base_only",
            10: "jepa_only", 20: "jepa_only", 24: "jepa_only", 30: "jepa_only",
            37: "jepa_only", 42: "jepa_only", 46: "jepa_only", 52: "jepa_only"}


def tb(x):
    return str(x).strip().lower() in ("true", "1", "1.0")


def stalled_stage(row):
    done = 0
    for c, name in STAGES:
        if tb(row.get(f"r_{c}_{name}_ever", "False")):
            done += 1
        else:
            return STAGES[done][1]
    return "done"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", required=True)
    ap.add_argument("--prefix", default="repeat_ic")
    args = ap.parse_args()

    root = Path(args.runs_root)
    folders = sorted(root.glob(f"{args.prefix}*"),
                     key=lambda p: int(re.sub(r"\D", "", p.name) or -1))
    if not folders:
        print(f"!! no folders matching '{args.prefix}*' under {root}")
        print(f"   what's actually there: {[p.name for p in sorted(root.iterdir())] if root.exists() else 'ROOT DOES NOT EXIST'}")
        return
    print(f"found {len(folders)} folders: {[p.name for p in folders]}")
    missing = [p.name for p in folders if not (p / 'eval_results.csv').exists()]
    if missing:
        print(f"!! {len(missing)} folders have NO eval_results.csv yet (crashed or not started): {missing}")
    print(f"{'IC':>3} {'cat':>10} {'n':>3} {'succ':>6} {'prog mean(min-max)':>20}  stalls")
    borderline = []
    for f in folders:
        csvp = f / "eval_results.csv"
        if not csvp.exists():
            continue
        ic = int(re.sub(r"\D", "", f.name))
        rows = list(csv.DictReader(open(csvp)))
        n = len(rows)
        if n == 0:
            continue
        s = sum(tb(r["success"]) for r in rows)
        progs = [float(r["progress"]) for r in rows]
        hist = {}
        for r in rows:
            if not tb(r["success"]):
                st = stalled_stage(r)
                hist[st] = hist.get(st, 0) + 1
        stalls = " ".join(f"{k}x{v}" for k, v in sorted(hist.items(), key=lambda x: -x[1]))
        rate = s / n
        if 0 < rate < 1:
            borderline.append((ic, rate))
        print(f"{ic:>3} {CATEGORY.get(ic, '?'):>10} {n:>3} {s:>3}/{n:<2} "
              f"{sum(progs)/n:>6.2f} ({min(progs):.2f}-{max(progs):.2f})  {stalls}")

    print(f"\nBORDERLINE (0 < rate < 1), {len(borderline)} ICs: "
          + ", ".join(f"IC{ic}={r:.0%}" for ic, r in sorted(borderline, key=lambda x: x[1])))


if __name__ == "__main__":
    main()
