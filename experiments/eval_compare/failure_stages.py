#!/usr/bin/env python3
"""
Attribute each failed episode to the stage where it stalled, using the per-checker
`r_c<idx>_<name>_ever` columns that eval.py logs (1 = that sub-goal was reached at
some point, 0 = never). The stall stage = the first checker (in c0..cN order) that
was never reached. If every checker was reached but success is still False, the
episode is labelled `all_reached_but_unsuccessful` (usually an ordering/skip issue).

Generalises across environments: it auto-discovers the checker columns, so it works
for FoodBussing (reach/lift/inside x ice_cream/grapes), TapeIntoContainer, etc.

READ-ONLY: reads one <run>/eval_results.csv; optionally writes <out> csv. Never
touches existing runs, videos, or code.

Usage:
  python failure_stages.py --run ../../runs/food_bussing_goal_frames
  python failure_stages.py --run ../../runs/food_bussing_goal_jepa --out stalls_jepa.csv
"""
import argparse
import re
from collections import Counter
from pathlib import Path

import pandas as pd

EVER_RE = re.compile(r"^r_c(\d+)_(.*)_ever$")


def checker_cols(df):
    cols = []
    for c in df.columns:
        m = EVER_RE.match(c)
        if m:
            cols.append((int(m.group(1)), m.group(2), c))
    cols.sort(key=lambda t: t[0])
    return cols  # list of (idx, name, column)


def stall_stage(row, cols):
    for idx, name, col in cols:
        val = row.get(col)
        if pd.isna(val) or float(val) == 0.0:
            return f"c{idx}_{name}"
    return "all_reached_but_unsuccessful"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", default=None, help="optional csv of per-episode stalls")
    args = ap.parse_args()

    df = pd.read_csv(Path(args.run) / "eval_results.csv")
    df = df.drop_duplicates("episode", keep="last").sort_values("episode")
    cols = checker_cols(df)
    if not cols:
        raise SystemExit("no r_c<idx>_..._ever columns found; is this a PolaRiS eval csv?")

    print(f"run: {args.run}")
    print(f"checkers ({len(cols)}): " + " -> ".join(f"c{i}:{n}" for i, n, _ in cols))
    n = len(df)
    succ = int(df["success"].astype(bool).sum())
    print(f"episodes: {n} | success: {succ}/{n} ({100 * succ / n:.1f}%) | "
          f"failures: {n - succ}\n")

    fails = df[~df["success"].astype(bool)]
    stalls = fails.apply(lambda r: stall_stage(r, cols), axis=1)
    hist = Counter(stalls)

    print("where failures stall (most common first):")
    for stage, cnt in hist.most_common():
        print(f"  {cnt:3d}  {stage}")

    if args.out:
        per_ep = pd.DataFrame({
            "episode": fails["episode"].values,
            "progress": fails["progress"].values,
            "stall_stage": stalls.values,
        })
        per_ep.to_csv(args.out, index=False)
        print(f"\nsaved per-episode stalls -> {args.out}")


if __name__ == "__main__":
    main()
