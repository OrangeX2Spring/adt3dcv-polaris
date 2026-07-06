#!/usr/bin/env python3
"""
Compare two PolaRiS eval runs by initial condition (IC) -- e.g. baseline
(no V-JEPA verifier) vs the verifier run.

Why episodes are comparable: scripts/eval.py rolls out
`initial_conditions[episode % N]`, so the `episode` column IS the IC index and is
directly comparable across runs. This script pairs the two runs on that column.

READ-ONLY on the runs: it only reads <run>/eval_results.csv. It writes a single
new file, <out>/comparison.csv. It never modifies existing runs, videos, or code.

Usage:
  python compare_runs.py \
      --baseline ../../runs/food_bussing_goal_frames \
      --verifier ../../runs/food_bussing_goal_jepa \
      --out ./out_food_bussing
"""
import argparse
from pathlib import Path

import pandas as pd


def load(run):
    p = Path(run) / "eval_results.csv"
    if not p.exists():
        raise FileNotFoundError(p)
    df = pd.read_csv(p)
    # keep the last logged row per episode (eval.py appends; guard against dupes)
    df = df.drop_duplicates("episode", keep="last").set_index("episode").sort_index()
    return df


def rate(df):
    n = len(df)
    s = int(df["success"].astype(bool).sum())
    mp = float(df["progress"].mean())
    return n, s, mp


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline", required=True, help="run folder WITHOUT verifier")
    ap.add_argument("--verifier", required=True, help="run folder WITH verifier")
    ap.add_argument("--out", default="./eval_compare_out")
    ap.add_argument("--baseline-name", default="baseline")
    ap.add_argument("--verifier-name", default="verifier")
    args = ap.parse_args()

    b = load(args.baseline)
    v = load(args.verifier)
    bn, vn = args.baseline_name, args.verifier_name

    print("=" * 68)
    for name, df in [(bn, b), (vn, v)]:
        n, s, mp = rate(df)
        print(f"[overall] {name:12s} n={n:3d}  success={s:3d}/{n} "
              f"({100 * s / n:5.1f}%)  mean_progress={mp:.3f}")

    common = sorted(set(b.index) & set(v.index))
    only_b = sorted(set(b.index) - set(v.index))
    only_v = sorted(set(v.index) - set(b.index))
    print("=" * 68)
    print(f"[paired] common ICs: {len(common)}   "
          f"(only in {bn}: {len(only_b)}, only in {vn}: {len(only_v)})")
    if only_b or only_v:
        print(f"         NOTE: runs cover different ICs -> a fair total comparison "
              f"needs both runs over the SAME 100 ICs. Paired stats below use the "
              f"{len(common)} shared ICs only.")

    bs = b.loc[common, "success"].astype(bool)
    vs = v.loc[common, "success"].astype(bool)
    bp = b.loc[common, "progress"].astype(float)
    vp = v.loc[common, "progress"].astype(float)

    both = [ic for ic in common if bs[ic] and vs[ic]]
    neither = [ic for ic in common if not bs[ic] and not vs[ic]]
    v_only = [ic for ic in common if vs[ic] and not bs[ic]]   # verifier FIXES these
    b_only = [ic for ic in common if bs[ic] and not vs[ic]]   # verifier REGRESSES these

    print("-" * 68)
    print(f"  both succeed                         : {len(both)}")
    print(f"  both fail                            : {len(neither)}")
    print(f"  FIXED by verifier (V ok, B fail)     : {len(v_only)}  ICs {v_only}")
    print(f"  REGRESSED by verifier (B ok, V fail) : {len(b_only)}  ICs {b_only}")
    print("-" * 68)
    print(f"  paired success rate : {bn} {100 * bs.mean():5.1f}%  vs  "
          f"{vn} {100 * vs.mean():5.1f}%   (net {len(v_only) - len(b_only):+d} ICs)")
    print(f"  paired mean progress: {bn} {bp.mean():.3f}  vs  {vn} {vp.mean():.3f}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = pd.DataFrame({
        "episode": common,
        f"{bn}_success": bs.values,
        f"{vn}_success": vs.values,
        f"{bn}_progress": bp.values,
        f"{vn}_progress": vp.values,
    })

    def cat(r):
        if r[f"{bn}_success"] and r[f"{vn}_success"]:
            return "both_succeed"
        if not r[f"{bn}_success"] and not r[f"{vn}_success"]:
            return "both_fail"
        if r[f"{vn}_success"]:
            return "verifier_only"
        return "baseline_only"

    rows["category"] = rows.apply(cat, axis=1)
    rows["progress_delta"] = rows[f"{vn}_progress"] - rows[f"{bn}_progress"]
    rows.to_csv(out / "comparison.csv", index=False)
    print("=" * 68)
    print(f"saved {out / 'comparison.csv'}")
    print("  -> feed category=verifier_only into export_pairs.py for the 'money' clips")


if __name__ == "__main__":
    main()
