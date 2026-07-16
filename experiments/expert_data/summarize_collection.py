"""Summarize scripted-expert collection failures by rubric stage."""

import argparse
import csv
import re
from collections import Counter, defaultdict
from pathlib import Path


METRICS = (
    ("r_c0_reach_ice_cream_ever", "reach_ice"),
    ("r_c1_reach_grapes_ever", "reach_grapes"),
    ("r_c2_lift_ice_cream_ever", "lift_ice"),
    ("r_c3_lift_grapes_ever", "lift_grapes"),
    ("r_c4_inside_ice_cream__bowl_ever", "inside_ice"),
    ("r_c5_inside_grapes_bowl_ever", "inside_grapes"),
)


def as_bool(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "1.0", "true"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()

    attempts_by_ic: dict[int, list[dict]] = defaultdict(list)
    for csv_path in sorted(args.run_dir.glob("ic*/eval_results.csv")):
        match = re.fullmatch(r"ic(\d+)(?:_a(\d+))?", csv_path.parent.name)
        if match is None:
            continue
        rows = list(csv.DictReader(csv_path.open(newline="")))
        if not rows:
            continue
        ic_index = int(match.group(1))
        folder_attempt = int(match.group(2)) if match.group(2) is not None else None
        for row_index, row in enumerate(rows, start=1):
            attempts_by_ic[ic_index].append(
                {
                    "attempt": folder_attempt or row_index,
                    "success": as_bool(row.get("success")),
                    "progress": float(row.get("progress", 0.0) or 0.0),
                    "metrics": tuple(as_bool(row.get(column)) for column, _ in METRICS),
                }
            )

    attempts = [attempt for values in attempts_by_ic.values() for attempt in values]
    successful_ics = sum(any(attempt["success"] for attempt in values) for values in attempts_by_ic.values())
    print(
        f"[summary] attempted ICs={len(attempts_by_ic)} attempts={len(attempts)} "
        f"successful ICs={successful_ics}"
    )
    if not attempts:
        return

    print("[summary] attempt-level pass rates:")
    for metric_index, (_, label) in enumerate(METRICS):
        passed = sum(attempt["metrics"][metric_index] for attempt in attempts)
        print(f"  {label:14s} {passed:4d}/{len(attempts):4d} = {passed / len(attempts):6.1%}")

    failures = {
        ic_index: values
        for ic_index, values in attempts_by_ic.items()
        if not any(attempt["success"] for attempt in values)
    }
    patterns = Counter()
    for values in failures.values():
        best = max(values, key=lambda attempt: (sum(attempt["metrics"]), attempt["progress"]))
        patterns[best["metrics"]] += 1

    print("[summary] failed-IC best-attempt patterns:")
    for metrics, count in patterns.most_common():
        passed = ",".join(label for (_, label), value in zip(METRICS, metrics) if value)
        missing = ",".join(label for (_, label), value in zip(METRICS, metrics) if not value)
        print(f"  {count:3d} passed=[{passed or 'none'}] missing=[{missing or 'none'}]")


if __name__ == "__main__":
    main()
