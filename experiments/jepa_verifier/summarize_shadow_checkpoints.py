#!/usr/bin/env python3
import argparse
import csv
import json
import statistics
from pathlib import Path


def load_rollouts(run_dir: Path) -> list[dict[str, str]]:
    rows = []
    for csv_path in sorted(run_dir.glob("ic*/eval_results.csv")):
        with csv_path.open(newline="") as handle:
            rows.extend(csv.DictReader(handle))
    return rows


def completion_ranks(records: list[dict]) -> list[int]:
    ranks = []
    previous = None
    previous_ic = None
    for record in records:
        done = record["done"]
        if previous_ic != record["ic"] or (
            previous is not None
            and any(was_done and not is_done for was_done, is_done in zip(previous["done"], done))
        ):
            previous = None
        if previous is not None and any(
            is_done and not was_done for was_done, is_done in zip(previous["done"], done)
        ):
            order = sorted(
                range(len(previous["energies"])),
                key=lambda index: previous["energies"][index],
            )
            ranks.append(order.index(previous["executed_idx"]) + 1)
        previous = record
        previous_ic = record["ic"]
    return ranks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/workspace/polaris/runs/jepa_eval_compare"),
    )
    args = parser.parse_args()

    print(
        "| checkpoint | rollouts | success | mean progress | decisions | "
        "median spread | spread >=2% | completion rank |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for run_dir in sorted(path for path in args.root.iterdir() if path.is_dir()):
        rollout_rows = load_rollouts(run_dir)
        energy_path = run_dir / "energies.jsonl"
        records = [
            json.loads(line)
            for line in energy_path.read_text().splitlines()
            if line.strip()
        ]
        spreads = [float(record["spread"]) for record in records]
        ranks = completion_ranks(records)
        successes = sum(row["success"].lower() == "true" for row in rollout_rows)
        progress = [float(row["progress"]) for row in rollout_rows]
        rank_text = f"{statistics.mean(ranks):.2f} ({len(ranks)})" if ranks else "n/a (0)"
        print(
            f"| {run_dir.name} | {len(rollout_rows)} | {successes} | "
            f"{statistics.mean(progress):.3f} | {len(records)} | "
            f"{100 * statistics.median(spreads):.2f}% | "
            f"{100 * sum(spread >= 0.02 for spread in spreads) / len(spreads):.1f}% | "
            f"{rank_text} |"
        )

    print("\ncompletion rank is the primary checkpoint metric; lower is better (1 is best of 10).")
    print("Shadow-mode success and progress do not measure checkpoint control quality.")


if __name__ == "__main__":
    main()
