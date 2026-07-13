#!/usr/bin/env bash
# Scripted-expert data collection: >=1 successful episode per IC, max 5 attempts each.
# Prereqs: policy.py = policy_expert.py on the server, serve_policy running with
# POLARIS_IC_FILE set (see below). Run from /workspace/polaris.
#
# Usage:  bash experiments/expert_data/collect_expert.sh [first_ic] [last_ic]
set -u
trap 'echo; echo "[collect] interrupted by user, exiting"; exit 130' INT TERM
FIRST=${1:-0}
LAST=${2:-99}
STAGING=/workspace/polaris/runs/expert_staging
RUNS=/workspace/polaris/runs/expert_runs
FAILED_LIST=$RUNS/failed_ics.txt
mkdir -p "$STAGING" "$RUNS"

for ic in $(seq "$FIRST" "$LAST"); do
  # skip ICs that already have a staged success
  if ls "$STAGING"/ep_ic$(printf '%03d' "$ic")_*_success >/dev/null 2>&1; then
    echo "[collect] IC $ic already has a success, skipping"
    continue
  fi
  ok=0
  for attempt in 1 2 3 4 5; do
    folder=$RUNS/ic${ic}_a${attempt}
    echo "[collect] IC $ic attempt $attempt"
    uv run scripts/eval.py --environment DROID-FoodBussing --policy.port 8000 \
        --rollouts 1 --fix-ic "$ic" --send-subtask-state \
        --record-traj "$STAGING" \
        --run-folder "$folder"
    result=$(tail -1 "$folder/eval_results.csv" 2>/dev/null | cut -d, -f3,4)
    echo "[collect] IC $ic attempt $attempt -> success,progress = ${result:-NO CSV}"
    if grep -qi true "$folder/eval_results.csv" 2>/dev/null; then
      echo "[collect] IC $ic SUCCESS on attempt $attempt"
      ok=1
      break
    fi
  done
  if [ "$ok" -eq 0 ]; then
    echo "$ic" >> "$FAILED_LIST"
    echo "[collect] IC $ic FAILED after 5 attempts (recorded in $FAILED_LIST)"
  fi
done

echo
echo "staged successes: $(ls -d "$STAGING"/ep_*_success 2>/dev/null | wc -l)"
[ -f "$FAILED_LIST" ] && echo "failed ICs: $(cat "$FAILED_LIST" | tr '\n' ' ')"
