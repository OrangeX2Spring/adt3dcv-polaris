#!/usr/bin/env bash
# Collect at least one successful scripted-expert episode per initial condition.
# Run from /workspace/polaris after starting the adaptive expert on policy port 8000.
set -uo pipefail

FIRST=${1:-0}
LAST=${2:-99}
MAX_ATTEMPTS=${POLARIS_MAX_ATTEMPTS:-10}
STAGING=${POLARIS_STAGING_DIR:-/workspace/polaris/runs/expert_staging}
RUN_ROOT=${POLARIS_RUN_ROOT:-/workspace/polaris/runs/expert_runs}
COLLECTION_ID=${POLARIS_COLLECTION_ID:-$(date +%Y%m%d_%H%M%S)}
RUNS=$RUN_ROOT/$COLLECTION_ID
FAILED_LIST=$RUNS/failed_ics.txt
CURRENT_PID=""
KEEP_FAILURES=${POLARIS_KEEP_FAILURES:-0}

if ! [[ $FIRST =~ ^[0-9]+$ && $LAST =~ ^[0-9]+$ && $MAX_ATTEMPTS =~ ^[1-9][0-9]*$ ]]; then
  echo "usage: $0 [first_ic] [last_ic] (non-negative integers; POLARIS_MAX_ATTEMPTS > 0)" >&2
  exit 2
fi
if (( FIRST > LAST )); then
  echo "first_ic must be <= last_ic" >&2
  exit 2
fi

interrupt_collection() {
  trap - INT TERM
  echo
  echo "[collect] interrupted; stopping the active eval"
  if [[ -n $CURRENT_PID ]] && kill -0 "$CURRENT_PID" 2>/dev/null; then
    kill -INT -- "-$CURRENT_PID" 2>/dev/null || true
    for _ in $(seq 1 20); do
      kill -0 "$CURRENT_PID" 2>/dev/null || break
      sleep 0.1
    done
    if kill -0 "$CURRENT_PID" 2>/dev/null; then
      kill -TERM -- "-$CURRENT_PID" 2>/dev/null || true
    fi
    for _ in $(seq 1 20); do
      kill -0 "$CURRENT_PID" 2>/dev/null || break
      sleep 0.1
    done
    if kill -0 "$CURRENT_PID" 2>/dev/null; then
      kill -KILL -- "-$CURRENT_PID" 2>/dev/null || true
    fi
    wait "$CURRENT_PID" 2>/dev/null || true
  fi
  exit 130
}
trap interrupt_collection INT TERM

mkdir -p "$STAGING" "$RUNS"
: > "$FAILED_LIST"

read_result() {
  python3 -c 'import csv,sys
rows=list(csv.DictReader(open(sys.argv[1], newline="")))
if not rows: raise SystemExit(1)
row=rows[-1]
success=str(row.get("success", "")).strip().lower() == "true"
progress=row.get("progress", "")
print("{},{}".format(str(success).lower(), progress))' "$1"
}

run_eval() {
  local ic=$1
  local folder=$2
  set -- uv run scripts/eval.py --environment DROID-FoodBussing --policy.port 8000 \
    --rollouts 1 --fix-ic "$ic" --send-subtask-state --step-log --stop-on-success \
    --record-traj "$STAGING"
  if [[ $KEEP_FAILURES == 1 ]]; then
    set -- "$@" --record-keep-failures
  fi
  set -- "$@" --run-folder "$folder"
  python3 -c 'import os,signal,sys
os.setsid()
signal.signal(signal.SIGINT, signal.SIG_DFL)
signal.signal(signal.SIGTERM, signal.SIG_DFL)
os.execvp(sys.argv[1], sys.argv[1:])' "$@" &
  CURRENT_PID=$!
  wait "$CURRENT_PID"
  local status=$?
  CURRENT_PID=""
  return "$status"
}

echo "[collect] collection=$COLLECTION_ID ICs=$FIRST..$LAST attempts=$MAX_ATTEMPTS"
echo "[collect] runs=$RUNS"
echo "[collect] staging=$STAGING"

for ic in $(seq "$FIRST" "$LAST"); do
  ic_tag=$(printf '%03d' "$ic")
  existing_stage=$(find "$STAGING" -maxdepth 1 -type d \
    -name "ep_ic${ic_tag}_*_success" -print -quit)
  if [[ -n $existing_stage && -f $existing_stage/video.mp4 \
      && -f $existing_stage/joints.npy && -f $existing_stage/meta.json ]]; then
    echo "[collect] IC $ic already has a staged success; skipping"
    continue
  fi
  if [[ -n $existing_stage ]]; then
    echo "[collect] IC $ic has an incomplete staged success; recollecting" >&2
  fi

  succeeded=0
  for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
    folder=$RUNS/ic${ic_tag}_a$(printf '%02d' "$attempt")
    echo "[collect] IC $ic attempt $attempt/$MAX_ATTEMPTS"
    if ! run_eval "$ic" "$folder"; then
      echo "[collect] IC $ic attempt $attempt: eval exited with an error" >&2
      continue
    fi

    csv_path=$folder/eval_results.csv
    result=$(read_result "$csv_path" 2>/dev/null) || result=""
    if [[ -z $result ]]; then
      echo "[collect] IC $ic attempt $attempt: missing or empty CSV" >&2
      continue
    fi
    success=${result%%,*}
    progress=${result#*,}
    echo "[collect] IC $ic attempt $attempt -> success=$success progress=$progress"
    if [[ $success == true ]]; then
      staged_dir=$(find "$STAGING" -maxdepth 1 -type d \
        -name "ep_ic${ic_tag}_*_success" -print -quit)
      if [[ -z $staged_dir || ! -f $staged_dir/video.mp4 \
          || ! -f $staged_dir/joints.npy || ! -f $staged_dir/meta.json ]]; then
        echo "[collect] IC $ic succeeded but its staged training files are incomplete" >&2
        continue
      fi
      echo "[collect] IC $ic SUCCESS on attempt $attempt"
      succeeded=1
      break
    fi
  done

  if (( succeeded == 0 )); then
    echo "$ic" >> "$FAILED_LIST"
    echo "[collect] IC $ic FAILED after $MAX_ATTEMPTS attempts"
  fi
done

success_count=$(find "$STAGING" -maxdepth 1 -type d -name 'ep_*_success' | wc -l | tr -d ' ')
echo
echo "[collect] staged successes: $success_count"
if [[ -s $FAILED_LIST ]]; then
  echo "[collect] failed ICs: $(tr '\n' ' ' < "$FAILED_LIST")"
else
  echo "[collect] failed ICs: none"
fi
python3 experiments/expert_data/summarize_collection.py "$RUNS" || true
