#!/usr/bin/env bash
# Collect at least one successful scripted-expert episode per initial condition.
# Run from /workspace/polaris after starting the adaptive expert on policy port 8000.
set -uo pipefail

FIRST=${1:-0}
LAST=${2:-99}
MAX_ATTEMPTS=${POLARIS_MAX_ATTEMPTS:-10}
ATTEMPT_TIMEOUT_SECONDS=${POLARIS_ATTEMPT_TIMEOUT_SECONDS:-1200}
STAGING=${POLARIS_STAGING_DIR:-/workspace/polaris/runs/expert_staging}
RUN_ROOT=${POLARIS_RUN_ROOT:-/workspace/polaris/runs/expert_runs}
COLLECTION_ID=${POLARIS_COLLECTION_ID:-$(date +%Y%m%d_%H%M%S)}
RUNS=$RUN_ROOT/$COLLECTION_ID
FAILED_LIST=$RUNS/failed_ics.txt
HIGH_PROGRESS_FILE=$RUNS/high_progress_failed_ics.csv
CURRENT_PID=""
KEEP_FAILURES=${POLARIS_KEEP_FAILURES:-0}
SKIP_ICS=${POLARIS_SKIP_ICS:-}
HIGH_PROGRESS_THRESHOLD=${POLARIS_HIGH_PROGRESS_THRESHOLD:-0.8}

if ! [[ $FIRST =~ ^[0-9]+$ && $LAST =~ ^[0-9]+$ \
    && $MAX_ATTEMPTS =~ ^[1-9][0-9]*$ \
    && $ATTEMPT_TIMEOUT_SECONDS =~ ^[1-9][0-9]*$ ]]; then
  echo "usage: $0 [first_ic] [last_ic] (non-negative integers; attempts and timeout > 0)" >&2
  exit 2
fi
if (( FIRST > LAST )); then
  echo "first_ic must be <= last_ic" >&2
  exit 2
fi
if ! python3 -c 'import math,sys
try: value=float(sys.argv[1])
except ValueError: raise SystemExit(1)
raise SystemExit(0 if math.isfinite(value) and 0 <= value < 1 else 1)' \
    "$HIGH_PROGRESS_THRESHOLD"; then
  echo "POLARIS_HIGH_PROGRESS_THRESHOLD must be a finite number in [0, 1)" >&2
  exit 2
fi
if ! command -v timeout >/dev/null 2>&1; then
  echo "GNU timeout is required for the per-attempt watchdog" >&2
  exit 2
fi

is_skipped_ic() {
  local candidate=$1
  local skipped
  for skipped in ${SKIP_ICS//,/ }; do
    if ! [[ $skipped =~ ^[0-9]+$ ]]; then
      echo "POLARIS_SKIP_ICS must contain only comma/space-separated integers" >&2
      exit 2
    fi
    if [[ $candidate == "$skipped" ]]; then
      return 0
    fi
  done
  return 1
}

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
printf 'ic,best_progress,attempt\n' > "$HIGH_PROGRESS_FILE"

read_result() {
  python3 -c 'import csv,sys
rows=list(csv.DictReader(open(sys.argv[1], newline="")))
if not rows: raise SystemExit(1)
row=rows[-1]
success=str(row.get("success", "")).strip().lower() == "true"
progress=row.get("progress", "")
print("{},{}".format(str(success).lower(), progress))' "$1"
}

progress_is_greater() {
  python3 -c 'import math,sys
try: value=float(sys.argv[1]); reference=float(sys.argv[2])
except ValueError: raise SystemExit(1)
raise SystemExit(0 if math.isfinite(value) and value > reference else 1)' "$1" "$2"
}

run_eval() {
  local ic=$1
  local folder=$2
  set -- timeout --foreground --signal=INT --kill-after=30s \
    "${ATTEMPT_TIMEOUT_SECONDS}s" \
    uv run scripts/eval.py --environment DROID-FoodBussing --policy.port 8000 \
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
  if (( status == 124 )); then
    echo "[collect] IC $ic attempt timed out after ${ATTEMPT_TIMEOUT_SECONDS}s" >&2
  fi
  return "$status"
}

echo "[collect] collection=$COLLECTION_ID ICs=$FIRST..$LAST attempts=$MAX_ATTEMPTS"
echo "[collect] attempt timeout=${ATTEMPT_TIMEOUT_SECONDS}s"
echo "[collect] explicitly skipped ICs=${SKIP_ICS:-none}"
echo "[collect] high-progress failure threshold: progress > ${HIGH_PROGRESS_THRESHOLD}"
echo "[collect] runs=$RUNS"
echo "[collect] staging=$STAGING"

for ic in $(seq "$FIRST" "$LAST"); do
  if is_skipped_ic "$ic"; then
    echo "[collect] IC $ic explicitly accepted as skipped; not attempting"
    continue
  fi
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
  best_progress=-1
  best_attempt=0
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
    if progress_is_greater "$progress" "$best_progress"; then
      best_progress=$progress
      best_attempt=$attempt
    fi
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
    if progress_is_greater "$best_progress" "$HIGH_PROGRESS_THRESHOLD"; then
      printf '%s,%s,%s\n' "$ic" "$best_progress" "$best_attempt" \
        >> "$HIGH_PROGRESS_FILE"
      echo "[collect] IC $ic noted as high-progress failure: best=$best_progress"
    fi
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
high_progress_count=$(( $(wc -l < "$HIGH_PROGRESS_FILE") - 1 ))
if (( high_progress_count > 0 )); then
  echo "[collect] failed ICs above $HIGH_PROGRESS_THRESHOLD progress:"
  tail -n +2 "$HIGH_PROGRESS_FILE"
else
  echo "[collect] failed ICs above $HIGH_PROGRESS_THRESHOLD progress: none"
fi
echo "[collect] high-progress report: $HIGH_PROGRESS_FILE"
python3 experiments/expert_data/summarize_collection.py "$RUNS" || true
