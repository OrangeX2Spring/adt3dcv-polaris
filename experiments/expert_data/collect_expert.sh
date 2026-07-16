#!/usr/bin/env bash
# Collect at least one successful scripted-expert episode per initial condition.
# Run from /workspace/polaris after starting the adaptive expert on policy port 8000.
set -uo pipefail

FIRST=${1:-0}
LAST=${2:-99}
MAX_ATTEMPTS=${POLARIS_MAX_ATTEMPTS:-10}
BATCH_TIMEOUT_SECONDS=${POLARIS_BATCH_TIMEOUT_SECONDS:-${POLARIS_ATTEMPT_TIMEOUT_SECONDS:-3600}}
PROCESS_COOLDOWN_SECONDS=${POLARIS_PROCESS_COOLDOWN_SECONDS:-10}
MAX_PROCESS_RESTARTS=${POLARIS_MAX_PROCESS_RESTARTS:-3}
STAGING=${POLARIS_STAGING_DIR:-/workspace/polaris/runs/expert_staging}
RUN_ROOT=${POLARIS_RUN_ROOT:-/workspace/polaris/runs/expert_runs}
COLLECTION_ID=${POLARIS_COLLECTION_ID:-$(date +%Y%m%d_%H%M%S)}
RUNS=$RUN_ROOT/$COLLECTION_ID
FAILED_LIST=$RUNS/failed_ics.txt
ERROR_LIST=$RUNS/errored_ics.txt
HIGH_PROGRESS_FILE=$RUNS/high_progress_failed_ics.csv
CURRENT_PID=""
KEEP_FAILURES=${POLARIS_KEEP_FAILURES:-0}
SKIP_ICS=${POLARIS_SKIP_ICS:-}
HIGH_PROGRESS_THRESHOLD=${POLARIS_HIGH_PROGRESS_THRESHOLD:-0.8}
CUDA_OOM_STATUS=86
ACTIVE_EVAL_STATUS=87

if ! [[ $FIRST =~ ^[0-9]+$ && $LAST =~ ^[0-9]+$ \
    && $MAX_ATTEMPTS =~ ^[1-9][0-9]*$ \
    && $BATCH_TIMEOUT_SECONDS =~ ^[1-9][0-9]*$ \
    && $PROCESS_COOLDOWN_SECONDS =~ ^[0-9]+$ \
    && $MAX_PROCESS_RESTARTS =~ ^[1-9][0-9]*$ ]]; then
  echo "usage: $0 [first_ic] [last_ic] (invalid numeric collection setting)" >&2
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
  echo "GNU timeout is required for the per-process watchdog" >&2
  exit 2
fi
if command -v pgrep >/dev/null 2>&1; then
  active_evals=$(pgrep -af '[s]cripts/eval.py' 2>/dev/null || true)
  if [[ -n $active_evals ]]; then
    echo "[collect] refusing to start while another eval.py process is active:" >&2
    echo "$active_evals" >&2
    echo "[collect] stop the stale/parallel eval process, then rerun." >&2
    exit "$ACTIVE_EVAL_STATUS"
  fi
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
: > "$ERROR_LIST"
printf 'ic,best_progress,attempt\n' > "$HIGH_PROGRESS_FILE"

read_collection_state() {
  python3 -c 'import csv,sys
rows=list(csv.DictReader(open(sys.argv[1], newline="")))
if not rows: raise SystemExit(1)
def as_bool(value):
    return str(value or "").strip().lower() in {"1", "1.0", "true"}
def as_progress(row):
    try: return float(row.get("progress", -1))
    except (TypeError, ValueError): return -1.0
best_index=max(range(len(rows)), key=lambda index: as_progress(rows[index]))
print("{},{},{},{}".format(
    len(rows),
    str(any(as_bool(row.get("success")) for row in rows)).lower(),
    as_progress(rows[best_index]),
    best_index + 1,
))' "$1"
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
  local launch=$3
  local log_path=$folder/eval_launch_$(printf '%02d' "$launch").log
  local output_pipe=$folder/.eval_launch_$(printf '%02d' "$launch").pipe
  mkdir -p "$folder"
  rm -f "$output_pipe"
  mkfifo "$output_pipe"
  tee "$log_path" < "$output_pipe" &
  local tee_pid=$!
  set -- timeout --foreground --signal=INT --kill-after=30s \
    "${BATCH_TIMEOUT_SECONDS}s" \
    uv run scripts/eval.py --environment DROID-FoodBussing --policy.port 8000 \
    --rollouts "$MAX_ATTEMPTS" --fix-ic "$ic" --send-subtask-state \
    --step-log --stop-on-success \
    --record-traj "$STAGING"
  if [[ $KEEP_FAILURES == 1 ]]; then
    set -- "$@" --record-keep-failures
  fi
  set -- "$@" --run-folder "$folder"
  python3 -c 'import os,signal,sys
os.setsid()
signal.signal(signal.SIGINT, signal.SIG_DFL)
signal.signal(signal.SIGTERM, signal.SIG_DFL)
os.execvp(sys.argv[1], sys.argv[1:])' "$@" > "$output_pipe" 2>&1 &
  CURRENT_PID=$!
  wait "$CURRENT_PID"
  local status=$?
  CURRENT_PID=""
  wait "$tee_pid" 2>/dev/null || true
  rm -f "$output_pipe"
  if grep -Eqi \
      'CUDA error 2: out of memory|CUDA out of memory|Failed to create stream on device|cudaErrorMemoryAllocation|OutOfMemoryError' \
      "$log_path"; then
    return "$CUDA_OOM_STATUS"
  fi
  if (( status == 124 )); then
    echo "[collect] IC $ic process timed out after ${BATCH_TIMEOUT_SECONDS}s" >&2
  fi
  return "$status"
}

echo "[collect] collection=$COLLECTION_ID ICs=$FIRST..$LAST attempts=$MAX_ATTEMPTS"
echo "[collect] one Isaac process handles up to $MAX_ATTEMPTS attempts per IC"
echo "[collect] process timeout=${BATCH_TIMEOUT_SECONDS}s restarts=$MAX_PROCESS_RESTARTS"
echo "[collect] process cooldown=${PROCESS_COOLDOWN_SECONDS}s"
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

  folder=$RUNS/ic${ic_tag}
  completed_attempts=0
  succeeded=false
  best_progress=-1
  best_attempt=0
  process_launch=0
  while (( process_launch < MAX_PROCESS_RESTARTS )); do
    csv_path=$folder/eval_results.csv
    state=$(read_collection_state "$csv_path" 2>/dev/null) || state=""
    if [[ -n $state ]]; then
      IFS=, read -r completed_attempts succeeded best_progress best_attempt \
        <<< "$state"
    fi
    if [[ $succeeded == true ]] || (( completed_attempts >= MAX_ATTEMPTS )); then
      break
    fi

    process_launch=$((process_launch + 1))
    echo "[collect] IC $ic process launch $process_launch/$MAX_PROCESS_RESTARTS"
    echo "[collect] IC $ic completed attempts before launch: $completed_attempts/$MAX_ATTEMPTS"
    status=0
    run_eval "$ic" "$folder" "$process_launch" || status=$?
    if (( status == CUDA_OOM_STATUS )); then
      echo "[collect] CUDA OOM while starting/running IC $ic; stopping the collection" >&2
      echo "[collect] This is an infrastructure failure, not an IC failure." >&2
      echo "$ic" > "$RUNS/cuda_oom_ic.txt"
      if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi >&2 || true
      fi
      exit "$CUDA_OOM_STATUS"
    fi
    if (( status != 0 && status != 124 )); then
      fatal_log=$folder/eval_launch_$(printf '%02d' "$process_launch").log
      printf 'ic=%s status=%s log=%s\n' "$ic" "$status" "$fatal_log" \
        > "$RUNS/fatal_eval_error.txt"
      echo "[collect] IC $ic eval crashed with status $status; stopping the collection" >&2
      echo "[collect] log: $fatal_log" >&2
      exit "$status"
    fi
    if (( status == 0 )); then
      csv_path=$folder/eval_results.csv
      if ! post_state=$(read_collection_state "$csv_path"); then
        echo "[collect] IC $ic eval exited cleanly without readable results; stopping" >&2
        exit 1
      fi
      IFS=, read -r completed_attempts succeeded best_progress best_attempt \
        <<< "$post_state"
      if [[ $succeeded != true ]] && (( completed_attempts < MAX_ATTEMPTS )); then
        echo "[collect] IC $ic eval exited early after $completed_attempts/$MAX_ATTEMPTS attempts; stopping" >&2
        exit 1
      fi
    fi
    if (( PROCESS_COOLDOWN_SECONDS > 0 )); then
      echo "[collect] waiting ${PROCESS_COOLDOWN_SECONDS}s for GPU process cleanup"
      sleep "$PROCESS_COOLDOWN_SECONDS"
    fi
  done

  csv_path=$folder/eval_results.csv
  state=$(read_collection_state "$csv_path" 2>/dev/null) || state=""
  if [[ -n $state ]]; then
    IFS=, read -r completed_attempts succeeded best_progress best_attempt \
      <<< "$state"
  fi

  if [[ $succeeded == true ]]; then
    staged_dir=$(find "$STAGING" -maxdepth 1 -type d \
      -name "ep_ic${ic_tag}_*_success" -print -quit)
    if [[ -n $staged_dir && -f $staged_dir/video.mp4 \
        && -f $staged_dir/joints.npy && -f $staged_dir/meta.json ]]; then
      echo "[collect] IC $ic SUCCESS after $completed_attempts attempt(s)"
      continue
    fi
    echo "[collect] IC $ic succeeded but its staged training files are incomplete" >&2
    echo "$ic" >> "$ERROR_LIST"
    exit 1
  elif (( completed_attempts >= MAX_ATTEMPTS )); then
    echo "$ic" >> "$FAILED_LIST"
    echo "[collect] IC $ic FAILED after $MAX_ATTEMPTS attempts"
  else
    echo "$ic" >> "$ERROR_LIST"
    echo "[collect] IC $ic INCOMPLETE: $completed_attempts/$MAX_ATTEMPTS attempts recorded" >&2
  fi
  if progress_is_greater "$best_progress" "$HIGH_PROGRESS_THRESHOLD"; then
    printf '%s,%s,%s\n' "$ic" "$best_progress" "$best_attempt" \
      >> "$HIGH_PROGRESS_FILE"
    echo "[collect] IC $ic noted as high-progress failure: best=$best_progress"
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
if [[ -s $ERROR_LIST ]]; then
  echo "[collect] errored/incomplete ICs: $(tr '\n' ' ' < "$ERROR_LIST")"
else
  echo "[collect] errored/incomplete ICs: none"
fi
high_progress_count=$(( $(wc -l < "$HIGH_PROGRESS_FILE") - 1 ))
if (( high_progress_count > 0 )); then
  echo "[collect] failed ICs above $HIGH_PROGRESS_THRESHOLD progress:"
  tail -n +2 "$HIGH_PROGRESS_FILE"
else
  echo "[collect] failed ICs above $HIGH_PROGRESS_THRESHOLD progress: none"
fi
echo "[collect] high-progress report: $HIGH_PROGRESS_FILE"
python3 experiments/expert_data/summarize_collection.py "$RUNS"
