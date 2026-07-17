#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <tag> <checkpoint.pt>" >&2
  exit 2
fi

TAG=$1
CHECKPOINT=$2
ROOT=/workspace/polaris
OPENPI=$ROOT/third_party/openpi
POLICIES=$OPENPI/src/openpi/policies
GOALS=$ROOT/runs/goals_subtask
OUT=$ROOT/runs/jepa_eval_compare/$TAG
PORT=8000
ICS=(2 9 20 21)

if [[ ! -f "$CHECKPOINT" ]]; then
  echo "Checkpoint not found: $CHECKPOINT" >&2
  exit 1
fi
if [[ ! -d "$GOALS" ]]; then
  echo "Goal directory not found: $GOALS" >&2
  exit 1
fi
if [[ -e "$OUT" ]]; then
  echo "Output already exists: $OUT" >&2
  echo "Use a new tag or move the existing directory first." >&2
  exit 1
fi

mkdir -p "$OUT"
cp "$POLICIES/policy_chunquan.py" "$POLICIES/policy.py"

SERVER_PID=
stop_server() {
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill -INT "$SERVER_PID"
    wait "$SERVER_PID" || true
  fi
}
trap stop_server EXIT INT TERM

cd "$OPENPI"
VJEPA_CHECKPOINT="$CHECKPOINT" \
VJEPA_SHADOW=1 \
VJEPA_SPREAD_GATE=0.02 \
VJEPA_LOG_ACTIONS=0 \
VJEPA_ENERGY_LOG="$OUT/energies.jsonl" \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.35 \
uv run scripts/serve_policy.py \
  --port "$PORT" \
  --goal-image-path "$GOALS" \
  --pytorch-device cuda:0 \
  policy:checkpoint \
  --policy.config pi05_droid_jointpos_polaris \
  --policy.dir gs://openpi-assets/checkpoints/polaris/pi05_droid_jointpos_polaris \
  >"$OUT/server.log" 2>&1 &
SERVER_PID=$!

for _ in $(seq 1 180); do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "Policy server exited during startup." >&2
    tail -n 50 "$OUT/server.log" >&2
    exit 1
  fi
  if grep -q "server listening on" "$OUT/server.log"; then
    break
  fi
  sleep 1
done
if ! grep -q "server listening on" "$OUT/server.log"; then
  echo "Policy server did not become ready within 180 seconds." >&2
  exit 1
fi
if ! grep -Fq "Loaded V-JEPA checkpoint: $CHECKPOINT" "$OUT/server.log"; then
  echo "Policy server did not confirm the requested checkpoint." >&2
  exit 1
fi

cd "$ROOT"
for IC in "${ICS[@]}"; do
  RUN_DIR=$OUT/ic$IC
  mkdir -p "$RUN_DIR"
  echo "[shadow] checkpoint=$TAG IC=$IC rollouts=2"
  uv run scripts/eval.py \
    --environment DROID-FoodBussing \
    --policy.port "$PORT" \
    --rollouts 2 \
    --fix-ic "$IC" \
    --send-subtask-state \
    --step-log \
    --run-folder "$RUN_DIR" \
    2>&1 | tee "$RUN_DIR/eval.log"
done

stop_server
trap - EXIT INT TERM
echo "Finished: $OUT"
