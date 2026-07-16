# FoodBussing Expert Data

This pipeline collects successful simulation episodes with the adaptive scripted expert, stages
camera frames plus joint states, and packs them into the DROID layout used by V-JEPA2-AC.

## 1. Start the expert server

On the simulation machine:

```bash
cd /workspace/polaris/third_party/openpi/src/openpi/policies
cp policy_expert.py policy.py

cd /workspace/polaris/third_party/openpi
POLARIS_IC_FILE=/workspace/polaris/PolaRiS-Hub/food_bussing/initial_conditions.json \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.35 \
uv run scripts/serve_policy.py --port 8000 \
  policy:checkpoint \
  --policy.config pi05_droid_jointpos_polaris \
  --policy.dir gs://openpi-assets/checkpoints/polaris/pi05_droid_jointpos_polaris
```

The startup log must say `Adaptive expert ready: 100 ICs`.

## 2. Run the repair gate

Use fresh test directories so previously staged successes do not get skipped. ICs 0 and 70 are
known-success controls; ICs 3, 24, and 57 are far-reach failures from the first batch.

```bash
cd /workspace/polaris
for ic in 0 3 24 57 70; do
  POLARIS_MAX_ATTEMPTS=3 \
  POLARIS_KEEP_FAILURES=1 \
  POLARIS_STAGING_DIR=/workspace/polaris/runs/expert_staging_repair_gate \
  POLARIS_RUN_ROOT=/workspace/polaris/runs/expert_runs_repair_gate \
  bash experiments/expert_data/collect_expert.sh "$ic" "$ic"
done
```

Do not start another full collection unless both control ICs and at least two of the three hard ICs
succeed. Each run prints criterion pass rates and failed-stage patterns automatically.

## 3. Collect all ICs

```bash
cd /workspace/polaris
bash experiments/expert_data/collect_expert.sh 0 99
```

Each IC gets up to 10 outer attempts. One `Ctrl+C` stops the loop and its active eval process.
Rerunning is safe: ICs with an existing staged success are skipped, while each invocation uses a
new run directory. Each attempt has a 20-minute watchdog so a blocked simulator or policy request
cannot stall the batch. Expert IK planning is separately bounded to three minutes so the policy
server also recovers from difficult replanning requests. Useful overrides:

```bash
POLARIS_MAX_ATTEMPTS=15 bash experiments/expert_data/collect_expert.sh 0 99
POLARIS_ATTEMPT_TIMEOUT_SECONDS=900 bash experiments/expert_data/collect_expert.sh 0 99
POLARIS_KEEP_FAILURES=1 bash experiments/expert_data/collect_expert.sh 0 0
```

Successful staging directories are under `runs/expert_staging/` and contain:

- `video.mp4`: external-camera frames at 3.75 fps
- `joints.npy`: synchronized `[T, 8]` arm and gripper states
- `meta.json`: IC, rubric result, sample rate, and frame-to-control-step mapping

The normal collector discards failed staging data. Set `POLARIS_KEEP_FAILURES=1` only when a
failure video is needed for diagnosis.

## 4. Pack training data

```bash
cd /workspace/polaris/third_party/openpi
uv run python /workspace/polaris/experiments/expert_data/pack_droid.py \
  --staging /workspace/polaris/runs/expert_staging \
  --out /workspace/polaris/runs/droid_foodbussing
```

The packer validates required files, joint shape, finite values, and video/state length before
writing `trajectory.h5`, `metadata.json`, `recordings/MP4/ext.mp4`, and `dataset.csv`.

Before training, the final line should report `distinct ICs: 100`.
