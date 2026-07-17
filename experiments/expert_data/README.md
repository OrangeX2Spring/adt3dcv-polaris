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

## 2. Collect all ICs

```bash
cd /workspace/polaris
bash experiments/expert_data/collect_expert.sh 0 99
```

Each IC gets up to 10 attempts inside one Isaac Sim process, avoiding repeated CUDA/Omniverse
startup and teardown. One `Ctrl+C` stops the loop and its active eval process. Rerunning is safe:
ICs with an existing staged success are skipped, while a timed-out IC process resumes from its
existing `eval_results.csv`. Each IC process has a one-hour watchdog, up to three process
launches, and a short GPU cleanup delay between launches. A detected CUDA OOM stops the collection
instead of incorrectly recording the remaining ICs as task failures. The collector also refuses to
start while another `scripts/eval.py` process is active. Any other eval crash, clean exit without
complete results, or incomplete staged success stops immediately instead of being retried or
reported as an IC failure. Useful overrides:

Simulator joint divergence, invalid live object poses, and safe-planning failures abort only the
current rollout. `eval.py` records its current rubric progress and resets without applying another
action, so the expert server and the remaining attempts continue running.

```bash
POLARIS_MAX_ATTEMPTS=15 bash experiments/expert_data/collect_expert.sh 0 99
POLARIS_BATCH_TIMEOUT_SECONDS=5400 bash experiments/expert_data/collect_expert.sh 0 99
POLARIS_MAX_PROCESS_RESTARTS=5 bash experiments/expert_data/collect_expert.sh 0 99
POLARIS_PROCESS_COOLDOWN_SECONDS=20 bash experiments/expert_data/collect_expert.sh 0 99
POLARIS_HIGH_PROGRESS_THRESHOLD=0.8 bash experiments/expert_data/collect_expert.sh 0 99
POLARIS_SKIP_ICS="24 57" bash experiments/expert_data/collect_expert.sh 0 99
POLARIS_KEEP_FAILURES=1 bash experiments/expert_data/collect_expert.sh 0 0
```

`POLARIS_SKIP_ICS` accepts comma- or space-separated IC indices that have been reviewed and accepted
as unrealizable. Skipped ICs are reported explicitly and are not added to the run's failure list.
The collector writes failed ICs whose best attempt exceeds `POLARIS_HIGH_PROGRESS_THRESHOLD` to
`high_progress_failed_ics.csv` in the new run directory, including their best progress and attempt.

Successful staging directories are under `runs/expert_staging/` and contain:

- `video.mp4`: external-camera frames at 3.75 fps
- `joints.npy`: synchronized `[T, 8]` arm and gripper states
- `meta.json`: IC, rubric result, sample rate, and frame-to-control-step mapping

The normal collector discards failed staging data. Set `POLARIS_KEEP_FAILURES=1` only when a
failure video is needed for diagnosis.

## 3. Pack training data

```bash
cd /workspace/polaris/third_party/openpi
uv run python /workspace/polaris/experiments/expert_data/pack_droid.py \
  --staging /workspace/polaris/runs/expert_staging \
  --out /workspace/polaris/runs/droid_foodbussing
```

The packer validates required files, joint shape, finite values, and video/state length before
writing `trajectory.h5`, `metadata.json`, `recordings/MP4/ext.mp4`, and `dataset.csv`.

For the current collection, the final line should report `49 episodes` and `distinct ICs: 49`.

## 4. Adapt the V-JEPA2-AC predictor

Stop Isaac Sim and the expert server first so this is the only process using the GPU. Install the
small set of data-loader dependencies into the PolaRiS environment if they are not already present:

```bash
cd /workspace/polaris
uv pip install --python .venv/bin/python pyyaml decord pandas h5py scipy timm einops iopath
```

The FoodBussing config initializes both networks from the trained `vjepa2-ac-vitg.pt` checkpoint,
keeps the encoder frozen, and updates only the action-conditioned predictor. It uses one random
8-frame clip from each episode per epoch, so 49 episodes and 20 epochs produce 980 optimizer steps.

```bash
cd /workspace/polaris/third_party/vjepa2
/workspace/polaris/.venv/bin/python -m app.main \
  --fname configs/train/vitg16/foodbussing-256px-8f.yaml \
  --devices cuda:0
```

The run writes `latest.pt` plus snapshots at epochs 5, 10, 15, and 20 under
`/workspace/polaris/runs/vjepa_foodbussing/`. Rerunning the same command resumes from `latest.pt`.
The original checkpoint is never overwritten.

To load the adapted checkpoint in the verifier, set:

```bash
VJEPA_CHECKPOINT=/workspace/polaris/runs/vjepa_foodbussing/latest.pt
```

on the same command that starts the OpenPI policy server. If the variable is omitted, the verifier
continues to load the original `vjepa2-ac-vitg.pt` checkpoint.
