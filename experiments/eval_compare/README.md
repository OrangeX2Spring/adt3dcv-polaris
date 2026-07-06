# eval_compare — baseline vs V-JEPA-verifier analysis

Read-only analysis of PolaRiS eval runs. **None of these scripts modify existing
runs, videos, or code** — they only read `<run>/eval_results.csv` + `episode_<k>.mp4`
and write into their own `--out` folders.

Episodes are comparable across runs because `scripts/eval.py` rolls out
`initial_conditions[episode % N]`, so the `episode` column == the initial-condition
(IC) index.

## Scripts

| script | what | in | out |
| --- | --- | --- | --- |
| `compare_runs.py` | pair two runs by IC; overall + paired success/progress; auto-list ICs the verifier FIXED vs REGRESSED | 2 × `eval_results.csv` | `comparison.csv` |
| `failure_stages.py` | attribute each failure to the checker where it stalled; histogram | 1 × `eval_results.csv` | optional csv |
| `export_pairs.py` | side-by-side baseline\|verifier mp4 per IC with SUCCESS/FAIL banner | `episode_<k>.mp4` | new mp4s |

## Run (FoodBussing: goal_frames = no verifier, goal_jepa = verifier)

```bash
cd polaris/experiments/eval_compare
python compare_runs.py --baseline ../../runs/food_bussing_goal_frames \
    --verifier ../../runs/food_bussing_goal_jepa \
    --baseline-name noverif --verifier-name jepa --out out_food_bussing
python failure_stages.py --run ../../runs/food_bussing_goal_frames
python export_pairs.py --baseline ../../runs/food_bussing_goal_frames \
    --verifier ../../runs/food_bussing_goal_jepa \
    --baseline-name noverif --verifier-name jepa \
    --from-comparison out_food_bussing/comparison.csv --category verifier_only \
    --out pairs/verifier_fixed
```

## Findings so far (2026-07-06)

- The headline `22% (baseline, 100 IC) vs 18.3% (verifier, 60 IC)` is **not a fair
  comparison** — the two runs cover different IC sets. A clean total comparison needs
  both runs over the SAME 100 ICs (a GPU re-run).
- On the **60 shared ICs**: noverif 15.0% vs jepa 18.3% — verifier slightly ahead:
  **8 ICs fixed** (10,20,24,30,37,42,46,52), **6 regressed** (3,9,13,15,43,56), net +2.
- Dominant failure mode is **grasping**: failures stall most at `lift_ice_cream`,
  `reach_grapes`, `lift_grapes` — the policy reaches but fails to pick up. Few
  failures are at the final `inside_*_bowl` placement.

Outputs (`out_food_bussing/`, `pairs/`) are regeneratable and not committed.
