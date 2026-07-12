import tyro
import mediapy

# import wandb
import json
import tqdm
import gymnasium as gym
import torch
import argparse
import pandas as pd


from pathlib import Path
from isaaclab.app import AppLauncher

from polaris.config import EvalArgs


def main(eval_args: EvalArgs):
    # This must be done before importing anything from IsaacLab
    # Inside main function to avoid launching IsaacLab in global scope
    # >>>> Isaac Sim App Launcher <<<<
    parser = argparse.ArgumentParser()
    args_cli, _ = parser.parse_known_args()
    args_cli.enable_cameras = True
    args_cli.headless = eval_args.headless
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app
    # >>>> Isaac Sim App Launcher <<<<

    from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
    from polaris.environments.manager_based_rl_splat_environment import (
        ManagerBasedRLSplatEnv,
    )
    from polaris.utils import load_eval_initial_conditions
    from polaris.policy import InferenceClient
    # from real2simeval.autoscoring import TASK_TO_SUCCESS_CHECKER

    env_cfg = parse_env_cfg(
        eval_args.environment,
        device="cuda",
        num_envs=1,
        use_fabric=True,
    )
    env: MangerBasedRLSplatEnv = gym.make(eval_args.environment, cfg=env_cfg)  # type: ignore

    language_instruction, initial_conditions = load_eval_initial_conditions(
        usd=env.usd_file,
        initial_conditions_file=eval_args.initial_conditions_file,
        # With --fix-ic, keep ALL ICs loaded (rollouts would truncate the list and any
        # fix_ic >= rollouts would go out of range); rollouts then means "number of repeats".
        rollouts=None if eval_args.fix_ic is not None else eval_args.rollouts,
    )
    if eval_args.fix_ic is not None:
        if not 0 <= eval_args.fix_ic < len(initial_conditions):
            raise ValueError(
                f"--fix-ic {eval_args.fix_ic} out of range (0..{len(initial_conditions) - 1})"
            )
        if eval_args.rollouts is None:
            raise ValueError("--fix-ic requires --rollouts (number of repeats)")
        rollouts = eval_args.rollouts
    else:
        rollouts = len(initial_conditions)

    def _reset_positions(ep: int):
        idx = eval_args.fix_ic if eval_args.fix_ic is not None else ep % len(initial_conditions)
        tag = "  (pinned via --fix-ic)" if eval_args.fix_ic is not None else ""
        print(f"[eval] rollout uses initial-condition index {idx}{tag}")
        return initial_conditions[idx]

    _STAGE_KEYS = [
        "c0_reach_ice_cream", "c1_reach_grapes", "c2_lift_ice_cream",
        "c3_lift_grapes", "c4_inside_ice_cream__bowl", "c5_inside_grapes_bowl",
    ]

    def _subtask_state(ep: int, latest_info) -> dict:
        """IC index + rubric done-mask for the subtask verifier's oracle goal switching."""
        metrics = (latest_info or {}).get("rubric", {}).get("metrics", {})
        return {
            "ic_index": eval_args.fix_ic if eval_args.fix_ic is not None else ep % len(initial_conditions),
            "done": [bool(metrics.get(f"{k}_ever", False)) for k in _STAGE_KEYS],
        }

    step_records: list[dict] = []
    # Resume CSV logging
    run_folder = Path(eval_args.run_folder)
    run_folder.mkdir(parents=True, exist_ok=True)
    csv_path = run_folder / "eval_results.csv"
    if csv_path.exists():
        episode_df = pd.read_csv(csv_path)
    else:
        episode_df = pd.DataFrame(
            {
                "episode": pd.Series(dtype="int"),
                "episode_length": pd.Series(dtype="int"),
                "success": pd.Series(dtype="bool"),
                "progress": pd.Series(dtype="float"),
            }
        )
    episode = len(episode_df)
    if episode >= rollouts:
        print("All rollouts have been evaluated. Exiting.")
        env.close()
        simulation_app.close()
        return

    policy_client: InferenceClient = InferenceClient.get_client(eval_args.policy)

    video = []
    horizon = env.max_episode_length
    bar = tqdm.tqdm(range(horizon))
    obs, info = env.reset(
        object_positions=_reset_positions(episode)
    )
    policy_client.reset()
    print(f" >>> Starting eval job from episode {episode + 1} of {rollouts} <<< ")
    while True:
        if eval_args.send_subtask_state:
            action, viz = policy_client.infer(
                obs, language_instruction, subtask_state=_subtask_state(episode, info)
            )
        else:
            action, viz = policy_client.infer(obs, language_instruction)
        if viz is not None:
            video.append(viz)
        obs, rew, term, trunc, info = env.step(
            torch.tensor(action).reshape(1, -1), expensive=policy_client.rerender
        )
        if eval_args.step_log:
            step_records.append(
                {
                    "step": bar.n,
                    "frame": max(0, len(video) - 1),
                    "progress": float(info["rubric"]["progress"]),
                    **{
                        k: bool(v)
                        for k, v in info["rubric"]["metrics"].items()
                        if k.endswith("_ever")
                    },
                }
            )

        bar.update(1)
        if term[0] or trunc[0] or bar.n >= horizon:
            policy_client.reset()

            # Save video and metadata
            filename = run_folder / f"episode_{episode}.mp4"
            mediapy.write_video(filename, video, fps=15)

            if eval_args.step_log and step_records:
                (run_folder / f"episode_{episode}_steps.jsonl").write_text(
                    "\n".join(json.dumps(r) for r in step_records)
                )
                step_records = []

            # Log episode results to CSV
            episode_data = {
                "episode": episode,
                "episode_length": bar.n,
                "success": info["rubric"]["success"],
                "progress": info["rubric"]["progress"],
            }
            # Per-checker metrics (r_c*_ever columns; summarize_repeats.py needs them
            # for the failure-stage histogram).
            episode_data.update(
                {f"r_{key}": value for key, value in info["rubric"]["metrics"].items()}
            )
            episode_df = pd.concat(
                [episode_df, pd.DataFrame([episode_data])], ignore_index=True
            )
            episode_df.to_csv(csv_path, index=False)

            bar.close()
            print(f"Episode {episode} finished. Episode length: {bar.n}")
            bar = tqdm.tqdm(range(horizon))
            obs, info = env.reset(
                object_positions=_reset_positions(episode)
            )

            episode += 1
            video = []
            if episode >= rollouts:
                break

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    args: EvalArgs = tyro.cli(EvalArgs)
    main(args)