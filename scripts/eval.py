import tyro
import mediapy

# import wandb
import tqdm
import gymnasium as gym
import torch
import argparse
import pandas as pd
import numpy as np
import cv2


from pathlib import Path
from isaaclab.app import AppLauncher

from polaris.config import EvalArgs


def _select_goal_frame(images: dict, camera: str):
    if camera == "both":
        return np.concatenate([images["external_cam"], images["wrist_cam"]], axis=1)
    if camera not in images:
        raise ValueError(
            f"Goal frame camera '{camera}' not found. Available cameras: {list(images.keys())}"
        )
    return images[camera]


def _save_goal_frame(env, run_folder: Path, episode: int, tag: str, camera: str):
    goal_dir = run_folder / "goal_frames"
    goal_dir.mkdir(parents=True, exist_ok=True)
    images = env.custom_render(expensive=True)
    frame = np.asarray(_select_goal_frame(images, camera))
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    path = goal_dir / f"episode_{episode:04d}_{tag}_{camera}.jpg"
    if frame.shape[-1] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
    else:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95]):
        raise RuntimeError(f"Failed to save goal frame to {path}")
    print(f"Saved {tag} goal frame to {path}")


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
        rollouts=eval_args.rollouts,
    )
    rollouts = len(initial_conditions)
    # Resume CSV logging
    run_folder = Path(eval_args.run_folder)
    run_folder.mkdir(parents=True, exist_ok=True)
    if eval_args.goal_frame_when not in {"success", "final", "both"}:
        raise ValueError("goal_frame_when must be one of: success, final, both")
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
        object_positions=initial_conditions[episode % len(initial_conditions)]
    )
    policy_client.reset()
    success_goal_frame_saved = False
    print(f" >>> Starting eval job from episode {episode + 1} of {rollouts} <<< ")
    while True:
        action, viz = policy_client.infer(obs, language_instruction)
        if viz is not None:
            video.append(viz)
        obs, rew, term, trunc, info = env.step(
            torch.tensor(action).reshape(1, -1), expensive=policy_client.rerender
        )
        if (
            eval_args.save_goal_frames
            and eval_args.goal_frame_when in {"success", "both"}
            and not success_goal_frame_saved
            and info["rubric"]["success"]
        ):
            _save_goal_frame(
                env,
                run_folder,
                episode,
                "success",
                eval_args.goal_frame_camera,
            )
            success_goal_frame_saved = True

        bar.update(1)
        if term[0] or trunc[0] or bar.n >= horizon:
            policy_client.reset()
            if eval_args.save_goal_frames and eval_args.goal_frame_when in {
                "final",
                "both",
            }:
                _save_goal_frame(
                    env,
                    run_folder,
                    episode,
                    "final",
                    eval_args.goal_frame_camera,
                )

            # Save video and metadata
            filename = run_folder / f"episode_{episode}.mp4"
            mediapy.write_video(filename, video, fps=15)

            # Log episode results to CSV
            episode_data = {
                "episode": episode,
                "episode_length": bar.n,
                "success": info["rubric"]["success"],
                "progress": info["rubric"]["progress"],
            }
            episode_data.update(
                {
                    f"r_{key}": value
                    for key, value in info["rubric"]["metrics"].items()
                }
            )
            episode_df = pd.concat(
                [episode_df, pd.DataFrame([episode_data])], ignore_index=True
            )
            episode_df.to_csv(csv_path, index=False)

            bar.close()
            print(f"Episode {episode} finished. Episode length: {bar.n}")
            bar = tqdm.tqdm(range(horizon))
            obs, info = env.reset(
                object_positions=initial_conditions[episode % len(initial_conditions)]
            )

            episode += 1
            video = []
            success_goal_frame_saved = False
            if episode >= rollouts:
                break

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    args: EvalArgs = tyro.cli(EvalArgs)
    main(args)
