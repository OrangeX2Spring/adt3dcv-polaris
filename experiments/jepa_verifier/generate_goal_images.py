"""
Generate a per-initial-condition GOAL image for V-JEPA verifier.

Why: the bowl position varies across the 100 initial conditions, so a single goal image
(from one episode) is spatially misaligned with every other scene — latent-L1-to-goal then
carries no progress signal (see experiments/jepa_verifier/REPORT). Fix: for each IC, stage the
success state directly in sim (place the foods `ice_cream_` and `grapes` into that IC's `bowl`),
render one frame from the SAME observation path the policy sees (`obs["splat"]["external_cam"]`,
resized with pad to 224 exactly like droid_jointpos_client), and save it as that IC's goal.

Runs on the eval machine (needs IsaacLab). Example:
  cd /workspace/polaris
  uv run python experiments/jepa_verifier/generate_goal_images.py \
      --environment DROID-FoodBussing \
      --out-dir /workspace/polaris/runs/goals_food_bussing \
      --headless

Then serve with per-episode goals (needs the serve-side plumbing) or, as a quick check, point
--goal-image-path at one of these to sanity-check a single scene.
"""
import argparse
from pathlib import Path

import numpy as np


def build_goal_poses(ic_pose: dict, foods, container, z_offset, xy_spread):
    """Copy the IC poses but move each food to the container's xy, lifted into the bowl."""
    goal = {k: list(v) for k, v in ic_pose.items()}
    cx, cy, cz = ic_pose[container][0], ic_pose[container][1], ic_pose[container][2]
    # spread foods around the container center so they don't occupy identical space
    offsets = np.linspace(-xy_spread, xy_spread, len(foods)) if len(foods) > 1 else [0.0]
    for food, dx in zip(foods, offsets):
        if food not in goal:
            continue
        p = goal[food]
        p[0] = cx + float(dx)          # x -> container x (+ spread)
        p[1] = cy                      # y -> container y
        p[2] = cz + z_offset           # z -> above container base
        goal[food] = p                 # keep original orientation (p[3:7])
    return goal


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--environment", default="DROID-FoodBussing")
    ap.add_argument("--initial-conditions-file", default=None)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--rollouts", type=int, default=None, help="limit number of ICs")
    ap.add_argument("--foods", nargs="+", default=["ice_cream_", "grapes"])
    ap.add_argument("--container", default="bowl")
    ap.add_argument("--z-offset", type=float, default=0.03, help="food height above container base (m)")
    ap.add_argument("--xy-spread", type=float, default=0.015, help="lateral spacing between foods (m)")
    ap.add_argument("--camera", default="external_cam")
    ap.add_argument("--headless", action="store_true")
    args = ap.parse_args()

    # >>>> IsaacLab launcher (must precede any IsaacLab import), mirroring scripts/eval.py <<<<
    import argparse as _argparse
    from isaaclab.app import AppLauncher

    _cli = _argparse.Namespace()
    _cli.enable_cameras = True
    _cli.headless = args.headless
    app_launcher = AppLauncher(_cli)
    simulation_app = app_launcher.app  # noqa: F841

    import gymnasium as gym
    from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
    from polaris.environments.manager_based_rl_splat_environment import ManagerBasedRLSplatEnv
    from polaris.utils import load_eval_initial_conditions
    from openpi_client import image_tools
    import mediapy

    env_cfg = parse_env_cfg(args.environment, device="cuda", num_envs=1, use_fabric=True)
    env: ManagerBasedRLSplatEnv = gym.make(args.environment, cfg=env_cfg)  # type: ignore

    instruction, initial_conditions = load_eval_initial_conditions(
        usd=env.usd_file,
        initial_conditions_file=args.initial_conditions_file,
        rollouts=args.rollouts,
    )
    print(f"instruction: {instruction}")
    print(f"{len(initial_conditions)} initial conditions; foods={args.foods} container={args.container}")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    for i, ic_pose in enumerate(initial_conditions):
        goal_poses = build_goal_poses(
            ic_pose, args.foods, args.container, args.z_offset, args.xy_spread
        )
        obs, _ = env.reset(object_positions=goal_poses, expensive=True)
        frame = np.asarray(obs["splat"][args.camera])
        if frame.dtype != np.uint8:
            frame = (255 * np.clip(frame, 0, 1)).astype(np.uint8)
        # match the exact obs path the policy sees (droid_jointpos_client resizes to 224)
        frame224 = image_tools.resize_with_pad(frame, 224, 224)
        path = out / f"goal_{i:04d}.png"
        mediapy.write_image(path, frame224)
        if i % 10 == 0:
            print(f"[{i:>3}/{len(initial_conditions)}] saved {path}")

    print(f"\nDone. {len(initial_conditions)} goal images in {out}")
    print("Next: validate offline with offline_encoder_check.py (per-episode goal), then wire "
          "per-episode goal switching into the serve path.")
    simulation_app.close()


if __name__ == "__main__":
    main()
