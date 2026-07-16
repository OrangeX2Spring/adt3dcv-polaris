"""Adaptive scripted expert for DROID-FoodBussing data collection.

The expert executes one pickup/drop leg at a time and uses the rubric state sent by
``scripts/eval.py --send-subtask-state`` to decide what to do next. Failed legs are retried
inside the same episode with deterministic grasp-orientation and position variants; completed
foods are never touched again.

Deploy by copying this file to ``policy.py`` and starting ``serve_policy.py`` with
``POLARIS_IC_FILE`` pointing to FoodBussing's ``initial_conditions.json``.

Important tuning variables (meters / radians / control steps):
  EXPERT_TCP_OFFSET=0.105
  EXPERT_GRASP_OFFSET=0.02
  EXPERT_GRASP_OFFSET_GRAPES=0.005
  EXPERT_HOVER=0.12
  EXPERT_TRANSIT=0.20
  EXPERT_DROP=0.09
  EXPERT_DWELL=10
  EXPERT_MAX_DQ=0.05
  EXPERT_MAX_INTERPOLATION_STEPS=256
  EXPERT_JITTER=0.005
  EXPERT_DROP_SPREAD=0.02
  EXPERT_BOWL_CLEARANCE=0.13
  EXPERT_GRASP_SHIFT_MAX=0.008
  EXPERT_IK_ROT_WEIGHT=0.20
  EXPERT_IK_POS_TOL=0.004
  EXPERT_IK_ROT_TOL=0.05
  EXPERT_MAX_IK_POS_ERROR=0.12
  EXPERT_MIN_OBJECT_Z=-0.02
  EXPERT_MAX_OBJECT_Z=0.60
  EXPERT_MAX_OBJECT_RADIUS=1.00
  EXPERT_PLAN_TIMEOUT=180
  EXPERT_YAW_ALIGN=1
  EXPERT_GRIP_INVERT=0
"""

from collections.abc import Sequence
import json
import logging
import os
import pathlib
import time
from typing import Any, TypeAlias

import numpy as np
from openpi_client import base_policy as _base_policy
from scipy.spatial.transform import Rotation
from typing_extensions import override

from vjepa2.FK import PandaFK


BasePolicy: TypeAlias = _base_policy.BasePolicy

CHUNK = 16
ADVANCE = 8
GRASP_VARIANTS = 10
GRASP_HEIGHT_DELTAS = (0.0, -0.006, 0.006, -0.012, 0.012, -0.003, 0.009, -0.009, 0.015, 0.003)
FOOD_SPECS = (
    ("grapes", 5, 1.0),
    ("ice_cream", 4, -1.0),
)
FOOD_RETRY_WEIGHTS = {"grapes": 2.0, "ice_cream": 1.0}


class Policy(BasePolicy):
    NEEDS_MODEL = False

    def __init__(
        self,
        model: Any = None,
        *,
        rng: Any = None,
        transforms: Sequence[Any] = (),
        output_transforms: Sequence[Any] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        pytorch_device: str = "cpu",
        is_pytorch: bool = False,
        goal_image_path: str | pathlib.Path | None = None,
    ):
        self._metadata = metadata or {}
        ic_file = os.environ.get("POLARIS_IC_FILE")
        if not ic_file or not pathlib.Path(ic_file).is_file():
            raise FileNotFoundError(
                "Scripted expert needs POLARIS_IC_FILE=<path to initial_conditions.json> "
                f"(got: {ic_file!r})"
            )

        data = json.loads(pathlib.Path(ic_file).read_text())
        self._poses: list[dict] = data["poses"]
        self._robot = PandaFK(device="cpu")

        self._tcp = float(os.environ.get("EXPERT_TCP_OFFSET", 0.105))
        self._grasp_off = float(os.environ.get("EXPERT_GRASP_OFFSET", 0.02))
        self._grasp_off_grapes = float(
            os.environ.get("EXPERT_GRASP_OFFSET_GRAPES", 0.005)
        )
        self._hover = float(os.environ.get("EXPERT_HOVER", 0.12))
        self._transit = float(os.environ.get("EXPERT_TRANSIT", 0.20))
        self._drop = float(os.environ.get("EXPERT_DROP", 0.09))
        self._dwell = int(os.environ.get("EXPERT_DWELL", 10))
        self._max_dq = float(os.environ.get("EXPERT_MAX_DQ", 0.05))
        self._max_interpolation_steps = int(
            os.environ.get("EXPERT_MAX_INTERPOLATION_STEPS", 256)
        )
        self._jitter = float(os.environ.get("EXPERT_JITTER", 0.005))
        self._drop_spread = float(os.environ.get("EXPERT_DROP_SPREAD", 0.02))
        self._bowl_clearance = float(os.environ.get("EXPERT_BOWL_CLEARANCE", 0.13))
        self._grasp_shift_max = float(os.environ.get("EXPERT_GRASP_SHIFT_MAX", 0.008))
        self._via_distance = float(os.environ.get("EXPERT_VIA_DISTANCE", 0.15))
        self._ik_rot_weight = float(os.environ.get("EXPERT_IK_ROT_WEIGHT", 0.20))
        self._ik_pos_tol = float(os.environ.get("EXPERT_IK_POS_TOL", 0.004))
        self._ik_rot_tol = float(os.environ.get("EXPERT_IK_ROT_TOL", 0.05))
        self._max_ik_position_error = float(
            os.environ.get("EXPERT_MAX_IK_POS_ERROR", 0.12)
        )
        self._min_object_z = float(os.environ.get("EXPERT_MIN_OBJECT_Z", -0.02))
        self._max_object_z = float(os.environ.get("EXPERT_MAX_OBJECT_Z", 0.60))
        self._max_object_radius = float(
            os.environ.get("EXPERT_MAX_OBJECT_RADIUS", 1.00)
        )
        self._plan_timeout = float(os.environ.get("EXPERT_PLAN_TIMEOUT", 180))
        self._yaw_align = os.environ.get("EXPERT_YAW_ALIGN", "1") == "1"
        invert = os.environ.get("EXPERT_GRIP_INVERT", "0") == "1"
        self._closed, self._open = (0.0, 1.0) if invert else (1.0, 0.0)

        if (
            self._max_dq <= 0
            or self._max_interpolation_steps <= 0
            or self._via_distance <= 0
            or self._ik_rot_weight < 0
            or self._ik_pos_tol <= 0
            or self._ik_rot_tol <= 0
            or self._max_ik_position_error <= 0
            or self._min_object_z >= self._max_object_z
            or self._max_object_radius <= 0
            or self._plan_timeout <= 0
        ):
            raise ValueError(
                "EXPERT_MAX_DQ, EXPERT_MAX_INTERPOLATION_STEPS, and "
                "EXPERT_VIA_DISTANCE must be positive; EXPERT_IK_POS_TOL and "
                "EXPERT_IK_ROT_TOL must be positive; object workspace and IK "
                "error limits must be valid; EXPERT_PLAN_TIMEOUT must be positive; "
                "EXPERT_IK_ROT_WEIGHT must be non-negative"
            )

        self._plan: np.ndarray | None = None
        self._motion_steps = 0
        self._ptr = 0
        self._episode_attempt = 0
        self._episode_ic: int | None = None
        self._q_home: np.ndarray | None = None
        self._active_food: str | None = None
        self._food_retries: dict[str, int] = {}

        logging.info(
            "Adaptive expert ready: %d ICs from %s | grasp=%.3f grapes=%.3f "
            "drop=%.3f dwell=%d max_dq=%.3f jitter=%.3f shift_max=%.3f ik_rot=%.2f",
            len(self._poses),
            ic_file,
            self._grasp_off,
            self._grasp_off_grapes,
            self._drop,
            self._dwell,
            self._max_dq,
            self._jitter,
            self._grasp_shift_max,
            self._ik_rot_weight,
        )

    @staticmethod
    def _find(pose_dict: dict, needle: str) -> np.ndarray:
        matches = [np.asarray(value, dtype=np.float32) for key, value in pose_dict.items() if needle in key]
        if len(matches) != 1:
            raise KeyError(
                f"Expected exactly one object matching {needle!r}; found {len(matches)} "
                f"in keys {list(pose_dict)}"
            )
        return matches[0]

    @staticmethod
    def _food_spec(food_name: str) -> tuple[str, int, float]:
        for spec in FOOD_SPECS:
            if spec[0] == food_name:
                return spec
        raise KeyError(food_name)

    @staticmethod
    def _object_yaw(object_pose: np.ndarray) -> float:
        quaternion = np.asarray(object_pose[3:7], dtype=np.float64)
        norm = float(np.linalg.norm(quaternion))
        if norm < 1e-8:
            return 0.0
        quaternion /= norm
        rotation = Rotation.from_quat(
            [quaternion[1], quaternion[2], quaternion[3], quaternion[0]]
        )
        axes = rotation.apply(np.eye(3))
        axis = axes[int(np.argmax(np.linalg.norm(axes[:, :2], axis=1)))]
        return float(np.arctan2(axis[1], axis[0]))

    def _grasp_orientation(
        self,
        home_rpy: np.ndarray,
        object_pose: np.ndarray,
        bowl_xyz: np.ndarray,
        variant_index: int,
    ) -> tuple[np.ndarray, float, str]:
        if not self._yaw_align:
            return np.asarray(home_rpy, dtype=np.float32), 0.0, "fixed"

        home_rotation = Rotation.from_euler("xyz", home_rpy)
        close_direction = home_rotation.apply([0.0, 1.0, 0.0])
        home_close_yaw = float(np.arctan2(close_direction[1], close_direction[0]))
        object_yaw = self._object_yaw(object_pose)
        object_close_yaw = object_yaw + np.pi / 2

        away = np.asarray(object_pose[:2]) - np.asarray(bowl_xyz[:2])
        distance = float(np.linalg.norm(away))
        tangent_yaw = float(np.arctan2(away[1], away[0]) + np.pi / 2)
        if 1e-6 < distance < self._bowl_clearance:
            candidates = (
                (object_close_yaw, "object-perpendicular"),
                (tangent_yaw, "bowl-tangent"),
                (object_close_yaw + np.pi / 2, "object-parallel"),
                (object_close_yaw + np.pi / 4, "object-plus45"),
                (object_close_yaw - np.pi / 4, "object-minus45"),
            )
        else:
            candidates = (
                (object_close_yaw, "object-perpendicular"),
                (object_close_yaw + np.pi / 4, "object-plus45"),
                (object_close_yaw - np.pi / 4, "object-minus45"),
                (object_close_yaw + np.pi / 2, "object-parallel"),
                (object_close_yaw + np.pi / 8, "object-plus22"),
            )

        desired_yaw, mode = candidates[variant_index % len(candidates)]
        yaw_delta = float(desired_yaw - home_close_yaw)
        yaw_delta = float((yaw_delta + np.pi / 2) % np.pi - np.pi / 2)
        grasp_rotation = Rotation.from_euler("z", yaw_delta) * home_rotation
        return grasp_rotation.as_euler("xyz").astype(np.float32), yaw_delta, mode

    @staticmethod
    def _rotation_error(target_rpy: np.ndarray, actual_rpy: np.ndarray) -> float:
        target = Rotation.from_euler("xyz", target_rpy)
        actual = Rotation.from_euler("xyz", actual_rpy)
        return float((target * actual.inv()).magnitude())

    def _sanitize_observed_joints(self, joints: np.ndarray) -> np.ndarray:
        q = np.asarray(joints, dtype=np.float32).reshape(-1)
        if len(q) < 7:
            raise ValueError(f"Expected seven arm joints, got shape {q.shape}")
        q = q[:7]
        joint_limits = np.asarray(self._robot._JOINT_LIMITS, dtype=np.float32)
        finite = np.isfinite(q)
        severely_invalid = (
            not np.all(finite)
            or np.any(q < joint_limits[:, 0] - 0.25)
            or np.any(q > joint_limits[:, 1] + 0.25)
        )
        if severely_invalid:
            raise RuntimeError(
                "Simulator joint state diverged; aborting this attempt: "
                f"{q.tolist()}"
            )
        elif not np.all(
            (q >= joint_limits[:, 0]) & (q <= joint_limits[:, 1])
        ):
            logging.warning(
                "expert clipped slightly out-of-limit observed joints %s", q.tolist()
            )
        return np.clip(q, joint_limits[:, 0], joint_limits[:, 1]).astype(np.float32)

    def _validate_live_pose(self, name: str, pose: np.ndarray) -> np.ndarray:
        pose = np.asarray(pose, dtype=np.float32).reshape(-1)
        if len(pose) < 7 or not np.all(np.isfinite(pose[:7])):
            raise RuntimeError(
                f"Simulator pose diverged for {name}; aborting this attempt: {pose.tolist()}"
            )
        xyz = pose[:3]
        quaternion_norm = float(np.linalg.norm(pose[3:7]))
        radius = float(np.linalg.norm(xyz[:2]))
        if (
            xyz[2] < self._min_object_z
            or xyz[2] > self._max_object_z
            or radius > self._max_object_radius
            or quaternion_norm < 1e-6
        ):
            raise RuntimeError(
                f"Simulator pose left the recoverable workspace for {name}; "
                f"aborting this attempt: {pose[:7].tolist()}"
            )
        return pose

    def _solve(
        self,
        target_pose: np.ndarray,
        q_init: np.ndarray,
        q_home: np.ndarray,
        yaw_delta: float,
        deadline: float,
    ) -> tuple[np.ndarray, float, float]:
        joint_limits = np.asarray(self._robot._JOINT_LIMITS, dtype=np.float32)

        def wrist_seed(base: np.ndarray) -> np.ndarray:
            seed = np.asarray(base, dtype=np.float32).copy()
            seed[6] = np.clip(seed[6] + yaw_delta, joint_limits[6, 0], joint_limits[6, 1])
            return seed

        def reach_seed(base: np.ndarray) -> np.ndarray:
            seed = np.asarray(base, dtype=np.float32).copy()
            seed[0] = np.clip(
                np.arctan2(target_pose[1], target_pose[0]),
                joint_limits[0, 0],
                joint_limits[0, 1],
            )
            if np.hypot(target_pose[0], target_pose[1]) > 0.50:
                seed[1] = -0.35
                seed[3] = -1.35
                seed[5] = 1.20
            return seed

        extended = reach_seed(q_home)
        seeds = (
            (np.asarray(q_init, dtype=np.float32), 140, self._ik_rot_weight),
            (wrist_seed(q_init), 180, self._ik_rot_weight),
            (np.asarray(q_home, dtype=np.float32), 220, self._ik_rot_weight),
            (extended, 260, min(self._ik_rot_weight, 0.08)),
            (wrist_seed(extended), 260, min(self._ik_rot_weight, 0.08)),
        )
        best_q: np.ndarray | None = None
        best_position_error = np.inf
        best_rotation_error = np.inf
        best_score = np.inf
        seen: list[np.ndarray] = []

        for seed, iterations, rotation_weight in seeds:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            if any(np.allclose(seed, previous, atol=1e-5) for previous in seen):
                continue
            seen.append(seed)
            q = np.asarray(
                self._robot.ik(
                    target_pose,
                    seed,
                    iters=iterations,
                    pos_tol=self._ik_pos_tol,
                    rot_tol=self._ik_rot_tol,
                    rot_weight=rotation_weight,
                    time_limit_s=remaining,
                ),
                dtype=np.float32,
            ).reshape(-1)
            if len(q) != 7 or not np.all(np.isfinite(q)):
                logging.warning("expert IK returned invalid joints: %s", q.tolist())
                continue
            q = np.clip(q, joint_limits[:, 0], joint_limits[:, 1])
            actual = np.asarray(
                self._robot.state(np.concatenate([q, [0.0]]).astype(np.float32))
            )
            if not np.all(np.isfinite(actual)):
                logging.warning("expert FK returned invalid pose for joints %s", q.tolist())
                continue
            position_error = float(np.linalg.norm(actual[:3] - target_pose[:3]))
            rotation_error = self._rotation_error(target_pose[3:6], actual[3:6])
            if not np.isfinite(position_error) or not np.isfinite(rotation_error):
                continue
            score = position_error + 0.005 * rotation_error
            if score < best_score:
                best_q = q
                best_position_error = position_error
                best_rotation_error = rotation_error
                best_score = score
            rotation_limit = 0.65 if rotation_weight <= 0.08 else 0.35
            if position_error < 0.010 and rotation_error < rotation_limit:
                break

        if best_q is None:
            raise TimeoutError("Expert IK planning exceeded its time limit")
        return best_q, best_position_error, best_rotation_error

    def _build_leg(
        self,
        ic_index: int,
        food_name: str,
        q0: np.ndarray,
        grip0: float,
        retry_index: int,
        live_poses: dict[str, Any] | None,
        sim_ee_pose: np.ndarray | None,
    ) -> tuple[np.ndarray, int]:
        if self._q_home is None:
            raise RuntimeError("Episode home joints are not initialized")
        planning_start = time.monotonic()
        planning_deadline = planning_start + self._plan_timeout

        ic = self._poses[ic_index]
        _, _, drop_sign = self._food_spec(food_name)
        initial_food_pose = self._find(ic, food_name)
        initial_bowl_pose = self._find(ic, "bowl")
        if live_poses:
            try:
                food_pose = self._validate_live_pose(
                    food_name, self._find(live_poses, food_name)
                )
                bowl_pose = self._validate_live_pose(
                    "bowl", self._find(live_poses, "bowl")
                )
            except KeyError:
                food_pose = initial_food_pose
                bowl_pose = initial_bowl_pose
        else:
            food_pose = initial_food_pose
            bowl_pose = initial_bowl_pose
        bowl_xyz = bowl_pose[:3]
        base_grasp_offset = (
            self._grasp_off_grapes if food_name == "grapes" else self._grasp_off
        )
        variant_index = (self._episode_attempt - 1 + retry_index) % GRASP_VARIANTS
        grasp_offset = base_grasp_offset + GRASP_HEIGHT_DELTAS[variant_index]

        seed = 100_000 * ic_index + 1_000 * self._episode_attempt + 10 * retry_index
        seed += 1 if food_name == "ice_cream" else 2
        rng = np.random.default_rng(seed)
        jitter = rng.uniform(-self._jitter, self._jitter, size=3) * [1.0, 1.0, 0.6]

        current_pose = np.asarray(
            self._robot.state(np.concatenate([q0, [grip0]]).astype(np.float32))
        )
        if sim_ee_pose is not None:
            logging.info(
                "expert frame check: sim_ee=%s fk_link8=%s delta=%s",
                np.round(sim_ee_pose[:3], 3).tolist(),
                np.round(current_pose[:3], 3).tolist(),
                np.round(sim_ee_pose[:3] - current_pose[:3], 3).tolist(),
            )
        home_rpy = np.asarray(current_pose[3:6], dtype=np.float32)
        grasp_rpy, yaw_delta, orientation_mode = self._grasp_orientation(
            home_rpy, food_pose, bowl_xyz, variant_index
        )

        target = np.asarray(food_pose[:3], dtype=np.float64) + jitter
        away = target[:2] - bowl_xyz[:2]
        bowl_distance = float(np.linalg.norm(away))
        grasp_shift = 0.0
        if 1e-6 < bowl_distance < self._bowl_clearance:
            grasp_shift = min(self._bowl_clearance - bowl_distance, self._grasp_shift_max)
            target[:2] += away / bowl_distance * grasp_shift

        drop = np.asarray(bowl_xyz, dtype=np.float64).copy()
        drop[0] += drop_sign * self._drop_spread
        waypoints = (
            (target + [0.0, 0.0, self._hover], grasp_rpy, self._open, 0),
            (target + [0.0, 0.0, grasp_offset], grasp_rpy, self._open, 0),
            (target + [0.0, 0.0, grasp_offset], grasp_rpy, self._closed, self._dwell),
            (target + [0.0, 0.0, self._transit], grasp_rpy, self._closed, 0),
            (drop + [0.0, 0.0, self._transit], grasp_rpy, self._closed, 0),
            (drop + [0.0, 0.0, self._drop], grasp_rpy, self._closed, 0),
            (drop + [0.0, 0.0, self._drop], grasp_rpy, self._open, self._dwell),
            (drop + [0.0, 0.0, self._transit], grasp_rpy, self._open, 0),
        )

        expanded: list[tuple[np.ndarray, np.ndarray, float, int]] = []
        previous_xyz = np.asarray(current_pose[:3], dtype=np.float64) - [0.0, 0.0, self._tcp]
        for xyz, rpy, gripper, dwell in waypoints:
            xyz = np.asarray(xyz, dtype=np.float64)
            via_count = int(np.linalg.norm((xyz - previous_xyz)[:2]) // self._via_distance)
            for via_index in range(1, via_count + 1):
                ratio = via_index / (via_count + 1)
                intermediate = previous_xyz + (xyz - previous_xyz) * ratio
                intermediate[2] = max(previous_xyz[2], xyz[2])
                expanded.append((intermediate, rpy, gripper, 0))
            expanded.append((xyz, rpy, gripper, dwell))
            previous_xyz = xyz

        rows: list[np.ndarray] = []
        q_previous = np.asarray(q0, dtype=np.float32)
        for waypoint_index, (xyz, rpy, gripper, dwell) in enumerate(expanded):
            if time.monotonic() >= planning_deadline:
                raise TimeoutError(
                    f"Expert planning exceeded {self._plan_timeout:.0f}s for {food_name}"
                )
            flange_xyz = np.asarray(xyz, dtype=np.float32) + [0.0, 0.0, self._tcp]
            target_pose = np.concatenate([flange_xyz, np.asarray(rpy, dtype=np.float32)])
            q_target, position_error, rotation_error = self._solve(
                target_pose,
                q_previous,
                self._q_home,
                yaw_delta,
                planning_deadline,
            )
            if time.monotonic() >= planning_deadline:
                raise TimeoutError(
                    f"Expert planning exceeded {self._plan_timeout:.0f}s for {food_name}"
                )
            if position_error > 0.02 or rotation_error > 0.20:
                logging.warning(
                    "expert IK weak: food=%s waypoint=%d pos_err=%.3f m rot_err=%.1f deg "
                    "target=%s",
                    food_name,
                    waypoint_index,
                    position_error,
                    np.degrees(rotation_error),
                    np.round(target_pose[:3], 3).tolist(),
                )
            if position_error > self._max_ik_position_error:
                raise RuntimeError(
                    f"Expert IK cannot safely reach {food_name} waypoint "
                    f"{waypoint_index}: position error {position_error:.3f} m"
                )
            interpolation_steps = max(
                int(np.ceil(np.max(np.abs(q_target - q_previous)) / self._max_dq)), 1
            )
            if interpolation_steps > self._max_interpolation_steps:
                logging.warning(
                    "expert capped interpolation from %d to %d steps; previous=%s target=%s",
                    interpolation_steps,
                    self._max_interpolation_steps,
                    np.round(q_previous, 3).tolist(),
                    np.round(q_target, 3).tolist(),
                )
                interpolation_steps = self._max_interpolation_steps
            for step_index in range(interpolation_steps):
                alpha = (step_index + 1) / interpolation_steps
                joints = q_previous + alpha * (q_target - q_previous)
                rows.append(np.concatenate([joints, [gripper]]))
            rows.extend(np.concatenate([q_target, [gripper]]) for _ in range(dwell))
            q_previous = q_target

        plan = np.stack(rows).astype(np.float32)
        motion_steps = len(plan)
        aligned_steps = ((motion_steps + ADVANCE - 1) // ADVANCE) * ADVANCE
        hold_steps = aligned_steps - motion_steps + CHUNK
        plan = np.concatenate([plan, np.repeat(plan[-1:], hold_steps, axis=0)], axis=0)

        logging.info(
            "expert leg: IC %d episode %d food=%s retry=%d variant=%d mode=%s "
            "yaw=%.1fdeg grasp_z=%.3f shift=%.3f bowl_distance=%.3f moved=%.3f "
            "-> %d motion steps",
            ic_index,
            self._episode_attempt,
            food_name,
            retry_index + 1,
            variant_index + 1,
            orientation_mode,
            np.degrees(yaw_delta),
            grasp_offset,
            grasp_shift,
            bowl_distance,
            float(np.linalg.norm(food_pose[:3] - initial_food_pose[:3])),
            motion_steps,
        )
        logging.info(
            "expert planning finished: IC %d food=%s retry=%d elapsed=%.1fs",
            ic_index,
            food_name,
            retry_index + 1,
            time.monotonic() - planning_start,
        )
        return plan, motion_steps

    def _next_food(self, done: np.ndarray) -> str | None:
        missing = [spec for spec in FOOD_SPECS if not done[spec[1]]]
        if not missing:
            return None
        missing.sort(
            key=lambda spec: (
                self._food_retries.get(spec[0], 0) / FOOD_RETRY_WEIGHTS[spec[0]],
                FOOD_SPECS.index(spec),
            )
        )
        return missing[0][0]

    def _reset_episode(self, ic_index: int, q0: np.ndarray) -> None:
        if not 0 <= ic_index < len(self._poses):
            raise ValueError(f"IC index {ic_index} out of range 0..{len(self._poses) - 1}")
        self._episode_attempt += 1
        self._episode_ic = ic_index
        self._q_home = np.asarray(q0, dtype=np.float32).copy()
        self._plan = None
        self._motion_steps = 0
        self._ptr = 0
        self._active_food = None
        self._food_retries = {spec[0]: 0 for spec in FOOD_SPECS}
        logging.info("expert episode reset: IC %d episode-attempt %d", ic_index, self._episode_attempt)

    @override
    def infer(self, obs: dict, *, noise: Any = None) -> dict:
        episode_reset = bool(obs.pop("_episode_reset", False))
        ic_index_value = obs.pop("subtask/ic_index", None)
        done_value = obs.pop("subtask/done", None)
        live_poses = obs.pop("subtask/object_poses", None)
        sim_ee_value = obs.pop("subtask/ee_pose", None)

        q0 = self._sanitize_observed_joints(obs["observation/joint_position"])
        grip0 = float(np.asarray(obs["observation/gripper_position"]).reshape(-1)[0])

        if episode_reset:
            if ic_index_value is None:
                raise ValueError(
                    "Scripted expert needs subtask/ic_index; run eval with --send-subtask-state"
                )
            self._reset_episode(int(ic_index_value), q0)
        elif self._episode_ic is None:
            raise RuntimeError("Expert received an observation before episode reset")
        elif ic_index_value is not None and int(ic_index_value) != self._episode_ic:
            raise ValueError(
                f"IC changed without episode reset: {self._episode_ic} -> {ic_index_value}"
            )

        if done_value is None:
            raise ValueError(
                "Scripted expert needs subtask/done; run eval with --send-subtask-state"
            )
        done = np.asarray(done_value, dtype=bool).reshape(-1)
        if len(done) < 6:
            raise ValueError(f"Expected six rubric flags, got {done.tolist()}")

        plan_finished = self._plan is None or self._ptr >= self._motion_steps

        if plan_finished:
            food_name = self._next_food(done)
            if food_name is None:
                hold = np.concatenate([q0, [self._open]]).astype(np.float32)
                return {"actions": np.repeat(hold[None], CHUNK, axis=0)}

            retry_index = self._food_retries[food_name]
            self._food_retries[food_name] += 1
            self._active_food = food_name
            logging.info(
                "expert planning start: IC %d food=%s retry=%d",
                self._episode_ic,
                food_name,
                retry_index + 1,
            )
            sim_ee_pose = (
                None
                if sim_ee_value is None
                else np.asarray(sim_ee_value, dtype=np.float32).reshape(-1)
            )
            self._plan, self._motion_steps = self._build_leg(
                int(self._episode_ic),
                food_name,
                q0,
                grip0,
                retry_index,
                live_poses,
                sim_ee_pose,
            )
            self._ptr = 0

        if self._plan is None:
            raise RuntimeError("Expert plan was not initialized")
        chunk = self._plan[self._ptr : self._ptr + CHUNK]
        if len(chunk) < CHUNK:
            chunk = np.concatenate(
                [chunk, np.repeat(self._plan[-1:], CHUNK - len(chunk), axis=0)]
            )
        self._ptr += ADVANCE
        return {"actions": chunk.astype(np.float32)}

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata
