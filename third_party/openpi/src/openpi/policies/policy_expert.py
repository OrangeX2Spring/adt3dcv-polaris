"""
Scripted expert for DROID-FoodBussing data collection (BEHAVIOR-1K-style classical planner).

Plan: for each food (ice cream, then grapes): hover above -> descend -> close gripper ->
lift -> move above bowl -> lower -> open -> retreat. Waypoints in EE space with the arm's
INITIAL orientation kept throughout (no orientation-convention risk); EE->joints via the
self-tested PandaFK.ik (warm-started); joint-space interpolation under a per-step cap.

It is a drop-in policy.py variant:
  - NEEDS_MODEL = False  -> serve_policy skips pi0.5 weights (same mechanism as the CEM run)
  - reads the SAME initial_conditions.json the eval uses (env var POLARIS_IC_FILE)
  - gets the IC index per episode from "subtask/ic_index"  -> run eval with
    --send-subtask-state --fix-ic <k>
  - re-plans on "_episode_reset" (sent by the client at every episode start), adding a small
    fresh jitter each attempt so retries after a failed grasp are not identical

Tuning via env vars (meters / radians / steps):
  EXPERT_TCP_OFFSET (0.105)   flange-to-fingertip distance (FK ends at panda_link8!)
  EXPERT_GRASP_OFFSET (0.02)  fingertip height above object-center z at the grasp waypoint
  EXPERT_HOVER (0.12)  EXPERT_TRANSIT (0.20)  EXPERT_DROP (0.12)   relative heights
  EXPERT_DWELL (10)    steps to hold still while the gripper closes/opens
  EXPERT_MAX_DQ (0.05) max joint delta per control step (speed cap)
  EXPERT_JITTER (0.005) per-attempt xyz jitter on grasp waypoints
  EXPERT_GRIP_INVERT (0) set 1 if gripper convention is reversed (default: 1.0 = closed)

Deploy: cp policy_expert.py policy.py, restart serve_policy (same command as always;
POLARIS_IC_FILE must point at the task's initial_conditions.json).
"""
from collections.abc import Sequence
import json
import logging
import os
import pathlib
from typing import Any, TypeAlias

import numpy as np
from openpi_client import base_policy as _base_policy
from typing_extensions import override

from vjepa2.FK import PandaFK

BasePolicy: TypeAlias = _base_policy.BasePolicy

CHUNK = 16          # actions returned per request (matches pi0.5's action horizon)
ADVANCE = 8         # actions the client consumes before requesting again (open_loop_horizon)


class Policy(BasePolicy):
    NEEDS_MODEL = False  # policy_config skips loading pi0.5 weights

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

        # PandaFK's chain ends at panda_link8 (the wrist flange); the fingertips sit
        # ~10.5 cm further along the tool axis. Waypoints are specified for the
        # FINGERTIPS; this offset lifts the flange target accordingly (world z,
        # valid for the near-vertical tool orientation of the DROID home pose).
        self._tcp = float(os.environ.get("EXPERT_TCP_OFFSET", 0.105))
        self._grasp_off = float(os.environ.get("EXPERT_GRASP_OFFSET", 0.02))
        # grapes lie tilted with a lower center (z~0.036 vs ice cream ~0.051) — allow a
        # separate grasp height; defaults to the shared offset if unset
        self._grasp_off_grapes = float(
            os.environ.get("EXPERT_GRASP_OFFSET_GRAPES", self._grasp_off)
        )
        self._hover = float(os.environ.get("EXPERT_HOVER", 0.12))
        self._transit = float(os.environ.get("EXPERT_TRANSIT", 0.20))
        self._drop = float(os.environ.get("EXPERT_DROP", 0.12))
        self._dwell = int(os.environ.get("EXPERT_DWELL", 10))
        self._max_dq = float(os.environ.get("EXPERT_MAX_DQ", 0.05))
        self._jitter = float(os.environ.get("EXPERT_JITTER", 0.005))
        invert = os.environ.get("EXPERT_GRIP_INVERT", "0") == "1"
        self._closed, self._open = (0.0, 1.0) if invert else (1.0, 0.0)

        self._plan: np.ndarray | None = None
        self._ptr = 0
        self._attempt = 0
        logging.info(
            "Scripted expert ready: %d ICs from %s | grasp_off=%.3f dwell=%d max_dq=%.3f "
            "jitter=%.3f closed=%.1f",
            len(self._poses), ic_file, self._grasp_off, self._dwell, self._max_dq,
            self._jitter, self._closed,
        )

    # ---------------- planning ----------------

    @staticmethod
    def _find(pose_dict: dict, needle: str) -> np.ndarray:
        for k, v in pose_dict.items():
            if needle in k:
                return np.asarray(v, dtype=np.float32)
        raise KeyError(f"No object matching '{needle}' in IC (keys: {list(pose_dict)})")

    def _build_plan(self, ic_index: int, q0: np.ndarray, grip0: float) -> np.ndarray:
        ic = self._poses[ic_index]
        rng = np.random.default_rng(1000 * ic_index + self._attempt)
        jit = lambda: rng.uniform(-self._jitter, self._jitter, size=3) * [1, 1, 0.6]

        ice = self._find(ic, "ice_cream")[:3]
        grapes = self._find(ic, "grapes")[:3]
        bowl = self._find(ic, "bowl")[:3]

        p0 = self._robot.state(np.concatenate([q0, [grip0]]).astype(np.float32))
        rpy = np.asarray(p0[3:6], dtype=np.float32)  # keep initial EE orientation everywhere
        logging.info(
            "expert targets: ice=%s grapes=%s bowl=%s | ee0(flange)=%s | ic_keys=%s",
            np.round(ice, 3).tolist(), np.round(grapes, 3).tolist(),
            np.round(bowl, 3).tolist(), np.round(np.asarray(p0[:3]), 3).tolist(),
            list(ic.keys()),
        )

        # (xyz, gripper, dwell_steps) waypoint list. Drop points are spread laterally per
        # food (else food #2 is released onto food #1 and bounces out of the bowl) —
        # same trick as generate_goal_images.py's xy_spread.
        spread = float(os.environ.get("EXPERT_DROP_SPREAD", 0.02))
        wps: list[tuple[np.ndarray, float, int]] = []
        for food, g_off, dx in (
            (ice, self._grasp_off, -spread),
            (grapes, self._grasp_off_grapes, +spread),
        ):
            f = food + jit()
            drop = bowl + [dx, 0.0, 0.0]
            wps += [
                (f + [0, 0, self._hover], self._open, 0),
                (f + [0, 0, g_off], self._open, 0),
                (f + [0, 0, g_off], self._closed, self._dwell),   # close & dwell
                (f + [0, 0, self._transit], self._closed, 0),                # lift
                (drop + [0, 0, self._transit], self._closed, 0),             # transit
                (drop + [0, 0, self._drop], self._closed, 0),                # lower
                (drop + [0, 0, self._drop], self._open, self._dwell),        # release & dwell
                (drop + [0, 0, self._transit], self._open, 0),               # retreat
            ]

        # Insert via-points so every IK hop stays short: the DLS IK in FK.py is built for
        # small deltas and can stall (silently returning ~the warm start) on long
        # cross-table jumps — which turns a whole leg of the plan into a no-op.
        expanded: list[tuple[np.ndarray, float, int]] = []
        prev_xyz = np.asarray(p0[:3], dtype=np.float32) - [0.0, 0.0, self._tcp]
        for xyz, grip, dwell in wps:
            xyz = np.asarray(xyz, dtype=np.float32)
            n_via = int(np.linalg.norm((xyz - prev_xyz)[:2]) // 0.15)
            for v in range(1, n_via + 1):
                mid = prev_xyz + (xyz - prev_xyz) * v / (n_via + 1)
                mid[2] = max(prev_xyz[2], xyz[2])
                expanded.append((mid, grip, 0))
            expanded.append((xyz, grip, dwell))
            prev_xyz = xyz

        def _solve(target6, q_init, q_home):
            best_q, best_err = None, np.inf
            for q_seed, iters in ((q_init, 120), (q_home, 200)):
                q = np.asarray(self._robot.ik(target6, q_seed, iters=iters), dtype=np.float32)
                fk = np.asarray(self._robot.state(np.concatenate([q, [0.0]]).astype(np.float32)))
                err = float(np.linalg.norm(fk[:3] - target6[:3]))
                if err < best_err:
                    best_q, best_err = q, err
                if err < 0.015:
                    break
            return best_q, best_err

        rows: list[np.ndarray] = []
        q_prev = np.asarray(q0, dtype=np.float32)
        for wi, (xyz, grip, dwell) in enumerate(expanded):
            xyz = np.asarray(xyz, dtype=np.float32) + [0.0, 0.0, self._tcp]  # fingertip -> flange
            target6 = np.concatenate([xyz, rpy])
            q_t, err = _solve(target6, q_prev, np.asarray(q0, dtype=np.float32))
            if err > 0.02:
                logging.warning(
                    "expert IK poor convergence at waypoint %d: err=%.3f m target=%s",
                    wi, err, np.round(target6[:3], 3).tolist(),
                )
            n = max(int(np.ceil(np.max(np.abs(q_t - q_prev)) / self._max_dq)), 1)
            for a in np.linspace(1.0 / n, 1.0, n):
                rows.append(np.concatenate([q_prev + a * (q_t - q_prev), [grip]]))
            for _ in range(dwell):
                rows.append(np.concatenate([q_t, [grip]]))
            q_prev = q_t

        plan = np.stack(rows).astype(np.float32)
        # hold the final pose so we never run out of actions within the 450-step horizon
        tail = np.repeat(plan[-1:], 600, axis=0)
        plan = np.concatenate([plan, tail], axis=0)
        logging.info(
            "expert plan: IC %d attempt %d -> %d motion steps (+hold)",
            ic_index, self._attempt, len(rows),
        )
        return plan

    # ---------------- serving ----------------

    @override
    def infer(self, obs: dict, *, noise: Any = None) -> dict:
        episode_reset = bool(obs.pop("_episode_reset", False))
        ic_index = obs.pop("subtask/ic_index", None)
        obs.pop("subtask/done", None)

        if episode_reset or self._plan is None:
            if ic_index is None:
                raise ValueError(
                    "Scripted expert needs 'subtask/ic_index' — run eval with "
                    "--send-subtask-state (and --fix-ic <k>)."
                )
            q0 = np.asarray(obs["observation/joint_position"], dtype=np.float32).reshape(-1)[:7]
            grip0 = float(np.asarray(obs["observation/gripper_position"]).reshape(-1)[0])
            self._attempt += 1
            self._plan = self._build_plan(int(ic_index), q0, grip0)
            self._ptr = 0

        chunk = self._plan[self._ptr:self._ptr + CHUNK]
        if len(chunk) < CHUNK:  # paranoid: hold-tail should make this unreachable
            chunk = np.concatenate([chunk, np.repeat(self._plan[-1:], CHUNK - len(chunk), axis=0)])
        self._ptr += ADVANCE
        # raw actions, deliberately NOT passed through pi0.5's output transforms
        return {"actions": chunk.astype(np.float32)}

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata
