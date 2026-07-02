"""
V-JEPA2-AC as a DIRECT CEM planner/controller for PolaRiS (replaces pi0.5).

Purpose: test whether V-JEPA2-AC's world model can complete the task ON THIS DOMAIN when it
plans actions itself (official CEM from notebooks/utils/mpc_utils), rather than only re-ranking
pi0.5 samples. If even the official planner fails, the world model does not transfer here; if it
works, our re-rank usage was the bottleneck.

Confounds to keep in mind when reading results:
  - CEM minimizes the SAME latent-L1-to-goal energy shown to be weak on the external cam.
  - EE-delta output is converted to joint targets via IK (PandaFK.ik) — an added link that can
    fail independently; run experiments/jepa_verifier/ik_selftest.py first.
  - CEM is slow (rollout*cem_steps batched predictor passes per decision) — use a few episodes.

Deploy by renaming to policy.py on the eval machine. Set the eval client's open_loop_horizon
small (e.g. 1-4) so it replans often. The pi0.5 `model` arg is accepted but unused.
"""
from collections.abc import Sequence
import logging
import os
import pathlib
import sys
import time
from pathlib import Path
from typing import Any, TypeAlias

import cv2
import numpy as np
import torch
from torch.nn import functional as F
import torchvision.transforms as T
from openpi_client import base_policy as _base_policy
from typing_extensions import override

# Put third_party/ on the path so the OFFICIAL vjepa2 (notebooks/, src/) resolves, merging with
# the glue vjepa2 package (FK, rollout) under openpi/src as one namespace package. Same trick as
# vjepa2/rollout.py; must run before importing from vjepa2.notebooks.
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from vjepa2.FK import PandaFK
from vjepa2.notebooks.utils.mpc_utils import cem, compute_new_pose

BasePolicy: TypeAlias = _base_policy.BasePolicy


class Policy(BasePolicy):
    """V-JEPA2-AC CEM controller. Same __init__ signature as the pi0.5 wrapper (model unused)."""

    def __init__(
        self,
        model: Any,
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
        self._device = torch.device(pytorch_device or ("cuda" if torch.cuda.is_available() else "cpu"))
        logging.info("V-JEPA CEM controller on device %s", self._device)

        self._robot = PandaFK(device=str(self._device))
        self._encoder, self._predictor = torch.hub.load(
            "facebookresearch/vjepa2", "vjepa2_ac_vit_giant", pretrained=False
        )
        ckpt = torch.load("/workspace/polaris/third_party/vjepa2/checkpoints/vjepa2-ac-vitg.pt", map_location="cpu")
        strip = lambda sd: {k.replace("module.", "", 1): v for k, v in sd.items()}
        self._encoder.load_state_dict(strip(ckpt["encoder"]))
        self._predictor.load_state_dict(strip(ckpt["predictor"]))
        self._encoder = self._encoder.to(self._device).eval()
        self._predictor = self._predictor.to(self._device).eval()

        crop = 256
        self._tokens_per_frame = int((crop // self._encoder.patch_size) ** 2)
        self._transform = T.Compose([
            T.ToPILImage(),
            T.Resize((crop, crop)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self._normalize_reps = True
        self._goal_image_path = pathlib.Path(goal_image_path or "last_frame.jpg").expanduser()
        self._z_goal = self._encode_goal_image()

        # CEM hyperparameters (official notebook defaults; override via env for speed/tuning)
        self._mpc_args = dict(
            rollout=int(os.environ.get("CEM_ROLLOUT", 2)),
            samples=int(os.environ.get("CEM_SAMPLES", 400)),
            topk=int(os.environ.get("CEM_TOPK", 10)),
            cem_steps=int(os.environ.get("CEM_STEPS", 10)),
            momentum_mean=0.15,
            momentum_std=0.15,
            maxnorm=float(os.environ.get("CEM_MAXNORM", 0.05)),
            verbose=False,
        )
        logging.info("CEM args: %s", self._mpc_args)

    def _encode_np(self, frame_rgb_uint8: np.ndarray) -> torch.Tensor:
        t = self._transform(frame_rgb_uint8)                 # (3,256,256)
        clip = np.stack([t, t], axis=0)                      # tubelet=2
        clip = np.expand_dims(clip, axis=0)                  # (1,2,3,256,256)
        tensor = torch.from_numpy(clip).float().permute(0, 2, 1, 3, 4).to(self._device)
        with torch.inference_mode():
            h = self._encoder(tensor)[:, -self._tokens_per_frame:, :]
            if self._normalize_reps:
                h = F.layer_norm(h, (h.size(-1),))
        return h                                             # (1, tokens, D)

    def _encode_goal_image(self) -> torch.Tensor:
        bgr = cv2.imread(str(self._goal_image_path))
        if bgr is None:
            raise FileNotFoundError(f"Could not read V-JEPA goal image at {self._goal_image_path}.")
        return self._encode_np(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

    @staticmethod
    def _parse_image(image) -> np.ndarray:
        img = np.asarray(image)
        if np.issubdtype(img.dtype, np.floating):
            img = (255 * img).astype(np.uint8)
        if img.shape[0] == 3:
            img = np.transpose(img, (1, 2, 0))
        return img

    def _step_predictor(self, reps, actions, poses):
        B, Tn, N_T, D = reps.size()
        reps = reps.flatten(1, 2)
        nxt = self._predictor(reps, actions, poses)[:, -self._tokens_per_frame:]
        if self._normalize_reps:
            nxt = F.layer_norm(nxt, (nxt.size(-1),))
        nxt = nxt.view(B, 1, N_T, D)
        next_pose = compute_new_pose(poses[:, -1:], actions[:, -1:])
        return nxt, next_pose

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:
        t0 = time.monotonic()
        # current joint state (7) + gripper (1) from the raw request
        joints = np.asarray(obs["observation/joint_position"], dtype=np.float32).reshape(-1)[:7]
        grip = np.asarray(obs["observation/gripper_position"], dtype=np.float32).reshape(-1)[:1]
        state8 = np.concatenate([joints, grip])
        ee_pose7 = self._robot.state(state8)                 # [x,y,z,r,p,y,gripper]

        frame = self._parse_image(obs["observation/exterior_image_1_left"])
        rep = self._encode_np(frame)                         # (1, tokens, D)

        pose_t = torch.from_numpy(ee_pose7).float().to(self._device)[None, None]  # (1,1,7)
        with torch.inference_mode():
            action_traj = cem(
                context_frame=rep,
                context_pose=pose_t,
                goal_frame=self._z_goal,
                world_model=self._step_predictor,
                **self._mpc_args,
            )                                                # (1, rollout, 7) EE deltas
        ee_delta = action_traj[0, 0].detach().cpu().numpy()  # first step (7,)

        # EE delta -> new EE pose -> IK -> joint targets
        new_pose = compute_new_pose(pose_t[:, -1:], torch.from_numpy(ee_delta).float().to(self._device)[None, None])
        new_pose6 = new_pose[0, 0, :6].detach().cpu().numpy()
        target_joints = self._robot.ik(new_pose6, joints)    # (7,)
        gripper_cmd = 1.0 if float(new_pose[0, 0, 6]) > 0.5 else 0.0

        action = np.concatenate([target_joints, [gripper_cmd]]).astype(np.float32)[None]  # (1,8)
        logging.info("CEM step %.2fs | ee_delta xyz=%s grip=%.2f",
                     time.monotonic() - t0, np.round(ee_delta[:3], 4).tolist(), gripper_cmd)
        return {"actions": action}

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata
