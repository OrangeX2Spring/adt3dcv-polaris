from collections.abc import Sequence
import json
import logging
import os
import pathlib
import time
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
from vjepa2.FK import PandaFK
from vjepa2 import rollout
import torch
from torch.nn import functional as F
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils
from vjepa2 import FK, rollout
import torchvision.transforms as T
import cv2
import glob
BasePolicy: TypeAlias = _base_policy.BasePolicy


class Policy(BasePolicy):
    """Wraps a pi0.5 policy with V-JEPA AC predictor for action selection via energy minimization."""

    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        pytorch_device: str = "cuda:0",
        is_pytorch: bool = False,
    ):
        self._model = model
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._is_pytorch_model = is_pytorch
        self._pytorch_device = pytorch_device
        self._vjepa_device = torch.device(
            pytorch_device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        logging.info("Using V-JEPA device: %s", self._vjepa_device)
        self._num_candidates = 10
        self._robot = PandaFK(device=str(self._vjepa_device))
        if self._is_pytorch_model:
            self._model = self._model.to(pytorch_device)
            self._model.eval()
            self._sample_actions = model.sample_actions
        else:
            # JAX model setup
            self._sample_actions = nnx_utils.module_jit(model.sample_actions)
            self._rng = rng or jax.random.key(0)

        self._encoder, self._predictor = torch.hub.load("facebookresearch/vjepa2", "vjepa2_ac_vit_giant", pretrained=False)
        ckpt= torch.load("/workspace/polaris/third_party/vjepa2/checkpoints/vjepa2-ac-vitg.pt", map_location="cpu")
        def strip_module_prefix(state_dict):
            return {k.replace("module.", "", 1): v for k, v in state_dict.items()}

        self._encoder.load_state_dict(strip_module_prefix(ckpt["encoder"]))
        self._predictor.load_state_dict(strip_module_prefix(ckpt["predictor"]))
        self._encoder = self._encoder.to(self._vjepa_device).eval()
        self._predictor = self._predictor.to(self._vjepa_device).eval()
        # Initialize transform
        crop_size = 256
        self._tokens_per_frame = int((crop_size // self._encoder.patch_size) ** 2)
        self._transform = T.Compose([
            T.ToPILImage(),
            T.Resize((256, 256)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])
        self._goal_frames_dir = "/workspace/polaris/runs/test/downsampled_frames/"
        self._episode_ids, self._z_goal_init = self._encode_all_episode_goals()
        logging.info(
            "预编码了 %d 个 episode 的目标图，来自: %s",
            len(self._episode_ids), self._goal_frames_dir,
        )

        # Per-step energy log (JSONL). Override location with VJEPA_ENERGY_LOG.
        self._energy_log_path = pathlib.Path(
            os.environ.get(
                "VJEPA_ENERGY_LOG",
                f"vjepa_energy_{time.strftime('%Y%m%d_%H%M%S')}.jsonl",
            )
        ).expanduser()
        self._infer_step = 0
        logging.info("Logging V-JEPA energies to %s", self._energy_log_path.resolve())
        self._z_goal = None              # 锁定后使用的目标 embedding，初始为空
        self._locked_episode_id = None   # 锁定后的 episode id，用于日志/调试
        self._goal_sequence: list[pathlib.Path] = []   # 锁定 episode 后的候选目标帧路径序列（按顺序）
        self._goal_ptr = 0                              # 当前使用的是序列里第几个目标
        self._prev_best_loss: float | None = None       # 上一步的最小 loss，用于判断趋势
        self._rising_count = 0                          # 连续上升计数
        self._rising_patience = 2 


    def _encode_all_episode_goals(self) -> tuple[list[str], torch.Tensor]:
        if not os.path.isdir(self._goal_frames_dir):
            raise FileNotFoundError(f"目标帧根目录不存在: {self._goal_frames_dir}")

        episode_dirs = sorted(
            d for d in os.listdir(self._goal_frames_dir)
            if os.path.isdir(os.path.join(self._goal_frames_dir, d))
        )
        if not episode_dirs:
            raise FileNotFoundError(f"{self._goal_frames_dir} 下没有找到任何 episode 子目录")

        episode_ids: list[str] = []
        z_goals: list[torch.Tensor] = []

        for video_name in episode_dirs:
            ep_dir = os.path.join(self._goal_frames_dir, video_name)
            first_frame_path = os.path.join(ep_dir, f"{video_name}_frame00000.jpg")

            if not os.path.isfile(first_frame_path):
                candidates = sorted(glob.glob(os.path.join(ep_dir, f"{video_name}_frame*.jpg")))
                if not candidates:
                    logging.warning("跳过 %s：目录下没有找到任何帧图片", ep_dir)
                    continue
                first_frame_path = candidates[0]

            goal_frame = cv2.imread(first_frame_path)
            goal_frame = self._transform(cv2.cvtColor(goal_frame, cv2.COLOR_BGR2RGB))
            goal_np = np.stack([goal_frame, goal_frame], axis=0)
            goal_np = np.expand_dims(goal_np, axis=0)
            goal_tensor = (
                torch.from_numpy(goal_np)
                .float()
                .permute(0, 2, 1, 3, 4)
                .to(self._vjepa_device)
            )
            with torch.inference_mode():
                h = self._encoder(goal_tensor)[:, -self._tokens_per_frame :, :]
                # Match official WorldModel.encode: reps are layer-normed before use
                z = F.layer_norm(h, (h.size(-1),))

            episode_ids.append(video_name)
            z_goals.append(z.squeeze(0))

        if not z_goals:
            raise RuntimeError(f"{self._goal_frames_dir} 下没有成功编码出任何目标图")

        z_goal_init = torch.stack(z_goals, dim=0)
        return episode_ids, z_goal_init
    


    def _build_goal_sequence(self, episode_id: str) -> list[pathlib.Path]:
        """
        根据 episode_id 拿到该文件夹下按文件名排序的所有帧，
        去掉第一帧（idx=0，只用来做初始 episode 匹配），
        剩下的按顺序作为后续目标帧序列：[idx=20, idx=40, idx=60, ...]
        """
        ep_dir = os.path.join(self._goal_frames_dir, episode_id)
        if not os.path.isdir(ep_dir):
            raise FileNotFoundError(f"episode 目录不存在: {ep_dir}")

        frame_paths = sorted(glob.glob(os.path.join(ep_dir, f"{episode_id}_frame*.jpg")))
        if len(frame_paths) < 2:
            raise FileNotFoundError(f"{ep_dir} 下帧数不足 2 张，无法构建目标序列")

        return frame_paths[1:]
    
    def _encode_frame(self, image_path: pathlib.Path) -> torch.Tensor:
        goal_frame = cv2.imread(str(image_path))
        if goal_frame is None:
            raise FileNotFoundError(
                f"Could not read V-JEPA goal image at {image_path}. "
                "Pass --goal-image-path to scripts/serve_policy.py with an existing image path."
            )
        goal_frame = self._transform(cv2.cvtColor(goal_frame, cv2.COLOR_BGR2RGB))
        goal_np = np.stack([goal_frame, goal_frame], axis=0)
        goal_np = np.expand_dims(goal_np, axis=0)
        goal_tensor = (
            torch.from_numpy(goal_np)
            .float()
            .permute(0, 2, 1, 3, 4)
            .to(self._vjepa_device)
        )
        with torch.inference_mode():
            h = self._encoder(goal_tensor)[:, -self._tokens_per_frame :, :]
            # Match official WorldModel.encode: reps are layer-normed before use
            return F.layer_norm(h, (h.size(-1),))
        
    def reset_episode(self):
        """在开始新的一个 episode/rollout 之前调用，清空锁定状态。"""
        self._z_goal = None
        self._locked_episode_id = None
        self._goal_sequence = []
        self._goal_ptr = 0
        self._prev_best_loss = None
        self._rising_count = 0
        self._infer_step = 0
            
    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:
        if obs.get("_episode_reset", False):
            self.reset_episode()
        obs = {k: v for k, v in obs.items() if k != "_episode_reset"}
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        if not self._is_pytorch_model:
            # Make a batch and convert to jax.Array.
            inputs = jax.tree.map(
                lambda x: jnp.repeat(
                    jnp.asarray(x)[None, ...],
                    self._num_candidates,
                    axis=0,
                ),
                inputs,
            )
            self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
        else:
            # Convert inputs to PyTorch tensors and move to correct device
            inputs = jax.tree.map(
                    lambda x: (
                        torch.from_numpy(np.asarray(x))
                        .to(self._pytorch_device)
                        .unsqueeze(0)
                        .repeat(self._num_candidates, *([1] * np.asarray(x).ndim))
                    ),
                    inputs,
                )
            sample_rng_or_pytorch_device = self._pytorch_device

        # Prepare kwargs for sample_actions
        sample_kwargs = dict(self._sample_kwargs)
        if noise is not None:
            noise = torch.from_numpy(noise).to(self._pytorch_device) if self._is_pytorch_model else jnp.asarray(noise)

            if noise.ndim == 2:  # If noise is (action_horizon, action_dim), add batch dimension
                if noise.ndim == 2:
                    noise = jnp.repeat(
                        noise[None, ...],
                        self._num_candidates,
                        axis=0,
                    )
            sample_kwargs["noise"] = noise

        observation = _model.Observation.from_dict(inputs)
        start_time = time.monotonic()
        outputs = {
            "state": inputs["state"],
            "actions": self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs),
        }

        actions_joint = outputs["actions"][:,:,:8]  # (num_candidates, action_horizon, action_dim)
        # A2: sample 4 frames ~0.27s apart (16-step chunk at 15Hz) so each V-JEPA action
        # spans ~one DROID 4fps step (~0.25s), matching the predictor's training granularity.
        # Previously [7, 15] gave 2 steps of ~0.53s each (~2x training magnitude, OOD).
        actions_joint_downsampled = actions_joint[:, [3, 7, 11, 15], :]  # (10, 4, 8)

        state_joint = outputs["state"][:,:8]
        curr_state_np = np.array(state_joint)      # (10, 8)
        future_action_np = np.array(actions_joint_downsampled)  # (10, 4, 8)

        # 2. 给当前帧状态插入时序维度 (Time Dimension)
        # 从 (10, 8) 变成 (10, 1, 8)，代表 t0 帧
        curr_state_expanded = curr_state_np[:, np.newaxis, :]

        # 3. 沿时序轴（axis=1）无缝拼接，组合成长为 T=5 的完整轨迹序列
        # [t0] + [t1..t4] = [t0, t1, t2, t3, t4]
        full_trajectory = np.concatenate([curr_state_expanded, future_action_np], axis=1) # Shape: (10, 5, 8)

        result = self._robot.convert_trajectory(full_trajectory)
        ee_actions = torch.from_numpy(result["actions"]).float().to(self._vjepa_device)
        ee_states = torch.from_numpy(result["states"]).float().to(self._vjepa_device)
        
        if self._encoder is not None:
            # A1: encode the RAW uint8 camera frame (RGB, H,W,C), matching the goal-image
            # path exactly. openpi's pipeline has already mapped observation.images to
            # [-1,1] float; feeding that to self._transform's ToPILImage (which assumes
            # [0,1]) wraps negative pixels and corrupts the current-frame embedding.
            raw_base = np.asarray(obs["observation/exterior_image_1_left"])
            if np.issubdtype(raw_base.dtype, np.floating):
                raw_base = (255 * raw_base).astype(np.uint8)
            if raw_base.shape[0] == 3:  # C,H,W -> H,W,C
                raw_base = np.transpose(raw_base, (1, 2, 0))
            frame = self._transform(raw_base)
            frames_np = np.stack([frame, frame], axis=0)  # (2, 256, 256, 3)
            frames_np = np.expand_dims(frames_np, axis=0)
            frames_tensor = (
                torch.from_numpy(frames_np)
                .float()
                .permute(0, 2, 1, 3, 4)
                .to(self._vjepa_device)
            )

            with torch.inference_mode():
                z_current = self._encoder(frames_tensor)[:, -self._tokens_per_frame :, :]
                z_current = F.layer_norm(z_current, (z_current.size(-1),))
                z_hat = rollout.forward_actions(z_current, self._predictor, ee_states, ee_actions)
                if self._z_goal is None:
                    num_candidates = z_hat.shape[0]
                    num_episodes = self._z_goal_init.shape[0]

                    z_hat_exp = z_hat.unsqueeze(1).expand(-1, num_episodes, -1, -1)
                    z_goal_exp = self._z_goal_init.unsqueeze(0).expand(num_candidates, -1, -1, -1)
                    z_hat_flat = z_hat_exp.reshape(num_candidates * num_episodes, *z_hat.shape[1:])
                    z_goal_flat = z_goal_exp.reshape(num_candidates * num_episodes, *z_hat.shape[1:])
                    losses_flat = rollout.loss_fn(z_hat_flat, z_goal_flat)
                    losses_matrix = np.array(losses_flat).reshape(num_candidates, num_episodes)

                    best_loss_per_episode = losses_matrix.min(axis=0)
                    best_episode_idx = int(np.argmin(best_loss_per_episode))
                    best_episode_id = self._episode_ids[best_episode_idx]

                    self._locked_episode_id = best_episode_id
                    self._goal_sequence = self._build_goal_sequence(best_episode_id)
                    self._goal_ptr = 0
                    self._z_goal = self._encode_frame(self._goal_sequence[self._goal_ptr])
                    self._prev_best_loss = None
                    self._rising_count = 0

                    logging.info(
                        "首次 infer：锁定 episode=%s（匹配 loss=%.4f），目标序列长度=%d，"
                        "初始目标帧=%s",
                        best_episode_id, best_loss_per_episode[best_episode_idx],
                        len(self._goal_sequence), os.path.basename(self._goal_sequence[self._goal_ptr]),
                    )
                
                losses = rollout.loss_fn(z_hat, self._z_goal)
                losses_np = np.asarray(losses)
                best_idx = int(np.argmin(losses_np))
                current_best_loss = float(losses_np[best_idx])
            switched = False
            if self._prev_best_loss is not None:
                if current_best_loss > self._prev_best_loss:
                    self._rising_count += 1
                else:
                    self._rising_count = 0  # loss 在下降或持平，重置计数

                if self._rising_count >= self._rising_patience:
                    # 连续上升达到耐心阈值，切换到序列里的下一帧（如果还有）
                    if self._goal_ptr < len(self._goal_sequence) - 1:
                        self._goal_ptr += 1
                        next_frame_path = self._goal_sequence[self._goal_ptr]
                        self._z_goal = self._encode_frame(next_frame_path)
                        switched = True
                        logging.info(
                            "Loss 连续上升 %d 步，切换目标帧 -> %s (episode=%s, ptr=%d/%d)",
                            self._rising_count, os.path.basename(next_frame_path),
                            self._locked_episode_id, self._goal_ptr, len(self._goal_sequence) - 1,
                        )
                        # 切换后重新算一次 loss 和 best_idx，保证本步动作选择基于新目标
                        losses = rollout.loss_fn(z_hat, self._z_goal)
                        losses_np = np.asarray(losses)
                        best_idx = int(np.argmin(losses_np))
                        current_best_loss = float(losses_np[best_idx])
                    else:
                        logging.info("已到达目标序列末尾，不再切换。")
                    # 无论是否成功切换，重置计数，避免连续多次切换
                    self._rising_count = 0

            self._prev_best_loss = current_best_loss
            logging.info(
                "V-JEPA energies: min=%.4f max=%.4f spread=%.4f best_idx=%d all=%s",
                min(losses), max(losses), max(losses) - min(losses), best_idx,
                [round(l, 4) for l in losses],
            )
            with self._energy_log_path.open("a") as f:
                f.write(
                    json.dumps(
                        {
                            "wall_time": time.time(),
                            "step": self._infer_step,
                            "locked_episode_id": self._locked_episode_id,
                            "best_idx": int(best_idx),
                            "energies": [round(l, 6) for l in losses],
                        }
                    )
                    + "\n"
                )
            self._infer_step += 1

            best_action = outputs["actions"][best_idx]  # (7,) delta EE
            

            outputs = {
                "state": inputs["state"][0][None, ...],
                "actions": best_action[None, ...],
            }
        model_time = time.monotonic() - start_time
        if self._is_pytorch_model:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
        else:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)

        outputs = self._output_transform(outputs)
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        return outputs
    
    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results
