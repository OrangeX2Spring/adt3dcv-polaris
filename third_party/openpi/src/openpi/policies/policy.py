from collections.abc import Sequence
import logging
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
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils
from vjepa2 import FK, rollout
import torchvision.transforms as T
import cv2

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
        pytorch_device: str = "cpu",
        is_pytorch: bool = False,
        goal_image_path: str | pathlib.Path | None = None,
    ):
        self._model = model
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._is_pytorch_model = is_pytorch
        self._pytorch_device = pytorch_device
        self._num_candidates = 10
        self._robot =  PandaFK()
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
        self._goal_image_path = pathlib.Path(goal_image_path or "last_frame.jpg").expanduser()

            
    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:
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
        actions_joint_downsampled = actions_joint[:, [7, 15], :]  # Downsample to match V-JEPA input requirements

        state_joint = outputs["state"][:,:8]
        curr_state_np = np.array(state_joint)      # (10, 8)
        future_action_np = np.array(actions_joint_downsampled)  # (10, 2, 8)

        # 2. 给当前帧状态插入时序维度 (Time Dimension)
        # 从 (10, 8) 变成 (10, 1, 8)，代表 t0 帧
        curr_state_expanded = curr_state_np[:, np.newaxis, :]

        # 3. 沿时序轴（axis=1）无缝拼接，组合成长为 T=3 的完整轨迹序列
        # [t0] + [t1, t2] = [t0, t1, t2]
        full_trajectory = np.concatenate([curr_state_expanded, future_action_np], axis=1) # Shape: (10, 3, 8)

        result = self._robot.convert_trajectory(full_trajectory)
        ee_actions = torch.from_numpy(result["actions"]).float()
        ee_states = torch.from_numpy(result["states"]).float()
        
        if self._encoder is not None:
            # device = inputs.device
            goal_frame = cv2.imread(str(self._goal_image_path))
            if goal_frame is None:
                raise FileNotFoundError(
                    f"Could not read V-JEPA goal image at {self._goal_image_path}. "
                    "Pass --goal-image-path to scripts/serve_policy.py with an existing image path."
                )
            goal_frame = self._transform(cv2.cvtColor(goal_frame, cv2.COLOR_BGR2RGB))
            goal_np = np.stack([goal_frame, goal_frame], axis=0) 
            goal_np = np.expand_dims(goal_np, axis=0)
            goal_tensor = torch.from_numpy(goal_np).float().permute(0, 2, 1, 3, 4)
            z_goal = self._encoder(goal_tensor)[:,-256:,:]
            frame = self._transform(np.array(observation.images["base_0_rgb"][0]))
            frames_np = np.stack([frame, frame], axis=0)  # (2, 256, 256, 3)
            frames_np = np.expand_dims(frames_np, axis=0)
            frames_tensor = torch.from_numpy(frames_np).float().permute(0, 2, 1, 3, 4)
            z_current = self._encoder(frames_tensor)[:,-256:,:]
            batch_size = ee_actions.shape[0]  # 10
            action_dim = ee_actions.shape[2]  # 7
         
            z_hat = rollout.forward_actions(z_current, self._predictor, ee_states, ee_actions)
        
            losses = rollout.loss_fn(z_hat, z_goal)  # list[10]
            best_idx = np.argmin(losses)

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
