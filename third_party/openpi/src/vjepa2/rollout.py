import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.nn import functional as F
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
# from app.vjepa_droid.transforms import make_transforms
from vjepa2.notebooks.utils.mpc_utils import (
    compute_new_pose,
    poses_to_diff
)
tokens_per_frame = 256

def forward_target(c, encoder, normalize_reps=True):
    B, C, T, H, W = c.size()
    c = c.permute(0, 2, 1, 3, 4).flatten(0, 1).unsqueeze(2).repeat(1, 1, 2, 1, 1)
    h = encoder(c)
    h = h.view(B, T, -1, h.size(-1)).flatten(1, 2)
    if normalize_reps:
        h = F.layer_norm(h, (h.size(-1),))
    return h


# def forward_actions(z, predictor, states, actions, normalize_reps=True, action_repeat=1):

#     S = actions.shape[0]  # 10 candidates
#     T = actions.shape[1]  # 4 帧

#     def step_predictor(_z, _a, _s):
#         _z = predictor(_z, _a, _s)[:, -tokens_per_frame:]
#         if normalize_reps:
#             _z = F.layer_norm(_z, (_z.size(-1),))
#         _s = compute_new_pose(_s[:, -1:], _a[:, -1:])
#         return _z, _s

#     # Context frame rep and context pose
#     z_hat = z[:, :tokens_per_frame].repeat(S, 1, 1)   # (10, N, D)
#     # s_hat = states[:, :1].repeat(S, 1, 1)              # (10, 1, 7)

#     for t in range(T):
#         a_t = actions[:, t:t+1, :]               # (10, 1, 7)
#         _z, _s = step_predictor(z_hat, a_t, s_hat)
#         z_hat = torch.cat([z_hat, _z], dim=1)          # (10, N*(t+2), D)
#         s_hat = torch.cat([s_hat, _s], dim=1)          # (10, t+2, 7)

#     return z_hat, s_hat
def forward_actions(z, predictor, states, actions, normalize_reps=True):
    """
    z       : (1, 256, 1408)   当前帧 encoder 输出（已 layer_norm），T=1
    states  : (10, 3, 7)       FK 算好，t0/t1/t2
    actions : (10, 2, 7)       delta EE，a0/a1

    与官方 cem() rollout 一致：预测帧 concat 进序列，predictor 每步
    看到完整历史 (z_traj, actions[:, :t+1], states[:, :t+1])。
    """
    S = actions.shape[0]   # 10
    T = actions.shape[1]   # 2

    z_traj = z[:, :tokens_per_frame].repeat(S, 1, 1)   # (10, 256, 1408)

    for t in range(T):
        a_t = actions[:, : t + 1, :]   # (10, t+1, 7)
        s_t = states[:, : t + 1, :]    # (10, t+1, 7)

        z_next = predictor(z_traj, a_t, s_t)[:, -tokens_per_frame:]   # (10, 256, 1408)

        if normalize_reps:
            z_next = F.layer_norm(z_next, (z_next.size(-1),))

        z_traj = torch.cat([z_traj, z_next], dim=1)   # (10, 256*(t+2), 1408)

    # 返回最终预测帧，用于和 goal 算 loss
    return z_traj[:, -tokens_per_frame:]   # (10, 256, 1408)

def loss_fn(z, h):
    z, h = z[:, -tokens_per_frame:], h[:, -tokens_per_frame:]
    loss = torch.abs(z - h)  # [B, N, D]
    loss = torch.mean(loss, dim=[1, 2])
    return loss.tolist()