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
    z       : (1, 256, 1408)   当前帧 encoder 输出，T=1
    states  : (10, 3, 7)       FK 算好，t0/t1/t2
    actions : (10, 2, 7)       delta EE，a0/a1
    
    predictor 每次只接受 T=1，所以逐步调用，z 滚动替换（不累积）
    """
    S = actions.shape[0]   # 10
    T = actions.shape[1]   # 2

    # z_curr 始终是单帧 (10, 256, D)，不累积
    z_curr = z[:, :tokens_per_frame].repeat(S, 1, 1)   # (10, 256, 1408)

    for t in range(T):
        a_t = actions[:, t:t+1, :]    # (10, 1, 7)  只取当前步
        s_t = states[:, t:t+1, :]     # (10, 1, 7)  只取当前帧真实位姿
        # print(f"t={t}")
        # print(f"  z_curr shape: {z_curr.shape}")
        # print(f"  a_t    shape: {a_t.shape}")
        # print(f"  s_t    shape: {s_t.shape}")
        # print(f"  grid_height={predictor.grid_height}, grid_width={predictor.grid_width}")
        # print(f"  T inferred = {z_curr.shape[1]} // {predictor.grid_height * predictor.grid_width} = {z_curr.shape[1] // (predictor.grid_height * predictor.grid_width)}")

        # 三者 T 维全部 = 1，对齐
        z_next = predictor(z_curr, a_t, s_t)   # (10, 256, 1408)

        if normalize_reps:
            z_next = F.layer_norm(z_next, (z_next.size(-1),))

        z_curr = z_next   # 滚动：下一步用预测帧作为输入

    # 返回最终预测帧，用于和 goal 算 loss
    return z_curr   # (10, 256, 1408)

def loss_fn(z, h):
    z, h = z[:, -tokens_per_frame:], h[:, -tokens_per_frame:]
    loss = torch.abs(z - h)  # [B, N, D]
    loss = torch.mean(loss, dim=[1, 2])
    return loss.tolist()