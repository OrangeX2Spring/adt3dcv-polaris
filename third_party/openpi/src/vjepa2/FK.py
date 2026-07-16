"""
panda_fk.py — Franka Panda 正运动学，全接口支持 batch 输入
=============================================================
将 PolaRiS 的关节位置动作转换为 V-JEPA 2-AC 的 EE 笛卡尔格式。

安装：
    pip install pytorch-kinematics scipy

输入约定（PolaRiS）：
    单帧  (8,)      = [joint0..6, gripper]
    batch (*B, 8)   = 任意前置 batch 维度均可

输出约定（V-JEPA 2-AC）：
    state  (*B, 7)  = [x, y, z, roll, pitch, yaw, gripper]
    action (*B, 7)  = [Δx, Δy, Δz, Δroll, Δpitch, Δyaw, Δgripper]
"""

from __future__ import annotations
import warnings
import numpy as np
import torch
from scipy.spatial.transform import Rotation
import pytorch_kinematics as pk

# ── URDF ──────────────────────────────────────────────────────────────────────
_URDF_URL = (
    "https://raw.githubusercontent.com/RobotLocomotion/models"
    "/master/franka_description/urdf/panda_arm.urdf"
)

def _load_urdf(urdf_path: str | None) -> bytes:
    if urdf_path is not None:
        with open(urdf_path, "rb") as f:
            return f.read()
    import urllib.request, os, tempfile
    cache = os.path.join(tempfile.gettempdir(), "panda_arm_pk.urdf")
    if not os.path.exists(cache):
        print(f"[panda_fk] 下载 URDF → {cache}")
        urllib.request.urlretrieve(_URDF_URL, cache)
    with open(cache, "rb") as f:
        return f.read()


# ── 核心类 ─────────────────────────────────────────────────────────────────────
class PandaFK:
    """
    全 batch 的 Franka Panda 正运动学封装。

    所有方法均接受：
        单帧  np.ndarray shape (..., 8) 或 (8,)
        batch torch.Tensor shape (..., 8)
    并返回与输入相同 batch shape 的结果。
    """

    def __init__(
        self,
        urdf_path: str | None = None,
        ee_link: str = "panda_link8",   # flange 帧，与 DROID 一致
        device: str = "cpu",
        euler_seq: str = "xyz",         # DROID 使用外旋 xyz (roll-pitch-yaw)
    ):
        urdf_bytes = _load_urdf(urdf_path)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._chain = pk.build_serial_chain_from_urdf(urdf_bytes, ee_link)
        self._chain = self._chain.to(device=device)
        self.device = device
        self.euler_seq = euler_seq

    # ── 内部工具 ───────────────────────────────────────────────────────────────

    def _to_tensor(self, x) -> torch.Tensor:
        """numpy / list / tensor → float32 tensor，保留 batch 维度。"""
        if isinstance(x, torch.Tensor):
            return x.float().to(self.device)
        return torch.tensor(np.asarray(x, dtype=np.float32), device=self.device)

    def _fk_matrices(self, joints_7: torch.Tensor) -> torch.Tensor:
        """
        输入：(..., 7) tensor
        输出：(..., 4, 4) 齐次变换矩阵
        """
        batch_shape = joints_7.shape[:-1]           # e.g. (B,) or (B, T)
        flat = joints_7.reshape(-1, 7)              # (N, 7)
        T = self._chain.forward_kinematics(flat).get_matrix()  # (N, 4, 4)
        return T.reshape(*batch_shape, 4, 4)        # (..., 4, 4)

    @staticmethod
    def _mats_to_rpy(T: np.ndarray, euler_seq: str) -> np.ndarray:
        """
        (..., 4, 4) → (..., 3) 欧拉角，使用 scipy batch Rotation。
        """
        batch_shape = T.shape[:-2]
        R_flat = Rotation.from_matrix(T.reshape(-1, 4, 4)[:, :3, :3])
        rpy_flat = R_flat.as_euler(euler_seq, degrees=False)   # (N, 3)
        return rpy_flat.reshape(*batch_shape, 3)

    # ── 公共 API ───────────────────────────────────────────────────────────────

    def fk(self, joint_pos_8) -> np.ndarray:
        """
        关节位置 → 6D EE pose。

        输入：(*B, 8) 或 (8,)   [joint0..6, gripper]（gripper 被忽略）
        输出：(*B, 6)            [x, y, z, roll, pitch, yaw]
        """
        x = self._to_tensor(joint_pos_8)            # (*B, 8)
        T = self._fk_matrices(x[..., :7])           # (*B, 4, 4)
        T_np = T.cpu().numpy()
        xyz = T_np[..., :3, 3]                      # (*B, 3)
        rpy = self._mats_to_rpy(T_np, self.euler_seq)  # (*B, 3)
        return np.concatenate([xyz, rpy], axis=-1)  # (*B, 6)

    def state(self, joint_pos_8) -> np.ndarray:
        """
        PolaRiS action → V-JEPA 2-AC state。

        输入：(*B, 8)
        输出：(*B, 7)  [x, y, z, roll, pitch, yaw, gripper]
        """
        x = np.asarray(joint_pos_8, dtype=np.float32)
        pose6   = self.fk(x)                        # (*B, 6)
        gripper = x[..., 7:8]                       # (*B, 1)
        return np.concatenate([pose6, gripper], axis=-1)  # (*B, 7)

    def delta_action(self, joint_pos_t, joint_pos_t1) -> np.ndarray:
        """
        两帧 PolaRiS action → V-JEPA 2-AC delta action。

        输入：(*B, 8), (*B, 8)
        输出：(*B, 7)  [Δx, Δy, Δz, Δroll, Δpitch, Δyaw, Δgripper]

        旋转 delta = R_{t+1} · R_t^{-1}（world 系增量，
        与官方 mpc_utils.poses_to_diff 的 e_rotation @ s_rotation.T 一致）。
        gripper 为增量 gripper_t1 - gripper_t（官方 action 约定）。
        """
        q_t  = self._to_tensor(joint_pos_t)         # (*B, 8)
        q_t1 = self._to_tensor(joint_pos_t1)

        # 一次 FK 调用处理两帧（沿新 batch 维拼接）
        both   = torch.stack([q_t[..., :7], q_t1[..., :7]], dim=-2)  # (*B, 2, 7)
        T_both = self._fk_matrices(both).cpu().numpy()                # (*B, 2, 4, 4)

        T_t  = T_both[..., 0, :, :]   # (*B, 4, 4)
        T_t1 = T_both[..., 1, :, :]

        delta_xyz = T_t1[..., :3, 3] - T_t[..., :3, 3]              # (*B, 3)

        batch_shape = T_t.shape[:-2]
        R_t  = Rotation.from_matrix(T_t.reshape(-1, 4, 4)[:, :3, :3])
        R_t1 = Rotation.from_matrix(T_t1.reshape(-1, 4, 4)[:, :3, :3])
        delta_rpy = (R_t1 * R_t.inv()).as_euler(self.euler_seq, degrees=False)
        delta_rpy = delta_rpy.reshape(*batch_shape, 3)               # (*B, 3)

        gripper_delta = (
            np.asarray(joint_pos_t1, dtype=np.float32)[..., 7:8]
            - np.asarray(joint_pos_t, dtype=np.float32)[..., 7:8]
        )  # (*B, 1)
        return np.concatenate([delta_xyz, delta_rpy, gripper_delta], axis=-1)  # (*B, 7)

    def convert_trajectory(self, joint_pos_seq) -> dict:
        """
        轨迹批量转换（支持多轨迹并行）。

        输入：(*B, T, 8)
        输出：{
            "states"  : (*B, T,   7)
            "actions" : (*B, T-1, 7)
        }
        """
        x = np.asarray(joint_pos_seq, dtype=np.float32)  # (*B, T, 8)
        states  = self.state(x)                           # (*B, T, 7)
        actions = self.delta_action(x[..., :-1, :], x[..., 1:, :])  # (*B, T-1, 7)
        return {"states": states, "actions": actions}

    # ── 逆运动学（雅可比 DLS，用于 V-JEPA CEM 的 EE→关节 转换）──────────────────────
    # Panda 关节限位 (rad)，用于 clamp IK 解
    _JOINT_LIMITS = np.array([
        [-2.8973, 2.8973], [-1.7628, 1.7628], [-2.8973, 2.8973], [-3.0718, -0.0698],
        [-2.8973, 2.8973], [-0.0175, 3.7525], [-2.8973, 2.8973],
    ], dtype=np.float32)

    def ik(
        self,
        target_pose6,
        q_init7,
        iters: int = 60,
        damping: float = 0.05,
        step: float = 0.5,
        pos_tol: float = 2e-3,
        rot_tol: float = 5e-3,
        rot_weight: float = 1.0,
    ) -> np.ndarray:
        """
        目标 6D EE 位姿 → 7 关节角（阻尼最小二乘雅可比 IK，从 q_init7 热启动）。

        输入：target_pose6 (6,) [x,y,z,roll,pitch,yaw]；q_init7 (7,) 当前关节角
        输出：(7,) 关节角（已 clamp 到限位）；小 delta 下几步即收敛。
        专为 CEM 的小步长（|Δxyz|≤~0.05m/步）设计——热启动使其稳定不跳变。
        """
        target_pose6 = np.asarray(target_pose6, dtype=np.float32)
        p_tgt = target_pose6[:3]
        R_tgt = Rotation.from_euler(self.euler_seq, target_pose6[3:6], degrees=False).as_matrix()

        if rot_weight < 0:
            raise ValueError("rot_weight must be non-negative")

        lim = torch.tensor(self._JOINT_LIMITS, device=self.device)
        q_init = np.asarray(q_init7, dtype=np.float32).reshape(-1)
        if len(q_init) < 7:
            raise ValueError(f"Expected seven initial joints, got shape {q_init.shape}")
        q_init = q_init[:7]
        q_init = np.nan_to_num(q_init, nan=0.0, posinf=0.0, neginf=0.0)
        q = torch.tensor(q_init, device=self.device)
        q = torch.clamp(q, lim[:, 0], lim[:, 1])

        for _ in range(iters):
            T_cur = self._chain.forward_kinematics(q[None]).get_matrix()[0].cpu().numpy()  # (4,4)
            p_cur = T_cur[:3, 3]
            R_cur = T_cur[:3, :3]
            e_pos = p_tgt - p_cur                                      # (3,)
            e_rot = Rotation.from_matrix(R_tgt @ R_cur.T).as_rotvec()  # (3,) world-frame
            if (
                np.linalg.norm(e_pos) < pos_tol
                and (rot_weight == 0 or np.linalg.norm(e_rot) < rot_tol)
            ):
                break
            err = torch.tensor(
                np.concatenate([e_pos, rot_weight * e_rot]),
                dtype=torch.float32,
                device=self.device,
            )
            J = self._chain.jacobian(q[None])[0]                       # (6,7) [v;w] in base frame
            weighted_J = J.clone()
            weighted_J[3:] *= rot_weight
            JT = weighted_J.transpose(0, 1)
            reg = (damping ** 2) * torch.eye(6, device=self.device)
            dq = JT @ torch.linalg.solve(weighted_J @ JT + reg, err)   # (7,)
            if not torch.isfinite(dq).all():
                break
            q_next = q + step * dq
            if not torch.isfinite(q_next).all():
                break
            q = torch.clamp(q_next, lim[:, 0], lim[:, 1])

        return q.detach().cpu().numpy()


# # ── 验证 ───────────────────────────────────────────────────────────────────────
# if __name__ == "__main__":
#     robot = PandaFK()
#     q_neutral = np.array([0., -np.pi/4, 0., -3*np.pi/4, 0., np.pi/2, np.pi/4])

#     print("=" * 50)
#     print("1. 单帧 (8,)")
#     q = np.concatenate([q_neutral, [0.0]])
#     print("  state :", robot.state(q).round(4))
#     print("  fk    :", robot.fk(q).round(4))

#     print("\n2. 单 batch (B=4, 8)")
#     qs = np.tile(q, (4, 1))
#     qs[:, 0] += np.linspace(0, 0.3, 4)
#     s = robot.state(qs)
#     print("  states shape:", s.shape)
#     print("  states[0]   :", s[0].round(4))

#     print("\n3. delta_action batch (B=4, 8)")
#     q0 = qs
#     q1 = qs.copy(); q1[:, 0] += 0.01; q1[:, 7] = 1.0
#     a = robot.delta_action(q0, q1)
#     print("  actions shape:", a.shape)
#     print("  actions[0]   :", a[0].round(5))

#     print("\n4. 多维 batch (*B, 8) = (2, 3, 8)")
#     qs_md = np.tile(q, (2, 3, 1))
#     qs_md[..., 0] += np.random.uniform(0, 0.1, (2, 3))
#     s_md = robot.state(qs_md)
#     print("  states shape:", s_md.shape)   # (2, 3, 7)

#     print("\n5. 轨迹转换 (T=16, 8)")
#     traj = np.tile(q, (16, 1))
#     traj[:, 0] += np.linspace(0, 0.2, 16)
#     traj[8:, 7] = 1.0
#     result = robot.convert_trajectory(traj)
#     print("  states  shape:", result["states"].shape)   # (16, 7)
#     print("  actions shape:", result["actions"].shape)  # (15, 7)

#     print("\n6. 多轨迹批量 (B=8, T=16, 8)")
#     trajs = np.tile(traj, (8, 1, 1))
#     result_b = robot.convert_trajectory(trajs)
#     print("  states  shape:", result_b["states"].shape)   # (8, 16, 7)
#     print("  actions shape:", result_b["actions"].shape)  # (8, 15, 7)

#     print("\n中立姿态位置验证:", robot.fk(q)[:3].round(4), "预期 [0.3069, 0, 0.5903]",
#           "✓" if np.allclose(robot.fk(q)[:3], [0.3069, 0., 0.5903], atol=1e-3) else "❌")
