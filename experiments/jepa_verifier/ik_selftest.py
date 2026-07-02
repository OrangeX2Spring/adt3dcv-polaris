"""
Standalone IK sanity check — RUN THIS BEFORE the CEM eval.

Round-trip: random joints q -> FK -> EE pose -> IK(pose, seed=q+noise) -> q' -> FK -> pose'.
A working IK must recover pose' ~= pose (position < few mm, rotation < ~0.5deg). Also tests the
small-delta regime the CEM planner actually uses (perturb EE by <=5cm and re-solve from current q).

Run in the openpi venv on the eval machine (needs pytorch_kinematics):
  cd /workspace/polaris/third_party/openpi
  uv run python /workspace/polaris/experiments/jepa_verifier/ik_selftest.py
"""
import numpy as np
from scipy.spatial.transform import Rotation
from vjepa2.FK import PandaFK

rng = np.random.default_rng(0)
robot = PandaFK(device="cpu")
q_home = np.array([0.0, -np.pi / 4, 0.0, -3 * np.pi / 4, 0.0, np.pi / 2, np.pi / 4], np.float32)

print("=== round-trip from perturbed home ===")
pos_err, rot_err = [], []
for _ in range(20):
    q = q_home + rng.uniform(-0.3, 0.3, 7).astype(np.float32)
    pose = robot.state(np.concatenate([q, [0.0]]))[:6]         # target EE pose
    seed = q + rng.uniform(-0.2, 0.2, 7).astype(np.float32)     # IK seed (off by a bit)
    q_sol = robot.ik(pose, seed)
    pose2 = robot.state(np.concatenate([q_sol, [0.0]]))[:6]
    pos_err.append(np.linalg.norm(pose2[:3] - pose[:3]))
    R = Rotation.from_euler("xyz", pose2[3:6]) * Rotation.from_euler("xyz", pose[3:6]).inv()
    rot_err.append(np.linalg.norm(R.as_rotvec()))
print(f"pos err: mean {np.mean(pos_err)*1000:.2f}mm  max {np.max(pos_err)*1000:.2f}mm")
print(f"rot err: mean {np.degrees(np.mean(rot_err)):.3f}deg  max {np.degrees(np.max(rot_err)):.3f}deg")

print("\n=== small-delta regime (CEM-like: <=5cm step, seed=current q) ===")
pos_err = []
for _ in range(20):
    q = q_home + rng.uniform(-0.3, 0.3, 7).astype(np.float32)
    pose = robot.state(np.concatenate([q, [0.0]]))[:6]
    tgt = pose.copy(); tgt[:3] += rng.uniform(-0.05, 0.05, 3)   # 5cm EE delta
    q_sol = robot.ik(tgt, q)                                    # seed = current joints
    pose2 = robot.state(np.concatenate([q_sol, [0.0]]))[:6]
    pos_err.append(np.linalg.norm(pose2[:3] - tgt[:3]))
print(f"pos err vs target: mean {np.mean(pos_err)*1000:.2f}mm  max {np.max(pos_err)*1000:.2f}mm")
print("\nPASS if errors are sub-mm / sub-degree. If not, IK is broken — fix before CEM eval.")
