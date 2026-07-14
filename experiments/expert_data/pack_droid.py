"""
Pack staged trajectories (from scripts/eval.py --record-traj) into the DROID episode
format that third_party/vjepa2/app/vjepa_droid/droid.py's DROIDVideoDataset consumes:

  <out>/<episode>/metadata.json                     "left/right_mp4_path" -> recordings/MP4/ext.mp4
  <out>/<episode>/trajectory.h5                     observation/robot_state/cartesian_position [T,6]
                                                    observation/robot_state/gripper_position   [T]
                                                    observation/camera_extrinsics/ext_left     [T,6]
  <out>/<episode>/recordings/MP4/ext.mp4            T frames @ 3.75 fps
  <out>/dataset.csv                                 one episode dir per line (init_data's data_path)

Cartesian poses come from PandaFK forward kinematics on the staged joint states, so run
this in the openpi venv (pytorch_kinematics available there):
  cd /workspace/polaris/third_party/openpi
  uv run python /workspace/polaris/experiments/expert_data/pack_droid.py \
      --staging /workspace/polaris/runs/expert_staging \
      --out /workspace/polaris/runs/droid_foodbussing
"""
import argparse
import json
import shutil
from pathlib import Path

import cv2
import h5py
import numpy as np

from vjepa2.FK import PandaFK


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--staging", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--include-failures", action="store_true")
    args = ap.parse_args()

    robot = PandaFK(device="cpu")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    packed = []

    for ep_dir in sorted(Path(args.staging).glob("ep_*")):
        required = (ep_dir / "meta.json", ep_dir / "joints.npy", ep_dir / "video.mp4")
        missing = [path.name for path in required if not path.is_file()]
        if missing:
            print(f"!! {ep_dir.name}: missing {', '.join(missing)} -> skipped")
            continue
        meta = json.loads((ep_dir / "meta.json").read_text())
        if not meta["success"] and not args.include_failures:
            continue
        joints = np.load(ep_dir / "joints.npy")            # (T, 8)
        if joints.ndim != 2 or joints.shape[1] != 8 or not np.isfinite(joints).all():
            print(f"!! {ep_dir.name}: invalid joints array {joints.shape} -> skipped")
            continue
        T = len(joints)

        cap = cv2.VideoCapture(str(ep_dir / "video.mp4"))
        vlen = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if vlen != T:
            print(f"!! {ep_dir.name}: video has {vlen} frames but {T} joint rows -> skipped")
            continue

        poses = np.asarray(robot.state(joints.astype(np.float32)))  # (T, 7) [xyz rpy grip]
        dst = out / ep_dir.name
        (dst / "recordings/MP4").mkdir(parents=True, exist_ok=True)
        shutil.copy(ep_dir / "video.mp4", dst / "recordings/MP4/ext.mp4")
        with h5py.File(dst / "trajectory.h5", "w") as f:
            f.create_dataset("observation/robot_state/cartesian_position", data=poses[:, :6])
            f.create_dataset("observation/robot_state/gripper_position", data=joints[:, 7])
            # static sim camera; only indexed (not used) when camera_frame=False
            f.create_dataset("observation/camera_extrinsics/ext_left", data=np.zeros((T, 6), np.float32))
        (dst / "metadata.json").write_text(json.dumps({
            "left_mp4_path": "recordings/MP4/ext.mp4",
            "right_mp4_path": "recordings/MP4/ext.mp4",
            "source_episode": ep_dir.name,
            **{
                key: meta[key]
                for key in (
                    "ic_index", "success", "progress", "control_hz", "record_every",
                    "frame_control_steps",
                )
                if key in meta
            },
        }))
        packed.append(str(dst.resolve()))
        print(f"packed {ep_dir.name}  (T={T}, IC {meta['ic_index']})")

    csv_path = out / "dataset.csv"
    csv_path.write_text("\n".join(packed) + "\n")
    ics = {json.loads((Path(p) / "metadata.json").read_text())["ic_index"] for p in packed}
    print(f"\n{len(packed)} episodes -> {csv_path}   (distinct ICs: {len(ics)})")


if __name__ == "__main__":
    main()
