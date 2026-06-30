# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
from logging import getLogger
from math import ceil

import numpy as np
import pandas as pd
import torch
import torch.utils.data
from decord import VideoReader, cpu
from scipy.spatial.transform import Rotation
_GLOBAL_SEED = 0
logger = getLogger()


def init_behavior_data(
    data_path,              
    batch_size,
    frames_per_clip=16,
    fps=4,                  
    rank=0,
    world_size=1,
    camera_views=["head","left_wrist", "right_wrist"],  
    drop_last=True,
    num_workers=10,
    pin_mem=True,
    persistent_workers=True,
    collator=None,
    transform=None,
    camera_frame=False,   
    tubelet_size=2,
):
    dataset = Behavior1KVideoDataset(
        base_path=data_path,
        frames_per_clip=frames_per_clip,
        transform=transform,
        fps=fps,
        camera_views=camera_views,
        tubelet_size=tubelet_size,
    )

    # dist_sampler = torch.utils.data.distributed.DistributedSampler(
    #     dataset, num_replicas=world_size, rank=rank, shuffle=True
    # )
    dist_sampler = torch.utils.data.RandomSampler(dataset)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=collator,
        sampler=dist_sampler,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0) and persistent_workers,
    )

    logger.info("Behavior1KDataset unsupervised data loader created successfully (Global Random Shuffle).")

    return data_loader, dist_sampler


class Behavior1KVideoDataset(torch.utils.data.Dataset):
    """Behavior1K dataset specifically formatted for V-JEPA AC Predictor training.

    Scans a base path, flattens all tasks/episodes, and synchronizes actions into 4fps via Delta Summation.
    """

    def __init__(
        self,
        base_path,
        camera_views=["head","left_wrist", "right_wrist"],
        frames_per_clip=16,
        fps=4,
        transform=None,
        tubelet_size=2,
    ):
        self.base_path = base_path
        self.frames_per_clip = frames_per_clip
        self.fps = fps
        self.transform = transform
        self.camera_views = camera_views
        self.tubelet_size = tubelet_size
        if VideoReader is None:
            raise ImportError('Unable to import "decord" which is required to read videos.')

        self.total_data_dir = os.path.join(base_path, "data")
        self.total_video_dir = os.path.join(base_path, "videos")

        all_tasks = [t for t in os.listdir(self.total_data_dir) if os.path.isdir(os.path.join(self.total_data_dir, t))]
        
        self.samples = []
        for task_id in all_tasks:
            task_data_path = os.path.join(self.total_data_dir, task_id)
            all_files = os.listdir(task_data_path)
            
            for f in all_files:
                episode_id = os.path.splitext(f)[0]
                self.samples.append({
                    "task_id": task_id,
                    "episode_id": episode_id
                })

        self.samples.sort(key=lambda x: (x["task_id"], x["episode_id"]))
        
        logger.info(f"共{len(all_tasks)} 个 Tasks，共 {len(self.samples)} 个 Episodes。")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample_info = self.samples[index]
        task_id = sample_info["task_id"]
        episode_id = sample_info["episode_id"]

        loaded_video = False
        while not loaded_video:
            try:
                buffer, actions, states, extrinsics, indices = self.load_episode_data(task_id, episode_id)
                loaded_video = True
            except Exception as e:
                logger.info(f"加载失败，自动跳过并随机替换样本。错误: {task_id=}, {episode_id=}, {e=}")
                loaded_video = False
                index = np.random.randint(self.__len__())
                sample_info = self.samples[index]
                task_id = sample_info["task_id"]
                episode_id = sample_info["episode_id"]

        return buffer, actions, states, extrinsics, indices

    def load_episode_data(self, task_id, episode_id):
        task_data_dir = os.path.join(self.total_data_dir, task_id)
        parquet_path = os.path.join(task_data_dir, f"{episode_id}.parquet")

        df = pd.read_parquet(parquet_path)

        selected_view = self.camera_views[torch.randint(0, len(self.camera_views), (1,)).item()]
        folder_name = f"observation.images.rgb.{selected_view}"
        vpath = os.path.join(self.total_video_dir, task_id, folder_name, f"{episode_id}.mp4")
        


       
            
        n_steps = self.frames_per_clip  
        time_step = 1.0 / self.fps


        clip_duration = (n_steps - 1) * time_step 
        
        timestamps = df['timestamp'].values
        total_duration = timestamps[-1] - timestamps[0]

        if total_duration < clip_duration:
            raise Exception(f"视频或数据表总时长不足以覆盖一个 CLIP 的窗口 {total_duration=}s < {clip_duration=}s")

        max_start_time = total_duration - clip_duration
        start_time = np.random.uniform(0, max_start_time) + timestamps[0]
        
        target_timestamps = start_time + np.arange(n_steps) * time_step
        indices_state = np.searchsorted(timestamps, target_timestamps)
        indices_state = np.clip(indices_state, 0, len(timestamps) - 1)


        # full_actions = df['action'].iloc[indices_state[:-1]].values
        # actions = np.stack(full_actions, axis=0)
        full_state = df['observation.state'].iloc[indices_state].values
        if isinstance(full_state[0], str):
            full_state = np.array([np.fromstring(s.strip('[]'), sep=',') for s in full_state])
        elif not isinstance(full_state, np.ndarray) or full_state.ndim == 1:
            full_state = np.stack(full_state, axis=0)
        states = np.concatenate([
            full_state[:, 253:256],
            full_state[:, 236:240],
            full_state[:, 158:165],
            ((full_state[:, 193] + full_state[:, 194]) * 10).reshape(-1, 1),
            full_state[:, 197:204],
            ((full_state[:, 232] + full_state[:, 233]) * 10).reshape(-1, 1),
        ], axis=-1)  # [n_steps, 23]
        # actions = states[1:] - states[:-1]
        actions = states[1:]
        # extrinsics 用 indices_state
        extrinsics_raw = df['observation.cam_rel_poses'].iloc[indices_state].values
        if isinstance(extrinsics_raw[0], str):
            extrinsics = np.array([np.fromstring(e.strip('[]'), sep=',') for e in extrinsics_raw])
        # [n_steps, extrinsics_dim]
        extrinsics = []
        for e in extrinsics_raw:
            cams = []
            for i in range(3):
                xyz = e[i*7 : i*7+3]
                quat = e[i*7+3 : i*7+7]  # [qx, qy, qz, qw]
                euler = Rotation.from_quat(quat).as_euler('xyz', degrees=False)
                cams.append(np.concatenate([xyz, euler]))  # 6维
            extrinsics.append(np.concatenate(cams))  # 18维
        extrinsics = np.stack(extrinsics, axis=0)  # [n_steps, 18]
        # buffer 用 indices_video
        vr = VideoReader(vpath, num_threads=-1, ctx=cpu(0))
        vfps = vr.get_avg_fps() 
        vlen = len(vr)
        
        indices_video = ((target_timestamps - timestamps[0]) * vfps).astype(np.int64)
        if indices_video.max() >= vlen or indices_video.min() < 0:
            raise Exception(f"video length error: {vpath}, max_req={indices_video.max()}, vlen={vlen}")
        indices_video = np.clip(indices_video, 0, vlen - 1)
        
        buffer = vr.get_batch(indices_video).asnumpy()
        if self.transform is not None:
            buffer = self.transform(buffer)

        return buffer, actions, states, extrinsics, indices_video