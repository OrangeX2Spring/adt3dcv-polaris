#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import logging

import numpy as np
import torch
from torch import nn
from polaris.splat_renderer.utils.graphics_utils import (
    getWorld2View2,
    getProjectionMatrix,
)


class Camera(nn.Module):
    def __init__(
        self,
        colmap_id,
        R,
        T,
        FoVx,
        FoVy,
        image,
        gt_alpha_mask,
        image_name,
        uid,
        trans=np.array([0.0, 0.0, 0.0]),
        scale=1.0,
        data_device="cuda",
    ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(
                f"[Warning] Custom device {data_device} failed, fallback to default cuda device"
            )
            self.data_device = torch.device("cuda")

        self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        if gt_alpha_mask is not None:
            # self.original_image *= gt_alpha_mask.to(self.data_device)
            self.gt_alpha_mask = gt_alpha_mask.to(self.data_device)
        else:
            self.original_image *= torch.ones(
                (1, self.image_height, self.image_width), device=self.data_device
            )
            self.gt_alpha_mask = None

        self.zfar = 100
        self.znear = 0.05
        # self.znear = 10

        self.trans = trans
        self.scale = scale

        self.world_view_transform = (
            torch.tensor(getWorld2View2(R, T, trans, scale))
            .transpose(0, 1)
            .to(self.data_device)
        )
        self.projection_matrix = (
            getProjectionMatrix(
                znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy
            )
            .transpose(0, 1)
            .to(self.data_device)
        )
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0).bmm(
                self.projection_matrix.unsqueeze(0)
            )
        ).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

    def set_extrinsics(self, R, T):
        prepared = self._prepare_extrinsics(R, T)
        if prepared is None:
            return False
        R, T = prepared
        center = np.zeros(3)
        world_view_transform = (
            torch.tensor(getWorld2View2(R, center, T, self.scale))
            .transpose(0, 1)
            .to(self.data_device)
        )
        try:
            camera_center = torch.linalg.inv(world_view_transform)[3, :3]
        except RuntimeError:
            logging.warning("Rejected singular camera transform for %s", self.image_name)
            return False
        full_proj_transform = (
            world_view_transform.unsqueeze(0).bmm(
                self.projection_matrix.unsqueeze(0)
            )
        ).squeeze(0)
        self.R = R
        self.T = T
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        self.camera_center = camera_center
        return True

    def set_extrinsics2(self, R, T):
        prepared = self._prepare_extrinsics(R, T)
        if prepared is None:
            return False
        R, T = prepared
        world_view_transform = (
            torch.tensor(getWorld2View2(R, T, self.trans, self.scale))
            .transpose(0, 1)
            .to(self.data_device)
        )
        try:
            camera_center = torch.linalg.inv(world_view_transform)[3, :3]
        except RuntimeError:
            logging.warning("Rejected singular camera transform for %s", self.image_name)
            return False
        full_proj_transform = (
            world_view_transform.unsqueeze(0).bmm(
                self.projection_matrix.unsqueeze(0)
            )
        ).squeeze(0)
        self.R = R
        self.T = T
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        self.camera_center = camera_center
        return True

    def _prepare_extrinsics(self, R, T):
        rotation = np.asarray(R, dtype=np.float64)
        translation = np.asarray(T, dtype=np.float64).reshape(-1)
        if (
            rotation.shape != (3, 3)
            or len(translation) != 3
            or not np.all(np.isfinite(rotation))
            or not np.all(np.isfinite(translation))
        ):
            logging.warning("Rejected invalid camera extrinsics for %s", self.image_name)
            return None

        try:
            left, singular_values, right = np.linalg.svd(rotation)
        except np.linalg.LinAlgError:
            logging.warning("Rejected unstable camera rotation for %s", self.image_name)
            return None
        if singular_values[-1] < 1e-4:
            logging.warning("Rejected near-singular camera rotation for %s", self.image_name)
            return None
        rotation = left @ right
        if np.linalg.det(rotation) < 0:
            left[:, -1] *= -1
            rotation = left @ right
        return rotation.astype(np.float32), translation.astype(np.float32)


class MiniCam:
    def __init__(
        self,
        width,
        height,
        fovy,
        fovx,
        znear,
        zfar,
        world_view_transform,
        full_proj_transform,
    ):
        self.image_width = width
        self.image_height = height
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]
