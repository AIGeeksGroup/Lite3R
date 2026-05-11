"""Synthetic multi-view data for local smoke tests.

Produces tensors of the right shape and self-consistent geometry so that
loss / training pipelines can be exercised end-to-end without downloading the
real BlendedMVS dataset.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


def _project_world_to_pixels(world: np.ndarray, extr: np.ndarray, intr: np.ndarray):
    """world: (H, W, 3) → cam space → pixel space. Returns depth map (H, W)."""
    R = extr[:3, :3]
    t = extr[:3, 3]
    cam = world @ R.T + t  # (H, W, 3)
    return cam[..., 2].astype(np.float32)


class DummyMultiViewDataset(Dataset):
    def __init__(self, length: int = 16, img_per_seq: int = 2, img_size: int = 224, seed: int = 0):
        self.length = length
        self.S = img_per_seq
        self.img_size = img_size
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return self.length

    def __getitem__(self, idx: int):
        H = W = self.img_size
        # build a random fronto-parallel "scene" by sampling depths in [1, 5]
        depth0 = self.rng.uniform(1.0, 5.0, size=(H, W)).astype(np.float32)
        # build pinhole intrinsics
        f = max(H, W) * 1.0
        intr = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]], dtype=np.float32)
        # camera 0 is identity; subsequent cameras translate slightly along x
        imgs, depths, extrs, intrs, masks, wpts = [], [], [], [], [], []
        # build world points from camera 0
        yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
        x_n = (xx - W / 2) / f
        y_n = (yy - H / 2) / f
        cam0 = np.stack([x_n * depth0, y_n * depth0, depth0], axis=-1)  # (H, W, 3)
        # cam0 is in world frame because extr0 = I
        world = cam0
        for s in range(self.S):
            extr = np.eye(4, dtype=np.float32)
            extr[0, 3] = -0.05 * s  # baseline translation along x
            # project world to camera s
            R = extr[:3, :3]; t = extr[:3, 3]
            cam_s = world @ R.T + t
            depth_s = cam_s[..., 2].astype(np.float32)
            # color = a function of depth and frame index, plus noise
            base = (depth_s - depth_s.min()) / max(depth_s.max() - depth_s.min(), 1e-6)
            color = np.stack([base, np.full_like(base, 0.5), 1.0 - base], axis=-1)
            color = np.clip(color + 0.02 * self.rng.standard_normal(color.shape), 0, 1).astype(np.float32)
            mask = (depth_s > 0).astype(bool)
            imgs.append(color)
            depths.append(depth_s)
            extrs.append(extr)
            intrs.append(intr)
            masks.append(mask)
            wpts.append(world.astype(np.float32))
        return {
            "images": torch.from_numpy(np.stack(imgs)).permute(0, 3, 1, 2).contiguous().float(),
            "depths": torch.from_numpy(np.stack(depths)).float(),
            "extrinsics": torch.from_numpy(np.stack(extrs)).float(),
            "intrinsics": torch.from_numpy(np.stack(intrs)).float(),
            "point_masks": torch.from_numpy(np.stack(masks)).bool(),
            "world_points": torch.from_numpy(np.stack(wpts)).float(),
            "seq_name": f"dummy_{idx}",
        }
