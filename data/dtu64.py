"""DTU-64 dataset loader (pose-only) for cross-dataset evaluation.

DTU-64 is a 64-frame-per-scene subset of DTU. It has RGB + camera params but
**no GT depth maps** (the BlendedMVS-style PFM rendered_depth_maps are not
distributed with this subset). So it is usable for **pose** and **efficiency**
metrics only — depth_metrics will see all-False masks and return NaN, which
`aggregate()` already drops. Chamfer/F-score likewise skipped.

Layout (server):
    {root}/
      Cameras/{frame:08d}_cam.txt          # shared across all scans
      scan{N}/image/{frame:06d}.png        # 64 frames per scan
      ...

cam.txt format is identical to BlendedMVS, so we reuse `_read_cam` from
data.blendedmvs.
"""

from __future__ import annotations

import os
from glob import glob
from typing import List

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from data.blendedmvs import _read_cam, collate_fn  # re-exported for eval_common


class DTU64Dataset(Dataset):
    """Pose-only multi-view dataset producing VGGT-style batches.

    Mirrors `BlendedMVSDataset` interface so the eval pipeline is unchanged.
    Returns zeros + all-False masks for depth/world_points so depth and
    geometry metrics are safely NaN-filtered downstream.
    """

    def __init__(
        self,
        root: str,
        img_per_seq: int = 2,
        img_size: int = 518,
        max_scenes: int | None = None,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.root = root
        self.S = img_per_seq
        self.img_size = img_size
        self.rng = np.random.default_rng(seed)
        self.cam_root = os.path.join(root, "Cameras")
        if not os.path.isdir(self.cam_root):
            raise FileNotFoundError(f"DTU64 missing Cameras/: {self.cam_root}")

        scans = sorted(
            p for p in glob(os.path.join(root, "scan*"))
            if os.path.isdir(p) and os.path.isdir(os.path.join(p, "image"))
        )
        if max_scenes is not None:
            scans = scans[:max_scenes]
        if not scans:
            raise FileNotFoundError(f"No DTU64 scans under {root}")
        self.scans = scans

        # Index every scan's frame list once.
        self.scene_frames: List[List[int]] = []
        for s in self.scans:
            ims = sorted(glob(os.path.join(s, "image", "*.png")))
            ids = sorted({int(os.path.basename(p).split(".")[0]) for p in ims})
            self.scene_frames.append(ids if len(ids) >= self.S else [])
        self.scans = [s for s, f in zip(self.scans, self.scene_frames) if f]
        self.scene_frames = [f for f in self.scene_frames if f]

    def __len__(self) -> int:
        return len(self.scans)

    def _load_one(self, scan: str, frame_id: int):
        img_path = os.path.join(scan, "image", f"{frame_id:06d}.png")
        cam_path = os.path.join(self.cam_root, f"{frame_id:08d}_cam.txt")
        img = np.array(Image.open(img_path).convert("RGB"), dtype=np.float32) / 255.0
        extr, intr, _ = _read_cam(cam_path)
        return img, extr, intr

    def _resize_pack(self, img, intr, target):
        H0, W0 = img.shape[:2]
        scale = target / min(H0, W0)
        Hs, Ws = int(round(H0 * scale)), int(round(W0 * scale))
        img = np.array(Image.fromarray((img * 255).astype(np.uint8)).resize((Ws, Hs)),
                        dtype=np.float32) / 255.0
        intr = intr.copy()
        intr[0, 0] *= scale; intr[1, 1] *= scale
        intr[0, 2] *= scale; intr[1, 2] *= scale
        y0 = max(0, (Hs - target) // 2)
        x0 = max(0, (Ws - target) // 2)
        img = img[y0:y0 + target, x0:x0 + target]
        intr[0, 2] -= x0; intr[1, 2] -= y0
        return img, intr

    def __getitem__(self, idx: int):
        scan = self.scans[idx]
        frames = self.scene_frames[idx]

        anchor = int(self.rng.choice(frames))
        others = self.rng.choice([f for f in frames if f != anchor],
                                  size=self.S - 1, replace=len(frames) - 1 < self.S - 1)
        sel = [anchor] + [int(o) for o in np.atleast_1d(others)]

        imgs, extrs, intrs = [], [], []
        for fid in sel:
            try:
                img, extr, intr = self._load_one(scan, fid)
            except FileNotFoundError:
                fid2 = int(self.rng.choice([f for f in frames if f != fid]))
                img, extr, intr = self._load_one(scan, fid2)
            img, intr = self._resize_pack(img, intr, self.img_size)
            imgs.append(img); extrs.append(extr); intrs.append(intr)

        extrs_arr = np.stack(extrs).astype(np.float32, copy=True)
        # Translation magnitudes in DTU cam.txt are in mm (~hundreds). Normalise
        # by the max view-translation so trans_err interpretation is comparable
        # to BlendedMVS (which we also unit-scale-normalised at dataset time).
        # Pose convention is *view-0-relative w2c* downstream so absolute scale
        # is mostly irrelevant — but trans_err is dim-ful, so we still want
        # consistent units across datasets.
        t_max = float(np.linalg.norm(extrs_arr[:, :3, 3], axis=-1).max())
        scene_scale = max(t_max, 1e-3)
        extrs_arr[:, :3, 3] /= scene_scale

        H, W = self.img_size, self.img_size
        S = len(sel)
        imgs_t = torch.from_numpy(np.stack(imgs)).permute(0, 3, 1, 2).contiguous().float()
        extrs_t = torch.from_numpy(extrs_arr).float()
        intrs_t = torch.from_numpy(np.stack(intrs)).float()
        # Depth GT not available — emit zero-filled placeholders + all-False
        # masks so depth_metrics returns NaN (filtered by aggregate()) and
        # chamfer hits its except branch cleanly.
        zeros_d = torch.zeros((S, H, W), dtype=torch.float32)
        zeros_wp = torch.zeros((S, H, W, 3), dtype=torch.float32)
        masks = torch.zeros((S, H, W), dtype=torch.bool)
        return {
            "images": imgs_t,
            "depths": zeros_d,
            "extrinsics": extrs_t,
            "intrinsics": intrs_t,
            "point_masks": masks,
            "world_points": zeros_wp,
            "seq_name": os.path.basename(scan),
        }
