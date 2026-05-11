"""BlendedMVS Low-Res dataset loader compatible with both VGGT and DA3.

Produces batches in the format expected by VGGT's `MultitaskLoss`:
    {
      "images":      (B, S, 3, H, W) float32 in [0, 1],
      "extrinsics":  (B, S, 4, 4)   float32  (OpenCV w2c convention),
      "intrinsics":  (B, S, 3, 3)   float32,
      "depths":      (B, S, H, W)   float32,
      "point_masks": (B, S, H, W)   bool,
      "world_points":(B, S, H, W, 3) float32,
      "seq_name":    str (collated as list),
    }

The same dict is consumed unchanged by DA3 training; the DA3 head outputs are
converted to VGGT's `pose_enc_list` format inside the training loop.

BlendedMVS layout assumed (after unzipping BlendedMVS.z01..zNN + BlendedMVS.zip):
    {root}/{scene_id}/blended_images/{frame:08d}.jpg
    {root}/{scene_id}/cams/{frame:08d}_cam.txt
    {root}/{scene_id}/rendered_depth_maps/{frame:08d}.pfm
    {root}/{scene_id}/cams/pair.txt          (optional, for multi-view sampling)
"""

from __future__ import annotations

import os
import re
from glob import glob
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def _read_pfm(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        header = f.readline().rstrip().decode("ascii")
        if header == "PF":
            color = True
        elif header == "Pf":
            color = False
        else:
            raise ValueError(f"Not a PFM file: {path}")
        dims = f.readline().decode("ascii").strip()
        m = re.match(r"^(\d+)\s+(\d+)\s*$", dims)
        if not m:
            raise ValueError(f"Bad PFM dim line: {dims}")
        w, h = int(m.group(1)), int(m.group(2))
        scale = float(f.readline().decode("ascii").strip())
        endian = "<" if scale < 0 else ">"
        data = np.fromfile(f, endian + "f")
    shape = (h, w, 3) if color else (h, w)
    data = data.reshape(shape)
    data = np.flipud(data)
    return data.astype(np.float32)


def _read_cam(path: str) -> Tuple[np.ndarray, np.ndarray, Tuple[float, float]]:
    """Parse BlendedMVS *_cam.txt → (extrinsic 4x4, intrinsic 3x3, (dmin, dmax))."""
    with open(path, "r") as f:
        text = f.read()
    extr_block = re.search(r"extrinsic\s+([-\d\.\sEe\+]+)", text)
    intr_block = re.search(r"intrinsic\s+([-\d\.\sEe\+]+)", text)
    assert extr_block and intr_block, f"Bad cam file {path}"
    extr_vals = [float(x) for x in extr_block.group(1).split()][:16]
    intr_vals = [float(x) for x in intr_block.group(1).split()][:9]
    extr = np.array(extr_vals, dtype=np.float32).reshape(4, 4)
    intr = np.array(intr_vals, dtype=np.float32).reshape(3, 3)
    # depth range is the last two/four numbers after intrinsic
    tail = re.search(r"intrinsic[\s\S]+?\n\n([-\d\.\sEe\+]+)$", text)
    dmin, dmax = 0.0, 100.0
    if tail:
        nums = [float(x) for x in tail.group(1).split()]
        if len(nums) >= 2:
            dmin = nums[0]
            interval = nums[1]
            if len(nums) >= 4:
                dmax = nums[2]
            else:
                # use 192 planes as in MVSNet conventions
                dmax = dmin + interval * 192
    return extr, intr, (float(dmin), float(dmax))


def _depth_to_world_points(depth: np.ndarray, extr_w2c: np.ndarray, intr: np.ndarray) -> np.ndarray:
    """Convert depth map to world-space 3D points. depth: (H, W). returns (H, W, 3)."""
    H, W = depth.shape
    fx, fy = intr[0, 0], intr[1, 1]
    cx, cy = intr[0, 2], intr[1, 2]
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    xn = (xx - cx) / max(fx, 1e-6)
    yn = (yy - cy) / max(fy, 1e-6)
    cam = np.stack([xn * depth, yn * depth, depth], axis=-1)  # (H, W, 3)
    R = extr_w2c[:3, :3]
    t = extr_w2c[:3, 3]
    # world = R^T @ (cam - t)
    flat = cam.reshape(-1, 3)
    world = (flat - t[None, :]) @ R  # equiv to R^T (flat - t)^T
    return world.reshape(H, W, 3).astype(np.float32)


class BlendedMVSDataset(Dataset):
    """Multi-view dataset producing VGGT-style batches.

    Args:
        root: path to dir containing per-scene folders
        img_per_seq: number of frames per training item (S)
        img_size: height (and width) after square center-crop
        scene_filter: optional substring to filter scene ids
        max_scenes: limit number of scenes (for debugging)
    """

    def __init__(
        self,
        root: str,
        img_per_seq: int = 4,
        img_size: int = 224,
        scene_filter: str | None = None,
        max_scenes: int | None = None,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.root = root
        self.S = img_per_seq
        self.img_size = img_size
        self.rng = np.random.default_rng(seed)

        scenes = sorted(
            p for p in glob(os.path.join(root, "*"))
            if os.path.isdir(p) and os.path.isdir(os.path.join(p, "blended_images"))
        )
        if scene_filter:
            scenes = [s for s in scenes if scene_filter in os.path.basename(s)]
        if max_scenes is not None:
            scenes = scenes[:max_scenes]
        if not scenes:
            raise FileNotFoundError(f"No BlendedMVS scenes found under {root}")
        self.scenes = scenes

        # Index every scene's frame list once.
        self.scene_frames: List[List[int]] = []
        for s in self.scenes:
            ims = sorted(glob(os.path.join(s, "blended_images", "*.jpg")))
            ids = [
                int(os.path.basename(p).split(".")[0].split("_")[0])
                for p in ims
                if "_masked" not in p
            ]
            ids = sorted(set(ids))
            if len(ids) >= self.S:
                self.scene_frames.append(ids)
            else:
                self.scene_frames.append([])

        # Filter out scenes too small for S frames.
        self.scenes = [s for s, f in zip(self.scenes, self.scene_frames) if f]
        self.scene_frames = [f for f in self.scene_frames if f]

    def __len__(self) -> int:
        return len(self.scenes)

    def _load_one(self, scene: str, frame_id: int):
        img_path = os.path.join(scene, "blended_images", f"{frame_id:08d}.jpg")
        cam_path = os.path.join(scene, "cams", f"{frame_id:08d}_cam.txt")
        dep_path = os.path.join(scene, "rendered_depth_maps", f"{frame_id:08d}.pfm")
        img = np.array(Image.open(img_path).convert("RGB"), dtype=np.float32) / 255.0
        depth = _read_pfm(dep_path)
        extr, intr, depth_range = _read_cam(cam_path)
        return img, depth, extr, intr, depth_range

    def _resize_pack(self, img, depth, intr, target):
        H0, W0 = img.shape[:2]
        # scale to fit target on the short side, then center-crop to (target, target)
        scale = target / min(H0, W0)
        Hs, Ws = int(round(H0 * scale)), int(round(W0 * scale))
        img = np.array(Image.fromarray((img * 255).astype(np.uint8)).resize((Ws, Hs)),
                        dtype=np.float32) / 255.0
        depth = np.array(Image.fromarray(depth).resize((Ws, Hs), resample=Image.NEAREST),
                          dtype=np.float32)
        intr = intr.copy()
        intr[0, 0] *= scale; intr[1, 1] *= scale
        intr[0, 2] *= scale; intr[1, 2] *= scale

        y0 = max(0, (Hs - target) // 2)
        x0 = max(0, (Ws - target) // 2)
        img = img[y0:y0 + target, x0:x0 + target]
        depth = depth[y0:y0 + target, x0:x0 + target]
        intr[0, 2] -= x0; intr[1, 2] -= y0
        return img, depth, intr

    def __getitem__(self, idx: int):
        scene = self.scenes[idx]
        frames = self.scene_frames[idx]

        # pick a random anchor and (S-1) other frames
        anchor = self.rng.choice(frames)
        others = self.rng.choice([f for f in frames if f != anchor],
                                  size=self.S - 1, replace=len(frames) - 1 < self.S - 1)
        sel = [int(anchor)] + [int(o) for o in np.atleast_1d(others)]

        imgs, deps, extrs, intrs, masks, wpts = [], [], [], [], [], []
        for fid in sel:
            try:
                img, depth, extr, intr, drange = self._load_one(scene, fid)
            except FileNotFoundError:
                # broken frame; replace with a different one from the scene
                fid2 = int(self.rng.choice([f for f in frames if f != fid]))
                img, depth, extr, intr, drange = self._load_one(scene, fid2)
            img, depth, intr = self._resize_pack(img, depth, intr, self.img_size)
            dmin, dmax = drange  # official per-frame depth range from cam.txt
            # base mask: positive + finite
            mask = (depth > 0) & np.isfinite(depth)
            # use cam.txt range to drop outliers (sky pixels, behind-camera artifacts)
            if dmax > 0 and np.isfinite(dmax):
                mask = mask & (depth >= 0.5 * dmin) & (depth <= 1.5 * dmax)
            # per-frame outlier filter: drop pixels > 5x median valid depth
            if mask.any():
                med = float(np.median(depth[mask]))
                if med > 0:
                    mask = mask & (depth <= 5.0 * med)
            # zero out invalid depth so loss/world_points never see junk values
            depth = np.where(mask, depth, 0.0).astype(np.float32)
            wp = _depth_to_world_points(depth, extr, intr)
            wp[~mask] = 0.0
            imgs.append(img)
            deps.append(depth)
            extrs.append(extr)
            intrs.append(intr)
            masks.append(mask)
            wpts.append(wp)

        deps_arr = np.stack(deps)
        extrs_arr = np.stack(extrs).astype(np.float32, copy=True)
        wpts_arr = np.stack(wpts)
        masks_arr = np.stack(masks)

        # Per-batch scene-scale normalization. BlendedMVS scenes range from
        # ~0.5m close-ups to ~125m flythroughs, so absolute depth and the
        # camera-translation magnitudes vary by 2-3 orders of magnitude.
        # The pretrained VGGT and DA3 weights, however, were trained on
        # normalized scenes (|T| ~ 1, depth ~ 1). Without rescaling, the L1
        # camera loss directly takes |T_gt|=125 as a 100+ contribution and
        # spikes the total loss. Normalize all spatial quantities by
        # max(median valid depth, max |T|) so every sample lives in unit scale.
        valid_d = deps_arr[masks_arr]
        if valid_d.size > 100:
            d_med = float(np.median(valid_d))
        else:
            d_med = 0.0
        t_max = float(np.linalg.norm(extrs_arr[:, :3, 3], axis=-1).max())
        scene_scale = max(d_med, t_max, 1e-3)
        deps_arr = deps_arr / scene_scale
        extrs_arr[:, :3, 3] /= scene_scale
        wpts_arr = wpts_arr / scene_scale

        return {
            "images": torch.from_numpy(np.stack(imgs)).permute(0, 3, 1, 2).contiguous().float(),
            "depths": torch.from_numpy(deps_arr).float(),
            "extrinsics": torch.from_numpy(extrs_arr).float(),
            "intrinsics": torch.from_numpy(np.stack(intrs)).float(),
            "point_masks": torch.from_numpy(masks_arr).bool(),
            "world_points": torch.from_numpy(wpts_arr).float(),
            "seq_name": os.path.basename(scene),
        }


def collate_fn(batch):
    out = {}
    keys = batch[0].keys()
    for k in keys:
        if k == "seq_name":
            out[k] = [b[k] for b in batch]
        else:
            out[k] = torch.stack([b[k] for b in batch], dim=0)
    return out
