#!/usr/bin/env python3
"""Lite3R inference entry point.

Loads a trained Lite3R checkpoint (VGGT or DA3-L variant) and runs
feed-forward 3D reconstruction on a directory of multi-view images,
exporting the predicted point cloud as a PLY file.

Usage:
    python inference.py \\
        --model vggt \\
        --checkpoint checkpoints/fp8_qat_1ep/vggt/vggt_fp8_qat_1ep.pt \\
        --input_dir examples/input \\
        --output examples/output/reconstruction.ply
"""
from __future__ import annotations

import argparse
import glob
import os
import struct
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Lite3R feed-forward 3D reconstruction inference.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--model", choices=["vggt", "da3"], required=True,
                   help="Backbone family: vggt or da3 (Depth Anything V3).")
    p.add_argument("--checkpoint", required=True, type=Path,
                   help="Path to trained Lite3R checkpoint (.pt).")
    p.add_argument("--input_dir", required=True, type=Path,
                   help="Directory containing input images (jpg/png).")
    p.add_argument("--output", required=True, type=Path,
                   help="Output PLY file path.")
    p.add_argument("--num_views", type=int, default=8,
                   help="Number of views to use for reconstruction (default: 8).")
    p.add_argument("--img_size", type=int, default=518,
                   help="Input image resolution (default: 518).")
    p.add_argument("--conf_threshold", type=float, default=2.0,
                   help="Depth confidence threshold for point filtering (default: 2.0).")
    p.add_argument("--device", default="cuda",
                   help="Device for inference (default: cuda).")
    return p.parse_args()


def load_images(input_dir: Path, num_views: int, img_size: int) -> torch.Tensor:
    """Load and preprocess input images."""
    patterns = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG"]
    image_files = []
    for pat in patterns:
        image_files.extend(sorted(glob.glob(str(input_dir / pat))))
    image_files = sorted(set(image_files))[:num_views]

    if not image_files:
        raise FileNotFoundError(f"No images found in {input_dir}")

    print(f"Loading {len(image_files)} images from {input_dir}")
    images = []
    for f in image_files:
        img = Image.open(f).convert("RGB")
        arr = np.array(img).transpose(2, 0, 1) / 255.0
        images.append(arr)

    images = torch.from_numpy(np.stack(images)).float()
    images = F.interpolate(images, size=(img_size, img_size),
                           mode="bilinear", align_corners=False)
    return images


def build_vggt_model(checkpoint_path: Path, device: str) -> torch.nn.Module:
    """Build VGGT Lite3R model and load checkpoint."""
    sys.path.insert(0, str(PROJECT_ROOT / "model_VGGT" / "Lite"))
    from vggt.models.vggt import VGGT
    from vggt.lite3r_apply import apply_sla

    model = VGGT(img_size=518, patch_size=14, embed_dim=1024,
                 enable_camera=True, enable_depth=True,
                 enable_point=False, enable_track=False)
    apply_sla(model, keep_ratio=0.3, lambda_init=0.5)

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state, strict=False)
    model = model.to(device).eval()

    # Switch to deployment-time kernels: linear-only SLA + INT4 weight-only,
    # mirroring eval/eval_vggt.py. Required to avoid the O(N^2) Top-K
    # branch's >20 GiB activation memory at inference.
    from lite3r_kit import apply_real_inference_kernels
    apply_real_inference_kernels(model)
    return model


def build_da3_model(checkpoint_path: Path, device: str) -> torch.nn.Module:
    """Build DA3-L Lite3R model and load checkpoint."""
    da3_src = PROJECT_ROOT / "model_DA3-Large" / "Lite" / "src"
    sys.path.insert(0, str(da3_src))
    from depth_anything_3.cfg import create_object, load_config
    from depth_anything_3.registry import MODEL_REGISTRY
    from depth_anything_3.lite3r_apply import apply_sla

    model = create_object(load_config(MODEL_REGISTRY["da3-large"]))
    apply_sla(model, keep_ratio=0.2, lambda_init=0.5)

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state, strict=False)
    model = model.to(device).eval()

    from lite3r_kit import apply_real_inference_kernels
    apply_real_inference_kernels(model)
    return model


def run_vggt_inference(model, images, device, conf_threshold):
    """Run VGGT inference and produce a fused multi-view point cloud."""
    sys.path.insert(0, str(PROJECT_ROOT / "model_VGGT" / "Lite"))
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    from vggt.utils.geometry import unproject_depth_map_to_point_map

    images = images.to(device).to(torch.bfloat16)
    with torch.no_grad():
        batch = images[None]
        tokens, ps_idx = model.aggregator(batch)
        pose_enc = model.camera_head(tokens)[-1]
        extr, intr = pose_encoding_to_extri_intri(pose_enc, batch.shape[-2:])
        depth, depth_conf = model.depth_head(tokens, batch, ps_idx)

    depth = depth.squeeze(0).float().cpu()
    depth_conf = depth_conf.squeeze(0).float().cpu()
    extr = extr.squeeze(0).float().cpu()
    intr = intr.squeeze(0).float().cpu()
    images_cpu = images.float().cpu()

    all_points, all_colors = [], []
    for v in range(depth.shape[0]):
        pts = unproject_depth_map_to_point_map(
            depth[v:v + 1], extr[v:v + 1], intr[v:v + 1]
        )[0].reshape(-1, 3)
        conf = depth_conf[v].flatten().numpy()
        valid = conf > conf_threshold
        pts = pts[valid]
        cols = images_cpu[v].numpy().reshape(3, -1).T[valid]
        all_points.append(pts)
        all_colors.append(cols)

    points = np.vstack(all_points)
    colors = (np.vstack(all_colors) * 255).astype(np.uint8)
    return points, colors


def run_da3_inference(model, images, device, conf_threshold):
    """Run DA3-L inference and produce a fused multi-view point cloud."""
    images = images.to(device).to(torch.bfloat16)
    with torch.no_grad():
        out = model(images[None])

    # DA3 returns an addict.Dict; depth shape (B, S, H, W).
    depth = out.depth.squeeze(0).float().cpu()           # (S, H, W)
    extr = out.extrinsics.squeeze(0).float().cpu()       # (S, 3, 4) or (S, 4, 4)
    intr = out.intrinsics.squeeze(0).float().cpu()       # (S, 3, 3)
    images_cpu = images.float().cpu()                    # (S, 3, H, W)

    S, H, W = depth.shape

    # Build per-view camera-frame points via pinhole unprojection.
    yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    all_points, all_colors = [], []
    for v in range(S):
        fx, fy = intr[v, 0, 0], intr[v, 1, 1]
        cx, cy = intr[v, 0, 2], intr[v, 1, 2]
        z = depth[v]
        x_cam = (xx - cx) / fx * z
        y_cam = (yy - cy) / fy * z
        pts_cam = torch.stack([x_cam, y_cam, z], dim=-1).reshape(-1, 3)

        # Transform to view-0 camera frame using DA3's view-0-relative w2c.
        E = extr[v]
        if E.shape[0] == 3:
            E = torch.cat([E, torch.tensor([[0, 0, 0, 1.0]])], dim=0)
        try:
            R = torch.linalg.inv(E)[:3, :3]
            t = torch.linalg.inv(E)[:3, 3]
        except RuntimeError:
            R = torch.linalg.pinv(E)[:3, :3]
            t = torch.linalg.pinv(E)[:3, 3]
        pts_world = (pts_cam @ R.t() + t).numpy()

        cols = images_cpu[v].numpy().reshape(3, -1).T
        all_points.append(pts_world)
        all_colors.append(cols)

    points = np.vstack(all_points)
    colors = (np.vstack(all_colors) * 255).astype(np.uint8)
    return points, colors


def save_ply(points: np.ndarray, colors: np.ndarray, path: Path) -> None:
    """Save coloured point cloud as a binary PLY file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        header = (
            "ply\n"
            "format binary_little_endian 1.0\n"
            f"element vertex {len(points)}\n"
            "property float x\nproperty float y\nproperty float z\n"
            "property uchar red\nproperty uchar green\nproperty uchar blue\n"
            "end_header\n"
        )
        f.write(header.encode("ascii"))
        for (x, y, z), (r, g, b) in zip(points, colors):
            f.write(struct.pack("<fffBBB", x, y, z, int(r), int(g), int(b)))


def main() -> None:
    args = parse_args()

    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    images = load_images(args.input_dir, args.num_views, args.img_size)

    print(f"Building {args.model.upper()} Lite3R model...")
    if args.model == "vggt":
        model = build_vggt_model(args.checkpoint, args.device)
        points, colors = run_vggt_inference(model, images, args.device,
                                            args.conf_threshold)
    else:
        model = build_da3_model(args.checkpoint, args.device)
        points, colors = run_da3_inference(model, images, args.device,
                                           args.conf_threshold)

    print(f"Saving point cloud ({len(points)} points) to {args.output}")
    save_ply(points, colors, args.output)
    print("Done.")


if __name__ == "__main__":
    main()
