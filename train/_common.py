"""Shared training utilities: config, device, optim, ckpt, DA3→VGGT-loss adapter."""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict

import torch
import torch.nn as nn
import yaml


def add_paths(*paths: str) -> None:
    """Prepend extra paths to sys.path (kept early so internal imports win)."""
    for p in paths:
        p = os.path.abspath(p)
        if p not in sys.path:
            sys.path.insert(0, p)


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def deep_update(d: Dict[str, Any], u: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in u.items():
        if isinstance(v, dict) and isinstance(d.get(k), dict):
            d[k] = deep_update(d[k], v)
        else:
            d[k] = v
    return d


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def autocast_dtype(device: torch.device) -> torch.dtype:
    if device.type != "cuda":
        return torch.float32
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def build_optimizer(model: nn.Module, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim <= 1 or n.endswith(".bias") or "lam" in n or "smooth" in n or "proj_lin" in n or "proj_topk" in n:
            no_decay.append(p)
        else:
            decay.append(p)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=lr,
        betas=(0.9, 0.999),
    )


def build_scheduler(optim: torch.optim.Optimizer, total_steps: int, warmup_steps: int = 0):
    def lr_lambda(step):
        if step < warmup_steps and warmup_steps > 0:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = max(0.0, min(1.0, progress))
        return 0.5 * (1.0 + (1 - progress) * (1 + 0))  # linear-ish to 0.5
    return torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_lambda)


def save_ckpt(path: str, model: nn.Module, optim=None, step: int = 0, extra: Dict[str, Any] | None = None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "optim": optim.state_dict() if optim is not None else None,
        "step": step,
        "extra": extra or {},
    }
    torch.save(payload, path)


def load_ckpt(path: str, model: nn.Module, optim=None, strict: bool = False) -> Dict[str, Any]:
    payload = torch.load(path, map_location="cpu")
    msg = model.load_state_dict(payload["model"], strict=strict)
    if optim is not None and payload.get("optim") is not None:
        try:
            optim.load_state_dict(payload["optim"])
        except Exception as e:
            print(f"[load_ckpt] optim load skipped: {e}")
    return payload


def adapt_da3_output_for_vggt_loss(out: Any, image_hw, pose_enc_fn) -> Dict[str, torch.Tensor]:
    """Convert DA3 net output (addict.Dict) into the dict expected by VGGT's
    MultitaskLoss.

    Args:
        out: DA3 forward output with `.depth (B,S,H,W)`, `.depth_conf (B,S,H,W)`,
             `.extrinsics (B,S,3,4|4,4)`, `.intrinsics (B,S,3,3)`.
        image_hw: (H, W) tuple.
        pose_enc_fn: VGGT's `extri_intri_to_pose_encoding` callable.

    Returns:
        Dict suitable for VGGT loss with `pose_enc_list`, `depth`, `depth_conf`.
    """
    extr = out.extrinsics
    if extr.shape[-2] == 3:
        # promote (B,S,3,4) → (B,S,4,4)
        B, S = extr.shape[:2]
        eye = torch.tensor([[0, 0, 0, 1.0]], device=extr.device, dtype=extr.dtype).expand(B, S, 1, 4)
        extr = torch.cat([extr, eye], dim=-2)
    pose_enc = pose_enc_fn(extr, out.intrinsics, image_hw, "absT_quaR_FoV")
    depth = out.depth
    if depth.dim() == 4:  # (B,S,H,W)
        depth = depth.unsqueeze(-1)
    return {
        "pose_enc_list": [pose_enc],
        "pose_enc": pose_enc,
        "depth": depth,
        "depth_conf": out.depth_conf,
    }


@dataclass
class TrainState:
    step: int = 0
    epoch: int = 0
    best_loss: float = float("inf")
    history: list = field(default_factory=list)
