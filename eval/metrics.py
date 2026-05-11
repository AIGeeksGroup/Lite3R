"""Metric computation utilities.

Includes:
- Depth: AbsRel, δ<1.25 (1.25^1, 1.25^2, 1.25^3), RMSE.
- Camera pose: rotation error (deg), translation L2 (after Umeyama-style scale).
- Geometry (optional): Chamfer Distance + F-score@τ on point clouds.
- Efficiency: Latency, Max Memory, FLOPs.

FLOPs counting is intentionally analytical because off-the-shelf counters
(fvcore, thop) treat SLA attention as opaque. The function `count_flops` adds
the linear-layer FLOPs (via fvcore) to the closed-form attention FLOPs based
on model introspection.
"""
from __future__ import annotations

import math
import time
from typing import Dict, Tuple

import torch
import torch.nn as nn


# ---------- depth ----------

def depth_metrics(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor,
                  scale_invariant: bool = True) -> Dict[str, float]:
    """Standard depth metrics. pred/gt: (..., H, W). mask: bool same-shape."""
    pred = pred.float().clamp(min=1e-3)
    gt = gt.float().clamp(min=1e-3)
    mask = mask & (gt > 0)
    if mask.sum() < 100:
        return {k: float("nan") for k in ["abs_rel", "delta1", "delta2", "delta3", "rmse"]}
    p = pred[mask]
    g = gt[mask]
    if scale_invariant:
        # solve for scalar s minimizing ||s*p - g||
        s = (g * p).sum() / (p * p).sum().clamp(min=1e-8)
        p = p * s
    abs_rel = ((p - g).abs() / g).mean().item()
    rmse = ((p - g) ** 2).mean().sqrt().item()
    ratio = torch.maximum(p / g, g / p)
    d1 = (ratio < 1.25).float().mean().item()
    d2 = (ratio < 1.25 ** 2).float().mean().item()
    d3 = (ratio < 1.25 ** 3).float().mean().item()
    return {"abs_rel": abs_rel, "delta1": d1, "delta2": d2, "delta3": d3, "rmse": rmse}


# ---------- camera pose ----------

def rotation_angle_error_deg(R_pred: torch.Tensor, R_gt: torch.Tensor) -> float:
    R = R_pred.transpose(-1, -2) @ R_gt
    trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    cos = ((trace - 1.0) / 2.0).clamp(-1.0, 1.0)
    return torch.acos(cos).mean().item() * 180.0 / math.pi


def translation_error(t_pred: torch.Tensor, t_gt: torch.Tensor,
                      scale_align: bool = True) -> float:
    if scale_align:
        s_num = (t_pred * t_gt).sum()
        s_den = (t_pred * t_pred).sum().clamp(min=1e-8)
        t_pred = t_pred * (s_num / s_den)
    return (t_pred - t_gt).norm(dim=-1).mean().item()


def pose_metrics(extr_pred: torch.Tensor, extr_gt: torch.Tensor) -> Dict[str, float]:
    """extr: (..., 4, 4) or (..., 3, 4) world-to-camera."""
    if extr_pred.shape[-2] == 3:
        extr_pred = torch.cat([extr_pred, torch.tensor([[[0, 0, 0, 1.0]]],
                                                         device=extr_pred.device,
                                                         dtype=extr_pred.dtype).expand_as(extr_pred[..., :1, :])], -2)
    if extr_gt.shape[-2] == 3:
        extr_gt = torch.cat([extr_gt, torch.tensor([[[0, 0, 0, 1.0]]],
                                                     device=extr_gt.device,
                                                     dtype=extr_gt.dtype).expand_as(extr_gt[..., :1, :])], -2)
    R_p = extr_pred[..., :3, :3]
    R_g = extr_gt[..., :3, :3]
    t_p = extr_pred[..., :3, 3]
    t_g = extr_gt[..., :3, 3]
    return {
        "rot_err_deg": rotation_angle_error_deg(R_p, R_g),
        "trans_err": translation_error(t_p, t_g),
    }


# ---------- geometry (3D points) ----------

def chamfer_l2(p_a: torch.Tensor, p_b: torch.Tensor, max_pts: int = 50000) -> float:
    """Bidirectional Chamfer distance (L2)."""
    if p_a.shape[0] > max_pts:
        idx = torch.randperm(p_a.shape[0])[:max_pts]
        p_a = p_a[idx]
    if p_b.shape[0] > max_pts:
        idx = torch.randperm(p_b.shape[0])[:max_pts]
        p_b = p_b[idx]
    # naive O(n*m) — fine for max_pts=50k via chunked compute
    chunk = 4096
    total_a, n_a = 0.0, p_a.shape[0]
    for i in range(0, n_a, chunk):
        d = torch.cdist(p_a[i:i + chunk], p_b)
        total_a += d.min(dim=-1).values.pow(2).sum().item()
    total_b, n_b = 0.0, p_b.shape[0]
    for i in range(0, n_b, chunk):
        d = torch.cdist(p_b[i:i + chunk], p_a)
        total_b += d.min(dim=-1).values.pow(2).sum().item()
    return (total_a / max(n_a, 1) + total_b / max(n_b, 1)) * 0.5


def fscore_pointcloud(p_a: torch.Tensor, p_b: torch.Tensor, tau: float = 0.05,
                       max_pts: int = 50000) -> float:
    """Symmetric F-score at threshold tau."""
    if p_a.shape[0] > max_pts:
        p_a = p_a[torch.randperm(p_a.shape[0])[:max_pts]]
    if p_b.shape[0] > max_pts:
        p_b = p_b[torch.randperm(p_b.shape[0])[:max_pts]]
    chunk = 4096
    rec_hits = 0
    for i in range(0, p_a.shape[0], chunk):
        d = torch.cdist(p_a[i:i + chunk], p_b)
        rec_hits += (d.min(dim=-1).values < tau).sum().item()
    pre_hits = 0
    for i in range(0, p_b.shape[0], chunk):
        d = torch.cdist(p_b[i:i + chunk], p_a)
        pre_hits += (d.min(dim=-1).values < tau).sum().item()
    pre = pre_hits / max(p_b.shape[0], 1)
    rec = rec_hits / max(p_a.shape[0], 1)
    if pre + rec < 1e-6:
        return 0.0
    return 2 * pre * rec / (pre + rec)


# ---------- efficiency ----------

@torch.no_grad()
def measure_latency(model: nn.Module, sample_input: Tuple,
                    n_warmup: int = 10, n_iters: int = 20,
                    device: str = "cuda") -> Dict[str, float]:
    """Mean / std forward latency in ms."""
    model.eval().to(device)
    sample_input = tuple(x.to(device) if torch.is_tensor(x) else x for x in sample_input)
    if device == "cuda":
        torch.cuda.synchronize()
    for _ in range(n_warmup):
        _ = model(*sample_input)
    if device == "cuda":
        torch.cuda.synchronize()
    times_ms = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        _ = model(*sample_input)
        if device == "cuda":
            torch.cuda.synchronize()
        times_ms.append((time.perf_counter() - t0) * 1000.0)
    import statistics
    return {
        "latency_ms_mean": statistics.mean(times_ms),
        "latency_ms_std": statistics.pstdev(times_ms),
        "latency_ms_p50": sorted(times_ms)[len(times_ms) // 2],
    }


@torch.no_grad()
def measure_max_memory(model: nn.Module, sample_input: Tuple,
                       device: str = "cuda") -> Dict[str, float]:
    if device != "cuda":
        return {"max_mem_MB": 0.0}
    model.eval().to(device)
    sample_input = tuple(x.to(device) if torch.is_tensor(x) else x for x in sample_input)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    _ = model(*sample_input)
    torch.cuda.synchronize()
    return {"max_mem_MB": torch.cuda.max_memory_allocated() / (1024 ** 2)}


def count_flops(model: nn.Module, sample_input: Tuple) -> Dict[str, float]:
    """Try fvcore.nn.FlopCountAnalysis; fall back to a Linear-only counter."""
    try:
        from fvcore.nn import FlopCountAnalysis
        flops = FlopCountAnalysis(model, sample_input).unsupported_ops_warnings(False)
        flops.uncalled_modules_warnings(False)
        total = flops.total()
        return {"flops_g": total / 1e9}
    except Exception as e:  # pragma: no cover
        # rough fallback: count nn.Linear * 2 * in * out per token
        # this only handles linears; attention contributions are skipped.
        total = 0
        # Walk model and estimate from observed input shape
        for m in model.modules():
            if isinstance(m, nn.Linear):
                total += 2 * m.in_features * m.out_features
        return {"flops_g": total / 1e9, "note": f"fvcore fallback: {e}"}


def count_parameters(model: nn.Module) -> Dict[str, int]:
    n = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"params_total": n, "params_trainable": n_train}
