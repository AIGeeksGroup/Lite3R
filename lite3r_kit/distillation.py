"""Knowledge distillation utilities for SLA training.

Records the post-attention output tensor of every Attention/SLAAttention layer
in a model via forward hooks, then computes per-layer MSE between teacher and
student outputs.

Usage:
    teacher_recorder = AttentionOutputRecorder(teacher_model)
    student_recorder = AttentionOutputRecorder(student_model)
    ...
    with torch.no_grad():
        _ = teacher_model(images)
    student_out = student_model(images)
    loss_kd = compute_kd_loss(teacher_recorder, student_recorder)
"""

from __future__ import annotations

import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


_ATTENTION_CLASS_NAMES = ("Attention", "MemEffAttention", "SLAAttention")


class AttentionOutputRecorder:
    """Registers forward hooks on every Attention-like submodule and stores
    their outputs for later KD loss computation."""

    def __init__(self, model: nn.Module, class_names: tuple[str, ...] = _ATTENTION_CLASS_NAMES):
        self.outputs: List[torch.Tensor] = []
        self._handles = []
        for m in model.modules():
            if m.__class__.__name__ in class_names:
                self._handles.append(m.register_forward_hook(self._make_hook()))

    def _make_hook(self):
        def _hook(_mod, _inp, out):
            if isinstance(out, torch.Tensor):
                self.outputs.append(out)
        return _hook

    def reset(self) -> None:
        self.outputs.clear()

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()


def compute_kd_loss(teacher_rec: AttentionOutputRecorder,
                    student_rec: AttentionOutputRecorder,
                    reduction: str = "mean") -> torch.Tensor:
    """Per-layer MSE loss between teacher and student attention outputs.

    Layers that are missing from one side are skipped. Outputs are flattened
    over (batch, tokens, features) before the MSE so different sequence lengths
    will raise (the architectures must match between teacher and student).
    """
    t_outs = teacher_rec.outputs
    s_outs = student_rec.outputs
    n = min(len(t_outs), len(s_outs))
    if n == 0:
        device = next(iter(s_outs), torch.zeros(1)).device if s_outs else torch.device("cpu")
        return torch.zeros((), device=device)

    losses = []
    for t, s in zip(t_outs[:n], s_outs[:n]):
        # cast to common dtype
        t = t.detach().to(s.dtype)
        # broadcast/skip if shapes mismatch
        if t.shape != s.shape:
            continue
        losses.append(F.mse_loss(s, t, reduction=reduction))

    if not losses:
        return torch.zeros((), device=s_outs[0].device)
    return torch.stack(losses).mean()


def cosine_kd_weight(step: int, total_steps: int, gamma_max: float = 0.5,
                     gamma_min: float = 0.05) -> float:
    """Cosine decay from gamma_max to gamma_min over `total_steps`."""
    if total_steps <= 0:
        return gamma_max
    t = min(max(step / total_steps, 0.0), 1.0)
    return gamma_min + 0.5 * (gamma_max - gamma_min) * (1.0 + math.cos(math.pi * t))
