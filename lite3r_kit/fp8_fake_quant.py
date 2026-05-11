"""FP8 fake-quantization modules for QAT experiments.

This module simulates scaled FP8 E4M3 quantization in the forward pass while
using a straight-through estimator for gradients. It is intended for
experiments where the deployment backend is FP8 weight-only, but we want the
training checkpoint to have seen FP8 weight and activation noise.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _scaled_fp8_ste(
    x: torch.Tensor,
    *,
    fp8_dtype: torch.dtype = torch.float8_e4m3fn,
    scale_dim: int | tuple[int, ...] = -1,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Dynamically scale, cast to FP8, dequantize, and apply STE."""
    if not x.is_floating_point():
        return x
    finfo = torch.finfo(fp8_dtype)
    reduce_dim = scale_dim
    amax = x.detach().abs().amax(dim=reduce_dim, keepdim=True).clamp(min=eps)
    scale = amax / float(finfo.max)
    x_scaled = (x / scale).clamp(min=float(finfo.min), max=float(finfo.max))
    x_deq = x_scaled.to(fp8_dtype).to(x.dtype) * scale
    return x + (x_deq - x).detach()


class FP8FakeQuantLinear(nn.Module):
    """Wraps an nn.Linear with scaled FP8 weight/activation fake quant."""

    def __init__(
        self,
        linear: nn.Linear,
        *,
        fp8_dtype: torch.dtype = torch.float8_e4m3fn,
        enable_act_quant: bool = True,
    ) -> None:
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.fp8_dtype = fp8_dtype
        self.enable_act_quant = enable_act_quant
        self.weight = nn.Parameter(linear.weight.detach().clone())
        if linear.bias is not None:
            self.bias = nn.Parameter(linear.bias.detach().clone())
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Per-token activation scaling and per-output-row weight scaling.
        x_q = _scaled_fp8_ste(x, fp8_dtype=self.fp8_dtype, scale_dim=-1) if self.enable_act_quant else x
        w_q = _scaled_fp8_ste(self.weight, fp8_dtype=self.fp8_dtype, scale_dim=-1)
        return F.linear(x_q, w_q, self.bias)


def quantize_model_fp8_(
    module: nn.Module,
    *,
    fp8_dtype: torch.dtype = torch.float8_e4m3fn,
    enable_act_quant: bool = True,
    skip_name_substrings: tuple[str, ...] = (),
) -> int:
    """Recursively replace nn.Linear with FP8FakeQuantLinear in-place."""
    n = 0
    for name, child in list(module.named_children()):
        if any(s in name for s in skip_name_substrings):
            continue
        if isinstance(child, nn.Linear):
            setattr(
                module,
                name,
                FP8FakeQuantLinear(
                    child,
                    fp8_dtype=fp8_dtype,
                    enable_act_quant=enable_act_quant,
                ),
            )
            n += 1
        else:
            n += quantize_model_fp8_(
                child,
                fp8_dtype=fp8_dtype,
                enable_act_quant=enable_act_quant,
                skip_name_substrings=skip_name_substrings,
            )
    return n
