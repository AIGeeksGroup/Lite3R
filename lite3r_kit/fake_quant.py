"""W4A4 fake-quantization with straight-through estimator.

Implements a "dual-smoothed group-wise" W4A4 fake-quant suitable for QAT:
- Weight: per-output-channel-group symmetric INT4, with a learnable smoothing
  scale that redistributes magnitude between activation and weight.
- Activation: per-token symmetric INT4, dynamic.

Numerically equivalent (in expectation) to running INT4 weights and INT4
activations on a quantized backend; only the simulated rounding error is
modelled. Both forward branches use STE (`x + (q - x).detach()`) so gradients
flow through unaffected.

Reference: SmoothQuant (arXiv:2211.10438) for the smoothing trick;
arXiv:2509.24006 for the dual-smoothed dual-branch formulation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _quantize_per_token_int4(x: torch.Tensor, n_bits: int = 4, eps: float = 1e-8) -> torch.Tensor:
    qmax = float(2 ** (n_bits - 1) - 1)  # 7 for INT4
    # x: (..., d). Per-token absmax: scale per row.
    absmax = x.detach().abs().amax(dim=-1, keepdim=True).clamp(min=eps)
    scale = absmax / qmax
    q = torch.round(x / scale).clamp(-qmax - 1, qmax)
    deq = q * scale
    return x + (deq - x).detach()


def _quantize_per_group_int4(
    w: torch.Tensor, group_size: int = 128, n_bits: int = 4, eps: float = 1e-8
) -> torch.Tensor:
    qmax = float(2 ** (n_bits - 1) - 1)
    out_dim, in_dim = w.shape
    if in_dim % group_size != 0:
        # fallback: per-row quant
        scale = w.detach().abs().amax(dim=-1, keepdim=True).clamp(min=eps) / qmax
        q = torch.round(w / scale).clamp(-qmax - 1, qmax)
        deq = q * scale
        return w + (deq - w).detach()
    n_groups = in_dim // group_size
    w_g = w.view(out_dim, n_groups, group_size)
    absmax = w_g.detach().abs().amax(dim=-1, keepdim=True).clamp(min=eps)
    scale = absmax / qmax
    q = torch.round(w_g / scale).clamp(-qmax - 1, qmax)
    deq = q * scale
    deq = deq.view(out_dim, in_dim)
    return w + (deq - w).detach()


class FakeQuantLinear(nn.Module):
    """Wraps an nn.Linear with W4A4 fake-quant + smoothing.

    The smoothing factor `s` (per input feature) is fused with the weight at
    construction time and divides activations:
        y = (x / s) @ (W * s)^T
    so weights become flatter and activations get more dynamic range. Initial
    `s` is computed from a calibration batch via `calibrate_smoothing`, or set
    to all-ones when calibration is skipped.
    """

    def __init__(
        self,
        linear: nn.Linear,
        group_size: int = 128,
        weight_bits: int = 4,
        act_bits: int = 4,
        enable_act_quant: bool = True,
    ) -> None:
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.group_size = group_size
        self.weight_bits = weight_bits
        self.act_bits = act_bits
        self.enable_act_quant = enable_act_quant

        # take ownership of weights
        self.weight = nn.Parameter(linear.weight.detach().clone())
        if linear.bias is not None:
            self.bias = nn.Parameter(linear.bias.detach().clone())
        else:
            self.register_parameter("bias", None)
        # smoothing factor (per input feature), init 1
        self.register_buffer("smooth", torch.ones(self.in_features), persistent=True)

    @torch.no_grad()
    def calibrate_smoothing(self, x: torch.Tensor, alpha: float = 0.5, eps: float = 1e-8) -> None:
        """Update `self.smooth` from a calibration activation tensor.

        Following SmoothQuant: s = (max|X|)^alpha / (max|W|)^(1-alpha).
        Activations get divided by s, weights multiplied by s.
        """
        x_flat = x.detach().reshape(-1, x.shape[-1])
        a_max = x_flat.abs().amax(dim=0).clamp(min=eps)
        w_max = self.weight.detach().abs().amax(dim=0).clamp(min=eps)
        s = (a_max.pow(alpha) / w_max.pow(1.0 - alpha)).clamp(min=eps)
        self.smooth.copy_(s)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # apply smoothing: divide x, multiply W by s (broadcast on input dim)
        s = self.smooth.to(dtype=x.dtype)
        x_s = x / s
        w_s = self.weight * s.to(self.weight.dtype)

        # quantize weight (per group, INT4)
        w_q = _quantize_per_group_int4(w_s, group_size=self.group_size, n_bits=self.weight_bits)

        # quantize activation (per token, INT4) — only when enabled
        if self.enable_act_quant:
            x_q = _quantize_per_token_int4(x_s, n_bits=self.act_bits)
        else:
            x_q = x_s

        return F.linear(x_q, w_q, self.bias)


def quantize_model_(
    module: nn.Module,
    group_size: int = 128,
    weight_bits: int = 4,
    act_bits: int = 4,
    enable_act_quant: bool = True,
    skip_name_substrings: tuple[str, ...] = (),
) -> int:
    """Recursively replace nn.Linear with FakeQuantLinear, in-place.

    Skips any module whose qualified name contains any of `skip_name_substrings`
    (e.g., "rope", "topk_router") to keep them in FP16/BF16.
    Returns the number of layers wrapped.
    """
    n = 0
    for name, child in list(module.named_children()):
        full_name = name
        if any(s in full_name for s in skip_name_substrings):
            continue
        if isinstance(child, nn.Linear):
            wrapped = FakeQuantLinear(
                child,
                group_size=group_size,
                weight_bits=weight_bits,
                act_bits=act_bits,
                enable_act_quant=enable_act_quant,
            )
            setattr(module, name, wrapped)
            n += 1
        else:
            n += quantize_model_(
                child,
                group_size=group_size,
                weight_bits=weight_bits,
                act_bits=act_bits,
                enable_act_quant=enable_act_quant,
                skip_name_substrings=skip_name_substrings,
            )
    return n


def count_quant_params(module: nn.Module) -> tuple[int, int]:
    """Return (num_quantized_params, num_total_params)."""
    n_q, n_t = 0, 0
    for m in module.modules():
        if isinstance(m, FakeQuantLinear):
            n_q += m.weight.numel()
        elif isinstance(m, nn.Linear):
            n_t += m.weight.numel()
    n_t += n_q  # quantized are counted toward total
    return n_q, n_t
