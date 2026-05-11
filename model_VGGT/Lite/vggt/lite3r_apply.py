"""Lite3R: convert a vanilla VGGT model into Stage-1 (SLA) or Stage-2 (W4A4) form.

Stage 1: replace every Attention/MemEffAttention layer with SLAAttention.
Stage 2: also wrap every nn.Linear inside the aggregator + heads with
         FakeQuantLinear (W4A4, group=128). The qkv linear inside SLAAttention
         is NOT wrapped because its activations are routed through Top-K /
         linear branches and quantizing them too aggressively destabilises
         training; users can override via `quantize_attn_linear=True`.
"""
from __future__ import annotations

from typing import Iterable

import torch.nn as nn

from lite3r_kit.sla import SLAAttention, replace_attention_with_sla
from lite3r_kit.fake_quant import quantize_model_, FakeQuantLinear


def apply_sla(model: nn.Module, keep_ratio: float = 0.2, lambda_init: float = 0.5) -> int:
    """Stage 1: swap dense attention blocks for SLAAttention.

    Returns number of replaced modules.
    """
    return replace_attention_with_sla(
        model,
        keep_ratio=keep_ratio,
        lambda_init=lambda_init,
        target_class_names=("Attention", "MemEffAttention"),
    )


def apply_w4a4(
    model: nn.Module,
    group_size: int = 128,
    weight_bits: int = 4,
    act_bits: int = 4,
    quantize_attn_linear: bool = False,
) -> int:
    """Stage 2: wrap nn.Linear with FakeQuantLinear.

    By default the qkv/proj linears inside SLAAttention are skipped (kept in
    BF16) because their activations route through the Top-K branch. Set
    `quantize_attn_linear=True` for a fully W4A4 lite model.
    Returns number of wrapped linears.
    """
    skip = ()
    if not quantize_attn_linear:
        # Walk the model and freeze SLA submodules first by blacklisting their
        # children. We do this by descending into modules that are *not*
        # SLAAttention.
        n = 0
        for name, m in model.named_children():
            if isinstance(m, SLAAttention):
                # quantise nothing inside SLA
                continue
            if isinstance(m, nn.Linear):
                wrapped = FakeQuantLinear(m, group_size=group_size,
                                          weight_bits=weight_bits, act_bits=act_bits)
                setattr(model, name, wrapped)
                n += 1
            else:
                n += apply_w4a4(m, group_size, weight_bits, act_bits,
                                quantize_attn_linear=False)
        return n
    # Even when fully quantizing attention linears, keep the SLA residual
    # projections (`proj_lin`, `proj_topk`) in FP16: they ARE the novelty of
    # the paper's `O = O^s + Proj(O^l)` formulation and are the only modules
    # R2-A QAT trains, so quantizing them would (a) weaken the paper story
    # and (b) round-trip the trainable weights.
    return quantize_model_(
        model,
        group_size=group_size,
        weight_bits=weight_bits,
        act_bits=act_bits,
        enable_act_quant=(act_bits < 16),
        skip_name_substrings=("proj_lin", "proj_topk"),
    )
