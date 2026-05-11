"""Lite3R: convert a vanilla DA3 model into Stage-1 (SLA) or Stage-2 (W4A4) form.

DA3 attention lives inside src/depth_anything_3/model/dinov2/layers/attention.py
(class `Attention`). We replace it the same way as for VGGT.
"""
from __future__ import annotations

import torch.nn as nn

from lite3r_kit.sla import SLAAttention, replace_attention_with_sla
from lite3r_kit.fake_quant import quantize_model_, FakeQuantLinear


def apply_sla(model: nn.Module, keep_ratio: float = 0.2, lambda_init: float = 0.5) -> int:
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
    if not quantize_attn_linear:
        n = 0
        for name, m in model.named_children():
            if isinstance(m, SLAAttention):
                continue
            if isinstance(m, nn.Linear):
                setattr(
                    model,
                    name,
                    FakeQuantLinear(m, group_size=group_size,
                                    weight_bits=weight_bits, act_bits=act_bits),
                )
                n += 1
            else:
                n += apply_w4a4(m, group_size, weight_bits, act_bits,
                                quantize_attn_linear=False)
        return n
    return quantize_model_(
        model,
        group_size=group_size,
        weight_bits=weight_bits,
        act_bits=act_bits,
        enable_act_quant=(act_bits < 16),
    )
