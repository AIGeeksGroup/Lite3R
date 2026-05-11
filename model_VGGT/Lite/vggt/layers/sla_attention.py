"""Lite3R: Sparse Linear Attention shim for VGGT.

Re-exports SLAAttention from the project-wide lite3r_kit. Kept here so that
imports inside the VGGT package look local (`from vggt.layers.sla_attention
import SLAAttention`).
"""
from lite3r_kit.sla import SLAAttention, replace_attention_with_sla

__all__ = ["SLAAttention", "replace_attention_with_sla"]
