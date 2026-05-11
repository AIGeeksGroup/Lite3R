"""lite3r_kit — shared utilities for the Lite3R lightweight 3D-geometry models.

Importable from anywhere on PYTHONPATH. Designed to drop into both VGGT and
DA3 codebases without modifying their package layout.
"""

from .sla import SLAAttention, replace_attention_with_sla
from .fake_quant import FakeQuantLinear, quantize_model_, count_quant_params
from .distillation import (
    AttentionOutputRecorder,
    compute_kd_loss,
    cosine_kd_weight,
)
from .inference import (
    apply_real_inference_kernels,
    patch_sla_to_linear_only,
)

__all__ = [
    "SLAAttention",
    "replace_attention_with_sla",
    "FakeQuantLinear",
    "quantize_model_",
    "count_quant_params",
    "AttentionOutputRecorder",
    "compute_kd_loss",
    "cosine_kd_weight",
    "apply_real_inference_kernels",
    "patch_sla_to_linear_only",
]
