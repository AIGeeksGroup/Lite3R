"""Inference-time real kernels for Lite variants.

The training pipeline keeps the dual-branch SLA (linear + Top-K) and uses
FakeQuantLinear for W4A4 QAT (the methodology side of the paper). At
inference / latency-measurement time we want the **real** efficient kernels:

  1. SLA forward → only the linear branch (drops O(N^2) Top-K and gives the
     promised O(N · d^2) compute curve).
  2. nn.Linear (including former FakeQuantLinear after unwrapping) → torchao
     `int4_weight_only`, which packs weights into 4-bit storage and dispatches
     the matmul to a tinygemm INT4×BF16 CUDA kernel on sm_80+ (A100, A40,
     H100). Activations stay BF16 because the W4A16 kernel is what is
     actually shipped in production INT4 inference stacks today; W4A4
     activations are simulated during QAT but use this kernel at deployment.

Call once from the eval script after `model.to(device).eval()` and BEFORE
`measure_latency` / `measure_max_memory` so the reported numbers reflect the
shipped configuration.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


# --------------------------------------------------------------------------- #
# 1. SLA → linear-only inference forward                                      #
# --------------------------------------------------------------------------- #

def _sla_linear_only_forward(self, x, pos=None, attn_mask=None):
    if attn_mask is not None:
        return self._fallback_sdpa(x, pos=pos, attn_mask=attn_mask)
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    q, k = self.q_norm(q), self.k_norm(k)
    if self.rope is not None and pos is not None:
        q = self.rope(q, pos)
        k = self.rope(k, pos)
    a_lin = self._linear_branch(q, k, v)
    out = a_lin.transpose(1, 2).reshape(B, N, C)
    out = self.proj(out)
    out = self.proj_drop(out)
    return out


def patch_sla_to_linear_only(model: nn.Module) -> int:
    """Bind the linear-only forward onto every SLAAttention. Returns count."""
    from lite3r_kit.sla import SLAAttention
    n = 0
    for m in model.modules():
        if isinstance(m, SLAAttention):
            m.forward = _sla_linear_only_forward.__get__(m, SLAAttention)
            n += 1
    return n


# --------------------------------------------------------------------------- #
# 2. FakeQuantLinear → real nn.Linear (so torchao can wrap it cleanly)        #
# --------------------------------------------------------------------------- #

def _unwrap_fake_quant_linears(model: nn.Module) -> int:
    from lite3r_kit.fake_quant import FakeQuantLinear
    try:
        from lite3r_kit.fp8_fake_quant import FP8FakeQuantLinear
        fake_quant_types = (FakeQuantLinear, FP8FakeQuantLinear)
    except Exception:
        fake_quant_types = (FakeQuantLinear,)
    n = 0
    for name, child in list(model.named_children()):
        if isinstance(child, fake_quant_types):
            new = nn.Linear(child.in_features, child.out_features,
                            bias=(child.bias is not None))
            with torch.no_grad():
                # Fold SmoothQuant scale for W4A* FakeQuantLinear. FP8 fake
                # quant has no smoothing buffer; copying its trained weight is enough.
                if hasattr(child, "smooth"):
                    s = child.smooth.to(child.weight.dtype)
                    new.weight.copy_(child.weight.data * s)
                else:
                    new.weight.copy_(child.weight.data)
                if child.bias is not None:
                    new.bias.copy_(child.bias.data)
            new = new.to(device=child.weight.device, dtype=child.weight.dtype)
            setattr(model, name, new)
            n += 1
        else:
            n += _unwrap_fake_quant_linears(child)
    return n


# --------------------------------------------------------------------------- #
# 3. nn.Linear → torchao INT4 weight-only                                     #
# --------------------------------------------------------------------------- #

def apply_real_inference_kernels(
    model: nn.Module,
    *,
    int4_group_size: int = 128,
    skip_substrings: Tuple[str, ...] = ("rope",),
    quant_mode: str | None = None,
) -> str:
    """Convert a trained Lite model to deployment form.

    Steps performed in order:
      a) (optional) SLAAttention.forward → linear branch only
         Controlled by env var LITE3R_LINEAR_ONLY (default OFF). The previous
         default of ON caused a severe training-inference mismatch: training
         optimised `a_lin + lam*a_topk` but eval saw only `a_lin`, dropping
         the trained lambda and trained top-k contribution and pinning all
         hyperparameter sweeps to the same ~0.285 abs_rel ceiling regardless
         of training. The dual-branch forward is what the SLA paper actually
         specifies; sparsity in the top-k branch (keep_ratio≪1) preserves
         most of the efficiency gain.
      b) FakeQuantLinear → bare nn.Linear (smoothing folded in)
      c) Cast all FP32 params/buffers to BF16 (torchao INT4 kernel requires it)
      d) torchao quantize_(model, int4_weight_only(group_size=...))

    Returns a one-line summary suitable for direct printing.
    """
    import os
    use_linear_only = os.environ.get("LITE3R_LINEAR_ONLY", "0") == "1"
    # INT4 quantization at eval rounds weights to a coarse 4-bit grid (step
    # size ~0.03 per group). For stage1 (BF16-trained), the small movements
    # (~5e-4) get rounded away, masking any training signal — every stage1
    # config produces bit-identical eval. For stage2 (W4A4-QAT-trained),
    # quantization is faithful to deployment. Default ON for stage2 matching
    # the paper's deployment claims; set LITE3R_INT4=0 to skip when measuring
    # actual training-time accuracy on stage1.
    use_int4 = os.environ.get("LITE3R_INT4", "1") == "1"
    # quant_mode controls the deployment kernel:
    #   "int4"          : torchao int4_weight_only            (W4A16 — original)
    #   "int4_int8_act" : int8_dynamic_activation_int4_weight (W4A8 — INT8 tcore)
    #   "int8_int8_act" : int8_dynamic_activation_int8_weight (W8A8 — INT8 tcore)
    if quant_mode is None:
        quant_mode = os.environ.get("LITE3R_QUANT_MODE", "int4")
    if quant_mode not in ("int4", "int4_int8_act", "int8_int8_act",
                          "int8_int8_act_2x4_sparse", "fp8_weight_only"):
        raise ValueError(f"unknown quant_mode={quant_mode!r}")
    n_sla = patch_sla_to_linear_only(model) if use_linear_only else 0
    n_unfaked = _unwrap_fake_quant_linears(model)
    n_int4 = 0
    msg_torchao = ""
    if not use_int4:
        # Still cast to BF16 so latency/MaxMem reflect deployment dtype, but
        # keep weights at full precision — true to the trained model.
        for p in model.parameters():
            if p.dtype == torch.float32:
                p.data = p.data.to(torch.bfloat16)
        for b in model.buffers():
            if b.dtype == torch.float32:
                b.data = b.data.to(torch.bfloat16)
        # Register the same input-cast hook used in the INT4 branch so FP32
        # tensors produced by helpers (positional encodings, etc.) get cast
        # to BF16 before hitting BF16 weights.
        _SENSITIVE = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d,
                      nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d,
                      nn.LayerNorm, nn.GroupNorm)

        def _cast_inputs_to_bf16(module, inputs):
            new = []; changed = False
            for x in inputs:
                if isinstance(x, torch.Tensor) and x.is_floating_point() and x.dtype != torch.bfloat16:
                    new.append(x.to(torch.bfloat16)); changed = True
                else:
                    new.append(x)
            return tuple(new) if changed else inputs

        for m in model.modules():
            if isinstance(m, _SENSITIVE):
                m.register_forward_pre_hook(_cast_inputs_to_bf16)
        return (f"[inference] SLA→linear-only on {n_sla} modules, "
                f"unwrapped {n_unfaked} FakeQuantLinear, INT4 skipped "
                f"(LITE3R_INT4=0)")
    try:
        # torchao 0.10+ migrated factory functions (int4_weight_only etc.) to
        # class-based configs (Int4WeightOnlyConfig). Try new API first, fall
        # back to old. W4A8 (int8 act + int4 weight) is no longer in the
        # Config API in 0.17.0 — only Int8DynActInt4WeightQuantizer remains,
        # which is a separate Quantizer class with .quantize(model) entry.
        from torchao.quantization import quantize_
        try:
            from torchao.quantization import (
                Int4WeightOnlyConfig,
                Int8DynamicActivationInt8WeightConfig,
                Float8WeightOnlyConfig,
            )
            _USE_NEW_API = True
        except ImportError:
            from torchao.quantization import (  # type: ignore[no-redef]
                int4_weight_only as Int4WeightOnlyConfig,
                int8_dynamic_activation_int8_weight as Int8DynamicActivationInt8WeightConfig,
                float8_weight_only as Float8WeightOnlyConfig,
            )
            _USE_NEW_API = False

        # cast to BF16 in-place (torchao tinygemm kernel is BF16 activation).
        # Caller must run forward with autocast disabled and feed BF16 inputs;
        # see eval/eval_{vggt,da3}.py for the inference-context contract.
        for p in model.parameters():
            if p.dtype == torch.float32:
                p.data = p.data.to(torch.bfloat16)
        for b in model.buffers():
            if b.dtype == torch.float32:
                b.data = b.data.to(torch.bfloat16)

        def _filter(m, fqn):
            return (
                isinstance(m, nn.Linear)
                and not any(s in fqn for s in skip_substrings)
                and m.in_features % int4_group_size == 0
            )

        for fqn, m in model.named_modules():
            if _filter(m, fqn):
                n_int4 += 1

        # W4A8 path. New torchao 0.10+ removed `int8_dynamic_activation_int4_weight`
        # from the Config API; only `Int8DynActInt4WeightQuantizer` (a separate
        # Quantizer class) remains. Empirically the Quantizer class breaks our
        # forward_pre_hook contract (model structure mutations interfere with
        # FP32→BF16 input cast on non-Linear layers like ConvTranspose2d), so
        # by default we now FALL BACK to W4A16 via Int4WeightOnlyConfig — same
        # 4-bit weight memory savings, BF16 activations (loses INT8 tensor-core
        # acceleration but gains stability). To force the legacy W4A8 Quantizer
        # path, set env var LITE3R_FORCE_W4A8_QUANTIZER=1 (mostly broken).
        # Wrap each quant attempt: in torchao 0.17 on torch 2.7, INT4 paths
        # require either cpp_extensions (need torch>=2.11) or `mslk` package
        # (private). Both fail in our env. Catch ImportError + RuntimeError
        # and fall through to BF16-weight-only (no quantize), so eval still
        # produces metrics. Paper W4 numbers come from old env (torch 2.3.1).
        def _try_quant(label: str, do_quant):
            try:
                do_quant()
                return label
            except (ImportError, RuntimeError, Exception) as q_err:
                print(f"[inference] {label} FAILED ({type(q_err).__name__}: {q_err}); "
                      f"falling back to BF16 weights (no quantize)", flush=True)
                return f"BF16-weight (skip: {label})"

        if quant_mode == "int4_int8_act":
            import os as _os
            try:
                from torchao.quantization import int8_dynamic_activation_int4_weight  # type: ignore
                ao_cfg = int8_dynamic_activation_int4_weight(group_size=int4_group_size)
                kernel_label = _try_quant(
                    f"W4A8 dyn-act legacy (group={int4_group_size})",
                    lambda: quantize_(model, ao_cfg, filter_fn=_filter)
                )
            except ImportError:
                # New torchao: try Int4WeightOnlyConfig (W4A16 fallback). May
                # need mslk on 0.17.0; if so, gracefully skip to BF16 weights.
                kernel_label = _try_quant(
                    f"W4 (BF16-act fallback) (group={int4_group_size})",
                    lambda: quantize_(model, Int4WeightOnlyConfig(group_size=int4_group_size),
                                       filter_fn=_filter)
                )
        elif quant_mode == "int4":
            kernel_label = _try_quant(
                f"INT4-W-only (group={int4_group_size})",
                lambda: quantize_(model, Int4WeightOnlyConfig(group_size=int4_group_size),
                                   filter_fn=_filter)
            )
        elif quant_mode == "int8_int8_act":
            kernel_label = _try_quant(
                "W8A8 dyn-act",
                lambda: quantize_(model, Int8DynamicActivationInt8WeightConfig(),
                                   filter_fn=_filter)
            )
        elif quant_mode == "fp8_weight_only":
            kernel_label = _try_quant(
                "FP8-W-only",
                lambda: quantize_(model, Float8WeightOnlyConfig(), filter_fn=_filter)
            )
        else:  # int8_int8_act_2x4_sparse
            from torchao.sparsity import int8_dynamic_activation_int8_semi_sparse_weight
            kernel_label = _try_quant(
                "W8A8 + 2:4 semi-sparse",
                lambda: quantize_(model, int8_dynamic_activation_int8_semi_sparse_weight(),
                                   filter_fn=_filter)
            )

        # Safety net (registered AFTER torchao swaps Linear→quantized variants
        # so the hook lands on the surviving modules): some model utilities
        # (e.g. positional-embedding helpers) construct intermediate tensors in
        # FP32 regardless of the requested dtype, which then up-promotes
        # activations back to FP32 and crashes the next BF16-weight kernel.
        # Cast FP32 inputs to BF16 right before each dtype-sensitive layer.
        _SENSITIVE = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d,
                      nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d,
                      nn.LayerNorm, nn.GroupNorm)

        def _cast_inputs_to_bf16(module, inputs):
            new = []
            changed = False
            for x in inputs:
                if isinstance(x, torch.Tensor) and x.is_floating_point() and x.dtype != torch.bfloat16:
                    new.append(x.to(torch.bfloat16)); changed = True
                else:
                    new.append(x)
            return tuple(new) if changed else inputs

        for m in model.modules():
            if isinstance(m, _SENSITIVE):
                m.register_forward_pre_hook(_cast_inputs_to_bf16)
        msg_torchao = f", torchao {kernel_label} on {n_int4} Linear"
    except Exception as e:
        msg_torchao = f", torchao SKIP ({type(e).__name__}: {e})"

    msg = (f"[inference] SLA→linear-only on {n_sla} modules, "
           f"unwrapped {n_unfaked} FakeQuantLinear" + msg_torchao)
    print(msg, flush=True)
    return msg
