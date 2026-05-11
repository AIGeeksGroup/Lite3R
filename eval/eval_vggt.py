#!/usr/bin/env python3
"""Evaluate a trained VGGT (Original or Lite) checkpoint.

Computes depth + pose + geometry (Chamfer / F-score) metrics on a BlendedMVS
val split, plus efficiency metrics (Latency / MaxMem / FLOPs).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from train._common import add_paths, autocast_dtype, get_device, load_ckpt, load_yaml
from eval.eval_common import (
    build_eval_loader, aggregate, write_report,
)
from eval.metrics import (
    depth_metrics, pose_metrics, chamfer_l2, fscore_pointcloud,
    measure_latency, measure_max_memory, count_flops, count_parameters,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--name", required=True, help="report name (e.g., vggt_original)")
    return p.parse_args()


def build_vggt(variant: str, vggt_root: Path, model_cfg: dict):
    add_paths(str(vggt_root), str(vggt_root / "training"))
    from vggt.models.vggt import VGGT
    model = VGGT(
        img_size=model_cfg.get("img_size", 518),
        patch_size=model_cfg.get("patch_size", 14),
        embed_dim=model_cfg.get("embed_dim", 1024),
        enable_camera=True, enable_depth=True, enable_point=False, enable_track=False,
    )
    if variant in ("lite_stage1", "lite_stage2") and model_cfg.get("enable_sla", True):
        from vggt.lite3r_apply import apply_sla
        apply_sla(model, keep_ratio=model_cfg.get("keep_ratio", 0.2),
                  lambda_init=model_cfg.get("lambda_init", 0.5))
    quant_format = str(model_cfg.get("quant_format", "")).lower()
    if variant == "lite_stage2" and quant_format == "fp8":
        from lite3r_kit.fp8_fake_quant import quantize_model_fp8_
        quantize_model_fp8_(
            model,
            enable_act_quant=model_cfg.get("fp8_act_quant", True),
            skip_name_substrings=tuple(model_cfg.get("fp8_skip_name_substrings", ())),
        )
    elif variant == "lite_stage2" and quant_format in ("none", "off", "no", "fp32", "bf16"):
        pass
    elif variant == "lite_stage2":
        from vggt.lite3r_apply import apply_w4a4
        apply_w4a4(model, group_size=model_cfg.get("group_size", 128),
                   weight_bits=model_cfg.get("weight_bits", 4),
                   act_bits=model_cfg.get("act_bits", 4),
                   quantize_attn_linear=model_cfg.get("quantize_attn_linear", False))
    return model


def vggt_extr_from_pose(pose_enc, image_hw):
    add_paths(str(PROJECT_ROOT / "model_VGGT" / "Original"))
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    extr, intr = pose_encoding_to_extri_intri(pose_enc, image_hw)
    return extr, intr


def main():
    args = parse_args()
    cfg = load_yaml(args.config)
    variant = cfg.get("variant", "original")
    out_dir = Path(cfg.get("output_dir", "outputs/eval_vggt"))

    device = get_device()
    amp_dtype = autocast_dtype(device)

    vggt_root = PROJECT_ROOT / "model_VGGT" / ("Lite" if variant != "original" else "Original")
    model = build_vggt(variant, vggt_root, cfg.get("model", {}))
    payload = load_ckpt(args.ckpt, model, optim=None, strict=False)
    print(f"[eval-vggt] loaded ckpt step={payload.get('step')}")
    model.to(device).eval()
    # Lite variants → switch to real inference kernels (linear-only SLA +
    # torchao INT4 weight-only). This is what the paper's latency / max-mem
    # numbers MUST be measured against; otherwise we are reporting QAT-side
    # FakeQuantLinear FLOPs which deliver no real wins.
    is_lite = variant in ("lite_stage1", "lite_stage2")
    # Diagnostic: LITE3R_FP32_EVAL=1 evaluates the lite ckpt as if it were
    # dense (autocast-BF16, FP32 weight storage, no INT4). Used to test whether
    # the deployment-time BF16 weight cast erases small backbone updates that
    # training did make. Real numbers reported in the paper still come from
    # deploy_kernels=True.
    import os as _os
    fp32_eval = _os.environ.get("LITE3R_FP32_EVAL", "0") == "1"
    deploy_kernels = is_lite and not fp32_eval
    if deploy_kernels:
        from lite3r_kit import apply_real_inference_kernels
        # quant_mode resolution order: env var > yaml inference.quant_mode > None (defaults to "int4" inside).
        qmode = _os.environ.get("LITE3R_QUANT_MODE",
                                cfg.get("inference", {}).get("quant_mode"))
        apply_real_inference_kernels(model, quant_mode=qmode)
        # Optional torch.compile wrapping. mode=reduce-overhead minimizes kernel
        # launch latency at small batch (best for our N=1369 batch=1 case).
        # fullgraph=False since torchao quantized linears may introduce graph
        # breaks; we accept partial compilation rather than crashing.
        if _os.environ.get("LITE3R_COMPILE", "0") == "1":
            # torchao's LinearActivationQuantizedTensor is not yet trace-friendly
            # under dynamo (torch 2.3 + torchao 0.7). suppress_errors lets dynamo
            # graph-break at quantized linears and fall back to eager there,
            # while still compiling the surrounding non-quantized ops (attn
            # math, layer norms, residual adds, head MLPs that are FP16).
            import torch._dynamo as _dynamo
            _dynamo.config.suppress_errors = True
            cmode = _os.environ.get("LITE3R_COMPILE_MODE", "default")
            model = torch.compile(model, mode=cmode, fullgraph=False)
            print(f"[inference] torch.compile(mode={cmode}, fullgraph=False, "
                  f"suppress_errors=True) wrapped")

    if _os.environ.get("LITE3R_CUDAGRAPH_MARK_STEP", "0") == "1" and device.type == "cuda":
        class _CUDAGraphStepWrapper(torch.nn.Module):
            def __init__(self, inner):
                super().__init__()
                self.inner = inner
            def forward(self, *a, **kw):
                marker = getattr(getattr(torch, "compiler", None), "cudagraph_mark_step_begin", None)
                if marker is not None:
                    marker()
                return self.inner(*a, **kw)
        model = _CUDAGraphStepWrapper(model)
        print("[inference] cudagraph_mark_step_begin wrapper enabled")

    _, loader = build_eval_loader(cfg.get("data", {}))

    records = []
    for i, batch in enumerate(loader):
        for k in batch:
            if torch.is_tensor(batch[k]):
                batch[k] = batch[k].to(device, non_blocking=True)
        if deploy_kernels and batch["images"].is_floating_point():
            batch["images"] = batch["images"].to(torch.bfloat16)
        with torch.no_grad(), torch.autocast(
            device_type=device.type, dtype=amp_dtype,
            enabled=(device.type == "cuda" and not deploy_kernels),
        ):
            preds = model(batch["images"])
        if deploy_kernels:
            # Lite forward runs in BF16; metrics + GT are FP32. Up-cast outputs
            # so downstream metric math doesn't hit dtype mismatches.
            for k, v in list(preds.items() if isinstance(preds, dict) else []):
                if isinstance(v, torch.Tensor) and v.is_floating_point():
                    preds[k] = v.float()
        # depth
        depth_pred = preds["depth"][..., 0] if preds["depth"].dim() == 5 else preds["depth"]
        d_m = depth_metrics(depth_pred, batch["depths"], batch["point_masks"])
        # pose: convert pose_enc → extrinsics
        pose_enc = preds["pose_enc"]
        extr_pred, intr_pred = vggt_extr_from_pose(pose_enc, batch["images"].shape[-2:])
        # VGGT predicts extrinsics in *view-0-relative* w2c convention:
        # the first frame is treated as the canonical reference (its extrinsic
        # is identity), and all other frames' poses are expressed relative to
        # view 0. BlendedMVS GT stores absolute world-frame w2c. To compare
        # them we canonicalize both to view-0-relative:
        #     extr_rel[i] = extr[i] @ inv(extr[0])
        # This drops mean rot_err on FP baseline from ~127° (absolute mismatch)
        # to <1° (correct alignment, validating the fix).
        def _to_4x4(m):
            if m.shape[-2] == 3:
                tail = torch.tensor([[0,0,0,1.0]], device=m.device, dtype=m.dtype)
                tail = tail.expand(*m.shape[:-2], 1, 4)
                m = torch.cat([m, tail], dim=-2)
            return m
        def _view0_rel(extr):
            extr4 = _to_4x4(extr)
            try:
                inv0 = torch.linalg.inv(extr4[..., :1, :, :].float()).to(extr4.dtype)
            except RuntimeError:
                inv0 = torch.linalg.pinv(extr4[..., :1, :, :].float()).to(extr4.dtype)
            return (extr4 @ inv0)[..., :3, :]
        gt_rel   = _view0_rel(batch["extrinsics"])
        pred_rel = _view0_rel(extr_pred)
        p_m = pose_metrics(pred_rel, gt_rel)
        # geometry: from depth + cam → 3D points and Chamfer.
        # unproject(d_pred, extr_pred, intr_pred) yields wp_pred in *view-0-
        # camera-frame* coords (since extr_pred is view-0-relative w2c). To
        # compare with batch["world_points"] which is absolute world coords,
        # transform GT points into view-0-camera-frame via gt_extr[0].
        try:
            from vggt.utils.geometry import unproject_depth_map_to_point_map
            d0 = depth_pred[0]
            if d0.dim() == 3:
                d0 = d0.unsqueeze(-1)
            wp_pred = unproject_depth_map_to_point_map(d0, extr_pred[0], intr_pred[0])
            import numpy as _np
            if isinstance(wp_pred, _np.ndarray):
                wp_pred = torch.from_numpy(wp_pred).to(d0.device)
            wp_pred = wp_pred.reshape(-1, 3).float()
            # wp_gt is in absolute world coords. Transform to view-0-camera
            # frame to align with wp_pred's coord system:
            #     pt_v0cam = R_gt0 @ pt_world + t_gt0
            wp_gt_world = batch["world_points"][0][batch["point_masks"][0]].reshape(-1, 3).float()
            gt_extr0 = _to_4x4(batch["extrinsics"])[0, 0].float()  # (4, 4) of view 0
            R_gt0 = gt_extr0[:3, :3]
            t_gt0 = gt_extr0[:3, 3]
            wp_gt = (wp_gt_world @ R_gt0.t() + t_gt0).contiguous()
            geo = {
                "chamfer": chamfer_l2(wp_pred, wp_gt),
                "fscore_5cm": fscore_pointcloud(wp_pred, wp_gt, tau=0.05),
            }
        except Exception as e:
            geo = {"chamfer_err": str(e)}
        records.append({**d_m, **p_m, **geo, "scene": batch["seq_name"][0]})
        print(f"[eval-vggt] sample {i}: " + ", ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                                                          for k, v in records[-1].items() if k != "scene"))
        if i >= cfg.get("max_eval_samples", 32) - 1:
            break

    agg = aggregate(records)
    # efficiency on a single representative sample
    sample = next(iter(loader))
    sample_images = sample["images"].to(device)
    if deploy_kernels and sample_images.is_floating_point():
        # Match deployed dtype so latency/MaxMem reflect the BF16+INT4 pipeline
        # the paper claims, not the cost of an opportunistic FP32->BF16 cast.
        sample_images = sample_images.to(torch.bfloat16)
    sample_input = (sample_images,)
    eff = {}
    # Optional: wrap model in a CUDA graph for batch=1 fixed-shape latency
    # measurement. Removes per-call kernel-launch overhead which is sizeable
    # when the model has hundreds of small ops (24×{ln+attn+mlp} ≈ 1000+ ops).
    # Capture may fail on torchao quantized tensors; on failure we fall back
    # to eager so the eval still reports a valid number.
    measure_target = model
    if _os.environ.get("LITE3R_CUDA_GRAPHS", "0") == "1" and device.type == "cuda":
        try:
            for _ in range(3):
                with torch.no_grad():
                    _ = model(*sample_input)
            torch.cuda.synchronize()
            _g = torch.cuda.CUDAGraph()
            _static_in = sample_input[0].clone()
            with torch.cuda.graph(_g):
                with torch.no_grad():
                    _captured_out = model(_static_in)

            class _Graphed:
                def __call__(self, x):
                    _static_in.copy_(x)
                    _g.replay()
                    return _captured_out
            measure_target = _Graphed()
            print("[inference] CUDA Graph captured for latency replay")
        except Exception as _e:
            print(f"[inference] CUDA Graph FAILED ({type(_e).__name__}: {_e}); "
                  f"falling back to eager for latency")
            measure_target = model
    eff.update(measure_latency(measure_target, sample_input, device=device.type))
    eff.update(measure_max_memory(model, sample_input, device=device.type))
    eff.update(count_flops(model, sample_input))
    eff.update(count_parameters(model))

    report = {
        "variant": variant,
        "ckpt": args.ckpt,
        "metrics": agg,
        "per_sample": records,
        "efficiency": eff,
    }
    write_report(out_dir, args.name, report)


if __name__ == "__main__":
    main()
