#!/usr/bin/env python3
"""Evaluate a trained DA3 (Original or Lite) checkpoint."""
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
    p.add_argument("--name", required=True)
    return p.parse_args()


def build_da3(variant: str, da3_root: Path, model_cfg: dict):
    add_paths(str(da3_root / "src"))
    from depth_anything_3.cfg import create_object, load_config
    from depth_anything_3.registry import MODEL_REGISTRY
    name = model_cfg.get("model_name", "da3-large")
    net = create_object(load_config(MODEL_REGISTRY[name]))
    if variant in ("lite_stage1", "lite_stage2") and model_cfg.get("enable_sla", True):
        from depth_anything_3.lite3r_apply import apply_sla
        apply_sla(net, keep_ratio=model_cfg.get("keep_ratio", 0.2),
                  lambda_init=model_cfg.get("lambda_init", 0.5))
    quant_format = str(model_cfg.get("quant_format", "")).lower()
    if variant == "lite_stage2" and quant_format == "fp8":
        from lite3r_kit.fp8_fake_quant import quantize_model_fp8_
        quantize_model_fp8_(
            net,
            enable_act_quant=model_cfg.get("fp8_act_quant", True),
            skip_name_substrings=tuple(model_cfg.get("fp8_skip_name_substrings", ())),
        )
    elif variant == "lite_stage2" and quant_format in ("none", "off", "no", "fp32", "bf16"):
        pass
    elif variant == "lite_stage2":
        from depth_anything_3.lite3r_apply import apply_w4a4
        apply_w4a4(net, group_size=model_cfg.get("group_size", 128),
                   weight_bits=model_cfg.get("weight_bits", 4),
                   act_bits=model_cfg.get("act_bits", 4),
                   quantize_attn_linear=model_cfg.get("quantize_attn_linear", False))
    return net


def main():
    args = parse_args()
    cfg = load_yaml(args.config)
    variant = cfg.get("variant", "original")
    out_dir = Path(cfg.get("output_dir", "outputs/eval_da3"))

    device = get_device()
    amp_dtype = autocast_dtype(device)

    da3_root = PROJECT_ROOT / "model_DA3-Large" / ("Lite" if variant != "original" else "Original")
    net = build_da3(variant, da3_root, cfg.get("model", {}))
    payload = load_ckpt(args.ckpt, net, optim=None, strict=False)
    print(f"[eval-da3] loaded ckpt step={payload.get('step')}")
    net.to(device).eval()
    # Lite variants → switch to real inference kernels (model becomes fully BF16
    # + INT4-W). We must skip autocast and feed BF16 inputs; autocast's per-op
    # policy would re-cast LayerNorm input to FP32 against the BF16 weights.
    is_lite = variant in ("lite_stage1", "lite_stage2")
    import os as _os
    fp32_eval = _os.environ.get("LITE3R_FP32_EVAL", "0") == "1"
    deploy_kernels = is_lite and not fp32_eval
    if deploy_kernels:
        from lite3r_kit import apply_real_inference_kernels
        qmode = _os.environ.get("LITE3R_QUANT_MODE",
                                cfg.get("inference", {}).get("quant_mode"))
        apply_real_inference_kernels(net, quant_mode=qmode)
        if _os.environ.get("LITE3R_COMPILE", "0") == "1":
            import torch._dynamo as _dynamo
            _dynamo.config.suppress_errors = True
            cmode = _os.environ.get("LITE3R_COMPILE_MODE", "default")
            net = torch.compile(net, mode=cmode, fullgraph=False)
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
        net = _CUDAGraphStepWrapper(net)
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
            out = net(batch["images"])
        if deploy_kernels:
            # Lite forward runs in BF16 end-to-end; metrics and GT are FP32, so
            # up-cast the model outputs once here. `out` is an addict.Dict
            # (dict subclass), so iterate via .items().
            for k, v in list(out.items()) if hasattr(out, "items") else []:
                if isinstance(v, torch.Tensor) and v.is_floating_point():
                    out[k] = v.float()
        depth_pred = out.depth  # (B, S, H, W)
        d_m = depth_metrics(depth_pred, batch["depths"], batch["point_masks"])
        # DA3 (like VGGT) outputs extrinsics in view-0-relative w2c convention,
        # while BlendedMVS/DTU64 GT is absolute world-frame w2c. Canonicalize
        # both to view-0-relative before pose comparison; otherwise rot_err
        # blows up to ~127° from coordinate-frame mismatch (verified on VGGT).
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
        pred_rel = _view0_rel(out.extrinsics)
        p_m = pose_metrics(pred_rel, gt_rel)
        # geometry: unproject DA3 depth to view-0-camera-frame points (using
        # predicted view-0-relative extrinsics), then transform GT world points
        # to the same view-0-camera frame for fair Chamfer comparison.
        try:
            extr_pred_w2c = _to_4x4(out.extrinsics)[0].float()  # (S, 4, 4)
            intr = batch["intrinsics"][0]
            depth0 = depth_pred[0].float()   # (S, H, W)
            S, H, W = depth0.shape
            yy, xx = torch.meshgrid(
                torch.arange(H, device=depth0.device),
                torch.arange(W, device=depth0.device), indexing="ij",
            )
            fx = intr[:, 0, 0][:, None, None]
            fy = intr[:, 1, 1][:, None, None]
            cx = intr[:, 0, 2][:, None, None]
            cy = intr[:, 1, 2][:, None, None]
            x_n = (xx[None] - cx) / fx
            y_n = (yy[None] - cy) / fy
            cam = torch.stack([x_n * depth0, y_n * depth0, depth0], dim=-1)
            # Transform predicted camera-frame points (in each view's local
            # camera frame) into view-0-camera frame, using predicted poses:
            #   pt_v0cam = inv(extr_pred[0]) @ extr_pred[i] @ pt_cam[i]
            try:
                inv_pred0 = torch.linalg.inv(extr_pred_w2c[:1])  # (1,4,4)
            except RuntimeError:
                inv_pred0 = torch.linalg.pinv(extr_pred_w2c[:1])
            T_to_v0 = inv_pred0 @ extr_pred_w2c              # (S,4,4)
            R = T_to_v0[:, :3, :3]; t = T_to_v0[:, :3, 3]
            cam_flat = cam.reshape(S, -1, 3).float()
            world = torch.einsum("sij,snj->sni", R, cam_flat) + t[:, None, :]
            wp_pred = world.reshape(-1, 3).contiguous()
            # GT world points → view-0-camera frame using GT view-0 extrinsic
            wp_gt_world = batch["world_points"][0][batch["point_masks"][0]].reshape(-1, 3).float()
            gt_extr0 = _to_4x4(batch["extrinsics"])[0, 0].float()
            R_gt0 = gt_extr0[:3, :3]; t_gt0 = gt_extr0[:3, 3]
            wp_gt = (wp_gt_world @ R_gt0.t() + t_gt0).contiguous()
            geo = {
                "chamfer": chamfer_l2(wp_pred, wp_gt),
                "fscore_5cm": fscore_pointcloud(wp_pred, wp_gt, tau=0.05),
            }
        except Exception as e:
            geo = {"chamfer_err": str(e)}
        records.append({**d_m, **p_m, **geo, "scene": batch["seq_name"][0]})
        print(f"[eval-da3] sample {i}: " + ", ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                                                          for k, v in records[-1].items() if k != "scene"))
        if i >= cfg.get("max_eval_samples", 32) - 1:
            break

    agg = aggregate(records)
    sample = next(iter(loader))
    sample_images = sample["images"].to(device)
    if deploy_kernels and sample_images.is_floating_point():
        # Match deployed dtype so latency/MaxMem reflect the BF16+INT4 pipeline
        # the paper claims, not the cost of an opportunistic FP32->BF16 cast.
        sample_images = sample_images.to(torch.bfloat16)
    sample_input = (sample_images,)
    eff = {}
    measure_target = net
    if _os.environ.get("LITE3R_CUDA_GRAPHS", "0") == "1" and device.type == "cuda":
        try:
            for _ in range(3):
                with torch.no_grad():
                    _ = net(*sample_input)
            torch.cuda.synchronize()
            _g = torch.cuda.CUDAGraph()
            _static_in = sample_input[0].clone()
            with torch.cuda.graph(_g):
                with torch.no_grad():
                    _captured_out = net(_static_in)

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
            measure_target = net
    eff.update(measure_latency(measure_target, sample_input, device=device.type))
    eff.update(measure_max_memory(net, sample_input, device=device.type))
    eff.update(count_flops(net, sample_input))
    eff.update(count_parameters(net))

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
