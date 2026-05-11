#!/usr/bin/env python3
"""Train a VGGT model (Original or Lite Stage-1/Stage-2) on BlendedMVS.

Usage:
    python train/train_vggt.py --config configs/vggt_original.yaml
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "lite3r_kit" / ".."))  # for lite3r_kit
from train._common import (
    add_paths, autocast_dtype, build_optimizer, build_scheduler,
    get_device, load_ckpt, load_yaml, save_ckpt,
)


@torch.no_grad()
def calibrate_smooth(model: torch.nn.Module, loader, device: torch.device,
                     n_batches: int = 4, alpha: float = 0.5) -> int:
    """SmoothQuant-style calibration of FakeQuantLinear.smooth buffers.

    Runs a few forward passes, accumulates per-input-feature absmax of
    activations into each FakeQuantLinear, then sets:
        s = max|X|^alpha / max|W|^(1-alpha)
    so that subsequent `x_smoothed = x / s` lowers activation outliers
    while `w_smoothed = w * s` keeps the matmul output unchanged. Reduces
    quant noise floor at zero training cost.

    Returns number of FakeQuantLinear modules calibrated.
    """
    from lite3r_kit.fake_quant import FakeQuantLinear

    activations: dict[str, torch.Tensor] = {}
    handles = []

    def make_hook(name: str):
        def hook(module, inputs):
            x = inputs[0]
            if not torch.is_tensor(x) or not x.is_floating_point():
                return
            x_flat = x.detach().abs().reshape(-1, x.shape[-1])
            cur = x_flat.amax(dim=0)
            prev = activations.get(name)
            activations[name] = cur if prev is None else torch.maximum(prev, cur)
        return hook

    for n, m in model.named_modules():
        if isinstance(m, FakeQuantLinear):
            handles.append(m.register_forward_pre_hook(make_hook(n)))

    was_training = model.training
    model.eval()
    seen = 0
    for i, batch in enumerate(loader):
        if i >= n_batches:
            break
        try:
            imgs = batch["images"].to(device, non_blocking=True)
            _ = model(imgs)
            seen += 1
        except Exception as e:
            print(f"[calibrate-smooth] batch {i} forward failed: {type(e).__name__}: {e}")
            break
    if was_training:
        model.train()
    for h in handles:
        h.remove()

    n_calib = 0
    for n, m in model.named_modules():
        if isinstance(m, FakeQuantLinear) and n in activations:
            a_max = activations[n].clamp(min=1e-8)
            w_max = m.weight.detach().abs().amax(dim=0).clamp(min=1e-8)
            s = (a_max.pow(alpha) / w_max.pow(1.0 - alpha)).clamp(min=1e-8)
            m.smooth.copy_(s.to(m.smooth.dtype))
            n_calib += 1
    print(f"[calibrate-smooth] calibrated {n_calib} FakeQuantLinear modules "
          f"from {seen} batches (alpha={alpha})")
    return n_calib


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="path to YAML config")
    p.add_argument("--variant", default=None,
                   help="override variant: original | lite_stage1 | lite_stage2")
    p.add_argument("--resume", default=None, help="path to checkpoint to resume from")
    return p.parse_args()


def build_vggt(variant: str, vggt_root: Path, model_cfg: dict):
    add_paths(str(vggt_root), str(vggt_root / "training"))
    from vggt.models.vggt import VGGT  # noqa: E402

    model = VGGT(
        img_size=model_cfg.get("img_size", 518),
        patch_size=model_cfg.get("patch_size", 14),
        embed_dim=model_cfg.get("embed_dim", 1024),
        enable_camera=True,
        enable_depth=True,
        enable_point=False,
        enable_track=False,
    )
    if model_cfg.get("load_pretrained", False):
        from huggingface_hub import hf_hub_download
        ckpt_id = model_cfg.get("pretrained_repo", "facebook/VGGT-1B")
        ckpt_path = hf_hub_download(ckpt_id, "model.safetensors")
        from safetensors.torch import load_file
        sd = load_file(ckpt_path)
        msg = model.load_state_dict(sd, strict=False)
        print(f"[vggt] loaded pretrained from {ckpt_id}: missing={len(msg.missing_keys)} "
              f"unexpected={len(msg.unexpected_keys)}")

    if variant in ("lite_stage1", "lite_stage2") and model_cfg.get("enable_sla", True):
        from vggt.lite3r_apply import apply_sla
        n_replaced = apply_sla(model,
                               keep_ratio=model_cfg.get("keep_ratio", 0.2),
                               lambda_init=model_cfg.get("lambda_init", 0.5))
        print(f"[vggt-lite] swapped {n_replaced} attention modules to SLAAttention")
    elif variant in ("lite_stage1", "lite_stage2"):
        print("[vggt-lite] enable_sla=false; keeping dense Attention modules")

    quant_format = str(model_cfg.get("quant_format", "")).lower()
    if variant == "lite_stage2" and quant_format == "fp8":
        from lite3r_kit.fp8_fake_quant import quantize_model_fp8_
        n_q = quantize_model_fp8_(
            model,
            enable_act_quant=model_cfg.get("fp8_act_quant", True),
            skip_name_substrings=tuple(model_cfg.get("fp8_skip_name_substrings", ())),
        )
        print(f"[vggt-lite] wrapped {n_q} nn.Linear with FP8FakeQuantLinear")
    elif variant == "lite_stage2" and quant_format in ("none", "off", "no", "fp32", "bf16"):
        print(f"[vggt-lite] quant_format={quant_format}; skipping training-time fake quant")
    elif variant == "lite_stage2":
        from vggt.lite3r_apply import apply_w4a4
        n_q = apply_w4a4(model,
                         group_size=model_cfg.get("group_size", 128),
                         weight_bits=model_cfg.get("weight_bits", 4),
                         act_bits=model_cfg.get("act_bits", 4),
                         quantize_attn_linear=model_cfg.get("quantize_attn_linear", False))
        print(f"[vggt-lite] wrapped {n_q} nn.Linear with FakeQuantLinear (W4A4)")

    return model


def build_dataset(data_cfg: dict):
    sys.path.insert(0, str(PROJECT_ROOT))
    from data.blendedmvs import BlendedMVSDataset, collate_fn
    from data.dummy import DummyMultiViewDataset

    if data_cfg.get("use_dummy", False) or not os.path.isdir(data_cfg["root"]):
        ds = DummyMultiViewDataset(
            length=data_cfg.get("dummy_len", 16),
            img_per_seq=data_cfg.get("img_per_seq", 4),
            img_size=data_cfg.get("img_size", 224),
        )
    else:
        ds = BlendedMVSDataset(
            root=data_cfg["root"],
            img_per_seq=data_cfg.get("img_per_seq", 4),
            img_size=data_cfg.get("img_size", 224),
            max_scenes=data_cfg.get("max_scenes", None),
        )
    cfn = collate_fn
    loader = DataLoader(
        ds,
        batch_size=data_cfg.get("batch_size", 1),
        shuffle=True,
        num_workers=data_cfg.get("num_workers", 2),
        collate_fn=cfn,
        drop_last=True,
        pin_memory=True,
    )
    return ds, loader


def setup_kd(model, vggt_root: Path, model_cfg: dict, variant: str):
    """Build a teacher (frozen FP-precision dense) for Stage-1 / Stage-2 KD.

    Stage-1 default: KD on (anchors lite student to dense pretrained attention).
    Stage-2 default: KD off (was empirically destructive in early experiments;
    can be re-enabled via `enable_kd: true` in config when probing whether
    KD anchors qkv against quantisation-noise drift).
    """
    if variant not in ("lite_stage1", "lite_stage2"):
        return None, None, None
    kd_default = True if variant == "lite_stage1" else False
    if not model_cfg.get("enable_kd", kd_default):
        return None, None, None
    from lite3r_kit import AttentionOutputRecorder

    add_paths(str(vggt_root), str(vggt_root / "training"))
    from vggt.models.vggt import VGGT
    teacher = VGGT(
        img_size=model_cfg.get("img_size", 518),
        patch_size=model_cfg.get("patch_size", 14),
        embed_dim=model_cfg.get("embed_dim", 1024),
        enable_camera=True, enable_depth=True, enable_point=False, enable_track=False,
    )
    # KD teacher must be a *useful* dense reference. Mirror the student's
    # pretrained init so distillation targets the actual pretrained attention
    # distribution (random-init teacher would just inject noise).
    if model_cfg.get("load_pretrained", False):
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        ckpt_id = model_cfg.get("pretrained_repo", "facebook/VGGT-1B")
        ckpt_path = hf_hub_download(ckpt_id, "model.safetensors")
        sd = load_file(ckpt_path)
        msg = teacher.load_state_dict(sd, strict=False)
        print(f"[vggt-kd-teacher] loaded pretrained from {ckpt_id}: "
              f"missing={len(msg.missing_keys)} unexpected={len(msg.unexpected_keys)}")
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    teacher_rec = AttentionOutputRecorder(teacher)
    student_rec = AttentionOutputRecorder(model)
    return teacher, teacher_rec, student_rec


def main():
    args = parse_args()
    cfg = load_yaml(args.config)
    if args.variant:
        cfg["variant"] = args.variant

    variant = cfg.get("variant", "original")
    out_dir = Path(cfg.get("output_dir", "outputs/vggt_run"))
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config_resolved.yaml", "w") as f:
        yaml.safe_dump(cfg, f)

    # Fix all randomness sources so retrains are reproducible.
    # Without this, abs_rel for the same recipe (same ckpt init, same yaml,
    # same code) was observed to vary 4× between runs (0.033 ↔ 0.138) due
    # to DataLoader shuffle / dropout / cudnn nondeterminism. Seed selection
    # is read from yaml `seed:` field, default 42.
    import random as _random, numpy as _np
    seed = int(cfg.get("seed", 42))
    _random.seed(seed)
    _np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[train-vggt] fixed seed={seed}")

    device = get_device()
    amp_dtype = autocast_dtype(device)

    vggt_lite_root = PROJECT_ROOT / "model_VGGT" / "Lite"
    vggt_orig_root = PROJECT_ROOT / "model_VGGT" / "Original"
    vggt_root = vggt_lite_root if variant != "original" else vggt_orig_root

    model = build_vggt(variant, vggt_root, cfg.get("model", {}))
    if cfg.get("init_from_stage1", None):
        ck = load_ckpt(cfg["init_from_stage1"], model, optim=None, strict=False)
        print(f"[vggt-lite] inited from {cfg['init_from_stage1']} step={ck.get('step')}")

    model.to(device)
    model.train()

    # Optional: train only proj_lin (freeze backbone qkv/MLP/head/etc).
    # Used as a QAT ablation to test whether the SLA residual branch alone
    # can compensate quantisation noise without poisoning the pretrained
    # representation in qkv/MLP/head.
    if cfg.get("trainable_only_proj_lin", False):
        n_frozen, n_trainable = 0, 0
        for n, p in model.named_parameters():
            if "proj_lin" in n:
                p.requires_grad = True
                n_trainable += 1
            else:
                p.requires_grad = False
                n_frozen += 1
        print(f"[freeze] trainable_only_proj_lin: {n_trainable} params trainable, "
              f"{n_frozen} frozen")
    elif cfg.get("trainable_substrings"):
        # Generalised version: keep params whose qualified name contains any
        # of the listed substrings; freeze everything else. Used for S2/S3
        # strict-QAT ablations where we want to train MLP+heads+proj_lin
        # (semi-strict, includes W4-quantised weights) or just heads+proj_lin.
        pats = list(cfg["trainable_substrings"])
        n_frozen, n_trainable = 0, 0
        for n, p in model.named_parameters():
            if any(pat in n for pat in pats):
                p.requires_grad = True
                n_trainable += 1
            else:
                p.requires_grad = False
                n_frozen += 1
        print(f"[freeze] trainable_substrings={pats}: {n_trainable} trainable, "
              f"{n_frozen} frozen")

    teacher, teacher_rec, student_rec = setup_kd(model, vggt_root, cfg.get("model", {}), variant)
    if teacher is not None:
        teacher.to(device)

    # data
    _, loader = build_dataset(cfg.get("data", {}))

    # Optional SmoothQuant calibration before training. Runs a few forward
    # passes through the model to set FakeQuantLinear.smooth buffers via
    # the standard SmoothQuant formula, lowering activation-side
    # quantisation noise from the first training step onward.
    if cfg.get("model", {}).get("calibrate_smooth", False):
        n_calib_batches = int(cfg.get("model", {}).get("calibrate_smooth_batches", 4))
        alpha = float(cfg.get("model", {}).get("calibrate_smooth_alpha", 0.5))
        calibrate_smooth(model, loader, device,
                         n_batches=n_calib_batches, alpha=alpha)

    # loss
    add_paths(str(vggt_root), str(vggt_root / "training"))
    from training.loss import MultitaskLoss
    loss_module = MultitaskLoss(
        camera=cfg.get("loss", {}).get("camera", {"weight": 1.0}),
        depth=cfg.get("loss", {}).get("depth", {"weight": 1.0,
                                                  "gradient_loss_fn": ""}),
    )
    loss_module.to(device)

    # optim & sched
    epochs = cfg.get("optim", {}).get("epochs", 20)
    lr = cfg.get("optim", {}).get("lr", 1e-4)
    wd = cfg.get("optim", {}).get("weight_decay", 0.05)
    optim = build_optimizer(model, lr=lr, weight_decay=wd)
    total_steps = max(1, epochs * len(loader))
    sched = build_scheduler(optim, total_steps=total_steps,
                              warmup_steps=cfg.get("optim", {}).get("warmup_steps", 100))

    grad_clip = cfg.get("optim", {}).get("grad_clip", 1.0)
    grad_accum = max(1, int(cfg.get("optim", {}).get("grad_accum", 1)))
    log_every = cfg.get("log_every", 10)
    save_every = cfg.get("save_every", 500)

    step = 0
    if args.resume:
        payload = load_ckpt(args.resume, model, optim)
        step = payload.get("step", 0)

    t0 = time.time()
    for epoch in range(epochs):
        for batch in loader:
            for k in batch:
                if torch.is_tensor(batch[k]):
                    batch[k] = batch[k].to(device, non_blocking=True)

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
                if student_rec is not None:
                    student_rec.reset()
                preds = model(batch["images"])
                loss_dict = loss_module(preds, batch)
                loss = loss_dict["objective"]

                if teacher is not None:
                    teacher_rec.reset()
                    with torch.no_grad():
                        _ = teacher(batch["images"])
                    from lite3r_kit import compute_kd_loss, cosine_kd_weight
                    gamma = cosine_kd_weight(step, total_steps,
                                              cfg.get("model", {}).get("kd_gamma_max", 0.1),
                                              cfg.get("model", {}).get("kd_gamma_min", 0.01))
                    loss_kd = compute_kd_loss(teacher_rec, student_rec)
                    loss = loss + gamma * loss_kd
                    loss_dict["loss_kd"] = loss_kd
                    loss_dict["kd_gamma"] = gamma

            # Gradient accumulation: average loss over `grad_accum` micro-steps
            # before doing one optimizer step. Effective batch is
            # data_batch_size * grad_accum. clip+step+sched advance once per
            # *effective* step; zero_grad happens after the optimizer step,
            # so the next accumulation cycle starts clean.
            (loss / grad_accum).backward()
            if (step + 1) % grad_accum == 0:
                if grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optim.step()
                sched.step()
                optim.zero_grad(set_to_none=True)

            if step % log_every == 0:
                elapsed = time.time() - t0
                msg = (f"step={step:6d} ep={epoch:3d} loss={loss.item():.4f} "
                       f"lr={optim.param_groups[0]['lr']:.2e} "
                       f"t={elapsed:.0f}s")
                if "loss_kd" in loss_dict:
                    msg += f" kd={loss_dict['loss_kd'].item():.4f}γ={loss_dict['kd_gamma']:.3f}"
                print(msg, flush=True)

            if step > 0 and step % save_every == 0:
                save_ckpt(str(out_dir / f"ckpt_step{step}.pt"), model, optim, step=step)

            step += 1

    # final ckpt: drop optimizer state (saves ~50% disk; eval / stage2 init
    # do not need it). Stage2 starts a fresh optimizer anyway.
    save_ckpt(str(out_dir / "last.pt"), model, optim=None, step=step)
    print(f"[vggt] done. saved to {out_dir/'last.pt'}")


if __name__ == "__main__":
    main()
