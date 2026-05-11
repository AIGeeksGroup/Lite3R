#!/usr/bin/env python3
"""Train a DA3 model (Original or Lite Stage-1/Stage-2) on BlendedMVS.

Mirrors train_vggt.py but uses the DA3 architecture; outputs are adapted to
VGGT's MultitaskLoss via `adapt_da3_output_for_vggt_loss`.
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
from train._common import (
    add_paths, adapt_da3_output_for_vggt_loss, autocast_dtype,
    build_optimizer, build_scheduler, get_device, load_ckpt, load_yaml, save_ckpt,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--variant", default=None)
    p.add_argument("--resume", default=None)
    return p.parse_args()


def build_da3(variant: str, da3_root: Path, model_cfg: dict):
    add_paths(str(da3_root / "src"))
    # Bypass DepthAnything3 wrapper (which pulls in moviepy/gradio/etc. via
    # api.py) and instantiate the underlying nn.Module directly.
    from depth_anything_3.cfg import create_object, load_config
    from depth_anything_3.registry import MODEL_REGISTRY
    name = model_cfg.get("model_name", "da3-large")
    config = load_config(MODEL_REGISTRY[name])
    net = create_object(config)

    if model_cfg.get("load_pretrained", False):
        # Bypass api.py entirely (it imports moviepy/gradio). Pull the
        # safetensors via huggingface_hub directly. PyTorchModelHubMixin
        # serialises with a "model." prefix on every key (because the wrapper
        # holds the net under .model); strip it before loading into the bare
        # nn.Module.
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        repo = model_cfg.get("pretrained_repo", "depth-anything/DA3-LARGE")
        ckpt_path = hf_hub_download(repo, "model.safetensors")
        sd = load_file(ckpt_path)
        if any(k.startswith("model.") for k in sd):
            sd = {k[len("model."):]: v for k, v in sd.items() if k.startswith("model.")}
        msg = net.load_state_dict(sd, strict=False)
        print(f"[da3] loaded pretrained from {repo}: "
              f"missing={len(msg.missing_keys)} unexpected={len(msg.unexpected_keys)}")

    if variant in ("lite_stage1", "lite_stage2") and model_cfg.get("enable_sla", True):
        from depth_anything_3.lite3r_apply import apply_sla
        n_replaced = apply_sla(net,
                               keep_ratio=model_cfg.get("keep_ratio", 0.2),
                               lambda_init=model_cfg.get("lambda_init", 0.5))
        print(f"[da3-lite] swapped {n_replaced} attention modules to SLAAttention")
    elif variant in ("lite_stage1", "lite_stage2"):
        print("[da3-lite] enable_sla=false; keeping dense Attention modules")

    quant_format = str(model_cfg.get("quant_format", "")).lower()
    if variant == "lite_stage2" and quant_format == "fp8":
        from lite3r_kit.fp8_fake_quant import quantize_model_fp8_
        n_q = quantize_model_fp8_(
            net,
            enable_act_quant=model_cfg.get("fp8_act_quant", True),
            skip_name_substrings=tuple(model_cfg.get("fp8_skip_name_substrings", ())),
        )
        print(f"[da3-lite] wrapped {n_q} nn.Linear with FP8FakeQuantLinear")
    elif variant == "lite_stage2" and quant_format in ("none", "off", "no", "fp32", "bf16"):
        print(f"[da3-lite] quant_format={quant_format}; skipping training-time fake quant")
    elif variant == "lite_stage2":
        from depth_anything_3.lite3r_apply import apply_w4a4
        n_q = apply_w4a4(net,
                         group_size=model_cfg.get("group_size", 128),
                         weight_bits=model_cfg.get("weight_bits", 4),
                         act_bits=model_cfg.get("act_bits", 4),
                         quantize_attn_linear=model_cfg.get("quantize_attn_linear", False))
        print(f"[da3-lite] wrapped {n_q} nn.Linear with FakeQuantLinear (W4A4)")

    return net


def build_dataset(data_cfg: dict):
    sys.path.insert(0, str(PROJECT_ROOT))
    from data.blendedmvs import BlendedMVSDataset, collate_fn
    from data.dummy import DummyMultiViewDataset

    if data_cfg.get("use_dummy", False) or not os.path.isdir(data_cfg["root"]):
        ds = DummyMultiViewDataset(
            length=data_cfg.get("dummy_len", 16),
            img_per_seq=data_cfg.get("img_per_seq", 2),
            img_size=data_cfg.get("img_size", 224),
        )
    else:
        ds = BlendedMVSDataset(
            root=data_cfg["root"],
            img_per_seq=data_cfg.get("img_per_seq", 2),
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


def main():
    args = parse_args()
    cfg = load_yaml(args.config)
    if args.variant:
        cfg["variant"] = args.variant
    variant = cfg.get("variant", "original")
    out_dir = Path(cfg.get("output_dir", "outputs/da3_run"))
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config_resolved.yaml", "w") as f:
        yaml.safe_dump(cfg, f)

    device = get_device()
    amp_dtype = autocast_dtype(device)

    da3_lite_root = PROJECT_ROOT / "model_DA3-Large" / "Lite"
    da3_orig_root = PROJECT_ROOT / "model_DA3-Large" / "Original"
    da3_root = da3_lite_root if variant != "original" else da3_orig_root

    net = build_da3(variant, da3_root, cfg.get("model", {}))
    if cfg.get("init_from_stage1", None):
        ck = load_ckpt(cfg["init_from_stage1"], net, optim=None, strict=False)
        print(f"[da3-lite] inited from {cfg['init_from_stage1']} step={ck.get('step')}")
    net.to(device).train()

    # Optional: train only proj_lin (freeze backbone qkv/MLP/head). See
    # train_vggt.py for rationale — the SLA residual branch alone learns
    # to compensate W4 quantisation noise without poisoning pretrained
    # backbone representation (the failure mode of full-fine-tuning QAT).
    if cfg.get("trainable_only_proj_lin", False):
        n_frozen, n_trainable = 0, 0
        for n, p in net.named_parameters():
            if "proj_lin" in n:
                p.requires_grad = True
                n_trainable += 1
            else:
                p.requires_grad = False
                n_frozen += 1
        print(f"[freeze] trainable_only_proj_lin: {n_trainable} params trainable, "
              f"{n_frozen} frozen")
    elif cfg.get("trainable_substrings"):
        pats = list(cfg["trainable_substrings"])
        n_frozen, n_trainable = 0, 0
        for n, p in net.named_parameters():
            if any(pat in n for pat in pats):
                p.requires_grad = True
                n_trainable += 1
            else:
                p.requires_grad = False
                n_frozen += 1
        print(f"[freeze] trainable_substrings={pats}: {n_trainable} trainable, "
              f"{n_frozen} frozen")

    # teacher for KD: must mirror the student's pretrained init (otherwise
    # distilling against a random teacher actively hurts).
    teacher = None
    teacher_rec = student_rec = None
    if variant in ("lite_stage1", "lite_stage2") and cfg.get("model", {}).get("enable_kd",
            variant == "lite_stage1"):
        from lite3r_kit import AttentionOutputRecorder
        add_paths(str(da3_orig_root / "src"))
        from depth_anything_3.cfg import create_object, load_config
        from depth_anything_3.registry import MODEL_REGISTRY
        t_name = cfg.get("model", {}).get("model_name", "da3-large")
        teacher = create_object(load_config(MODEL_REGISTRY[t_name])).eval()
        if cfg.get("model", {}).get("load_pretrained", False):
            from huggingface_hub import hf_hub_download
            from safetensors.torch import load_file
            repo = cfg.get("model", {}).get("pretrained_repo", "depth-anything/DA3-LARGE")
            ckpt_path = hf_hub_download(repo, "model.safetensors")
            sd = load_file(ckpt_path)
            if any(k.startswith("model.") for k in sd):
                sd = {k[len("model."):]: v for k, v in sd.items() if k.startswith("model.")}
            msg = teacher.load_state_dict(sd, strict=False)
            print(f"[da3-kd-teacher] loaded pretrained from {repo}: "
                  f"missing={len(msg.missing_keys)} unexpected={len(msg.unexpected_keys)}")
        for p in teacher.parameters():
            p.requires_grad = False
        teacher.to(device)
        teacher_rec = AttentionOutputRecorder(teacher)
        student_rec = AttentionOutputRecorder(net)

    _, loader = build_dataset(cfg.get("data", {}))

    add_paths(str(PROJECT_ROOT / "model_VGGT" / "Original"),
              str(PROJECT_ROOT / "model_VGGT" / "Original" / "training"))
    from vggt.utils.pose_enc import extri_intri_to_pose_encoding
    from training.loss import MultitaskLoss
    loss_module = MultitaskLoss(
        camera=cfg.get("loss", {}).get("camera", {"weight": 1.0}),
        depth=cfg.get("loss", {}).get("depth", {"weight": 1.0, "gradient_loss_fn": ""}),
    ).to(device)

    epochs = cfg.get("optim", {}).get("epochs", 20)
    lr = cfg.get("optim", {}).get("lr", 1e-4)
    wd = cfg.get("optim", {}).get("weight_decay", 0.05)
    optim = build_optimizer(net, lr=lr, weight_decay=wd)
    total_steps = max(1, epochs * len(loader))
    sched = build_scheduler(optim, total_steps=total_steps,
                              warmup_steps=cfg.get("optim", {}).get("warmup_steps", 100))
    grad_clip = cfg.get("optim", {}).get("grad_clip", 1.0)
    grad_accum = max(1, int(cfg.get("optim", {}).get("grad_accum", 1)))
    log_every = cfg.get("log_every", 10)
    save_every = cfg.get("save_every", 500)

    step = 0
    if args.resume:
        payload = load_ckpt(args.resume, net, optim)
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
                out = net(batch["images"])
                preds = adapt_da3_output_for_vggt_loss(
                    out, batch["images"].shape[-2:], extri_intri_to_pose_encoding,
                )
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

            # Gradient accumulation: see train_vggt.py for the rationale.
            (loss / grad_accum).backward()
            if (step + 1) % grad_accum == 0:
                if grad_clip:
                    torch.nn.utils.clip_grad_norm_(net.parameters(), grad_clip)
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
                save_ckpt(str(out_dir / f"ckpt_step{step}.pt"), net, optim, step=step)

            step += 1

    # final ckpt: drop optimizer state (saves ~50% disk; eval / stage2 init
    # do not need it). Stage2 starts a fresh optimizer anyway.
    save_ckpt(str(out_dir / "last.pt"), net, optim=None, step=step)
    print(f"[da3] done. saved to {out_dir/'last.pt'}")


if __name__ == "__main__":
    main()
