#!/usr/bin/env python3
"""60-second sanity check before `bash setup.sh` / `sbatch ...`.

Validates: Python version, GPU + sm_80 capability, disk space, mirror env
vars and reachability, key Python deps, project file structure, and a tiny
end-to-end smoke train + eval on dummy data.

Exit code:
    0 — all green (or only warnings); safe to proceed
    1 — at least one ✗; fix before running setup.sh

Usage:
    python scripts/preflight.py
    # or, if executable:
    ./scripts/preflight.py
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from urllib.error import URLError
from urllib.request import Request, urlopen

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(HERE)

# --- pretty-print helpers -------------------------------------------------
TTY = sys.stdout.isatty()
G = "\033[0;32m" if TTY else ""
R = "\033[0;31m" if TTY else ""
Y = "\033[0;33m" if TTY else ""
B = "\033[1m" if TTY else ""
N = "\033[0m" if TTY else ""

results = {"pass": 0, "warn": 0, "fail": 0}


def ok(msg, hint=""):
    print(f"  {G}✓{N} {msg}" + (f"  ({hint})" if hint else ""))
    results["pass"] += 1


def warn(msg, hint=""):
    print(f"  {Y}!{N} {msg}" + (f"  → {hint}" if hint else ""))
    results["warn"] += 1


def fail(msg, hint=""):
    print(f"  {R}✗{N} {msg}" + (f"  → {hint}" if hint else ""))
    results["fail"] += 1


def section(num, total, title):
    print(f"\n{B}[{num}/{total}] {title}{N}")


def probe(url, timeout=10):
    """HEAD-style probe; returns (ok: bool, info: str)."""
    try:
        req = Request(url, method="HEAD",
                      headers={"User-Agent": "lite3r-preflight/1.0"})
        with urlopen(req, timeout=timeout) as r:
            return True, f"HTTP {r.status}"
    except URLError as e:
        return False, str(getattr(e, "reason", e))
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


print(f"{B}==[ Lite3R preflight @ {HERE} ]=={N}")


# --- 1. Python -----------------------------------------------------------
section(1, 8, "Python interpreter")
v = sys.version_info
print(f"    {sys.executable}")
if v >= (3, 9):
    ok(f"Python {v.major}.{v.minor}.{v.micro}")
else:
    fail(f"Python {v.major}.{v.minor} < 3.9",
         "conda env create -f environment.yml && conda activate l3rsla")


# --- 2. GPU --------------------------------------------------------------
section(2, 8, "GPU / CUDA")
try:
    import torch
    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability(0)
        name = torch.cuda.get_device_name(0)
        bf16 = torch.cuda.is_bf16_supported()
        sm = f"sm_{cap[0]}{cap[1]}"
        if cap[0] >= 8:
            ok(f"{name}  {sm}  bf16={bf16}")
        else:
            warn(f"{name}  {sm}  — torchao INT4 needs sm_80+; will fall back to BF16",
                 "Latency/MaxMem won't show real wins on this GPU")
    else:
        fail("torch installed but CUDA not visible",
             "check `nvidia-smi` and that conda env points at CUDA torch wheel")
except ImportError:
    warn("torch not installed yet", "run `bash setup.sh` first, then re-run preflight")


# --- 3. Disk space --------------------------------------------------------
section(3, 8, "Disk space")
total, used, free = shutil.disk_usage(HERE)
free_gb = free / 1024 ** 3
if free_gb >= 50:
    ok(f"{free_gb:.1f} GB free in {HERE}")
elif free_gb >= 35:
    warn(f"{free_gb:.1f} GB free — BlendedMVS extracted is ~30GB, ckpts another few; tight")
else:
    fail(f"only {free_gb:.1f} GB free", "need 50GB+; free disk or move project root")


# --- 4. Mirror env vars + reachability -----------------------------------
section(4, 8, "Mirror env vars + reachability")
hf_ep = os.environ.get("HF_ENDPOINT", "").rstrip("/")
gh_mirror = os.environ.get("GITHUB_MIRROR", "")
pip_idx = os.environ.get("PIP_INDEX_URL", "")

# HF
hf_test_url = f"{hf_ep or 'https://huggingface.co'}/api/models/facebook/VGGT-1B"
hf_ok, hf_info = probe(hf_test_url)
if hf_ep:
    if hf_ok:
        ok(f"HF_ENDPOINT={hf_ep}", hf_info)
    else:
        fail(f"HF_ENDPOINT={hf_ep} unreachable", f"{hf_info}; try https://hf-mirror.com")
else:
    if hf_ok:
        ok("HF_ENDPOINT not set; direct huggingface.co reachable")
    else:
        warn("HF_ENDPOINT not set & direct HF unreachable",
             "export HF_ENDPOINT=https://hf-mirror.com")

# GitHub mirror — probe a small file (1KB) on the BlendedMVS release
gh_target = ("https://github.com/YoYo000/BlendedMVS/"
             "releases/download/v1.0.0/BlendedMVS.zip")
gh_url = f"{gh_mirror}{gh_target}" if gh_mirror else gh_target
gh_ok, gh_info = probe(gh_url)
if gh_mirror:
    if gh_ok:
        ok(f"GITHUB_MIRROR={gh_mirror}", gh_info)
    else:
        fail(f"GITHUB_MIRROR={gh_mirror} unreachable",
             f"{gh_info}; try https://gh-proxy.com/ or https://mirror.ghproxy.com/")
else:
    if gh_ok:
        ok("GITHUB_MIRROR not set; direct github.com reachable")
    else:
        warn("github.com unreachable from here",
             "export GITHUB_MIRROR=https://ghproxy.com/")

# pip mirror
if pip_idx:
    pip_ok, pip_info = probe(pip_idx)
    if pip_ok:
        ok(f"PIP_INDEX_URL={pip_idx}", pip_info)
    else:
        warn(f"PIP_INDEX_URL={pip_idx} unreachable", pip_info)
else:
    ok("PIP_INDEX_URL not set; using default pypi.org")


# --- 5. Key Python deps --------------------------------------------------
section(5, 8, "Key Python deps")
import importlib
DEPS_REQUIRED = [
    ("torch", "torch"),
    ("torchvision", "torchvision"),
    ("huggingface_hub", "huggingface_hub"),
    ("safetensors", "safetensors"),
    ("yaml", "PyYAML"),
    ("numpy", "numpy"),
    ("PIL", "Pillow"),
    ("einops", "einops"),
    ("addict", "addict"),
    ("omegaconf", "omegaconf"),
    ("fvcore", "fvcore"),
]
DEPS_LITE_RUNTIME = [
    ("torchao", "torchao", "Lite eval will skip the real INT4 kernel"),
    ("xformers", "xformers", "DA3 may print warnings (not fatal)"),
]

for mod, pkg in DEPS_REQUIRED:
    try:
        m = importlib.import_module(mod)
        ver = getattr(m, "__version__", "?")
        ok(f"{pkg} {ver}")
    except Exception as e:
        fail(f"{pkg} import failed", f"pip install {pkg}  ({type(e).__name__})")

for mod, pkg, impact in DEPS_LITE_RUNTIME:
    try:
        m = importlib.import_module(mod)
        ver = getattr(m, "__version__", "?")
        ok(f"{pkg} {ver}")
    except Exception:
        warn(f"{pkg} missing — {impact}", f"pip install {pkg}")


# --- 6. Project structure ------------------------------------------------
section(6, 8, "Project structure")
required_files = [
    "configs/vggt_lite_stage1.yaml",
    "configs/vggt_lite_stage2.yaml",
    "configs/da3_lite_stage1.yaml",
    "lite3r_kit/sla.py",
    "lite3r_kit/inference.py",
    "lite3r_kit/fake_quant.py",
    "model_VGGT/Original/vggt/models/vggt.py",
    "model_VGGT/Lite/vggt/lite3r_apply.py",
    "model_DA3-Large/Original/src/depth_anything_3/cfg.py",
    "model_DA3-Large/Lite/src/depth_anything_3/lite3r_apply.py",
    "train/train_vggt.py",
    "train/train_da3.py",
    "eval/eval_vggt.py",
    "eval/eval_da3.py",
    "setup.sh",
    "scripts/run_all_resume.sh",
    "scripts/sbatch_run_all.sh",
    "environment.yml",
    "requirements.txt",
]
for p in required_files:
    if os.path.exists(p):
        ok(p)
    else:
        fail(f"{p} MISSING", "tarball didn't extract correctly; re-download")


# --- 7. Project import sanity --------------------------------------------
section(7, 8, "Project imports (lite3r_kit + model packages)")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "model_DA3-Large", "Lite", "src"))
try:
    from lite3r_kit import (SLAAttention, FakeQuantLinear,
                             apply_real_inference_kernels,
                             AttentionOutputRecorder)
    ok("lite3r_kit  (SLA + FakeQuant + KD + inference helpers)")
except Exception as e:
    fail("lite3r_kit import failed", f"{type(e).__name__}: {e}")

try:
    from depth_anything_3.cfg import create_object, load_config
    from depth_anything_3.registry import MODEL_REGISTRY
    ok(f"depth_anything_3  ({len(MODEL_REGISTRY)} model presets)")
except Exception as e:
    fail("depth_anything_3 import failed", f"{type(e).__name__}: {e}")


# --- 8. End-to-end smoke (dummy data, ~30s) ------------------------------
section(8, 8, "End-to-end smoke train + eval (dummy data, ~30s)")

if results["fail"] > 0:
    warn("skipping smoke because earlier checks failed")
else:
    try:
        import yaml
        cfg = {
            "variant": "lite_stage1",
            "output_dir": "outputs/_preflight_smoke",
            "model": {"model_name": "da3-small", "load_pretrained": False,
                      "keep_ratio": 0.2, "lambda_init": 0.5, "enable_kd": False},
            "data": {"root": "__none__", "use_dummy": True, "dummy_len": 1,
                     "img_per_seq": 2, "img_size": 56, "batch_size": 1, "num_workers": 0},
            "loss": {"camera": {"weight": 1.0},
                     "depth": {"weight": 1.0, "gradient_loss_fn": ""}},
            "optim": {"epochs": 1, "lr": 1e-4, "weight_decay": 0.05,
                      "warmup_steps": 0, "grad_clip": 1.0},
            "log_every": 1, "save_every": 9999,
        }
        cfg_path = "/tmp/_preflight_train.yaml"
        eval_cfg = {**cfg, "output_dir": "outputs/_preflight_eval", "max_eval_samples": 1}
        eval_cfg_path = "/tmp/_preflight_eval.yaml"
        with open(cfg_path, "w") as f:
            yaml.safe_dump(cfg, f)
        with open(eval_cfg_path, "w") as f:
            yaml.safe_dump(eval_cfg, f)

        env = os.environ.copy()
        env["PYTHONPATH"] = HERE + (":" + env.get("PYTHONPATH", "") if env.get("PYTHONPATH") else "")

        # train
        p = subprocess.run(
            [sys.executable, "train/train_da3.py", "--config", cfg_path],
            env=env, capture_output=True, text=True, timeout=180,
        )
        if p.returncode == 0 and "loss=" in p.stdout:
            ok("dummy train step ran")
        else:
            tail = (p.stderr or p.stdout).strip().splitlines()[-3:]
            fail(f"smoke train failed (rc={p.returncode})", " | ".join(tail))

        # eval (verifies apply_real_inference_kernels)
        if os.path.exists("outputs/_preflight_smoke/last.pt"):
            p2 = subprocess.run(
                [sys.executable, "eval/eval_da3.py", "--config", eval_cfg_path,
                 "--ckpt", "outputs/_preflight_smoke/last.pt", "--name", "preflight"],
                env=env, capture_output=True, text=True, timeout=180,
            )
            stdout = p2.stdout
            if p2.returncode == 0:
                if "torchao INT4-W-only" in stdout:
                    ok("eval ran with REAL torchao INT4 kernel — Lite metrics will be authentic")
                elif "torchao SKIP" in stdout:
                    warn("eval ran but torchao SKIPPED",
                         "Lite Latency/MaxMem won't reflect real INT4; check torchao + sm_80+")
                else:
                    ok("eval ran (no inference kernel signature spotted)")
            else:
                tail = (p2.stderr or p2.stdout).strip().splitlines()[-3:]
                fail(f"smoke eval failed (rc={p2.returncode})", " | ".join(tail))

        # cleanup
        for d in ("outputs/_preflight_smoke", "outputs/_preflight_eval"):
            shutil.rmtree(d, ignore_errors=True)
        for f in (cfg_path, eval_cfg_path):
            try: os.unlink(f)
            except OSError: pass
    except Exception as e:
        fail(f"smoke wrapper crashed: {type(e).__name__}: {e}")


# --- summary -------------------------------------------------------------
print()
total = sum(results.values())
summary = (f"{B}{results['pass']}/{total} pass{N}, "
           f"{Y}{results['warn']} warn{N}, "
           f"{R}{results['fail']} fail{N}")
print(f"==[ {summary} ]==")
if results["fail"] > 0:
    print(f"{R}Fix the ✗ items above before running setup.sh / sbatch.{N}")
    sys.exit(1)
elif results["warn"] > 0:
    print(f"{Y}Warnings noted. You can proceed but read them.{N}")
    print()
    print("Next:")
    print("  bash setup.sh")
    print("  sbatch scripts/sbatch_run_all.sh")
    sys.exit(0)
else:
    print(f"{G}All green. Proceed:{N}")
    print("  bash setup.sh                   # ~30 min: data + deps")
    print("  sbatch scripts/sbatch_run_all.sh")
    sys.exit(0)
