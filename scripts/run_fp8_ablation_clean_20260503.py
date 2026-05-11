#!/usr/bin/env python3
"""Run paper-facing FP8 ablations with consistent deployment modes.

Differences from the raw fp8_ablation_20260503 runner:

* QAT rows are evaluated with FP8 weight-only deployment.
* no-QAT rows are evaluated without deployment quantization and without SAGE.
* Summary cells are never left blank; missing or invalid metrics are marked NA.
* A full SLA+FP8-QAT+KD0.1 row is included for context.
* Temporary checkpoints are removed after both dataset evals unless --keep-ckpt.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml


PROJECT = Path(__file__).resolve().parents[1]
RUN_TAG = "fp8_ablation_clean_20260503"
CONFIG_DIR = PROJECT / "configs" / RUN_TAG
LOG_DIR = PROJECT / "logs" / RUN_TAG
OUT_CSV = PROJECT / "outputs" / f"{RUN_TAG}.csv"
OUT_MD = PROJECT / "outputs" / f"{RUN_TAG}.md"
PYTHON = os.environ.get("PYTHON", sys.executable)


EXPERIMENTS = [
    {
        "name": "sla_qat_kd01",
        "route": "Full: SLA + FP8-QAT + KD0.1",
        "enable_sla": True,
        "qat": True,
        "kd": 0.1,
        "family": "full",
        "use_existing_final": True,
    },
    {
        "name": "sla_noqat_kd01",
        "route": "w/o QAT: SLA + KD0.1",
        "enable_sla": True,
        "qat": False,
        "kd": 0.1,
        "family": "component",
    },
    {
        "name": "nosla_qat_kd01",
        "route": "w/o SLA: FP8-QAT + KD0.1",
        "enable_sla": False,
        "qat": True,
        "kd": 0.1,
        "family": "component",
    },
    {
        "name": "nosla_noqat_kd01",
        "route": "w/o SLA & QAT: KD0.1",
        "enable_sla": False,
        "qat": False,
        "kd": 0.1,
        "family": "component",
    },
    {
        "name": "sla_qat_kd00",
        "route": "SLA + FP8-QAT + KD0.0",
        "enable_sla": True,
        "qat": True,
        "kd": 0.0,
        "family": "kd_gamma",
    },
    {
        "name": "sla_qat_kd02",
        "route": "SLA + FP8-QAT + KD0.2",
        "enable_sla": True,
        "qat": True,
        "kd": 0.2,
        "family": "kd_gamma",
    },
    {
        "name": "sla_qat_kd05",
        "route": "SLA + FP8-QAT + KD0.5",
        "enable_sla": True,
        "qat": True,
        "kd": 0.5,
        "family": "kd_gamma",
    },
]


MODEL_SPECS = {
    "vggt": {
        "label": "VGGT",
        "train_script": "train/train_vggt.py",
        "eval_script": "eval/eval_vggt.py",
        "init_from_stage1": "outputs/vggt_keep03_r2a_qat/last.pt",
        "existing_final_ckpt": "outputs/vggt_fp8_qat_1ep/last.pt",
        "trainable_no_sla": ["camera_head", "depth_head"],
        "model": {
            "img_size": 518,
            "patch_size": 14,
            "embed_dim": 1024,
            "load_pretrained": True,
            "pretrained_repo": "facebook/VGGT-1B",
            "keep_ratio": 0.3,
            "lambda_init": 0.5,
        },
    },
    "da3": {
        "label": "DA3",
        "train_script": "train/train_da3.py",
        "eval_script": "eval/eval_da3.py",
        "init_from_stage1": "outputs/da3_keep03_r2a_qat_w4a8/last.pt",
        "existing_final_ckpt": "outputs/da3_fp8_qat_1ep/last.pt",
        "trainable_no_sla": ["head", "cam_dec", "cam_enc"],
        "model": {
            "model_name": "da3-large",
            "load_pretrained": True,
            "pretrained_repo": "depth-anything/DA3-LARGE",
            "keep_ratio": 0.3,
            "lambda_init": 0.5,
        },
    },
}


LOSS_CFG = {
    "camera": {
        "weight": 1.0,
        "loss_type": "l1",
        "gamma": 0.6,
        "pose_encoding_type": "absT_quaR_FoV",
        "weight_trans": 1.0,
        "weight_rot": 1.0,
        "weight_focal": 0.5,
    },
    "depth": {
        "weight": 1.0,
        "gamma": 1.0,
        "alpha": 0.2,
        "gradient_loss_fn": "",
        "valid_range": 0.95,
    },
}


FIELDS = [
    "family", "ablation", "route", "model", "dataset", "status",
    "AbsRel", "d1", "d2", "d3", "RMSE", "Rot", "Trans", "Chamfer", "F5cm",
    "lat_ms", "p50_ms", "lat_std", "mem_MB", "GFLOPs", "params_M",
    "trainable_M", "eval_mode", "notes", "json",
]


def kd_min(kd: float) -> float:
    return 0.0 if kd == 0.0 else kd * 0.1


def output_dir(model_name: str, exp: dict[str, Any]) -> str:
    return f"outputs/{RUN_TAG}_{model_name}_{exp['name']}"


def eval_json_path(model_name: str, exp: dict[str, Any], dataset: str) -> Path:
    name = f"{model_name}_{exp['name']}_{dataset}"
    return PROJECT / output_dir(model_name, exp) / f"{name}.json"


def both_evals_done(model_name: str, exp: dict[str, Any]) -> bool:
    return eval_json_path(model_name, exp, "blended").exists() and eval_json_path(model_name, exp, "dtu64").exists()


def ckpt_path(model_name: str, exp: dict[str, Any]) -> Path:
    spec = MODEL_SPECS[model_name]
    if exp.get("use_existing_final"):
        return PROJECT / spec["existing_final_ckpt"]
    return PROJECT / output_dir(model_name, exp) / "last.pt"


def build_cfg(model_name: str, exp: dict[str, Any], *, dtu: bool = False) -> dict[str, Any]:
    spec = MODEL_SPECS[model_name]
    cfg: dict[str, Any] = {
        "variant": "lite_stage2" if exp["qat"] else "lite_stage1",
        "output_dir": output_dir(model_name, exp),
        "init_from_stage1": spec["init_from_stage1"],
        "seed": 42,
        "model": dict(spec["model"]),
        "data": {
            "root": "datasets/BlendedMVS_lowres/BlendedMVS",
            "use_dummy": False,
            "img_per_seq": 2,
            "img_size": 518,
            "batch_size": 1,
            "num_workers": 2,
        },
        "loss": LOSS_CFG,
        "optim": {
            "epochs": 1,
            "lr": 1.0e-3,
            "weight_decay": 0.05,
            "warmup_steps": 30,
            "grad_clip": 1.0,
        },
        "log_every": 10,
        "save_every": 9999,
        "max_eval_samples": 32,
        "ablation": {
            "run_tag": RUN_TAG,
            "name": exp["name"],
            "route": exp["route"],
            "family": exp["family"],
            "paper_eval_rule": "qat->fp8_weight_only, no_qat->no_deploy_quant",
        },
    }
    cfg["model"].update(
        {
            "enable_sla": bool(exp["enable_sla"]),
            "enable_kd": True,
            "kd_gamma_max": float(exp["kd"]),
            "kd_gamma_min": float(kd_min(exp["kd"])),
        }
    )
    if exp["qat"]:
        cfg["model"].update({"quant_format": "fp8", "fp8_act_quant": True})
    else:
        cfg["model"].update({"quant_format": "none"})

    if exp["enable_sla"]:
        cfg["trainable_only_proj_lin"] = True
    else:
        cfg["trainable_substrings"] = spec["trainable_no_sla"]

    if dtu:
        cfg["data"] = {
            "dataset": "dtu64",
            "root": "datasets/dtu64",
            "use_dummy": False,
            "img_per_seq": 2,
            "img_size": 518,
            "batch_size": 1,
            "num_workers": 0,
        }
        cfg["max_eval_samples"] = 14
    return cfg


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


def run_logged(cmd: list[str], log_path: Path, *, env: dict[str, str] | None = None) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(log_path, "a", buffering=1) as log:
        log.write(f"\n[$ {start}] {' '.join(cmd)}\n")
        proc = subprocess.run(cmd, cwd=PROJECT, env=env, stdout=log, stderr=subprocess.STDOUT)
        log.write(f"[$ exit={proc.returncode}]\n")
        return int(proc.returncode)


def write_failure_json(model_name: str, exp: dict[str, Any], dataset: str, message: str) -> None:
    path = eval_json_path(model_name, exp, dataset)
    path.parent.mkdir(parents=True, exist_ok=True)
    _, mode = eval_env(exp)
    payload = {
        "status": "failed",
        "error": message,
        "variant": "lite_stage2" if exp["qat"] else "lite_stage1",
        "metrics": {},
        "efficiency": {},
        "eval_mode": mode,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")


def train_one(model_name: str, exp: dict[str, Any], *, force: bool) -> None:
    if exp.get("use_existing_final") and not force:
        p = ckpt_path(model_name, exp)
        if not p.exists():
            raise FileNotFoundError(f"missing existing final checkpoint: {p}")
        print(f"[train-skip] {model_name}/{exp['name']} uses {p}")
        return
    if both_evals_done(model_name, exp) and not force:
        print(f"[train-skip] {model_name}/{exp['name']} already has both eval JSON files")
        return
    cfg_path = CONFIG_DIR / f"{model_name}_{exp['name']}.yaml"
    write_yaml(cfg_path, build_cfg(model_name, exp, dtu=False))
    ckpt = ckpt_path(model_name, exp)
    if ckpt.exists() and not force:
        print(f"[train-skip] {model_name}/{exp['name']} already has {ckpt}")
        return
    script = MODEL_SPECS[model_name]["train_script"]
    log = LOG_DIR / f"train_{model_name}_{exp['name']}.log"
    rc = run_logged([PYTHON, script, "--config", str(cfg_path)], log)
    if rc != 0:
        raise RuntimeError(f"train failed: {model_name}/{exp['name']} rc={rc} log={log}")


def eval_env(exp: dict[str, Any]) -> tuple[dict[str, str], str]:
    env = os.environ.copy()
    if exp["qat"]:
        env.update(
            {
                "LITE3R_QUANT_MODE": "fp8_weight_only",
                "LITE3R_SAGE_ATTN": "1",
                "LITE3R_SAGE_SMOOTH_K": "1",
            }
        )
        env.pop("LITE3R_FP32_EVAL", None)
        return env, "fp8_weight_only+sAGE"
    env.update(
        {
            "LITE3R_FP32_EVAL": "1",
            "LITE3R_SAGE_ATTN": "0",
            "LITE3R_SAGE_SMOOTH_K": "0",
            "LITE3R_INT4": "0",
            "LITE3R_QUANT_MODE": "none",
        }
    )
    return env, "no_deploy_quant"


def eval_one(model_name: str, exp: dict[str, Any], dataset: str, *, force: bool) -> None:
    is_dtu = dataset == "dtu64"
    cfg_path = CONFIG_DIR / f"{model_name}_{exp['name']}_{dataset}.yaml"
    cfg = build_cfg(model_name, exp, dtu=is_dtu)
    write_yaml(cfg_path, cfg)
    name = f"{model_name}_{exp['name']}_{dataset}"
    json_path = eval_json_path(model_name, exp, dataset)
    if json_path.exists() and not force:
        print(f"[eval-skip] {name} already has {json_path}")
        return
    ckpt = ckpt_path(model_name, exp)
    if not ckpt.exists():
        raise FileNotFoundError(f"missing checkpoint: {ckpt}")
    env, mode = eval_env(exp)
    log = LOG_DIR / f"eval_{name}.log"
    script = MODEL_SPECS[model_name]["eval_script"]
    print(f"[eval] {name} mode={mode}")
    rc = run_logged([PYTHON, script, "--config", str(cfg_path), "--ckpt", str(ckpt), "--name", name], log, env=env)
    if rc != 0:
        msg = f"eval failed: {name} rc={rc} log={log}"
        write_failure_json(model_name, exp, dataset, msg)
        print(f"[eval-failed] {msg}")


def clean_ckpt(model_name: str, exp: dict[str, Any], *, keep: bool) -> None:
    if keep or exp.get("use_existing_final"):
        return
    ckpt = ckpt_path(model_name, exp)
    if ckpt.exists():
        size_gb = ckpt.stat().st_size / (1024 ** 3)
        ckpt.unlink()
        print(f"[clean] removed {ckpt} ({size_gb:.2f} GiB)")


def scalar(v: Any) -> Any:
    if v is None:
        return "NA"
    if isinstance(v, float):
        if not math.isfinite(v):
            return "NA"
        return v
    return v


def get_num(d: dict[str, Any], key: str) -> Any:
    return scalar(d.get(key))


def fmt(v: Any, digits: int = 4) -> str:
    v = scalar(v)
    if v == "NA" or v == "":
        return "NA"
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)


def row_from_json(model_name: str, exp: dict[str, Any], dataset: str) -> dict[str, Any]:
    path = eval_json_path(model_name, exp, dataset)
    env_mode = "fp8_weight_only+sAGE" if exp["qat"] else "no_deploy_quant"
    row: dict[str, Any] = {
        "family": exp["family"],
        "ablation": exp["name"],
        "route": exp["route"],
        "model": MODEL_SPECS[model_name]["label"],
        "dataset": "BlendedMVS" if dataset == "blended" else "DTU64",
        "status": "missing",
        "eval_mode": env_mode,
        "notes": "json_missing",
        "json": str(path.relative_to(PROJECT)),
    }
    for field in FIELDS:
        row.setdefault(field, "NA")
    if not path.exists():
        return row
    try:
        payload = json.loads(path.read_text())
    except Exception as exc:
        row.update({"status": "json_error", "notes": f"{type(exc).__name__}: {exc}"})
        return row
    if payload.get("status") == "failed":
        row.update({"status": "failed", "notes": str(payload.get("error", "eval_failed"))})
        return row
    m = payload.get("metrics", {}) or {}
    e = payload.get("efficiency", {}) or {}
    missing = []
    mapping = {
        "AbsRel": (m, "abs_rel"),
        "d1": (m, "delta1"),
        "d2": (m, "delta2"),
        "d3": (m, "delta3"),
        "RMSE": (m, "rmse"),
        "Rot": (m, "rot_err_deg"),
        "Trans": (m, "trans_err"),
        "Chamfer": (m, "chamfer"),
        "F5cm": (m, "fscore_5cm"),
        "lat_ms": (e, "latency_ms_mean"),
        "p50_ms": (e, "latency_ms_p50"),
        "lat_std": (e, "latency_ms_std"),
        "mem_MB": (e, "max_mem_MB"),
        "GFLOPs": (e, "flops_g"),
        "params_M": (e, "params_total"),
        "trainable_M": (e, "params_trainable"),
    }
    for out_key, (src, in_key) in mapping.items():
        val = get_num(src, in_key)
        if val == "NA":
            missing.append(out_key)
        row[out_key] = val
    row["status"] = "ok"
    expected_missing = {"AbsRel", "d1", "d2", "d3", "RMSE", "Chamfer", "F5cm"} if dataset == "dtu64" else set()
    unexpected = [k for k in missing if k not in expected_missing]
    if unexpected:
        row["status"] = "partial"
        row["notes"] = "missing:" + "/".join(unexpected)
    else:
        row["notes"] = "OK" if not expected_missing else "DTU64 pose-only"
    return row


def collect_rows() -> list[dict[str, Any]]:
    rows = []
    for exp in EXPERIMENTS:
        for model_name in MODEL_SPECS:
            for dataset in ("blended", "dtu64"):
                rows.append(row_from_json(model_name, exp, dataset))
    return rows


def write_summary() -> None:
    rows = collect_rows()
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: fmt(row.get(k), 6) if k not in ("family", "ablation", "route", "model", "dataset", "status", "eval_mode", "notes", "json") else row.get(k, "NA") for k in FIELDS})

    lines = [
        "# Clean FP8 Lite3R Ablation Results",
        "",
        f"Run tag: `{RUN_TAG}`.",
        "",
        "Evaluation rule:",
        "",
        "- FP8-QAT rows: `LITE3R_QUANT_MODE=fp8_weight_only`, `LITE3R_SAGE_ATTN=1`.",
        "- no-QAT rows: `LITE3R_FP32_EVAL=1`, no deployment quantization, no SAGE.",
        "- `NA` is explicit. DTU64 is pose-only in this setup, so depth/Chamfer columns are expected `NA` there.",
        "",
        "|family|ablation|model|dataset|status|AbsRel|d1|RMSE|Rot|Trans|Chamfer|F5cm|lat_ms|p50_ms|mem_MB|eval_mode|notes|",
        "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for r in rows:
        lines.append(
            "|{family}|{ablation}|{model}|{dataset}|{status}|{AbsRel}|{d1}|{RMSE}|"
            "{Rot}|{Trans}|{Chamfer}|{F5cm}|{lat_ms}|{p50_ms}|{mem_MB}|{eval_mode}|{notes}|".format(
                family=r["family"],
                ablation=r["ablation"],
                model=r["model"],
                dataset=r["dataset"],
                status=r["status"],
                AbsRel=fmt(r["AbsRel"]),
                d1=fmt(r["d1"]),
                RMSE=fmt(r["RMSE"]),
                Rot=fmt(r["Rot"]),
                Trans=fmt(r["Trans"]),
                Chamfer=fmt(r["Chamfer"]),
                F5cm=fmt(r["F5cm"]),
                lat_ms=fmt(r["lat_ms"], 2),
                p50_ms=fmt(r["p50_ms"], 2),
                mem_MB=fmt(r["mem_MB"], 2),
                eval_mode=r["eval_mode"],
                notes=r["notes"],
            )
        )
    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f"[summary] wrote {OUT_CSV}")
    print(f"[summary] wrote {OUT_MD}")


def select(names: str, available: list[str]) -> list[str]:
    if names == "all":
        return available
    wanted = [x.strip() for x in names.split(",") if x.strip()]
    bad = [x for x in wanted if x not in available]
    if bad:
        raise SystemExit(f"unknown selection {bad}; available={available}")
    return wanted


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="all", help="all, vggt, da3, or comma list")
    parser.add_argument("--experiments", default="all", help="all or comma-separated experiment names")
    parser.add_argument("--force-train", action="store_true")
    parser.add_argument("--force-eval", action="store_true")
    parser.add_argument("--keep-ckpt", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()

    model_names = select(args.models, list(MODEL_SPECS.keys()))
    exp_names = select(args.experiments, [e["name"] for e in EXPERIMENTS])
    exps = [e for e in EXPERIMENTS if e["name"] in exp_names]

    if not args.summary_only:
        for exp in exps:
            for model_name in model_names:
                print(f"== {model_name}/{exp['name']} ==")
                try:
                    train_one(model_name, exp, force=args.force_train)
                except Exception as exc:
                    msg = f"train failed: {type(exc).__name__}: {exc}"
                    print(f"[train-failed] {model_name}/{exp['name']} {msg}")
                    write_failure_json(model_name, exp, "blended", msg)
                    write_failure_json(model_name, exp, "dtu64", msg)
                    write_summary()
                    continue
                eval_one(model_name, exp, "blended", force=args.force_eval)
                eval_one(model_name, exp, "dtu64", force=args.force_eval)
                clean_ckpt(model_name, exp, keep=args.keep_ckpt)
                write_summary()
                # Clean possible torch cache before next large model.
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass
    write_summary()


if __name__ == "__main__":
    main()
