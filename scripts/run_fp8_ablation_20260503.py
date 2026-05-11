#!/usr/bin/env python3
"""Run FP8 Lite3R ablations.

This wrapper only orchestrates existing train/eval entrypoints. It writes
temporary YAML configs under configs/fp8_ablation_20260503, runs 6 ablation
groups for VGGT and DA3, evaluates each checkpoint on BlendedMVS and DTU64,
then writes a compact CSV/Markdown summary under outputs/.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml


PROJECT = Path(__file__).resolve().parents[1]
RUN_TAG = "fp8_ablation_20260503"
CONFIG_DIR = PROJECT / "configs" / RUN_TAG
LOG_DIR = PROJECT / "logs" / RUN_TAG
OUT_CSV = PROJECT / "outputs" / f"{RUN_TAG}.csv"
OUT_MD = PROJECT / "outputs" / f"{RUN_TAG}.md"

PYTHON = os.environ.get("PYTHON", sys.executable)


EXPERIMENTS = [
    {
        "name": "sla_noqat_kd01",
        "route": "SLA + no-QAT + KD0.1",
        "enable_sla": True,
        "qat": False,
        "kd": 0.1,
        "family": "component",
    },
    {
        "name": "nosla_qat_kd01",
        "route": "no-SLA + FP8-QAT + KD0.1",
        "enable_sla": False,
        "qat": True,
        "kd": 0.1,
        "family": "component",
    },
    {
        "name": "nosla_noqat_kd01",
        "route": "no-SLA + no-QAT + KD0.1",
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
        "train_script": "train/train_vggt.py",
        "eval_script": "eval/eval_vggt.py",
        "init_from_stage1": "outputs/vggt_keep03_r2a_qat/last.pt",
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
        "train_script": "train/train_da3.py",
        "eval_script": "eval/eval_da3.py",
        "init_from_stage1": "outputs/da3_keep03_r2a_qat_w4a8/last.pt",
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


def kd_min(kd: float) -> float:
    return 0.0 if kd == 0.0 else kd * 0.1


def output_dir(model_name: str, exp_name: str) -> str:
    return f"outputs/{RUN_TAG}_{model_name}_{exp_name}"


def eval_json_path(model_name: str, exp: dict[str, Any], dataset: str) -> Path:
    name = f"{model_name}_{exp['name']}_{dataset}"
    return PROJECT / output_dir(model_name, exp["name"]) / f"{name}.json"


def both_evals_done(model_name: str, exp: dict[str, Any]) -> bool:
    return eval_json_path(model_name, exp, "blended").exists() and eval_json_path(model_name, exp, "dtu64").exists()


def build_cfg(model_name: str, exp: dict[str, Any], *, dtu: bool = False) -> dict[str, Any]:
    spec = MODEL_SPECS[model_name]
    cfg: dict[str, Any] = {
        "variant": "lite_stage2" if exp["qat"] else "lite_stage1",
        "output_dir": output_dir(model_name, exp["name"]),
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


def train_one(model_name: str, exp: dict[str, Any], force: bool) -> None:
    cfg_path = CONFIG_DIR / f"{model_name}_{exp['name']}.yaml"
    write_yaml(cfg_path, build_cfg(model_name, exp, dtu=False))
    if both_evals_done(model_name, exp) and not force:
        print(f"[skip-train] {model_name}/{exp['name']} already has both eval JSON files")
        return
    ckpt = PROJECT / output_dir(model_name, exp["name"]) / "last.pt"
    if ckpt.exists() and not force:
        print(f"[skip-train] {model_name}/{exp['name']} already has {ckpt}")
        return
    script = MODEL_SPECS[model_name]["train_script"]
    log = LOG_DIR / f"train_{model_name}_{exp['name']}.log"
    rc = run_logged([PYTHON, script, "--config", str(cfg_path)], log)
    if rc != 0:
        raise RuntimeError(f"train failed: {model_name}/{exp['name']} rc={rc} log={log}")


def eval_one(model_name: str, exp: dict[str, Any], dataset: str, force: bool) -> None:
    is_dtu = dataset == "dtu64"
    cfg_path = CONFIG_DIR / f"{model_name}_{exp['name']}_{dataset}.yaml"
    write_yaml(cfg_path, build_cfg(model_name, exp, dtu=is_dtu))
    ckpt = PROJECT / output_dir(model_name, exp["name"]) / "last.pt"
    name = f"{model_name}_{exp['name']}_{dataset}"
    json_path = eval_json_path(model_name, exp, dataset)
    if json_path.exists() and not force:
        print(f"[skip-eval] {name} already has {json_path}")
        return
    env = os.environ.copy()
    env.update(
        {
            "LITE3R_QUANT_MODE": "fp8_weight_only",
            "LITE3R_SAGE_ATTN": "1",
            "LITE3R_SAGE_SMOOTH_K": "1",
        }
    )
    script = MODEL_SPECS[model_name]["eval_script"]
    log = LOG_DIR / f"eval_{name}.log"
    rc = run_logged([PYTHON, script, "--config", str(cfg_path), "--ckpt", str(ckpt), "--name", name], log, env=env)
    if rc != 0:
        raise RuntimeError(f"eval failed: {name} rc={rc} log={log}")


def clean_ckpt_one(model_name: str, exp: dict[str, Any]) -> None:
    ckpt = PROJECT / output_dir(model_name, exp["name"]) / "last.pt"
    if ckpt.exists() and both_evals_done(model_name, exp):
        size_gb = ckpt.stat().st_size / (1024 ** 3)
        ckpt.unlink()
        print(f"[clean] removed {ckpt} ({size_gb:.2f} GiB)")


def get_nested(d: dict[str, Any], key: str) -> Any:
    return d.get(key, "")


def collect_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for exp in EXPERIMENTS:
        for model_name in MODEL_SPECS:
            for dataset in ("blended", "dtu64"):
                name = f"{model_name}_{exp['name']}_{dataset}"
                json_path = PROJECT / output_dir(model_name, exp["name"]) / f"{name}.json"
                row: dict[str, Any] = {
                    "family": exp["family"],
                    "ablation": exp["name"],
                    "route": exp["route"],
                    "model": model_name.upper() if model_name == "da3" else "VGGT",
                    "dataset": "BlendedMVS" if dataset == "blended" else "DTU64",
                    "json": str(json_path.relative_to(PROJECT)),
                }
                if not json_path.exists():
                    row["status"] = "missing"
                    rows.append(row)
                    continue
                with open(json_path) as f:
                    payload = json.load(f)
                m = payload.get("metrics", {})
                e = payload.get("efficiency", {})
                row.update(
                    {
                        "status": "ok",
                        "AbsRel": get_nested(m, "abs_rel"),
                        "d1": get_nested(m, "delta1"),
                        "d2": get_nested(m, "delta2"),
                        "d3": get_nested(m, "delta3"),
                        "RMSE": get_nested(m, "rmse"),
                        "Rot": get_nested(m, "rot_err_deg"),
                        "Trans": get_nested(m, "trans_err"),
                        "Chamfer": get_nested(m, "chamfer"),
                        "F5cm": get_nested(m, "fscore_5cm"),
                        "lat_ms": get_nested(e, "latency_ms_mean"),
                        "p50_ms": get_nested(e, "latency_ms_p50"),
                        "lat_std": get_nested(e, "latency_ms_std"),
                        "mem_MB": get_nested(e, "max_mem_MB"),
                        "flops_g": get_nested(e, "flops_g"),
                        "params_total": get_nested(e, "params_total"),
                        "params_trainable": get_nested(e, "params_trainable"),
                    }
                )
                rows.append(row)
    return rows


def fmt(v: Any, digits: int = 4) -> str:
    if v == "" or v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)


def write_summary() -> None:
    rows = collect_rows()
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "family",
        "ablation",
        "route",
        "model",
        "dataset",
        "status",
        "AbsRel",
        "d1",
        "d2",
        "d3",
        "RMSE",
        "Rot",
        "Trans",
        "Chamfer",
        "F5cm",
        "lat_ms",
        "p50_ms",
        "lat_std",
        "mem_MB",
        "flops_g",
        "params_total",
        "params_trainable",
        "json",
    ]
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# FP8 Lite3R Ablation Results",
        "",
        f"Run tag: `{RUN_TAG}`.",
        "",
        "Training recipe: 1 epoch on BlendedMVS low-res, batch size 1, seed 42. "
        "Deployment eval uses `LITE3R_QUANT_MODE=fp8_weight_only`, "
        "`LITE3R_SAGE_ATTN=1`, `LITE3R_SAGE_SMOOTH_K=1`.",
        "",
        "For no-SLA variants, the backbone has no `proj_lin`; the script trains only "
        "model heads (`camera_head/depth_head` for VGGT, `head/cam_dec/cam_enc` for DA3) "
        "to keep the trainable scope small and avoid full-backbone fine-tuning.",
        "",
        "|family|ablation|model|dataset|AbsRel|d1|RMSE|Rot|Trans|Chamfer|F5cm|lat_ms|mem_MB|",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            "|{family}|{ablation}|{model}|{dataset}|{AbsRel}|{d1}|{RMSE}|{Rot}|{Trans}|"
            "{Chamfer}|{F5cm}|{lat_ms}|{mem_MB}|".format(
                family=r["family"],
                ablation=r["ablation"],
                model=r["model"],
                dataset=r["dataset"],
                AbsRel=fmt(r.get("AbsRel")),
                d1=fmt(r.get("d1")),
                RMSE=fmt(r.get("RMSE")),
                Rot=fmt(r.get("Rot")),
                Trans=fmt(r.get("Trans")),
                Chamfer=fmt(r.get("Chamfer")),
                F5cm=fmt(r.get("F5cm")),
                lat_ms=fmt(r.get("lat_ms"), 2),
                mem_MB=fmt(r.get("mem_MB"), 2),
            )
        )
    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f"[summary] wrote {OUT_CSV}")
    print(f"[summary] wrote {OUT_MD}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=["all", "train", "eval", "report", "stream"], default="all")
    p.add_argument("--force", action="store_true")
    p.add_argument("--clean-ckpt", action="store_true",
                   help="after both eval JSON files exist, delete this run's temporary last.pt")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    if args.phase == "stream":
        for exp in EXPERIMENTS:
            for model_name in MODEL_SPECS:
                if both_evals_done(model_name, exp) and not args.force:
                    print(f"[stream-skip] {model_name}/{exp['name']} already evaluated", flush=True)
                    if args.clean_ckpt:
                        clean_ckpt_one(model_name, exp)
                    continue
                print(f"[stream-train] {model_name}/{exp['name']}: {exp['route']}", flush=True)
                train_one(model_name, exp, args.force)
                for dataset in ("blended", "dtu64"):
                    print(f"[stream-eval] {model_name}/{exp['name']}/{dataset}", flush=True)
                    eval_one(model_name, exp, dataset, args.force)
                write_summary()
                if args.clean_ckpt:
                    clean_ckpt_one(model_name, exp)
        write_summary()
        return

    if args.phase in ("all", "train"):
        for exp in EXPERIMENTS:
            for model_name in MODEL_SPECS:
                print(f"[train] {model_name}/{exp['name']}: {exp['route']}", flush=True)
                train_one(model_name, exp, args.force)

    if args.phase in ("all", "eval"):
        for exp in EXPERIMENTS:
            for model_name in MODEL_SPECS:
                for dataset in ("blended", "dtu64"):
                    print(f"[eval] {model_name}/{exp['name']}/{dataset}", flush=True)
                    eval_one(model_name, exp, dataset, args.force)

    write_summary()
    if args.clean_ckpt:
        for exp in EXPERIMENTS:
            for model_name in MODEL_SPECS:
                clean_ckpt_one(model_name, exp)


if __name__ == "__main__":
    main()
