#!/usr/bin/env python3
"""Collect eval JSON files into CSV and Markdown tables."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def pick(d: dict[str, Any], key: str) -> Any:
    return d.get(key, "")


def fmt(v: Any, digits: int = 4) -> str:
    if v == "" or v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)


def row_from_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    metrics = payload.get("metrics", {})
    eff = payload.get("efficiency", {})
    stem = path.stem
    parts = stem.split("_")
    model = "VGGT" if "vggt" in parts else "DA3" if "da3" in parts else ""
    dataset = "DTU64" if "dtu64" in parts else "BlendedMVS" if "blended" in parts else ""
    return {
        "name": stem,
        "model": model,
        "dataset": dataset,
        "AbsRel": pick(metrics, "abs_rel"),
        "d1": pick(metrics, "delta1"),
        "d2": pick(metrics, "delta2"),
        "d3": pick(metrics, "delta3"),
        "RMSE": pick(metrics, "rmse"),
        "Rot": pick(metrics, "rot_err_deg"),
        "Trans": pick(metrics, "trans_err"),
        "Chamfer": pick(metrics, "chamfer"),
        "F5cm": pick(metrics, "fscore_5cm"),
        "lat_ms": pick(eff, "latency_ms_mean"),
        "p50_ms": pick(eff, "latency_ms_p50"),
        "lat_std": pick(eff, "latency_ms_std"),
        "mem_MB": pick(eff, "max_mem_MB"),
        "flops_g": pick(eff, "flops_g"),
        "params_total": pick(eff, "params_total"),
        "params_trainable": pick(eff, "params_trainable"),
        "json": str(path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, help="output path without extension")
    parser.add_argument("jsons", nargs="+")
    args = parser.parse_args()

    rows = [row_from_json(Path(p)) for p in args.jsons if Path(p).exists()]
    rows.sort(key=lambda r: (r["model"], r["dataset"], r["name"]))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "name", "model", "dataset", "AbsRel", "d1", "d2", "d3", "RMSE",
        "Rot", "Trans", "Chamfer", "F5cm", "lat_ms", "p50_ms", "lat_std",
        "mem_MB", "flops_g", "params_total", "params_trainable", "json",
    ]
    with open(out.with_suffix(".csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Final FP8-QAT Evaluation",
        "",
        "|model|dataset|AbsRel|d1|RMSE|Rot|Trans|Chamfer|F5cm|lat_ms|mem_MB|",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            "|{model}|{dataset}|{AbsRel}|{d1}|{RMSE}|{Rot}|{Trans}|{Chamfer}|"
            "{F5cm}|{lat_ms}|{mem_MB}|".format(
                model=r["model"],
                dataset=r["dataset"],
                AbsRel=fmt(r["AbsRel"]),
                d1=fmt(r["d1"]),
                RMSE=fmt(r["RMSE"]),
                Rot=fmt(r["Rot"]),
                Trans=fmt(r["Trans"]),
                Chamfer=fmt(r["Chamfer"]),
                F5cm=fmt(r["F5cm"]),
                lat_ms=fmt(r["lat_ms"], 2),
                mem_MB=fmt(r["mem_MB"], 2),
            )
        )
    out.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print(f"[collect] wrote {out.with_suffix('.csv')}")
    print(f"[collect] wrote {out.with_suffix('.md')}")


if __name__ == "__main__":
    main()
