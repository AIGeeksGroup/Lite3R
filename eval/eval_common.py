"""Shared evaluation harness invoked by eval_vggt.py and eval_da3.py."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.blendedmvs import BlendedMVSDataset, collate_fn
from data.dummy import DummyMultiViewDataset


def build_eval_loader(data_cfg: Dict[str, Any]):
    name = (data_cfg.get("dataset") or "").lower()
    if data_cfg.get("use_dummy", False) or not os.path.isdir(data_cfg["root"]):
        ds = DummyMultiViewDataset(
            length=data_cfg.get("dummy_len", 4),
            img_per_seq=data_cfg.get("img_per_seq", 4),
            img_size=data_cfg.get("img_size", 224),
        )
    elif name == "dtu64":
        from data.dtu64 import DTU64Dataset
        ds = DTU64Dataset(
            root=data_cfg["root"],
            img_per_seq=data_cfg.get("img_per_seq", 2),
            img_size=data_cfg.get("img_size", 518),
            max_scenes=data_cfg.get("max_scenes", None),
        )
    else:
        ds = BlendedMVSDataset(
            root=data_cfg["root"],
            img_per_seq=data_cfg.get("img_per_seq", 4),
            img_size=data_cfg.get("img_size", 224),
            max_scenes=data_cfg.get("max_scenes", None),
        )
    cfn = collate_fn
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=cfn)
    return ds, loader


def aggregate(metric_records: list[dict]) -> Dict[str, float]:
    out = {}
    keys = set()
    for r in metric_records:
        keys.update(r.keys())
    for k in keys:
        vals = [r[k] for r in metric_records if k in r and not (isinstance(r[k], float) and r[k] != r[k])]
        if not vals:
            continue
        if isinstance(vals[0], (int, float)):
            out[k] = float(sum(vals) / len(vals))
    return out


def write_report(out_dir: Path, name: str, payload: Dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fp = out_dir / f"{name}.json"
    with open(fp, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[eval] wrote {fp}")
