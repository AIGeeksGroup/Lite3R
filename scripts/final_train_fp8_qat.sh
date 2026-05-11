#!/usr/bin/env bash
# Train the final FP8-QAT route from pretrained public backbones.

set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$HERE"

PY="${PYTHON:-python}"
MODEL="${1:-all}"  # all | vggt | da3

train_vggt() {
  if [ ! -f outputs/vggt_keep03_r2a_qat/last.pt ]; then
    "$PY" train/train_vggt.py --config configs/final/vggt_stage1_sla_w4a16.yaml
  else
    echo "[final-train] found outputs/vggt_keep03_r2a_qat/last.pt"
  fi
  "$PY" train/train_vggt.py --config configs/final/vggt_fp8_qat_1ep.yaml
}

train_da3() {
  if [ ! -f outputs/da3_keep03_r2a_qat_w4a8/last.pt ]; then
    "$PY" train/train_da3.py --config configs/final/da3_stage1_sla_w4a8.yaml
  else
    echo "[final-train] found outputs/da3_keep03_r2a_qat_w4a8/last.pt"
  fi
  "$PY" train/train_da3.py --config configs/final/da3_fp8_qat_1ep.yaml
}

case "$MODEL" in
  all) train_vggt; train_da3 ;;
  vggt) train_vggt ;;
  da3) train_da3 ;;
  *) echo "usage: $0 [all|vggt|da3]" >&2; exit 2 ;;
esac
