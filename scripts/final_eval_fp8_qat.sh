#!/usr/bin/env bash
# Evaluate the final FP8-QAT checkpoints on BlendedMVS and DTU64.

set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$HERE"

PY="${PYTHON:-python}"
MODEL="${1:-all}"  # all | vggt | da3
CKPT_ROOT="${CKPT_ROOT:-checkpoints/fp8_qat_1ep}"

export LITE3R_QUANT_MODE="${LITE3R_QUANT_MODE:-fp8_weight_only}"
export LITE3R_SAGE_ATTN="${LITE3R_SAGE_ATTN:-1}"
export LITE3R_SAGE_SMOOTH_K="${LITE3R_SAGE_SMOOTH_K:-1}"

run_vggt() {
  local ckpt="$CKPT_ROOT/vggt/last.pt"
  if [ "$CKPT_ROOT" = "outputs" ]; then
    ckpt="outputs/vggt_fp8_qat_1ep/last.pt"
  fi
  test -f "$ckpt" || { echo "[final-eval] missing VGGT ckpt: $ckpt" >&2; exit 1; }
  "$PY" eval/eval_vggt.py \
    --config configs/final/vggt_eval_blended.yaml \
    --ckpt "$ckpt" \
    --name final_vggt_fp8qat_blended
  "$PY" eval/eval_vggt.py \
    --config configs/final/vggt_eval_dtu64.yaml \
    --ckpt "$ckpt" \
    --name final_vggt_fp8qat_dtu64
}

run_da3() {
  local ckpt="$CKPT_ROOT/da3/last.pt"
  if [ "$CKPT_ROOT" = "outputs" ]; then
    ckpt="outputs/da3_fp8_qat_1ep/last.pt"
  fi
  test -f "$ckpt" || { echo "[final-eval] missing DA3 ckpt: $ckpt" >&2; exit 1; }
  "$PY" eval/eval_da3.py \
    --config configs/final/da3_eval_blended.yaml \
    --ckpt "$ckpt" \
    --name final_da3_fp8qat_blended
  "$PY" eval/eval_da3.py \
    --config configs/final/da3_eval_dtu64.yaml \
    --ckpt "$ckpt" \
    --name final_da3_fp8qat_dtu64
}

case "$MODEL" in
  all) run_vggt; run_da3 ;;
  vggt) run_vggt ;;
  da3) run_da3 ;;
  *) echo "usage: $0 [all|vggt|da3]" >&2; exit 2 ;;
esac

jsons=(
  outputs/final_eval_vggt_fp8_qat/final_vggt_fp8qat_blended.json
  outputs/final_eval_vggt_fp8_qat/final_vggt_fp8qat_dtu64.json
  outputs/final_eval_da3_fp8_qat/final_da3_fp8qat_blended.json
  outputs/final_eval_da3_fp8_qat/final_da3_fp8qat_dtu64.json
)
"$PY" scripts/collect_eval_reports.py --out outputs/final_fp8_qat_eval "${jsons[@]}"
