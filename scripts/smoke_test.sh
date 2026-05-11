#!/usr/bin/env bash
# Local smoke test using the dummy dataset.
#
# Purpose: validate that import paths, model construction, SLA / W4A4 swaps,
# loss adapters, and one full train + eval iteration work end-to-end on a
# small (12GB) GPU. Do NOT run real training here.

set -euo pipefail
HERE="$(cd "$(dirname "$0")"/.. && pwd)"
cd "$HERE"
export PYTHONPATH="$HERE${PYTHONPATH:+:$PYTHONPATH}"

PY="${PYTHON:-python3}"

# Override every config to use the dummy dataset for one epoch / one step.
DUMMY_FLAGS='--config /dev/stdin'

run_train() {
  CFG="$1"
  VARIANT="$2"
  SCRIPT="$3"
  cat > /tmp/_smoke_cfg.yaml <<YML
variant: $VARIANT
output_dir: outputs/smoke_${VARIANT}_${CFG%.*}
model: { img_size: 112, patch_size: 14, embed_dim: 1024, model_name: da3-small,
         keep_ratio: 0.2, lambda_init: 0.5, enable_kd: false,
         load_pretrained: false }
data:  { root: __none__, use_dummy: true, dummy_len: 2, img_per_seq: 2, img_size: 112, batch_size: 1, num_workers: 0 }
loss:  { camera: {weight: 1.0}, depth: {weight: 1.0, gradient_loss_fn: ""} }
optim: { epochs: 1, lr: 1.0e-4, weight_decay: 0.05, warmup_steps: 0, grad_clip: 1.0 }
log_every: 1
save_every: 1000
YML
  $PY train/${SCRIPT}.py --config /tmp/_smoke_cfg.yaml
}

echo "==[ smoke: VGGT original ]=="
run_train vggt_original original train_vggt

echo "==[ smoke: VGGT lite stage1 ]=="
run_train vggt_lite_stage1 lite_stage1 train_vggt

echo "==[ smoke: DA3 original ]=="
run_train da3_original original train_da3

echo "==[ smoke: DA3 lite stage1 ]=="
run_train da3_lite_stage1 lite_stage1 train_da3

echo "==[ smoke OK ]=="
