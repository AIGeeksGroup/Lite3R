# Lite3R-FP8

Reproducible code package for the final lightweight 3D-geometry route:

**Sparse Linear Attention (SLA) + FP8-aware QAT + partial attention-output
distillation**, evaluated on **VGGT** and **Depth Anything 3 Large (DA3-Large)**.

The package is prepared so a fresh single-GPU server can be brought up quickly
after the current AutoDL instance is shut down. Dataset files are intentionally
not included; see [docs/DATASETS.md](docs/DATASETS.md) for expected layouts and
download links. Final local checkpoints are stored under
`checkpoints/fp8_qat_1ep/`.

## Final Route

The paper-facing route is:

- Backbone: VGGT-1B or DA3-Large.
- Attention: dense attention blocks are replaced by `SLAAttention`.
- Quantization-aware training: FP8 E4M3 fake quantization is applied to Linear
  weights and activations with straight-through gradients.
- Distillation: frozen dense teacher, attention-output MSE only; this is partial
  attention distillation, not output/depth/pose distillation.
- Trainable scope: only SLA `proj_lin` parameters are trained in the final
  lightweight stage; the pretrained backbone is frozen.
- Deployment evaluation: trained fake-quant Linear layers are unwrapped and
  converted with torchao `Float8WeightOnlyConfig` using
  `LITE3R_QUANT_MODE=fp8_weight_only`. A100 does not run full native FP8
  activation inference in this code path; the claim is FP8-aware QAT plus FP8
  weight-only deployment.

Main configs:

- [configs/final/vggt_fp8_qat_1ep.yaml](configs/final/vggt_fp8_qat_1ep.yaml)
- [configs/final/da3_fp8_qat_1ep.yaml](configs/final/da3_fp8_qat_1ep.yaml)
- [configs/final/vggt_eval_blended.yaml](configs/final/vggt_eval_blended.yaml)
- [configs/final/da3_eval_blended.yaml](configs/final/da3_eval_blended.yaml)
- [configs/final/vggt_eval_dtu64.yaml](configs/final/vggt_eval_dtu64.yaml)
- [configs/final/da3_eval_dtu64.yaml](configs/final/da3_eval_dtu64.yaml)

## Quick Start on a Fresh Server

```bash
cd coding

# Optional China-mainland mirrors.
export HF_ENDPOINT=https://hf-mirror.com
export PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/
export GITHUB_MIRROR=https://ghproxy.com/

# Environment.
conda env create -f environment.yml
conda activate l3rsla-fp8
SKIP_DATA=1 bash setup.sh

# Put datasets at the documented paths, or symlink them:
#   datasets/BlendedMVS_lowres/BlendedMVS
#   datasets/dtu64

# Fast check without datasets.
bash scripts/smoke_test.sh

# Evaluate the included final checkpoints.
bash scripts/final_eval_fp8_qat.sh
```

The final checkpoints included locally are:

```text
checkpoints/fp8_qat_1ep/vggt/last.pt
checkpoints/fp8_qat_1ep/da3/last.pt
```

Their SHA256 hashes are recorded in
[checkpoints/fp8_qat_1ep/SHA256SUMS](checkpoints/fp8_qat_1ep/SHA256SUMS).

## Reproducing Training

The final FP8-QAT stage depends on a short SLA stage-1 checkpoint. To train
everything again on BlendedMVS low-res:

```bash
cd coding
bash scripts/final_train_fp8_qat.sh
```

The script trains missing stage-1 checkpoints first:

- `outputs/vggt_keep03_r2a_qat/last.pt`
- `outputs/da3_keep03_r2a_qat_w4a8/last.pt`

Then it trains:

- `outputs/vggt_fp8_qat_1ep/last.pt`
- `outputs/da3_fp8_qat_1ep/last.pt`

For evaluation from newly trained checkpoints instead of packaged checkpoints:

```bash
CKPT_ROOT=outputs bash scripts/final_eval_fp8_qat.sh
```

## Expected Hardware

Validated server:

- NVIDIA A100-PCIE-40GB
- Python 3.10
- PyTorch 2.11.0 + CUDA 12.6
- torchao 0.11.0

`fp8_weight_only` requires a recent torchao with `Float8WeightOnlyConfig`. On
older torchao versions the eval script may fall back or fail before producing
valid FP8-weight deployment numbers.

## Project Layout

```text
configs/final/          final train/eval YAMLs
checkpoints/            local final checkpoints and checksums
lite3r_kit/             SLA, FP8 fake quant, distillation, deployment kernels
train/                  train_vggt.py, train_da3.py
eval/                   eval_vggt.py, eval_da3.py, metrics
data/                   BlendedMVS and DTU64 loaders
scripts/                final train/eval scripts and experiment utilities
docs/                   dataset, checkpoint, method, and beginner guides
model_VGGT/             Original and Lite source trees
model_DA3-Large/        Original and Lite source trees
```

Legacy W4A4 notes from earlier exploration were moved to
[README_LEGACY_W4A4.md](README_LEGACY_W4A4.md). The current paper route should
use the FP8 files above.
