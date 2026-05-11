#!/usr/bin/env bash
#SBATCH --job-name=lite3r_all
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --output=outputs/slurm_%j.out
#SBATCH --error=outputs/slurm_%j.err
#
# Submit from the project root:
#   cd /path/to/coding
#   sbatch scripts/sbatch_run_all.sh
#
# Cluster-specific overrides (set on the command line, NOT in this file):
#   sbatch -p day --gres=gpu:tesla:1 scripts/sbatch_run_all.sh
#   sbatch --nodelist=ltu-hpc-1 scripts/sbatch_run_all.sh
#   PYTHON=/path/to/conda/envs/l3rsla/bin/python sbatch scripts/sbatch_run_all.sh
#
# China mirror support (set in the submit shell so SLURM inherits, OR pass
# via `sbatch --export=ALL,HF_ENDPOINT=https://hf-mirror.com,...`):
#   export HF_ENDPOINT=https://hf-mirror.com    # used by huggingface_hub
#   export GITHUB_MIRROR=https://ghproxy.com/   # only used by setup.sh

set -euo pipefail

# --- Resolve project root portably ---------------------------------------
# SLURM sets SLURM_SUBMIT_DIR to the dir from which sbatch was called; that's
# the project root. Outside SLURM, fall back to the script's own location.
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    HERE="$SLURM_SUBMIT_DIR"
else
    HERE="$(cd "$(dirname "$0")"/.. && pwd)"
fi
cd "$HERE"
mkdir -p outputs

# --- Resolve Python interpreter ------------------------------------------
# Priority: $PYTHON env var → currently activated conda env → system python3.
if [ -n "${PYTHON:-}" ]; then
    PY="$PYTHON"
elif [ -n "${CONDA_PREFIX:-}" ] && [ -x "$CONDA_PREFIX/bin/python" ]; then
    PY="$CONDA_PREFIX/bin/python"
else
    PY="$(command -v python3)"
fi
export PYTHON="$PY"
export PYTHONPATH="$HERE${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"

echo "==[ slurm ${SLURM_JOB_ID:-local} @ $(hostname) start $(date -Is) ]=="
echo "[env] HERE=$HERE"
echo "[env] PYTHON=$PY"
[ -n "${HF_ENDPOINT:-}" ] && echo "[env] HF_ENDPOINT=$HF_ENDPOINT"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
"$PY" -c "import torch; print(f'torch={torch.__version__} cuda={torch.cuda.is_available()} bf16={torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False}')"

bash scripts/run_all_resume.sh 2>&1
echo "==[ slurm ${SLURM_JOB_ID:-local} end $(date -Is) ]=="
