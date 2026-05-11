#!/usr/bin/env bash
# Lite3R from-scratch setup for a freshly-provisioned server.
#
# Usage:
#   bash setup.sh                   # use whatever python3 is on PATH
#   PYTHON=/path/to/python setup.sh # use a specific interpreter
#   SKIP_DATA=1 bash setup.sh       # skip BlendedMVS download (already there)
#   SKIP_DEPS=1 bash setup.sh       # skip pip install (env already prepped)
#
# China mirror support (set BEFORE running):
#   export HF_ENDPOINT=https://hf-mirror.com           # huggingface mirror
#   export PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/  # pip mirror
#   export GITHUB_MIRROR=https://ghproxy.com/          # GitHub releases proxy
#                                  # (note the trailing slash; final URL becomes
#                                  #  ${GITHUB_MIRROR}https://github.com/...)
#
# Recommended: create a fresh conda env first via the provided
# environment.yml (see README Quick Start).
#
# Idempotent: re-running skips already-completed steps.

set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

echo "==[ Lite3R from-scratch setup @ $HERE ]=="
[ -n "${HF_ENDPOINT:-}" ]    && echo "[setup] HF_ENDPOINT=$HF_ENDPOINT"
[ -n "${PIP_INDEX_URL:-}" ]  && echo "[setup] PIP_INDEX_URL=$PIP_INDEX_URL"
[ -n "${GITHUB_MIRROR:-}" ]  && echo "[setup] GITHUB_MIRROR=$GITHUB_MIRROR"

# --- 1. Resolve Python interpreter ---------------------------------------
if [ -n "${PYTHON:-}" ]; then
    PY="$PYTHON"
elif [ -n "${CONDA_PREFIX:-}" ] && [ -x "$CONDA_PREFIX/bin/python" ]; then
    PY="$CONDA_PREFIX/bin/python"
else
    PY="$(command -v python3)"
fi
echo "[setup] PYTHON=$PY"
"$PY" -c "import sys; assert sys.version_info >= (3,9), f'need Python 3.9+, have {sys.version}'"

# --- 2. Install pip deps -------------------------------------------------
if [ -z "${SKIP_DEPS:-}" ]; then
    echo "[setup] installing pip deps (this may take a few minutes)"
    "$PY" -m pip install --upgrade pip
    "$PY" -m pip install -r "$HERE/requirements.txt"
    # NOTE: we intentionally DO NOT `pip install -e` the model_*/Original|Lite
    # folders. Both VGGT/Original and VGGT/Lite declare `name = "vggt"` in
    # their pyproject.toml; installing both would clobber each other. Training
    # scripts manage sys.path explicitly via `add_paths(...)` and pick the
    # right Original/Lite tree depending on `variant`.
else
    echo "[setup] SKIP_DEPS=1 — skipping pip install"
fi

# --- 3. CUDA / GPU sanity check ------------------------------------------
"$PY" - <<'PYCHK' || true
import torch
print(f"[setup] torch={torch.__version__}, cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    cap = torch.cuda.get_device_capability(0)
    name = torch.cuda.get_device_name(0)
    print(f"[setup] GPU: {name}  sm_{cap[0]}{cap[1]}  bf16={torch.cuda.is_bf16_supported()}")
    if cap[0] < 8:
        print("[setup] WARNING: final FP8 weight-only deployment was validated on sm_80+. "
              "This GPU may run but latency/memory numbers will not match the paper route.")
    if cap < (8, 9):
        print("[setup] NOTE: native dynamic FP8 activation kernels are not expected here; "
              "A100 uses FP8-aware QAT plus FP8 weight-only deployment.")
else:
    print("[setup] WARNING: no CUDA visible; training will fall back to CPU.")
PYCHK

# --- 4. Download BlendedMVS Low-Res --------------------------------------
DATA_DIR="$HERE/datasets/BlendedMVS_lowres"
if [ -n "${SKIP_DATA:-}" ]; then
    echo "[setup] SKIP_DATA=1 — skipping dataset download"
elif [ -f "$DATA_DIR/BlendedMVS/_DOWNLOAD_OK" ]; then
    echo "[setup] BlendedMVS already present at $DATA_DIR/BlendedMVS — skipping"
else
    echo "[setup] downloading BlendedMVS Low-Res (16 parts ≈ 30GB)"
    mkdir -p "$DATA_DIR"
    cd "$DATA_DIR"
    GH_PREFIX="${GITHUB_MIRROR:-}"
    base="${GH_PREFIX}https://github.com/YoYo000/BlendedMVS/releases/download/v1.0.0"
    [ -n "$GH_PREFIX" ] && echo "[setup] using GitHub mirror prefix: $GH_PREFIX"
    for i in $(seq -w 1 15); do
        f="BlendedMVS.z${i}"
        # part files are ~1.95GB each; resume if partial
        if [ ! -f "$f" ] || [ "$(stat -c%s "$f" 2>/dev/null || echo 0)" -lt 1900000000 ]; then
            echo "  → fetching $f"
            curl -fL --retry 5 --retry-delay 5 -C - -o "$f" "$base/$f"
        fi
    done
    if [ ! -f "BlendedMVS.zip" ] || [ "$(stat -c%s BlendedMVS.zip 2>/dev/null || echo 0)" -lt 250000000 ]; then
        echo "  → fetching BlendedMVS.zip (last part)"
        curl -fL --retry 5 --retry-delay 5 -C - -o "BlendedMVS.zip" "$base/BlendedMVS.zip"
    fi
    echo "[setup] concatenating 16 parts → BlendedMVS.combined.zip"
    cat BlendedMVS.z01 BlendedMVS.z02 BlendedMVS.z03 BlendedMVS.z04 \
        BlendedMVS.z05 BlendedMVS.z06 BlendedMVS.z07 BlendedMVS.z08 \
        BlendedMVS.z09 BlendedMVS.z10 BlendedMVS.z11 BlendedMVS.z12 \
        BlendedMVS.z13 BlendedMVS.z14 BlendedMVS.z15 BlendedMVS.zip \
        > BlendedMVS.combined.zip
    echo "[setup] extracting (~70k files, this takes a few minutes) …"
    unzip -qq BlendedMVS.combined.zip
    rm -f BlendedMVS.z* BlendedMVS.zip BlendedMVS.combined.zip
    touch BlendedMVS/_DOWNLOAD_OK
    cd "$HERE"
fi

# --- 5. Final summary ----------------------------------------------------
echo
echo "==[ setup complete ]=="
echo "Project root : $HERE"
echo "Python       : $PY"
echo "Dataset      : $DATA_DIR/BlendedMVS  ($(ls -1 "$DATA_DIR/BlendedMVS" 2>/dev/null | wc -l) entries)"
echo
echo "Next:"
echo "  bash scripts/smoke_test.sh                # validate on dummy data (~1 min)"
echo "  bash scripts/final_eval_fp8_qat.sh        # evaluate packaged final checkpoints"
echo "  bash scripts/final_train_fp8_qat.sh       # retrain final VGGT + DA3 route"
