#!/usr/bin/env bash
# Run the full 4-experiment training + evaluation suite.
#
# Lite models are trained as Stage-1 (SLA) → Stage-2 (W4A4 QAT). Stage-2
# automatically initialises from Stage-1's last.pt.
#
# Outputs:
#   outputs/{vggt_original, vggt_lite_stage1, vggt_lite_stage2,
#            da3_original,  da3_lite_stage1,  da3_lite_stage2}/last.pt
#   outputs/eval_{vggt,da3}/{*.json}

set -euo pipefail
HERE="$(cd "$(dirname "$0")"/.. && pwd)"
cd "$HERE"
export PYTHONPATH="$HERE${PYTHONPATH:+:$PYTHONPATH}"

PY="${PYTHON:-python3}"

train() { echo "==[ TRAIN $1 ]=="; $PY train/$2.py --config "configs/$1.yaml"; }
eval_run() { echo "==[ EVAL $1 ]=="; $PY eval/$2.py --config "configs/$1.yaml" --ckpt "outputs/$1/last.pt" --name "$1"; }

# 1. VGGT
train vggt_original     train_vggt
eval_run vggt_original  eval_vggt

train vggt_lite_stage1  train_vggt
train vggt_lite_stage2  train_vggt
eval_run vggt_lite_stage2 eval_vggt

# 2. DA3
train da3_original      train_da3
eval_run da3_original   eval_da3

train da3_lite_stage1   train_da3
train da3_lite_stage2   train_da3
eval_run da3_lite_stage2 eval_da3

# 3. Aggregated CSV (recursive glob across the actual eval_*.json shape)
$PY - <<'PY'
import json, glob, os, csv
rows, seen = [], set()
for fp in sorted(glob.glob("outputs/**/*.json", recursive=True)):
    try:
        with open(fp) as f: r = json.load(f)
    except Exception:
        continue
    if not (isinstance(r, dict) and "metrics" in r and "efficiency" in r):
        continue
    name = os.path.splitext(os.path.basename(fp))[0]
    if name in seen:
        continue
    seen.add(name)
    rows.append({"name": name, **r["metrics"], **r["efficiency"]})
fields = ["name"] + sorted({k for r in rows for k in r if k != "name"})
with open("outputs/summary.csv", "w") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for r in rows:
        w.writerow(r)
print(f"wrote outputs/summary.csv with {len(rows)} rows")
PY

echo "==[ DONE ]=="
