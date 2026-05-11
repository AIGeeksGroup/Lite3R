#!/usr/bin/env bash
# Resume-aware version of run_all.sh.
#
# - Skips any train_X step whose outputs/<name>/last.pt already exists.
# - Includes vggt/da3 pretrained-only references first (epochs=0 → captures
#   bare HF weights so the ratio Original / Lite / pretrained-only is visible
#   in the final summary.csv).

set -euo pipefail
HERE="$(cd "$(dirname "$0")"/.. && pwd)"
cd "$HERE"
export PYTHONPATH="$HERE${PYTHONPATH:+:$PYTHONPATH}"

PY="${PYTHON:-python3}"

train() {
  local name="$1" script="$2"
  if [ -s "outputs/$name/last.pt" ]; then
    echo "==[ SKIP TRAIN $name (already have last.pt) ]=="
    return 0
  fi
  echo "==[ TRAIN $name ]=="
  $PY train/$script.py --config "configs/$name.yaml"
}
eval_run() {
  local name="$1" script="$2"
  echo "==[ EVAL $name ]=="
  $PY eval/$script.py --config "configs/$name.yaml" \
                      --ckpt "outputs/$name/last.pt" \
                      --name "$name"
}

# 0. Pretrained-only references (epochs=0 → save HF weights as last.pt → eval)
train vggt_pretrained_only train_vggt
eval_run vggt_pretrained_only eval_vggt
train da3_pretrained_only  train_da3
eval_run da3_pretrained_only  eval_da3

# 1. VGGT
train vggt_original     train_vggt
eval_run vggt_original   eval_vggt
train vggt_lite_stage1  train_vggt
train vggt_lite_stage2  train_vggt
eval_run vggt_lite_stage2 eval_vggt

# 2. DA3
train da3_original      train_da3
eval_run da3_original    eval_da3
train da3_lite_stage1   train_da3
train da3_lite_stage2   train_da3
eval_run da3_lite_stage2 eval_da3

# 3. Aggregated CSV
$PY - <<'PY'
import json, glob, os, csv
rows, seen = [], set()
for fp in sorted(glob.glob("outputs/**/*.json", recursive=True)):
    try:
        with open(fp) as f:
            r = json.load(f)
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
