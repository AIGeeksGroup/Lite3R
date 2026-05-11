# Lite3R-FP8 Experiment Inventory

Last updated: 2026-05-03.

This is the reproducibility-facing copy of `../../results/experiment_inventory.md`. It records what experiments have already been run and where to find their outputs.

## Main Route

```text
SLA + FP8-aware QAT + partial attention-output distillation
```

Main local checkpoints:

|backbone|checkpoint|
|---|---|
|VGGT|`../checkpoints/fp8_qat_1ep/vggt/last.pt`|
|DA3-Large|`../checkpoints/fp8_qat_1ep/da3/last.pt`|

Deployment/evaluation environment:

```bash
export LITE3R_QUANT_MODE=fp8_weight_only
export LITE3R_SAGE_ATTN=1
export LITE3R_SAGE_SMOOTH_K=1
```

On A100 SM80, the final tested path is FP8-aware fake-quant training plus FP8 weight-only deployment. Native dynamic FP8 activation inference was not available on this hardware.

## Experiments

|id|experiment|summary files|detailed artifacts|
|---|---|---|---|
|E0 Full candidate route matrix|Original FP backbones plus BF16, W4A8, W8A8, INT4W, BlockSparse, TopK and related probes on VGGT/DA3 x BlendedMVS/DTU64.|`../../results/full_lite_eval_report.md`, `../../results/full_lite_eval_results_compact.csv`|Report includes the complete matrix and failed/skipped probes.|
|E1 BF16/SAGE/compile last-bet|Latency-oriented BF16 QAT-lite + SAGE + `torch.compile` + CUDA-graph step marker.|`../../results/bf16_sage_compile_last_bet.md`, `../../results/bf16_sage_compile_last_bet.csv`|Server JSON paths are listed in the report.|
|E2 FP8 weight-only deployment probe|torchao FP8 weight-only deployment with BF16 activations and SAGE; also checked A100 FP8 kernel limits.|`../../results/fp8_weight_only_eval.md`, `../../results/fp8_weight_only_eval.csv`|Server JSON paths are listed in the report.|
|E3 FP8-aware QAT 1ep main probe|Main FP8 route: SLA + FP8 fake-quant QAT + attention-output KD, evaluated through FP8 weight-only deployment.|`../../results/fp8_qat_eval.md`, `../../results/fp8_qat_eval.csv`|Local checkpoints in `../checkpoints/fp8_qat_1ep/`; server JSON paths are listed in the report.|
|E4 FP8-QAT 20ep training|20-epoch flagship training completed, but checkpoints were not copied locally because of size.|`../../results/fp8_qat_20ep_flagship_status.md`|Logs/configs: `../../results/server_artifacts_20260503/logs/`, `../../results/server_artifacts_20260503/configs/`.|
|E5 FP8-QAT 20ep evaluation|20ep checkpoints evaluated on both datasets; results showed geometry drift, so they are not main paper checkpoints.|`../../results/fp8_qat_20ep_flagship_eval.md`, `../../results/fp8_qat_20ep_flagship_eval.csv`|Full artifacts: `../../results/server_artifacts_20260503/fp8_qat_20ep_eval_artifacts/`.|
|E6 Raw FP8 ablation suite|First component and KD-gamma ablation run. no-QAT rows still used FP8 weight-only deployment, so use only as raw history.|`../../results/fp8_ablation_20260503.md`, `../../results/fp8_ablation_20260503.csv`|Full artifacts: `../../results/server_artifacts_20260503/fp8_ablation_20260503/`.|
|E7 Clean FP8 ablation suite|Paper-facing rerun. FP8-QAT rows use FP8 weight-only + SAGE; no-QAT rows use no deployment quantization and no SAGE.|`../../results/fp8_ablation_clean_20260503.md`, `../../results/fp8_ablation_clean_20260503.csv`|Full artifacts: `../../results/server_artifacts_20260503/fp8_ablation_clean_20260503/`.|

## Practical Pointers

- Use CSV files for plotting and numerical post-processing.
- Use Markdown reports for interpretation and paper-writing context.
- DTU64 is pose-only in this local setup, so depth and Chamfer metrics are not expected there.
- Final reproducible configs live in `../configs/final/`.
- Final training/evaluation wrappers are `../scripts/final_train_fp8_qat.sh` and `../scripts/final_eval_fp8_qat.sh`.
