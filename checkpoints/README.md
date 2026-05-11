# Checkpoints

This directory stores metadata and checksums for model weights.

## Model Weights Location

Model weights are stored separately in the `../weights/` directory:

|backbone|filename|purpose|
|---|---|---|
|VGGT|`vggt_fp8_qat_1ep.pt`|main VGGT Lite3R-FP8 checkpoint|
|DA3-Large|`da3_fp8_qat_1ep.pt`|main DA3 Lite3R-FP8 checkpoint|

Checksums are recorded in `fp8_qat_1ep/SHA256SUMS`.

## Download Weights

The model weights are hosted on Hugging Face:
- [Download from Hugging Face](https://huggingface.co/YOUR_ORG/lite3r-weights)

After downloading, place the `.pt` files in the `../weights/` directory.

## Regenerating

If the checkpoint files are missing, run:

```bash
cd coding
bash scripts/final_train_fp8_qat.sh
```

The script first creates the required SLA stage-1 checkpoints, then trains the
final FP8-QAT checkpoints.
