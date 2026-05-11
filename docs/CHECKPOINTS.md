# Checkpoints And Reproduction

## Final Local Weights

The final paper-facing 1ep FP8-QAT weights are included locally:

|backbone|path|
|---|---|
|VGGT|`../checkpoints/fp8_qat_1ep/vggt/last.pt`|
|DA3-Large|`../checkpoints/fp8_qat_1ep/da3/last.pt`|

Verify them from `coding/`:

```bash
sha256sum -c checkpoints/fp8_qat_1ep/SHA256SUMS
```

Expected hashes:

```text
aa4314350c3400d50c9b06ebff524205049654a4189a0d4fcbe9720b00bcf074  checkpoints/fp8_qat_1ep/vggt/last.pt
9cfacfa89b28c54f573327063b8782cadc6b71e6a344950259a3435746526813  checkpoints/fp8_qat_1ep/da3/last.pt
```

## Original Server Paths

These were the source paths on the AutoDL server:

```text
/root/autodl-tmp/lite3r/coding/outputs/vggt_fp8_qat_1ep/last.pt
/root/autodl-tmp/lite3r/coding/outputs/da3_fp8_qat_1ep/last.pt
```

## If We Need To Recreate Them

```bash
cd coding
bash scripts/final_train_fp8_qat.sh
```

This creates:

```text
outputs/vggt_keep03_r2a_qat/last.pt
outputs/da3_keep03_r2a_qat_w4a8/last.pt
outputs/vggt_fp8_qat_1ep/last.pt
outputs/da3_fp8_qat_1ep/last.pt
```

Then evaluate those newly trained checkpoints with:

```bash
CKPT_ROOT=outputs bash scripts/final_eval_fp8_qat.sh
```
