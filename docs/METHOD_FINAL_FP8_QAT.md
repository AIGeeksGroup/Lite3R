# Final Method: Lite3R-FP8

This document describes the implemented route, not a speculative method.

## Overview

The final route combines three components:

```text
Sparse Linear Attention + FP8-aware QAT + partial attention-output distillation
```

It is implemented for both VGGT and DA3-Large.

## Sparse Linear Attention

Dense attention blocks are replaced by `SLAAttention` in `lite3r_kit/sla.py`.
The trainable lightweight branch is the SLA residual projection, named
`proj_lin`. Final configs train only these parameters while the pretrained
backbone remains frozen.

Important configs:

```text
configs/final/vggt_stage1_sla_w4a16.yaml
configs/final/da3_stage1_sla_w4a8.yaml
configs/final/vggt_fp8_qat_1ep.yaml
configs/final/da3_fp8_qat_1ep.yaml
```

## FP8-Aware QAT

FP8 QAT is implemented in `lite3r_kit/fp8_fake_quant.py` through
`FP8FakeQuantLinear`.

The implementation applies:

- E4M3-style fake quantization for Linear weights and activations.
- per-output-row dynamic scaling for weights.
- per-token dynamic scaling for activations.
- straight-through estimator gradients.

On A100, final deployment uses torchao FP8 weight-only conversion through
`LITE3R_QUANT_MODE=fp8_weight_only`. Native dynamic FP8 activation inference was
not available on the tested A100 SM80 stack.

## Partial Attention Distillation

Distillation is implemented in `lite3r_kit/distillation.py`.

This is not output, depth, pose, or point-cloud distillation. The student and
frozen dense teacher expose attention-block outputs through hooks, and the loss
adds a partial attention-output MSE term. The default final route uses KD gamma
`0.1`.

## Trainable Scope

Final training does not fine-tune the full backbone. The code freezes the
pretrained backbone and trains only the selected lightweight parameters, mainly
SLA `proj_lin`. This is important for how the method should be written in the
paper.

## Evaluation

Final eval configs:

```text
configs/final/vggt_eval_blended.yaml
configs/final/da3_eval_blended.yaml
configs/final/vggt_eval_dtu64.yaml
configs/final/da3_eval_dtu64.yaml
```

Convenience command:

```bash
cd coding
bash scripts/final_eval_fp8_qat.sh
```

The generated outputs are written to `outputs/final_eval_*`.
