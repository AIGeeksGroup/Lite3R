# Lite3R: A Model-Agnostic Framework for Efficient Feed-Forward 3D Reconstruction

Official implementation of **Lite3R**, a model-agnostic framework for efficient feed-forward 3D reconstruction from multi-view images.

## Overview

Lite3R introduces a systematic approach to compress large-scale 3D reconstruction models while maintaining reconstruction quality. The framework combines:

- **Sparse Linear Attention (SLA)**: Efficient attention mechanism that reduces computational complexity
- **FP8-Aware Quantization-Aware Training (QAT)**: Low-precision training for deployment efficiency
- **Partial Attention Distillation**: Knowledge transfer from dense teacher models

The framework has been validated on two state-of-the-art architectures:
- **VGGT** (Visual Geometry Grounding Transformer)
- **Depth Anything V3 Large (DA3-L)**

## Installation

```bash
# Clone the repository
git clone https://github.com/AIGeeksGroup/Lite3R.git
cd Lite3R

# Create conda environment
conda create -n lite3r python=3.10
conda activate lite3r

# Install dependencies
pip install -r requirements.txt
```

## Model Checkpoints

Pre-trained model weights are available on [Hugging Face](https://huggingface.co/AIGeeksGroup/Lite3R):

- `vggt_fp8_qat_1ep.pt` - VGGT with FP8 QAT
- `da3_fp8_qat_1ep.pt` - DA3-L with FP8 QAT

Download and place checkpoints in `checkpoints/fp8_qat_1ep/`.

## Quick Start

### Inference

```bash
python inference.py \
  --model vggt \
  --checkpoint checkpoints/fp8_qat_1ep/vggt/vggt_fp8_qat_1ep.pt \
  --input_dir examples/input \
  --output_dir examples/output
```

### Training

```bash
python train.py --config configs/final/vggt_fp8_qat_1ep.yaml
```

## Evaluation

Evaluate on BlendedMVS or DTU datasets:

```bash
python eval.py \
  --config configs/final/vggt_eval_blended.yaml \
  --checkpoint checkpoints/fp8_qat_1ep/vggt/vggt_fp8_qat_1ep.pt
```

## Citation

If you find this work useful, please cite:

```bibtex
@article{lite3r2025,
  title={Lite3R: A Model-Agnostic Framework for Efficient Feed-Forward 3D Reconstruction},
  author={Your Name and Collaborators},
  journal={arXiv preprint},
  year={2025}
}
```

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgments

This work builds upon [VGGT](https://github.com/naver/vggt) and [Depth Anything V3](https://github.com/DepthAnything/Depth-Anything-V3).
