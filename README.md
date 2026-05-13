# Lite3R: A Model-Agnostic Framework for Efficient Feed-Forward 3D Reconstruction

Official implementation of **Lite3R**, a model-agnostic framework for efficient feed-forward 3D reconstruction from multi-view images.

> **Lite3R: A Model-Agnostic Framework for Efficient Feed-Forward 3D Reconstruction**
>
> Haoyu Zhang\*, [Zeyu Zhang](https://steve-zeyu-zhang.github.io/)\*<sup>†</sup>, Zedong Zhou, Yang Zhao, and [Hao Tang](https://ha0tang.github.io/)<sup>#</sup>
>
> \*Equal contribution. <sup>†</sup>Project lead. <sup>#</sup>Corresponding author.
>
> ### [Paper](https://arxiv.org/abs/2605.11354) | [Website](https://aigeeksgroup.github.io/Lite3R/) | [Models](https://huggingface.co/AIGeeksGroup/Lite3R) | [HF Paper](https://huggingface.co/papers/)

<img width="2953" height="827" alt="mainfig_page-0001" src="https://github.com/user-attachments/assets/7667ae19-976f-4956-8b0c-038d9da88ab2" />


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

- `vggt_fp8_qat_1ep.pt` - Lite3R VGGT
- `da3_fp8_qat_1ep.pt` - Lite3R DA3-L

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
@article{zhang2026lite3r,
  title={Lite3R: A Model-Agnostic Framework for Efficient Feed-Forward 3D Reconstruction},
  author={Zhang, Haoyu and Zhang, Zeyu and Zhou, Zedong and Zhao, Yang and Tang, Hao},
  journal={arXiv preprint arXiv:2605.11354},
  year={2026}
}
```

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgments

This work builds upon [VGGT](https://github.com/naver/vggt) and [Depth Anything V3](https://github.com/DepthAnything/Depth-Anything-V3).
