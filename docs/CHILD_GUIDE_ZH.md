# 幼儿园级复现说明

这份说明给自己用，目标是：换一台新服务器后，不用猜，照着做就能跑起来。

## 0. 先看你在哪

进入项目：

```bash
cd coding
```

## 1. 创建环境

推荐用 conda：

```bash
conda env create -f environment.yml
conda activate l3rsla-fp8
```

如果在国内，先加镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com
export PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/
export GITHUB_MIRROR=https://ghproxy.com/
```

然后跑安装脚本：

```bash
SKIP_DATA=1 bash setup.sh
```

`SKIP_DATA=1` 的意思是：先不要下载数据集。

## 2. 放数据集

不要把数据集放进 Git。服务器上有数据集就用软链接。

BlendedMVS 应该长这样：

```text
coding/datasets/BlendedMVS_lowres/BlendedMVS
```

如果真实数据在 `/data/BlendedMVS`，就这样：

```bash
mkdir -p datasets/BlendedMVS_lowres
ln -s /data/BlendedMVS datasets/BlendedMVS_lowres/BlendedMVS
```

DTU64 应该长这样：

```text
coding/datasets/dtu64
```

如果真实数据在 `/data/dtu64`，就这样：

```bash
ln -s /data/dtu64 datasets/dtu64
```

## 3. 检查代码没坏

```bash
bash scripts/smoke_test.sh
```

这个不需要真实数据集。

## 4. 检查本地最终模型还在不在

```bash
sha256sum -c checkpoints/fp8_qat_1ep/SHA256SUMS
```

看到 `OK` 就说明本地模型没坏。

## 5. 直接评测最终模型

```bash
bash scripts/final_eval_fp8_qat.sh
```

会评测：

```text
VGGT + BlendedMVS
VGGT + DTU64
DA3 + BlendedMVS
DA3 + DTU64
```

结果会在：

```text
outputs/final_fp8_qat_eval.md
outputs/final_fp8_qat_eval.csv
```

## 6. 如果要重新训练

```bash
bash scripts/final_train_fp8_qat.sh
```

这个会先训练 stage-1，再训练 FP8-QAT 最终模型。跑完后模型在：

```text
outputs/vggt_fp8_qat_1ep/last.pt
outputs/da3_fp8_qat_1ep/last.pt
```

用新训练出来的模型评测：

```bash
CKPT_ROOT=outputs bash scripts/final_eval_fp8_qat.sh
```

## 7. 查历史实验结果

先看这份总清单：

```text
docs/EXPERIMENTS.md
../results/experiment_inventory.md
```

最重要的最终结果：

```text
../results/fp8_qat_eval.md
../results/fp8_ablation_20260503.md
```

CSV 用来画表：

```text
../results/fp8_qat_eval.csv
../results/fp8_ablation_20260503.csv
```
