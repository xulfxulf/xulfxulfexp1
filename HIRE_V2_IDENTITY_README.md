# HIRE-v2 版本二：身份随机效应工程说明

本覆盖包基于仓库主分支提交：

```text
09def7e11fe3a2f47b39929013aaad4038b98ac9
```

版本一正式模型代码提交为：

```text
48e61f81649aa2f3ea515d8e967faa4960b2f478
```

## 本版本只新增什么

```text
同身份、不同 image_id 的动态支持图；
共享身份均值映射；
逐维图像不确定性；
观测内不确定性 + 观测间异质性的可信交集；
严格留一的文本到图像身份组监督；
以版本一完整观测为锚点的有界身份残差。
```

本版本不包含状态头、支持文本、身份分类器、困难负包、教师标签或方差校准回归。

## 解压

```bash
cd /root/autodl-tmp/IRRA_light_baseline
unzip -o HIRE_v2_identity_overlay_20260716.zip \
  -d /root/autodl-tmp/IRRA_light_baseline
```

## 静态审计

```bash
python tools/hire_v2/audit_identity.py
```

## 单元测试

```bash
pytest -q \
  tests/test_hire_v2_anchor_components.py \
  tests/test_hire_v2_identity_components.py
```

## 一轮烟测

```bash
CUDA_VISIBLE_DEVICES=0 \
DATA_ROOT=/root/autodl-tmp/datasets \
OUTPUT_DIR=/root/autodl-tmp/HIRE_v2_identity_smoke \
PYTHON_BIN=/root/miniconda3/envs/irra190/bin/python \
NUM_EPOCH=1 \
SEED=1 \
BATCH_SIZE=64 \
SUPPORT_SIZE=3 \
AUX_WEIGHT=0.1 \
bash run_hire_v2_identity_smoke.sh
```

烟测日志必须出现：

```text
identity_group_loss
identity_group_nce
identity_gate
identity_score_delta_abs
observation_identity_cosine
identity_projection_delta_norm
support_valid_ratio
support_count_mean
mean_image_variance
variance_low_ratio
variance_high_ratio
mean_group_heterogeneity
```

## 正式 60 轮训练

```bash
CUDA_VISIBLE_DEVICES=0 \
DATA_ROOT=/root/autodl-tmp/datasets \
OUTPUT_DIR=/root/autodl-tmp/HIRE_v2_identity_logs \
PYTHON_BIN=/root/miniconda3/envs/irra190/bin/python \
NUM_EPOCH=60 \
SEED=1 \
BATCH_SIZE=64 \
SUPPORT_SIZE=3 \
AUX_WEIGHT=0.1 \
EXP_NAME=hire_v2_identity_tagpedes_60e_seed1 \
bash run_hire_v2_identity_4090_tag.sh
```

## 最佳检查点组件评测

```bash
python tools/hire_v2/eval_identity_components.py \
  --config-file <实验目录>/configs.yaml \
  --checkpoint <实验目录>/best.pth
```

输出五套分数：

```text
global
local
observation
identity
final
```

默认保存：

```text
<实验目录>/hire_v2_identity_components.json
```

## 旧版本兼容

版本一仍使用：

```bash
--hire_v2 --hire_v2_mode anchor
```

版本二使用：

```bash
--hire_v2 --hire_v2_mode identity
```

旧 IRRA、IRRA-light、支持包系列和完整 HIRE 路径均不修改其模式名称与前向逻辑。
