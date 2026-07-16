# HIRE-v2 锚定式完整观测基线：运行说明

**实验版本：** `v16.1.0`

## 1. 基线提交

本覆盖包以以下主分支提交为基线：

```text
610c2a405aec4acfdb0d6364872ec4f86d17c588
```

建议先确认仓库版本：

```bash
git rev-parse HEAD
```

## 2. 解压覆盖

建议先创建独立分支：

```bash
git switch -c hire-v2-anchor
```

然后在仓库根目录解压：

```bash
unzip -o HIRE_v2_anchor_overlay_20260716.zip \
  -d /root/autodl-tmp/IRRA_light_baseline
```

该包不会替换数据集文件、已有日志或检查点。

## 3. 静态检查

```bash
python tools/hire_v2/audit_anchor.py
```

审计不仅检查文件和参数，还会验证：

```text
零初始化融合与归一化全局特征一致；
局部残差适配器获得有限且非零的梯度；
损失聚合与设计公式完全一致；
填充和特殊文本词元不进入细粒度 MLP/BatchNorm；
文档不存在公式控制字符损坏。
```

## 4. 单元测试

```bash
pytest -q tests/test_hire_v2_anchor_components.py
```

当前覆盖八项核心行为，包括损失聚合、非零梯度、文本无效词元隔离和零初始化融合。

## 5. 一轮烟测

```bash
CUDA_VISIBLE_DEVICES=0 \
DATA_ROOT=/root/autodl-tmp/datasets \
OUTPUT_DIR=/root/autodl-tmp/HIRE_v2_anchor_smoke \
PYTHON_BIN=/root/miniconda3/envs/irra190/bin/python \
bash run_hire_v2_anchor_smoke.sh
```

烟测必须满足：

```text
训练完成一轮
评测完成
保存 best.pth
所有损失为有限值
全局、细粒度和完整观测诊断均出现在日志中
局部残差范数有记录
无 CUDA OOM、NaN 或 Inf
```

## 6. 正式训练

```bash
CUDA_VISIBLE_DEVICES=0 \
DATA_ROOT=/root/autodl-tmp/datasets \
OUTPUT_DIR=/root/autodl-tmp/HIRE_v2_anchor_logs \
PYTHON_BIN=/root/miniconda3/envs/irra190/bin/python \
SEED=1 \
BATCH_SIZE=64 \
NUM_EPOCH=60 \
EXP_NAME=hire_v2_anchor_tagpedes_60e_seed1 \
bash run_hire_v2_anchor_4090_tag.sh
```

## 7. 离线三分支评测

训练完成后：

```bash
python tools/hire_v2/eval_anchor_components.py \
  --config-file <实验目录>/configs.yaml \
  --checkpoint <实验目录>/best.pth
```

输出：

```text
global：CLIP 全局分数
local：RDE 风格词元选择分数
observation：零初始化残差融合后的正式分数
```

结果默认保存到：

```text
<实验目录>/hire_v2_anchor_components.json
```

## 8. 正式结果选择

训练期间使用 `observation` 分数保存最佳检查点。

离线结果必须同时报告：

```text
R1
R5
R10
mAP
mINP
```

## 9. 旧模式兼容

以下旧模式不应受影响：

```text
IRRA 原模式
IRRA-light 全部模式
旧 HIRE 主版本
v16 支持包和 fast3 模式
```

新模式只有在显式传入以下参数时启用：

```bash
--hire_v2 --hire_v2_mode anchor
```
