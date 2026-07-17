# HIRE-v2 v16.2.1 工程使用说明

## 1. 版本

```text
实验版本：v16.2.1
模式：--hire_v2 --hire_v2_mode identity_balanced
源主分支：7d89aa311eda5aaef8b7f6f200e2cd47de015ad0
```

本包是覆盖式工程包，应解压到上述提交或其后保持兼容的主分支根目录。

## 2. 相对 v16.2.0 的修改

```text
保留：
动态同身份支持图；
严格留一身份组监督；
共享身份映射；
有界身份残差门；
测试阶段无支持检索。

修改：
恢复完整观测直接 SDM + ITC；
完整观测与最终分数主损失固定各占 0.5；
删除离线审计证明未启动的方差头；
删除异质性感知加权；
支持身份组改为明确的掩码简单均值。
```

## 3. 解压

```bash
cd /root/autodl-tmp/IRRA_light_baseline

git rev-parse HEAD

unzip -o HIRE_v2_identity_balanced_overlay_20260717.zip \
  -d /root/autodl-tmp/IRRA_light_baseline
```

## 4. 静态审计

```bash
python tools/hire_v2/audit_identity_balanced.py \
  --output-json /root/autodl-tmp/HIRE_v2_identity_balanced_audit.json
```

预期：

```text
status: passed
masked_group_consensus: passed
objective_formula: passed
identity_initialization: passed
initial_score_equivalence: passed
initial_gradient_equivalence: passed
inference_equivalence: passed
source_invariants: passed
```

## 5. 单元测试

```bash
pytest -q \
  tests/test_hire_v2_anchor_components.py \
  tests/test_hire_v2_identity_components.py \
  tests/test_hire_v2_identity_balanced_components.py
```

新版本测试单独运行：

```bash
pytest -q tests/test_hire_v2_identity_balanced_components.py
```

## 6. 一轮烟测

```bash
CUDA_VISIBLE_DEVICES=0 \
DATA_ROOT=/root/autodl-tmp/datasets \
OUTPUT_DIR=/root/autodl-tmp/HIRE_v2_identity_balanced_smoke \
PYTHON_BIN=/root/miniconda3/envs/irra190/bin/python \
NUM_EPOCH=1 \
SEED=1 \
BATCH_SIZE=64 \
SUPPORT_SIZE=3 \
AUX_WEIGHT=0.1 \
bash run_hire_v2_identity_balanced_smoke.sh
```

日志必须出现：

```text
observation_sdm
observation_itc
final_sdm
final_itc
balanced_main_objective
identity_group_nce
identity_gate
support_valid_ratio
identity_group_dispersion
identity_group_support_cosine
observation_main_weight: 0.5000
final_main_weight: 0.5000
```

烟测通过条件：

```text
一轮训练和评测完成；
best.pth 正常保存；
所有损失有限；
无 NaN、Inf、CUDA OOM；
有效支持组比例大于 0；
身份组损失可反向传播；
固定权重均为 0.5。
```

## 7. 正式 60 轮

```bash
CUDA_VISIBLE_DEVICES=0 \
DATA_ROOT=/root/autodl-tmp/datasets \
OUTPUT_DIR=/root/autodl-tmp/HIRE_v2_identity_balanced_logs \
PYTHON_BIN=/root/miniconda3/envs/irra190/bin/python \
NUM_EPOCH=60 \
SEED=1 \
BATCH_SIZE=64 \
NUM_WORKERS=8 \
SUPPORT_SIZE=3 \
AUX_WEIGHT=0.1 \
EXP_NAME=hire_v2_identity_balanced_tagpedes_60e_seed1 \
bash run_hire_v2_identity_balanced_4090_tag.sh
```

## 8. 最佳检查点组件评测

```bash
python tools/hire_v2/eval_identity_balanced_components.py \
  --config-file <实验目录>/configs.yaml \
  --checkpoint <实验目录>/best.pth
```

默认输出：

```text
<实验目录>/hire_v2_identity_balanced_components.json
```

包含：

```text
global
local
observation
identity
final
identity_gate
```

## 9. 正式结果必须比较

```text
v16.1.0 observation
v16.2.0 observation
v16.2.0 final
v16.2.1 observation
v16.2.1 identity
v16.2.1 final
```

## 10. 注意

```text
本版本不从 v16.1.0 或 v16.2.0 检查点继续训练；
本版本不包含状态头；
本版本不包含方差和异质性加权；
本版本不搜索 0.5/0.5 权重；
本版本只跑一次正式身份修正实验；
完成后进入状态创新。
```
