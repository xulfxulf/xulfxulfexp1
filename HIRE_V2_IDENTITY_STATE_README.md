# HIRE-v2 v16.3.0 工程使用说明

## 一、版本

```text
实验版本：v16.3.0
模式：--hire_v2 --hire_v2_mode identity_state
直接基线：v16.2.1
源主分支：44749815b3b6769071b424472938913f3feb3ec3
```

本包是覆盖式工程包，应解压到上述提交或其后保持兼容的主分支根目录。

## 二、本版本新增内容

```text
保留完整 v16.2.1 身份平衡路径；
选择 8 个文本状态词元；
选择 16 个图像状态块；
共享 128 维状态投影；
对身份基础前 50 候选计算词元—图像块 MaxSim；
同 image_id 为状态正样本；
不同身份为状态负样本；
同身份不同 image_id 在状态损失中忽略；
零初始化状态门；
训练和测试均只重排前 50 候选。
```

本版本不修改身份组共识，不使用状态支持包，不引入 MLLM、三态标签、视角分类器或困难负样本。

## 三、解压

```bash
cd /root/autodl-tmp/IRRA_light_baseline

git rev-parse HEAD

unzip -o HIRE_v2_identity_state_overlay_20260717.zip \
  -d /root/autodl-tmp/IRRA_light_baseline
```

源提交应为：

```text
44749815b3b6769071b424472938913f3feb3ec3
```

## 四、静态审计

```bash
python tools/hire_v2/audit_identity_state.py \
  --output-json /root/autodl-tmp/HIRE_v2_identity_state_audit.json
```

预期结果：

```text
status: passed
zero_state_gate: passed
positive_candidate_coverage: passed
late_interaction: passed
initial_score_equivalence: passed
objective_formula: passed
initial_gradient_equivalence: passed
shared_state_projection: passed
source_invariants: passed
```

## 五、单元测试

```bash
pytest -q \
  tests/test_hire_v2_anchor_components.py \
  tests/test_hire_v2_identity_components.py \
  tests/test_hire_v2_identity_balanced_components.py \
  tests/test_hire_v2_state_components.py
```

只运行新测试：

```bash
pytest -q tests/test_hire_v2_state_components.py
```

## 六、一轮烟测

```bash
CUDA_VISIBLE_DEVICES=0 \
DATA_ROOT=/root/autodl-tmp/datasets \
OUTPUT_DIR=/root/autodl-tmp/HIRE_v2_identity_state_smoke \
PYTHON_BIN=/root/miniconda3/envs/irra190/bin/python \
NUM_EPOCH=1 \
SEED=1 \
BATCH_SIZE=64 \
SUPPORT_SIZE=3 \
AUX_WEIGHT=0.1 \
STATE_TOPK=50 \
STATE_IMAGE_TOKENS=16 \
STATE_TEXT_TOKENS=8 \
bash run_hire_v2_identity_state_smoke.sh
```

烟测日志必须包含：

```text
identity_final_sdm
identity_final_itc
state_final_sdm
state_final_itc
state_pair_loss
state_pair_nce
state_gate
state_candidate_ratio
state_positive_coverage
state_positive_negative_margin
state_text_token_count
state_image_token_count
observation_main_weight: 0.5000
identity_main_weight: 0.2500
state_final_main_weight: 0.2500
```

烟测通过条件：

```text
训练和状态专用评测完成；
保存 best.pth；
所有损失和统计有限；
无 NaN、Inf、CUDA OOM；
state_positive_coverage 等于 1；
state_text_token_count 大于 0；
state_image_token_count 等于 16；
训练期评测表同时输出 identity_final 和 state_final。
```

## 七、正式六十轮训练

```bash
CUDA_VISIBLE_DEVICES=0 \
DATA_ROOT=/root/autodl-tmp/datasets \
OUTPUT_DIR=/root/autodl-tmp/HIRE_v2_identity_state_logs \
PYTHON_BIN=/root/miniconda3/envs/irra190/bin/python \
NUM_EPOCH=60 \
SEED=1 \
BATCH_SIZE=64 \
NUM_WORKERS=8 \
SUPPORT_SIZE=3 \
AUX_WEIGHT=0.1 \
STATE_TOPK=50 \
STATE_IMAGE_TOKENS=16 \
STATE_TEXT_TOKENS=8 \
EXP_NAME=hire_v2_identity_state_tagpedes_60e_seed1 \
bash run_hire_v2_identity_state_4090_tag.sh
```

正式实验从 CLIP ViT-B/16 原始预训练权重开始，不加载 v16.2.1 检查点。

## 八、最佳检查点组件评测

```bash
python tools/hire_v2/eval_identity_state_components.py \
  --config-file <实验目录>/configs.yaml \
  --checkpoint <实验目录>/best.pth
```

默认输出：

```text
<实验目录>/hire_v2_identity_state_components.json
```

输出组件：

```text
global
local
observation
identity
identity_final
state_final
```

并输出：

```text
identity_final -> state_final 修复数；
破坏数；
净修复数；
identity_gate；
state_gate；
state_topk；
状态文本和图像词元数量；
状态投影维度。
```

## 九、最佳检查点保存协议

v16.3.0 不使用普通单向量评测结果保存最佳检查点。

每个 epoch 的验证流程为：

```text
一、计算 v16.2.1 identity_final 全图库分数；
二、每条查询选取 identity_final 前 50 图像；
三、只对这 50 张图计算状态晚交互；
四、生成 state_final；
五、按 state_final 的 R1 保存 best.pth。
```

测试阶段不会使用真实 image_id 强制补充正确候选。

## 十、正式结果必须比较

```text
v16.1.0 observation
v16.2.1 observation
v16.2.1 identity_final
v16.3.0 observation
v16.3.0 identity_final
v16.3.0 state_final
```

重点判断：

```text
state_gate 是否为正；
状态正负间隔是否为正；
state_final 是否高于同检查点 identity_final；
状态修复数是否多于破坏数；
最终 R1 是否超过 58.034；
mAP 是否不下降。
```

## 十一、显存或评测内存不足

训练状态交互只计算批内前 50 候选，不支持单独降低候选数后仍称为同一正式实验。

评测阶段可降低查询编码批次，使用原有：

```text
--test_batch_size
```

若状态专用评测显存不足，可将配置中的：

```text
hire_eval_query_chunk
```

从 `128` 改为 `64` 或 `32`。这只改变分块方式，不改变数学结果。

## 十二、注意事项

```text
不得从 v16.2.1 检查点继续训练；
不得打开图像增强；
不得修改随机采样器；
不得修改支持图数量；
不得修改身份辅助权重；
不得额外加入状态标签；
不得使用普通 Evaluator 结果代替状态专用评测；
不得将 identity_final 误写成最终 v16.3.0 结果。
```
