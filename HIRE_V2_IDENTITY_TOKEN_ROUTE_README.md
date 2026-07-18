# HIRE-v2 v16.4.0 工程使用说明

## 一、版本信息

```text
实验版本：v16.4.0
模式：--hire_v2 --hire_v2_mode identity_token_route
直接基线：v16.2.1
源主分支：07a9fe1cb86d5e223de23ccb02eb319f1eb4402d
```

本版本不沿用 v16.3.0 的独立状态分数和候选重排。它以 v16.2.1 为主体，只加入同身份支持关系监督的文本词元身份路由。

## 二、该版本实际做什么

```text
当前配对主体：
继续使用 v16.2.1 的全局—局部完整观测。

在线词元教师：
用锚点图、三张同身份支持图和一个批内最高相似异身份图，
为每个选中文本词元生成连续可传播性目标。

词元路由器：
测试时只看文本词元与文本完整观测，
预测该词元是否适合进入身份残差。

文本身份残差：
只池化高可传播概率词元，
再通过零初始化适配器修正文本身份表示。
```

本版本不需要任何外部标签文件、MLLM 输出或离线困难负样本表。

## 三、解压

```bash
cd /root/autodl-tmp/IRRA_light_baseline

git rev-parse HEAD

unzip -o HIRE_v2_identity_token_route_overlay_20260718.zip \
  -d /root/autodl-tmp/IRRA_light_baseline
```

建议当前主分支至少包含：

```text
v16.2.1 identity_balanced 模式；
v16.3.0 identity_state 模式；
HIREV2IdentityDataset；
最新 utils/metrics.py。
```

## 四、静态审计

```bash
python tools/hire_v2/audit_identity_token_route.py \
  --output-json /root/autodl-tmp/HIRE_v2_identity_token_route_audit.json
```

预期：

```text
status: passed
image_token_selection: passed
text_token_selection: passed
hard_negative_selection: passed
group_propagability_target: passed
router_initialization: passed
identity_initialization_equivalence: passed
objective_formula: passed
source_invariants: passed
```

## 五、单元测试

```bash
pytest -q \
  tests/test_hire_v2_anchor_components.py \
  tests/test_hire_v2_identity_components.py \
  tests/test_hire_v2_identity_balanced_components.py \
  tests/test_hire_v2_state_components.py \
  tests/test_hire_v2_token_route_components.py
```

只测试新模块：

```bash
pytest -q tests/test_hire_v2_token_route_components.py
```

## 六、一轮烟测

```bash
CUDA_VISIBLE_DEVICES=0 \
DATA_ROOT=/root/autodl-tmp/datasets \
OUTPUT_DIR=/root/autodl-tmp/HIRE_v2_identity_token_route_smoke \
PYTHON_BIN=/root/miniconda3/envs/irra190/bin/python \
NUM_EPOCH=1 \
SEED=1 \
BATCH_SIZE=64 \
SUPPORT_SIZE=3 \
AUX_WEIGHT=0.1 \
bash run_hire_v2_identity_token_route_smoke.sh
```

烟测日志必须出现：

```text
token_route_loss
token_route_bce
token_route_valid_ratio
token_route_probability_mean
token_route_probability_std
token_route_target_mean
token_route_target_std
token_route_target_correlation
token_route_hard_negative_valid_ratio
token_route_selected_count
identity_token_residual_norm
identity_group_nce
identity_gate
```

烟测通过条件：

```text
一轮训练和评测正常完成；
best.pth 正常保存；
全部损失有限；
无 NaN、Inf、CUDA OOM；
token_route_valid_ratio 大于 0；
token_route_hard_negative_valid_ratio 接近 1；
token_route_selected_count 大于 0；
最终评测使用标准 final 单向量。
```

## 七、正式六十轮训练

```bash
CUDA_VISIBLE_DEVICES=0 \
DATA_ROOT=/root/autodl-tmp/datasets \
OUTPUT_DIR=/root/autodl-tmp/HIRE_v2_identity_token_route_logs \
PYTHON_BIN=/root/miniconda3/envs/irra190/bin/python \
NUM_EPOCH=60 \
SEED=1 \
BATCH_SIZE=64 \
NUM_WORKERS=8 \
SUPPORT_SIZE=3 \
AUX_WEIGHT=0.1 \
EXP_NAME=hire_v2_identity_token_route_tagpedes_60e_seed1 \
bash run_hire_v2_identity_token_route_4090_tag.sh
```

正式训练必须从 OpenAI CLIP ViT-B/16 原始预训练权重开始，不加载 v16.2.1 或 v16.3.0 检查点。

## 八、最佳检查点组件评测

```bash
python tools/hire_v2/eval_identity_token_route_components.py \
  --config-file <实验目录>/configs.yaml \
  --checkpoint <实验目录>/best.pth
```

默认输出：

```text
<实验目录>/hire_v2_identity_token_route_components.json
```

组件包括：

```text
global
local
observation
identity
final
```

额外输出：

```text
测试词元路由概率均值；
测试词元路由概率标准差；
高于 0.5 的词元比例；
路由熵；
每条文本身份词元残差范数。
observation 到 final 的 Top-1 修复数、破坏数和净修复数。
```

## 九、正式比较对象

至少比较：

```text
v16.1.0 observation；
v16.2.1 observation；
v16.2.1 identity；
v16.2.1 final；
v16.4.0 observation；
v16.4.0 identity；
v16.4.0 final。
```

v16.3.0 作为独立状态分数负结果保留在论文消融或研究过程记录中，不作为 v16.4.0 的直接训练基线。

## 十、结果判读

优先看：

```text
final 是否超过 v16.2.1 的 58.034；
mAP 是否不低于 44.540；
final 是否高于同检查点 observation；
路由概率是否离开统一 0.5；
路由预测与在线目标是否形成正相关；
身份词元残差范数是否离开零点。
```

建议进入 MLLM 教师版本的条件：

```text
R1 至少达到 58.234；
mAP 不下降；
路由概率标准差不是接近零；
路由器未全部输出高或全部输出低；
身份残差仍保持修复多于破坏。
```

## 十一、显存问题

v16.4.0 会为主图和三张支持图保存选中的原始图像块关系证据，但不做全图库晚交互。

显存不足时，优先降低：

```text
BATCH_SIZE=48
```

但该结果不能与批次大小 64 的正式结果直接作为严格单变量对照。

不应通过修改选择比例、支持图数量或辅助权重规避显存，因为这些会改变方法定义。

## 十二、禁止同时修改

正式 v16.4.0 不得同时：

```text
打开图像增强；
修改随机采样器；
改变支持图数量；
改变辅助权重；
加入 MLLM 标签；
加入反事实文本；
加入状态分支；
加入困难负样本池；
从旧检查点继续训练。
```
