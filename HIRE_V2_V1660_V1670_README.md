# HIRE-v2 v16.6.0 / v16.7.0 执行说明

## 一、覆盖基线

```text
仓库：xulfxulf/xulfxulfexp1
源提交：0863840462dbd4dfdb9e42b2cdfb2f08010b4e6f
数据集：TAG-PEDES
v16.6 模式：identity_phrase_route
v16.7 模式：identity_phrase_route_cmp
```

两版共用一个模型结构。v16.7 只加载不同的训练教师 JSONL。

## 二、解压

```bash
cd /root/autodl-tmp/IRRA_light_baseline

unzip -o HIRE_v2_phrase_distillation_v1660_v1670_overlay_20260719.zip \
  -d /root/autodl-tmp/IRRA_light_baseline
```

## 三、静态检查

```bash
python tools/hire_v2/audit_identity_phrase_route.py \
  --output-json /root/autodl-tmp/HIRE_v2_phrase_route_static_audit.json
```

```bash
pytest -q \
  tests/test_hire_v2_phrase_route_components.py \
  tests/test_phrase_route_io.py \
  tests/test_phrase_teacher_common.py
```

预期：

```text
16 passed
static audit status: passed
```

## 四、离线环境

短语提取和 MLLM 推理建议在独立环境执行，不修改训练环境 `irra190`。

```bash
conda activate /root/autodl-tmp/envs/tag_mllm
pip install "spacy==3.7.5" pillow jsonschema
python -m spacy download en_core_web_sm
```

生成式教师路径示例：

```text
/root/autodl-tmp/models/Qwen3-VL-8B-Instruct
/root/autodl-tmp/models/InternVL3_5-8B-HF
```

固定目录：

```bash
export PROJECT_ROOT=/root/autodl-tmp/IRRA_light_baseline
export DATA_ROOT=/root/autodl-tmp/datasets
export PHRASE_ROOT=/root/autodl-tmp/HIRE_v2_phrase_distillation
mkdir -p ${PHRASE_ROOT}/{spans,v1660_quality,v1660_full,v1670_full}
cd ${PROJECT_ROOT}
```

## 五、生成训练和测试短语跨度

```bash
python tools/mllm/build_phrase_spans.py \
  --root-dir ${DATA_ROOT} \
  --output-dir ${PHRASE_ROOT}/spans \
  --spacy-model en_core_web_sm \
  --splits train test
```

必须生成：

```text
${PHRASE_ROOT}/spans/phrase_spans_train.jsonl
${PHRASE_ROOT}/spans/phrase_spans_test.jsonl
${PHRASE_ROOT}/spans/phrase_span_summary.json
```

训练、验证和测试都使用这套固定跨度。测试阶段不读取教师标签。

# v16.6.0

## 六、先构造三百五十例质量子集

```bash
python tools/mllm/build_phrase_teacher_cases.py \
  --root-dir ${DATA_ROOT} \
  --train-spans ${PHRASE_ROOT}/spans/phrase_spans_train.jsonl \
  --output-file ${PHRASE_ROOT}/v1660_quality/cases.jsonl \
  --support-size 3 \
  --seed 1 \
  --teacher-support-epoch 0 \
  --per-category 50
```

## 七、双教师三例烟测

Qwen 正序：

```bash
python tools/mllm/run_phrase_teacher.py \
  --teacher qwen3vl \
  --model-path /root/autodl-tmp/models/Qwen3-VL-8B-Instruct \
  --cases ${PHRASE_ROOT}/v1660_quality/cases.jsonl \
  --output-file ${PHRASE_ROOT}/v1660_quality/qwen_forward_smoke.jsonl \
  --order forward \
  --precision bf16 \
  --max-cases 3 \
  --overwrite
```

InternVL 正序：

```bash
python tools/mllm/run_phrase_teacher.py \
  --teacher internvl35 \
  --model-path /root/autodl-tmp/models/InternVL3_5-8B-HF \
  --cases ${PHRASE_ROOT}/v1660_quality/cases.jsonl \
  --output-file ${PHRASE_ROOT}/v1660_quality/intern_forward_smoke.jsonl \
  --order forward \
  --precision bf16 \
  --max-cases 3 \
  --overwrite
```

若某个教师 BF16 出现显存不足，该教师后续四组正式推理全部统一改用：

```text
--precision int8
```

不得同一个教师部分 BF16、部分 int8。

## 八、质量子集四组正式推理

```bash
python tools/mllm/run_phrase_teacher.py \
  --teacher qwen3vl \
  --model-path /root/autodl-tmp/models/Qwen3-VL-8B-Instruct \
  --cases ${PHRASE_ROOT}/v1660_quality/cases.jsonl \
  --output-file ${PHRASE_ROOT}/v1660_quality/qwen_forward.jsonl \
  --order forward --precision bf16

python tools/mllm/run_phrase_teacher.py \
  --teacher qwen3vl \
  --model-path /root/autodl-tmp/models/Qwen3-VL-8B-Instruct \
  --cases ${PHRASE_ROOT}/v1660_quality/cases.jsonl \
  --output-file ${PHRASE_ROOT}/v1660_quality/qwen_reverse.jsonl \
  --order reverse --precision bf16

python tools/mllm/run_phrase_teacher.py \
  --teacher internvl35 \
  --model-path /root/autodl-tmp/models/InternVL3_5-8B-HF \
  --cases ${PHRASE_ROOT}/v1660_quality/cases.jsonl \
  --output-file ${PHRASE_ROOT}/v1660_quality/intern_forward.jsonl \
  --order forward --precision bf16

python tools/mllm/run_phrase_teacher.py \
  --teacher internvl35 \
  --model-path /root/autodl-tmp/models/InternVL3_5-8B-HF \
  --cases ${PHRASE_ROOT}/v1660_quality/cases.jsonl \
  --output-file ${PHRASE_ROOT}/v1660_quality/intern_reverse.jsonl \
  --order reverse --precision bf16
```

## 九、合并并审计质量子集

质量子集只覆盖部分记录，因此先用相同跨度文件合并会给未标记录写入无监督零目标，这是预期行为。

```bash
python tools/mllm/merge_phrase_teacher.py \
  --train-spans ${PHRASE_ROOT}/spans/phrase_spans_train.jsonl \
  --cases ${PHRASE_ROOT}/v1660_quality/cases.jsonl \
  --qwen-forward ${PHRASE_ROOT}/v1660_quality/qwen_forward.jsonl \
  --qwen-reverse ${PHRASE_ROOT}/v1660_quality/qwen_reverse.jsonl \
  --intern-forward ${PHRASE_ROOT}/v1660_quality/intern_forward.jsonl \
  --intern-reverse ${PHRASE_ROOT}/v1660_quality/intern_reverse.jsonl \
  --output-labels ${PHRASE_ROOT}/v1660_quality/labels.jsonl \
  --output-summary ${PHRASE_ROOT}/v1660_quality/merge_summary.json
```

```bash
python tools/mllm/audit_phrase_teacher.py \
  --labels ${PHRASE_ROOT}/v1660_quality/labels.jsonl \
  --cases ${PHRASE_ROOT}/v1660_quality/cases.jsonl \
  --output-summary ${PHRASE_ROOT}/v1660_quality/audit_summary.json \
  --review-csv ${PHRASE_ROOT}/v1660_quality/review_template.csv \
  --per-category 50
```

人工填写 `manual_label` 和 `manual_comment`。`manual_label` 只允许：

```text
correct
incorrect
uncertain
```

填写后重新运行审计并增加：

```bash
--completed-review-csv ${PHRASE_ROOT}/v1660_quality/review_template.csv
```

正式全量标签前至少确认：

```text
未知比例不高于70%
三类关系均出现
人工抽查准确率不低于80%
双教师双顺序一致性满足项目门槛
```

## 十、构造 v16.6 全量案例

```bash
python tools/mllm/build_phrase_teacher_cases.py \
  --root-dir ${DATA_ROOT} \
  --train-spans ${PHRASE_ROOT}/spans/phrase_spans_train.jsonl \
  --output-file ${PHRASE_ROOT}/v1660_full/cases.jsonl \
  --support-size 3 \
  --seed 1 \
  --teacher-support-epoch 0
```

对全量案例重复四组教师命令，输出：

```text
qwen_forward.jsonl
qwen_reverse.jsonl
intern_forward.jsonl
intern_reverse.jsonl
```

然后严格合并：

```bash
python tools/mllm/merge_phrase_teacher.py \
  --train-spans ${PHRASE_ROOT}/spans/phrase_spans_train.jsonl \
  --cases ${PHRASE_ROOT}/v1660_full/cases.jsonl \
  --qwen-forward ${PHRASE_ROOT}/v1660_full/qwen_forward.jsonl \
  --qwen-reverse ${PHRASE_ROOT}/v1660_full/qwen_reverse.jsonl \
  --intern-forward ${PHRASE_ROOT}/v1660_full/intern_forward.jsonl \
  --intern-reverse ${PHRASE_ROOT}/v1660_full/intern_reverse.jsonl \
  --output-labels ${PHRASE_ROOT}/v1660_full/train_labels_v1660.jsonl \
  --output-summary ${PHRASE_ROOT}/v1660_full/merge_summary.json
```

## 十一、v16.6 一轮烟测

```bash
CUDA_VISIBLE_DEVICES=0 \
DATA_ROOT=${DATA_ROOT} \
TRAIN_LABELS=${PHRASE_ROOT}/v1660_full/train_labels_v1660.jsonl \
TEST_SPANS=${PHRASE_ROOT}/spans/phrase_spans_test.jsonl \
PYTHON_BIN=/root/miniconda3/envs/irra190/bin/python \
OUTPUT_DIR=/root/autodl-tmp/HIRE_v2_phrase_route_v1660_smoke \
BATCH_SIZE=64 \
NUM_EPOCH=1 \
bash run_hire_v2_phrase_route_v1660_smoke.sh
```

日志必须出现：

```text
phrase_route_loss
phrase_route_kl
phrase_route_supervision_ratio
phrase_route_teacher_entropy
phrase_route_student_entropy
phrase_route_spearman
phrase_route_top1_agreement
phrase_identity_residual_norm
```

## 十二、v16.6 正式六十轮

```bash
CUDA_VISIBLE_DEVICES=0 \
DATA_ROOT=${DATA_ROOT} \
TRAIN_LABELS=${PHRASE_ROOT}/v1660_full/train_labels_v1660.jsonl \
TEST_SPANS=${PHRASE_ROOT}/spans/phrase_spans_test.jsonl \
PYTHON_BIN=/root/miniconda3/envs/irra190/bin/python \
OUTPUT_DIR=/root/autodl-tmp/HIRE_v2_phrase_route_v1660_logs \
BATCH_SIZE=64 \
NUM_EPOCH=60 \
SEED=1 \
EXP_NAME=hire_v2_phrase_route_v1660_tagpedes_60e_seed1 \
bash run_hire_v2_phrase_route_v1660_4090_tag.sh
```

## 十三、v16.6 组件评测

```bash
python tools/hire_v2/eval_identity_phrase_route_components.py \
  --config-file <v16.6实验目录>/configs.yaml \
  --checkpoint <v16.6实验目录>/best.pth
```

# v16.7.0

## 十四、挖掘训练集最高相似异身份图像

必须使用 v16.6 最佳检查点，且只处理 TAG 训练集。

```bash
python tools/hire_v2/extract_phrase_hard_negatives.py \
  --config-file <v16.6实验目录>/configs.yaml \
  --checkpoint <v16.6实验目录>/best.pth \
  --v1660-train-labels ${PHRASE_ROOT}/v1660_full/train_labels_v1660.jsonl \
  --output-file ${PHRASE_ROOT}/v1670_full/hard_negatives.jsonl
```

## 十五、构造比较式教师案例

```bash
python tools/mllm/build_phrase_comparative_cases.py \
  --v1660-cases ${PHRASE_ROOT}/v1660_full/cases.jsonl \
  --hard-negatives ${PHRASE_ROOT}/v1670_full/hard_negatives.jsonl \
  --output-file ${PHRASE_ROOT}/v1670_full/cases.jsonl
```

对 `v1670_full/cases.jsonl` 再运行两个教师、两种顺序。教师输出中会增加：

```text
hard_negative
```

## 十六、合并 v16.7 比较式目标

```bash
python tools/mllm/merge_phrase_comparative_teacher.py \
  --v1660-labels ${PHRASE_ROOT}/v1660_full/train_labels_v1660.jsonl \
  --comparative-cases ${PHRASE_ROOT}/v1670_full/cases.jsonl \
  --qwen-forward ${PHRASE_ROOT}/v1670_full/qwen_forward.jsonl \
  --qwen-reverse ${PHRASE_ROOT}/v1670_full/qwen_reverse.jsonl \
  --intern-forward ${PHRASE_ROOT}/v1670_full/intern_forward.jsonl \
  --intern-reverse ${PHRASE_ROOT}/v1670_full/intern_reverse.jsonl \
  --output-labels ${PHRASE_ROOT}/v1670_full/train_labels_v1670.jsonl \
  --output-summary ${PHRASE_ROOT}/v1670_full/merge_summary.json
```

同样运行教师审计：

```bash
python tools/mllm/audit_phrase_teacher.py \
  --labels ${PHRASE_ROOT}/v1670_full/train_labels_v1670.jsonl \
  --cases ${PHRASE_ROOT}/v1670_full/cases.jsonl \
  --output-summary ${PHRASE_ROOT}/v1670_full/audit_summary.json \
  --review-csv ${PHRASE_ROOT}/v1670_full/review_template.csv \
  --per-category 50
```

## 十七、v16.7 烟测与正式训练

```bash
CUDA_VISIBLE_DEVICES=0 \
DATA_ROOT=${DATA_ROOT} \
TRAIN_LABELS=${PHRASE_ROOT}/v1670_full/train_labels_v1670.jsonl \
TEST_SPANS=${PHRASE_ROOT}/spans/phrase_spans_test.jsonl \
PYTHON_BIN=/root/miniconda3/envs/irra190/bin/python \
OUTPUT_DIR=/root/autodl-tmp/HIRE_v2_phrase_route_v1670_smoke \
BATCH_SIZE=64 \
NUM_EPOCH=1 \
bash run_hire_v2_phrase_route_v1670_smoke.sh
```

```bash
CUDA_VISIBLE_DEVICES=0 \
DATA_ROOT=${DATA_ROOT} \
TRAIN_LABELS=${PHRASE_ROOT}/v1670_full/train_labels_v1670.jsonl \
TEST_SPANS=${PHRASE_ROOT}/spans/phrase_spans_test.jsonl \
PYTHON_BIN=/root/miniconda3/envs/irra190/bin/python \
OUTPUT_DIR=/root/autodl-tmp/HIRE_v2_phrase_route_v1670_logs \
BATCH_SIZE=64 \
NUM_EPOCH=60 \
SEED=1 \
EXP_NAME=hire_v2_phrase_route_v1670_tagpedes_60e_seed1 \
bash run_hire_v2_phrase_route_v1670_4090_tag.sh
```

## 十八、禁止同时修改

两版正式实验均不得同时：

```text
加入v16.5同图双文本损失
改变身份支持数量
改变0.1辅助权重
打开可选图像增强
改变随机采样器
加载旧检查点继续训练
加入独立状态头
加入候选晚交互
测试时运行MLLM
测试时使用支持图
```
