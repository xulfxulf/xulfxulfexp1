# v16.2.0 身份机制无需训练审计：直接执行文档

## 一、目标

本审计不训练、不反向传播、不修改检查点，只回答两个问题。

第一，当前“逐维方差 + 组内异质性”的可信交集是否真的不同于简单均值。重点检查：

- 方差头是否产生了真实变化；
- 三张支持图是否被赋予不同权重；
- 异质性项在总不确定性中占多大比例；
- 完整可信交集与简单均值、仅方差加权均值有多接近；
- 三种身份组构造在严格留一组检索上的结果是否有差异。

第二，身份残差在测试排序中修复了什么、破坏了什么。默认执行两种对比：

- `v16.2 observation → v16.2 final`：隔离身份残差本身；
- `v16.1 observation → v16.2 final`：衡量相对上一版的真实净收益。

## 二、代码文件

将覆盖包解压到仓库根目录后新增：

```text
tools/hire_v2/audit_v162_identity.py
tests/test_v162_identity_audit.py
run_hire_v2_identity_audit.sh
```

不修改模型、训练、数据加载或评测代码。

## 三、默认路径

启动脚本已经填入已归档实验的服务器默认路径：

```text
v16.2 配置：
/root/autodl-tmp/HIRE_v2_identity_logs/82601f8/TAG-PEDES/20260716_201333_hire_v2_identity_tagpedes_60e_seed1_82601f8/configs.yaml

v16.2 检查点：
/root/autodl-tmp/HIRE_v2_identity_logs/82601f8/TAG-PEDES/20260716_201333_hire_v2_identity_tagpedes_60e_seed1_82601f8/best.pth

v16.1 配置：
/root/autodl-tmp/HIRE_v2_anchor_logs/48e61f8_full60/TAG-PEDES/20260716_104228_hire_v2_anchor_tagpedes_60e_seed1_48e61f8/configs.yaml

v16.1 检查点：
/root/autodl-tmp/HIRE_v2_anchor_logs/48e61f8_full60/TAG-PEDES/20260716_104228_hire_v2_anchor_tagpedes_60e_seed1_48e61f8/best.pth
```

路径不同就通过环境变量覆盖。

## 四、执行前检查

```bash
cd /root/autodl-tmp/IRRA_light_baseline

python -m py_compile tools/hire_v2/audit_v162_identity.py

pytest -q tests/test_v162_identity_audit.py
```

## 五、完整执行命令

```bash
cd /root/autodl-tmp/IRRA_light_baseline

CUDA_VISIBLE_DEVICES=0 \
PROJECT_ROOT=/root/autodl-tmp/IRRA_light_baseline \
PYTHON_BIN=/root/miniconda3/envs/irra190/bin/python \
OUTPUT_DIR=/root/autodl-tmp/HIRE_v2_identity_audit/v16.2.0_full \
bash run_hire_v2_identity_audit.sh
```

默认会：

- 编码全部 19,954 张训练图像；
- 编码全部 39,908 条训练文本；
- 在支持轮次 `0,15,30,45,54,59` 上统计可信交集结构；
- 在支持轮次 `54` 上执行完整严格留一组检索；
- 在 16,504 条测试文本上执行修复/破坏审计；
- 同时比较 v16.1 与 v16.2。

## 六、显存不足时

先降低批次和分块，不改变数学结果：

```bash
CUDA_VISIBLE_DEVICES=0 \
IMAGE_BATCH_SIZE=64 \
TEXT_BATCH_SIZE=256 \
QUERY_CHUNK=64 \
GALLERY_CHUNK=512 \
OUTPUT_DIR=/root/autodl-tmp/HIRE_v2_identity_audit/v16.2.0_lowmem \
bash run_hire_v2_identity_audit.sh
```

## 七、只跑 v16.2 内部修复/破坏，不加载 v16.1

```bash
V161_CONFIG="" \
V161_CHECKPOINT="" \
OUTPUT_DIR=/root/autodl-tmp/HIRE_v2_identity_audit/v16.2.0_no_v161 \
bash run_hire_v2_identity_audit.sh
```

## 八、快速试跑

先只编码前 2,000 条训练文本，并只审计最佳轮次支持集合：

```bash
SUPPORT_EPOCHS=54 \
RETRIEVAL_EPOCHS=54 \
MAX_TRAIN_QUERIES=2000 \
OUTPUT_DIR=/root/autodl-tmp/HIRE_v2_identity_audit/v16.2.0_quick \
bash run_hire_v2_identity_audit.sh
```

快速试跑只用于确认链路，不能替代完整结论。

## 九、输出目录

```text
OUTPUT_DIR/
├── audit_manifest.json
├── v162_identity_audit_report.md
├── v162_identity_audit_report.json
├── automatic_findings.json
├── trusted_intersection/
│   ├── uncertainty_head_summary.json
│   ├── trusted_intersection_audit.json
│   ├── epoch_000/
│   │   ├── group_summary.json
│   │   └── group_per_image.csv
│   ├── ...
│   └── epoch_054/
│       ├── group_summary.json
│       ├── group_per_image.csv
│       ├── paired_group_retrieval_summary.json
│       └── paired_group_retrieval_per_query.csv
└── fix_break/
    ├── fix_break_audit.json
    ├── v16.2_observation_vs_v16.2_final/
    │   ├── summary.json
    │   ├── all_query_deltas.csv
    │   ├── fix_cases.csv
    │   ├── break_cases.csv
    │   ├── rank_improved_cases.csv
    │   └── rank_worsened_cases.csv
    └── v16.1_observation_vs_v16.2_final/
        └── 同上
```

## 十、可信交集重点字段

### `variance_all_images.std`

所有训练图逐维预测方差的整体标准差。若极小，方差头几乎输出常数。

### `support_scalar_variance_cv`

同一身份组三张支持图的平均方差变异系数。接近零代表三张支持图几乎没有不同置信度。

### `precision_dim_cv`

每个身份维度上，三个支持图有效精度的变异系数，再对维度求平均。它直接反映可信交集是否真正非等权。

### `tau_share`

\[
\frac{\tau^2}{\bar{\sigma}^2+\tau^2}
\]

表示组内异质性在总不确定性中的占比。

### `trusted_simple_cos`

完整可信交集与简单均值身份中心的余弦相似度。

建议判断：

```text
中位数 > 0.9995：
两者几乎相同。

超过 95% 身份组的余弦 > 0.9995：
可信交集在绝大多数组上接近简单均值。
```

### 三种组检索

- `simple`：简单均值；
- `variance_only`：只用预测方差加权；
- `trusted`：预测方差加组内异质性。

若三者 R1 差距小于 `0.05`，不能把 v16.2 的收益主要归因于复杂可信加权。

## 十一、修复/破坏判读

### `fix`

基线第一名错误，候选版本第一名正确。

### `break`

基线第一名正确，候选版本第一名错误。

### `net_top1`

```text
fix_count - break_count
```

### 两个对比的区别

`v16.2 observation_vs_final` 若净修复明显为正，说明身份残差本身有价值。

`v16.1 observation_vs_v16.2 final` 若净修复很小，而前者明显为正，说明身份残差主要在补偿 v16.2 训练导致的观测锚点退化。

### 残差字段

v16.2 内部对比额外记录：

```text
identity_residual_on_base_top1
identity_residual_on_candidate_top1
identity_residual_on_source_image
identity_contribution_on_base_top1
identity_contribution_on_candidate_top1
identity_contribution_on_source_image
```

`identity_residual` 是原始的 `S_id-S_observation`，`identity_contribution` 是乘以实际身份门后的真实最终分数贡献。

## 十二、固定结论规则

满足以下两条时，建议做 v16.2.1 锚点平衡修正版，而不是继续强化方差：

```text
v16.2 observation→final 的净修复为正；
v16.1 observation→v16.2 final 的净修复明显更小。
```

满足以下条件时，可信交集应在论文中降级为普通组共识，或后续改为纯异质性版本：

```text
可信交集与简单均值中位余弦 > 0.9995；
有效精度变异系数接近零；
trusted 与 simple 的组检索 R1 差 < 0.05。
```

若方差和精度差异明显、trusted 组检索稳定高于 simple，则保留完整概率可信交集。
