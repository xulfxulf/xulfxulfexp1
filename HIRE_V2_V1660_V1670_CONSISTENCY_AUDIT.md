# v16.6.0 / v16.7.0 代码—文档—原设计一致性审计

## 一、审计结论

```text
v16.6.0 代码与 HIRE_V2_V1660_DESIGN.md 一致。
v16.7.0 代码与 HIRE_V2_V1670_DESIGN.md 一致。
两版共享同一模型结构和损失实现。
两版唯一方法差异是训练标签 JSONL 中的教师相对分布。
实现符合上一轮正式实验计划；其中 v16.5 条件基础按原计划选择 v16.2.1。
```

## 二、v16.6.0 逐项核对

### 1. 基础版本

文档规定 v16.5 未通过或未执行时使用 v16.2.1。

代码：

```python
class HIREV2IdentityPhraseRoute(HIREV2IdentityBalanced)
```

因此完整观测、身份映射、支持图选择、严格留一组共识和身份门均继承 v16.2.1。

### 2. 短语而非子词

代码由：

```text
tools/mllm/phrase_extraction.py
tools/mllm/build_phrase_spans.py
```

生成完整字符跨度和 `token_positions`。训练、验证、测试统一读取相同跨度。

模型接收：

```text
phrase_token_mask
phrase_valid_mask
```

并在跨度内部使用 EOT 注意力池化。

### 3. 双教师双顺序

`run_phrase_teacher.py` 固定参数：

```text
teacher = qwen3vl 或 internvl35
order = forward 或 reverse
```

`merge_phrase_teacher.py` 对每个关系调用：

```python
strict_label(values)
```

且 `strict_label` 要求恰好四个判断完全一致；任何缺失或分歧返回 `unknown`。

### 4. 支持图顺序不导致错配

教师输出支持关系使用：

```text
support_by_image_id
```

合并时按数值 `image_id` 读取，而不是按数组位置读取。

### 5. 锚点可靠性

代码公式：

```python
anchor_reliable = (
    anchor_label == "support"
    and sibling_label != "contradiction"
)
```

与文档：

\[
a=\mathbf1[anchor=support \land sibling\neq contradiction]
\]

一致。

### 6. 传播原始分数

代码：

```python
raw = anchor_reliable \
      * (n_support / count) \
      * (1.0 - n_contradiction / count)
```

与文档完全一致。

### 7. 相对分布

代码 `normalize_record_phrase_targets`：

```python
target_weight = raw / sum(raw)
```

只有至少两个正原始分数时 `route_supervision_valid=True`。

没有使用 0.5 硬阈值。

### 8. 学生路由器

代码：

```python
RelativePhraseRouter
```

结构为：

```text
LayerNorm
512→128
GELU
128→1
caption 内 masked softmax
```

输出层零初始化，因此有效短语初始均匀分布。

### 9. 路由蒸馏

代码：

```python
phrase_route_kl_divergence
```

只对 `phrase_route_supervision=True` 且至少两个正目标的文本计算：

\[
KL(q\Vert\pi).
\]

### 10. 零初始化身份短语残差

代码：

```python
ZeroInitializedPhraseIdentityResidual
nn.init.zeros_(self.proj.weight)
```

文本身份为：

```python
normalize(identity_mean.proj(stopgrad(observation)) + phrase_residual(pooled))
```

所以初始严格等于 v16.2.1。

### 11. 总损失

代码：

```python
aggregate_identity_phrase_route_objectives
```

实现：

\[
0.5(L_G+L_L)+0.5L_O+0.5L_F+0.1L_{group}+0.1L_{route}.
\]

### 12. 测试流程

`PhraseTextDataset` 在测试时只加载 `span-only` 文件。

`utils/metrics.py` 将短语跨度传给：

```python
model.encode_text(...)
```

测试不读取训练教师目标、支持图或 PID。

## 三、v16.7.0 逐项核对

### 1. 最高相似异身份来源

`extract_phrase_hard_negatives.py`：

- 使用 v16.6 配置和检查点；
- 编码 TAG 训练图像和训练文本；
- 对同 PID 图像全部掩码；
- 每条文本只保留最高分不同 PID 图像；
- 不读取测试集。

### 2. 比较式教师

`build_phrase_comparative_cases.py` 只给 v16.6 案例增加一个：

```text
hard_negative
```

其余锚点、兄弟文本、支持图和短语均保持不变。

### 3. 严格 hard-negative 标签

仍由两个教师、两种支持顺序产生四个判断；分歧归 `unknown`。

### 4. 区分因子

代码常量：

```python
support -> 0.0
unknown -> 0.5
contradiction -> 1.0
```

与文档一致。

### 5. 比较式原始分数

代码：

```python
comparative_raw_score
= propagation_raw_score * discrimination_factor
```

然后在文本内部重新归一化。

### 6. 模型不变

`model/__init__.py` 对：

```text
identity_phrase_route
identity_phrase_route_cmp
```

调用同一个：

```python
build_hire_v2_identity_phrase_route_model
```

模型根据模式只设置：

```text
hire_v2_experiment_version
```

没有条件创建额外层、额外损失或额外推理路径。

## 四、与原设计的符合性

| 原设计要求 | 实现 | 结论 |
|---|---|---|
| 当前配对完整观测为主体 | 完整继承 v16.2.1 observation/final 结构 | 符合 |
| v16.5 未通过时以 v16.2.1 为基础 | v16.5 尚未执行，直接继承 v16.2.1 | 符合原条件分支 |
| 语义短语而非 CLIP 子词 | 离线依存规则、字符跨度和词元跨度 | 符合 |
| 锚点图、两条同图文本、三张支持图 | 教师案例完整包含 | 符合 |
| 不发送 PID、视角和答案 | 提示词不包含这些字段 | 符合 |
| support/contradiction/unknown | 固定三标签与严格解析 | 符合 |
| Qwen 与 InternVL | 统一教师运行器支持两者 | 符合 |
| 两种支持图顺序 | forward/reverse | 符合 |
| 四判断一致才保留 | `strict_label` | 符合 |
| 同图文本明确冲突才取消可靠性 | sibling != contradiction | 符合 |
| 传播公式 | 代码逐项一致 | 符合 |
| 相对分布而非绝对阈值 | 文本内归一化 + KL | 符合 |
| 学生路由初始均匀 | 输出层零初始化 | 符合 |
| 短语残差零初始化 | 适配器权重全零 | 符合 |
| 测试不运行 MLLM | 只读取 span-only 文件 | 符合 |
| 测试不使用支持图 | 图像和文本独立编码 | 符合 |
| v16.7 只改教师目标 | 两模式共用同一模型类 | 符合 |
| hard negative 来自训练集 | 提取脚本只使用 TAG train | 符合 |
| support/unknown/contradiction 因子 0/0.5/1 | 固定常量 | 符合 |
| 不恢复独立状态头 | 模型无 state gate/NCE/rerank | 符合 |

## 五、已执行自动检查

```text
全部24个交付Python文件抽象语法树解析：通过
四个Bash启动脚本语法检查：通过
模型组件、数据IO和教师公式单元测试：18/18通过
数学与静态审计：8/8通过
```

静态审计通过项：

```text
uniform_phrase_router
v1621_initial_equivalence
relative_route_kl
v1660_teacher_formula
v1670_teacher_formula
objective_formula
shared_architecture_and_registration
teacher_target_source_invariants
```

## 六、当前环境未完成的检查

当前交付环境没有 TAG-PEDES 图像、两个本地 8B 教师和 RTX 4090，因此未虚构以下结果：

```text
真实短语跨度覆盖率
真实双教师一致率
真实人工抽查准确率
真实一轮烟测
真实显存峰值
六十轮训练结果
```

这些项目必须按 README 在服务器执行。
