# HIRE-v2 v16.4.0 代码—文档一致性审计

## 一、结论

代码与 `HIRE_V2_IDENTITY_TOKEN_ROUTE_DESIGN.md` 的核心公式一致。

```text
模式：
--hire_v2 --hire_v2_mode identity_token_route

直接基础：
v16.2.1 identity_balanced

新增机制：
组条件文本词元身份可传播性路由

不存在：
独立状态分数；
状态门；
候选重排；
MLLM 标签输入；
反事实文本；
离线困难负样本池。
```

## 二、逐项对应

### 1. 原始词元选择

设计：

```text
图像：CLS 注意力前 30% 图像块；
文本：EOT 注意力前 30% 有效内容词元。
```

代码：

```python
AttentionRawTokenSelector(ratio=self.select_ratio)
```

文本显式排除：

```text
SOS；
EOT；
padding。
```

文件：

```text
model/hire_v2_token_route_components.py
```

一致。

### 2. 批内最高异身份图

设计：

\[
j_i^-=
\arg\max_{p_j\neq p_i}S_{ij}^{O}.
\]

代码：

```python
choose_hard_negative_indices(
    observation_score,
    pids,
)
```

完整观测分数在选择时停止梯度。

一致。

### 3. 锚点、支持和异身份 MaxSim

设计：

\[
A_{i,m}
=
\max_n t_{i,m}^{\top}v_{i,n}^{0},
\]

\[
P_{i,m,k}
=
\max_n t_{i,m}^{\top}v_{i,k,n}^{+},
\]

\[
N_{i,m}
=
\max_n t_{i,m}^{\top}v_{j_i^{-},n}.
\]

代码：

```python
token_patch_maxsim
support_token_patch_maxsim
```

一致。

### 4. 在线可传播目标

设计：

\[
D^{stable}
=
mean(P)-std(P)-N,
\]

\[
D^{pair}=A-N,
\]

\[
q^{id}
=
sigmoid(z(D^{stable}))
sigmoid(z(D^{pair})).
\]

代码：

```python
stable_margin = support_mean - support_std - hard_negative_score
pair_margin = anchor_score - hard_negative_score
target = sigmoid(stable_z) * sigmoid(pair_z)
```

一致。

### 5. 路由器输入保护

设计要求：

```text
文本原始词元停止梯度；
文本完整观测停止梯度；
在线目标停止梯度。
```

代码：

```python
token_features.detach()
text_observation.detach()
target.detach()
```

一致。

### 6. 路由初始值

设计：

\[
\pi^{id}=0.5.
\]

代码：

```python
nn.init.zeros_(self.fc2.weight)
nn.init.zeros_(self.fc2.bias)
```

因此最后一层初始 logit 为零，概率为 0.5。

一致。

### 7. 身份词元池化

设计：

\[
z^{id}
=
\frac{
\sum_m a_m\pi_m^{id}t_m
}{
\sum_m a_m\pi_m^{id}+\epsilon
}.
\]

代码：

```python
weight = token_attention * identity_probability * token_mask
pooled = (token_features * weight).sum(1) / weight.sum(1)
```

一致。

### 8. 零初始化身份词元残差

设计：

\[
u^T
=
Norm(
W_{id}sg(o^T)
+
A_{token}z^{id}
).
\]

代码：

```python
base_raw = self.identity_mean.proj(text_observation.detach())
identity = normalize(base_raw + self.identity_token_residual.proj(pooled))
```

`identity_token_residual.proj.weight` 初始化为零。

一致。

### 9. 图像身份路径

代码的图像端直接复用父类：

```python
super().encode_image_retrieval(images)
```

训练图像身份仍为：

```python
self._identity_from_observation(image_observation)
```

没有引入图像词元路由。

与文档一致。

### 10. 身份组共识

代码继续复用：

```python
masked_identity_group_consensus
paired_identity_group_nce
```

支持图仍然：

```text
同 PID；
不同 image_id；
动态轮换；
视角均衡；
不作为普通图文正样本。
```

一致。

### 11. 最终分数

代码继续复用：

```python
identity_residual_score
build_identity_final_embedding
```

所以训练前向和测试单向量点积均等价于：

\[
(1-\alpha)S^O+\alpha S^I.
\]

一致。

### 12. 总损失

设计：

\[
0.5(L_G+L_L)
+0.5L_O
+0.5L_F
+0.1L_{group}
+0.1L_{route}.
\]

代码：

```python
aggregate_identity_token_route_objectives
```

返回：

```text
sdm_loss
itc_loss
identity_group_loss
token_route_loss
```

一致。

## 三、初始化等价性

身份词元适配器初始为零，因此：

```text
文本身份表示初始等于 v16.2.1；
图像身份表示等于 v16.2.1；
最终分数初始等于 v16.2.1；
标准测试向量初始等于 v16.2.1。
```

路由辅助只训练路由器，不直接改变 CLIP 主干或完整观测。

## 四、数据关系

`datasets/build.py` 对：

```text
identity
identity_balanced
identity_state
identity_token_route
```

统一使用 `HIREV2IdentityDataset`。

v16.4.0 没有新增数据文件依赖。

## 五、向后兼容

保留：

```text
anchor
identity
identity_balanced
identity_state
```

新增：

```text
identity_token_route
```

旧版本模型、评测和日志字段未删除。

## 六、已执行检查

当前交付环境已执行：

```text
全部新增和修改 Python 文件语法解析；
两个 Bash 启动脚本语法检查；
v16.4.0 组件单元测试 10/10；
v16.4.0 数学与静态审计 8/8。
```

单元测试覆盖：

```text
图像块选择排除 CLS；
文本词元排除特殊和填充位置；
不同身份困难图选择；
稳定词元目标高于状态式词元目标；
路由器初始输出 0.5；
词元残差初始严格为零；
掩码路由 BCE；
填充支持槽在方差计算前被屏蔽，有限支持不会产生 NaN；
总损失公式；
掩码相关系数。
```

静态审计覆盖：

```text
模式注册；
数据集注册；
日志注册；
在线目标公式；
零初始化；
停止梯度；
排除状态头和候选重排；
排除反事实训练和离线困难负样本文件。
```

## 七、服务器仍需验证

当前交付环境没有真实 TAG-PEDES 图像和 RTX 4090，因此服务器必须完成：

```text
真实一轮烟测；
实际显存峰值；
六十轮训练；
最佳检查点五组件评测；
词元路由测试统计；
最终修复／破坏审计。
```

组件评测脚本以同一检查点的 `observation` 为参照，额外记录
`final` 的 Top-1 修复数、破坏数、净修复数以及稳定正确／错误数。
这些标签只用于离线统计，不进入模型前向或训练目标。
