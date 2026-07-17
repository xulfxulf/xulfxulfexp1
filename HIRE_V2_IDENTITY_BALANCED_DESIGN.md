# HIRE-v2 v16.2.1：锚点平衡的留一身份组共识残差

**工程仓库：** `xulfxulf/xulfxulfexp1`

**源主分支提交：** `7d89aa311eda5aaef8b7f6f200e2cd47de015ad0`

**代码模式：** `--hire_v2 --hire_v2_mode identity_balanced`

**实验版本：** `v16.2.1`

**直接对照：** `v16.1.0` 与 `v16.2.0`

**主干：** OpenAI CLIP ViT-B/16

**完整观测：** CLIP 全局观测 + RDE 风格词元选择 + 零初始化局部残差融合

**本版本唯一研究主题：** 在保护完整观测锚点的前提下，保留有效的身份组共识残差

**训练方式：** 从同一 CLIP 权重开始的单阶段端到端训练，不从 v16.1.0 或 v16.2.0 检查点继续训练
**测试输入：** 单条文本和单张图库图像，测试阶段不需要同身份支持集合

---

# 1. 版本定位

v16.1.0 建立了全局—局部完整观测基线。它在 TAG-PEDES 上达到约：

```text
R1 57.774
mAP 44.238
mINP 24.113
```

v16.2.0 在其上加入同身份多图身份随机效应，最终约为：

```text
R1 57.901
mAP 44.408
mINP 24.329
```

但无需训练的离线审计发现：

1. v16.2.0 内部从完整观测到最终身份融合，净修复 96 条 Top-1 查询，说明身份残差本身有效；
2. v16.1.0 完整观测到 v16.2.0 最终结果只净修复 27 条，说明大部分身份收益被 v16.2.0 完整观测锚点的退化抵消；
3. 预测方差均值为 1.050010、标准差仅 0.000220；
4. 完整可信交集与简单支持均值的中位余弦为 1.000000；
5. 100% 的有效身份组二者余弦不低于 0.9995；
6. 完整可信交集相对简单均值的严格组检索 R1 只提高 0.002542。

因此，v16.2.1 不继续加强方差或异质性加权，而执行两项直接由证据支持的修正：

```text
修正一：
恢复完整观测自身的直接检索约束，
并与最终身份融合分数各占一半主任务权重。

修正二：
删除未启动的预测方差与异质性加权，
将身份集合明确实现为严格留一的简单组共识。
```

这里第二项不是引入另一个新方法变量，而是删除已由离线审计证明没有形成实际差异的无效机制，使代码与可证实的研究结论一致。

---

# 2. 本版本明确包含与不包含的内容

## 2.1 包含

```text
CLIP ViT-B/16 图像和文本编码器；
RDE 风格注意力词元选择；
全局和局部零初始化残差融合；
全局与局部观测锚点监督；
完整观测直接检索监督；
图像与文本共享身份映射；
同身份、不同 image_id 的动态支持图；
严格留一身份组共识；
严格对角身份组 NCE；
有界身份残差门；
最终身份残差检索监督。
```

## 2.2 不包含

```text
状态头；
状态残差；
支持文本；
弱正样本对损失；
身份分类器；
固定训练身份原型；
方差预测头；
方差校准损失；
组内异质性加权；
MLLM 教师标签；
三态标签；
困难负样本池；
视角分类器；
图像质量分支；
第二阶段训练；
测试阶段支持集合。
```

---

# 3. 符号定义

设一个随机训练批次包含 \(B\) 个图文样本：

\[
\mathcal B=
\left\{
(I_i,T_i,p_i,g_i)
\right\}_{i=1}^{B},
\]

其中：

- \(I_i\)：第 \(i\) 张训练图像；
- \(T_i\)：与该训练样本绑定的文本；
- \(p_i\)：训练身份编号；
- \(g_i\)：训练图像编号 `image_id`；
- \(B=64\)。

同一个训练图像通常具有两条文本，因此不同训练样本可能具有相同 \(g_i\)。

向量二范数归一化：

\[
\operatorname{Norm}(x)
=
\frac{x}
{\max(\lVert x\rVert_2,\epsilon)}.
\]

温度沿用已有配置：

\[
\tau=0.02,
\qquad
s=\frac1\tau.
\]

---

# 4. 固定的 v16.1.0 完整观测锚点

v16.2.1 不修改 v16.1.0 的主干、词元选择与融合定义。

## 4.1 CLIP 全局观测

图像编码器输出：

\[
X_i^I=
[x_{i,\mathrm{cls}}^I,
x_{i,1}^I,\ldots,x_{i,N}^I].
\]

文本编码器输出：

\[
X_i^T=
[x_{i,\mathrm{sos}}^T,
x_{i,1}^T,\ldots,
x_{i,\mathrm{eot}}^T,\ldots].
\]

图像全局观测：

\[
b_i^I
=
\operatorname{Norm}
(x_{i,\mathrm{cls}}^I).
\]

文本全局观测：

\[
b_i^T
=
\operatorname{Norm}
(x_{i,\mathrm{eot}}^T).
\]

全局相似度：

\[
S_{ij}^{G}
=
(b_i^T)^\top b_j^I.
\]

## 4.2 RDE 风格细粒度观测

图像端使用最后一层类别词元注意力，从局部图像块中选择注意力最高的前 \(30\%\)。

文本端使用结束词元注意力，在排除开始词元、结束词元和填充词元后，选择注意力最高的有效词元。

选中词元分别经过：

\[
h
=
\operatorname{MLP}(\hat x)
+
W_{\mathrm{skip}}\hat x,
\]

再做逐维最大池化，得到：

\[
l_i^I,\qquad l_i^T.
\]

局部相似度：

\[
S_{ij}^{L}
=
(l_i^T)^\top l_j^I.
\]

词元选择比例 `0.3` 和局部维度 `1024` 沿用 v16.1.0，不属于 v16.2.1 新增超参数。

## 4.3 零初始化残差融合

图像和文本分别具有一个局部残差适配器：

\[
e_i^I=A_I l_i^I,
\qquad
e_i^T=A_T l_i^T.
\]

\(A_I\) 和 \(A_T\) 均为无偏置线性层，权重初始化为零。

完整观测：

\[
o_i^I
=
\operatorname{Norm}
(b_i^I+e_i^I),
\]

\[
o_i^T
=
\operatorname{Norm}
(b_i^T+e_i^T).
\]

训练开始时：

\[
A_I=A_T=0,
\]

因此：

\[
o_i^I=b_i^I,
\qquad
o_i^T=b_i^T.
\]

完整观测相似度：

\[
S_{ij}^{O}
=
(o_i^T)^\top o_j^I.
\]

---

# 5. 基础检索目标

每个相似度矩阵 \(S\) 使用：

\[
L_{\mathrm{ret}}(S)
=
L_{\mathrm{SDM}}(S)
+
L_{\mathrm{ITC}}(S).
\]

## 5.1 身份分布匹配

身份关系：

\[
Y_{ij}
=
\mathbf 1[p_i=p_j].
\]

真实分布：

\[
\hat Y_{ij}
=
\frac{Y_{ij}}
{\sum_kY_{ik}}.
\]

预测文本到图像分布：

\[
P_{ij}^{t2i}
=
\frac{\exp(sS_{ij})}
{\sum_k\exp(sS_{ik})}.
\]

预测图像到文本分布使用 \(S^\top\)。

代码完全复用仓库现有 SDM 定义：

\[
L_{\mathrm{SDM}}
=
L_{\mathrm{KL}}^{t2i}
+
L_{\mathrm{KL}}^{i2t}.
\]

## 5.2 原配图文对比

第 \(i\) 个训练样本的原配位置为对角位置 \(i\)。

\[
L_{\mathrm{ITC}}^{t2i}
=
-\frac1B
\sum_i
\log
\frac{\exp(sS_{ii})}
{\sum_j\exp(sS_{ij})}.
\]

图像到文本方向同理：

\[
L_{\mathrm{ITC}}
=
\frac12
\left(
L_{\mathrm{ITC}}^{t2i}
+
L_{\mathrm{ITC}}^{i2t}
\right).
\]

---

# 6. 全局与局部观测锚点

全局目标：

\[
L_G
=
L_{\mathrm{ret}}(S^G).
\]

局部目标：

\[
L_L
=
L_{\mathrm{ret}}(S^L).
\]

观测锚点目标：

\[
L_{\mathrm{anchor}}
=
\frac12(L_G+L_L).
\]

它持续保护：

```text
CLIP 全局语义；
RDE 风格局部语义；
主干和词元选择模块的独立检索能力。
```

---

# 7. 共享单样本身份映射

完整观测进入图像和文本共享的身份映射：

\[
u_i^m
=
\operatorname{Norm}
\left[
W_{\mathrm{id}}
\operatorname{sg}(o_i^m)
\right],
\qquad
m\in\{I,T\}.
\]

其中：

- \(W_{\mathrm{id}}\in\mathbb R^{512\times512}\)；
- 图像和文本共用同一个 \(W_{\mathrm{id}}\)；
- 无偏置；
- 单位矩阵初始化；
- `sg` 表示停止梯度。

单位初始化保证：

\[
u_i^m=o_i^m
\]

在训练开始时严格成立。

停止梯度保证：

```text
身份组辅助损失只更新身份映射；
身份组辅助损失不能直接改写完整观测；
完整观测只由全局、局部、完整观测和最终检索目标更新。
```

---

# 8. 同身份动态支持图

每个主训练样本额外返回最多 \(K=3\) 张支持图。

## 8.1 关系约束

每张支持图必须满足：

```text
support_pid == anchor_pid；
support_image_id != anchor_image_id；
同一支持集合内 image_id 不重复。
```

禁止：

```text
锚点原图；
同图另一条文本对应的原图；
不同身份图像；
重复支持图。
```

## 8.2 视角均衡

若存在跨视角和同视角图像，选择顺序为：

```text
一张跨视角；
一张同视角；
第三张从剩余候选中交替补充。
```

某类不足时使用另一类补足。

## 8.3 动态轮换

支持选择由：

```text
固定随机种子；
当前 epoch；
anchor image_id
```

共同决定。

同一张图的两条文本共享相同支持集合；不同训练轮次轮换支持图。

## 8.4 支持编码梯度

支持图的：

```text
CLIP 主干；
图像词元选择；
完整观测融合
```

在 `torch.no_grad()` 下运行。

图像词元选择模块暂时进入评测状态，避免支持图更新 BatchNorm 运行统计。

支持完整观测经过共享身份映射，身份映射仍正常获得梯度。

---

# 9. 严格留一身份组共识

设第 \(i\) 个锚点的有效支持身份表示为：

\[
\left\{
u_{i,a}^{I}
\right\}_{a=1}^{K_i},
\qquad
K_i\leq3.
\]

当：

\[
K_i\geq2
\]

时，该身份组有效。

v16.2.1 使用离线审计支持的确定性组共识：

\[
\tilde C_i^I
=
\frac1{K_i}
\sum_{a=1}^{K_i}
u_{i,a}^{I}.
\]

归一化后：

\[
C_i^I
=
\operatorname{Norm}
(\tilde C_i^I).
\]

支持图不作为普通图文正样本；它们只共同构造一个身份组级目标。

## 9.1 组内离散度诊断

为了记录同身份支持图是否一致，计算：

\[
D_i
=
\frac1{K_iD}
\sum_a
\left\|
u_{i,a}^I
-
\operatorname{sg}(\tilde C_i^I)
\right\|_2^2.
\]

该量只写入日志：

```text
identity_group_dispersion
```

不参与训练加权，也不构成额外损失。

同时记录支持身份表示到组共识的平均余弦：

```text
identity_group_support_cosine
```

---

# 10. 严格对角身份组监督

查询文本身份表示与所有批内身份组共识计算：

\[
S_{ij}^{\mathrm{group}}
=
(u_i^T)^\top C_j^I.
\]

第 \(i\) 列身份组是专门为第 \(i\) 个查询构造的留一组，并严格排除其原配图像。因此：

```text
对角身份组 j=i：
唯一身份组正目标。

相同 PID 的非对角身份组：
忽略，不作为额外正样本，也不作为负样本。
原因是这些组可能包含查询 i 的原配图像。

不同 PID 的有效身份组：
负样本。

无效身份组：
全部屏蔽。
```

允许矩阵：

\[
M_{ij}
=
\mathbf1[p_i\neq p_j]
+
\mathbf1[i=j].
\]

只保留有效身份组列。

严格身份组 NCE：

\[
L_{\mathrm{group}}
=
-\frac1{|\mathcal V|}
\sum_{i\in\mathcal V}
\log
\frac{
\exp(sS_{ii}^{\mathrm{group}})
}{
\exp(sS_{ii}^{\mathrm{group}})
+
\sum_{j:p_j\neq p_i,\;C_j\text{有效}}
\exp(sS_{ij}^{\mathrm{group}})
}.
\]

\(\mathcal V\) 还要求查询具有至少一个不同身份有效负组。

---

# 11. 身份残差分数

单样本身份分数：

\[
S_{ij}^{I}
=
(u_i^T)^\top u_j^I.
\]

身份门：

\[
\alpha
=
\operatorname{sigmoid}
(\theta_{\mathrm{id}}).
\]

初始化：

\[
\alpha_0=0.1.
\]

身份门始终满足：

\[
0<\alpha<1.
\]

最终训练分数：

\[
S^{F}
=
S^{O}
+
\alpha
\left[
S^{I}
-
\operatorname{sg}(S^{O})
\right].
\]

前向数值等价于：

\[
S^{F}
=
(1-\alpha)S^O
+
\alpha S^I.
\]

但梯度设计不同：

```text
S^O 在最终检索损失中保留完整梯度；
减去的 S^O 使用停止梯度；
身份表示读取停止梯度后的完整观测；
因此身份辅助和身份支路不能通过捷径重写完整观测。
```

---

# 12. v16.2.1 的核心锚点平衡目标

完整观测直接检索目标：

\[
L_O
=
L_{\mathrm{ret}}(S^O).
\]

最终身份融合检索目标：

\[
L_F
=
L_{\mathrm{ret}}(S^F).
\]

固定平衡主目标：

\[
L_{\mathrm{main}}
=
\frac12L_O
+
\frac12L_F.
\]

`0.5 / 0.5` 是根据离线审计提出的固定结构，不开放为命令行搜索参数。

它具有关键初始化性质。

因为 \(W_{\mathrm{id}}\) 为单位初始化：

\[
S^I=S^O.
\]

所以：

\[
S^F=S^O.
\]

进而：

\[
L_{\mathrm{main}}
=
\frac12L_O
+
\frac12L_O
=
L_O.
\]

因此，训练第一步的主检索目标严格等于 v16.1.0 的完整观测主目标。

同时，代码中的梯度审计验证：

\[
\nabla_{S^O}
\left[
\frac12L(S^O)
+
\frac12L(S^F)
\right]
=
\nabla_{S^O}L(S^O)
\]

在初始化等价点成立。

这意味着 v16.2.1 不会因为多加一条最终分数监督而在训练开始时把完整观测主梯度放大或缩小。

---

# 13. 总损失

v16.2.1 总损失为：

\[
L_{\mathrm{v16.2.1}}
=
L_{\mathrm{anchor}}
+
L_{\mathrm{main}}
+
\lambda L_{\mathrm{group}}.
\]

其中：

\[
L_{\mathrm{anchor}}
=
\frac12(L_G+L_L),
\]

\[
L_{\mathrm{main}}
=
\frac12L_O
+
\frac12L_F,
\]

\[
\lambda=0.1.
\]

完整展开：

\[
L_{\mathrm{v16.2.1}}
=
\frac12(L_G+L_L)
+
\frac12L_O
+
\frac12L_F
+
0.1L_{\mathrm{group}}.
\]

工程中拆成三个包含 `loss` 的返回项：

```text
sdm_loss
= 0.5 × (global_sdm + local_sdm)
+ 0.5 × observation_sdm
+ 0.5 × final_sdm

itc_loss
= 0.5 × (global_itc + local_itc)
+ 0.5 × observation_itc
+ 0.5 × final_itc

identity_group_loss
= 0.1 × identity_group_nce
```

训练器只求和这三个损失项。

---

# 14. 与 v16.1.0 和 v16.2.0 的严格区别

| 项目 | v16.1.0 | v16.2.0 | v16.2.1 |
|---|---|---|---|
| 全局/局部完整观测 | 有 | 有 | 有 |
| 同身份支持图 | 无 | 有 | 有 |
| 共享身份映射 | 无 | 有 | 有 |
| 身份组监督 | 无 | 有 | 有 |
| 预测方差 | 无 | 有但未启动 | 删除 |
| 异质性加权 | 无 | 有但等价均值 | 删除 |
| 组共识 | 无 | 概率公式，实际近似均值 | 明确简单均值 |
| 完整观测直接主损失 | 有 | 无 | 恢复为 0.5 |
| 最终身份融合主损失 | 无 | 1.0 | 0.5 |
| 身份组辅助权重 | 无 | 0.1 | 0.1 |
| 状态分支 | 无 | 无 | 无 |

v16.2.1 的核心因果问题是：

> 在身份残差和组监督保持不变的情况下，显式保护完整观测是否能够保留 v16.1.0 的强锚点，并把 v16.2.0 内部已经存在的身份残差收益转化为更大的净收益？

---

# 15. 推理

测试阶段只输入：

```text
一条文本；
一张图库图像。
```

不使用：

```text
训练身份编号；
同身份支持图；
身份组共识；
方差；
视角标签。
```

文本和图像分别得到完整观测与身份表示：

\[
o^m,\qquad u^m.
\]

构造最终单向量：

\[
f^m
=
\left[
\sqrt{1-\alpha}\,o^m,
\sqrt{\alpha}\,u^m
\right].
\]

其点积为：

\[
(f_i^T)^\top f_j^I
=
(1-\alpha)S_{ij}^{O}
+
\alpha S_{ij}^{I}.
\]

与训练前向分数数值完全一致。

因此，最终仍可使用仓库原有的单向量余弦评测器。

---

# 16. 训练设置

正式实验固定：

```text
数据集：TAG-PEDES
主干：OpenAI CLIP ViT-B/16
图像尺寸：384 × 128
训练轮数：60
批次大小：64
随机种子：1
主采样器：random
可选图像增强：关闭
学习率：1e-5
新模块学习率倍率：沿用仓库 lr_factor=5
温度：0.02
支持图数量：3
身份组辅助权重：0.1
身份门初始值：0.1
验证协议：沿用历史内部对照，test 每轮评测
```

v16.2.1 不从 v16.1.0 或 v16.2.0 检查点继续训练。

---

# 17. 单批次训练流程

以一个训练批次为例：

1. 编码主批次图像和文本；
2. 生成全局观测 \(b^I,b^T\)；
3. 生成局部观测 \(l^I,l^T\)；
4. 生成完整观测 \(o^I,o^T\)；
5. 计算全局和局部锚点损失；
6. 直接计算完整观测的 SDM 与 ITC；
7. 对停止梯度后的完整观测应用共享身份映射；
8. 编码同身份支持图；
9. 对支持图应用共享身份映射；
10. 用简单均值构造严格留一身份组共识；
11. 计算严格对角身份组 NCE；
12. 计算单样本身份分数；
13. 使用有界身份门构造最终分数；
14. 计算最终分数的 SDM 与 ITC；
15. 按固定公式聚合总损失；
16. 一次反向传播和一次优化器更新。

---

# 18. 训练日志必须记录

## 18.1 观测锚点

```text
global_sdm
global_itc
local_sdm
local_itc
observation_sdm
observation_itc
anchor_objective
observation_objective
```

## 18.2 最终身份残差

```text
final_sdm
final_itc
final_objective
balanced_main_objective
identity_gate
identity_score_delta_abs
observation_final_score_delta_abs
observation_identity_cosine
identity_projection_delta_norm
```

## 18.3 身份组共识

```text
identity_group_loss
identity_group_nce
support_valid_ratio
support_count_mean
identity_group_dispersion
identity_group_support_cosine
```

## 18.4 固定主权重

```text
observation_main_weight = 0.5
final_main_weight = 0.5
```

---

# 19. 组件评测

最佳检查点必须分别输出：

```text
global
local
observation
identity
final
```

每个组件计算：

```text
R1
R5
R10
mAP
mINP
```

重点判断：

1. `observation` 是否恢复到 v16.1.0 水平；
2. `final` 是否继续高于 `observation`；
3. 身份门是否保持保守而非饱和；
4. `final - observation` 的增益是否仍然稳定；
5. 最终净结果是否高于 v16.2.0。

---

# 20. 成功标准

最低要求：

```text
observation R1 不低于 57.65；
final R1 不低于 v16.2.0 的 57.913；
final mAP 不低于 44.409；
final mINP 不低于 24.323；
身份门不接近 0 或 1；
身份组损失正常下降；
有效支持组比例保持约 98.6%。
```

理想要求：

```text
observation 恢复到约 57.75；
final R1 达到或超过 58.1；
mAP 与 mINP 同时提高；
身份残差内部修复数继续明显多于破坏数。
```

理想范围不是性能保证，只是由 v16.2.0 内部净修复与锚点退化数量推导的目标。

---

# 21. 失败后的固定决策

v16.2.1 只允许训练这一版，不再继续搜索身份参数。

```text
若 v16.2.1 高于 v16.2.0：
以 v16.2.1 作为后续状态版本的身份基础。

若 v16.2.1 不高于 v16.2.0：
保留 v16.2.0 作为后续状态版本的身份基础。

无论哪种情况：
停止身份侧调参，进入 v16.3.0 状态实例残差。
```

不继续尝试：

```text
改变支持图数量；
增大身份组损失；
手动增大身份门；
重新加入方差；
增加身份分类器；
增加身份原型；
改变采样器。
```

---

# 22. 代码文件对应关系

| 设计部分 | 工程文件 |
|---|---|
| 简单组共识与平衡损失 | `model/hire_v2_identity_balanced_components.py` |
| 完整 v16.2.1 前向与推理 | `model/hire_v2_identity_balanced_model.py` |
| 模型模式分发 | `model/__init__.py` |
| 命令行模式与校验 | `utils/options.py` |
| 复用动态支持数据集 | `datasets/build.py` |
| 新诊断日志 | `processor/processor.py` |
| 组件离线评测 | `tools/hire_v2/eval_identity_balanced_components.py` |
| 数学与静态审计 | `tools/hire_v2/audit_identity_balanced.py` |
| 单元测试 | `tests/test_hire_v2_identity_balanced_components.py` |
| 一轮烟测 | `run_hire_v2_identity_balanced_smoke.sh` |
| 正式训练 | `run_hire_v2_identity_balanced_4090_tag.sh` |

---

# 23. 最终研究表述

v16.2.1 暂不再声称“概率可信交集”已经得到验证。更准确的身份级方法表述是：

> 我们将同身份不同图像视为潜在身份因素的多次留一观测，以动态组共识监督共享身份映射；该身份空间不替代完整图文观测，而是通过有界残差对完整观测排序进行保守修正。为了防止身份训练削弱已经有效的图文观测空间，我们同时优化完整观测和身份修正后的最终分数，使身份级监督只补充跨图稳定信息，而不接管当前实例匹配。

该版本仍然只验证身份创新，不包含状态创新。
