# HIRE 主版本：异质性感知的层级身份随机效应与状态残差模型

**对应工程基线提交：** `xulfxulf/xulfxulfexp1@90228fe720a82e36b04c4ac62e8d3247016c48d8`  
**主干：** CLIP ViT-B/16  
**细粒度观测：** RDE 风格词元选择模块  
**训练方式：** 单阶段联合训练  
**测试要求：** 单条文本与单张图像，不需要测试身份包  
**代码模式：** `--hire`

---

# 1. 研究问题

现有文本行人检索模型大多把训练单位理解为一张图像与一条文本，或者把同身份不同图像简单扩展为强正样本、弱正样本、支持集合或身份原型。空—地场景中，这种处理存在根本困难：

1. 同一个身份的多张图像只共享部分稳定证据；
2. 航拍与地面图像在视角、清晰度、遮挡和局部可见性上差异很大；
3. 一条文本只描述当前图像中被观察和被标注的部分状态；
4. 同身份不同图像不保证满足当前文本的全部局部描述；
5. 将整张同身份支持图直接拉近，会把状态差异和不可观测细节误传播到身份空间。

HIRE 不再把同身份支持样本视为若干普通正样本，而将它们解释为：

> 对同一个潜在身份随机效应的多次、异质、有噪观测。

每个图文实例则被解释为：

\[
\text{实例观测}
=
\text{身份组级随机效应}
+
\text{当前实例状态效应}
+
\text{模态与标注噪声}.
\]

HIRE 的目标是同时完成两种不同层级的匹配：

- **身份级匹配：** 单个图像或文本匹配由同身份多图动态估计的身份后验；
- **状态级匹配：** 当前文本的实例状态残差匹配当前原配图像的实例状态残差。

---

# 2. 总体结构

完整模型由五部分组成：

1. CLIP ViT-B/16 图像与文本编码器；
2. RDE 风格注意力词元选择模块；
3. 全局与细粒度观测融合模块；
4. 概率身份后验头；
5. 状态残差头。

模型不保留独立的 RDE 原始检索分数，也不把 RDE 直接称为身份头或状态头。RDE 只提供两个可迁移能力：

- 通过注意力选择有判别力的图像局部块和文本词元；
- 使用全部负样本的稳定排序思想。

最终检索分数只由身份概率匹配和状态残差匹配共同生成：

\[
S_{ij}^{\mathrm{final}}
=
\exp(\gamma_{\mathrm{id}})S_{ij}^{\mathrm{id}}
+
\exp(\gamma_{\mathrm{st}})S_{ij}^{\mathrm{st}}.
\]

其中两个尺度参数均为可学习标量，不需要人工搜索身份与状态融合比例。状态尺度初始化为身份尺度的十分之一，使训练从较安全的身份主导状态开始，但从第一轮起身份与状态模块同时训练。

---

# 3. 层级生成假设

设身份为 \(p\)，该身份的第 \(a\) 张图像为 \(I_{pa}\)，该图像的第 \(c\) 条文本为 \(T_{pac}\)。

身份组级潜变量：

\[
u_p \sim \mathcal N(0,I).
\]

图像状态与文本状态分别记为：

\[
s_{pa}^{I}\sim p(s^I),
\qquad
s_{pac}^{T}\sim p(s^T).
\]

两种模态的观测表示为：

\[
g_{pa}^{I}=u_p+s_{pa}^{I}+\epsilon_{pa}^{I},
\]

\[
g_{pac}^{T}=u_p+s_{pac}^{T}+\epsilon_{pac}^{T}.
\]

其中：

- \(u_p\) 表示同身份多图中可重复推断的公共身份因素；
- \(s^I\) 表示当前图像的视角、姿态、遮挡、局部可见性和成像状态；
- \(s^T\) 表示标注者从当前图像中实际描述出的状态子集；
- \(\epsilon\) 表示图像质量、模态差异和文本标注噪声。

文本状态不是图像状态的完整复制，可写成：

\[
s_{pac}^{T}=M_{pac}s_{pa}^{I}+\eta_{pac},
\]

其中 \(M_{pac}\) 是未知的描述选择与可见性算子。工程中不显式估计该矩阵，而由概率身份不确定性和状态残差共同吸收。

---

# 4. CLIP 全局观测

图像经过 CLIP ViT-B/16 后得到最终图像词元：

\[
X^I=[x_{\mathrm{cls}}^I,x_1^I,\ldots,x_N^I].
\]

文本经过 CLIP 文本编码器后得到：

\[
X^T=[x_{\mathrm{sos}}^T,x_1^T,\ldots,x_{\mathrm{eot}}^T,\ldots].
\]

全局观测为：

\[
b^I=x_{\mathrm{cls}}^I,
\qquad
b^T=x_{\mathrm{eot}}^T.
\]

工程代码通过 `CLIPAttentionAdapter` 在不修改原仓库 `clip_model.py` 的前提下，复现最后一层 Transformer 前向，并取得最后一层平均多头注意力矩阵。旧模型路径不受影响。

---

# 5. RDE 风格细粒度词元选择

## 5.1 图像词元选择

使用最后一层图像自注意力中类别词元对所有局部块的注意力：

\[
a_n^I=A^I_{\mathrm{cls},n}.
\]

去掉类别词元后，选择注意力最高的前 \(K_I\) 个局部块：

\[
K_I=\max\left(1,\left\lfloor \rho N\right\rfloor\right),
\qquad \rho=0.3.
\]

对选中的局部块做归一化、线性旁路与两层感知机残差映射：

\[
h_n^I=\operatorname{MLP}(\hat x_n^I)+W_{\mathrm{skip}}^I\hat x_n^I.
\]

通过逐维最大池化得到图像细粒度观测：

\[
l^I=\max_{n\in\operatorname{TopK}(a^I)} h_n^I.
\]

## 5.2 文本词元选择

CLIP 的结束词元承担句子级聚合作用，因此使用结束词元对其他词元的最后一层注意力：

\[
a_m^T=A^T_{\mathrm{eot},m}.
\]

开始词元、结束词元和填充词元不参与候选。选择前 \(K_T\) 个有效词元：

\[
K_T=\max\left(1,\left\lfloor \rho(L-2)\right\rfloor\right).
\]

映射与池化为：

\[
h_m^T=\operatorname{MLP}(\hat x_m^T)+W_{\mathrm{skip}}^T\hat x_m^T,
\]

\[
l^T=\max_{m\in\operatorname{TopK}(a^T)}h_m^T.
\]

代码中的 `RDEVisualTokenSelection` 与 `RDETextTokenSelection` 保留了 RDE 的注意力选择、残差映射和最大池化核心，但修正了无效文本词元的显式屏蔽。

---

# 6. 全局与细粒度观测融合

对图像和文本分别建立融合模块：

\[
g^m
=
\operatorname{LN}
\left[
W_g^m b^m
+
\alpha_m W_l^m l^m
\right],
\qquad m\in\{I,T\}.
\]

其中：

\[
\alpha_m=\operatorname{sigmoid}(a_m).
\]

全局映射初始化为单位映射，局部映射权重初始化为零，局部门控初始值为 \(0.5\)。因此模型初始观测严格等于 CLIP 全局表示；局部映射从第一轮起获得梯度，随后逐步引入细粒度证据。

工程默认：

```text
CLIP 全局维度：512
RDE 词元选择维度：1024
融合观测维度：512
```

---

# 7. 单样本概率身份后验

图像和文本分别从观测表示预测对角高斯身份分布：

\[
q^m(u\mid g^m)
=
\mathcal N
\left(
\mu^m,
\operatorname{diag}[(\sigma^m)^2]
\right).
\]

身份均值：

\[
\mu^m
=
\operatorname{LN}(W_\mu^m g^m).
\]

方差：

\[
(\sigma^m)^2
=
\operatorname{softplus}
\left(
W_\sigma^m\operatorname{sg}(g^m)+b_\sigma^m
\right)
+10^{-6}.
\]

其中 \(\operatorname{sg}\) 表示停止梯度。方差头读取停止梯度后的观测，防止共享编码器为了方便预测方差而扭曲主特征。

均值映射以单位矩阵初始化；方差映射权重置零、偏置初始化为使初始方差接近 1。这样训练开始时不会产生随机的极端置信度。

---

# 8. 留一、跨模态身份支持集合

每个训练锚点额外选择最多三张：

```text
同一身份；
不同 image_id；
优先同时覆盖跨视角和同视角；
每张支持图只使用一条对应文本。
```

查询文本的身份后验由同身份其他图像构造：

\[
T_i\longrightarrow C_{p_i}^{I,-i}.
\]

查询图像的身份后验由同身份其他图像对应的文本构造：

\[
I_i\longrightarrow C_{p_i}^{T,-i}.
\]

由于数据加载器严格排除相同 `image_id`，原配图像及同图另一条文本不能进入支持集合。

支持样本不会作为普通正样本进入对比学习分子；它们唯一的身份作用是估计潜在身份随机效应。

---

# 9. 异质性感知的可信交集

设同身份支持集合中第 \(a\) 个观测在第 \(d\) 个身份维度上输出：

\[
q_a(u_d)=\mathcal N(\mu_{a,d},\sigma_{a,d}^2).
\]

## 9.1 初始观测内精度

\[
w_{a,d}^{(0)}
=
\frac{1}{\sigma_{a,d}^2+\varepsilon}.
\]

初始精度加权中心：

\[
\mu_{p,d}^{(0)}
=
\frac{\sum_a w_{a,d}^{(0)}\mu_{a,d}}
{\sum_a w_{a,d}^{(0)}}.
\]

计算异质性时，代码对该初始中心停止梯度，避免模型通过移动中心人为降低组内分歧。

## 9.2 组内异质性

\[
\tau_{p,d}^2
=
\frac{
\sum_a w_{a,d}^{(0)}
(\mu_{a,d}-\operatorname{sg}(\mu_{p,d}^{(0)}))^2
}
{
\sum_a w_{a,d}^{(0)}
}.
\]

该项描述同身份不同图像对某个身份维度是否达成一致。

## 9.3 随机效应有效精度

\[
w_{a,d}
=
\frac{1}
{\sigma_{a,d}^2+\tau_{p,d}^2+\varepsilon}.
\]

最终身份后验均值：

\[
\bar\mu_{p,d}
=
\frac{\sum_a w_{a,d}\mu_{a,d}}
{\sum_a w_{a,d}}.
\]

最终身份后验方差：

\[
\bar\sigma_{p,d}^2
=
\frac{1}{\sum_a w_{a,d}}
+
\tau_{p,d}^2.
\]

该可信交集同时考虑：

- **观测内不确定性：** 单张图像或文本本身是否模糊、遗漏或低质量；
- **观测间异质性：** 同身份多图是否对该维度形成一致意见。

只有自身置信度高且多图一致的维度，才能形成高置信身份后验。

---

# 10. 概率身份匹配分数

对单样本身份分布：

\[
q_i=\mathcal N(\mu_i,\sigma_i^2)
\]

和身份集合后验：

\[
C_p=\mathcal N(\bar\mu_p,\bar\sigma_p^2),
\]

定义逐维互似然距离：

\[
D(q_i,C_p)
=
\frac{1}{2d}
\sum_{k=1}^{d}
\left[
\frac{(\mu_{i,k}-\bar\mu_{p,k})^2}
{\sigma_{i,k}^2+\bar\sigma_{p,k}^2}
+
\log(\sigma_{i,k}^2+\bar\sigma_{p,k}^2)
\right].
\]

身份相似度：

\[
S_{\mathrm{id}}(q_i,C_p)=-D(q_i,C_p).
\]

测试阶段的单图—单文本身份分数使用同一个公式，只需将集合后验替换为另一单样本后验：

\[
S_{ij}^{\mathrm{id}}
=
-D(q_i^T,q_j^I).
\]

---

# 11. 身份后验分类损失

设一个主批次中的身份编号集合由 `pids` 给出。文本查询与每一行动态图像身份后验形成分数矩阵：

\[
A_{ij}^{T\rightarrow I\text{-set}}
=
S_{\mathrm{id}}(q_i^T,C_{p_j}^{I,-j}).
\]

同身份行均为正目标，目标分布在同身份后验之间均匀分配：

\[
y_{ij}
=
\frac{\mathbf 1[p_i=p_j]}
{\sum_k\mathbf 1[p_i=p_k]}.
\]

文本到图像集合损失：

\[
L_{T\rightarrow I\text{-set}}
=
-\frac{1}{B}
\sum_i\sum_j
 y_{ij}
\log
\frac{\exp(A_{ij}/\tau_0)}
{\sum_k\exp(A_{ik}/\tau_0)}.
\]

图像到文本集合方向同理。工程还对分数矩阵转置方向执行对称约束，以保证单样本和集合后验在两个方向上都可识别。

记全部身份集合识别项为：

\[
L_{\mathrm{set}}.
\]

---

# 12. 身份不确定性校准

仅依靠身份分类可能使方差退化为常数。HIRE 使用同身份组内偏差构造方差监督目标。

对支持观测 \(a\)：

\[
v_{a,d}^{\mathrm{target}}
=
(\mu_{a,d}-\operatorname{sg}(\bar\mu_{p,d}))^2
+
\operatorname{sg}(\bar\sigma_{p,d}^2).
\]

在对数方差空间使用平滑绝对误差：

\[
L_{\mathrm{cal}}
=
\operatorname{SmoothL1}
\left[
\log\sigma_{a,d}^2,
\log v_{a,d}^{\mathrm{target}}
\right].
\]

身份层总损失：

\[
L_{\mathrm{identity}}
=
L_{\mathrm{set}}
+
L_{\mathrm{cal}}.
\]

---

# 13. 状态残差头

状态头不再是另一个自由的全局检索头，而是学习当前观测相对于跨模态身份公共因素的剩余部分。

图像和文本状态预测：

\[
r_i^I
=
\operatorname{Norm}(W_s^I g_i^I),
\]

\[
r_i^T
=
\operatorname{Norm}(W_s^T g_i^T).
\]

训练时，文本状态的目标由图像身份后验提供：

\[
\tilde r_i^T
=
\operatorname{Norm}
\left[
 g_i^T-
\operatorname{sg}(\bar\mu_{p_i}^{I,-i})
\right].
\]

图像状态的目标由文本身份后验提供：

\[
\tilde r_i^I
=
\operatorname{Norm}
\left[
 g_i^I-
\operatorname{sg}(\bar\mu_{p_i}^{T,-i})
\right].
\]

残差对齐损失：

\[
L_{\mathrm{res}}
=
\frac12
\left[
1-\cos(r_i^I,\tilde r_i^I)
+
1-\cos(r_i^T,\tilde r_i^T)
\right].
\]

身份后验目标停止梯度，因此状态损失不能反向修改“什么是身份公共因素”。

---

# 14. 原配状态匹配损失

状态正关系只定义为当前原配图文：

\[
S_{ij}^{\mathrm{st}}=
(r_i^T)^\top r_j^I.
\]

对于文本查询 \(i\)：

- 原配图像 \(i\) 是正样本；
- 不同身份图像是负样本；
- 同身份但非原配图像既不当正样本，也不当负样本。

文本到图像状态损失：

\[
L_{\mathrm{pair}}^{t2i}
=
-rac1B\sum_i
\log
\frac{
\exp(S_{ii}^{\mathrm{st}}/\tau_0)
}
{
\exp(S_{ii}^{\mathrm{st}}/\tau_0)
+
\sum_{j:p_j\neq p_i}
\exp(S_{ij}^{\mathrm{st}}/\tau_0)
}.
\]

图像到文本方向同理：

\[
L_{\mathrm{pair}}
=
\frac12
\left(
L_{\mathrm{pair}}^{t2i}
+
L_{\mathrm{pair}}^{i2t}
\right).
\]

---

# 15. 同身份非伤害安全约束

状态差异不能被解释成身份错误。HIRE 不要求同身份不同图像具有高状态相似度，只要求状态头不要对真实同身份支持产生明显负分。

文本查询到同身份支持图像：

\[
P_{ia}^{t2i}
=
\operatorname{ReLU}
\left[-(r_i^T)^\top r_a^I\right].
\]

图像查询到同身份支持文本：

\[
P_{ia}^{i2t}
=
\operatorname{ReLU}
\left[-(r_i^I)^\top r_a^T\right].
\]

安全损失：

\[
L_{\mathrm{safe}}
=
\frac{1}{2|A|}
\sum_{i,a\in A_i}
\left(
P_{ia}^{t2i}+P_{ia}^{i2t}
\right).
\]

状态层总损失：

\[
L_{\mathrm{state}}
=
L_{\mathrm{pair}}
+
L_{\mathrm{res}}
+
L_{\mathrm{safe}}.
\]

该约束与此前失败的支持包拉近损失本质不同：支持样本不会被强制得到高分，只在状态头试图把真实同身份样本判成负证据时产生惩罚。

---

# 16. 最终身份—状态联合分数

单样本身份分数：

\[
S_{ij}^{\mathrm{id}}
=
-D(q_i^T,q_j^I).
\]

单样本状态分数：

\[
S_{ij}^{\mathrm{st}}
=
(r_i^T)^\top r_j^I.
\]

最终分数：

\[
S_{ij}^{\mathrm{final}}
=
\exp(\gamma_{\mathrm{id}})
S_{ij}^{\mathrm{id}}
+
\exp(\gamma_{\mathrm{st}})
S_{ij}^{\mathrm{st}}.
\]

初始化：

```text
exp(gamma_id) = 1.0
exp(gamma_state) = 0.1
```

两个尺度端到端学习并在计算时限制在安全范围内，不需要额外融合超参数。

## 16.1 基线保护初始化

HIRE 通过初始化使第一轮排序尽量接近 CLIP 全局基线：

1. 全局融合映射为单位矩阵；
2. 局部融合映射为零，因此初始观测就是全局 CLIP 表示；
3. 身份均值映射为单位矩阵；
4. 身份方差初始为 1；
5. 状态映射为单位矩阵，但状态尺度仅为 0.1。

当图像和文本均值经过层归一化且方差均为 1 时，高斯身份分数与全局余弦相似度只相差正比例缩放和查询无关常数，因此初始排序与 CLIP 排序近似等价。新模块从该安全起点联合学习，而不是用随机身份—状态分数破坏基础空间。

---

# 17. 最终稳定排序损失

对最终分数使用 RDE 风格的全部负样本排序思想。

身份标签构造正负矩阵：

\[
Y_{ij}=\mathbf 1[p_i=p_j],
\qquad
M_{ij}=1-Y_{ij}.
\]

对每个文本查询，以温度加权聚合同身份正样本：

\[
\alpha_{ij}
=
\frac{
Y_{ij}\exp(S_{ij}^{\mathrm{final}}/\tau)
}
{
\sum_kY_{ik}\exp(S_{ik}^{\mathrm{final}}/\tau)
}.
\]

正样本聚合分数：

\[
P_i=\sum_j\alpha_{ij}S_{ij}^{\mathrm{final}}.
\]

全部不同身份负样本的平滑上界：

\[
N_i
=
\tau\log
\sum_{j:p_j\neq p_i}
\exp(S_{ij}^{\mathrm{final}}/\tau).
\]

文本到图像方向：

\[
L_{\mathrm{TAL}}^{t2i}
=
\frac1B\sum_i
\left[m+N_i-P_i\right]_+.
\]

图像到文本方向同理：

\[
L_{\mathrm{joint}}
=
\frac12
\left(
L_{\mathrm{TAL}}^{t2i}
+
L_{\mathrm{TAL}}^{i2t}
\right).
\]

工程沿用 RDE 默认：

```text
TAL 温度 tau = 0.015
间隔 margin = 0.1
```

---

# 18. 单阶段总损失

代码只向现有训练循环返回三个带 `loss` 的聚合项：

\[
L_{\mathrm{total}}
=
L_{\mathrm{joint}}
+
L_{\mathrm{identity}}
+
L_{\mathrm{state}}.
\]

展开为：

\[
L_{\mathrm{total}}
=
L_{\mathrm{TAL-final}}
+
L_{\mathrm{set}}
+
L_{\mathrm{cal}}
+
L_{\mathrm{pair}}
+
L_{\mathrm{res}}
+
L_{\mathrm{safe}}.
\]

所有子项都按自身有效样本数求平均后直接相加，不设置人工损失权重。

训练从第一轮开始同时优化全部模块，不存在先训练身份头、再训练状态头的二阶段过程。

---

# 19. 梯度路径

## 19.1 主图文样本

主图文样本完整通过 CLIP 主干、词元选择、观测融合、身份后验和状态残差模块，主干获得梯度。

## 19.2 同身份支持样本

为控制显存和训练开销：

- 支持图像和支持文本经过 CLIP 主干时使用无梯度模式；
- 主干输出的词元与注意力被视为当前迭代的观测；
- RDE 词元选择、观测融合、身份均值头、方差头和状态头仍获得梯度。

每个训练样本在其他迭代中都会作为主锚点更新 CLIP 主干，因此支持样本并非永久冻结。

## 19.3 身份与状态隔离

- 身份集合损失更新身份均值空间和共享观测编码；
- 方差头读取停止梯度的观测；
- 状态残差目标中的组级身份均值停止梯度；
- 状态安全损失只作用于状态表示；
- 最终 TAL 同时校准身份与状态的可用性。

---

# 20. 训练流程

每个训练迭代执行：

1. 随机采样 64 条主图文样本，尽可能保留大量不同身份负样本；
2. 为每个主样本读取最多三组同身份、不同图像的支持图文；
3. 主样本经过 CLIP，取得全局词元、全部局部词元和最终注意力；
4. RDE 词元选择模块生成图像与文本细粒度观测；
5. 融合全局与细粒度观测；
6. 预测主样本身份均值、身份方差和状态残差；
7. 无梯度编码支持样本的 CLIP 词元，再有梯度计算其词元选择、融合、身份分布和状态表示；
8. 分别构造图像身份后验和文本身份后验；
9. 计算跨模态留一身份集合识别损失；
10. 计算不确定性校准损失；
11. 计算原配状态匹配、残差对齐和同身份安全损失；
12. 用单样本身份与状态分数形成最终分数矩阵；
13. 对最终分数计算全部负样本 TAL；
14. 三个聚合损失直接相加，单次反向传播。

---

# 21. 测试流程

测试时不需要身份标签、支持包或同身份多图信息。

对每条文本输出：

```text
身份均值 mu_T
身份方差 var_T
状态残差 r_T
```

对每张图像输出：

```text
身份均值 mu_I
身份方差 var_I
状态残差 r_I
```

两两计算身份概率分数和状态余弦分数，得到最终排序。

由于完整测试矩阵较大，代码按查询块和图库块计算高斯分数，默认：

```text
查询块：128
图库块：512
```

不会构造形状为 `[查询数, 图库数, 特征维度]` 的巨型张量。

---

# 22. 参数与超参数

## 22.1 新方法核心设置

```text
同身份支持数量：3
```

这是主版本唯一需要视为新方法超参数的设置。

## 22.2 继承 RDE 的固定设置

```text
词元选择比例：0.3
TAL 温度：0.015
TAL 间隔：0.1
```

主实验不搜索这些值。

## 22.3 固定结构维度

```text
CLIP 嵌入：512
RDE 词元选择输出：1024
融合观测：512
身份潜变量：512
状态残差：512
```

## 22.4 训练设置

```text
主干：OpenAI CLIP ViT-B/16
批次大小：64
训练轮数：60
采样器：随机采样
随机种子：1
图像增强：默认关闭
优化器、学习率和调度器：沿用当前仓库
新模块学习率：主干基础学习率的 5 倍
```

---

# 23. 与现有 TBPR 方法的区别

## 23.1 不同于强弱正样本

强弱正样本仍然直接优化查询与单个同身份样本之间的距离。HIRE 不把任何支持样本直接加入普通正样本集合；多个支持只共同估计潜在身份后验。

## 23.2 不同于身份原型

HIRE 后验不是训练身份的可学习参数，也不是固定类别中心。它由当前支持观测动态计算，具有逐维方差和组内异质性，可用于未见身份。

## 23.3 不同于支持集合蒸馏

HIRE 不把支持集合融合为查询表示，也不训练富模型再蒸馏到轻模型。支持集合只承担随机效应推断。

## 23.4 不同于两个全局投影头

身份头输出概率分布；状态头受组级后验残差目标约束。两者具有不同统计定义、不同正关系和不同损失，不再是两个名称不同但容易同构的线性头。

## 23.5 不同于 RDE

RDE 的全局与词元选择分支都是混合检索表示，核心问题是噪声图文对应。HIRE 只迁移其词元选择和 TAL 思想，删除 CCD、高斯混合样本筛选、噪声注入和 RDE 原始独立分数。

---

# 24. 工程文件对应关系

| 文件 | 作用 |
|---|---|
| `model/hire_components.py` | RDE 词元选择、观测融合、随机效应后验、概率分数和全部损失 |
| `model/hire_model.py` | HIRE 完整模型、CLIP 注意力适配、训练前向与分块测试 |
| `model/__init__.py` | 根据 `--hire` 路由到 HIRE，旧模型保持原样 |
| `datasets/build.py` | 为 HIRE 构造平衡的同身份不同图支持集合 |
| `utils/options.py` | HIRE 参数与互斥检查 |
| `utils/metrics.py` | HIRE 概率身份—状态联合分数评测 |
| `solver/build.py` | 主干基础学习率、新模块 5 倍学习率 |
| `processor/processor.py` | HIRE 损失、诊断量和检查点训练日志 |
| `run_hire_4090_tag.sh` | TAG-PEDES 六十轮正式训练 |
| `run_hire_smoke.sh` | 一轮端到端烟测 |
| `tools/hire/audit_hire.py` | 无数据、无权重的数学与代码审计 |
| `tests/test_hire_components.py` | 随机效应、概率匹配、状态与 TAL 单元测试 |

---

# 25. 运行命令

## 25.1 覆盖代码后审计

```bash
python tools/hire/audit_hire.py
pytest -q tests/test_hire_components.py
```

## 25.2 一轮烟测

```bash
DATA_ROOT=/root/autodl-tmp/datasets \
PYTHON_BIN=/root/miniconda3/envs/irra190/bin/python \
bash run_hire_smoke.sh
```

## 25.3 TAG-PEDES 正式六十轮

```bash
CUDA_VISIBLE_DEVICES=0 \
DATA_ROOT=/root/autodl-tmp/datasets \
OUTPUT_DIR=/root/autodl-tmp/HIRE_logs \
PYTHON_BIN=/root/miniconda3/envs/irra190/bin/python \
SEED=1 \
BATCH_SIZE=64 \
NUM_EPOCH=60 \
bash run_hire_4090_tag.sh
```

## 25.4 测试最佳检查点

```bash
python test.py --config_file <训练目录>/configs.yaml
```

现有 `test.py` 会通过 HIRE 自定义评测路径计算最终概率身份—状态联合分数。

---

# 26. 训练前验收标准

一轮烟测必须满足：

```text
成功构建 CLIP ViT-B/16；
支持图像和支持文本均来自同身份不同 image_id；
三项聚合损失均为有限值；
身份方差和组内异质性均为有限正数；
状态尺度和身份尺度有日志；
验证评测成功完成；
生成 best.pth；
无 NaN、Inf、显存溢出或异常退出。
```

建议在正式训练日志中重点观察：

```text
identity_set_nce
uncertainty_calibration
state_pair_nce
residual_alignment
state_safety
mean_image_variance
mean_text_variance
mean_group_heterogeneity
identity_scale
state_scale
```

如果方差迅速贴近数值下限、异质性长期为零或状态尺度异常爆炸，应停止正式训练并检查实现或输入关系。

---

# 27. 参考思想来源

本设计迁移并重新组合了以下领域的成熟思想，但核心身份—状态层级定义和异质性感知交集面向本项目重新设计：

1. RDE：注意力词元选择与全部负样本 TAL；
2. Neural Statistician：从样本集合推断可泛化组级统计量；
3. MLVAE：组共享内容潜变量与实例风格潜变量；
4. 概率人脸表征：均值—方差身份表示与不确定性感知匹配；
5. 随机效应元分析：观测内方差与观测间异质性的联合加权；
6. 传统重识别身份—姿态分解：组级公共因素与实例状态偏差。

HIRE 的关键区别是：

> 将空—地行人图文数据中的同身份多图定义为潜在身份随机效应的多次异质观测，通过跨模态留一可信交集估计身份后验，再将当前图文相对于该身份后验的偏差定义为状态残差，并以单阶段联合损失完成身份级和状态级匹配。
