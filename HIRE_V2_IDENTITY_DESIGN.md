# HIRE-v2 版本二：锚定式身份随机效应模型

**实验版本：** `v16.2.0`

**直接基线版本：** `v16.1.0`

**工程仓库：** `xulfxulf/xulfxulfexp1`  
**源主分支提交：** `09def7e11fe3a2f47b39929013aaad4038b98ac9`  
**版本一正式代码提交：** `48e61f81649aa2f3ea515d8e967faa4960b2f478`  
**代码模式：** `--hire_v2 --hire_v2_mode identity`  
**主干：** OpenAI CLIP ViT-B/16  
**固定观测锚点：** CLIP 全局观测 + RDE 风格词元选择 + 零初始化局部残差融合  
**本版本唯一研究增量：** 同身份多图的异质性感知身份随机效应  

---

# 1. 本版本定位

版本一在 TAG-PEDES 上已经建立了稳定的全局—局部完整观测基线：融合观测基本保持纯单头的一号检索率，同时明显提高平均精度和平均逆负惩罚。版本二不再修改版本一的全局编码、词元选择、残差融合与锚点监督，只增加身份级创新。

本版本只回答一个问题：

> 同身份不同图像能否不被当作普通强正样本或加权弱正样本，而是共同估计一个潜在身份随机效应，并在不破坏版本一完整观测空间的前提下改善身份级检索？

本版本明确不包含：

```text
状态头；
状态残差；
状态三态标签；
支持文本；
身份分类器；
困难负包；
视角分类器；
MLLM 教师标签；
方差校准回归；
原型参数表；
测试阶段支持集合。
```

因此，版本二相对版本一只有一个机制变量：身份随机效应。

---

# 2. 版本一固定观测锚点

## 2.1 CLIP 全局观测

设第 \(i\) 个训练样本为：

\[
(I_i,T_i,p_i,g_i),
\]

其中 \(I_i\) 为图像，\(T_i\) 为原配文本，\(p_i\) 为身份编号，\(g_i\) 为图像编号。

CLIP 图像编码器输出类别词元和图像块词元：

\[
X_i^I=[x_{i,\mathrm{cls}}^I,x_{i,1}^I,\ldots,x_{i,N}^I].
\]

CLIP 文本编码器输出开始词元、文本词元和结束词元：

\[
X_i^T=[x_{i,\mathrm{sos}}^T,x_{i,1}^T,\ldots,x_{i,\mathrm{eot}}^T,\ldots].
\]

全局图像观测：

\[
b_i^I=\operatorname{Norm}(x_{i,\mathrm{cls}}^I).
\]

全局文本观测：

\[
b_i^T=\operatorname{Norm}(x_{i,\mathrm{eot}}^T).
\]

其中：

\[
\operatorname{Norm}(x)=\frac{x}{\|x\|_2}.
\]

全局相似度：

\[
S_{ij}^{G}=(b_i^T)^\top b_j^I.
\]

## 2.2 RDE 风格细粒度观测

图像端使用最后一层类别词元注意力，从图像块中选择注意力最高的前 \(30\%\)；文本端使用结束词元注意力，排除开始词元、结束词元和填充词元后选择注意力最高的有效文本词元。

选中词元经过：

\[
h=\operatorname{MLP}(\hat x)+W_{\mathrm{skip}}\hat x,
\]

再做逐维最大池化，得到图像和文本细粒度观测：

\[
l_i^I,\qquad l_i^T.
\]

细粒度相似度：

\[
S_{ij}^{L}=(l_i^T)^\top l_j^I.
\]

## 2.3 零初始化残差融合

局部观测通过零初始化适配器映射到 512 维：

\[
e_i^I=A_I l_i^I,
\qquad
e_i^T=A_T l_i^T.
\]

完整混合观测：

\[
g_i^I=\operatorname{Norm}(b_i^I+e_i^I),
\]

\[
g_i^T=\operatorname{Norm}(b_i^T+e_i^T).
\]

由于 \(A_I\) 与 \(A_T\) 零初始化，训练开始时严格满足：

\[
g_i^I=b_i^I,
\qquad
g_i^T=b_i^T.
\]

完整观测分数：

\[
S_{ij}^{O}=(g_i^T)^\top g_j^I.
\]

---

# 3. 观测锚点损失保持不变

全局和细粒度分支分别继续使用版本一的检索目标：

\[
L_{\mathrm{ret}}(S)=L_{\mathrm{SDM}}(S)+L_{\mathrm{ITC}}(S).
\]

全局损失：

\[
L_G=L_{\mathrm{ret}}(S^G).
\]

细粒度损失：

\[
L_L=L_{\mathrm{ret}}(S^L).
\]

观测锚点损失：

\[
L_{\mathrm{anchor}}=\frac12(L_G+L_L).
\]

与版本一不同的是，版本二不再单独使用 \(L_{\mathrm{ret}}(S^O)\) 作为最终主损失，而是用身份残差后的最终分数替代它。训练开始时最终分数严格等于 \(S^O\)，因此版本二初始目标与版本一一致。

---

# 4. 层级随机效应假设

本版本采用以下层级假设：

\[
g_{p,a}^{m}=u_p+\delta_{p,a}^{m}+\epsilon_{p,a}^{m},
\]

其中：

- \(p\) 表示身份；
- \(a\) 表示该身份的一次图像或文本观测；
- \(m\in\{I,T\}\) 表示图像或文本模态；
- \(u_p\) 表示同身份多图背后的组级身份随机效应；
- \(\delta_{p,a}^{m}\) 表示视角、姿态、遮挡、可见性和描述选择等实例变化；
- \(\epsilon_{p,a}^{m}\) 表示模态噪声和标注噪声。

版本二只估计 \(u_p\)，不显式建模 \(\delta\)。状态实例效应留给下一版本。

---

# 5. 单样本身份均值

完整观测经过图像与文本共享的身份映射：

\[
\mu_i^m=\operatorname{Norm}(W_{\mathrm{id}}g_i^m),
\qquad m\in\{I,T\}.
\]

其中：

- \(W_{\mathrm{id}}\in\mathbb R^{512\times512}\)；
- 图像与文本共享同一个矩阵；
- 权重初始化为单位矩阵；
- 不使用偏置。

所以训练开始时：

\[
\mu_i^I=g_i^I,
\qquad
\mu_i^T=g_i^T.
\]

## 5.1 锚点梯度隔离

身份映射读取停止梯度后的完整观测：

\[
\mu_i^m=\operatorname{Norm}
\left[
W_{\mathrm{id}}\operatorname{sg}(g_i^m)
\right].
\]

其含义是：

```text
全局—局部观测空间继续只由版本一锚点目标和最终检索目标训练；
身份组级辅助目标只更新共享身份映射和不确定性模块；
身份创新不能通过辅助损失直接扭曲已经验证有效的观测空间。
```

这是一项有意的保护设计，不是两阶段训练。所有模块从第一轮同时训练，但梯度职责不同。

---

# 6. 支持图像构造

## 6.1 主批次

主批次保持版本一设置：

```text
随机采样；
批次大小 64；
不使用大 P 小 K；
不改变不同身份负样本结构。
```

## 6.2 支持集合

对每个主样本额外选择最多三张支持图像：

```text
同一身份；
不同 image_id；
严格排除锚点原图；
不读取支持文本；
不把支持图当作普通正样本；
不进入 SDM 或 ITC 的正样本分子。
```

支持选择规则：

1. 若存在跨视角图像，优先选择一张跨视角图；
2. 若存在同视角不同图像，选择一张同视角图；
3. 第三张从剩余不同图像中补充；
4. 某类不足时由另一类补齐；
5. 同一支持集合不得重复 image_id。

支持数量：

\[
K=3.
\]

这是本版本第一个新增方法超参数。

## 6.3 按轮动态轮换

支持选择由以下量确定：

```text
固定随机种子；
当前 epoch；
锚点 image_id。
```

因此：

- 同一实验完全可复现；
- 同一锚点在不同训练轮次可看到更多同身份图像；
- 不再长期绑定固定三张支持图。

## 6.4 有效身份组

至少具有两张有效支持图像时，才估计组内异质性并计算身份组级损失：

\[
K_i\ge2.
\]

只有一张或没有支持图时，该查询仍参与全局、局部和最终检索训练，但不参与身份组级辅助目标。

---

# 7. 支持图像逐维不确定性

每张支持图像的完整观测 \(g_{i,a}^I\) 经过身份映射得到均值：

\[
\mu_{i,a}^{I}=\operatorname{Norm}
\left[
W_{\mathrm{id}}\operatorname{sg}(g_{i,a}^I)
\right].
\]

支持图像同时预测逐维方差：

\[
q_{i,a}=W_\sigma\operatorname{sg}(g_{i,a}^I)+b_\sigma.
\]

\[
(\sigma_{i,a}^{I})^2
=
\sigma_{\min}^2+
(\sigma_{\max}^2-\sigma_{\min}^2)
\operatorname{sigmoid}(q_{i,a}).
\]

固定数值范围：

\[
\sigma_{\min}^2=0.1,
\qquad
\sigma_{\max}^2=2.0.
\]

方差头权重和偏置初始化为零，所以初始方差为：

\[
1.05.
\]

方差只用于支持图可信交集，不直接进入测试阶段的单图排序。

---

# 8. 异质性感知可信交集

设某个锚点的有效支持图像集合为：

\[
\mathcal A_i=\{1,\ldots,K_i\}.
\]

对身份维度 \(d\)，单图初始精度为：

\[
\lambda_{i,a,d}^{(0)}
=
\frac1{(\sigma_{i,a,d}^{I})^2+\varepsilon}.
\]

初始组中心：

\[
c_{i,d}^{(0)}
=
\frac{
\sum_{a\in\mathcal A_i}
\lambda_{i,a,d}^{(0)}\mu_{i,a,d}^{I}
}{
\sum_{a\in\mathcal A_i}
\lambda_{i,a,d}^{(0)}
}.
\]

组内异质性：

\[
\tau_{i,d}^{2}
=
\frac{
\sum_{a\in\mathcal A_i}
\lambda_{i,a,d}^{(0)}
\left(
\mu_{i,a,d}^{I}-
\operatorname{sg}(c_{i,d}^{(0)})
\right)^2
}{
\sum_{a\in\mathcal A_i}
\lambda_{i,a,d}^{(0)}
}.
\]

最终有效精度：

\[
\lambda_{i,a,d}
=
\frac1{
(\sigma_{i,a,d}^{I})^2+
\tau_{i,d}^{2}+
\varepsilon
}.
\]

身份组级后验均值：

\[
\bar\mu_{i,d}^{I}
=
\frac{
\sum_{a\in\mathcal A_i}
\lambda_{i,a,d}\mu_{i,a,d}^{I}
}{
\sum_{a\in\mathcal A_i}
\lambda_{i,a,d}
}.
\]

最终可信身份交集：

\[
C_i^I=\operatorname{Norm}(\bar\mu_i^I).
\]

该设计同时考虑：

```text
观测内不确定性：单张图自己是否可靠；
观测间异质性：同身份不同图是否在该维度达成一致。
```

只有单图自身可靠且多图之间一致的维度，才会在身份后验中具有较高有效精度。

---

# 9. 严格留一身份组级损失

文本查询身份均值与图像身份交集的分数：

\[
S_{ij}^{\mathrm{group}}
=(\mu_i^T)^\top C_j^I.
\]

## 9.1 对角身份组是唯一正目标

第 \(i\) 个身份组 \(C_i^I\) 由第 \(i\) 个锚点的同身份、不同 image_id 支持图构造，因此严格排除了查询原配图像。

若主批次中恰好出现另一个相同身份样本 \(j\)，其支持集合可能包含第 \(i\) 个查询的原配图像。为避免隐式泄漏：

```text
列 i 是行 i 的唯一正身份组；
相同 PID 的非对角身份组全部忽略；
不同 PID 的有效身份组作为负样本。
```

定义允许矩阵：

\[
M_{ij}
=
\mathbf1[i=j]
+
\mathbf1[p_i\ne p_j].
\]

对角身份组无效或没有不同身份负组时，该行不计算组级损失。

组级损失：

\[
L_{\mathrm{group}}
=
-\frac1{|\mathcal V|}
\sum_{i\in\mathcal V}
\log
\frac{
\exp(S_{ii}^{\mathrm{group}}/\tau)
}{
\exp(S_{ii}^{\mathrm{group}}/\tau)
+
\sum_{j:p_j\ne p_i,\,C_j^I\text{有效}}
\exp(S_{ij}^{\mathrm{group}}/\tau)
}.
\]

这是单方向文本到图像身份组监督，不构造支持文本方向。

---

# 10. 单样本身份残差

单样本文本—图像身份分数：

\[
S_{ij}^{\mathrm{id}}
=(\mu_i^T)^\top\mu_j^I.
\]

完整观测分数：

\[
S_{ij}^{O}
=(g_i^T)^\top g_j^I.
\]

身份残差：

\[
\Delta_{ij}^{\mathrm{id}}
=
S_{ij}^{\mathrm{id}}
-
\operatorname{sg}(S_{ij}^{O}).
\]

身份门：

\[
\alpha=\operatorname{sigmoid}(\theta_{\mathrm{id}}),
\qquad
0<\alpha<1.
\]

身份门初始化为：

\[
\alpha_0=0.1.
\]

最终分数：

\[
S_{ij}^{\mathrm{final}}
=
S_{ij}^{O}
+
\alpha\Delta_{ij}^{\mathrm{id}}.
\]

其数值等价于：

\[
S_{ij}^{\mathrm{final}}
=
(1-\alpha)S_{ij}^{O}
+
\alpha S_{ij}^{\mathrm{id}}.
\]

但训练梯度并不完全等同于普通凸组合：被减去的观测分数停止梯度，因此最终主损失对完整观测路径保留完整梯度，身份映射只承担残差学习。

由于 \(W_{\mathrm{id}}\) 单位初始化，训练开始时：

\[
S^{\mathrm{id}}=S^O,
\]

所以：

\[
S^{\mathrm{final}}=S^O.
\]

版本二在第一步严格继承版本一检索行为。

---

# 11. 最终检索损失

最终分数矩阵直接计算版本一相同的两种检索目标：

\[
L_{\mathrm{final}}
=
L_{\mathrm{SDM}}(S^{\mathrm{final}})
+
L_{\mathrm{ITC}}(S^{\mathrm{final}}).
\]

全局与局部锚点保持：

\[
L_{\mathrm{anchor}}
=
\frac12
\left[
L_{\mathrm{ret}}(S^G)
+
L_{\mathrm{ret}}(S^L)
\right].
\]

本版本总损失：

\[
L_{\mathrm{V2}}
=
L_{\mathrm{anchor}}
+
L_{\mathrm{final}}
+
\lambda L_{\mathrm{group}}.
\]

固定：

\[
\lambda=0.1.
\]

展开到工程返回键：

```text
sdm_loss
= 0.5 × (global_sdm + local_sdm) + final_sdm

itc_loss
= 0.5 × (global_itc + local_itc) + final_itc

identity_group_loss
= 0.1 × identity_group_nce
```

训练循环只求和以上三个损失键。

---

# 12. 推理阶段

测试阶段不使用支持图像、训练身份或身份组后验。

图像和文本分别输出：

```text
完整观测 g；
单样本身份均值 μ。
```

为了与标准余弦检索评测兼容，构造最终拼接表示：

\[
f^m
=
\left[
\sqrt{1-\alpha}\,g^m,
\sqrt{\alpha}\,\mu^m
\right].
\]

其点积严格等于：

\[
(f_i^T)^\top f_j^I
=
(1-\alpha)S_{ij}^{O}
+
\alpha S_{ij}^{\mathrm{id}}.
\]

因此标准评测器无需特殊矩阵逻辑即可准确评测最终分数。

---

# 13. 梯度职责

## 13.1 CLIP、词元选择与观测融合

由以下目标更新：

```text
全局锚点损失；
局部锚点损失；
最终检索损失中的完整观测主路径。
```

不接收身份组级辅助损失梯度。

## 13.2 共享身份映射

由以下目标更新：

```text
最终检索损失中的身份残差；
身份组级辅助损失。
```

## 13.3 图像不确定性头

仅由身份组级辅助损失更新。

## 13.4 身份门

由最终检索损失自动学习。

这种梯度分工保证：

> 版本一完整观测保持为稳定固定效应，身份创新作为受约束的组级随机效应残差进入，而不是重新训练一套会覆盖锚点的检索空间。

---

# 14. 新增超参数

本版本新增方法超参数只有两个：

| 参数 | 数值 | 作用 |
|---|---:|---|
| 同身份支持图数量 | 3 | 构造身份可信交集 |
| 身份组级辅助权重 | 0.1 | 防止辅助身份目标压过检索主任务 |

以下属于固定数值保护，不进行搜索：

```text
最少有效支持数：2；
身份门初始值：0.1；
方差下界：0.1；
方差上界：2.0；
词元选择比例：0.3，沿用版本一/RDE；
温度：0.02，沿用版本一。
```

---

# 15. 单批次训练流程

每个训练迭代严格执行：

1. 随机采样 64 个主图文样本；
2. 用 CLIP 编码主图像与主文本；
3. 计算全局观测和细粒度观测；
4. 计算全局与局部锚点损失；
5. 通过零初始化残差融合得到完整观测；
6. 从停止梯度的完整观测得到图像和文本身份均值；
7. 读取每个锚点最多三张同身份不同图像支持；
8. 支持图像的 CLIP、词元选择和观测融合路径使用无梯度编码；支持图像经过词元选择时临时使用已学习的批归一化运行统计，不更新版本一锚点的批归一化状态；
9. 支持完整观测经过可训练共享身份映射和图像不确定性头；
10. 构造异质性感知可信身份交集；
11. 计算严格留一的文本到图像身份组损失；
12. 计算完整观测分数、单样本身份分数和身份残差最终分数；
13. 计算最终 SDM 与 ITC；
14. 按公式聚合三项损失并反向传播一次。

---

# 16. 训练设置

与版本一保持一致：

```text
数据集：TAG-PEDES；
主干：OpenAI CLIP ViT-B/16；
图像尺寸：384×128；
训练轮数：60；
批次大小：64；
随机种子：1；
主批次采样器：random；
图像增强开关：关闭；
验证划分：test，与历史内部对照一致；
优化器：Adam；
主干学习率：1e-5；
余弦学习率调度；
新模块学习率倍率：5。
```

注意：当前仓库在 `img_aug=False` 时仍保留随机水平翻转，这是历史基线的数据变换行为，本版本不修改，以保持严格公平。

---

# 17. 必须记录的训练统计

## 17.1 主损失

```text
sdm_loss；
itc_loss；
identity_group_loss；
identity_group_nce。
```

## 17.2 锚点统计

```text
global_sdm；
global_itc；
local_sdm；
local_itc；
anchor_objective；
image_local_residual_norm；
text_local_residual_norm。
```

## 17.3 身份统计

```text
final_sdm；
final_itc；
final_objective；
identity_gate；
identity_score_delta_abs；
observation_identity_cosine；
identity_projection_delta_norm。
```

## 17.4 支持后验统计

```text
support_valid_ratio；
support_count_mean；
mean_image_variance；
variance_low_ratio；
variance_high_ratio；
mean_group_heterogeneity。
```

理想训练现象：

```text
identity_gate 不应永久停在初值，也不应迅速饱和到 1；
identity_projection_delta_norm 应从 0 平稳增加；
observation_identity_cosine 应保持较高但低于完全同构；
方差不能全部压到上下界；
identity_group_nce 应下降；
final 结果应至少保持 observation 基线。
```

---

# 18. 离线组件评测

最佳检查点必须分别输出：

| 分数 | 含义 |
|---|---|
| global | CLIP 全局观测 |
| local | RDE 风格细粒度观测 |
| observation | 版本一完整混合观测 |
| identity | 单样本身份均值 |
| final | 完整观测与身份残差的有界联合 |

每套报告：

```text
R1；
R5；
R10；
mAP；
mINP。
```

关键判断：

```text
identity 高于 observation：身份映射独立具有收益；
identity 低于 observation、final 高于 observation：身份残差具有互补性；
final 低于 observation：身份创新当前有害；
final 的 mAP/mINP 提升而 R1 不降：符合身份级建模目标。
```

---

# 19. 准入下一版本的标准

版本二只有满足以下条件，才进入状态创新：

```text
final R1 相对版本一 observation 下降不超过 0.15；
final mAP 或 mINP 至少提高 0.30；
identity_group_nce 明显下降；
identity_gate 未退化到接近 0 或 1；
方差上下界命中率没有大面积饱和；
支持组有效率符合数据统计；
跨视角同身份正确图像平均排名改善。
```

理想目标：

```text
final R1 高于 57.774；
final mAP 高于 44.238；
final mINP 高于 24.113。
```

如果版本二明显下降，只允许一次定向回退：删除预测方差，保留纯组内异质性可信交集。暂不搜索支持数量、辅助权重或身份门初值。

---

# 20. 与此前支持包方法的区别

此前支持包方法：

```text
把同身份支持图作为正样本；
直接要求查询文本与支持图靠近；
支持关系进入全局检索空间；
容易把当前状态差异错误传播为身份一致性。
```

本版本：

```text
支持图不作为普通正样本；
支持图之间先形成潜在身份统计量；
查询匹配的是留一身份交集，而不是单张支持图；
单图不确定性和多图异质性共同决定贡献；
身份辅助梯度不能直接改写观测锚点。
```

因此，本版本不是强弱正样本加权，也不是支持包拉近。

---

# 21. 与第一版完整 HIRE 的区别

第一版完整 HIRE 的主要问题包括：

```text
概率身份分数完全替代成熟观测分数；
图像与支持文本双向构造后验；
方差校准和残差回归占据大部分训练量；
状态自由头与身份头共同重建检索空间；
可学习无界尺度配合间隔损失快速饱和。
```

版本二修正为：

```text
版本一完整观测始终是固定效应锚点；
只使用支持图像；
只增加身份组级辅助目标；
方差只做支持可信度，不直接改测试分数；
最终身份分数是有界残差；
没有状态分支；
没有方差校准和残差回归；
辅助损失固定乘 0.1。
```

---

# 22. 代码文件与职责

```text
model/hire_v2_identity_components.py
    身份均值、不确定性、可信交集、组级损失、分数矩阵损失、最终拼接表示。

model/hire_v2_identity_model.py
    主模型前向、支持图编码、损失聚合、训练诊断、推理接口。

datasets/hire_v2_identity_dataset.py
    动态同身份支持图选择与严格关系校验。

datasets/build.py
    identity 模式使用专用训练数据集。

model/__init__.py
    anchor 与 identity 模式分发。

utils/options.py
    新增 identity 模式、支持数量和辅助权重参数。

processor/processor.py
    每轮调用 set_epoch，并记录身份版诊断。

solver/build.py
    新模块继续使用已有 5 倍学习率规则。

tools/hire_v2/eval_identity_components.py
    五套分数组件离线评测。

tools/hire_v2/audit_identity.py
    数学、初始化和交付路径审计。

tests/test_hire_v2_identity_components.py
    身份组件单元测试。

run_hire_v2_identity_4090_tag.sh
    正式 60 轮启动脚本。

run_hire_v2_identity_smoke.sh
    一轮烟测脚本。
```

---

# 23. 执行命令

## 23.1 静态审计

```bash
python tools/hire_v2/audit_identity.py
```

## 23.2 单元测试

```bash
pytest -q \
  tests/test_hire_v2_anchor_components.py \
  tests/test_hire_v2_identity_components.py
```

## 23.3 一轮烟测

```bash
CUDA_VISIBLE_DEVICES=0 \
DATA_ROOT=/root/autodl-tmp/datasets \
OUTPUT_DIR=/root/autodl-tmp/HIRE_v2_identity_smoke \
PYTHON_BIN=/root/miniconda3/envs/irra190/bin/python \
NUM_EPOCH=1 \
SEED=1 \
BATCH_SIZE=64 \
SUPPORT_SIZE=3 \
AUX_WEIGHT=0.1 \
bash run_hire_v2_identity_smoke.sh
```

## 23.4 正式训练

```bash
CUDA_VISIBLE_DEVICES=0 \
DATA_ROOT=/root/autodl-tmp/datasets \
OUTPUT_DIR=/root/autodl-tmp/HIRE_v2_identity_logs \
PYTHON_BIN=/root/miniconda3/envs/irra190/bin/python \
NUM_EPOCH=60 \
SEED=1 \
BATCH_SIZE=64 \
SUPPORT_SIZE=3 \
AUX_WEIGHT=0.1 \
EXP_NAME=hire_v2_identity_tagpedes_60e_seed1 \
bash run_hire_v2_identity_4090_tag.sh
```

## 23.5 最佳检查点组件评测

```bash
python tools/hire_v2/eval_identity_components.py \
  --config-file <实验目录>/configs.yaml \
  --checkpoint <实验目录>/best.pth
```

默认输出：

```text
<实验目录>/hire_v2_identity_components.json
```

---

# 24. 训练前验收

正式训练前必须确认：

```text
anchor 模式仍能正常构建；
identity 模式构建 HIREV2Identity；
支持图 pid 全部等于 anchor pid；
支持图 image_id 全部不同于 anchor image_id；
支持集合内 image_id 不重复；
不同 epoch 的支持集合会轮换；
初始 identity 与 observation 完全相同；
初始 final 分数与 observation 完全相同；
方差位于 [0.1, 2.0]；
组级损失有限且可反向传播；
身份辅助梯度不进入观测锚点；
标准评测 final 点积与公式一致。
```

---

# 25. 本版本的最终研究表述

版本二可以概括为：

> 我们将同身份不同图像重新解释为潜在身份随机效应的多次异质观测，而不是普通强正或弱正图文对。模型通过观测内不确定性与观测间异质性形成可信身份交集，并用严格留一的文本到图像身份组监督学习共享身份映射。身份随机效应以有界残差形式修正强完整观测，同时辅助身份梯度与观测锚点隔离，从而在保护实例检索能力的前提下补充身份级监督。
