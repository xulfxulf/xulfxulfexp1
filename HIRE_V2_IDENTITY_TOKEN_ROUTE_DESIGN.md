# HIRE-v2 v16.4.0：当前配对主体与组条件词元身份路由残差

**工程仓库：** `xulfxulf/xulfxulfexp1`  
**源主分支提交：** `07a9fe1cb86d5e223de23ccb02eb319f1eb4402d`  
**代码模式：** `--hire_v2 --hire_v2_mode identity_token_route`  
**实验版本：** `v16.4.0`  
**直接基线：** `v16.2.1`  
**不以其为基线的版本：** `v16.3.0` 状态晚交互负结果  
**主干：** OpenAI CLIP ViT-B/16  
**完整观测：** CLIP 全局观测 + RDE 风格词元选择 + 零初始化局部残差融合  
**身份基础：** v16.2.1 锚点平衡的严格留一身份组共识残差  
**本版本唯一新增机制：** 同身份支持图监督的文本词元可传播性路由  
**训练方式：** 从同一 CLIP 预训练权重开始单阶段训练，不加载 v16.2.1 或 v16.3.0 检查点  
**测试阶段：** 不使用同身份支持图，不运行多模态大模型，不做候选重排  
**是否引入新的可调方法超参数：** 否

---

# 一、为什么安排这一版

v16.3.0 已经验证：

```text
同图局部状态关系可以在训练批次中被学习；
但独立状态分数不能直接替换或重排身份基础；
状态最终结果低于同检查点身份基础；
状态修复 182 条、破坏 228 条；
状态门最终为负。
```

这说明下一版不应继续训练第二套状态分数。

v16.4.0 改用更保守的非对称结构：

```text
当前配对主体：
继续使用完整观测，保留所有对当前图文匹配有用的语义。

身份残差：
只允许模型预测为“跨同身份图像可传播”的文本词元进入。

非可传播词元：
不会被删除；
仍然完整保留在当前配对主体中；
只是不进入新增的身份词元残差。
```

这使方法与实际实验结论一致：

> 强图文基线负责原配图文和当前实例匹配；身份模块只补充跨图稳定信息。

---

# 二、为什么 v16.4.0 暂时不使用多模态大模型

用户尚未确认：

```text
语义短语如何切分；
双教师不一致如何处理；
支持、矛盾、未知如何映射到身份和状态；
教师标签覆盖范围；
是否生成反事实文本。
```

为了先得到可训练结果，v16.4.0 使用一个不依赖外部文件的在线关系教师。

在线教师只使用：

```text
当前锚点图像；
三张同身份、不同 image_id 的支持图；
批内完整观测最相似的一个不同身份图像；
当前文本中由已有 RDE 注意力规则选出的词元。
```

它生成连续的“身份可传播性”软目标。

本版本不声称在线目标等于最终的人工或多模态大模型身份／状态真值。它只用于快速验证：

> 将高可传播词元限制到身份残差，是否比 v16.2.1 的整句身份映射更有效。

若 v16.4.0 有效，后续再把在线目标替换或补充为多模态大模型三态标签；若无效，则不急于投入昂贵教师标注。

---

# 三、版本边界

## 3.1 完整保留的 v16.2.1 内容

```text
CLIP ViT-B/16 图像与文本编码；
RDE 风格全局—局部完整观测；
全局和局部锚点监督；
完整观测直接检索监督；
动态同身份支持图；
严格留一身份组简单共识；
共享身份映射；
严格身份组 NCE；
有界身份残差门；
完整观测与最终分数各占 0.5 的主检索目标；
标准单向量全图库推理。
```

## 3.2 v16.4.0 新增

```text
原始 CLIP 文本词元选择；
原始 CLIP 图像块选择；
批内最高完整观测不同身份图像；
词元—图像块 MaxSim 关系证据；
跨支持图一致性软目标；
文本词元身份可传播性路由器；
零初始化文本身份词元残差；
词元路由辅助损失。
```

## 3.3 明确不包含

```text
独立状态头；
状态晚交互最终分数；
状态门；
候选前五十重排；
多模态大模型标签；
支持、矛盾、未知三分类；
反事实文本；
支持文本；
离线困难负样本池；
身份分类器；
视角分类器；
图像质量标签；
图像增强；
二阶段训练。
```

---

# 四、符号定义

一个随机训练批次：

\[
\mathcal B
=
\{(I_i,T_i,p_i,g_i)\}_{i=1}^{B},
\]

其中：

- \(I_i\)：锚点图像；
- \(T_i\)：锚点文本；
- \(p_i\)：身份编号；
- \(g_i\)：图像编号 `image_id`；
- \(B=64\)。

第 \(i\) 个锚点还对应最多三张同身份不同图像支持：

\[
\mathcal S_i
=
\{I_{i,k}^{+}\}_{k=1}^{K_i},
\qquad
K_i\leq3.
\]

支持图必须满足：

\[
p(I_{i,k}^{+})=p_i,
\]

\[
g(I_{i,k}^{+})\neq g_i.
\]

向量归一化：

\[
\operatorname{Norm}(x)
=
\frac{x}{\max(\lVert x\rVert_2,\epsilon)}.
\]

---

# 五、当前配对主体保持不变

## 5.1 全局观测

图像全局观测：

\[
b_i^I
=
\operatorname{Norm}(x_{i,\mathrm{cls}}^I).
\]

文本全局观测：

\[
b_i^T
=
\operatorname{Norm}(x_{i,\mathrm{eot}}^T).
\]

全局分数：

\[
S_{ij}^{G}
=
(b_i^T)^\top b_j^I.
\]

## 5.2 RDE 风格局部观测

图像端根据类别词元最后一层注意力，选择前 `30%` 图像块。

文本端根据结束词元最后一层注意力，排除开始词元、结束词元和填充词元后选择前 `30%` 有效词元。

选中词元经过已有的两层映射、残差旁路和最大池化，形成：

\[
l_i^I,\qquad l_i^T.
\]

局部分数：

\[
S_{ij}^{L}
=
(l_i^T)^\top l_j^I.
\]

## 5.3 完整观测

零初始化局部残差融合：

\[
o_i^I
=
\operatorname{Norm}
(b_i^I+A_I l_i^I),
\]

\[
o_i^T
=
\operatorname{Norm}
(b_i^T+A_T l_i^T).
\]

完整观测分数：

\[
S_{ij}^{O}
=
(o_i^T)^\top o_j^I.
\]

完整观测就是本版本的当前配对主体。

它继续使用：

```text
全局身份代理；
当前衣着；
局部物品；
姿态；
视角；
当前可见细节；
其他对当前图文匹配有帮助的证据。
```

v16.4.0 不要求它成为纯状态空间。

---

# 六、原始词元关系证据

v16.4.0 使用已有 `hire_v2_select_ratio=0.3` 选择原始 CLIP 词元，不新增词元数量参数。

## 6.1 文本词元

设选出的有效文本词元为：

\[
\{t_{i,m}\}_{m=1}^{M_i}.
\]

其结束词元注意力经过掩码软最大化，得到：

\[
a_{i,m},
\qquad
\sum_m a_{i,m}=1.
\]

## 6.2 图像块

锚点图像选出的图像块：

\[
\{v_{i,n}^{0}\}_{n=1}^{N}.
\]

第 \(k\) 张支持图选出的图像块：

\[
\{v_{i,k,n}^{+}\}_{n=1}^{N}.
\]

图像块同样沿用类别词元注意力前 `30%` 规则。

原始词元和图像块在计算关系证据前全部二范数归一化。

---

# 七、批内最高不同身份图像

对每条文本，根据停止梯度后的完整观测分数，从当前随机批次选择一个最高相似的不同身份图像：

\[
j_i^{-}
=
\arg\max_{j:p_j\neq p_i}
S_{ij}^{O}.
\]

该图像记为：

\[
I_i^{-}.
\]

它只用于构造词元可传播性软目标。

它不是：

```text
离线困难负样本池；
跨批次队列；
额外采样器；
新的训练正负关系。
```

因此不增加数据预处理和候选数量超参数。

---

# 八、逐词元图像关系分数

## 8.1 锚点图像分数

文本词元 \(t_{i,m}\) 在锚点图中的最佳局部匹配：

\[
A_{i,m}
=
\max_n
t_{i,m}^{\top}v_{i,n}^{0}.
\]

## 8.2 支持图分数

在第 \(k\) 张同身份支持图中的最佳局部匹配：

\[
P_{i,m,k}
=
\max_n
t_{i,m}^{\top}v_{i,k,n}^{+}.
\]

支持图平均：

\[
\bar P_{i,m}
=
\frac1{K_i}
\sum_k P_{i,m,k}.
\]

支持图标准差：

\[
\sigma_{i,m}^{P}
=
\sqrt{
\frac1{K_i}
\sum_k
(P_{i,m,k}-\bar P_{i,m})^2
}.
\]

## 8.3 不同身份困难图分数

在批内最高不同身份图像中的最佳局部匹配：

\[
N_{i,m}
=
\max_n
t_{i,m}^{\top}v_{j_i^{-},n}.
\]

---

# 九、在线词元可传播性软目标

一个词元要成为身份残差证据，必须同时满足两项直觉。

第一，它对当前锚点图有辨识力：

\[
D_{i,m}^{\mathrm{pair}}
=
A_{i,m}-N_{i,m}.
\]

第二，它在同身份其他图像中保持稳定且仍然能够排除相似异身份：

\[
D_{i,m}^{\mathrm{stable}}
=
\bar P_{i,m}
-
\sigma_{i,m}^{P}
-
N_{i,m}.
\]

在同一条文本的有效词元内部进行标准化：

\[
\hat D_{i,m}
=
\frac{
D_{i,m}
-
\operatorname{Mean}_{r}(D_{i,r})
}{
\operatorname{Std}_{r}(D_{i,r})
+\epsilon
}.
\]

最终身份可传播性软目标：

\[
q_{i,m}^{\mathrm{id}}
=
\operatorname{sigmoid}
(\hat D_{i,m}^{\mathrm{stable}})
\cdot
\operatorname{sigmoid}
(\hat D_{i,m}^{\mathrm{pair}}).
\]

该目标没有人工阈值。

它的含义是：

```text
当前图中相关；
同身份支持图中持续相关；
支持图之间不高度冲突；
对批内最相似异身份仍有区分力。
```

只有满足以上组合的词元具有较高目标。

## 9.1 有效掩码

词元路由目标仅在以下条件同时满足时有效：

```text
该位置是有效文本词元；
当前身份至少有两张有效支持图；
批次中至少存在一个不同身份图像。
```

无效位置不进入词元路由损失。

## 9.2 状态和未知的处理

v16.4.0 暂不显式区分：

```text
当前状态词元；
不可观察词元；
不稳定或冲突词元；
通用无辨识力词元。
```

它们统一构成“非身份可传播”剩余部分。

这不是把它们删除。它们仍然全部保留在完整观测当前配对主体中，只是不进入新增的身份词元残差。

---

# 十、词元可传播性路由器

路由器接收：

```text
选中的原始文本词元；
当前文本完整观测；
```

并输出一个身份可传播概率。

对第 \(m\) 个词元：

\[
h_{i,m}
=
\operatorname{LN}
(t_{i,m}+\operatorname{sg}(o_i^T)).
\]

\[
r_{i,m}
=
W_2
\operatorname{GELU}
(W_1h_{i,m}).
\]

\[
\pi_{i,m}^{\mathrm{id}}
=
\operatorname{sigmoid}(r_{i,m}).
\]

其中：

- 隐藏维度固定为 `512/4=128`；
- 最后一层权重和偏置初始化为零；
- 初始输出严格为 `0.5`；
- 文本词元和完整观测输入均停止梯度。

词元路由损失：

\[
L_{\mathrm{route}}
=
-
\frac1{|\Omega|}
\sum_{(i,m)\in\Omega}
\left[
q_{i,m}^{\mathrm{id}}
\log\pi_{i,m}^{\mathrm{id}}
+
(1-q_{i,m}^{\mathrm{id}})
\log(1-\pi_{i,m}^{\mathrm{id}})
\right].
\]

该损失只直接训练词元路由器，不直接改变 CLIP 主干和完整观测。

---

# 十一、身份词元池化

词元身份权重：

\[
w_{i,m}^{\mathrm{id}}
=
a_{i,m}
\pi_{i,m}^{\mathrm{id}}.
\]

身份词元池：

\[
z_i^{T,\mathrm{id}}
=
\frac{
\sum_m
w_{i,m}^{\mathrm{id}}
t_{i,m}
}{
\sum_m
w_{i,m}^{\mathrm{id}}
+\epsilon
}.
\]

注意力 \(a_{i,m}\) 表示原模型认为词元对当前文本重要。

路由概率 \(\pi_{i,m}^{\mathrm{id}}\) 表示该词元是否适合跨图传播到身份残差。

---

# 十二、文本身份词元残差

v16.2.1 的共享身份映射原始输出：

\[
\tilde u_i^T
=
W_{\mathrm{id}}
\operatorname{sg}(o_i^T).
\]

v16.4.0 增加一个无偏置词元残差适配器：

\[
A_{\mathrm{token}}
\in
\mathbb R^{512\times512}.
\]

文本身份表示：

\[
u_i^T
=
\operatorname{Norm}
\left[
\tilde u_i^T
+
A_{\mathrm{token}}
z_i^{T,\mathrm{id}}
\right].
\]

初始化：

\[
A_{\mathrm{token}}=0.
\]

因此训练开始时：

\[
u_i^T
=
\operatorname{Norm}(\tilde u_i^T),
\]

严格等于 v16.2.1 的文本身份表示。

图像身份表示保持不变：

\[
u_j^I
=
\operatorname{Norm}
\left[
W_{\mathrm{id}}
\operatorname{sg}(o_j^I)
\right].
\]

同身份支持图身份组共识保持不变：

\[
C_i^I
=
\operatorname{Norm}
\left(
\frac1{K_i}
\sum_k u_{i,k}^I
\right).
\]

---

# 十三、身份组监督保持不变

文本身份表示与批内严格留一身份组共识计算：

\[
S_{ij}^{\mathrm{group}}
=
(u_i^T)^\top C_j^I.
\]

第 \(i\) 个查询只把第 \(i\) 个严格留一身份组作为正目标。

同身份非对角身份组忽略。

不同身份有效组作为负样本。

身份组损失仍记为：

\[
L_{\mathrm{group}}.
\]

支持图不进入普通图文正样本分子。

---

# 十四、最终检索分数

单样本身份分数：

\[
S_{ij}^{I}
=
(u_i^T)^\top u_j^I.
\]

完整观测当前配对分数：

\[
S_{ij}^{O}
=
(o_i^T)^\top o_j^I.
\]

身份门：

\[
\alpha
=
\operatorname{sigmoid}
(\theta_{\mathrm{id}}),
\qquad
\alpha_0=0.1.
\]

最终分数继续沿用 v16.2.1：

\[
S_{ij}^{F}
=
S_{ij}^{O}
+
\alpha
\left[
S_{ij}^{I}
-
\operatorname{sg}(S_{ij}^{O})
\right].
\]

前向数值等价于：

\[
S_{ij}^{F}
=
(1-\alpha)S_{ij}^{O}
+
\alpha S_{ij}^{I}.
\]

当前配对完整观测仍占主体。

身份词元路由只改变约一成身份残差内部使用的文本语义。

---

# 十五、总损失

v16.4.0 的总损失为：

\[
L_{\mathrm{v16.4.0}}
=
\frac12(L_G+L_L)
+
\frac12L_O
+
\frac12L_F
+
0.1L_{\mathrm{group}}
+
0.1L_{\mathrm{route}}.
\]

其中：

\[
L_G=L_{\mathrm{SDM}}(S^G)+L_{\mathrm{ITC}}(S^G),
\]

\[
L_L=L_{\mathrm{SDM}}(S^L)+L_{\mathrm{ITC}}(S^L),
\]

\[
L_O=L_{\mathrm{SDM}}(S^O)+L_{\mathrm{ITC}}(S^O),
\]

\[
L_F=L_{\mathrm{SDM}}(S^F)+L_{\mathrm{ITC}}(S^F).
\]

工程返回四个包含 `loss` 的项：

```text
sdm_loss
itc_loss
identity_group_loss
token_route_loss
```

其中：

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

token_route_loss
= 0.1 × token_route_bce
```

词元路由辅助复用已有 `0.1` 系数，不新增损失权重。

---

# 十六、初始化等价性

训练开始时：

\[
A_{\mathrm{token}}=0.
\]

因此：

\[
u_i^T(\mathrm{v16.4.0})
=
u_i^T(\mathrm{v16.2.1}).
\]

进一步有：

\[
S^I_{\mathrm{v16.4.0}}
=
S^I_{\mathrm{v16.2.1}},
\]

\[
S^F_{\mathrm{v16.4.0}}
=
S^F_{\mathrm{v16.2.1}}.
\]

主检索目标在初始化时严格等于 v16.2.1。

词元路由辅助在初期只训练路由器，因为：

```text
路由输入停止梯度；
在线目标停止梯度；
身份词元适配器为零，主检索损失尚不能通过该分支改变路由器。
```

随着身份词元适配器离开零点，最终检索和身份组监督才会联合调整身份词元残差。

---

# 十七、测试阶段

测试时只输入：

```text
一条查询文本；
一张图库图像。
```

文本执行：

```text
全局—局部完整观测；
文本词元选择；
词元路由器预测可传播概率；
身份词元池化；
文本身份残差；
最终单向量构造。
```

图像执行：

```text
全局—局部完整观测；
原有 v16.2.1 图像身份映射；
最终单向量构造。
```

测试时不使用：

```text
同身份支持图；
身份编号；
批内困难图像；
在线关系目标；
多模态大模型；
候选重排。
```

最终单向量：

\[
f^m
=
[
\sqrt{1-\alpha}\,o^m,
\sqrt{\alpha}\,u^m
].
\]

仍然使用标准全图库矩阵乘法评测和保存最佳检查点。

---

# 十八、方法超参数

v16.4.0 不新增需要搜索的方法超参数。

复用：

```text
词元／图像块选择比例：0.3；
同身份支持图数量：3；
辅助损失权重：0.1；
身份门初始值：0.1；
温度：0.02。
```

固定派生设置：

```text
路由隐藏维度：512/4=128；
每条查询一个批内最高完整观测不同身份图像；
身份适配器维度：512。
```

---

# 十九、单批次训练流程

1. 编码主批次图像和文本；
2. 生成全局、局部和完整观测；
3. 计算全局和局部锚点损失；
4. 计算完整观测检索损失；
5. 选择原始文本词元与锚点图像块；
6. 编码三张同身份支持图；
7. 生成支持图身份组共识；
8. 选择支持图原始图像块；
9. 从完整观测分数中选择一个批内最高不同身份图像；
10. 计算每个文本词元的锚点、支持组和不同身份图像 MaxSim；
11. 构造可传播性软目标；
12. 路由器预测身份可传播概率；
13. 计算词元路由二元交叉熵；
14. 根据注意力和路由概率池化身份词元；
15. 通过零初始化适配器修正文本身份表示；
16. 计算严格留一身份组 NCE；
17. 计算最终身份残差分数；
18. 计算最终检索损失；
19. 按固定公式求和；
20. 一次反向传播和一次优化器更新。

---

# 二十、训练日志必须记录

## 20.1 v16.2.1 基础

```text
global_sdm
global_itc
local_sdm
local_itc
observation_sdm
observation_itc
final_sdm
final_itc
identity_group_nce
identity_gate
support_valid_ratio
support_count_mean
identity_group_dispersion
identity_group_support_cosine
```

## 20.2 v16.4.0 词元路由

```text
token_route_loss
token_route_bce
token_route_valid_ratio
token_route_probability_mean
token_route_probability_std
token_route_target_mean
token_route_target_std
token_route_high_ratio
token_route_entropy
token_route_target_correlation
token_route_stable_margin
token_route_pair_margin
token_route_support_std
token_route_hard_negative_valid_ratio
token_route_selected_count
identity_token_residual_norm
identity_token_weight_sum
```

---

# 二十一、最佳检查点组件评测

最佳检查点分别输出：

```text
global
local
observation
identity
final
```

并额外输出测试文本上的：

```text
路由概率均值；
路由概率标准差；
高于 0.5 的词元比例；
路由熵；
每条文本身份词元残差范数。
```

---

# 二十二、成功标准

主结果最低要求：

```text
final R1 高于 v16.2.1 的 58.034；
final mAP 不低于 44.540；
final mINP 不出现明显下降；
final 高于同检查点 observation。
```

进入进一步教师版本的建议条件：

```text
final R1 至少达到 58.234；
mAP 不下降；
词元路由 BCE 正常下降；
预测与在线目标相关系数为正；
路由概率标准差不接近零；
高路由概率比例不接近 0 或 1；
身份词元残差范数离开零点；
最终身份残差修复数多于破坏数。
```

---

# 二十三、失败判定

若出现：

```text
路由概率长期约为 0.5；
预测标准差接近零；
在线目标与预测相关性接近零；
身份词元残差仍接近零；
最终结果不高于 v16.2.1；
```

则说明当前无教师关系目标不足以学习可迁移的词元路由。

此时不应立即：

```text
增加路由网络深度；
调整大量阈值；
提高辅助权重；
加入更多支持图；
```

而应根据离线诊断决定是否进入：

```text
多模态大模型支持／矛盾／未知教师；
语义短语级而非 CLIP 子词级路由；
反事实文本校准。
```

---

# 二十四、与 v16.3.0 的区别

| 项目 | v16.3.0 | v16.4.0 |
|---|---|---|
| 主体 | 身份基础后再做状态候选重排 | 完整配对主体保持不变 |
| 新分数 | 独立局部状态分数 | 不增加第二套状态分数 |
| 推理 | 前五十候选成对晚交互 | 标准单向量全图库检索 |
| 状态门 | 有，最终为负 | 无 |
| 同身份支持图 | 只服务身份组 | 服务身份组和离线式在线词元目标 |
| 新监督 | 同图状态 NCE | 词元可传播性 BCE |
| 非可传播词元 | 进入状态分数 | 留在配对主体，不进入身份词元残差 |
| 训练测试候选错位 | 存在 | 不存在 |

---

# 二十五、代码文件对应关系

| 设计部分 | 工程文件 |
|---|---|
| 原始词元选择、关系目标、路由器、词元残差与损失 | `model/hire_v2_token_route_components.py` |
| 完整 v16.4.0 前向和推理 | `model/hire_v2_identity_token_route_model.py` |
| 模式分发 | `model/__init__.py` |
| 模式参数注册 | `utils/options.py` |
| 复用身份支持数据集 | `datasets/build.py` |
| 路由诊断日志 | `processor/processor.py` |
| 组件离线评测 | `tools/hire_v2/eval_identity_token_route_components.py` |
| 数学与静态审计 | `tools/hire_v2/audit_identity_token_route.py` |
| 组件测试 | `tests/test_hire_v2_token_route_components.py` |
| 一轮烟测 | `run_hire_v2_identity_token_route_smoke.sh` |
| 正式训练 | `run_hire_v2_identity_token_route_4090_tag.sh` |

---

# 二十六、最终研究表述

v16.4.0 的可解释结构为：

\[
\boxed{
\text{当前配对完整观测}
+
\text{组条件词元身份残差}
}
\]

当前配对完整观测保留全部有效图文语义。

同身份支持图只负责训练一个可迁移的文本词元路由器，使身份残差优先使用：

```text
当前图中可辨认；
同身份其他图中持续出现；
支持图之间一致；
对相似异身份仍具区分力；
```

的词元。

本版本是对核心假设的快速、无外部教师验证，不是最终多模态大模型三态版本。
