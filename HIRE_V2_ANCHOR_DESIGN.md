# HIRE-v2 版本一：锚定式完整观测基线

**实验版本：** `v16.1.0`

**工程基线：** `xulfxulf/xulfxulfexp1@610c2a405aec4acfdb0d6364872ec4f86d17c588`  
**代码模式：** `--hire_v2 --hire_v2_mode anchor`  
**主干：** （OpenAI CLIP ViT-B/16）  
**细粒度模块：** （RDE）风格注意力词元选择  
**训练方式：** 单阶段端到端训练  
**测试输入：** 单条文本和单张图库图像  
**本版本不包含：** 同身份支持图、概率身份后验、可信交集、身份—状态分配门、状态残差

---

# 1. 版本定位

此前完整层级模型已经验证了同身份多图后验可能增强困难正样本的一致性，但一号检索率明显下降。进一步分析表明，在进入身份随机效应和状态实例残差之前，必须先保证作为“完整混合观测”的基础空间本身足够强，并且其全局和细粒度检索能力受到显式保护。

因此，版本一只回答一个问题：

> 在不引入任何身份包和状态分解的情况下，（CLIP）全局语义、（RDE）风格细粒度语义以及零初始化残差融合，能否形成不低于现有纯单头基线的强完整观测空间？

版本一不是最终创新模型，而是后续身份随机效应版本和身份—状态完整版本的必要准入基线。

---

# 2. 设计原则

版本一遵循以下原则。

1. 保留（CLIP ViT-B/16）预训练图文语义空间。
2. 保留（RDE）中“根据最后一层注意力选择高信息词元”的细粒度建模思想。
3. 全局分支和细粒度分支分别接受直接图文检索监督。
4. 融合观测也接受直接图文检索监督。
5. 融合使用零初始化残差，使训练初始时融合观测严格等于归一化后的（CLIP）全局表示。
6. 不使用层归一化改写预训练全局几何。
7. 不使用同身份支持包，不改变数据采样器，不引入新的身份级关系。
8. 不使用（RDE）的可信一致划分、噪声注入、高斯混合模型或逐轮全数据重估。
9. 最佳检查点只根据融合观测的一号检索率保存。
10. 最佳检查点离线额外报告全局、细粒度和融合观测三套结果。

---

# 3. 符号定义

设训练批次大小为 \(B\)。第 \(i\) 个训练样本包括：

\[
(I_i,T_i,p_i),
\]

其中：

- \(I_i\) 为图像；
- \(T_i\) 为原配文本；
- \(p_i\) 为身份编号。

（CLIP）图像编码器输出：

\[
X_i^I=
[x_{i,\mathrm{cls}}^I,x_{i,1}^I,\ldots,x_{i,N}^I].
\]

（CLIP）文本编码器输出：

\[
X_i^T=
[x_{i,\mathrm{sos}}^T,x_{i,1}^T,\ldots,x_{i,\mathrm{eot}}^T,\ldots].
\]

最后一层图像和文本自注意力矩阵分别记为：

\[
A_i^I\in\mathbb R^{(N+1)\times(N+1)},
\]

\[
A_i^T\in\mathbb R^{L\times L}.
\]

向量二范数归一化定义为：

\[
\operatorname{Norm}(x)=\frac{x}{\|x\|_2}.
\]

---

# 4. （CLIP）全局观测

图像全局观测使用类别词元：

\[
b_i^I
=
\operatorname{Norm}
\left(
 x_{i,\mathrm{cls}}^I
\right).
\]

文本全局观测使用结束词元：

\[
b_i^T
=
\operatorname{Norm}
\left(
 x_{i,\mathrm{eot}}^T
\right).
\]

全局图文相似度矩阵：

\[
S_{ij}^{G}
=
(b_i^T)^\top b_j^I.
\]

该分支不增加任何投影头和层归一化，直接保留预训练（CLIP）余弦几何。

---

# 5. （RDE）风格细粒度词元选择

## 5.1 图像词元选择

取图像最后一层自注意力中类别词元对局部图像块的注意力：

\[
a_{i,n}^{I}
=
A_{i,\mathrm{cls},n}^{I}.
\]

排除类别词元后，选择注意力最高的前 \(K_I\) 个图像块：

\[
K_I
=
\max
\left(
1,
\left\lfloor \rho N\right\rfloor
\right),
\]

其中：

\[
\rho=0.3.
\]

该比例沿用（RDE）的公开设置，不作为本项目新增超参数。本实现保留（RDE）的注意力选择、残差映射和最大池化思想，同时对文本填充词元采用更严格的掩码处理。

对选中的局部块先做二范数归一化：

\[
\hat x_{i,n}^{I}
=
\operatorname{Norm}
\left(
 x_{i,n}^{I}
\right).
\]

经过线性旁路和两层感知机：

\[
h_{i,n}^{I}
=
\operatorname{MLP}_{I}
\left(
\hat x_{i,n}^{I}
\right)
+
W_{\mathrm{skip}}^{I}
\hat x_{i,n}^{I}.
\]

对选中词元逐维最大池化：

\[
\tilde l_i^I
=
\max_{n\in\operatorname{TopK}(a_i^I)}
 h_{i,n}^{I}.
\]

最终图像细粒度观测：

\[
l_i^I
=
\operatorname{Norm}
\left(
\tilde l_i^I
\right).
\]

默认细粒度维度为：

\[
D_L=1024.
\]

## 5.2 文本词元选择

取文本结束词元对其他文本词元的最后一层注意力：

\[
a_{i,m}^{T}
=
A_{i,\mathrm{eot},m}^{T}.
\]

以下词元必须屏蔽：

- 开始词元；
- 结束词元；
- 填充词元。

设（CLIP）固定文本上下文长度为 \(L_{\max}=77\)，先定义与公开（RDE）实现一致的最大候选预算：

\[
K_T^{\max}
=
\max
\left(
1,
\left\lfloor \rho(L_{\max}-2)\right\rfloor
\right).
\]

对于第 \(i\) 条文本，设去除开始、结束和填充词元后的有效词元数为 \(n_i\)。实际参与映射和池化的词元数量为：

\[
K_{T,i}
=
\min
\left(
K_T^{\max},
n_i
\right).
\]

代码先在掩码后的注意力上取最大候选集合，再使用有效掩码保证只有这 \(K_{T,i}\) 个有效词元进入两层感知机、（批归一化）统计和最大池化。填充词元及开始、结束词元既不参与池化，也不污染（批归一化）运行统计。

对实际选中的有效词元归一化并映射：

\[
h_{i,m}^{T}
=
\operatorname{MLP}_{T}
\left(
\hat x_{i,m}^{T}
\right)
+
W_{\mathrm{skip}}^{T}
\hat x_{i,m}^{T}.
\]

逐维最大池化：

\[
\tilde l_i^T
=
\max_{m\in\operatorname{TopK}(a_i^T)}
 h_{i,m}^{T}.
\]

最终文本细粒度观测：

\[
l_i^T
=
\operatorname{Norm}
\left(
\tilde l_i^T
\right).
\]

细粒度图文相似度矩阵：

\[
S_{ij}^{L}
=
(l_i^T)^\top l_j^I.
\]

---

# 6. 零初始化残差式观测融合

全局观测维度为 \(512\)，细粒度观测维度默认为 \(1024\)。图像和文本分别使用独立的局部残差适配器：

\[
R_I:\mathbb R^{1024}\rightarrow\mathbb R^{512},
\]

\[
R_T:\mathbb R^{1024}\rightarrow\mathbb R^{512}.
\]

两个适配器均为无偏置线性层，权重初始化为全零：

\[
R_I^{(0)}=0,
\qquad
R_T^{(0)}=0.
\]

图像完整观测：

\[
g_i^I
=
\operatorname{Norm}
\left[
 b_i^I+R_I(l_i^I)
\right].
\]

文本完整观测：

\[
g_i^T
=
\operatorname{Norm}
\left[
 b_i^T+R_T(l_i^T)
\right].
\]

训练初始时：

\[
R_I(l_i^I)=0,
\qquad
R_T(l_i^T)=0,
\]

因此：

\[
g_i^I=b_i^I,
\qquad
g_i^T=b_i^T.
\]

这不是近似相等，而是在浮点误差范围内严格等于归一化后的（CLIP）全局特征。

完整观测相似度：

\[
S_{ij}^{O}
=
(g_i^T)^\top g_j^I.
\]

局部适配器虽然零初始化，但能够从第一步接收梯度；同时细粒度分支本身有独立检索监督，因此词元选择模块从第一步即可学习，不依赖局部残差适配器先离开零点。

---

# 7. 图文检索目标

版本一沿用当前纯单头基线已经验证有效的两类目标：

1. 同身份相似度分布匹配；
2. 原配图文一一匹配。

统一定义：

\[
L_{\mathrm{ret}}(I,T,p)
=
L_{\mathrm{SDM}}(I,T,p)
+
L_{\mathrm{ITC}}(I,T).
\]

## 7.1 同身份相似度分布匹配

对特征 \(f_i^T\) 和 \(f_j^I\)，余弦相似度为：

\[
s_{ij}
=
\frac{(f_i^T)^\top f_j^I}
{\|f_i^T\|_2\|f_j^I\|_2}.
\]

身份标签矩阵：

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

文本到图像预测分布：

\[
P_{ij}^{t2i}
=
\frac{\exp(\alpha s_{ij})}
{\sum_k\exp(\alpha s_{ik})},
\]

其中：

\[
\alpha=\frac1{\tau},
\qquad
\tau=0.02.
\]

文本到图像损失：

\[
L_{\mathrm{SDM}}^{t2i}
=
\frac1B
\sum_i\sum_j
P_{ij}^{t2i}
\left[
\log P_{ij}^{t2i}
-
\log(\hat Y_{ij}+\varepsilon)
\right].
\]

图像到文本方向同理：

\[
L_{\mathrm{SDM}}^{i2t}.
\]

最终：

\[
L_{\mathrm{SDM}}
=
L_{\mathrm{SDM}}^{t2i}
+
L_{\mathrm{SDM}}^{i2t}.
\]

## 7.2 原配图文匹配

文本到图像方向：

\[
L_{\mathrm{ITC}}^{t2i}
=
-\frac1B
\sum_i
\log
\frac{
\exp(\alpha s_{ii})
}{
\sum_j\exp(\alpha s_{ij})
}.
\]

图像到文本方向：

\[
L_{\mathrm{ITC}}^{i2t}
=
-\frac1B
\sum_i
\log
\frac{
\exp(\alpha s_{ii})
}{
\sum_j\exp(\alpha s_{ji})
}.
\]

最终：

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

# 8. 三个受监督的检索空间

## 8.1 全局检索目标

\[
L_G
=
L_{\mathrm{ret}}
\left(
 b^I,b^T,p
\right).
\]

## 8.2 细粒度检索目标

\[
L_L
=
L_{\mathrm{ret}}
\left(
 l^I,l^T,p
\right).
\]

## 8.3 完整观测检索目标

\[
L_O
=
L_{\mathrm{ret}}
\left(
 g^I,g^T,p
\right).
\]

## 8.4 锚点目标

\[
L_{\mathrm{anchor}}
=
\frac12
\left(
L_G+L_L
\right).
\]

## 8.5 版本一总损失

\[
\boxed{
L_{\mathrm{V1}}
=
L_{\mathrm{anchor}}
+
L_O
}
\]

展开后：

\[
L_{\mathrm{V1}}
=
\frac12
\left(
L_{\mathrm{SDM}}^G
+
L_{\mathrm{ITC}}^G
+
L_{\mathrm{SDM}}^L
+
L_{\mathrm{ITC}}^L
\right)
+
L_{\mathrm{SDM}}^O
+
L_{\mathrm{ITC}}^O.
\]

工程中为了兼容现有训练循环，返回两个聚合损失键：

```text
sdm_loss = 0.5 × (global_sdm + local_sdm) + observation_sdm
itc_loss = 0.5 × (global_itc + local_itc) + observation_itc
```

现有训练器只会求和这两个损失键，其他统计名称不包含“loss”，不会误入总损失。

需要明确的是，版本一采用“正式融合观测目标权重为 1、两个锚点分支合计权重为 1”的深监督设计。因此在训练初始时，由于融合观测与全局观测相同，全局几何会同时收到全局锚点和融合观测两条梯度；这属于有意的锚定机制，而不是代码重复求和错误。全局、细粒度和融合观测的相对权重分别为 \(0.5\)、\(0.5\) 和 \(1.0\)。

---

# 9. 为什么锚点分数不作为第三条最终分数

版本一尚未进行身份—状态分解，因此测试时完整观测本身就是正式检索表示：

\[
S_{ij}^{\mathrm{final}}
=
S_{ij}^{O}.
\]

全局和细粒度分支的作用是：

- 直接保护预训练全局检索能力；
- 直接训练细粒度词元选择能力；
- 为完整观测提供两个稳定来源；
- 为后续身份随机效应和状态残差模型提供可解释的混合观测。

在后续完整模型中，全局和细粒度分数仍不会作为第三条最终分数，而是作为观测层的直接监督。

---

# 10. 单次训练流程

每个训练批次严格执行以下步骤。

## 第一步：读取随机主批次

输入：

```text
64 张图像
64 条原配文本
64 个身份编号
64 个图像编号
```

不读取支持图像，不改变随机采样器。

## 第二步：取得（CLIP）最终词元和注意力

图像端取得：

```text
类别词元
所有局部图像块
最后一层平均多头自注意力
```

文本端取得：

```text
结束词元
所有文本词元
最后一层平均多头自注意力
```

## 第三步：生成全局观测

计算：

\[
b^I,b^T.
\]

## 第四步：生成细粒度观测

计算：

\[
l^I,l^T.
\]

## 第五步：生成完整观测

计算：

\[
g^I,g^T.
\]

## 第六步：计算六个基础损失

```text
global_sdm
global_itc
local_sdm
local_itc
observation_sdm
observation_itc
```

## 第七步：聚合总损失

```text
sdm_loss = 0.5 × (global_sdm + local_sdm) + observation_sdm
itc_loss = 0.5 × (global_itc + local_itc) + observation_itc
L_total = sdm_loss + itc_loss
```

## 第八步：一次反向传播

同时更新：

- （CLIP）图像和文本编码器；
- 图像词元选择模块；
- 文本词元选择模块；
- 图像局部残差适配器；
- 文本局部残差适配器。

---

# 11. 推理流程

测试时分别提供图像和文本。

图像侧计算：

```text
global
local
observation
```

文本侧计算：

```text
global
local
observation
```

训练期间标准评测器调用：

```text
encode_image → observation
encode_text  → observation
```

因此最佳检查点按完整观测的一号检索率保存。

正式分数：

\[
S^{\mathrm{final}}
=
S^O.
\]

最佳检查点训练完成后，离线脚本额外报告：

```text
全局分数
细粒度分数
完整观测分数
```

每套分数均输出：

```text
R1
R5
R10
mAP
mINP
```

---

# 12. 优化设置

统一训练设置：

```text
数据集：TAG-PEDES
主干：OpenAI CLIP ViT-B/16
图像尺寸：384×128
训练轮数：60
批次大小：64
随机种子：1
图像增强：关闭
主采样器：随机采样
验证划分：测试集
优化器：Adam
主干学习率：1×10⁻⁵
调度器：沿用当前仓库余弦调度
```

新初始化模块使用仓库已有的五倍学习率规则：

```text
图像词元选择模块
文本词元选择模块
图像局部残差适配器
文本局部残差适配器
```

（CLIP）主干使用基础学习率。

---

# 13. 训练日志必须记录的内容

## 13.1 聚合损失

```text
loss
sdm_loss
itc_loss
```

## 13.2 三个空间的基础项

```text
global_sdm
global_itc
local_sdm
local_itc
observation_sdm
observation_itc
```

## 13.3 汇总目标

```text
anchor_objective
observation_objective
```

其中：

\[
\text{anchor\_objective}
=
\frac12
\left(
L_G+L_L
\right).
\]

\[
\text{observation\_objective}
=
L_O.
\]

## 13.4 局部残差强度

```text
image_local_residual_norm
text_local_residual_norm
```

训练初期应接近零，随后逐渐增长；若长期严格为零，说明融合适配器没有学习；若迅速异常增大，说明局部残差可能压过全局语义。

---

# 14. 组件评测和结果解释

## 14.1 全局结果

全局结果接近现有纯单头基线，说明：

- 注意力适配器没有破坏（CLIP）前向；
- 主干训练仍然稳定；
- 全局检索锚点成立。

## 14.2 细粒度结果

细粒度结果应具有独立检索能力。它不一定必须高于全局，但不能显著接近随机。

若细粒度结果很差：

- 检查文本有效词元屏蔽；
- 检查最后一层注意力索引；
- 检查（BatchNorm）运行统计；
- 检查最大池化是否被无效词元污染。

## 14.3 完整观测结果

理想关系：

\[
R1_O\geq R1_G.
\]

至少应满足：

\[
R1_O
\text{ 不显著低于 }
R1_G.
\]

若全局和细粒度均正常，但完整观测下降：

- 检查局部残差适配器学习率；
- 检查残差范数是否过大；
- 检查图像和文本残差方向是否失配。

---

# 15. 代码—文档一致性约束

工程将损失聚合集中在 `aggregate_anchor_objectives` 函数中，并由单元测试逐项验证：

```text
sdm_loss = 0.5 × (global_sdm + local_sdm) + observation_sdm
itc_loss = 0.5 × (global_itc + local_itc) + observation_itc
```

静态审计还必须验证：

```text
零初始化融合严格保留归一化全局特征；
局部残差适配器获得有限且非零的梯度；
填充与特殊文本词元不影响细粒度输出和 BatchNorm 统计；
文档不存在控制字符导致的公式损坏；
新模式分发和参数解析完整。
```

# 16. 正式准入条件

版本一建议使用以下准入条件：

```text
最佳 R1 ≥ 57.2
最佳检查点 mAP ≥ 43.2
全局分支接近纯单头基线
完整观测不低于全局分支超过 0.5 个百分点
训练无 NaN、Inf、显存溢出或异常退出
```

理想目标：

```text
R1 > 57.719
mAP ≥ 43.713
```

若版本一未达到准入条件，不应直接加入概率身份后验和状态残差，因为后续结果将无法区分是层级机制问题还是观测底座问题。

---

# 16. 与此前完整 HIRE 主版本的区别

| 此前完整版本 | HIRE-v2 版本一 |
|---|---|
| 全局和细粒度先融合，再进入概率身份与状态分支 | 全局、细粒度和融合观测分别直接监督 |
| 融合后使用层归一化 | 使用零初始化局部残差并二范数归一化 |
| 初始融合观测不严格等于（CLIP）全局表示 | 初始融合观测严格等于归一化全局表示 |
| 使用同身份支持图和支持文本 | 不使用任何支持样本 |
| 使用概率身份均值和方差 | 不使用概率身份后验 |
| 使用自由状态头 | 不使用状态分支 |
| 使用方差校准和残差回归 | 不使用辅助回归项 |
| 最终分数为概率身份加状态 | 最终分数为完整观测余弦 |
| （TAL）主损失易较早饱和 | 使用当前纯基线已验证的（SDM）与（ITC） |

---

# 17. 代码文件映射

## 新增文件

```text
model/hire_v2_anchor_components.py
model/hire_v2_anchor_model.py
tools/hire_v2/audit_anchor.py
tools/hire_v2/eval_anchor_components.py
tests/test_hire_v2_anchor_components.py
run_hire_v2_anchor_4090_tag.sh
run_hire_v2_anchor_smoke.sh
HIRE_V2_ANCHOR_DESIGN.md
HIRE_V2_ANCHOR_README.md
```

## 修改文件

```text
model/__init__.py
utils/options.py
solver/build.py
processor/processor.py
```

## 不修改文件

```text
datasets/bases.py
datasets/build.py
model/clip_model.py
model/objectives.py
utils/metrics.py
```

版本一不需要支持包，因此不修改数据集路径；标准评测器通过 `encode_image` 和 `encode_text` 自动使用完整观测分数。

---

# 18. 一句话总结

HIRE-v2 版本一将（CLIP）全局表示和（RDE）风格细粒度表示明确作为受监督的完整混合观测来源，并通过零初始化局部残差融合在训练起点严格保留预训练图文几何，为后续“身份组级随机效应 + 状态实例残差”的可解释层级分解建立一个可验证的强基线。
