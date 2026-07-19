# HIRE-v2 v16.6.0：多视角短语相对可传播性蒸馏

## 一、版本定位

**源仓库：** `xulfxulf/xulfxulfexp1`  
**源主分支提交：** `0863840462dbd4dfdb9e42b2cdfb2f08010b4e6f`  
**代码模式：** `--hire_v2 --hire_v2_mode identity_phrase_route`  
**直接性能基础：** v16.2.1  
**数据集：** TAG-PEDES  
**主干：** OpenAI CLIP ViT-B/16  
**训练方式：** 单阶段、从原始 CLIP 权重开始，不加载旧检查点  
**测试阶段：** 不运行 MLLM，不读取同身份支持图，不进行候选重排

v16.5.0 尚未训练和验收。原实验计划规定：若 v16.5.0 未通过，则 v16.6.0 直接以 v16.2.1 为基础。因此本实现不包含同图双文本损失替换，保持 v16.2.1 的主检索损失和身份组损失不变，只新增短语路由蒸馏。

## 二、研究假设

当前强图文模型主要学习当前图像与当前文本的完整配对关系。该表示同时包含衣着、携带物、姿态、视角、可见细节及部分身份代理信息，不应被强制净化成“纯状态空间”。

同身份多图只用于估计跨图稳定信息。身份或状态不是一个短语的固定语言类别，而是该短语在当前身份的多图观测中是否可以可靠传播。

因此模型采用非对称结构：

\[
\boxed{
\text{当前配对完整观测为主体}
+
\text{跨图稳定短语只修正身份残差}
}
\]

## 三、固定的 v16.2.1 基础

### 3.1 全局观测

图像和文本分别取得 CLIP 的类别词元与结束词元：

\[
b_i^I=\operatorname{Norm}(x_{i,\mathrm{cls}}^I),
\qquad
b_i^T=\operatorname{Norm}(x_{i,\mathrm{eot}}^T).
\]

全局相似度：

\[
S_{ij}^{G}=(b_i^T)^\top b_j^I.
\]

### 3.2 RDE 风格局部观测

图像端使用类别词元最后一层注意力选择前 30% 图像块；文本端使用结束词元注意力，在排除开始词元、结束词元和填充词元后选择前 30% 有效词元。选中词元经过原有残差映射和逐维最大池化，得到：

\[
l_i^I,\qquad l_i^T.
\]

局部相似度：

\[
S_{ij}^{L}=(l_i^T)^\top l_j^I.
\]

### 3.3 完整观测

零初始化局部残差融合：

\[
o_i^I=\operatorname{Norm}(b_i^I+A_I l_i^I),
\]

\[
o_i^T=\operatorname{Norm}(b_i^T+A_T l_i^T).
\]

完整观测相似度：

\[
S_{ij}^{O}=(o_i^T)^\top o_j^I.
\]

### 3.4 身份表示与严格留一身份组

共享身份映射：

\[
u_i^m=
\operatorname{Norm}
\left[
W_{\mathrm{id}}\operatorname{sg}(o_i^m)
\right],
\quad m\in\{I,T\}.
\]

对锚点图像排除原图后，选取最多三张同身份不同图像作为支持：

\[
\mathcal S_i=\{I_{i,k}^{+}\}_{k=1}^{K_i},
\qquad 2\le K_i\le3.
\]

身份组共识：

\[
C_i^I=
\operatorname{Norm}
\left(
\frac1{K_i}\sum_{k=1}^{K_i}u_{i,k}^{I}
\right).
\]

支持图不作为普通图文正样本。第 \(i\) 条文本只把为其构造、且严格排除锚点原图的第 \(i\) 个身份组作为正目标；同身份非对角组忽略，不同身份有效组作为负样本。

### 3.5 有界身份残差

身份分数：

\[
S_{ij}^{I}=(u_i^T)^\top u_j^I.
\]

身份门：

\[
\alpha=\operatorname{sigmoid}(\theta_{\mathrm{id}}),
\qquad \alpha_0=0.1.
\]

最终分数：

\[
S_{ij}^{F}
=
S_{ij}^{O}
+
\alpha
\left[
S_{ij}^{I}-\operatorname{sg}(S_{ij}^{O})
\right].
\]

前向数值等价于：

\[
S_{ij}^{F}=(1-\alpha)S_{ij}^{O}+\alpha S_{ij}^{I}.
\]

## 四、离线短语跨度

### 4.1 提取单位

教师与学生均不直接监督 CLIP 子词，而监督完整视觉语义短语。离线脚本使用固定依存句法规则抽取：

- 上衣；
- 下装；
- 鞋子；
- 包；
- 帽子；
- 头发或面部毛发；
- 姿态或动作；
- 其他可视觉验证名词短语。

重叠候选保留最长完整跨度。无可用视觉短语时生成一条整句回退跨度；回退跨度只用于推理保持连续性，不进入教师路由监督。

### 4.2 字符跨度到 CLIP 词元

每个短语保存：

```text
char_start
char_end
token_positions
category
phrase_id
```

词元位置由仓库同一个 `SimpleTokenizer` 对前缀进行编码得到，并严格排除开始词元、结束词元和填充位置。

训练、验证与测试都读取离线生成的同一套短语跨度。测试只读取跨度，不读取教师标签。

## 五、MLLM 教师案例

教师支持集合固定使用与训练支持选择器一致的 `seed=1`、`teacher_support_epoch=0` 结果；它只决定离线教师观测，不成为模型输入或可搜索参数。

每个训练记录对应一个多图案例，输入包含：

1. 锚点图像；
2. 锚点原始文本；
3. 同图另一条人工文本；
4. 三张同身份、不同 image_id 的支持图；
5. 当前文本中全部非回退短语。

不发送：

```text
PID
关系真值
航拍或地面标签
模型排名
正确答案
```

教师只判断每个短语与各观测之间的关系：

```text
support
contradiction
unknown
```

其中 `unknown` 必须用于不可见、过小、模糊、遮挡、歧义或证据不足；遗漏不能被当作矛盾。

## 六、双教师与双顺序严格一致

固定教师：

```text
Qwen3-VL-8B-Instruct
InternVL3.5-8B-HF
```

每个案例分别运行：

```text
Qwen + 正向支持图顺序
Qwen + 反向支持图顺序
InternVL + 正向支持图顺序
InternVL + 反向支持图顺序
```

对锚点、同图兄弟文本和每张支持图，只有四个判断完全一致时保留该标签；任何解析失败或分歧统一归为 `unknown`。

支持标签使用 `image_id` 映射，不依赖图像在输入中的位置，因此反向顺序不会错配支持图。

## 七、锚点可靠性与传播分数

短语 \(m\) 的锚点可靠性：

\[
a_{i,m}=
\mathbf1[
\text{anchor}=\text{support}
\land
\text{sibling}\neq\text{contradiction}
].
\]

同图另一条文本可以省略该短语；只有明确冲突才取消锚点可靠性。

对 \(K_i\) 张支持图统计：

\[
n_{i,m}^{s}=\#\text{support},
\]

\[
n_{i,m}^{c}=\#\text{contradiction},
\]

\[
n_{i,m}^{u}=\#\text{unknown}.
\]

满足：

\[
n_{i,m}^{s}+n_{i,m}^{c}+n_{i,m}^{u}=K_i.
\]

传播原始分数：

\[
r_{i,m}^{\mathrm{prop}}
=
a_{i,m}
\cdot
\frac{n_{i,m}^{s}}{K_i}
\cdot
\left(
1-\frac{n_{i,m}^{c}}{K_i}
\right).
\]

该公式满足：

- 支持越多，分数越高；
- 明确矛盾比未知惩罚更强；
- 锚点本身不可靠时分数为零；
- 不新增阈值。

## 八、相对可传播分布

身份短语池化关心的是一条文本内部哪些短语相对更适合跨图传播，而不是每个短语是否超过固定 0.5。

因此：

\[
q_{i,m}^{\mathrm{prop}}
=
\frac{r_{i,m}^{\mathrm{prop}}}
{\sum_n r_{i,n}^{\mathrm{prop}}+\epsilon}.
\]

只有同时满足以下条件的文本才计算路由损失：

- 至少有两个传播原始分数大于零的短语；
- 原始分数总和大于零；
- 短语跨度和教师结果完整。

其他文本仍参与全部检索损失和身份组损失，只是不参与短语路由蒸馏。

## 九、学生短语表示

设短语 \(m\) 覆盖的词元集合为 \(\Omega_{i,m}\)。结束词元最后一层注意力记为 \(a_{i,t}\)。短语内注意力归一化：

\[
\omega_{i,m,t}
=
\frac{
\exp(a_{i,t})
}
{
\sum_{r\in\Omega_{i,m}}\exp(a_{i,r})
},
\quad t\in\Omega_{i,m}.
\]

短语表示：

\[
h_{i,m}
=
\operatorname{Norm}
\left(
\sum_{t\in\Omega_{i,m}}
\omega_{i,m,t}x_{i,t}^{T}
\right).
\]

原始文本词元在进入路由器前停止梯度，路由辅助不能直接重写 CLIP 主干。

## 十、相对短语路由器

上下文短语表示：

\[
c_{i,m}
=
\operatorname{LN}
\left(
h_{i,m}+\operatorname{sg}(o_i^T)
\right).
\]

路由 logit：

\[
z_{i,m}
=
w_2^\top
\operatorname{GELU}(W_1c_{i,m}).
\]

隐藏维度固定为：

\[
512/4=128.
\]

文本内部相对分布：

\[
\pi_{i,m}
=
\frac{\exp(z_{i,m})}
{\sum_{n\in\mathcal M_i}\exp(z_{i,n})}.
\]

最后一层权重和偏置初始化为零，所以有效短语初始均匀分配，不新增固定概率阈值。

## 十一、短语蒸馏损失

\[
L_{\mathrm{route}}
=
\frac1{|\mathcal V|}
\sum_{i\in\mathcal V}
\operatorname{KL}
\left(
q_i^{\mathrm{prop}}
\Vert
\pi_i
\right).
\]

其中 \(\mathcal V\) 为有效路由监督文本集合。

教师目标和学生路由均在有效短语上归一化。无效短语、回退短语和无路由监督文本不产生路由梯度。

## 十二、身份短语残差

身份短语池：

\[
z_i^{T,\mathrm{id}}
=
\sum_m \pi_{i,m}h_{i,m}.
\]

v16.2.1 文本身份映射的未归一化输出：

\[
\widetilde u_i^T
=
W_{\mathrm{id}}\operatorname{sg}(o_i^T).
\]

加入无偏置短语适配器：

\[
u_i^T
=
\operatorname{Norm}
\left[
\widetilde u_i^T
+
A_{\mathrm{phrase}}z_i^{T,\mathrm{id}}
\right].
\]

初始化：

\[
A_{\mathrm{phrase}}^{(0)}=0.
\]

所以训练开始时：

\[
u_i^T(\mathrm{v16.6.0})
=
u_i^T(\mathrm{v16.2.1}).
\]

图像身份表示、支持图身份组共识和身份门完全不变。

## 十三、总损失

每个表示分支继续使用 v16.2.1 的：

\[
L_{\mathrm{ret}}=L_{\mathrm{SDM}}+L_{\mathrm{ITC}}.
\]

总损失：

\[
L_{16.6}
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

工程返回四个包含 `loss` 的字段：

```text
sdm_loss
itc_loss
identity_group_loss
phrase_route_loss
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

phrase_route_loss
= 0.1 × phrase_route_kl
```

短语路由复用已有辅助系数 `0.1`，不新增方法损失权重。

## 十四、推理流程

测试时只输入查询文本和图库图像。

文本侧：

1. 读取离线确定性短语跨度；
2. 编码完整文本；
3. 按跨度池化短语；
4. 学生路由器预测相对分布；
5. 生成文本身份短语残差；
6. 构造最终单向量。

图像侧完全沿用 v16.2.1。

最终单向量：

\[
f^m
=
\left[
\sqrt{1-\alpha}\,o^m,
\sqrt{\alpha}\,u^m
\right].
\]

不使用 MLLM、支持图、PID、视角标签或候选重排。

## 十五、固定训练配置

```text
数据集：TAG-PEDES
主干：OpenAI CLIP ViT-B/16
训练轮数：60
批次大小：64
随机种子：1
采样器：random
可选图像增强：关闭
基础学习率：1e-5
新模块学习率倍率：5
预热：5轮
调度：余弦
支持图数量：3
身份组辅助权重：0.1
短语路由辅助权重：复用0.1
身份门初始值：0.1
不加载旧检查点
```

## 十六、质量门与验收

教师数据先运行每类五十个案例的质量子集。要求：

- 双教师双顺序严格一致率达到计划门槛；
- 人工抽查准确率由 `review_template.csv` 填写；
- `unknown` 比例不高于 70%；
- 支持、矛盾、未知三类均出现；
- 至少部分文本具有两个以上有效传播短语。

训练机制指标：

```text
phrase_route_kl 正常下降
phrase_route_spearman 为正
phrase_route_top1_agreement 高于均匀随机
学生路由熵不长期等于均匀熵
短语身份残差范数离开零点
身份门保持有界且不饱和
```

性能门：

```text
最终 R1 至少达到 58.234
最终 mAP 不低于 44.540
最终结果高于同检查点 observation
```

## 十七、代码对应

| 设计部分 | 文件 |
|---|---|
| 短语跨度与固定张量 | `datasets/phrase_route_io.py` |
| 训练与测试短语数据集 | `datasets/hire_v2_phrase_route_dataset.py` |
| 短语池化、路由器、KL、零初始化残差 | `model/hire_v2_phrase_route_components.py` |
| 完整模型前向和推理 | `model/hire_v2_identity_phrase_route_model.py` |
| 短语提取 | `tools/mllm/phrase_extraction.py` |
| 生成 train/test 跨度 | `tools/mllm/build_phrase_spans.py` |
| 构造多图教师案例 | `tools/mllm/build_phrase_teacher_cases.py` |
| 双教师推理 | `tools/mllm/run_phrase_teacher.py` |
| 严格合并传播标签 | `tools/mllm/merge_phrase_teacher.py` |
| 教师质量审计 | `tools/mllm/audit_phrase_teacher.py` |
| 组件评测 | `tools/hire_v2/eval_identity_phrase_route_components.py` |
| 静态审计 | `tools/hire_v2/audit_identity_phrase_route.py` |
