# HIRE-v2 v16.7.0：相似异身份条件下的比较式短语传播蒸馏

## 一、版本定位

**代码模式：** `--hire_v2 --hire_v2_mode identity_phrase_route_cmp`  
**直接基线：** v16.6.0  
**模型结构：** 与 v16.6.0 完全相同  
**唯一实验变量：** 离线教师分布从“跨图可传播”升级为“跨图可传播且能区分最高相似异身份”

v16.7.0 不增加网络层、不增加损失项、不增加推理步骤、不增加候选数超参数。它只替换训练 JSONL 中的教师目标。

## 二、动机

一个短语可能在同身份多图中稳定，但不具有身份判别力。例如“穿着衣服”“一个人”“黑色裤子”可能同时适用于大量相似异身份。

v16.6.0 解决：

\[
\text{该短语能否在同身份多图中可靠传播？}
\]

v16.7.0 进一步解决：

\[
\text{该短语在可传播的同时，能否排除模型最容易混淆的异身份？}
\]

## 三、最高相似异身份图像

先完整训练 v16.6.0，并使用其最佳检查点在 **TAG 训练集内部** 提取特征。

对第 \(i\) 条训练文本，使用 v16.6.0 最终单向量分数：

\[
I_i^{-}
=
\arg\max_{j:p_j\neq p_i}
S_{ij}^{F,16.6}.
\]

只使用训练集图像、训练集文本和 v16.6.0 检查点。测试集图像、文本、标签和排名均不参与。

每条文本固定一个最高相似异身份图像，不新增候选数量参数。

## 四、比较式教师案例

在 v16.6.0 的教师案例中额外加入：

```text
hard_negative 图像
```

教师仍然接收：

```text
锚点图像
锚点文本
同图另一条文本
三张同身份支持图
当前短语列表
```

并继续运行两个教师、两种支持图顺序。

对 hard negative，教师仍只输出：

```text
support
contradiction
unknown
```

只有四个判断完全一致时保留；分歧或解析失败统一为 `unknown`。

## 五、区分因子

对短语 \(m\) 在最高相似异身份图像上的严格标签，定义：

\[
d_{i,m}
=
\begin{cases}
0, & \text{hard negative 支持该短语},\\
0.5, & \text{hard negative 上未知},\\
1, & \text{hard negative 与该短语矛盾}.
\end{cases}
\]

解释：

- `support`：短语同样适用于最相似异身份，不应作为身份区分证据；
- `unknown`：无法确认是否具有区分力，保留一半；
- `contradiction`：短语明确排除该异身份，完整保留。

该映射固定，不开放为超参数。

## 六、比较式传播原始分数

v16.6.0 的传播原始分数：

\[
r_{i,m}^{\mathrm{prop}}
=
a_{i,m}
\cdot
\frac{n_{i,m}^{s}}{K_i}
\cdot
\left(1-\frac{n_{i,m}^{c}}{K_i}\right).
\]

v16.7.0 比较式分数：

\[
r_{i,m}^{\mathrm{cmp}}
=
r_{i,m}^{\mathrm{prop}}\cdot d_{i,m}.
\]

相对教师分布：

\[
q_{i,m}^{\mathrm{cmp}}
=
\frac{r_{i,m}^{\mathrm{cmp}}}
{\sum_n r_{i,n}^{\mathrm{cmp}}+\epsilon}.
\]

同样要求至少两个比较式原始分数大于零，否则该文本不计算路由蒸馏。

## 七、模型和损失完全复用 v16.6.0

学生短语路由器仍为：

\[
\pi_{i,m}
=
\operatorname{softmax}_m
\left[
w_2^\top\operatorname{GELU}(W_1c_{i,m})
\right].
\]

蒸馏损失改为：

\[
L_{\mathrm{route}}^{16.7}
=
\frac1{|\mathcal V|}
\sum_{i\in\mathcal V}
\operatorname{KL}
\left(
q_i^{\mathrm{cmp}}
\Vert
\pi_i
\right).
\]

身份短语残差、图像身份映射、身份组共识、身份门和最终分数均不变：

\[
u_i^T
=
\operatorname{Norm}
\left[
W_{\mathrm{id}}\operatorname{sg}(o_i^T)
+
A_{\mathrm{phrase}}\sum_m\pi_{i,m}h_{i,m}
\right],
\]

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

总损失仍为：

\[
L_{16.7}
=
\frac12(L_G+L_L)
+
\frac12L_O
+
\frac12L_F
+
0.1L_{\mathrm{group}}
+
0.1L_{\mathrm{route}}^{16.7}.
\]

## 八、训练流程

1. 训练 v16.6.0；
2. 用 v16.6.0 最佳检查点提取全部 TAG 训练图像和训练文本最终特征；
3. 每条训练文本检索最高相似不同身份图像；
4. 将该图像加入原 v16.6.0 多图教师案例；
5. 运行两个教师和两种支持图顺序；
6. 严格合并 hard-negative 标签；
7. 读取 v16.6.0 已合并传播分数；
8. 乘固定区分因子并重新归一化；
9. 从原始 CLIP 权重训练 v16.7.0，不从 v16.6.0 检查点继续训练。

## 九、推理

推理代码与 v16.6.0 完全相同：

- 不读取 hard negative；
- 不运行教师；
- 不读取支持图；
- 不进行候选比较或重排；
- 仍为标准单向量全图库矩阵乘法。

## 十、验收标准

相对 v16.6.0：

```text
R1 至少提高 0.10
mAP 不下降
相似异身份错误修复数多于破坏数
高权重短语在 hard negative 上的 support 比例降低
身份门保持保守
```

若 v16.7.0 不高于 v16.6.0，则保留 v16.6.0，不增加第二或第三个困难异身份候选。

## 十一、代码对应

| 设计部分 | 文件 |
|---|---|
| 训练集最高异身份挖掘 | `tools/hire_v2/extract_phrase_hard_negatives.py` |
| 构造比较式案例 | `tools/mllm/build_phrase_comparative_cases.py` |
| 教师推理 | `tools/mllm/run_phrase_teacher.py` |
| 合并比较式目标 | `tools/mllm/merge_phrase_comparative_teacher.py` |
| 共享模型结构 | `model/hire_v2_identity_phrase_route_model.py` |
| 共享组件与损失 | `model/hire_v2_phrase_route_components.py` |
| 比较模式数据加载 | `datasets/hire_v2_phrase_route_dataset.py` |
