# HIRE-v2 v16.2.1 代码—文档一致性审计

## 一、审计结论

代码与 `HIRE_V2_IDENTITY_BALANCED_DESIGN.md` 的核心定义一致。

```text
实验模式：
--hire_v2 --hire_v2_mode identity_balanced

实验版本：
v16.2.1

主结构：
v16.1.0 全局—局部完整观测
+ 严格留一身份组共识
+ 共享身份残差
+ 完整观测/最终分数平衡主损失

明确不存在：
状态分支；
支持文本；
预测方差；
异质性加权；
方差校准；
弱正样本损失；
身份分类器；
困难负样本。
```

---

## 二、离线审计结论如何进入新代码

v16.2.0 无需训练审计得到：

```text
方差均值：1.050010
方差标准差：0.000220
可信交集与简单均值中位余弦：1.000000
余弦 >= 0.9995 的组：100%
完整可信交集相对简单均值组 R1：+0.002542
v16.2 observation -> final 净修复：+96
v16.1 observation -> v16.2 final 净修复：+27
```

v16.2.1 对应处理：

```text
删除未启动的 BoundedImageUncertainty；
删除 heterogeneity_aware_identity_intersection 训练路径；
新增 masked_identity_group_consensus；
恢复 observation_sdm / observation_itc；
observation 与 final 主损失固定 0.5 / 0.5；
保留身份映射、身份门、严格组 NCE 和动态支持关系。
```

---

## 三、公式与实现逐项核对

### 1. 简单身份组共识

文档：

\[
C_i
=
\operatorname{Norm}
\left(
\frac{1}{K_i}
\sum_a u_{i,a}
\right).
\]

代码：

```python
raw_mean = (means * mask_f).sum(dim=1) / denominator
normalized_mean = F.normalize(raw_mean, dim=-1)
```

文件：

```text
model/hire_v2_identity_balanced_components.py
```

一致。

### 2. 最小支持数量

文档：

```text
至少两个不同 image_id 支持图才构成有效身份组。
```

代码：

```python
valid = count.ge(int(min_supports))
```

模型固定：

```python
min_supports=2
```

一致。

### 3. 严格留一身份组 NCE

v16.2.1 直接复用已经测试通过的：

```python
paired_identity_group_nce
```

关系为：

```text
对角身份组：唯一正目标；
同 PID 非对角身份组：忽略；
不同 PID 有效身份组：负样本；
无效身份组：屏蔽。
```

一致。

### 4. 身份映射

文档：

\[
u
=
\operatorname{Norm}
[
W_{\mathrm{id}}\operatorname{sg}(o)
].
\]

代码：

```python
return self.identity_mean(observation.detach())
```

`SharedIdentityMean` 使用无偏置线性层和单位矩阵初始化。

一致。

### 5. 身份残差分数

文档：

\[
S^F
=
S^O
+
\alpha
[
S^I-\operatorname{sg}(S^O)
].
\]

代码复用：

```python
identity_residual_score(
    observation_score,
    identity_score,
    gate,
)
```

一致。

### 6. 完整观测直接损失

文档要求 v16.2.1 恢复：

```text
observation_sdm
observation_itc
```

模型前向显式计算：

```python
observation_sdm = sdm_from_similarity(...)
observation_itc = itc_from_similarity(...)
```

一致。

### 7. 平衡主损失

文档：

\[
L_{\mathrm{main}}
=
0.5L_O+0.5L_F.
\]

代码：

```python
sdm_loss
= 0.5 * (global_sdm + local_sdm)
+ 0.5 * observation_sdm
+ 0.5 * final_sdm

itc_loss
= 0.5 * (global_itc + local_itc)
+ 0.5 * observation_itc
+ 0.5 * final_itc
```

一致。

### 8. 身份组辅助

文档：

\[
L_{\mathrm{group\ weighted}}
=
0.1L_{\mathrm{group}}.
\]

代码：

```python
identity_group_loss
= auxiliary_weight * group_nce
```

正式脚本固定：

```bash
AUX_WEIGHT=0.1
```

一致。

### 9. 推理分数

文档：

\[
f
=
[
\sqrt{1-\alpha}\,o,
\sqrt{\alpha}\,u
].
\]

代码复用：

```python
build_identity_final_embedding
```

单元测试验证其点积严格等于：

\[
(1-\alpha)S^O+\alpha S^I.
\]

一致。

---

## 四、数据关系核对

`datasets/build.py` 对以下两个身份版本使用同一个数据集包装器：

```text
identity
identity_balanced
```

因此 v16.2.0 与 v16.2.1 的支持关系保持一致：

```text
同身份；
不同 image_id；
动态轮换；
视角均衡；
无支持文本；
不改变随机主批次。
```

v16.2.1 没有引入新的离线文件或数据预处理依赖。

---

## 五、向后兼容

保留：

```text
--hire_v2_mode anchor
--hire_v2_mode identity
```

新增：

```text
--hire_v2_mode identity_balanced
```

原 v16.1.0 和 v16.2.0 模型文件、模式名称、损失和推理行为未修改。

---

## 六、已执行检查

本地已执行：

```text
新增及修改 Python 文件语法编译：通过；
两个 Bash 启动脚本语法检查：通过；
v16.2.1 组件单元测试：9/9 通过；
数学与静态审计：通过。
```

数学审计通过项目：

```text
掩码身份组共识；
损失聚合公式；
身份映射单位初始化；
初始最终分数等于完整观测；
初始主梯度等价于 v16.1.0；
推理拼接表示与训练分数等价；
新模型未实例化方差头。
```

---

## 七、当前环境未执行的项目

当前交付环境没有真实 TAG-PEDES 数据、服务器检查点和 RTX 4090，因此尚未执行：

```text
真实一轮数据烟测；
真实显存峰值测试；
60 轮正式训练；
最佳检查点组件评测。
```

这些项目必须在服务器执行 README 中的命令后确认。

---

## 八、正式烟测必须验证

```text
模式正确解析为 identity_balanced；
使用 HIREV2IdentityDataset；
support_valid_ratio 大于 0；
observation_main_weight 恒为 0.5；
final_main_weight 恒为 0.5；
identity_group_nce 为有限值；
identity_group_dispersion 为有限值；
best.pth 正常保存；
无 NaN、Inf、CUDA OOM。
```
