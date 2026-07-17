# HIRE-v2 v16.3.0 代码—文档一致性审计

## 一、总体结论

代码实现与 `HIRE_V2_IDENTITY_STATE_DESIGN.md` 的核心定义一致。

```text
模式：
--hire_v2 --hire_v2_mode identity_state

直接基线：
v16.2.1 identity_balanced

唯一新增机制：
前五十候选上的文本词元—图像块状态晚交互残差
```

代码保持 v16.2.1 身份组共识、身份组 NCE、身份映射和身份门不变。

---

## 二、公式与代码对应

### 1. 状态词元选择

文档：

```text
文本：EOT 注意力最高的 8 个有效词元；
图像：CLS 注意力最高的 16 个图像块。
```

代码：

```python
AttentionStateTokenEncoder(
    image_token_count=args.hire_v2_state_image_tokens,
    text_token_count=args.hire_v2_state_text_tokens,
)
```

图像选择显式排除 CLS；文本选择显式排除 SOS、EOT 和填充位置。

文件：

```text
model/hire_v2_state_components.py
```

一致。

### 2. 状态共享投影

文档：

\[
q=\operatorname{Norm}(W_s x^T),
\qquad
v=\operatorname{Norm}(W_s x^I).
\]

代码中图像和文本共用：

```python
self.projection = SharedStateTokenProjection(...)
```

状态维度由：

```python
self.state_dim = self.embed_dim // 4
```

得到 128，不开放第四个命令行参数。

一致。

### 3. 状态输入停止梯度

模型调用：

```python
image_tokens.detach()
text_tokens.detach()
```

后再进入状态编码器。

因此状态配对辅助不修改 CLIP、RDE 局部分支、完整观测或身份映射。

一致。

### 4. 状态晚交互

文档：

\[
c_{ijm}
=
\max_n q_{im}^{\top}v_{jn},
\]

\[
S_{ij}^{S}
=
\sum_m w_{im}c_{ijm}.
\]

代码：

```python
similarity = torch.einsum("qmd,qknd->qkmn", ...)
peak = similarity.max(dim=-1)
score = (peak * weight).sum(dim=-1)
```

权重来自选中文本词元的 EOT 注意力掩码 softmax。

一致。

### 5. 状态候选

训练代码：

```python
build_state_candidate_indices(
    identity_final_score,
    image_ids,
    state_topk,
)
```

它以停止梯度的身份平衡分数选择前 K，并将同 image_id 正样本优先加入。

测试代码：

```python
base_score.topk(k=state_topk)
```

只根据身份基础前 K，不读取测试标签。

一致。

### 6. 状态关系

代码：

```python
positive = image_ids[:, None].eq(image_ids[None, :])
negative = pids[:, None].ne(pids[None, :])
```

因此：

```text
同 image_id：正样本；
不同 PID：负样本；
同 PID 不同 image_id：既非正也非负。
```

一致。

### 7. 状态门

文档：

\[
\beta=\tanh(\theta_s),
\qquad
\theta_s^{(0)}=0.
\]

代码：

```python
self.raw = nn.Parameter(torch.zeros([]))
return torch.tanh(self.raw)
```

状态门初始严格为零，且在零点导数为一。

一致。

### 8. 最终状态残差

文档：

\[
S^F
=
S^B
+
M^C\odot\beta
(S^S-\operatorname{sg}(S^B)).
\]

代码：

```python
residual = state_score - identity_base_score.detach()
return identity_base_score + candidate_mask * state_gate * residual
```

一致。

### 9. 主损失

文档：

\[
L_{\mathrm{main}}
=
0.5L_O
+
0.25L_B
+
0.25L_F.
\]

代码：

```python
sdm_loss
= 0.5 * (global_sdm + local_sdm)
+ 0.5 * observation_sdm
+ 0.25 * identity_final_sdm
+ 0.25 * state_final_sdm

itc_loss
= 0.5 * (global_itc + local_itc)
+ 0.5 * observation_itc
+ 0.25 * identity_final_itc
+ 0.25 * state_final_itc
```

一致。

### 10. 辅助损失

文档：

\[
0.1L_{\mathrm{group}}
+
0.1L_{\mathrm{state}}.
\]

代码：

```python
identity_group_loss = auxiliary_weight * identity_group_nce
state_pair_loss = auxiliary_weight * state_nce
```

正式配置固定：

```text
auxiliary_weight = 0.1
```

一致。

### 11. 初始化等价性

状态门为零时：

\[
S^F=S^B.
\]

因此：

\[
0.5L_O+0.25L_B+0.25L_F
=
0.5L_O+0.5L_B.
\]

静态审计和单元测试验证函数值与梯度均和 v16.2.1 初始主目标一致。

### 12. 训练期评测

`utils/metrics.py` 在检测到：

```python
is_hire_v2_state_model
```

后使用：

```python
compute_state_reranked_similarity
```

计算 identity_final 和 state_final，并以 state_final R1 作为最佳检查点标准。

一致。

---

## 三、数据关系

`datasets/build.py` 对：

```text
identity
identity_balanced
identity_state
```

统一使用 `HIREV2IdentityDataset`。

因此身份支持关系与 v16.2.1 完全相同。

状态分支仅使用主批次：

```text
images
caption_ids
pids
image_ids
```

不读取：

```text
support_images
support_mask
support_pids
support_image_ids
```

进行状态匹配。

---

## 四、向后兼容

保留模式：

```text
anchor
identity
identity_balanced
```

新增模式：

```text
identity_state
```

旧模型文件、旧损失公式、旧训练和旧评测分支均未删除。

---

## 五、静态审计项目

`tools/hire_v2/audit_identity_state.py` 检查：

```text
状态门精确零初始化并可学习；
训练候选包含所有同图状态正样本；
晚交互与手工 MaxSim 一致；
状态最终分数初始等于身份基础；
总损失公式一致；
初始主梯度等于 v16.2.1；
状态投影为图文共享；
模式在参数、模型、数据和评测中全部注册；
状态输入已停止梯度；
状态模型不含状态支持包、MLLM、困难负样本或分类器。
```

---

## 六、单元测试项目

`tests/test_hire_v2_state_components.py` 覆盖：

```text
状态门初始化；
图像块选择；
文本词元掩码；
晚交互手工值；
状态候选正样本覆盖；
同身份不同图状态梯度屏蔽；
状态残差候选稀疏性；
损失聚合公式；
初始主梯度等价性。
```

---

## 七、尚需服务器验证的内容

当前交付环境没有真实 TAG-PEDES 图像和 RTX 4090，必须在服务器完成：

```text
真实一轮烟测；
状态专用逐轮评测；
实际显存峰值；
状态评测耗时；
六十轮正式训练；
最佳检查点组件评测。
```

---

## 八、烟测必须满足

```text
state_positive_coverage = 1；
state_pair_nce 有限；
state_text_token_count > 0；
state_image_token_count = 16；
state_gate 初始接近 0；
identity_final 和 state_final 同时出现在评测表；
best.pth 按 state_final 保存；
无 NaN、Inf、CUDA OOM。
```


---

## 九、当前交付环境已执行结果

```text
全部交付 Python 文件抽象语法树解析：通过；
两个 Bash 启动脚本语法检查：通过；
v16.3.0 状态组件单元测试：9/9 通过；
v16.3.0 数学与静态审计：8/8 通过。
```

静态审计实际通过项：

```text
zero_state_gate
positive_candidate_coverage
late_interaction
initial_score_equivalence
objective_formula
initial_gradient_equivalence
shared_state_projection
source_invariants
```

当前环境没有真实 TAG-PEDES 图像和服务器显卡，因此未宣称真实一轮前向、显存峰值或训练结果已通过。
