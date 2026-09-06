# TBPS-CLIP-S + 四视角图像残差专家 · 修订版 1.1.0

本版基于上传的 `TBPS_CLIP_S_View4_train_infer_20260905_153217.zip`，修复上次审查的 **1、2、3、5、7、8** 项。模型、损失、数据增强、采样表、学习率、E0–E3 定义不变。正式运行环境仍为 Python 3.8、PyTorch 1.13.0/CUDA 11.7、torchvision 0.14.0，单机四卡。

## 本次修改

| 审查项 | 实现 |
|---|---|
| 1. 激活重计算未开启 | 构造模型时显式开启视觉 Transformer 的 checkpoint；训练且有梯度时重计算前11层，最后一层直算。文本塔、no-grad目标分支、评测不新增重计算。 |
| 2. best/last 保存中断 | 先写版本化 best，再提交 last，最后发布 best.pth。last 保存最佳权重引用；异常恢复时依据 last 修复别名，不误选未提交的 best。 |
| 3. 队列不能原地恢复 | 增加 `queue --resume`。跳过已完成任务，失败任务有 last 则恢复；没有 last 则归档失败目录后重启。GPU验收、资源检查和子任务异常统一记录。 |
| 5. 检查旧路径却使用新配置 | 校验当前配置的实际输入路径、轻量文件信息、标注与角度内容，以及图片和采样表存在性；不再只检查旧 manifest 指向的文件。 |
| 7. 评测数值后端不一致 | train、evaluate、reference-val、GPU等价验收共用 TF32/cuDNN 配置和目标 Torch/torchvision 版本检查。 |
| 8. 修改代码必须重训 E0 | 删除代码哈希一致门禁。检查 E0 的实验身份、最佳轮次、数据路径和权重键/形状；代码修复后可复用已有 E0。恢复仍拒绝改变关键训练配置。 |

详见 `PATCH_NOTES.md`。测试原始日志见 `review/patch_tests/`。

## 哈希检查已简化

正常 prepare/train/evaluate/queue **不再**：

- 扫描每张图片或 CLIP 权重的 SHA256；
- 每个 batch 拷回图像/tokens 计算哈希；
- 每轮为整个模型计算哈希；
- 因代码文件变化拒绝 E0 初始化或断点恢复；
- 要求带代码哈希的批准文件。

保留必要的配置、数据契约、权重形状及路径检查。输入记录仅保存路径、文件大小、mtime；模型跨卡诊断比较数值统计，不是完整逐位一致证明。每步只记录较小的样本 ID、视角和随机种子。

`seed32()` 仍使用旧版的 SHA256 **派生随机种子**，不是文件校验；保留它是为了不改变现有采样表和增强序列。除此之外，运行代码不计算哈希。旧 `prepared/G0/run_manifest.json` 中的哈希字段仅为历史记录，程序只读取其路径以兼容随包旧数据准备结果。

## 保持不变的训练与推理

- E0：固定官方 S 配置等价复现，训练5轮，按 val R1 选 best，不使用此前完整版 checkpoint。
- E1：从 E0 续训，无视角模块。
- E2：从 E0 续训，增加四个图像残差头，无去相关损失。
- E3：从 E0 续训，增加相同残差头及同身份跨视角残差去相关。
- E1–E3 均不冻结原模型，基础 LR 比例1、专家比例10；相同种子使用相同采样表。
- 图像特征为 `normalize(h + 0.2 * Head_view(h))`。文本保持统一表示。
- 正交分支使用 `Head_view(h.detach())`，该项只更新专家。
- no-grad 软目标重新执行同一模型的前向，包含相同视角残差路径，不是 EMA。
- 推理按图库图片自身视角选择专家，查询文本不读取配对图片视角，统一余弦相似度排序。

## 运行

配置文件为 `configs/cuhk_g0.yaml`；输入路径仍为原服务器绝对路径。请使用已有 `tbpsclip_official` 环境，不要修改现有训练环境。

```bash
cd /root/autodl-tmp/TBPS_CLIP_S_View4
PY=/root/autodl-tmp/envs/tbpsclip_official/bin/python
CFG=$PWD/configs/cuhk_g0.yaml
OUT=$PWD/runs/g01

# 只生成队列计划，不启动训练。
$PY run_view4.py queue --config "$CFG" --output-dir "$OUT"

# 在上述计划目录启动执行；--resume 同样适用于失败后的继续运行。
$PY run_view4.py queue --config "$CFG" --output-dir "$OUT" --execute --resume
```

全新目录也可直接使用 `--execute`，无需先生成计划。`--review-approval` 仅保留为可选确认文件；若提供，只检查 `approved: true`，不校验代码哈希。没有 `--execute` 不会训练。

### 手动恢复单次训练

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 "$PY" -m torch.distributed.run \
  --standalone --nproc_per_node=4 run_view4.py train \
  --config "$CFG" --output-dir "$OUT/E3_seed1_formal" \
  --experiment E3 --seed 1 --run-kind formal \
  --resume "$OUT/E3_seed1_formal/checkpoints/last.pth"
```

已有 complete 结果不自动重训或重测。失败且没有 last 的任务由队列移入 `failed_attempts/` 后，从原初始化重新运行；不会覆盖失败日志。成功验收按普通 `implementation_version` 记录，新版本首次恢复会重做版本不符的验收，但不会因此重训已完成的 E0。

### 数据准备

随包 catalog/NPZ 未修改，可在原配置路径上直接验证使用。若迁移数据路径或更换源标注/权重，请指定新的 `prepared_dir` 后运行：

```bash
$PY run_view4.py prepare --config "$CFG" --output-dir /absolute/new/prepared_dir
```

新 prepared 目录须与配置一致。新准备结果保存轻量 `prepared_sources.json`，不生成图片/采样表哈希清单。

### 独立验证

```bash
$PY run_view4.py evaluate --config "$CFG" \
  --checkpoint "$OUT/E3_seed1_formal/checkpoints/best.pth" \
  --split val --device cuda:0 --output-dir /absolute/new/eval_result
```

独立评测与训练内评测使用相同数值后端。`best.pth` 是公开入口；程序根据 last 中提交的最佳权重引用选择文件。不要手动删除运行中的 `best_epoch_*.pth`。

## checkpoint 与兼容性

当前有效文件通常是 `last.pth`、一个 `best_epoch_XXX.pth` 及其 `best.pth` 别名。Linux 上别名使用硬链接，不复制一份大型权重。旧版本在新 last 成功提交后清理；保存中断前的已提交权重仍可恢复。

一致的旧版 best/last 可以加载；旧代码若已经产生缺失或互相矛盾的存档，本版明确报错，不猜测或伪造最佳权重。只加载本人保存、可信来源的训练 `.pth` 文件。

源代码修改不再触发重训 E0。模型键/形状仍严格检查；更改身份采样规模、角度门槛、数据路径等实验条件，应新建实验组，不强行恢复旧 optimizer。

## 测试

```bash
# 本次六项修复的专项测试，不依赖真实图片或预训练权重。
$PY -m unittest tests.test_patch -v

# 事务保存和广播错误的 CPU/Gloo 多进程测试。
$PY -m torch.distributed.run --standalone --nproc_per_node=4 \
  -m tests.patch_distributed_checks --output-dir /absolute/new/gloo_check

# 原工程的完整单元测试（需要原项目依赖）。
$PY run_view4.py audit --config "$CFG" --suite cpu --output-dir /absolute/new/cpu_audit
```

本次已执行27项专项 CPU 回归、2进程与4进程 Gloo checkpoint 故障恢复测试、Python 3.8 语法解析。测试使用当前 CPU 环境，不冒充目标 PyTorch 1.13/CUDA 环境。

**没有执行真实 CUHK 图片＋CLIP 权重＋四卡满 batch 训练，也没有完成目标 CUDA/AMP 验收。** 原审查第4项所述真实模型四卡满 batch 预检不在此次六项修改范围中，正式运行前仍需服务器实测。本包不包含真实训练权重或数据集图片。

`review/legacy/` 为原包的历史说明，不作为本次代码测试通过的证据。
