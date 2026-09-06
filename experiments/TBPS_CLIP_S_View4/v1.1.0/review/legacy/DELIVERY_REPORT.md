# 代码交付与审核说明

日期：2026-09-05。服务器：TBPR-PRO1。

## 交付状态

已在 `/root/autodl-tmp/TBPS_CLIP_S_View4` 建立独立实现，包含 E0/E1/E2/E3、数据预检、固定采样表生成、训练/恢复/评测、串行队列及测试代码。
本次没有启动 E0/E1/E2/E3 训练，没有修改原数据集、旧实验目录或 conda 环境，没有推送 GitHub，也没有配置新的 SSH key。

代码冻结 SHA256：`5c992abaeba333cf3ec10cc8fa467a1ab82cf0dc1752486185be68ba0e3dbedd`。
该值是 `run_freeze/code_checksums.json` 的规范 JSON 哈希，不是压缩包哈希。包传回后还需逐文件验证。
当前工程未创建 Git commit；官方来源提交为 `ba090bc7dcb3b2787e80f701fc34e19f9205cee4`，原始 ZIP 中的 37 个文件均未改动。

## 已验证

| 检查 | 结果 | 证据 |
|---|---|---|
| Python 语法编译 | 通过 | 服务器 compileall 成功 |
| CPU 单元测试 | 16 项通过，1 项 GPU 测试跳过 | `cpu_final/unittest.log` |
| 原/新 S 检索 forward、梯度及两次 optimizer 更新 | 小型测试编码器通过，调用原官方 CLIP.forward | `cpu_final/unittest.log` |
| 官方/新图像增强像素一致 | 40 个固定 seed 通过 | 同上 |
| 路由边界、连续两次翻转、raw/aug 配对筛选 | 通过 | 同上 |
| 去相关仅更新专家，空配对零损失可 backward | 通过 | 同上 |
| E2/E3 lambda=0 时的参数更新和 RNG 一致 | 小型模型多步测试通过 | 同上 |
| mid-step 模型、optimizer、RNG 恢复 | 小型模型逐 tensor 完全一致 | 同上 |
| 图库顺序扰动后按 index 写回、诊断不消耗训练 RNG | 通过 | 同上 |
| CPU 双进程 DDP 与全局参考梯度一致 | 通过，包括某 rank 无有效配对 | `gloo_final/distributed_result.json` |
| 全 rank 无配对反传、连续更新、模型同步检查及 RNG gather 恢复 | 通过 | 同上及测试源码 |
| 真实 ViT-B/16 + 四专家结构构造 | 通过，未加载预训练权重 | `construction_final/run_manifest.json` |
| 全部原参数可训练、train() 不重新冻结 conv1 | 通过 | 同上 |
| 四专家参数量 | 530944，整个 E3 模型 150151681 | 同上 |
| CUHK 原始图片 SHA256、PID/split/角度关联 | 40206 张全部通过 | `../prepared/G0/summary.json` |
| 三个 seed 各五轮固定采样表 | 已生成，E1-E3 共用 PK 表 | `../prepared/G0/plan_checksums.json` |
| 队列命令生成 | 已生成但未执行 | `queue_preview/queue_status.json` |

## 未完成，不能写成通过

1. 当前 `torch.cuda.is_available()` 为 False，GPU 数量为0。因此未验证 CUDA FP16 融合、真实 CLIP 反传/两次更新、多卡 AMP 单 rank 溢出注入、真实四卡运行。
2. 真实 OpenAI 预训练权重的 CPU 前向核验进程被外部终止（shell 显示 Killed，未产生完整结果）。容器内存上限为2GB，但本 cgroup 的 oom_kill 计数为0，不能将具体终止机制断言为已证实的内核 OOM。见 `backbone_r1/external_stop.json`。
3. 完整 E0 验证集原/新入口数值与指标核验、E1-E3 两轮 smoke、完整第一轮 E2/E3 权重对照和正式训练均未执行。本交付没有可报告的 R1。
4. 所有“小型模型通过”仅证明实现的相关数学/工程接口，不等价于真实 backbone 或完整训练已经通过。真实 GPU 验收是串行训练队列的硬门槛。
5. 完整队列的保守磁盘门槛约为54.22GiB；环境快照可用约53.85GiB，尚差约0.4GB，打包也会占少量空间。未来执行前需要重新检查；此次未删除任何旧数据或保存点。详见 `environment/resource_status.json`。

## 审核入口

- `../README.md`：完整行为规范、运行步骤、阈值与公平性边界。
- `../view4/model.py`：原 S 检索损失、图像专家、target 分支、跨视角残差去相关及参数分组。
- `../view4/data.py`、`../view4/sampling.py`：视角规则、增强、caption/BT 处理、身份采样及确定性输入。
- `../view4/train.py`、`../view4/runtime.py`：DDP、AMP 同步、全参数训练、checkpoint/RNG、验证 best 选择。
- `../view4/evaluation.py`、`../view4/audit.py`：图库对齐、专家分化诊断、官方数值核验。
- `../view4/queue.py`：代码审核确认与验收门槛、串行任务、失败暂停，默认仅生成计划。

## 已修复与保留记录

- 第一次 CPU 测试尝试 FP16 normalize，torch1.13 报 `clamp_min_cpu not implemented for Half`。未改变正式模型精度路径；改为单独的 GPU 必测项。`cpu_r1` 失败记录保留。
- 原官方 EDA 在单词句上返回 list 的问题只在新输入处理层规范为字符串，参考文件未改动。
- GPU 未挂载时不启动训练，CPU 核验失败没有触发重新初始化、低精度替代、删保存点或启动其他任务。
- 队列目前只有预览；没有生成 `approved=true` 的审核文件。完成审核及 GPU 验收之前不会自动进入训练。
