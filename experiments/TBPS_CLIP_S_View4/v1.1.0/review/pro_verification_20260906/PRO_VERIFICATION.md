# v1.1 PRO 服务器验证记录

检查时间：2026-09-06。此记录是接收包的部署和定向验证，不是完整 CUDA 训练通过证明。

## 版本与部署

- 原始包：`D:/downloadfile/TBPS_CLIP_S_View4_patched_v1_1.zip`。
- 服务器代码：`/root/autodl-tmp/TBPS_CLIP_S_View4_patched_v1_1_20260906`。
- 本机代码：`D:/004SSH/TBPS_CLIP_S_View4_patched_v1_1_20260906`。
- 原服务器 `/root/autodl-tmp/TBPS_CLIP_S_View4` 及原本机版本均保留。
- 接收包的训练、推理源码和配置未修改；新增内容仅为本目录的验证脚本与记录。
- 配置仍显式使用 `/root/autodl-tmp/TBPS_CLIP_S_View4/prepared/G0` 作为既有采样表输入，验证通过；没有改写该目录。
- 未恢复文件、模型、代码或逐批次张量哈希门禁，也未设置强制审批文件。
- 未启动正式训练，未推送 GitHub。

## 目标环境

Python 3.8.20；PyTorch 1.13.0+cu117；torchvision 0.14.0+cu117；CUDA runtime 11.7。

当前 GPU 数量为 0，`torch.cuda.is_available()` 为 False；容器内存上限 2 GiB。
检查时可用磁盘 57,739,702,272 字节，13 项队列的现有磁盘门禁要求 54,442,065,920 字节，空间门禁通过。

## 已执行验证

| 检查 | 实际结果 | 证据 |
|---|---|---|
| Python 编译检查 | 通过 | 服务器 compileall 返回 0 |
| 补丁专项 CPU 测试 | 27/27 通过 | `patch_unittest.log` |
| 完整 CPU 测试 | 44 项，43 通过、1 跳过、0 失败 | `cpu_audit/run_manifest.json`、`cpu_audit/unittest.log` |
| 2 进程 Gloo 事务测试 | 通过 | `gloo2/result.json` |
| 4 进程 Gloo 事务测试 | 通过 | `gloo4/result.json` |
| 真实 CUHK 数据关联、配置路径、角度、采样表检查 | 通过 | `input_check.json` |
| 队列预览及非执行模式 resume | 通过，仍为 planned_only | `queue_preview/queue_status.json` |

27 项专项测试已包含在 44 项完整测试内，不能相加计数。跳过项为 GPU FP16 融合测试。
Gloo 使用小型 CPU 测试权重，覆盖提交失败广播、旧提交保留、重试、各 rank RNG 保存和数值分叉检测，不代表真实 CLIP 四卡训练。

CUHK 实际关联数量：train 34,054 图 / 11,003 身份 / 68,126 caption；val 3,078 图 / 1,000 身份 / 6,158 caption；test 3,074 图 / 1,000 身份 / 6,156 caption。
输入验证没有扫描图片或权重字节计算哈希，也没有修改原始数据集。

源码对照确认：官方 vendor、采样算法、评测特征计算、原 catalog 和 30 份采样表未变化；损失、残差公式及学习率函数未改动。数据处理改动仅为 tokenize 延迟导入。

## 已确认的检查缺口

`view4/validation.py:48` 的 E0 复用检查仅逐项比较 PATH_KEYS，未比较源 checkpoint 的 TRAIN_KEYS。

使用合成小型 checkpoint 复现：源 `local_batch=40`、当前 `local_batch=80` 时，E0 复用检查仍放行；源 `steps_per_epoch=106`、当前为 212 时也放行。同样的配置差异在同任务 `check_resume` 中能被正确拒绝。

证据：`probe_e0_config.py`、`e0_config_probe.json`。没有加载或改动真实 E0 权重。
这不表示现有输入已经不公平，但说明“允许代码修复后复用 E0，仍检查关键配置”尚未完整实现。建议复用 E0 前补充源训练配置兼容检查及对应回归测试，不需要恢复代码哈希限制。本次保留来包原样，未擅自修补。

## 未完成与运行边界

- GPU 不可见，尚未执行真实 CLIP 加载后的 CUDA 等价检查、四卡 AMP、真实满 batch 训练或显存峰值测试。
- 队列记录仅为预览；没有后台训练或等待 GPU 后自动启动的任务。
- 不将包内历史测试日志冒充本次服务器测试结果。
- 当前结论：已部署，CPU 与输入检查通过；存在上述 E0 配置校验缺口，GPU 验收仍待执行。
