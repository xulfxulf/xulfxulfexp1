# v1.1.0 修改说明

仅更新审查编号1、2、3、5、7、8及与其直接相关的调用、测试和文档。没有修改 E0–E3 的训练超参数、损失公式或采样算法。

## 文件与改动

- `view4/upstream.py`、`model.py`：显式启用视觉 checkpoint；训练有梯度时前11层重计算，最后一层直算，目标分支和评测跳过。
- `view4/runtime.py`：统一后端与目标版本检查；版本化 best/last 原子提交与恢复；用数值统计替代整模型哈希。
- `view4/train.py`：接入事务式保存/恢复，删除代码哈希门禁和逐步张量哈希；保留普通样本轨迹；逐次记录实现版本。
- `view4/queue.py`、`run_view4.py`：新增队列 `--resume`，跳过完成任务，恢复 last，归档无 last 的失败尝试；资源/GPU验收异常统一暂停；取消强制 SHA 审批。
- `view4/prepare.py`：比较当前配置的输入路径和准备数据，保留标注/角度/图片存在性检查，去掉全量图片哈希扫描。
- `view4/validation.py`：显式检查 E0、恢复运行的实验身份、训练配置、数据来源；不限制修复代码的版本。
- `view4/standalone.py`、`audit.py`：评测数值后端统一；best 根据已提交记录选择；parity 比较样本轨迹和权重，不依赖输入哈希。
- `view4/common.py`：运行记录使用路径/大小/mtime；删除文件、模型和代码哈希工具，只保留兼容旧随机序列的 seed32。
- `tests/test_patch.py`：27项新增回归测试。
- `tests/patch_distributed_checks.py`：多进程 Gloo 事务故障测试。

## 不变部分

`vendor/TBPS-CLIP` 的37个文件与原包逐字节一致；原 catalog 和30份 NPZ 采样表逐字节一致。图像/文本编码与损失的数值公式、残差结构、梯度隔离、采样、增强及学习率未改变。

## 旧运行迁移

旧 source/code SHA 字段不再参与准入判断。数据路径、关键训练设置和权重形状相容时可复用原 E0；旧 checkpoint 读取不重新初始化 optimizer。若是开始 E1–E3 新任务，仍然按原方案重新创建 optimizer/scaler；若恢复本任务，恢复全部状态。

原 prepared manifest 可继续提供源路径列表；若修改输入路径或输入文件元数据，请在新的 prepared 目录重新准备，避免悄悄用旧角度/采样映射。

旧 best/last 已损坏时明确停止。新事务流程防止再次出现 last 已推进而最佳权重尚未保存的问题。

## 测试边界

所有新增队列测试通过模拟子进程返回值验证调度逻辑，不表示启动了 CUDA 训练。Gloo 测试真实启动多个 CPU 进程，验证集体错误广播、提交回退/重试、各 rank RNG 和数值同步诊断，不表示验证了完整 CLIP 图文梯度。
