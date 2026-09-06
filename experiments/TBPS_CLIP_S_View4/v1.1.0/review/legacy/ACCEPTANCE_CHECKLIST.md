# 验收清单

本文件区分“源码已实现”与“真实 GPU 上已验证”。不得把待测项视为通过。

| 要求 | 实现位置 | 当前验证 |
|---|---|---|
| E0 caption-pair 展开与 DistributedSampler | sampling.e0_plan | 真实数据计划 + 单元测试通过 |
| 新 forward 不读关闭分支输入 | model.ViewCLIP.forward | CPU 原/新对照通过 |
| 真正 S E0，而非沿用旧 full TBPS checkpoint | config.official_config / train.check_e0 | 源码检查，E0 未训练 |
| 构建后重置训练 RNG，数据 RNG 隔离 | common / train / data | CPU 多步一致性通过 |
| 先解冻，再 DDP，再 optimizer | model.build_model / train.train | 真实结构构造通过 |
| FP32 专家/融合、回转 h.dtype 后原 normalize | model.route / fuse | FP32 通过；CUDA FP16 待测 |
| no-grad target 使用相同专家路径且保持 train dropout | model.forward | 调用路径与小型数值测试通过 |
| 不同图片 + raw/aug 双重跨视角过滤 | model.pair_mask | 单元测试通过 |
| 去相关不更新 backbone，空集合图连接零损失 | model.decorrelation | 单卡/双进程通过 |
| DDP 按全球有效对数平均 | model.decorrelation | 双进程 vs 全局梯度通过 |
| 同一 h 的四专家六对 cos² 诊断 | evaluation.expert_diagnostics | 顺序/RNG/退化值测试通过 |
| raw/aug 视角、零范数对、rho 固定告警 | train.telemetry | 源码完成，真实训练日志待测 |
| 独立 val/test，图库 index 对齐，文本不读视角 | evaluation | 顺序扰动测试通过；完整数据评测待测 |
| 不依赖 person2text，num_train_ids 正确 | data / prepare | 真实 11003 身份通过 |
| 全 rank RNG gather + 原子 checkpoint | runtime | 双进程 RNG roundtrip 通过 |
| AMP 溢出同步与增长状态一致 | runtime.SynchronizedScaler | torch1.13 适配完成，CUDA 注入待测 |
| 每轮参数/scaler 全 rank 一致检查 | runtime.assert_model_synchronized | CPU 双进程连续更新通过 |
| 第一轮 E2/E3 权重和输入一致门槛 | audit.compare_epoch1 / queue | 小型模型通过；完整 epoch 待测 |
| 原始官方入口 loss/gradient/update/val 等价 | audit / tests | 小型编码器通过；真实权重 GPU 验收待测 |
| 固定 seed1/2/3、同 E0 五轮，串行且失败暂停 | queue | 仅预览，未启动 |
| 原始数据只读、原官方源码不改 | prepare / freeze_review | 40206 图片校验、37 源文件校验通过 |

结论：代码可交由用户审核；尚不能宣称真实 GPU 训练已经跑通。
