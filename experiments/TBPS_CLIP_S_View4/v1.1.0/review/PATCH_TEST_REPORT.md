# 本次补丁测试报告

## 已运行

- `python -m unittest tests.test_patch -v`：27项通过。
- 真实2进程和4进程 CPU/Gloo 测试：通过。
- 新项目源文件按 Python 3.8 语法解析：通过。
- 与原上传 ZIP 的逐字节比较：vendor 37文件未变；catalog及30份NPZ未变。

实际环境：Python 3.13.5、PyTorch 2.10.0+cpu、torchvision 0.25.0+cpu。详细版本见 `patch_tests/static_checks.json`。

专项测试涵盖：视觉 checkpoint 开关、前11层反向重计算及梯度等价、no-grad/eval旁路、best写入失败、last提交失败、别名发布失败及恢复、旧存档兼容、当前输入路径切换、源数据过期、代码修复后复用E0、恢复配置保护、后端统一、队列暂停/恢复/跳过和失败目录归档。

## 未运行

- 目标 Python 3.8 / Torch 1.13 / CUDA 11.7 实测。
- 原 `tests/test_core.py` 整套模型/词法测试：当前容器缺少 easydict、ftfy；未用替代实现伪造通过结果。
- 真实 CUHK 数据和预训练 CLIP 权重加载。
- 四GPU/NCCL、真实每卡80张的显存、AMP、吞吐和完整训练。
- E0–E3 正式实验及检索指标。

没有替换原项目 tokenizer/第三方依赖。服务器仍使用既有 tbpsclip_official 环境。

## 日志

- `patch_tests/cpu_regression.log`
- `patch_tests/gloo.log`、`patch_tests/gloo/result.json`
- `patch_tests/gloo_four_process.log`、`patch_tests/gloo_four_process/result.json`
- `patch_tests/static_checks.json`

队列测试采用模拟训练子进程，只证明控制流程；Gloo测试采用真实CPU多进程，但不是完整模型训练。
