# HIRE-v2 v16.3.0 覆盖包清单

## 源代码基线

```text
仓库：xulfxulf/xulfxulfexp1
提交：44749815b3b6769071b424472938913f3feb3ec3
```

## 新增文件

```text
model/hire_v2_state_components.py
model/hire_v2_identity_state_model.py
tools/hire_v2/eval_identity_state_components.py
tools/hire_v2/audit_identity_state.py
tests/test_hire_v2_state_components.py
run_hire_v2_identity_state_4090_tag.sh
run_hire_v2_identity_state_smoke.sh
HIRE_V2_IDENTITY_STATE_DESIGN.md
HIRE_V2_IDENTITY_STATE_README.md
HIRE_V2_IDENTITY_STATE_CONSISTENCY_AUDIT.md
SOURCE_COMMIT_V1630.txt
```

## 修改文件

```text
model/__init__.py
utils/options.py
datasets/build.py
processor/processor.py
utils/metrics.py
```

## 不修改

```text
v16.1.0 观测模型；
v16.2.0 身份模型；
v16.2.1 身份平衡模型；
HIREV2IdentityDataset 的支持选择实现；
CLIP 主干；
RDE 风格词元选择；
优化器和学习率调度；
历史日志和检查点。
```
