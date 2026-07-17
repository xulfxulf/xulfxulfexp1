# HIRE-v2 v16.2.1 覆盖包清单

## 源代码基线

```text
仓库：xulfxulf/xulfxulfexp1
提交：7d89aa311eda5aaef8b7f6f200e2cd47de015ad0
```

## 新增文件

```text
model/hire_v2_identity_balanced_components.py
model/hire_v2_identity_balanced_model.py
tools/hire_v2/eval_identity_balanced_components.py
tools/hire_v2/audit_identity_balanced.py
tests/test_hire_v2_identity_balanced_components.py
run_hire_v2_identity_balanced_4090_tag.sh
run_hire_v2_identity_balanced_smoke.sh
HIRE_V2_IDENTITY_BALANCED_DESIGN.md
HIRE_V2_IDENTITY_BALANCED_README.md
HIRE_V2_IDENTITY_BALANCED_CONSISTENCY_AUDIT.md
SOURCE_COMMIT_V1621.txt
```

## 修改文件

```text
model/__init__.py
utils/options.py
datasets/build.py
processor/processor.py
```

## 不修改

```text
v16.1.0 模型代码；
v16.2.0 模型代码；
HIREV2IdentityDataset 的支持关系实现；
CLIP 主干；
RDE 风格词元选择；
优化器和调度器；
测试集评测公式；
历史实验日志和检查点。
```
