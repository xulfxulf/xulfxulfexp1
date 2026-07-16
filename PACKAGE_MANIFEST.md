# HIRE-v2 Identity Overlay Manifest

**实验版本：** `v16.2.0`

**直接基线版本：** `v16.1.0`

## 基线

```text
repository: xulfxulf/xulfxulfexp1
branch: main
source main commit: 09def7e11fe3a2f47b39929013aaad4038b98ac9
version-one model commit: 48e61f81649aa2f3ea515d8e967faa4960b2f478
```

## 保留并随包提供的版本一文件

```text
HIRE_V2_ANCHOR_DESIGN.md
HIRE_V2_ANCHOR_README.md
HIRE_V2_ANCHOR_CONSISTENCY_AUDIT.md
model/hire_v2_anchor_components.py
model/hire_v2_anchor_model.py
run_hire_v2_anchor_4090_tag.sh
run_hire_v2_anchor_smoke.sh
tools/hire_v2/audit_anchor.py
tools/hire_v2/eval_anchor_components.py
tests/test_hire_v2_anchor_components.py
```

## 版本二新增

```text
HIRE_V2_IDENTITY_DESIGN.md
HIRE_V2_IDENTITY_README.md
HIRE_V2_IDENTITY_CONSISTENCY_AUDIT.md
model/hire_v2_identity_components.py
model/hire_v2_identity_model.py
datasets/hire_v2_identity_dataset.py
run_hire_v2_identity_4090_tag.sh
run_hire_v2_identity_smoke.sh
tools/hire_v2/audit_identity.py
tools/hire_v2/eval_identity_components.py
tests/test_hire_v2_identity_components.py
```

## 版本二修改

```text
model/__init__.py
datasets/build.py
utils/options.py
solver/build.py
processor/processor.py
```

## 明确未包含

```text
数据集；
模型权重；
训练日志；
检查点；
测试集诊断标签；
MLLM 输出；
相似身份负包；
本地 Python 环境。
```
