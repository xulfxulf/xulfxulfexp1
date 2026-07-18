# HIRE-v2 v16.4.0 覆盖包清单

## 源代码基线

```text
仓库：xulfxulf/xulfxulfexp1
提交：07a9fe1cb86d5e223de23ccb02eb319f1eb4402d
```

## 新增文件

```text
model/hire_v2_token_route_components.py
model/hire_v2_identity_token_route_model.py
tools/hire_v2/eval_identity_token_route_components.py
tools/hire_v2/audit_identity_token_route.py
tests/test_hire_v2_token_route_components.py
run_hire_v2_identity_token_route_4090_tag.sh
run_hire_v2_identity_token_route_smoke.sh
HIRE_V2_IDENTITY_TOKEN_ROUTE_DESIGN.md
HIRE_V2_IDENTITY_TOKEN_ROUTE_README.md
HIRE_V2_IDENTITY_TOKEN_ROUTE_CONSISTENCY_AUDIT.md
SOURCE_COMMIT_V1640.txt
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
utils/metrics.py；
v16.1.0 模型；
v16.2.0 模型；
v16.2.1 模型；
v16.3.0 模型；
HIREV2IdentityDataset 支持选择；
CLIP 主干；
RDE 词元选择模块；
优化器；
学习率调度器。
```
