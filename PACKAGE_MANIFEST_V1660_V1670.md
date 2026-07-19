# v16.6.0 / v16.7.0 覆盖包清单

## 源基线

```text
xulfxulf/xulfxulfexp1
0863840462dbd4dfdb9e42b2cdfb2f08010b4e6f
```

## 新增模型与数据文件

```text
datasets/phrase_route_io.py
datasets/hire_v2_phrase_route_dataset.py
model/hire_v2_phrase_route_components.py
model/hire_v2_identity_phrase_route_model.py
```

## 新增离线教师工具

```text
tools/mllm/phrase_teacher_common.py
tools/mllm/phrase_extraction.py
tools/mllm/build_phrase_spans.py
tools/mllm/build_phrase_teacher_cases.py
tools/mllm/run_phrase_teacher.py
tools/mllm/merge_phrase_teacher.py
tools/mllm/audit_phrase_teacher.py
tools/mllm/build_phrase_comparative_cases.py
tools/mllm/merge_phrase_comparative_teacher.py
```

## 新增训练集困难异身份工具

```text
tools/hire_v2/extract_phrase_hard_negatives.py
```

## 新增评测、审计和测试

```text
tools/hire_v2/eval_identity_phrase_route_components.py
tools/hire_v2/audit_identity_phrase_route.py
tests/test_hire_v2_phrase_route_components.py
tests/test_phrase_route_io.py
tests/test_phrase_teacher_common.py
```

## 修改文件

```text
model/__init__.py
utils/options.py
datasets/build.py
processor/processor.py
utils/metrics.py
```

## 启动脚本

```text
run_hire_v2_phrase_route_v1660_4090_tag.sh
run_hire_v2_phrase_route_v1660_smoke.sh
run_hire_v2_phrase_route_v1670_4090_tag.sh
run_hire_v2_phrase_route_v1670_smoke.sh
```
