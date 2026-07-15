# HIRE 工程包清单

- 基础仓库提交：`90228fe720a82e36b04c4ac62e8d3247016c48d8`
- 交付形式：覆盖式工程包，解压到仓库根目录
- 已完成检查：
  - Python 语法编译通过；
  - `tests/test_hire_components.py` 共 9 项通过；
  - `tools/hire/audit_hire.py` 通过；
  - 轻量伪主干前向已完成；
  - 未在本环境运行真实 CLIP 权重、TAG-PEDES 数据和 GPU 一轮烟测。

## 覆盖文件

```text
model/__init__.py
model/hire_components.py
model/hire_model.py
datasets/build.py
utils/options.py
utils/metrics.py
solver/build.py
processor/processor.py
run_hire_4090_tag.sh
run_hire_smoke.sh
tools/hire/audit_hire.py
tests/test_hire_components.py
HIRE_MAIN_DESIGN.md
HIRE_README.md
SOURCE_COMMIT.txt
```
