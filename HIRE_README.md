# HIRE 工程覆盖包使用说明

本压缩包针对以下仓库版本制作：

```text
仓库：https://github.com/xulfxulf/xulfxulfexp1
提交：90228fe720a82e36b04c4ac62e8d3247016c48d8
```

## 1. 安装

建议先在仓库中创建分支：

```bash
cd /root/autodl-tmp/IRRA_light_baseline
git status --short
git switch -c hire-main
```

将压缩包解压到仓库根目录并允许同名文件覆盖：

```bash
unzip HIRE_engineering_overlay.zip -d /root/autodl-tmp/IRRA_light_baseline
```

覆盖包不会替换 `model/build.py`、`model/clip_model.py` 或 `datasets/bases.py`，旧实验主体仍保留。它通过 `model/__init__.py` 增加独立 HIRE 路由，并对参数、数据构建、优化器、训练日志和评测器做兼容扩展。

## 2. 无数据审计

```bash
python tools/hire/audit_hire.py
pytest -q tests/test_hire_components.py
```

## 3. 一轮烟测

修改或传入数据集路径：

```bash
DATA_ROOT=/root/autodl-tmp/datasets \
PYTHON_BIN=/root/miniconda3/envs/irra190/bin/python \
bash run_hire_smoke.sh
```

## 4. 正式训练

```bash
CUDA_VISIBLE_DEVICES=0 \
DATA_ROOT=/root/autodl-tmp/datasets \
OUTPUT_DIR=/root/autodl-tmp/HIRE_logs \
PYTHON_BIN=/root/miniconda3/envs/irra190/bin/python \
SEED=1 \
BATCH_SIZE=64 \
NUM_EPOCH=60 \
bash run_hire_4090_tag.sh
```

## 5. 测试

```bash
python test.py --config_file <训练输出目录>/configs.yaml
```

## 6. 重要限制

- 主版本只支持 CLIP ViT-B/16；
- 不需要 MLLM 标签、同图文本一致性表或离线困难负包；
- 测试时不需要同身份支持集合；
- 主实验新增方法超参数只有支持数量 3；
- RDE 词元比例、TAL 温度和间隔沿用公开默认值，不建议在首轮搜索；
- 正式六十轮前必须先完成一轮烟测。

完整公式与训练逻辑见 `HIRE_MAIN_DESIGN.md`。
