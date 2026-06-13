# TAG-PEDES IRRA-light Split Pure Run

- Server: 4090
- Dataset: TAG-PEDES
- Mode: split_pure
- Epochs configured: 60
- Batch size: 64
- Seed: 1
- Image augmentation: off
- Identity loss: sdm
- Sampler: random
- Evaluation split: test
- Backbone: ViT-B/16
- Run directory on server: `/root/autodl-tmp/IRRA_light_baseline/logs_4090/TAG-PEDES/20260612_231536_irra_light_split_pure_4090_tagpedes_60e`
- Status: interrupted by scheduled server shutdown after epoch 56 and during epoch 57; user accepted `best.pth` at epoch 52 as the result to record.
- Checkpoint used for result tracking: `best.pth`, checkpoint epoch 52.

This file records the run result only. It does not rank or compare modes.

The recorded headline result is best R1.

## Recorded Best Metrics

Epoch 52 (`best.pth`):

| task | R1 | R5 | R10 | mAP | mINP |
|---|---:|---:|---:|---:|---:|
| t2i | 57.235 | 76.442 | 82.701 | 43.597 | 23.321 |

## Last Complete Validation Before Interruption

Epoch 56:

| task | R1 | R5 | R10 | mAP | mINP |
|---|---:|---:|---:|---:|---:|
| t2i | 57.174 | 76.357 | 82.707 | 43.626 | 23.378 |

## Files

- `train_log_interrupted.txt`: training log through Epoch[57] Iteration[500/624]
- `configs.yaml`: saved run configuration

Note: model weights are intentionally not stored in GitHub. The stopped resume attempt is intentionally not synced.
