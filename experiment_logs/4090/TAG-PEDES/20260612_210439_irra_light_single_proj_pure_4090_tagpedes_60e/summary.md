# TAG-PEDES IRRA-light Single Projection Pure Full Run

- Server: 4090
- Dataset: TAG-PEDES
- Mode: single_proj_pure
- Epochs: 60
- Batch size: 64
- Seed: 1
- Image augmentation: off
- Identity loss: sdm
- Sampler: random
- Evaluation split: test
- Backbone: ViT-B/16
- Status: completed

This file records the run result only. It does not rank or compare modes.

The recorded headline result is best R1.

## Recorded Best Metrics

Best R1 epoch 45:

| task | R1 | R5 | R10 | mAP | mINP |
|---|---:|---:|---:|---:|---:|
| t2i | 57.538 | 76.072 | 82.653 | 43.694 | 23.408 |

## Final Metrics

Epoch 60:

| task | R1 | R5 | R10 | mAP | mINP |
|---|---:|---:|---:|---:|---:|
| t2i | 57.386 | 76.151 | 82.574 | 43.792 | 23.462 |

## Files

- `train_log_full.txt`: full successful training log
- `configs.yaml`: saved run configuration
- `run_stdout_full_pair.txt`: paired launcher stdout/nohup log
