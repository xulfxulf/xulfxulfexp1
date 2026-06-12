# TAG-PEDES IRRA-light Single Pure Full Run

- Server: 4090
- Dataset: TAG-PEDES
- Mode: single_pure
- Epochs: 60
- Batch size: 64
- Seed: 1
- Run directory on server: `/root/autodl-tmp/IRRA_light_baseline/logs_4090/TAG-PEDES/20260612_181706_irra_light_single_pure_4090_tagpedes_60e`
- Status: completed

## Final Metrics

Epoch 60:

| task | R1 | R5 | R10 | mAP | mINP |
|---|---:|---:|---:|---:|---:|
| t2i | 57.647 | 76.321 | 82.647 | 43.713 | 23.526 |

Best R1:

- R1: 57.719337463378906
- Epoch: 52

## Files

- `train_log_full.txt`: full successful training log
- `configs.yaml`: saved run configuration
- `run_stdout_full.txt`: successful run stdout/nohup log
