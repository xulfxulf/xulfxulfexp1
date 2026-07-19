# HIRE-v2 RSTPReid v16 Transfer Results

## Run scope

- Dataset: `RSTPReid`
- Frozen code commit: `e6b31594846ac61cf4376eb99c447ebd49a34134`
- Snapshot SHA256: `f5b7bf724dde45e04c79b4fbb811d50f726b3ee66cab26f1add4be4db8acda15`
- Environment: Python 3.8.20, PyTorch 1.9.0+cu111, torchvision 0.10.0+cu111, CUDA 11.1, NVIDIA RTX 4090
- Shared settings: OpenAI CLIP ViT-B/16, seed 1, batch size 64, random sampler, image augmentation off
- RSTPReid train split: 3,701 identities, 18,505 images, 37,010 captions
- RSTPReid test split: 200 identities, 1,000 images, 2,000 captions

The checkpoint files remain on the server and are intentionally excluded from GitHub.

## Primary validation records

| Version | Status | Best epoch | R1 | R5 | R10 | mAP | mINP |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `v16.1.0` anchor | Completed 60 epochs | 4 | 57.50 | 79.55 | 86.95 | 44.573 | 22.753 |
| `v16.2.1` identity balanced | Completed 60 epochs | 4 | 57.00 | 80.25 | 87.65 | 44.702 | 22.705 |
| `v16.4.0` identity token route | Stopped by user during epoch 35 | 4 | 57.70 | 79.15 | 87.80 | 45.079 | 22.947 |

The last completed validation records were:

| Version | Validation epoch | R1 | R5 | R10 | mAP | mINP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `v16.1.0` | 60 | 55.20 | 78.65 | 86.10 | 44.708 | 23.684 |
| `v16.2.1` | 60 | 55.15 | 77.05 | 83.80 | 44.142 | 23.742 |
| `v16.4.0` | 34 | 53.05 | 76.55 | 85.35 | 44.139 | 24.112 |

`v16.4.0` was stopped at epoch 35 iteration 500/579 on 2026-07-19 21:56:33 CST. Its best checkpoint had already been saved at epoch 4. No training error, OOM, NaN, or Inf was observed before the user-requested stop.

## Best-checkpoint component evaluation

| Version | Component | R1 | R5 | R10 | mAP | mINP |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `v16.1.0` | global | 58.30 | 79.25 | 87.10 | 44.381 | 22.397 |
| `v16.1.0` | local | 49.40 | 73.55 | 83.35 | 39.817 | 20.177 |
| `v16.1.0` | observation | 57.50 | 79.60 | 86.95 | 44.576 | 22.753 |
| `v16.2.1` | global | 57.75 | 80.20 | 87.40 | 44.557 | 22.211 |
| `v16.2.1` | local | 49.00 | 74.30 | 82.75 | 39.658 | 19.726 |
| `v16.2.1` | observation | 57.15 | 80.05 | 87.60 | 44.707 | 22.648 |
| `v16.2.1` | identity | 55.00 | 79.35 | 87.35 | 43.859 | 22.515 |
| `v16.2.1` | final | 57.05 | 80.20 | 87.65 | 44.703 | 22.704 |
| `v16.4.0` | global | 57.90 | 80.00 | 88.00 | 44.862 | 22.321 |
| `v16.4.0` | local | 49.25 | 74.85 | 83.35 | 40.302 | 20.563 |
| `v16.4.0` | observation | 57.80 | 79.10 | 87.80 | 45.083 | 22.899 |
| `v16.4.0` | identity | 56.65 | 79.10 | 87.05 | 44.409 | 22.358 |
| `v16.4.0` | final | 57.75 | 79.15 | 87.85 | 45.084 | 22.936 |

For `v16.4.0`, the best-checkpoint route mean was 0.3003, route standard deviation was 0.0369, and the ratio above 0.5 was 0.0. Observation-to-final top-1 changes contained 5 fixes and 6 breaks.

## Reproducibility files

- `rstpreid_v16_three_full_manifest_e6b3159_run1.txt`: frozen runtime and dataset manifest.
- `rstpreid_v16_three_full_queue_e6b3159_run1.sh`: exact serial launcher used on the server.
- Each version directory contains its configuration, complete or partial training log, component result JSON, evaluator log, timestamps, and exit/stop records.
