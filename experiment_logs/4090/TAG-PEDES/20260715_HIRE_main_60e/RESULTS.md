# HIRE Main TAG-PEDES 60-Epoch Results

## Run contract

- Code commit: `5aa91044da70bde770d749bcc21b29b15b3b5bbe`
- HIRE overlay commit: `9d6eed842faca1d21ef02e254d63741c57756a87`
- Base commit: `90228fe720a82e36b04c4ac62e8d3247016c48d8`
- Hardware: one NVIDIA GeForce RTX 4090
- Environment: Python 3.8.20, PyTorch 1.9.0+cu111, torchvision 0.10.0+cu111, CUDA 11.1
- Dataset: TAG-PEDES
- Training: 60 epochs, seed 1, batch size 64, random sampler, no image augmentation
- Backbone initialization: OpenAI CLIP ViT-B/16
- HIRE support size: 3

## Recorded result

This file records the completed run without adding a baseline comparison.

| Status | Best R1 | Best epoch | Final R1 | Final R5 | Final R10 | Final mAP | Final mINP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| completed 60/60 | 48.96994400024414 | 24 | 48.794 | 67.747 | 75.085 | 40.675 | 23.995 |

At the best-R1 epoch, the logged metrics were R1 48.970, R5 68.208, R10 75.697, mAP 40.793, and mINP 24.077.

## Validation

- The formal run started at 2026-07-15 19:31:45 CST and finished at 2026-07-16 00:10:15 CST.
- The log contains `Epoch 60 done` and the final `best R1` record.
- No traceback, runtime error, CUDA OOM, CUDA error, NaN, Inf, segmentation fault, or killed-process marker was found.
- The server-side `best.pth` was present and 1,120,846,582 bytes.
- The checkpoint is intentionally not stored in GitHub. This directory contains the full text log, stdout, configuration, command, and run manifest.
- The only compatibility change was importing `distutils.version` before PyTorch 1.9's TensorBoard shim; no HIRE loss, formula, or training flow was changed.

## Archived-file hashes

| File | SHA256 |
| --- | --- |
| `train_log_full.txt` | `298BF955D6C871577D975F6695DD036989D22E1F28380B6E4515B4477721BFEE` |
| `configs.yaml` | `CA065D3439A51990F5368B2F8C12319ED21F19D282A87ACC1ADD79C238388454` |
| `run_stdout_full.txt` | `0F89D467FF5D0981F634C3430DE8A4DBFDE589A51B154E2C1FC8DD0F54DD0230` |
