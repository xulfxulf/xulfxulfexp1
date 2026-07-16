# HIRE-v2 Anchor TAG-PEDES 60-Epoch Results

## Run contract

- Final code commit: `48e61f81649aa2f3ea515d8e967faa4960b2f478`
- Raw overlay commit: `2dc6dd7e24038e217398140eda97227df4b0c04f`
- Base commit: `610c2a405aec4acfdb0d6364872ec4f86d17c588`
- Source overlay SHA256: `F2F21DD414ABD0AB14A2F8F415C5DD70DEB60E7CDFAF497A3F361D65374D2652`
- Hardware: one NVIDIA GeForce RTX 4090
- Environment: Python 3.8.20, PyTorch 1.9.0+cu111, torchvision 0.10.0+cu111, CUDA 11.1
- Dataset: TAG-PEDES
- Training: 60 epochs, seed 1, batch size 64, random sampler, no image augmentation
- Backbone initialization: OpenAI CLIP ViT-B/16
- Mode: `--hire_v2 --hire_v2_mode anchor`

## Recorded training result

This file records the completed run without adding a baseline comparison.

| Status | Best R1 | Best epoch | Best-epoch R5 | Best-epoch R10 | Best-epoch mAP | Best-epoch mINP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| completed 60/60 | 57.773868560791016 | 59 | 76.260 | 82.380 | 44.238 | 24.113 |

The final Epoch 60 evaluation was R1 57.719, R5 76.291, R10 82.319, mAP 44.241, and mINP 24.150.

## Best-checkpoint component evaluation

| Component | R1 | R5 | R10 | mAP | mINP |
| --- | ---: | ---: | ---: | ---: | ---: |
| `global` | 57.186 | 75.915 | 82.265 | 43.643 | 23.538 |
| `local` | 54.750 | 73.982 | 80.853 | 41.934 | 22.447 |
| `observation` | 57.750 | 76.260 | 82.380 | 44.238 | 24.117 |

The training evaluator and the separately executed chunked component evaluator are retained as distinct raw outputs. Their best-checkpoint observation R1 values are 57.774 and 57.750, respectively.

## Validation

- The formal run started at 2026-07-16 10:42:28 CST and finished at 2026-07-16 13:05:22 CST.
- The log contains `Epoch 60 done` and the final `best R1` record.
- The server-side `best.pth` was present and 1,095,567,765 bytes.
- No traceback, runtime error, CUDA OOM, CUDA error, NaN, Inf, segmentation fault, or killed-process marker was found.
- Local and server HIRE-v2 tests passed: `8 passed`.
- The only compatibility change adds the repository root to the offline evaluator's Python import path. No model function, formula, loss, data flow, training flow, or evaluation calculation was changed.
- The checkpoint is intentionally not stored in GitHub.

## Archived-file hashes

| File | SHA256 |
| --- | --- |
| `train_log_full.txt` | `3AD459141831326D37C0CD8D9AA857E6BFCD11F3B332A4C82D514EB601F27471` |
| `run_stdout_full.txt` | `30CABEEE99DE073D3AF7AEA78F6A89AB0D15E764566750D4E45E85B0B65DE05E` |
| `configs.yaml` | `D116845D9FDFF54AAEA63410BE44752ABAAB441ABD33F3E842CB6F4988CC88A2` |
| `hire_v2_anchor_components.json` | `13241602E31BB17F6F5266791846EBE49B53F3F9BA096C629DB3C357FA71B2C3` |
| `component_eval.log` | `D16D8EECA34A26B6BA0FFBA082CBD8C1719B6DF33AFCA259EB6688E82B19345B` |
| `run_manifest.json` | `AE174926B350F15F26E1E3180EB7ABA21D1C3A19887A3BE5AAB616B86E3EB9CB` |
| `command.sh` | `D84FB9D3A7560E7C67601E2605AF3151C525972DCC4BC54A42C3C23E541FF7B7` |
