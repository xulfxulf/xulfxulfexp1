# HIRE-v2 Identity-Balanced TAG-PEDES 60-Epoch Results

## Run contract

- Experiment version: `v16.2.1`
- Direct baseline version: `v16.2.0`
- Final code commit: `9bbfc167df064ea14f1fc3e88f5ccf8edd126119`
- Source main commit: `7d89aa311eda5aaef8b7f6f200e2cd47de015ad0`
- Source overlay SHA256: `44B7C4498F768FFA2B3D7C16EE2168FAE9374B934CEB69F22CCC8E94CD2E5520`
- Source design SHA256: `44D776E37D8A94DB7D26434B96A7A56E6438FE6C01A6F1F3E956F73C03CAEF15`
- Hardware: one NVIDIA GeForce RTX 4090
- Environment: Python 3.8.20, PyTorch 1.9.0+cu111, torchvision 0.10.0+cu111, CUDA 11.1
- Dataset: TAG-PEDES
- Training: 60 epochs, seed 1, batch size 64, random sampler, no optional image augmentation
- Backbone initialization: OpenAI CLIP ViT-B/16
- Mode: `--hire_v2 --hire_v2_mode identity_balanced`
- Identity settings: support size 3, auxiliary group weight 0.1, observation/final main weights 0.5/0.5

## Recorded training result

This file records the completed run without adding an experimental comparison.

| Status | Best R1 | Best epoch | Best-epoch R5 | Best-epoch R10 | Best-epoch mAP | Best-epoch mINP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| completed 60/60 | 58.016239166259766 | 54 | 76.412 | 83.022 | 44.534 | 24.495 |

The final Epoch 60 evaluation was R1 57.919, R5 76.539, R10 82.962, mAP 44.537, and mINP 24.499.

## Best-checkpoint component evaluation

| Component | R1 | R5 | R10 | mAP | mINP |
| --- | ---: | ---: | ---: | ---: | ---: |
| `global` | 57.241 | 75.921 | 82.525 | 43.651 | 23.510 |
| `local` | 54.466 | 74.049 | 80.659 | 41.996 | 22.633 |
| `observation` | 57.495 | 76.236 | 82.713 | 44.049 | 23.995 |
| `identity` | 52.902 | 72.825 | 79.865 | 41.869 | 23.133 |
| `final` | 58.034 | 76.394 | 82.998 | 44.540 | 24.503 |

The training evaluator and the separately executed chunked component evaluator are retained as distinct raw outputs. Their best-checkpoint final R1 values are 58.016 and 58.034, respectively. The small difference comes from the separate similarity-ranking paths.

## Training telemetry

- The valid support-group ratio ended at 98.64%.
- `identity_group_nce` was 0.5750 at Epoch 60 iteration 600.
- The learned identity gate ended at 0.102637 and did not saturate near 0 or 1.
- The final logged mean support count was 2.7002.
- The final observation/identity cosine was 0.8733.
- The final absolute identity score delta was 0.0955; the observation-to-final absolute score delta was 0.0098.
- Observation and final main weights remained fixed at 0.5/0.5.

## Validation

- The formal run started at 2026-07-17 15:29:42 CST and finished at 2026-07-17 19:31:49 CST with exit code 0.
- The log contains 60 validation blocks, `Epoch 60 done`, and the final best-R1 record.
- The server-side `best.pth` was present and 1,098,715,657 bytes.
- No traceback, runtime error, CUDA OOM, CUDA error, NaN, or Inf marker was found.
- Local and server identity-balanced tests passed: `26 passed`; the seven-item static mathematical audit passed.
- The best-checkpoint five-component evaluation finished at 2026-07-18 00:33:38 CST with exit code 0.
- No model function, formula, loss, data flow, training flow, or evaluation calculation was changed for environment compatibility.
- The checkpoint is intentionally not stored in GitHub.

## Archived-file hashes

| File | SHA256 |
| --- | --- |
| `train_log.txt` | `F1B69C227543086740CEC4F6D6DB7FFA6E93E15E918F4367C1995C059E964A4B` |
| `configs.yaml` | `19B559C420BAAD0B80213429C7CC26B0142FDDFA26B64CCF010E2FCC72224958` |
| `identity_balanced_components.json` | `C8D234D0D67D05ABC9061419D71F81AAB72F2B7D45C5AB46283758A47A751648` |
| `component_eval.log` | `324599545FBD80EAB26E9424D41C473584A7BF601B94F93629E73CF5A4122631` |
| `static_audit.json` | `8DD202531045AD0DFB00ED47226F202DB6A747A2FAE78A74027BFE1C35477C53` |
| `run_record/nohup.log` | `82DDF9E60D67BD23ECF15D63602A158E42499D10F62E9A4D85071D85E8B0B340` |
| `run_record/run_manifest.txt` | `DA0AD413CD8466F348D7A39DAB56132B62DC68C158166D785790F117378F9559` |
| `component_eval_run_record/command.sh` | `992968F5C679969D6AA18A42AF7C3BAD69AA71904D2BE270558FC6284FC3FA32` |
| `component_eval_run_record/run_manifest.txt` | `4BE428CDC4D6880BB7AA072108FB40CED041788A8BAE99C9153B2216B35E2BE7` |
