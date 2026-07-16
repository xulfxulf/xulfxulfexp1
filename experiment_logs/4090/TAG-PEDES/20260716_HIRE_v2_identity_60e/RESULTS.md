# HIRE-v2 Identity TAG-PEDES 60-Epoch Results

## Run contract

- Experiment version: `v16.2.0`
- Direct baseline version: `v16.1.0`
- Final code commit: `82601f865e3df205f88ec69f880284f215751fd0`
- Raw overlay commit: `0648ae4e985d057c945a0c561eada183e3e9c353`
- Source main commit: `09def7e11fe3a2f47b39929013aaad4038b98ac9`
- Source overlay SHA256: `3410B17E3EF3C2F2FEBD83069D6E711822F92202CEEA4B40527F01EFAB356DD2`
- Source design SHA256: `141F35F05B43F8F15D266A83F7122C76D2786AD68C0859667D9A4CA985E94156`
- Hardware: one NVIDIA GeForce RTX 4090
- Environment: Python 3.8.20, PyTorch 1.9.0+cu111, torchvision 0.10.0+cu111, CUDA 11.1
- Dataset: TAG-PEDES
- Training: 60 epochs, seed 1, batch size 64, random sampler, no optional image augmentation
- Backbone initialization: OpenAI CLIP ViT-B/16
- Mode: `--hire_v2 --hire_v2_mode identity`
- Identity settings: support size 3 and auxiliary group weight 0.1

## Recorded training result

This file records the completed run without adding a baseline comparison.

| Status | Best R1 | Best epoch | Best-epoch R5 | Best-epoch R10 | Best-epoch mAP | Best-epoch mINP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| completed 60/60 | 57.90111541748047 | 54 | 76.460 | 82.962 | 44.408 | 24.329 |

The final Epoch 60 evaluation was R1 57.786, R5 76.436, R10 82.931, mAP 44.400, and mINP 24.326.

## Best-checkpoint component evaluation

| Component | R1 | R5 | R10 | mAP | mINP |
| --- | ---: | ---: | ---: | ---: | ---: |
| `global` | 57.307 | 76.163 | 82.780 | 43.647 | 23.468 |
| `local` | 54.611 | 73.782 | 81.089 | 41.947 | 22.557 |
| `observation` | 57.332 | 76.145 | 82.689 | 43.823 | 23.662 |
| `identity` | 53.890 | 72.788 | 80.156 | 42.232 | 23.390 |
| `final` | 57.913 | 76.478 | 82.937 | 44.409 | 24.323 |

The training evaluator and the separately executed chunked component evaluator are retained as distinct raw outputs. Their best-checkpoint final R1 values are 57.901 and 57.913, respectively. The small difference comes from the separate GPU/CPU similarity-ranking paths.

## Training telemetry

- The valid support-group ratio stayed near 98.6%, consistent with the frozen TAG relation audit.
- `identity_group_nce` decreased from 2.585 at Epoch 1 iteration 100 to 0.635 at Epoch 60 iteration 600.
- The learned identity gate ended at 0.111861 and did not saturate near 0 or 1.
- The variance lower- and upper-bound hit ratios remained 0.0 in logged telemetry.
- The final logged mean support count was 2.7002.

## Validation

- The formal run started at 2026-07-16 20:13:32 CST and finished at 2026-07-17 00:16:58 CST with exit code 0.
- The log contains 60 validation blocks, `Epoch 60 done`, and the final best-R1 record.
- The server-side `best.pth` was present and 1,101,869,629 bytes.
- No traceback, runtime error, CUDA OOM, CUDA error, NaN, or Inf marker was found.
- Local and server HIRE-v2 anchor plus identity tests passed: `17 passed`; both mathematical audits passed.
- The real TAG audit found no PID-boundary crossing, anchor-image reuse, duplicate support image, or missing image.
- All 18,046 rotation-eligible caption samples used at least two distinct support sets over 60 epochs.
- The only compatibility change restores the repository-root import path for the retained anchor offline evaluator. No model function, formula, loss, data flow, training flow, or evaluation calculation was changed.
- The checkpoint is intentionally not stored in GitHub.

## Archived-file hashes

| File | SHA256 |
| --- | --- |
| `train_log_full.txt` | `7C5EB5D9482A28DE2976ADC1AAB4BDC14468C436B8D09388D872DFC6E6D2912C` |
| `run_stdout_full.txt` | `C83C8F0C99995E7F86EE106C8986E2DA91F4A185671871D8BA7973609176A63B` |
| `configs.yaml` | `869427D73365405973D62F3946772B378575DF932C63DF6A3E7DC29DCAC370EF` |
| `identity_components.json` | `E89A964DBB9C8F3CD2139C971D3A815C626CB0D1CC56F13F8CC1AFFBC89F9005` |
| `component_eval.log` | `7AD43BA444FF2DD292E57708CD45E76AEC9625000583AB27D764943B32E3F9F4` |
| `run_manifest.json` | `7A40082F2234A5BDC1C96AF581681D1BE58C03136ABFDDD92623CC5B84F56B36` |
| `command.sh` | `7D4665FA22D18A2C7F04FD9500476952074CF71710B85F4CF11C72E5DC065766` |
| `tag_relation_audit.json` | `676E86A8D9BEB79176B383F6A7FA7E2DDA3D18F92B97F64B6E8DE23B67CF9476` |
| `support_rotation_60e_audit.json` | `69F5A30AC3D545A964639AA423066FBE113072DBB0B6113493387549C64204BC` |
| `model_build_audit.json` | `685CF3591BB6616F4C14337A28E86D6280C92F8977E8C41D522368F580E02FF7` |
| `exit_code.txt` | `9A271F2A916B0B6EE6CECB2426F0B3206EF074578BE55D9BC94F6F3FE3AB86AA` |
| `started_at.txt` | `F2AD7185473E32C00A53D2425CA3054EAC0B3FCA4ECC782115C9ACD9D72ADDB4` |
| `finished_at.txt` | `BC2BDD30D0751D0C34382A6D716F419B9960514F0782F28C05CC6BC41B42FCD3` |
