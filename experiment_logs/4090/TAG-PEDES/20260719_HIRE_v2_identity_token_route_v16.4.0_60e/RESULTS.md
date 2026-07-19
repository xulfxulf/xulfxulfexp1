# HIRE-v2 Identity Token Route TAG-PEDES 60-Epoch Results

## Run contract

- Experiment version: `v16.4.0`
- Direct baseline version: `v16.2.1`
- Final code commit: `e8fe98cd7ede709f93814f795aebb678df3a365e`
- Source main commit: `07a9fe1cb86d5e223de23ccb02eb319f1eb4402d`
- Source overlay SHA256: `1F75F883CE98BD3EE2A373A7569FE55D2BAFCED4E86FA326CC8C681965A9543F`
- Source design SHA256: `A05F3B29D8E75BFA74A3582C2520B012E749C2B19F5E981FF41FEF96F7DF82D2`
- Immutable server snapshot SHA256: `7326E6A94AAD6BE03F8926AEB8F278B37BEB9A5E4CB6EF2F8AA499E9A64E3BDB`
- Hardware: one NVIDIA GeForce RTX 4090
- Environment: Python 3.8.20, PyTorch 1.9.0+cu111, torchvision 0.10.0+cu111, CUDA 11.1
- Dataset: TAG-PEDES
- Training: 60 epochs, seed 1, batch size 64, random sampler, no optional image augmentation
- Backbone initialization: OpenAI CLIP ViT-B/16; no checkpoint resume
- Mode: `--hire_v2 --hire_v2_mode identity_token_route`
- Identity settings: support size 3, identity-group weight 0.1, token-route weight 0.1, observation/final main weights 0.5/0.5

## Recorded training result

The standard training evaluator selected the checkpoint by `final` R1.

| Status | Best R1 | Best epoch | R5 | R10 | mAP | mINP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| completed 60/60 | 57.8829345703125 | 53 | 76.594 | 82.998 | 44.379 | 24.244 |

The final Epoch 60 evaluation was R1 57.871, R5 76.503, R10 82.980, mAP 44.392, and mINP 24.278.

## Best-checkpoint component evaluation

| Component | R1 | R5 | R10 | mAP | mINP |
| --- | ---: | ---: | ---: | ---: | ---: |
| `global` | 57.138 | 75.927 | 82.538 | 43.525 | 23.354 |
| `local` | 53.999 | 74.188 | 81.005 | 41.819 | 22.533 |
| `observation` | 57.556 | 76.412 | 82.756 | 44.025 | 23.868 |
| `identity` | 53.763 | 73.134 | 80.041 | 42.881 | 24.135 |
| `final` | 57.925 | 76.587 | 82.998 | 44.387 | 24.246 |

Relative to the same-checkpoint `observation`, the token-routed identity residual improved final R1 by 0.370 and mAP by 0.361. It fixed 114 observation Top-1 errors and broke 53 observation Top-1 successes, for a net gain of 61 queries. It retained 9,446 correct and 6,891 wrong Top-1 cases.

The standard training evaluator and the chunked component evaluator are preserved as distinct raw outputs. Their selected-checkpoint final R1 values are 57.883 and 57.925. The 0.042 difference comes from the component evaluator's explicit feature renormalization and separate ranking path; the formal run result remains 57.883.

## Token-route telemetry

- `token_route_bce` decreased from 0.6877 at Epoch 1 iteration 600 to 0.5659 at the selected Epoch 53 checkpoint and 0.5661 at Epoch 60.
- The train-time prediction/target correlation increased from 0.0139 to 0.4062 at Epoch 53 and remained 0.4041 at Epoch 60.
- Best-checkpoint test route probability mean/std were 0.2676/0.0763; entropy was 0.5655.
- Only approximately 0.000279% of valid test tokens had probability above 0.5, so the router was non-uniform but almost entirely on the low side of the nominal threshold.
- The test identity-token residual norm was 0.7929; the residual therefore did not remain at its zero initialization.
- The final train-time route valid ratio was 97.28%, support-group valid ratio was 98.64%, and hard-negative valid ratio was 100%.
- The identity gate at the selected checkpoint was 0.09296 and remained bounded.

## Baseline and acceptance decision

The design contract uses the v16.2.1 best-checkpoint component result as its direct baseline: R1 58.034, mAP 44.540, and mINP 24.503. On the same component-evaluation path, v16.4.0 changed these metrics by -0.109 R1, -0.153 mAP, and -0.257 mINP. The standard training-evaluator best R1 also decreased from v16.2.1's 58.016 to 57.883.

The mechanism learned a meaningful soft route target and its residual improved its own observation representation. However, the formal route does not pass the v16.4.0 method gate:

- final R1 did not exceed 58.034;
- final mAP did not reach 44.540;
- the stronger 58.234 R1 teacher-version gate was not reached;
- the proportion above 0.5 was effectively zero, violating the non-collapsed high/low routing condition;
- the positive evidence is that correlation was positive, route variance and residual norm were nonzero, and fixes exceeded breaks.

This is a method-level negative main result with a positive mechanism diagnostic. Under the supplied design contract, it must not proceed directly to a formal MLLM-teacher training version without a new offline diagnosis and a separately versioned design.

## Validation

- The one-epoch TAG-PEDES smoke completed training, evaluation, and checkpoint saving before the formal run.
- Local and server-side suites both passed 66 tests; the v16.4.0 static mathematical audit passed 8/8 checks.
- A formula-preserving numerical fix masks padded support slots before variance squaring; this prevents finite padding values from overflowing into NaN for two-support identities.
- The formal run started on 2026-07-19 at 03:46 CST and finished at 07:51 CST, taking about 4 hours and 5 minutes.
- The log contains 60 validation blocks, `Epoch 60 done`, and the final best-R1 record.
- No traceback, runtime error, CUDA OOM, CUDA error, non-finite loss, NaN, or Inf marker was found.
- The server-side `best.pth` was present and 1,102,670,568 bytes; it is intentionally not stored in GitHub.
- No supplied loss, formula, training flow, or checkpoint-selection rule was changed.

## Archived-file hashes

| File | SHA256 |
| --- | --- |
| `configs.yaml` | `62DC5DCA1ED1825D64056157959E61E7E95B9C1DC7608002B3F4722024C8CE86` |
| `train_log.txt` | `ECA14D4B1BF10D5DC0592085722C5B2D38E417446F23D154A9CE0ED61BA4F0A5` |
| `test_log.txt` | `88B9814308DBF28EF46ABA7B7E21C7CC353DCE6D6154AEC4641788969444E08B` |
| `hire_v2_identity_token_route_components.json` | `E6826C6E6C234E9A1205BD752B8168B127C450A3E15F586C055950DCD0E87F09` |
| `run_record/nohup.log` | `ECB08B762811E310BAC7E31150245A99348F7351EA3B3E51B0078C7812475458` |
| `run_record/v1640_full60_command_e8fe98c_run1.sh` | `A4B5665F5069020B0B4D590A6350D3CEFEB4E88CDBC794E7CA79D6707DC2E4A7` |
| `run_record/v1640_full60_manifest_e8fe98c_run1.txt` | `5E308D55E3E94B93069E23C0DBF9CE9F11567EB9B4832975AE56B10F7D7036AB` |
| `smoke_record/nohup.log` | `1B90DC735C53C81F6853FD518FF27636DCD06D48239659FFEE324939AD6F60C1` |

