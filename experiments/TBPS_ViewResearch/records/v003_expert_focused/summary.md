# V003 validation result

Source commit: `27e0339`. Seed 1, CUHK-PEDES, 5 continuation epochs.
All 1060 batches completed; two initial synchronized AMP skips; exit code 0.
Elapsed launcher time: 1072.84 seconds. Test was not evaluated.

| Epoch | Validation R1 | Validation mAP |
| --- | ---: | ---: |
| 1 (best R1) | 72.767128 | 65.696274 |
| 2 | 72.523544 | 65.661079 |
| 3 | 72.669693 | 65.604492 |
| 4 | 72.620979 | 65.529472 |
| 5 | 72.556023 | 65.419029 |

At the best-R1 checkpoint, gains against the frozen E0 validation reference
are +0.129913 R1 and +0.188530 mAP percentage points. Keep the optimizer
allocation as a validation candidate, not as a successful RDE comparison.

## Selection limitation

`selected_pre_decorrelation`: epoch 1 has zero decorrelation weight. Therefore
the selected checkpoint has retrieval and shared-consistency training, but
has NOT received residual decorrelation updates. It must not be presented as
a checkpoint validating all required mechanisms. Epoch 3 is the best-R1
checkpoint among epochs 2-5, but that post-hoc subset is not the run's original
checkpoint-selection rule. Do not substitute it silently.

The same selection limitation also applies to V001 and V002, whose best
validation checkpoints are epoch 1. Their complete runs exercised the
decorrelation path, but their selected checkpoints did not.

Raw audit and audit-launch logs are retained locally, excluded from GitHub.
