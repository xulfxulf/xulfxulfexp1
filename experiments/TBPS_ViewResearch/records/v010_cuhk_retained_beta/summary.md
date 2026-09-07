# V010: retained continuation beta did not improve the joint objective

Immutable source `2a7927c`. Same CUHK portrait E0 checkpoint, seed1, 4x80 main
batch, five additional epochs, augmentation, soft-view patch experts, shared
consistency and decorrelation as V009. Only beta differs: fixed0.5 rather than
restarted0-to-0.5 in epoch1. This is a validation-only experiment.

| Epoch | Validation R1 | Validation mAP | Actual decorrelation updates |
| --- | ---: | ---: | ---: |
| 1, selected | 72.994476 | 65.633606 | 0 |
| 2 | 72.767128 | 65.663689 | 211 |
| 3 | 72.832085 | 65.553459 | 212 |
| 4 | 72.734650 | 65.431122 | 212 |
| 5 | 72.702171 | 65.348724 | 212 |

Matched E0 is72.848328/65.622398. The selected V010 checkpoint gains0.146149
R1 and0.011208mAP points; the minimum paired gain0.011208 is below V009's
0.018570. Against V009, selected R1 rises0.081192 but mAP falls0.007362.
Therefore discard this change under the registered joint objective. It does
not establish that every beta schedule is ineffective; it fails to solve the
observed post-epoch1 decline in this controlled trial.

All1060 batches completed, with two initial synchronized AMP skips and none
later. Runtime772.55seconds, exit0. All26 targeted unit tests passed and the
four-step four-GPU audit performed three actual optimizer updates. Source and
data remained unchanged during the run.

Best is selected by validation R1, earliest tie; mAP comes from the same epoch.
Although later epochs execute decorrelation, the selected epoch1 does not.
It is not a full-mechanism final-test candidate. No test was evaluated and no
claim of exceeding RDE is made.

Server compact best and all run records were persisted successfully. Full
best+resumable-last archive in
`D:/004SSH/TBPS_ViewResearch_v010_archives_20260907/validation_run/` passed
local model/config/selection/optimizer/four-rank-RNG restore verification.
See `archive_verification.json`. RAM source remains intact; server redundant
last is eligible for the separately authorized archive-then-delete cleanup.
