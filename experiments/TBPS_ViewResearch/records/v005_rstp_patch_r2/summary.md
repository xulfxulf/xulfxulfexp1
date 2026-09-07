# V005 RSTP continuation result

Source `4a4d270`, five additional epochs / 575 batches / seed1. Completed
in 612.14 seconds, exit 0. Four-step preflight had three actual synchronized
optimizer updates. Model and loss formulas are the same as CUHK V004.

| Epoch | Validation R1 | Validation mAP |
| --- | ---: | ---: |
| 1 (best R1) | 45.150002 | 36.771713 |
| 2 | 44.250004 | 36.753342 |
| 3 | 44.050003 | 36.478024 |
| 4 | 43.650002 | 36.275650 |
| 5 | 43.400002 | 36.083370 |

Best-R1 deltas against RSTP E0: R1 -0.100002 and mAP -0.080799 percentage
points. Minimum joint gain -0.100002. Discard in the RSTP ledger; retain all
source and records. Test was not evaluated. Selected epoch1 is before
decorrelation starts, so it is not a full-mechanism final-test candidate.

More cross-view support availability alone did not produce an improvement.
This does not isolate which component caused the decline. In particular,
the auxiliary shared loss currently contains positive cosine attraction
only; identity discrimination is provided indirectly by the main retrieval
loss. Inspect that interaction before further scalar LR tuning.
