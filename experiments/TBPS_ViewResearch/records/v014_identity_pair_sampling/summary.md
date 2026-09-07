# V014: retain for the newly authorized mAP objective

Source4b9be6b; CUHK384x128,OpenAI initialization,seed1,five epochs,4x80.
Only main sampling changed from V012 to existing view-aware P160 K2 plans.
The complete model/loss source differs only in a version-name error string;
shared consistency,soft confidence-weighted patch experts and decorrelation
remain unchanged. Actual main batches have160 PIDs,two candidates per PID,
and about one different-image positive per anchor. Single-image repeats are
explicitly counted and cannot replace different images for multi-image IDs.

| Epoch | Validation R1 | Paired validation mAP |
| --- | ---: | ---: |
| 1 | 57.681065 | 53.916954 |
| 2 | 60.344265 | 56.873806 |
| 3 | 66.921074 | 63.312679 |
| 4 | 68.999672 | 65.797386 |
| 5,selected | 69.990257 | 67.439011 |

All1060 batches completed,exit0,five initial synchronized AMP skips,847
actual decorrelation updates. Runtime about767seconds. All32 server unit
tests passed; eight-step four-GPU audit performed three actual updates.
Raw audit logs are local-only.

Versus retained V01273.335495/65.948112: R1-3.345238,mAP+1.490898 percentage
points. This is a meaningful mAP gain with a substantial R1 tradeoff,not a
joint-metric improvement. Retain under the user's amended single-metric
objective only. Preserve V012 as the stronger-R1 alternative. Do not
silently rewrite any historical joint-metric ledger decisions.

No test was evaluated. Validation67.44 does not establish exceeding RDE's
test67.56,let alone the prospective68.56 working target. More validation-
based improvement is needed before choosing the one final test candidate.
Compact best and records are persisted. Full local archival is pending;
do not claim it complete before its verification receipt exists.
