# V006 RSTP backbone learning-rate result

Source: db5f60e. Same RSTP E0 initializer and patch-expert mechanism as V005.
Only base peak LR changes from 1e-6 to 1e-5; expert/query peak remains 1e-3.
Seed 1, four ranks x 80, five continuation epochs, validation-only selection.

| Epoch | Validation R1 | Validation mAP |
| --- | ---: | ---: |
| 1, selected | 44.250004 | 36.616604 |
| 2 | 42.950001 | 35.643120 |
| 3 | 43.250004 | 35.280598 |
| 4 | 42.800003 | 35.016460 |
| 5 | 41.600002 | 34.609241 |

Baseline best validation R1/mAP: 45.250003814697266 / 36.85251235961914.
Joint validation gain: -1.0 percentage point. Decision: discard as an
improvement; preserve code, logs and checkpoints. Increased backbone LR did
not help in this controlled run. No test scores were computed.

All 575 steps were visited; two initial synchronized AMP skips, no later
skips or execution errors. Exit code 0; result guard passed. Elapsed launch
time 617.19 seconds. The selected epoch1 preceded decorrelation, so it is
not a full-mechanism final-test candidate. Raw smoke logs are local-only.

The next independent direction is input geometry. A seed-1 sample of 512
RSTP training image headers has median H/W 2.574; 91.8% have log-aspect
distance closer to 3 than 1. This supports investigating portrait inputs,
but does not establish a retrieval benefit or explain all observed losses.
