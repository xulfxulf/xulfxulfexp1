# RSTPReid E0 five-epoch reference

Source `4a4d270` (`v005_rstp_patch_r2`). Single seed 1, four GPUs, 115 batches
per epoch and 575 total. Completed in 617.09 seconds, exit 0. Five initial
synchronized AMP skips; no subsequent skips. The eight-step preflight had
three actual optimizer updates; the earlier four-step no-update audit was
inconclusive and is retained locally. Tests: 18/18 on the pinned server.

| Epoch | Validation R1 | Validation mAP |
| --- | ---: | ---: |
| 1 | 39.500000 | 31.332979 |
| 2 | 42.650002 | 33.897327 |
| 3 | 43.600002 | 35.446747 |
| 4 | 44.450001 | 37.075336 |
| 5 (best R1) | 45.250004 | 36.852512 |

Reference: R1 **45.250003814697266**, mAP **36.85251235961914** at epoch 5.
Do not independently select epoch4 mAP. This is a fixed five-epoch reference,
not evidence of convergence. Test has not been evaluated; do not compare
these validation values directly with the paper's test scores.

Official-source check: the vendored S config uses five epochs, batch80 per
rank, N-ITC+R-ITC. The author README launches four ranks. Training labels,
source associations and all 20505 images passed input checks. A train-image
visual spot-check also matched its caption; it is not a complete label audit.

Sources: [official repository](https://github.com/Flame-Chasers/TBPS-CLIP),
[published paper](https://ojs.aaai.org/index.php/AAAI/article/download/27801/27634).

The RSTP research ledger starts from zero joint gain against this measured
baseline. It is separate from CUHK so gains across different datasets do not
compete as though they shared the same validation reference. Only final-test
R1 >65.35 AND mAP >50.88 at the same full-mechanism checkpoint can satisfy the
RSTP goal. Initializer stays on server; no checkpoint is uploaded to GitHub.
