# V004 validation result

Source `eda758f`, CUHK-PEDES, seed 1. Completed 5 epochs / 1060 batches in
1098.00 seconds; exit 0. Two synchronized initial AMP skips. Targeted tests
13/13 passed on the server, plus a four-GPU real-update audit. Raw audit logs
remain local only. No test evaluation.

| Epoch | Validation R1 | Validation mAP | Decorrelation updates in epoch |
| --- | ---: | ---: | ---: |
| 1 | 72.783371 | 65.681473 | 0 |
| 2 | 72.653458 | 65.709846 | 211 |
| 3 | 72.734650 | 65.619255 | 212 |
| 4 (best R1) | 72.815849 | 65.565125 | 212 |
| 5 | 72.750893 | 65.440483 | 212 |

Selected checkpoint received all mechanisms. Relative to E0 validation,
R1 +0.178635 and mAP +0.057381 percentage points. The minimum joint gain
0.057381 is below the retained V003 validation metric 0.129913. Discard as the
CUHK numeric incumbent while preserving this first full-mechanism candidate.
These tiny validation gains do not justify claiming or checking an RDE win.

## Next data-dependent direction

Train-only existing OEFormer annotations, same-PID different-image, different
peak F/S/B and circular angle gap >=60 (before random training transforms):

| Dataset | Train images | Images with eligible support | Fraction |
| --- | ---: | ---: | ---: |
| CUHK-PEDES | 34054 | 20702 | 60.79% |
| RSTPReid | 18505 | 16467 | 88.99% |

Train identities with 1/2/3 distinct F/S/B groups: CUHK 3371/6563/1069;
RSTPReid 150/1704/1847. This motivates a RSTPReid trial, not a guaranteed gain.
Its official annotation has complete paired training back-translations and
disjoint 3701/200/200 train/val/test identities. No TBPS RSTP E0 exists on the
server, so establish a train-only/validation-selected E0 before continuation.
Do not reuse CUHK weights as if they were a RSTP-converged baseline.
