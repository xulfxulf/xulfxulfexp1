# V001 result

Five additional CUHK epochs completed, seed 1, source commit `df68e52`.
Runtime: 2026-09-07 09:12:58 to 09:30:51 China standard time, about 18 minutes.
Source: `experiments/TBPS_ViewResearch/v001_soft3_shared`.

| Epoch | Validation R1 | Validation mAP |
| --- | ---: | ---: |
| 1 (selected) | 72.2637 | 65.1536 |
| 2 | 72.0688 | 64.8908 |
| 3 | 71.8577 | 64.5030 |
| 4 | 71.8577 | 64.4992 |
| 5 | 72.0851 | 64.5157 |

E0 best validation: R1 72.6372 / mAP 65.5077. Same-checkpoint deltas:
R1 -0.3735 pp, mAP -0.3541 pp. Joint validation score -0.3735 pp.
No test was evaluated. Do not interpret validation numbers as paper test scores.

1060 batches, two globally synchronized initial AMP skips, all other updates
completed. No OOM/nonfinite loss/model divergence. 20702 of 34054 train images
had at least one eligible cross-view same-ID support (34352 directed pairs).

By epoch 5, same-input expert pair cosine-squared means were approximately
0.00505, 0.00051 and 0.00551. This confirms the heads differ numerically, but
does not establish useful retrieval information. Better decorrelation did not
translate into better validation retrieval.

Decision: do not retain as an improvement; preserve source, both server
checkpoints and all logs. The subsequent read-only gradient audit diagnoses
relative objective strength using train images only, with zero optimizer steps.
