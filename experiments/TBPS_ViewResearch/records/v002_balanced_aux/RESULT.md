# V002 result

Five additional CUHK epochs completed, seed 1, code commit `778037e`.
Training/validation runtime 1082.83 seconds (18.05 minutes).

| Epoch | Validation R1 | Validation mAP |
| --- | ---: | ---: |
| 1 (selected) | 72.3287 | 65.4154 |
| 2 | 71.9065 | 65.1569 |
| 3 | 72.1988 | 64.8537 |
| 4 | 72.2475 | 64.8219 |
| 5 | 72.1825 | 64.8383 |

E0 best validation R1 72.6372 / mAP 65.5077. Best-checkpoint differences are
-0.3085 pp R1 and -0.0923 pp mAP. V002 is slightly better than V001, but is
not a retained improvement over E0. Test remains unevaluated.

1060 batches completed, two synchronized initial AMP skips; no OOM,
nonfinite loss, cross-rank divergence or residual-norm alarm. The final-epoch
median residual/shared ratio was 0.002506, with maximum batch p95 0.004583
across the run. All code, records and server checkpoints are preserved.

Interpretation: auxiliary gradient balancing alone was insufficient. The
small residual contribution and early validation peak motivate an optimizer
allocation experiment, not a claim that the proposed mechanism is disproved.
