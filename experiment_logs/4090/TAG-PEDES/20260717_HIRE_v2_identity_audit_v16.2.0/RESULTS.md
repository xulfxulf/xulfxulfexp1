# v16.2.0 Identity Mechanism Audit Results

## Audit contract

- Target experiment: `v16.2.0` HIRE-v2 identity
- Direct baseline: `v16.1.0` HIRE-v2 anchor
- Status: completed successfully, exit code `0`
- Started: `2026-07-17T02:20:51+08:00`
- Finished: `2026-07-17T02:24:14+08:00`
- Hardware: one NVIDIA GeForce RTX 4090
- Environment: Python 3.8, PyTorch 1.9.0+cu111, CUDA 11.1
- Training/backward/checkpoint update: none
- Audited train images: `19,954`
- Audited train captions: `39,908`
- Audited test captions: `16,504`
- Support epochs: `0, 15, 30, 45, 54, 59`
- Strict paired-group retrieval epoch: `54`

The audit ran from an isolated copy of the exact `v16.2.0` training source snapshot. Core source files matched the local `82601f8` snapshot by SHA256. The original training code, checkpoints, configurations, and archived logs were not modified.

## Trusted-intersection audit

| Diagnostic | Full result |
| --- | ---: |
| Predicted variance mean | 1.050010 |
| Predicted variance standard deviation | 0.000220 |
| Median trusted/simple cosine | 1.000000 |
| Groups with trusted/simple cosine >= 0.9995 | 100.0000% |
| Median effective precision dimension CV | 0.000129 |
| Mean heterogeneity share | 0.039363% |
| Simple group R1 | 31.523 |
| Variance-only group R1 | 31.526 |
| Trusted group R1 | 31.526 |
| Trusted minus simple group R1 | +0.002542 |

All six support epochs show the same pattern. The full heterogeneity-aware trusted intersection is numerically almost identical to the simple support mean. Its `+0.002542` R1 difference is far below the guide's `0.05` attribution threshold, so the current gain cannot be attributed mainly to complex probabilistic weighting.

The epoch-54 paired-group CSV contains `39,368` rows rather than `39,908` because only `98.6469%` of queries have a valid strict leave-one support group. The remaining `540` queries are intentionally excluded by the minimum-support rule.

## Identity-residual audit

| Comparison | Fix | Break | Net top-1 | Rank improved | Rank worsened | Mean AP delta | Mean mINP delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `v16.2 observation -> v16.2 final` | 185 | 89 | +96 | 2,604 | 1,249 | +0.005861 | +0.006603 |
| `v16.1 observation -> v16.2 final` | 892 | 865 | +27 | 3,572 | 3,377 | +0.001713 | +0.002054 |

The identity residual is useful inside `v16.2.0`: it repairs 96 more top-1 queries than it breaks. However, the net repair relative to the `v16.1.0` observation baseline is only 27 queries. This supports the audit guide's diagnosis that the identity residual is partly compensating for degradation of the `v16.2.0` observation anchor.

## Decision

1. Do not further strengthen or claim the current variance-based trusted weighting as the main source of improvement.
2. Retain the identity-residual direction because its internal net top-1 effect is positive.
3. Prioritize a `v16.2.1` anchor-balance correction that protects the `v16.1.0` observation space while retaining the useful identity residual.
4. Treat the current trusted intersection as ordinary group consensus unless a later implementation produces meaningful precision variation and a stable group-retrieval gain.

The generated report is preserved in `v162_identity_audit_report.md`; all query-level fix, break, rank-change, and group-level CSV files are archived alongside it.
