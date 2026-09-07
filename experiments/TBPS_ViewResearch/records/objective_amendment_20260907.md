# User objective amendment,2026-09-07

The user now permits concentrating on either R1 or mAP, provided that the
gain is larger. Simultaneous improvement of both paper metrics is no longer
a hard requirement. The working single-metric target is at least1.00
percentage point above the RDE paper's corresponding test metric on one
dataset. This numeric margin is the agent's explicitly stated conservative
working interpretation, not a verbatim user-specified threshold.

| Dataset | RDE R1 | RDE mAP | Working R1 target | Working mAP target |
| --- | ---: | ---: | ---: | ---: |
| CUHK-PEDES | 75.94 | 67.56 | 76.94 | 68.56 |
| ICFG-PEDES | 67.68 | 40.06 | 68.68 | 41.06 |
| RSTPReid | 65.35 | 50.88 | 66.35 | 51.88 |

Primary reference remains the author paper,0%noise,Best column:
https://github.com/QinYang79/RDE/blob/main/src/RDE_main.pdf
Do not compare our validation scores directly with these test thresholds.

Prospective search preference: mAP, with R1 always reported from the same
checkpoint. Existing result rows and selection rules stay unchanged. Do
not claim an old both-exceeded label for a new single-metric success. A new
single-metric ledger must be initialized through the bundled helper from
an already measured baseline before running the next trial. A checkpoint
must have received actual full-mechanism updates and be selected on val
before its final test is opened. Test feedback must not choose subsequent
settings. For continuity, maximum validation R1 with earliest tie remains
the checkpoint selection rule; optimize the mAP at that selection rather
than retrospectively selecting an unsaved maximum-mAP epoch.

The user also prefers five-epoch experiments but delegates tuning. Future
trials default to five epochs. V013's already-started ten-epoch method will
finish once as the planned matched-horizon comparison; no repeated ten-epoch
search is intended. Overall shared representation, soft confidence-weighted
view experts and cross-view residual decorrelation remain mandatory.

Checkpoint cleanup authorization now covers server checkpoints. Current
cleanup remains deliberately narrow: completed current-round last files
only, after verified local archival, with best, code, logs and raw datasets
preserved. Active run checkpoints are never removed.
