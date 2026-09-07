# V011 discriminative shared objective: completed, no improvement

Source `d9efb89`, seed1, CUHK portrait, four GPUs x80, five additional epochs.
Same measured E0 and all other settings as V009. Only the shared objective is
replaced by symmetric temperature-scaled cross-view NCE, with same-PID bank
entries excluded and tau0.07. V010's retained-beta change is not included.

| Epoch | Validation R1 | Validation mAP | Actual decorrelation updates |
| --- | ---: | ---: | ---: |
| 1, selected | 72.750893 | 65.609901 | 0 |
| 2 | 72.702171 | 65.509048 | 211 |
| 3 | 72.539780 | 65.397011 | 212 |
| 4 | 72.572258 | 65.276733 | 212 |
| 5 | 72.588501 | 65.259079 | 212 |

Matched E0:72.848328/65.622398. Selected gains are-0.097435R1 and-0.012497mAP
points; minimum paired gain-0.097435. Discard under the registered objective.
This is evidence against this particular continuation objective, not proof
that discriminative cross-view learning is universally ineffective.

All1060 batches completed in777.24seconds, exit0. Two initial synchronized AMP
skips; none later. All30 targeted tests passed, including formula, same-PID
masks, graph-connected zero, and encoder/head gradient isolation. Two CPU
ranks reproduced the single-global-reference loss/gradient for uneven support
counts, including one rank with no support. Four-GPU audit made two updates.

The ten-batch train-only gradient audit used the frozen V009 selected model as
a diagnostic reference, not as the formal training initializer. Weighted
shared/retrieval gradient norms were13.08%-15.30% for the last visual block,
5.60%-6.98% for visual projection. Direction cosine ranged-0.107to0.187 and
-0.045to0.076 respectively. Shared loss had zero gradient to experts/queries;
decorrelation had zero gradient to the visual encoder. No diagnostic optimizer
step or test evaluation was performed.

Best=max validation R1, earliest tie; mAP from the same checkpoint. Selected
epoch1 precedes actual decorrelation, so this is not a full-mechanism final-test
candidate. No test was opened; there is no claim of exceeding RDE.

All records and compact best are persisted on the server. The full local
archive in `D:/004SSH/TBPS_ViewResearch_v011_archives_20260907/validation_run`
passed model/config/selection/optimizer/four-rank-RNG restore checks; see
`archive_verification.json`. Redundant server last is eligible for the
separately approved cleanup, which preserves best and all experiment records.
