# V012 initial joint training: retained complete-mechanism improvement

Source `0b5017e`, seed1, CUHK384x128,4x80, five total epochs from OpenAI.
The V009 mechanism and weights are unchanged: soft F/S/B patch experts,
shared cross-view cosine consistency and encoder-detached residual
decorrelation. Change is training stage/profile: OpenAI rather than E0
initialization and S initial-training base peak LR1e-4 rather than1e-6.
Expert/query peak LR remains1e-3. No V010/V011 modifications are included.

| Epoch | Validation R1 | Validation mAP | Actual decorrelation updates |
| --- | ---: | ---: | ---: |
| 1 | 57.469955 | 51.720039 | 0 |
| 2 | 63.153618 | 57.188000 | 211 |
| 3 | 68.869759 | 61.687668 | 212 |
| 4 | 72.247482 | 64.773857 | 212 |
| 5, selected | 73.335495 | 65.948112 | 212 |

Frozen portrait E0:72.848328/65.622398. Selected gains:+0.487167 R1,
+0.325714 mAP percentage points; joint minimum+0.325714. Against original
square E0: +0.698280/+0.440369. Both comparisons use the same selected
epoch5 checkpoint, maximum validation R1 with earliest tie.

Retain this version. Unlike the best V009/V010/V011 checkpoints, this
selection follows actual decorrelation updates (847 in total). It is a
complete-mechanism candidate, not a pre-decorrelation selection. The portrait
ledger iteration4 description mistakenly says635; this is a clerical typo.
The immutable epoch records above, whose sum is847, are authoritative.

All1060 batches finished in777.59seconds, exit0. Five initial synchronized
AMP skips and no later skips. All27 server unit tests passed. Eight-step
four-GPU audit performed three actual updates. The model rejects E0 weights
as its initializer; the measured E0 is used only as reference evidence.
Original parameters are all unfrozen, whereas E0 retains original conv1
freezing, so this is not an isolated conv1-controlled component ablation.

No test was evaluated. Validation73.34/65.95 is NOT a claim of exceeding RDE
test75.94/67.56. A future fixed selection still needs an unopened final test.

All records, compact best and full last are protected on persistent server
storage. Full local archive transfer is pending; keep this status until the
subsequent local restore verification receipt exists. Raw smoke logs are
local-only; source and compact completed results are published separately.
