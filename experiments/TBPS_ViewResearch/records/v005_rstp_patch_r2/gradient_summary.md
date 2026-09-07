# V005 train-only gradient diagnosis

Checkpoint: V005 RSTP best validation R1, epoch 1. Ten fixed train batches
from the seed-1 epoch-3 plan, four ranks of 80 main pairs plus up to 16
cross-view supports/rank. Zero optimizer updates. All parameter gradients
are averaged across ranks. No validation/test samples enter this audit.
Raw structured measurements: `gradient_audit.json`.

| Parameter scope | Weighted auxiliary / retrieval gradient norm | Direction evidence |
| --- | --- | --- |
| Last visual block, shared | 0.135-0.170 | cosine positive in 10/10 batches |
| Visual projection, shared | 0.053-0.067 | cosine positive in 10/10 batches |
| Residual heads, decorrelation | 0.0067-0.0167 | mixed sign, small norm |
| Patch queries, decorrelation | 0.0180-0.0344 | mixed sign, small norm |

Decorrelation encoder gradient is exactly zero; shared-loss expert/query
gradient is exactly zero. These checks confirm the intended gradient paths.
The sampled evidence does not support pervasive shared/retrieval conflict
or decorrelation dominating the current RSTP heads. It does not prove the
absence of conflict at other checkpoints or on unseen batches.

The first audit attempt failed on a noncontiguous query gradient passed to
NCCL. The standalone audit was corrected with `.contiguous()` before reduction
and rerun in a new output directory. Training code and weights were unchanged;
the failed log is local-only, not published.

Next focused hypothesis: increase shared-backbone continuation peak LR from
1e-6 to 1e-5 while retaining every other V005 method/training setting. Reuse
the same measured RSTP E0 and unopened test protocol. Do not reinterpret
these measurements as proof of the new learning rate's effectiveness.
