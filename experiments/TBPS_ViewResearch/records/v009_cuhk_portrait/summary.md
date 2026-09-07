# V009 CUHK portrait continuation: completed validation experiment

Immutable source: `449b3ac`, seed 1, four GPUs, global batch 320, five additional
epochs from the independently trained five-epoch 384x128 S baseline. All losses,
sampling, soft routing, learned patch queries, and learning rates match V004.
Only input geometry and its matched baseline differ from the square experiment.

| Epoch | Validation R1 | Validation mAP | Actual decorrelation updates |
| --- | ---: | ---: | ---: |
| 1, selected | 72.913284 | 65.640968 | 0 |
| 2 | 72.848328 | 65.680679 | 211 |
| 3 | 72.669693 | 65.545883 | 212 |
| 4 | 72.799606 | 65.446320 | 212 |
| 5 | 72.750893 | 65.359001 | 212 |

Matched portrait E0: validation R1 72.848328, mAP 65.622398, epoch 5. The selected
method checkpoint improves these by 0.064957 and 0.018570 percentage points.
The minimum paired gain is 0.018570. This is a marginal numerical result, not
evidence of a practically useful improvement or a result exceeding RDE.
Selection is maximum validation R1, with earliest tie; mAP is from that same
checkpoint, not its independently best epoch. Test was not evaluated.

Training completed all 1060 batches in 767.30 seconds. Two initial AMP steps
were synchronously skipped; subsequent steps were finite and no run error was
observed. Although epochs 2-5 execute decorrelation, the selected checkpoint is
from epoch 1 and has received zero decorrelation updates. It is therefore not
eligible as the final full-mechanism candidate.

## Storage verification

The complete best and resumable last checkpoints, optimizer state, all four
rank RNG states, configurations, and logs were archived locally under
`D:/004SSH/TBPS_ViewResearch_v009_archives_20260907/validation_run/`.
`archive_verification.json` records a successful local load, 321 compatible
model keys, optimizer presence, four-rank RNG presence, and resume position
epoch 6 / step 0. No file-content hashes were introduced. The server retains a
persistent compact best plus all records; the RAM source was not deleted.
Old experiments and their checkpoints were not removed.

## Train-only continuation-beta diagnosis

After training, the original portrait E0 checkpoint was loaded read-only with
zero-initialized experts. Ten fixed train batches, four ranks x 80 examples,
were evaluated at beta 0 and beta 0.5 using identical forward RNG for each pair.
The actual S retrieval implementation was differentiated; auxiliary weights
were zero. No optimizer update, test evaluation, or checkpoint modification
occurred. Per-rank gradients were reduced before computing the following means.

| Parameter scope | norm(beta=0) / norm(beta=.5) | Direction cosine |
| --- | ---: | ---: |
| Visual last block | 3.253884 | 0.449911 |
| Visual projection | 1.833303 | 0.733522 |
| Text last block | 3.314437 | 0.449087 |
| Text projection | 1.870411 | 0.719117 |
| Residual heads | 1.773630 | 0.796326 |

This shows that resetting beta changes the initial gradient substantially. It
does not establish that the reset caused validation deterioration. A separate
one-variable continuation experiment is required to test that hypothesis.
Full measurements, exact train indices, and runtime metadata are retained in
`beta_gradient_diagnosis.json` and `beta_gradient_manifest.json`.
