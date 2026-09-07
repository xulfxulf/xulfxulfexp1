# V007 portrait soft-expert result

Source 262a2f8. RSTP, own portrait E0 best epoch4 initialization. Five
continuation epochs, seed1, four ranks x80, unchanged V005 soft-expert and
loss formulas. No test evaluation.

| Epoch | Validation R1 | Validation mAP |
| --- | ---: | ---: |
| 1, selected | 46.400002 | 37.581055 |
| 2 | 45.900002 | 37.671566 |
| 3 | 45.550003 | 37.496410 |
| 4 | 45.900002 | 37.283108 |
| 5 | 45.400002 | 37.295425 |

Compare only metrics of the best-R1 checkpoint, not epoch2's larger mAP.
Portrait E0 reference is 46.55000305175781 / 37.43480682373047. Thus R1
changes by -0.1500015pp and mAP by +0.1462479pp; joint gain -0.1500015pp.
Decision: discard as a dual-metric improvement. Preserve all records.
The selected epoch1 has not received decorrelation updates and is not an
eligible full-mechanism final-test candidate.

All575 batches visited, two initial synchronized AMP skips, no later skips.
Exit code0, guard passed, launch elapsed597.05s. Compact best/full last and
transactional saving operated successfully; original datasets and older
checkpoints were untouched.

## Train-only patch diagnosis

`patch_evidence.json` compares E0, method best(epoch1), and method last(epoch5)
on the SAME fixed256 training images, one caption/image, zero updates.

- At epoch1 attention normalized entropy is0.991-0.997, with mean maximum
  patch weight0.0075-0.0092 versus uniform0.00521. Residual ratio median1.04%.
- At epoch5 entropy is0.965-0.972; maximum weight0.0186-0.0243. Pairwise
  attention cosine is0.783-0.857; same-input residual output cos^2 is
  0.0038-0.0104. Residual ratio median3.12%, p95 7.05%.
- On this small TRAIN subset, epoch5 shared-only R1 is94.14 and fused R1
  93.36. These are NOT validation metrics. The residual branch is active
  and differentiated, but this does not establish helpful generalization.
- The first standalone diagnostic lacked the configured NLTK search path;
  a new attempt called the existing bootstrap helper and passed. No training
  change was made. Its failed log is stored outside the Git repository.

Disk remaining after this trial is about5.46GiB, below the new-run safety
budget. No prior checkpoint is deleted without the pending user approval.
Further read-only validation can proceed without new training checkpoints.
