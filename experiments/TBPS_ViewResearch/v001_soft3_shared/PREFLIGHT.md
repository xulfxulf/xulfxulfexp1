# Preflight evidence, 2026-09-07

- New unit tests: 9/9 passed locally and in the actual server environment.
  Covers boundaries, full probability mirroring, invalid probabilities,
  zero-head equivalence, detached decorrelation, shared-feature gradients,
  empty-pair backward, support identity and unchanged main sampler/RNG.
- Legacy tests: target Linux environment, 49 run, 48 passed and one GPU-only
  FP16 test skipped by its CPU guard. The same suite is not a Windows target:
  a local attempt failed on Linux paths, missing NLTK stopwords and fcntl.
  No model code was changed to hide those environment failures.
- Actual four-GPU audit: 4 x 80 main samples plus at most 16 supports per rank;
  4 steps completed, exit 0, parameter/scaler statistics synchronized across
  all ranks. Steps 1 and 3 overflow-skipped together during initial AMP scale
  adjustment; steps 2 and 4 updated normally. No OOM or nonfinite forward loss.
- Main batch distinct PIDs: 314, 317, 313, 314. Support pairs: 62, 64, 62, 61.
- Rank-0 peak allocated memory: 6,742,703,104 bytes (6.28 GiB).
- The audit uses no validation or test evaluation and saves no checkpoint.
- Author-source original retrieval equations remain unchanged in the new
  forward. New auxiliary terms are separately logged and have gradient tests.

Target: Python 3.8, torch 1.13.0+cu117, torchvision 0.14.0+cu117;
four 32 GiB RTX 4080 GPUs on TBPR-PRO1. No environment reinstall.
