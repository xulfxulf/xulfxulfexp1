# TBPS view-conditioned representation research

Goal: keep shared cross-view identity/state features plus front/side/back soft
image residual experts, cross-view shared consistency, and residual-only
decorrelation. Exceed BOTH R1 and mAP of RDE (CVPR 2024) on one fixed test set.

## Frozen reference targets

Author paper: https://github.com/QinYang79/RDE/blob/main/src/RDE_main.pdf
Table 1, 0% synthetic noise, RDE Best (validation-selected checkpoint).

| Dataset | R1 | mAP |
| --- | ---: | ---: |
| CUHK-PEDES | 75.94 | 67.56 |
| ICFG-PEDES | 67.68 | 40.06 |
| RSTPReid | 65.35 | 50.88 |

Both metrics must be strictly higher (at published two-decimal precision) for
the SAME validation-selected checkpoint, not separately selected epochs.

## Starting evidence

Verified 2026-09-07 CUHK E0 five-epoch S reproduction, seed 1:
best validation epoch 5, R1 72.63721466064453, mAP 65.50774383544922.
Historical test R1 73.0506820678711, mAP 65.152099609375. Its checkpoint stays
on TBPR-PRO1 in `TBPS_CLIP_S_View4_E0_5e_E1_5e_20260907_r2/E0/runs/`.
E1 five-epoch continuation changed the sampler and unfreezing together; its
test R1 71.97856140136719 and mAP 67.43281555175781 are not an isolated ablation.

## Research rules

- New versions have independent full code directories and run records.
- Seed 1 only; no 60-epoch or three-seed queue is implied.
- Do not alter original images/annotations or train on validation/test identity.
- Choose experiments and checkpoints on validation only. Do not inspect test
  after every trial. Freeze final selection before one-time test evaluation.
- Main code and compact experiment records are published after each completed
  experiment. No checkpoints, credentials, dataset images, or model downloads
  are uploaded to GitHub.
- User explicitly waived human calibration for now. Existing OEFormer outputs
  are used directly; entropy-derived confidence remains uncalibrated.
- Foreground execution uses the app's active goal. The optional skill hooks do
  not support Windows and were not installed or bypassed.

`research-results.tsv` and `autoresearch-state.json` are updated with the
autoresearch helper after each completed experiment, including failures.
Selection metric is the minimum of R1 and mAP validation gains against E0,
at the best-validation-R1 epoch. Meeting a validation score is NOT test success.

## Experiments

| Version | Hypothesis | State |
| --- | --- | --- |
| v001_soft3_shared | Keep E0 main negatives; add soft3 residuals and auxiliary cross-view supports | Complete; no validation improvement; test not evaluated |
