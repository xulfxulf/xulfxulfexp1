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
| v002_balanced_aux | Balance shared/decorrelation gradients using V001 train-only audit | Complete; best val 72.3287 / 65.4154, below E0; no test |
| v003_expert_focused | Smaller backbone LR, larger new-expert LR; all losses and data unchanged | Complete; best val 72.7671 / 65.6963; no test; selected before decorrelation |
| v004_patch_evidence | Add learned view-specific patch pooling to the residual input, retain shared CLS | Complete; best val epoch4 72.8158 / 65.5651; full mechanism; no test |
| v005_rstp_patch | Same mechanism on RSTP, where 88.99% of train images have eligible cross-view support | Initial audit had 4 AMP skips and no actual update; no full run; superseded by r2 audit guard |
| v005_rstp_patch_r2 | Same training code/settings; eight-step E0 audit must show actual optimizer updates | E0 val45.2500/36.8525; continuation best45.1500/36.7717, no gain; no test |
| v006_rstp_backbone_lr | Test base peak LR 1e-5 instead of 1e-6 after train-only gradient diagnosis; all formulas unchanged | Complete; best val44.2500/36.6166, no gain; no test |
| v007_rstp_portrait | Test 384x128 geometry with its own portrait S baseline, then unchanged V005 soft experts | Portrait E0 complete, best val46.5500/37.4348; method audit next; no test |

RSTP references and iteration metrics use `rstp/research-results.tsv` and
`rstp/autoresearch-state.json`, independently of the CUHK metric ledger.
Portrait RSTP uses `rstp_portrait/` with its separately measured E0 reference.

V001-V003 select epoch 1, whose decorrelation weight is zero. Their complete
runs execute all mechanisms, but their selected checkpoints do not yet count
as full-mechanism candidates. See the V003 result summary for this distinction.
