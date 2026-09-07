# TBPS view-conditioned representation research

Goal: keep shared cross-view identity/state features plus front/side/back soft
image residual experts, cross-view shared consistency, and residual-only
decorrelation. The user's2026-09-07 amendment permits a stronger improvement
in either R1 OR mAP,with the other metric fully reported. The active search
focuses on mAP; prospective working margin is at least1.00 percentage point
over RDE paper test mAP on one fixed dataset. Both-metric improvement is no
longer mandatory. See records/objective_amendment_20260907.md.

## Frozen reference targets

Author paper: https://github.com/QinYang79/RDE/blob/main/src/RDE_main.pdf
Table 1, 0% synthetic noise, published RDE Best row.

| Dataset | R1 | mAP |
| --- | ---: | ---: |
| CUHK-PEDES | 75.94 | 67.56 |
| ICFG-PEDES | 67.68 | 40.06 |
| RSTPReid | 65.35 | 50.88 |

Always report R1 and mAP from the SAME validation-selected checkpoint,not
separately selected epochs. Current working CUHK test mAP target is68.56;
R1 is fully disclosed,not hidden if it falls. Validation scores are not paper
test scores and cannot establish this target.

## Starting evidence

Verified 2026-09-07 CUHK E0 five-epoch S reproduction, seed 1:
best validation epoch 5, R1 72.63721466064453, mAP 65.50774383544922.
Historical test R1 73.0506820678711, mAP 65.152099609375. Its checkpoint stays
on TBPR-PRO1 in `TBPS_CLIP_S_View4_E0_5e_E1_5e_20260907_r2/E0/runs/`.
E1 five-epoch continuation changed the sampler and unfreezing together; its
test R1 71.97856140136719 and mAP 67.43281555175781 are not an isolated ablation.

## Research rules

- New versions have independent full code directories and run records.
- Seed1 only; default five epochs. V013's completed ten-epoch control was
  worse than retained V012 and does not authorize repeated ten-epoch search.
  No60-epoch or three-seed queue is implied.
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
The original joint-gain ledgers remain historical evidence. Active ledger
is cuhk_map_focus/,initialized against measured V012. Its metric is mAP at
the max-validation-R1 checkpoint (earliest tie),not maximum mAP across
independently selected epochs. Meeting a validation score is NOT test success.

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
| v007_rstp_portrait | Test 384x128 geometry with its own portrait S baseline, then unchanged V005 soft experts | E0 val46.5500/37.4348; method best46.4000/37.5811, no dual gain; no test |
| v008_flip_gallery | Fixed CUHK V004 epoch4, average original/mirrored gallery embeddings; matched E0 control | V004 flip val72.7509/65.9936, below E0 flip73.1569/66.0684; discard; no test |
| v009_cuhk_portrait | Transfer the measured portrait-input benefit to CUHK with a fresh matching E0 before unchanged patch experts | E0 72.8483/65.6224; method 72.9133/65.6410 at epoch1, marginal numeric gain only; both full archives verified; no test |
| v010_cuhk_retained_beta | Keep converged S soft-label mixing beta=.5 throughout otherwise identical continuation | Complete; val72.9945/65.6336 at epoch1, joint gain below V009; discard; no test |
| v011_discriminative_shared | Add different-PID negatives to cross-view shared consistency; keep the V009 training profile | Complete; val72.7509/65.6099 at epoch1, below matched E0; discard; no test |
| v012_joint_initialization | Train unchanged V009 mechanism from OpenAI with S initial-training backbone LR, instead of appending it after E0 | Complete; best val73.3355/65.9481 at epoch5, +0.4872/+0.3257 vs portrait E0; retained full-mechanism candidate; no test |
| v013_joint_ten_epochs | Extend unchanged V012 initial-training horizon to10 epochs,with separately measured10-epoch E0 | Complete; E0 72.0039/64.9814,method72.2637/65.4230; below V012,discard; no test |
| v014_identity_pair_sampling | Existing view-aware P160 K2 sampling instead of random caption-pair batches; unchanged five-epoch V012 model/loss | Complete;69.9903/67.4390 at epoch5; mAP+1.4909,R1-3.3452 vs V012; retain for amended mAP objective; no test |
| v015_ritc_floor | R-ITC epsilon0.01->0.001 only,otherwise unchanged five-epoch V014 |33server tests and8-step four-GPU audit passed; formal running; no test |

RSTP references and iteration metrics use `rstp/research-results.tsv` and
`rstp/autoresearch-state.json`, independently of the CUHK metric ledger.
Portrait RSTP uses `rstp_portrait/` with its separately measured E0 reference.
Portrait CUHK uses `cuhk_portrait/`, likewise initialized after measuring its
own E0. Geometry changes never silently replace the original square reference.

V001-V003 select epoch 1, whose decorrelation weight is zero. Their complete
runs execute all mechanisms, but their selected checkpoints do not yet count
as full-mechanism candidates. See the V003 result summary for this distinction.
V009 likewise selects epoch 1, before decorrelation. Its higher numerical score
does not supersede V004 as a fully trained mechanism checkpoint, and no new
test score has been used to select these experiments.

V012 and V014 both select epoch5 and include847 actual decorrelation updates.
Retain V012 as the stronger-R1 option and V014 as current stronger-mAP option;
neither has been evaluated on test. Do not claim a joint improvement for V014.
