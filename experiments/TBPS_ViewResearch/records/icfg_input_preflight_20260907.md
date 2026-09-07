# ICFG input preflight: unresolved same-image identity conflict

Read-only inspection on TBPR-PRO1 while V012 CUHK trains. No ICFG model,
training run, holdout split or test retrieval evaluation was created.

Sources:
- `/root/autodl-tmp/datasets/ICFG-PEDES/ICFG-PEDES.json`
- `/root/autodl-tmp/TBPS_CLIP_repro_20260904/inputs/official_annotation/annotation/ICFG-PEDES/train_reid.json`
- Same official annotation directory's `val_reid.json` and `test_reid.json`.

Raw total54522 rows. Official train34674 rows,34673 distinct image paths,
3102 IDs numbered0..3101. Official test19848 rows,1000 IDs3102..4101. Official
val is empty. Train/test ID intersection and image-path intersection are zero.
These are annotation-structure checks, not evaluation scores.

One train image path occurs at row indices3 and21 with conflicting IDs:
`test/2868/2868_005_05_0114afternoon_0833_1_ex.jpg`, PID1 and PID5 respectively.
Each row has one caption. The raw dataset JSON has the SAME conflicting
records at the same indices; the conflict was not introduced by TBPS export.
PID1 has3 image rows, PID5 has4. The path's `test/` folder does not determine
the experimental split: both records belong to official train.

Before an ICFG experiment, its derived input preparation must explicitly
resolve or quarantine this image without guessing the correct PID. Random
PID holdout must not place its two copies across train/val. Existing strict
CUHK preparation rejects duplicate paths and must not be bypassed silently.
Possible documented handling is excluding BOTH conflicting records from
derived train/holdout pools while leaving originals unchanged, and applying
the identical filtered pool to E0 and the method. This is a proposal, not an
executed data change or established ground-truth correction.

A train-ID-disjoint fixed validation split is still required. Test must not
be used for checkpoint or hyperparameter selection. No evidence in this
preflight changes the ongoing CUHK run or claims a retrieval improvement.
