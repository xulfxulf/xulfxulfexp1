# V017: five-epoch ICFG mAP-focused result

## Outcome

The user-amended single-metric objective is met on the first frozen ICFG
test evaluation. It is NOT a joint R1/mAP improvement. Stop parameter search;
do not use this test result to choose another checkpoint or hyperparameter.

| Official ICFG test | R1 | R5 | R10 | mAP |
|---|---:|---:|---:|---:|
|RDE paper,0% noise,Best|67.68|82.47|87.36|40.06|
|V017,validation-selected epoch5|58.958080|75.473602|81.459091|41.812618|
|Difference,percentage points|-8.721920|-6.996398|-5.900909|+1.752618|

Reference: [RDE paper Table1](https://openaccess.thecvf.com/content/CVPR2024/papers/Qin_Noisy-Correspondence_Learning_for_Text-to-Image_Person_Re-identification_CVPR_2024_paper.pdf).
The working criterion was mAP at least1pp above40.06,with R1 disclosed.
This comparison demonstrates a substantial R1 tradeoff,not overall superiority.

## Frozen selection

Source96b8827; selection committed4a30f57 at2026-09-07 18:35:52+08:00.
Test started18:38:36+08:00 and completed18:41:04+08:00,exit0.
Selection=max validationR1,earliest tie. Both metrics always use that same
checkpoint. No test scores were inspected during V001-V017 model selection.
Historical pre-research CUHK baseline tests remain separately identified.

| Fixed ICFG train-ID holdout | Selected epoch | R1 | mAP |
|---|---:|---:|---:|
|S E0,5epochs|5|79.958611|60.705067|
|V017,5epochs|5|76.559265|63.369148|
|Difference||-3.399345|+2.664082|

Validation and official test are different retrieval pools. Never compare
validation63.37 against paper test40.06. E0 was only evaluated on validation;
no matched E0 test gain is claimed. V017 starts independently from OpenAI,
not from this E0 checkpoint. E0 runtime377.008s; method426.827s; test148.056s.

## Preserved mechanism

- Shared CLIP global image feature and unified CLIP text representation.
- Three F/S/B image residual experts with learned patch-query pooling.
- Fixed OEFormer soft3 probabilities and uncalibrated entropy confidence;
  manual calibration was explicitly waived by the user.
- Confidence-weighted same-ID,cross-view support consistency on shared features.
- Residual-only cross-view decorrelation with BOTH shared/patch encoder
  outputs detached in that auxiliary branch; retrieval still updates all.
- Unified normalized cosine retrieval. No reranking,test-PID feature,input
  from the query's paired-image angle,or test-adapted parameters.

Read v017_icfg_joint/README.md and configs/research_v017.json for the complete
formulas and executable configuration. Compared with E0,the full setting also
changes main sampling,conv1 unfreezing and R-ITC floor. This experiment does
not isolate the causal contribution of experts or decorrelation.

## Training and data

ViT-B/16,384x128,seed1,4xRTX4080,global320=P160K2. Five epochs,97steps each.
AdamW; base peakLR1e-4; expert/query peakLR1e-3; R-ITC epsilon0.001.
Shared weight ramps to0.4 in epoch1; decorrelation is0 in epoch1,ramps to
0.0003 in epoch2,and stays there. Residualalpha0.2; up to16supports/rank.
Python3.8.20,torch1.13.0+cu117,torchvision0.14.0+cu117.

37server tests passed. Both startup audits had3actual updates in8steps.
The method completed485batches,480updates and387actual decorrelation updates.
Five initial AMP skips were globally synchronized; no later skips,nonfinite
forward losses,or residual-size alarms. All selected expert outputs are
nonzero; same-input head cos2 diagnostics are in method/result.json.
Training maximum recorded allocated memory6761197056bytes. Final-test GPU
memory in completion.json is only a5-second nvidia-smi sample maximum,not
the true allocator peak.

Original datasets/annotations/OEFormer are read-only. Deterministic train-ID
holdout:31289images/2792IDs for training,3383images/310IDs for validation.
Both rows of one cross-PID duplicated training image are quarantined; no
identity was guessed. Test content/order stays19848images,19848queries,
1000IDs. This uses fewer training identities than a full-original-train
protocol; disclose the holdout difference when presenting paper comparisons.
No test/validation identities enter training. No model/file/image/batch hashes.

## Artifacts

- Code: experiments/TBPS_ViewResearch/v017_icfg_joint,source96b8827.
- Selection: frozen_test_selection.json,committed before the first test.
- Exact test evidence: test/result.json,test/verification.json,and launch records.
- Epoch-level validation: e0/result.json and method/result.json.
- Server best: /root/autodl-tmp/TBPS_ViewResearch_v017_20260907/validation_run/checkpoints/best.pth.
- Server code: /root/autodl-tmp/TBPS_ViewResearch_v017_20260907.
- Full last/best remain protected on persistent server storage. Local E0
  archive transfer is running; method archive is ready. Do not claim local
  checkpoint completion until the corresponding verification receipts exist.

One seed only,no isolated causal ablation,no manual orientation calibration.
The supported conclusion is numerical mAP improvement under this fixed
complete mechanism and protocol,not robustness or statistical significance.
