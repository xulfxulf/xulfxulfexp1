# V017 ICFG retained five-epoch validation result

Source96b8827; complete V015 mechanism transferred to ICFG. All485batches,
480actual updates,5initial globally synchronized AMP skips,387actual
decorrelation updates,37server tests and eight-step four-GPU audit3updates.
Runtime426.827seconds. Selection is epoch5=max validationR1,earliest tie.

| Validation epoch | R1 | mAP |
|---|---:|---:|
|1|58.291454|45.111977|
|2|67.898315|53.241600|
|3|71.386337|57.489933|
|4|74.963051|61.487228|
|5,selected|76.559265|63.369148|

Against measured E0 epoch5 79.958611/60.705067: R1-3.399345pp,
mAP+2.664082pp. Retain for the user-amended mAP priority; this is NOT
a joint-metric gain. E0 and method share the fixed train-ID holdout.
Method initializes independently from OpenAI and uses PK2/unfreezing/
R-ITC epsilon0.001 as well as complete shared/soft-expert/decorrelation
mechanism. No isolated expert contribution is claimed from this comparison.

Per-batch auxiliary support count58-64; final-epoch zero-residual-pair ratio0.
No residual-size alarm. Last-epoch mean of batch median residual ratios
0.005423; maximum batchp95=0.017885. Same-input pairwise expert cos2 at
epoch5: F/S0.022848,F/B0.041108,S/B0.012797; all256diagnostic images valid.
Maximum recorded allocated GPU memory6761197056bytes. These diagnostics
show nonzero,distinct expert outputs,not proof of isolated causal benefit.

The code,weights,selection and evaluation settings are frozen before the
first official ICFG test. No new research test was opened while choosing
this candidate. The working target is testmAP>=41.06 (RDE40.06+1pp),with
R1 fully reported. Validation63.37 is not directly comparable to test40.06.
See frozen_test_selection.json for the prospective selection and scope.
