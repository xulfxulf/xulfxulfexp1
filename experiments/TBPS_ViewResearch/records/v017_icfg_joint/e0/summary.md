# V017 ICFG five-epoch S baseline

Source96b8827; Python3.8.20/torch1.13.0+cu117,4xRTX4080,seed1,
global320,384x128,OpenAI initialization,caption-pair DistributedSampler.
All485batches completed,480actual updates,5initial globally synchronized
AMP skips at epoch1 steps0-4. No later skips. All37server tests passed;
the eight-step four-GPU startup audit performed3actual updates. Runtime
377.008seconds; maximum recorded allocated memory6614261760bytes.

| Validation epoch | R1 | mAP |
|---|---:|---:|
|1|65.710907|47.121532|
|2|73.012115|52.484535|
|3|77.298256|55.802925|
|4|79.308304|59.784866|
|5,selected|79.958611|60.705067|

Selection=max validationR1,earliest tie; paired mAP uses the SAME checkpoint.
These are NOT official test results. The fixed validation split reserves310
original training identities; test remains19848images/1000IDs,unopened.
Training is31289images/2792IDs after removing both rows of one conflicting
cross-PID image and reserving3383images/310IDs for validation. See split_manifest.
Original annotations/images/OEFormer remain unmodified. No content hashes.

This measured baseline initializes the separate ICFG search ledger. V017
method independently starts from OpenAI,not this E0 checkpoint. It changes
sampling/unfreezing/R-ITC floor as well as adding the complete mechanism;
the comparison is full-system,not an isolated expert-module ablation.

Compact best,full last and records are persisted; a full archive is being
transferred locally. No successful local archive is claimed until its
verification receipt exists. Raw startup-audit logs are local-only under
D:/004SSH/TBPS_ViewResearch_audit_logs_20260907/v017.
