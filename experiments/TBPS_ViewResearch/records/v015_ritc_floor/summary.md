# V015 completed five-epoch result

Source:d24657a; parent V014:4b9be6b. Only training change is R-ITC target
floor0.01->0.001. Independent OpenAI initialization,seed1,five epochs,4GPUs,
global320 P160K2,same augmentations/model/other losses. Test was not evaluated.

| Validation epoch | R1 | mAP |
|---|---:|---:|
|1|54.611885|51.156761|
|2|60.880154|57.401802|
|3|65.979210|62.652630|
|4|68.674889|65.881607|
|5,selected|70.298798|67.798599|

Selection=max validation R1,earliest tie. R1/mAP are from the same epoch5
checkpoint. Against V014: R1+0.308540,mAP+0.359589pp. Retain for the amended
mAP objective; V012 remains the stronger-R1 option. No claim of RDE test gain.

33server tests passed,8-step four-GPU audit had3actual updates,formal exit0.
Formal runtime762.133seconds;1060batches,5initial globally synchronized AMP
skips,847actual decorrelation updates,0residual-size alarm batches. Maximum
allocated CUDA memory6761197056bytes per reported rank. All1060sample-trace
rows match V014 for each of4ranks. These are structured row comparisons,
not image/tensor/file hashes.

The source and compact best/logs are persistent on PRO1. Full best/last archive
is locally verified,including all model/AdamW tensor finiteness and four-rank
RNG state. Only the redundant persistent server last was deleted afterward;
server best/source/RAM records remain. See the three archive/cleanup receipts.
No checkpoint or raw smoke log belongs in GitHub. Raw startup-audit logs are local under
D:/004SSH/TBPS_ViewResearch_audit_logs_20260907/v015.
