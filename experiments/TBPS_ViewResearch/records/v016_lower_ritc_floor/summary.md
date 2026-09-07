# V016 completed: reject the further epsilon reduction

Source38694c7,parentV015 d24657a. Only training change:R-ITC epsilon0.001
to0.0001. Independent OpenAI initialization,five epochs,seed1,4GPUs,
unchanged V015 model/losses/PK plans/other hyperparameters. No test evaluated.

| Validation epoch | R1 | mAP |
|---|---:|---:|
|1|51.997398|49.234631|
|2|58.687885|55.531204|
|3|64.290352|61.491932|
|4|67.976616|65.168724|
|5,selected|70.071449|67.153687|

Selection is max validationR1,earliest tie. Selected paired mAP is0.644913pp
below V015,R1 is0.227348pp lower. Discard and retain V015 epsilon0.001.

33server tests passed;8-step four-GPU audit had3actual updates. Formal exit0,
runtime757.095seconds,1060batches,1054actual updates,847decorrelation updates.
Six synchronized AMP skips were in epoch1 at zero-based steps0,1,2,3,4,8;
none later. No forward nonfinite loss or residual-size alarms. The extra
early scale reduction is disclosed,not hidden as an execution failure.

Best/logs are persistent; full last and RAM tar are protected pending local
archive. Raw audit logs remain local-only. No official data files were edited.
ICFG structural diagnosis motivates a separate dataset-specific baseline next;
its scores must not be mixed into the CUHK validation ledger.
