# V013 ten-epoch method result: do not replace five-epoch V012

Immutable source e45c55e,OpenAI initialization,CUHK384x128,seed1,four GPUs,
global320. All2120 batches completed in about1518seconds,exit0. Five initial
synchronized AMP skips,none later. All27 server tests passed; eight-step
four-GPU audit made three actual updates. Actual decorrelation updates1907.

Selected maximum-validation-R1 checkpoint is epoch10:
R1=72.26371765136719,mAP=65.4229736328125.
Epoch9 has slightly higher mAP65.46144104003906 but is NOT the selected
checkpoint and is not substituted in comparisons.

Same ten-epoch E0 reference:72.00389862060547/64.98136901855469.
Within-horizon gains:+0.259819 R1,+0.441605 mAP. Thus the mechanism improves
this ten-epoch reference,but the absolute result is below retained V012
five-epoch73.33549499511719/65.94811248779297 by1.071777 R1 and0.525139 mAP.
Record a within-horizon keep in the ten-epoch ledger and a global discard;
do not confuse these decisions. The new mAP-focused ledger also discards it.

V012 remains retained. Return to five epochs as the user prefers. The next
focused hypothesis is same-ID image-pair sampling based on the saved train
plan diagnosis,not further increasing epochs. No test has been evaluated.

Compact best and records are persisted. Both E0 and method full archives now
pass local restoration checks,including AdamW moments and all4rank RNG states.
Method archive:D:/004SSH/TBPS_ViewResearch_v013_archives_20260907/validation_run;
321model keys,best600277159bytes,last1800722249bytes,next_resume_epoch11.
Each redundant persistent last was deleted only after its individual local
verification receipt passed. Best/code/logs/raw data and RAM source remain.
Raw audit logs are local-only. The manifest's inherited v009 implementation
text is historical metadata; experiment/config/source/history identify V013.
