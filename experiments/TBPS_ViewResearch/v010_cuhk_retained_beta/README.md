# V010: retain converged S soft-label mixing during continuation

Independent full snapshot of V009. Only training-method change: retain beta=0.5
from the first additional step instead of restarting its epoch-one 0-to-0.5
ramp. N-ITC and R-ITC formulas are unchanged. This tests the matched train-only
gradient diagnosis, not a proven explanation for previous validation losses.

Use the SAME V009 portrait E0 best (epoch5) as V009, not the trained V009 method:
`/root/autodl-tmp/TBPS_ViewResearch_v009_20260907/e0_validation_run/checkpoints/best.pth`.
Measured validation baseline: R1 72.84832763671875, mAP 65.62239837646484.
No E0 retraining, no resume of V009 optimizer, and no changed data or RNG plan.
Inherited baseline scripts remain available but are not part of this run.

## Fixed training and model

- CUHK-PEDES official train/val/test split, seed1, four GPUs x80, 212 steps/epoch,
  five additional epochs. Main samples are the E0 expanded caption-pair plan.
- Main/support/gallery geometry 384x128; preserve all V009 transforms, text
  augmentation and model dropout. Initial RNG is reset after model/DDP creation.
- Shared projected CLS h; three FP32 residual heads with zero final layers,
  each LN512-Linear128-GELU-Linear512. Learned zero-initialized patch queries
  produce view-specific pooled patch inputs, added to h before each head.
- OEFormer 72 bins aggregate into F[150,210], B[330,360) union[0,30], S remainder.
  `c=clamp(1-H(p3)/log3,0,1)` remains uncalibrated, as authorized by the user.
  `r=c*sum_g p3[g]*Head_g(h+pool_g(patches))`; `v=normalize(h+0.2*r)`.
  Fusion is FP32 then cast to h.dtype before normalization, unchanged from V009.
- Unified normalized text; cosine retrieval. Same residual path for gradient
  image, no-grad target, and gallery. Query-source angle is never used.
- At most16 train-only supports/rank: same PID, different image and peak3-view,
  circular gap>=60 before/after augmentation. Supports do not change negatives.
- Detached `w=c_a*c_b*(1-dot(p_a,p_b))`. Shared consistency is weighted global
  mean `1-cos(h_a,h_b)`, encoder gradients only. Decorrelation is weighted mean
  `cos(r_a,r_b)^2` with CLS/patches detached, gradients to heads/queries only.
- Shared weight ramps to0.4 in epoch1. Decorrelation is0 in epoch1, ramps to
  0.0003 in epoch2, then fixed. V010 changes neither auxiliary schedule.
- All original parameters unfreeze before DDP/AdamW. Base peak LR1e-6; expert
  and query peak LR1e-3; original one-epoch LR warm-up/cosine schedule retained.
- The only new fixed config value is `continuation_beta=0.5`, validated on load
  and recorded in config plus every training telemetry row.

## Execution

Pinned environment: Python3.8.20, torch1.13.0+cu117, torchvision0.14.0+cu117.
All output paths must be new; never run the inherited legacy queue.

```bash
python -m unittest discover -s tests_research -v
python launch_research.py --stage audit --config configs/research_v010.json \
  --record-dir /root/autodl-tmp/TBPS_ViewResearch_v010_20260907/launch_audit \
  --output-dir /dev/shm/TBPS_ViewResearch_v010_20260907/audit_run
python launch_research.py --stage train --config configs/research_v010.json \
  --record-dir /root/autodl-tmp/TBPS_ViewResearch_v010_20260907/launch_train \
  --output-dir /dev/shm/TBPS_ViewResearch_v010_20260907/validation_run
python run_research.py verify-result \
  --result /dev/shm/TBPS_ViewResearch_v010_20260907/validation_run/result.json
python persist_completed_run.py --phase validation_run \
  --output-dir /root/autodl-tmp/TBPS_ViewResearch_v010_20260907/validation_run
```

Run the four-step real-update audit before formal training. Do not accept an
audit containing only AMP skips. Monitor synchronized updates and all five
validation epochs. Test is not opened during search. Best=max validation R1,
earliest tie, reporting mAP from that same checkpoint. A selected checkpoint
without actual decorrelation updates is not a full-mechanism final candidate.

Storage policy is identical to V009: existing RAM transaction safeguards remain,
old checkpoints are not touched. After completion immediately persist compact
best/logs and download full resumable last plus best to
`D:/004SSH/TBPS_ViewResearch_v010_archives_20260907/validation_run/`.
Verify archive lengths, checkpoint contents, AdamW and four-rank RNG before
claiming backup. No shutdown before backup; no automatic old-run cleanup.
Code and compact results go to GitHub, never weights or raw smoke/failure logs.
