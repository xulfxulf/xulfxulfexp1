# V011: discriminative cross-view shared consistency

Independent full source copied from V009, not the rejected V010 beta trial.
Only method change: replace the positive-only shared cosine pull with a
symmetric, temperature-scaled cross-view contrastive objective. Main TBPS
N-ITC+R-ITC, soft-view patch experts, and residual decorrelation stay unchanged.
This is a new shared-consistency design, not a claimed full migration of an
external method. No new classifier, text-view labels, or test training.

## Exact new shared objective

Each eligible train-only pair (a,b) is the same PID, different images and peak
F/S/B views, with circular angle distance>=60 before and after augmentation.
Use normalized shared CLS features A=normalize(h_a), B=normalize(h_b).
The differentiably gathered shared features of the existing global main batch
form bank N. For a given pair, remove EVERY bank entry with its PID. Do not
treat another caption/image of that PID as a negative. No cached/test features
or additional negative images are introduced.

For fixed tau=0.07, and `p=dot(A,B)`:

```
L_A = tau * log(1 + sum_{j:pid_j!=pid_a} exp((dot(A,N_j)-p)/tau))
L_B = tau * log(1 + sum_{j:pid_j!=pid_a} exp((dot(B,N_j)-p)/tau))
L_pair = (L_A + L_B)/2
w = stop_gradient(c_a*c_b*(1-dot(view_prob_a,view_prob_b)))
L_shared = sum_global(w*L_pair) / sum_global(w)
```

Use stable logsumexp over one positive logit and masked negative logits. The
tau factor preserves a bounded similarity-gradient scale rather than silently
introducing a1/tau multiplier. All positive features and negative-bank shared
features receive gradients, only through the image encoder, never experts.
Bank features are gathered with upstream differentiable AllGather. Per-rank
loss is world_size*local_weighted_sum/global_weight_mass before DDP averaging.
No support pairs or no different-PID negatives gives graph-connected zero.
Unit tests cover explicit formula, masks, zero pairs and gradient separation;
two-rank tests compare actual reduced gradients to one global reference.

## Unchanged mechanism and settings

Three front/side/back experts: FP32 LN512-Linear128-GELU-Linear512 with zero last
layers. Learned zero queries pool projected visual patches excluding CLS;
each head sees h+its pooled evidence. OEFormer72 bins aggregate to F[150,210],
B[330,360) union[0,30], S remainder. Uncalibrated confidence is
`c=clamp(1-H(p3)/log3,0,1)`; human calibration is waived. All routes use
`r=c*sum_g p3_g*Head_g(h+pool_g(X))`, `v=normalize(h+0.2*r)` and one normalized
text embedding/cosine score. Fusion FP32 then cast to h.dtype before normalize.
Main, no-grad target and gallery share this path; query-source view is unused.

Residual decorrelation remains weighted cos(r_a,r_b)^2 with both encoder CLS
and patches detached, so only heads/queries update. Confidence pair weights,
same-input expert diagnostics, no-grad branch, and support RNG isolation stay.

- Same V009 portrait E0 best epoch5, validation72.84832764/65.62239838.
- Seed1, four GPUs x80, five additional epochs,212steps/epoch; exact existing
  caption-pair plans, up to16 supports/rank, no change to TBPS negatives.
- Geometry384x128 and all V009 image/text augmentations unchanged.
- Shared weight ramps0-to0.4 epoch1. Decorrelation0 epoch1, ramps0-to0.0003
  epoch2, fixed epochs3-5. Beta retains V009's epoch1 ramp0-to0.5, then0.5.
- All original parameters unfrozen before DDP/AdamW. Peak baseLR1e-6,
  expert/queryLR1e-3. Original warm-up/cosine schedule and AdamW unchanged.
- Full 5+5 protocol, no new E0 run, no additional training seed.

## Execution and evidence policy

Pinned Python3.8.20/torch1.13.0+cu117/torchvision0.14.0+cu117. Use new absolute
output paths. Do not run the inherited legacy queue or restart an old version.

```bash
python -m unittest discover -s tests_research -v
python -m torch.distributed.run --standalone --nproc_per_node=2 \
  tests_research/distributed_shared_checks.py
python launch_research.py --stage audit --config configs/research_v011.json \
  --record-dir /root/autodl-tmp/TBPS_ViewResearch_v011_20260907/launch_audit \
  --output-dir /dev/shm/TBPS_ViewResearch_v011_20260907/audit_run
python launch_research.py --stage train --config configs/research_v011.json \
  --record-dir /root/autodl-tmp/TBPS_ViewResearch_v011_20260907/launch_train \
  --output-dir /dev/shm/TBPS_ViewResearch_v011_20260907/validation_run
python run_research.py verify-result \
  --result /dev/shm/TBPS_ViewResearch_v011_20260907/validation_run/result.json
python persist_completed_run.py --phase validation_run \
  --output-dir /root/autodl-tmp/TBPS_ViewResearch_v011_20260907/validation_run
```

The four-step audit must perform real updates and have synchronized parameters.
Train/val/test identity separation and source-path checks remain mandatory.
Use validation only for search; select maxR1, earliest tie, with mAP from the
same checkpoint. No full-mechanism claim for a selected pre-decorrelation epoch.
Final test remains separate and frozen before use. A positive validation delta
is not evidence of exceeding RDE's test result.

Keep RAM output only during execution; persist best/logs and immediately back
up full best+last to `D:/004SSH/TBPS_ViewResearch_v011_archives_20260907/` after
completion. Do not shut down before local load verification. User authorized
cleanup only after archival safety checks; do not erase a running checkpoint.
Publish main source/compact results, not raw smoke/failure logs or checkpoints.
