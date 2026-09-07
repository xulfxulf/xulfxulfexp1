# V013: ten-epoch horizon, with a matching portrait S control

## Focused change

V012 completed five initial-training epochs with the full mechanism and
selected epoch5 at validation73.335495/65.948112, above its five-epoch
portrait E0 by0.487167/0.325714 points. V013 changes only training horizon:
five -> ten epochs, including the corresponding cosine-decay duration.
No loss formula, routing, encoder, augmentation, sampler or loss weight is
changed. This is not a resume of V012 with a changed scheduler.

First measure a NEW ten-epoch portrait S E0 from OpenAI. Then train the method
independently from that same OpenAI file for ten epochs. The new E0 best is
comparison evidence, NOT the method initializer. Do not use V009's five-epoch
E0 as the ten-epoch reference and do not silently change an old run's config.
Only seed1, four GPUs, validation-based selection; no automatic final test.

## Fixed inputs and representation

CUHK official train/val/test image/PID/caption counts remain
34054/11003/68126;3078/1000/6158;3074/1000/6156. Reuse the existing checked
catalog and ten E0 caption-pair sampling plans in the explicit prepared_dir.
Expanded caption pairs, global320=4x80,212steps/epoch; all original images and
annotations remain read-only. Supports use train identities only.

Image size384x128, projected24x8 patches plus CLS. Upstream bilinear OpenAI
14x14 position resize keeps CLS unchanged. The exact V012 transforms remain:
original two-draw augmentation pool, ImageNet normalization, crop scale(.9,1),
portrait W/H ratio(.25,4/9), original flip/rotation/color/erasing, paired
backtranslation0.1, text deletion0.05 and text dropout0.05.

Shared h is projected image CLS; X is projected patches excluding CLS. Sum
OEFormer72 probabilities into F=[150,210],B=[330,360) union[0,30],S=rest.
c=clamp(1-H(p3)/log(3),0,1), explicitly uncalibrated as approved by the user.
Horizontal flip preserves merged F/S/B probabilities.

Each expert has learned zero-initial query q_g and FP32 head
LN512-Linear128-GELU-Linear512, with the final linear layer initialized zero:

```
a_g = softmax(LayerNorm_no_affine(X) @ q_g)
z_g = sum_n a_g[n] * X[n]
r_g = Head_g(h + z_g)
r = c * sum_g p3[g] * r_g
v = normalize(h + 0.2*r)
t = normalize(TextEncoder(caption))
score = dot(t,v)
```

Fusion remains FP32 then casts to h.dtype before normalization. Main, fresh
no-grad target, validation and gallery use the same route. Text never reads
source-image view or PID. Zero heads start at OpenAI S retrieval, not the
fine-tuned E0 reference. E0 has no experts or auxiliary losses.

## Fixed losses and optimization

Original S N-ITC+R-ITC remain unchanged. Method adds up to16 supports per rank:
same PID, distinct image, different peak F/S/B, circular angle gap>=60 before
and after augmentation. They do not enter main negatives. Support RNG is
isolated. With detached w=c_a*c_b*(1-dot(p_a,p_b)):

- Shared loss: weighted global mean1-cos(h_a,h_b), encoder only.
- Decorrelation: weighted global meancos(r_a,r_b)^2; detach encoder CLS AND
  patches, retaining head/query gradients.
- Total=N-ITC+R-ITC+lambda_shared*shared+lambda_dec*decorrelation.

Distributed weighted means and graph-connected zeros remain unchanged.
Shared weight ramps0->0.4 in epoch1. Decorrelation0 in epoch1, ramps0->0.0003
in epoch2, then0.0003 through epoch10. Beta ramps0->0.5 in epoch1 then stays.
Base peak LR1e-4; method heads/queries peak1e-3. Warmup is still ONE epoch,
from1% to peak. Cosine decay now occupies epochs2-10, ending near5% of peak.
AdamW betas(.9,.98),eps1e-8,wd0.02 except bias/norm0 are unchanged.

Method unfreezes all original parameters before DDP/optimizer, including
freeze_conv1=False. E0 preserves original S conv1 freezing. This is the same
method-vs-E0 distinction as V012, not an isolated conv1-controlled ablation.
Reset Torch/CUDA RNG after construction/loading/DDP; synchronized AMP skips,
visual activation checkpointing, train-only expert diagnostics and atomic
best/last with all-rank RNG remain unchanged.

## Run order and verification

Pinned server Python3.8.20,torch1.13.0+cu117,torchvision0.14.0+cu117,4xRTX4080.
Each output/record directory must be new and absolute. No legacy View4 queue.
Both audits use8 full-size four-GPU steps and require an actual optimizer
update. The initial five AMP skips are scaler settling, not eight successful
updates; the actual count is audited separately.

```bash
python -m unittest discover -s tests_research -v
python launch_research.py --stage e0-audit --config configs/cuhk_portrait_e0.yaml \
  --record-dir /root/autodl-tmp/TBPS_ViewResearch_v013_20260907/e0_audit_launch \
  --output-dir /dev/shm/TBPS_ViewResearch_v013_20260907/e0_audit_run
python launch_research.py --stage e0-train --config configs/cuhk_portrait_e0.yaml \
  --record-dir /root/autodl-tmp/TBPS_ViewResearch_v013_20260907/e0_validation_launch \
  --output-dir /dev/shm/TBPS_ViewResearch_v013_20260907/e0_validation_run
python run_cuhk_e0.py verify-result --result /dev/shm/TBPS_ViewResearch_v013_20260907/e0_validation_run/result.json
python persist_completed_run.py --phase e0_validation_run \
  --output-dir /root/autodl-tmp/TBPS_ViewResearch_v013_20260907/e0_validation_run
# Log/initialize the measured ten-epoch portrait baseline BEFORE method training.
python launch_research.py --stage audit --config configs/research_v013.json \
  --record-dir /root/autodl-tmp/TBPS_ViewResearch_v013_20260907/audit_launch \
  --output-dir /dev/shm/TBPS_ViewResearch_v013_20260907/audit_run
python launch_research.py --stage train --config configs/research_v013.json \
  --record-dir /root/autodl-tmp/TBPS_ViewResearch_v013_20260907/validation_launch \
  --output-dir /dev/shm/TBPS_ViewResearch_v013_20260907/validation_run
python run_research.py verify-result --result /dev/shm/TBPS_ViewResearch_v013_20260907/validation_run/result.json
python persist_completed_run.py --phase validation_run \
  --output-dir /root/autodl-tmp/TBPS_ViewResearch_v013_20260907/validation_run
```

Both complete stages require10 epoch records and2120 full batches. Tests
reject a five-epoch comparator and verify epoch10 LR bounds, loader origins,
gradient isolation and resume state at next_epoch11. Best is maximum
validation R1 with earliest tie; use mAP from that SAME checkpoint. Report
against both the new ten-epoch E0 and V012; do not call training duration a
new architectural benefit. E0 success alone does not satisfy the mechanism.

No test during search. A final frozen selection must have actual decorrelation
updates and be evaluated once on the untouched test split. RDE CUHK target
75.94/67.56 must both be strictly exceeded at published precision.

After EACH stage persist logs/configs and compact best, temporarily protect
full last if local transfer is pending, and download/verify a full local
archive. RAM is volatile on shutdown. Best is selection-only; last carries
AdamW/scaler and four-rank RNG. Never lower save-space guards or delete
another run automatically. Existing authorized cleanup runs independently,
after local full-checkpoint verification. Publish main code and compact
completed results; raw smoke logs remain local, weights/images never GitHub.
