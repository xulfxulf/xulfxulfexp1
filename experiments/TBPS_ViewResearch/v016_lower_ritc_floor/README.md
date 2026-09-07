# V016: one further reduction of the R-ITC target floor

## Hypothesis and comparison

Only training change from retained V015: R-ITC target floor0.001->0.0001.
Keep V014/V015's existing view-aware P160 K2 plans. Train from the same OpenAI
ViT-B/16 file for five epochs,seed1,with the same learning rates,augmentations,
auxiliary supports and architecture. This is NOT a continuation of V015.
V013's ten-epoch change is NOT included. No additional E0 is trained.

The previous0.01->0.001 change improved selected validation R1 by0.30854pp
and mAP by0.35959pp,with1060identical sample-trace rows per rank,5initial AMP
skips only,847actual decorrelation updates,and no residual-size alarms.
This motivates one further factor10 reduction,not a guarantee of improvement.
The earlier train-only gradient diagnosis tested0.001,not this new0.0001.

Frozen comparator is V009 portrait E0 (five epochs from the same OpenAI file):
validation R1=72.84832763671875, mAP=65.62239837646484. Its selected weights are
checked as reference evidence but NEVER loaded into this training model.
`initializer` is the original OpenAI file; `baseline_checkpoint` is explicitly
separate. The loader rejects substituting E0 best for the initializer.

The direct retained method comparator is V015,whose selected validation
epoch5 has R1=70.29879760742188,mAP=67.79859924316406. V012 remains a separate
stronger-R1 option:73.33549499511719/65.94811248779297. The user permits focus
on one metric and prefers five epochs. Search metric is mAP at the maximum
validation-R1 checkpoint,not a retrospectively selected maximum-mAP epoch.
E0 comparison remains secondary; E0 freezes conv1 while both methods do not.
V010 retained beta and V011 discriminative shared loss are NOT included.

## Inputs and model

CUHK official train/val/test image/PID/caption counts:
34054/11003/68126;3078/1000/6158;3074/1000/6156. Reuse the existing checked
catalog and seed1 PK plans. Global batch320=4x80,212steps/epoch. Each global
batch has160 different PIDs,each with two images,allocated as40 whole PID
pairs per rank. Existing sampler prioritizes different raw four-views with
angle gap>=60,then balances raw four-view counts. Caption is randomly drawn
per image. If an identity has only one image,it is repeated; this is counted,
not misreported as cross-image support. Validation rejects repeated images
for identities that do have multiple images and rejects split-rank PID pairs.

Five-epoch plan diagnosis: random main batches have0.023703 cross-image
same-ID candidates per anchor,PK has0.999735. PK still covers all11003 train
IDs but32637 of34054 images and63513 of68126 caption pairs. This reduction
in image/caption coverage is a documented tradeoff,not a proven gain. F/S/B
auxiliary eligibility below remains separate from four-view main sampling.
Auxiliary images come only from train and never enter the main negatives.
No original dataset file or prepared plan is modified.

All image paths use384x128,24x8 patches plus CLS. Upstream bilinear positional
embedding resize maps OpenAI14x14 to24x8 and preserves CLS. Keep the original
two-draw augmentation pool, ImageNet normalization, portrait crop-ratio
(.25,4/9), scale(.9,1), flips/rotation/color/erasing, backtranslation0.1,
random deletion0.05 and text dropout0.05, exactly as V009.

Shared h is projected image CLS; X is projected patch tokens excluding CLS.
OEFormer72 probabilities sum into F=[150,210],B=[330,360) union[0,30],S=rest.
Confidence c=clamp(1-H(p3)/log(3),0,1), explicitly uncalibrated. Human
calibration was waived. Horizontal flip preserves these merged F/S/B masses.

Each view has learned query q_g (initially zero), softmax attention over
LayerNorm_no_affine(X), and a FP32 head LN512-Linear128-GELU-Linear512:

```
a_g = softmax(LayerNorm_no_affine(X) @ q_g)
z_g = sum_n a_g[n] * X[n]
r_g = Head_g(h + z_g)
r = c * sum_g p3[g] * r_g
v = normalize(h + 0.2*r)
t = normalize(TextEncoder(caption))
score = dot(t, v)
```

Last head layers are zero initialized. Initial retrieval equals the original
OpenAI-initialized S model, NOT the fine-tuned E0 model. Fusion uses FP32 then
casts to h.dtype before normalization. Gradient main, fresh no-grad target,
validation and final-test gallery all use the same image route. Text uses no
paired source image, angle, or view label.

## Losses and updates

Keep S N-ITC+R-ITC formulas,with only R-ITC's existing epsilon parameter
changed to0.0001. For normalized same-PID q, R-ITC uses log(q+0.0001),without
renormalizing q+epsilon. This is not an additional loss or an RDE reimplementation.
The research model builder sets and asserts epsilon in both train and final
evaluation construction; legacy E0's official config stays0.01. Up to16 auxiliary supports per rank:
same PID, distinct image, different peak F/S/B, circular angle gap>=60 both
before and after augmentation. Support RNG is isolated from the main path.
With detached w=c_a*c_b*(1-dot(p_a,p_b)):

- Shared: globally weighted mean of1-cos(h_a,h_b), encoder gradients only.
- Decorrelation: globally weighted mean ofcos(r_a,r_b)^2; both encoder CLS
  and patch tensors are detached. Only heads/queries get these gradients.
- L=N-ITC+R-ITC+lambda_shared*shared+lambda_dec*decorrelation.

Global weighted means use world-scaled local numerator/global mass, with
graph-connected zero for empty supports. Shared weight ramps0->0.4 in epoch1;
decorrelation0 in epoch1,0->0.0003 in epoch2,0.0003 in epochs3-5. Beta0->0.5
in epoch1, then0.5. All original parameters unfreeze before DDP/optimizer;
freeze_conv1=False remains effective after model.train().

AdamW betas(.9,.98),eps1e-8,wd0.02 except bias/norm0. Base peak1e-4, new heads
and queries peak1e-3. Epoch1 linearly warms from1% to peak, remaining epochs
cosine-decay toward5% of peak. Seed1 only; reset Torch/CUDA RNG after model
construction/loading/DDP. Four-card AMP overflow decisions are synchronized.
Visual activation checkpointing, train-only diagnostics, raw/aug view counts,
residual ratios, same-input expert comparisons and transactional best/last
saving remain unchanged.

## Execution and acceptance

Pinned server: Python3.8.20,torch1.13.0+cu117,torchvision0.14.0+cu117,4xRTX4080.
Do NOT use the legacy run_view4 queue or train another E0. Every output and
launcher directory must be new and absolute. The launcher prevents duplicates.

```bash
python -m unittest discover -s tests_research -v
python launch_research.py --stage audit --config configs/research_v016.json \
  --record-dir /root/autodl-tmp/TBPS_ViewResearch_v016_20260907/audit_launch \
  --output-dir /dev/shm/TBPS_ViewResearch_v016_20260907/audit_run
# Require >=1 actual optimizer update across8 full-size four-GPU audit steps.
python launch_research.py --stage train --config configs/research_v016.json \
  --record-dir /root/autodl-tmp/TBPS_ViewResearch_v016_20260907/validation_launch \
  --output-dir /dev/shm/TBPS_ViewResearch_v016_20260907/validation_run
python run_research.py verify-result --result /dev/shm/TBPS_ViewResearch_v016_20260907/validation_run/result.json
python persist_completed_run.py --phase validation_run \
  --output-dir /root/autodl-tmp/TBPS_ViewResearch_v016_20260907/validation_run
```

The longer8-step audit accounts for initial OpenAI AMP scale settling; it does
not alter the formal scaler. Audit must check actual OpenAI loading, all
encoder keys/shape compatibility, finite loss, updates and four-rank agreement.
Tests cover initial zero-head equivalence, gradient isolation, missing-key
rejection, initializer/reference separation and checkpoint recovery.

Select maximum validation R1 with earliest tie; mAP must come from that same
checkpoint. Primary metric is that mAP; paired minimum gain against the
frozen portrait E0 is retained only as secondary provenance. No
test evaluation during training or trial selection. Final test requires a
separate frozen selection and a selected checkpoint with actual decorrelation
updates. Prospective single-metric working target is CUHK test mAP>=68.56
(RDE67.56+1.00),with R1 always reported. Validation success alone never
establishes this test goal. The current trial does not automatically open test.

RAM outputs are volatile. After completion persist logs/configs and compact
best, then archive full last plus best locally and verify restoration. Last
holds optimizer and all four rank RNG states; best is selection-only. No
automatic old-run deletion or lowered disk guard. Publish completed source
and compact results only, never weights, images, credentials or smoke logs.
