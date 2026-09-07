# V012: joint soft-view learning from OpenAI initialization

## Hypothesis and comparison

V009/V010/V011 all peak in continuation epoch1, before decorrelation begins.
V012 tests whether adding view structure only after E0 has already converged
is the limitation. Train the complete V009 mechanism from OpenAI ViT-B/16 for
five epochs, using the original S initial-training backbone LR profile.
This is NOT continuation, not a new loss, and not a ten-epoch run.

Frozen comparator is V009 portrait E0 (five epochs from the same OpenAI file):
validation R1=72.84832763671875, mAP=65.62239837646484. Its selected weights are
checked as reference evidence but NEVER loaded into this training model.
`initializer` is the original OpenAI file; `baseline_checkpoint` is explicitly
separate. The loader rejects substituting E0 best for the initializer.

Compared with V009 method, the focused change is its initial-training profile:
OpenAI rather than E0 weights, base peak LR1e-4 rather than1e-6. New heads and
queries retain peak LR1e-3. All mechanisms, weights, inputs and five-epoch
schedule shapes stay the same. Relative to E0, this method includes experts,
auxiliary supports and conv1 unfreezing; it is not an isolated conv1 ablation.
V010 retained beta and V011 discriminative shared loss are NOT included.

## Inputs and model

CUHK official train/val/test image/PID/caption counts:
34054/11003/68126;3078/1000/6158;3074/1000/6156. Reuse the existing checked
catalog and seed1 E0 caption-pair plans. Global batch320=4x80,212steps/epoch.
The training set expands caption pairs. Auxiliary images come only from train
and never enter the main negatives. No original dataset file is written.

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

Keep S N-ITC+R-ITC without modification. Up to16 auxiliary supports per rank:
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
python launch_research.py --stage audit --config configs/research_v012.json \
  --record-dir /root/autodl-tmp/TBPS_ViewResearch_v012_20260907/audit_launch \
  --output-dir /dev/shm/TBPS_ViewResearch_v012_20260907/audit_run
# Require >=1 actual optimizer update across8 full-size four-GPU audit steps.
python launch_research.py --stage train --config configs/research_v012.json \
  --record-dir /root/autodl-tmp/TBPS_ViewResearch_v012_20260907/validation_launch \
  --output-dir /dev/shm/TBPS_ViewResearch_v012_20260907/validation_run
python run_research.py verify-result --result /dev/shm/TBPS_ViewResearch_v012_20260907/validation_run/result.json
python persist_completed_run.py --phase validation_run \
  --output-dir /root/autodl-tmp/TBPS_ViewResearch_v012_20260907/validation_run
```

The longer8-step audit accounts for initial OpenAI AMP scale settling; it does
not alter the formal scaler. Audit must check actual OpenAI loading, all
encoder keys/shape compatibility, finite loss, updates and four-rank agreement.
Tests cover initial zero-head equivalence, gradient isolation, missing-key
rejection, initializer/reference separation and checkpoint recovery.

Select maximum validation R1 with earliest tie; mAP must come from that same
checkpoint. Log the paired minimum gain against the frozen portrait E0. No
test evaluation during training or trial selection. Final test requires a
separate frozen selection and a selected checkpoint with actual decorrelation
updates. Validation success alone never establishes the RDE test goal.

RAM outputs are volatile. After completion persist logs/configs and compact
best, then archive full last plus best locally and verify restoration. Last
holds optimizer and all four rank RNG states; best is selection-only. No
automatic old-run deletion or lowered disk guard. Publish completed source
and compact results only, never weights, images, credentials or smoke logs.
