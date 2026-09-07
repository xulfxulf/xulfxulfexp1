# V002: balance the two auxiliary gradient strengths

Full independent source snapshot. The sole method change from V001 is joint
rescaling of the existing two auxiliary terms. No backbone, head, sampler,
input, initialization, retrieval formula, loss formula or evaluation change.
This is NOT an isolated ablation attributing an effect to one of the two weights.

## Evidence and hypothesis

V001 best validation epoch 1: R1 72.2637 / mAP 65.1536, below E0.
A train-only, four-rank, three-batch gradient audit of that checkpoint found:

- Weighted shared gradient / retrieval gradient: 0.54%-0.64% in the last visual
  block and 0.24%-0.27% in the visual projection.
- Weighted decorrelation gradient / retrieval gradient in residual heads:
  2.62-3.57 times. The original .01 penalty can dominate small residual heads
  when it switches on, although its scalar loss looks small.
- Shared/retrieval gradient cosines were mildly negative, not aligned. Strong
  unrestricted shared alignment is therefore not justified by this audit.

Fix maximum `shared_weight` to **0.4** (from .02) and maximum
`decorrelation_weight` to **0.0003** (from .01). At the audited point these
correspond to approximately 5%-13% shared/backbone gradient and 8%-11%
decorrelation/head gradient. These are starting-point estimates, not guaranteed
ratios throughout training. No adaptive reweighting is implemented.

## Model and training contract

F [150,210], B [330,360) union [0,30], S remaining. Sum all 72 OEFormer angle-bin
probabilities into these three groups. Human calibration is waived; entropy
confidence is explicitly uncalibrated. Horizontal flip leaves F/S/B probabilities
unchanged and updates angles to `(360-theta)%360`.

For global shared feature h, three FP32 LN-Linear(512,128)-GELU-Linear(128,512)
heads with zero final layers produce
`r = confidence(p) * sum_g p_g Head_g(h)` and `v = normalize(h + .2*r)`.
Fusion is FP32 then cast to h.dtype before normalization, as in V001.
Text remains the original normalized CLIP feature. Main and no-grad target
images use identical routing. All original parameters are trainable.

Use the same E0 five-epoch checkpoint, seed 1, five additional epochs,
caption-pair batches of 4x80, original BT/deletion/augmentations, fixed per-sample
RNG and unchanged optimizer/LR schedule. Up to 16 auxiliary supports per rank
are train-only, same PID, different image_id, different peak-angle F/S/B group
and >=60 degrees apart before and after augmentation. No added main negatives.

Reliability is `w = c_a*c_b*(1-dot(p_a,p_b))`. Shared loss is the globally
weighted mean of `1-cos(h_a,h_b)`; residual decorrelation is the globally
weighted mean of `cos(r(stopgrad(h_a),p_a),r(stopgrad(h_b),p_b))^2`.
The latter cannot update the image backbone. Empty sets return graph-connected
zero; global means account for DDP averaging. Main N-ITC/R-ITC stay unchanged.

Shared weight ramps 0 to .4 during epoch 1. Decorrelation stays off epoch 1,
ramps 0 to .0003 during epoch 2, then stays fixed. Backbone peak LR 1e-5,
experts 1e-4, AdamW, warmup then cosine, synchronized AMP overflow handling.

## Commands and selection

Pinned Python 3.8 / torch 1.13.0+cu117 / torchvision 0.14.0+cu117, four GPUs.

```bash
python -m unittest discover -s tests_research -v
python launch_research.py --stage audit --config configs/research_v002.json \
  --record-dir /absolute/audit_launch --output-dir /absolute/audit
python launch_research.py --stage train --config configs/research_v002.json \
  --record-dir /absolute/train_launch --output-dir /absolute/run
python run_research.py verify-result --result /absolute/run/result.json
```

Training is validation-only. Select highest validation R1, earliest tie, and
read mAP at that checkpoint. Freeze selection before any separate final test.
No query source angle enters text encoding or scoring. The test target remains
RDE CUHK R1 75.94 and mAP 67.56, both strictly exceeded by one checkpoint.
Never start the inherited legacy multi-seed queue. Keep old records unchanged.
