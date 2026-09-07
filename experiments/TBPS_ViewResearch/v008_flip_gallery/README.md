# V004: give residual experts access to visual patch evidence

Independent full code snapshot; same seed-1 CUHK E0 five-epoch checkpoint.
Five additional epochs, validation-only selection. No original parameter is
frozen. This changes the expert input from V003, not the loss, data, optimizer
settings or schedule. V003 gave a small best-validation improvement, but its
best checkpoint was before decorrelation started. The next hypothesis is that
global CLS alone discards detail that a residual MLP cannot reconstruct.

| Group | V003 peak LR | V004 peak LR |
| --- | ---: | ---: |
| Original image/text backbone and logit scale | 1e-6 | 1e-6 |
| Residual experts (including patch-pooling queries) | 1e-3 | 1e-3 |

Shared h is still the projected CLS token. Let X be the other projected visual
tokens `[batch,196,512]`, explicitly excluding CLS. Each view expert has one
learnable zero-initialized FP32 query q_g `[512]`. For each image:

```
a_g = softmax(LayerNorm_no_affine(X, eps=1e-5) @ q_g)  # over patches
z_g = sum_n a_g[n] * X[n]
r_g = Head_g(h + z_g)
```

Queries start with uniform attention, not a fixed human-part box. Three
queries add 1536 trainable parameters. Zero final layers preserve the exact
initial global embedding. Pooling is used identically in main, no-grad target,
gallery and same-input expert diagnostics. It is not text-conditioned.

## Complete method contract

OEFormer supplies 72 probabilities for bin centers 0,5,...,355 degrees. Sum into
F [150,210], B [330,360) union [0,30], and S the remaining angles. Confidence
`c=clamp(1-entropy(p)/log(3),0,1)` is uncalibrated; human annotation was waived.
Horizontal flip updates theta to `(360-theta)%360`; F/S/B probabilities are
unchanged because both sides are grouped into S.

CLIP produces unnormalized shared image h and normalized text t. Three FP32
heads each use LN(512), Linear(512,128), GELU, Linear(128,512). Final linear
weights and biases start at zero. Route `r=c*sum_g p_g*Head_g(h+z_g)` and retrieve
with `v=normalize(h+.2*r)` and score `dot(t,v)`. Fusion is FP32, cast to h.dtype
before normalization to preserve initial numeric equivalence. The no-grad
image target uses identical routing. Text never receives a query source angle.

Keep the original N-ITC + R-ITC and expanded-caption-pair main batches of 4x80.
Use unchanged BT, deletion, image transforms and deterministic per-sample RNG.
Each rank adds up to 16 auxiliary train-only supports: same PID, different
image_id, different peak-angle F/S/B view, circular distance >=60 degrees both
before and after augmentation. Supports do not change main negative batches.

For selected support pairs, detached `w=c_a*c_b*(1-dot(p_a,p_b))` weights:

- `L_shared=weighted_mean(1-cos(h_a,h_b))`, updating the image tower.
- `L_dec=weighted_mean(cos(r(stopgrad(h_a),p_a,stopgrad(X_a)),
  r(stopgrad(h_b),p_b,stopgrad(X_b)))^2)`, updating only experts and pooling
  queries. Both shared and patch encoder outputs are detached. No same-view,
  same-image or different-ID pairs.
- Global weighted means use correct DDP averaging; empty pairs have graph-connected zero.
- `L=N-ITC+R-ITC+lambda_shared*L_shared+lambda_dec*L_dec`.

Shared weight ramps 0 to .4 during epoch 1. Decorrelation is zero in epoch 1,
ramps 0 to .0003 in epoch 2, and then stays fixed. AdamW and weight-decay rules
are unchanged; both LR groups warm up for one epoch, then cosine decay. All
original parameters remain trainable before DDP and optimizer construction.
Reset training Torch/CUDA RNG after model construction. AMP overflow decisions
are synchronized, and best/last checkpoints use the existing atomic transaction.

## Running

Environment: Python 3.8 / torch 1.13.0+cu117 / torchvision 0.14.0+cu117.

```bash
python -m unittest discover -s tests_research -v
python launch_research.py --stage audit --config configs/research_v004.json \
  --record-dir /absolute/audit_launch --output-dir /absolute/audit
python launch_research.py --stage train --config configs/research_v004.json \
  --record-dir /absolute/train_launch --output-dir /absolute/run
python run_research.py verify-result --result /absolute/run/result.json
```

Highest validation R1 chooses the checkpoint (earliest tie); report mAP from
the same checkpoint. No test evaluation during training/search. A separate
frozen selection is required for final test. CUHK RDE target is R1 75.94 and
mAP 67.56, BOTH exceeded at two-decimal precision. Do not start old multi-seed
queues. Do not overwrite or delete previous version records.

The run records actual non-skipped decorrelation updates per epoch. Final test
refuses a selected checkpoint with no such updates. This reporting/eligibility
guard does not modify the validation selection rule or the loss schedule.
