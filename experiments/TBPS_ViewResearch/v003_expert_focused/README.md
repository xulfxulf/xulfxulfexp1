# V003: focus adaptation on the new residual experts

Independent full code snapshot; same seed-1 CUHK E0 five-epoch checkpoint.
Five additional epochs, validation-only selection. No original parameter is
frozen. This changes ONLY optimizer learning-rate allocation from V002:

| Group | V002 peak LR | V003 peak LR |
| --- | ---: | ---: |
| Original image/text backbone and logit scale | 1e-5 | 1e-6 |
| Three new residual experts | 1e-4 | 1e-3 |

V001 and V002 both peaked on validation at additional epoch 1 and then fell.
V002 best validation was R1 72.3287 / mAP 65.4154, below E0 72.6372 / 65.5077.
Its final-epoch median `alpha*norm(r)/norm(h)` was just 0.002506. These observations
motivate less backbone drift and faster new-head fitting. They do not prove
either cause separately; both learning rates change in this optimizer-allocation
experiment. All other settings and equations remain byte-for-byte unchanged.

## Complete method contract

OEFormer supplies 72 probabilities for bin centers 0,5,...,355 degrees. Sum into
F [150,210], B [330,360) union [0,30], and S the remaining angles. Confidence
`c=clamp(1-entropy(p)/log(3),0,1)` is uncalibrated; human annotation was waived.
Horizontal flip updates theta to `(360-theta)%360`; F/S/B probabilities are
unchanged because both sides are grouped into S.

CLIP produces unnormalized shared image h and normalized text t. Three FP32
heads each use LN(512), Linear(512,128), GELU, Linear(128,512). Final linear
weights and biases start at zero. Route `r=c*sum_g p_g*Head_g(h)` and retrieve
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
- `L_dec=weighted_mean(cos(r(stopgrad(h_a),p_a),r(stopgrad(h_b),p_b))^2)`,
  updating only experts. No same-view, same-image or different-ID pairs.
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
python launch_research.py --stage audit --config configs/research_v003.json \
  --record-dir /absolute/audit_launch --output-dir /absolute/audit
python launch_research.py --stage train --config configs/research_v003.json \
  --record-dir /absolute/train_launch --output-dir /absolute/run
python run_research.py verify-result --result /absolute/run/result.json
```

Highest validation R1 chooses the checkpoint (earliest tie); report mAP from
the same checkpoint. No test evaluation during training/search. A separate
frozen selection is required for final test. CUHK RDE target is R1 75.94 and
mAP 67.56, BOTH exceeded at two-decimal precision. Do not start old multi-seed
queues. Do not overwrite or delete previous version records.
