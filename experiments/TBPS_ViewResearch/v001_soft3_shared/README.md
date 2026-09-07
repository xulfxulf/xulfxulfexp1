# V001: shared cross-view features and soft F/S/B residuals

Independent full source snapshot based on the verified five-epoch TBPS-CLIP-S
adapter. The vendored official implementation is unchanged. Only seed 1 is run.

## Hypothesis and boundaries

Continue CUHK E0's validation-selected five-epoch checkpoint. Retain its
expanded caption-pair main batches (4 x 80), rather than E1's identity sampler.
Add three view residuals plus real cross-image, same-identity auxiliary supports.
This is a mechanism experiment, not the earlier E1 continuation.

The user waived human annotation for this first stage. OEFormer probabilities
are used directly. They are **not target-domain calibrated confidence**.

## Executable definition

- OEFormer 72 bin centers are 0,5,...,355 degrees. Sum probabilities in F
  [150,210], B [330,360) union [0,30], and S the remaining angles.
- `c = clamp(1 - entropy(p_F,p_S,p_B)/log(3), 0, 1)` is an uncertainty proxy.
- Three FP32 heads: LayerNorm(512), Linear(512,128), GELU, Linear(128,512).
  Final linear weights/biases are zero. All original parameters are unfrozen
  before DDP/AdamW. Torch/CUDA training RNG is reset after model construction.
- `r(h,p) = c * sum_g p_g Head_g(h)`; `v = normalize(h + 0.2*r)`.
  Fusion uses FP32 then casts to the backbone feature dtype before normalizing,
  preserving initial numeric equivalence with the source adapter.
- Text is unchanged: `normalize(TextEncoder(text))`. Scoring is its dot product
  with v. No query's source image angle is accessed by text encoding/scoring.
- Both online and no-grad image target branches use the identical soft routing.
  The original N-ITC + R-ITC equations, BT/deletion, transforms and main samples
  are retained. Auxiliary support RNG does not advance main-path RNG.
- Per rank select at most 16 eligible anchors from the unchanged 80-item batch.
  For each, choose one train image of the same PID, a different image_id and a
  different peak-angle F/S/B view, with circular distance at least 60 degrees.
  Require these angle/view conditions both before and after augmentation.
  Supports are used only by the auxiliary losses, never as extra main negatives.
- Pair reliability `w = c_a*c_b*(1-dot(p_a,p_b))` is detached.
- `L_shared = weighted_mean(1-cos(h_a,h_b))`, updating the shared image tower.
- `L_dec = weighted_mean(cos(r(stopgrad(h_a),p_a),
  r(stopgrad(h_b),p_b))^2)`, updating only residual heads.
  Means use globally summed weights with correct four-rank DDP scaling.
  Empty pairs return graph-connected zero. No different-ID decorrelation.
- `L = N-ITC + R-ITC + lambda_shared*L_shared + lambda_dec*L_dec`.
  Shared weight ramps 0 to .02 over epoch 1. Decorrelation is off epoch 1,
  ramps 0 to .01 in epoch 2, then stays .01. No automatic loss changes.
- AdamW as the source adapter; peak backbone LR 1e-5, experts 1e-4. One-epoch
  warmup then cosine decay for five additional epochs. AMP overflow is synced.

## Execution

Use the existing pinned environment (Python 3.8, torch 1.13.0+cu117,
torchvision 0.14.0+cu117). Do not use the old `run_view4.py queue` entry point.

```bash
python -m unittest discover -s tests_research -v
torchrun --standalone --nproc_per_node=4 run_research.py audit \
  --config configs/research_v001.json --output-dir /absolute/audit --steps 4
torchrun --standalone --nproc_per_node=4 run_research.py train \
  --config configs/research_v001.json --output-dir /absolute/run
python run_research.py verify-result --result /absolute/run/result.json
```

For an interrupted run use the same config/root and `--resume
/absolute/run/checkpoints/last.pth`. Never overwrite a finished run or old code.

## Selection, evidence and final test

Training evaluates **validation only**. Best means maximum validation R1,
earliest epoch on ties; mAP is taken at this same checkpoint. Record the joint
validation score `min(R1-R1_E0, mAP-mAP_E0)`. Do not tune against test results.

Before the one-time final test, freeze a selection JSON with `run_dir`,
`validation_epoch`, and `reason`, then run `run_research.py final-test` with
`--config`, `--run-dir`, `--selection-file`, `--output-dir`. The test dataset's
gallery orientation is allowed, but test paired-query image angles are not.

Log support counts, distinct main PIDs, raw/augmented views, loss components,
AMP skips, memory and residual/shared norm ratios; median ratio > .5 or p95 > 1
only raises a recorded warning. Evaluate all three heads on the same train
feature subset to measure functional differentiation without backpropagation.

The source E0 validation baseline is R1 72.63721466064453 / mAP 65.50774383544922.
The RDE paper CUHK target is test R1 75.94 and mAP 67.56; both must be exceeded
by one fixed checkpoint. No success is claimed by this source-only artifact.
