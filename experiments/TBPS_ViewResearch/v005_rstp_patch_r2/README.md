# V005 r2: RSTPReid baseline and soft-view patch experts

Independent full code snapshot. This assesses the V004 mechanism on RSTPReid,
not an additional CUHK hyperparameter trial. Seed 1 only, 4 GPUs, local batch
80 (global 320), 5 baseline epochs followed by 5 continuation epochs. All
model and loss definitions of V004 are unchanged. Do not use the legacy
`run_view4.py queue` or its inherited CUHK-only audit tests as this entry.

## Inputs and baseline

Use the existing official TBPS RSTP `train_reid.json`, `val_reid.json`, and
`test_reid.json`, with paired training back-translations. Expected image /
identity / caption counts are 18505/3701/37010, 1000/200/2000, 1000/200/2000.
Check unique images, disjoint split identities, orientation-to-image/PID/split
associations and complete captions. Original datasets are never written.
Use only train identities to construct same-person support candidates.

Precompute seed-1 sampling plans and small image-ID traces outside datasets.
Main batches retain the original expanded-caption-pair DistributedSampler
(sampler seed 0, epoch set to epoch-1). There are 115 full 4x80 steps per
epoch, not the CUHK-specific 212. PK plans are generated only for inherited
input-validation compatibility; they are NOT used by either training stage.

E0 initializes from the OpenAI ViT-B/16 checkpoint, not CUHK-trained weights.
It has no experts and keeps S N-ITC + R-ITC, original conv1 freeze, transforms,
text augmentation, AdamW groups and 5-epoch LR schedule (peak 1e-4). LR and
soft-label warm-up are scaled using 115 steps/epoch. E0 evaluates only val;
it does not run the old automatic final test. Its completed best-R1 record
supplies the frozen RSTP validation baseline for continuation.

## Continuation mechanism

Shared h is CLIP's projected image CLS. Text t is the single normalized CLIP
text feature. OEFormer 72 probabilities are summed into F [150,210],
B [330,360) union [0,30], and S the remainder. Confidence is
`c=clamp(1-H(p3)/log(3),0,1)`, explicitly uncalibrated; human labels were waived.
Horizontal flip swaps the fine angles but leaves merged F/S/B mass invariant.

For projected visual patches X excluding CLS, each expert has a zero-initial
query q_g and FP32 head LN(512)-Linear(512,128)-GELU-Linear(128,512):

```
a_g = softmax(LayerNorm_no_affine(X) @ q_g)  # over patches
z_g = sum_n a_g[n] * X[n]
r_g = Head_g(h + z_g)
r = c * sum_g p3[g] * r_g
v = normalize(h + 0.2*r)
score = dot(t, v)
```

Final head layers start at zero. Initial retrieval equals E0. Fusion is FP32
then cast to h.dtype before normalization, as in V004. Main, fresh no-grad
target and gallery use the same route. Text never reads query-source views.

Keep original N-ITC+R-ITC. Each rank adds up to 16 train-only supports from
same PID, different image, different peak F/S/B, with circular angle gap >=60
both raw and augmented. Main negatives and caption sampling are unchanged.
With detached weight `w=c_a*c_b*(1-dot(p_a,p_b))`:

- Shared consistency: weighted global mean `1-cos(h_a,h_b)`.
- Residual decorrelation: weighted global mean `cos(r_a,r_b)^2`, with both
  CLS and patches detached from the encoder; heads and queries still update.
- Total: N-ITC + R-ITC + lambda_shared*shared + lambda_dec*decorrelation.

Shared lambda ramps to 0.4 in epoch1; decorrelation is zero in epoch1, ramps
to 0.0003 in epoch2, then fixed. Base peak LR 1e-6, expert/query peak LR 1e-3;
all original parameters unfreeze before DDP/optimizer. No loss or sampler
retuning relative to V004. Preserve RNG isolation, synchronized AMP skip
decisions, same-input expert diagnostics and transactional best/last saving.

## Commands

Pinned server environment: Python3.8, torch1.13.0+cu117, torchvision0.14.0+cu117.
Use explicit absolute output directories, each previously nonexistent.

```bash
python -m unittest discover -s tests_research -v
python run_rstp_e0.py prepare --config configs/rstp_e0.yaml \
  --output-dir /root/autodl-tmp/TBPS_ViewResearch_RSTP_inputs_20260907
python launch_research.py --stage e0-audit --config configs/rstp_e0.yaml \
  --record-dir /absolute/e0_audit_launch --output-dir /absolute/e0_audit
python launch_research.py --stage e0-train --config configs/rstp_e0.yaml \
  --record-dir /absolute/e0_launch --output-dir /root/autodl-tmp/TBPS_ViewResearch_v005_20260907_r2/e0_validation_run
python run_rstp_e0.py verify-result --result /absolute/e0_validation_run/result.json
python launch_research.py --stage audit --config configs/research_v005.json \
  --record-dir /absolute/research_audit_launch --output-dir /absolute/research_audit
python launch_research.py --stage train --config configs/research_v005.json \
  --record-dir /absolute/research_launch --output-dir /absolute/validation_run
python run_research.py verify-result --result /absolute/validation_run/result.json
```

E0 audit uses eight real steps; continuation audit uses four. Both require at
least one non-skipped optimizer update. The original four-step E0 audit had
four synchronized AMP skips and is preserved, not accepted as an update test.
Audit length does not change full-training AMP settings or losses. Log E0 result
and initialize a RSTP-specific validation ledger before continuation. The
research JSON resolves baseline metrics from its own completed E0 result,
then writes concrete values in the run config. Missing/mismatched E0 fails.

Max val R1 selects best, earliest tie, with mAP from that same checkpoint.
Test remains unopened during search. Separate frozen final-test selection
requires the selected checkpoint to have actually received decorrelation
updates. RDE RSTP target is R1 65.35 AND mAP 50.88 on the SAME test checkpoint,
both strictly exceeded at two decimals. A baseline-only win cannot satisfy
the innovation goal. Code/results are published; raw smoke/failed logs stay
local, and checkpoints/data/credentials are never pushed to GitHub.
