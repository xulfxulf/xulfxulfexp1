# V009: CUHK portrait inputs and soft-view patch experts

Independent full code snapshot based on V007's tested portrait implementation,
with CUHK counts/paths restored. Seed 1, four GPUs, global batch 320. First measure
a NEW five-epoch portrait S baseline from OpenAI, then five epochs with the
unchanged V005 soft-view mechanism. This is NOT the official 224x224 E0.
The method's gain must be reported against its own measured portrait baseline.
Do not use the legacy `run_view4.py queue`. No inference flip averaging is used.

## Geometry change

Train main/support images, val and final-test gallery all use H/W=384/128.
The fixed profile is asserted in both baseline and method configs. Visual
tokens become 24x8 patches plus CLS. The upstream `misc.build.resize_pos_embed`
bilinearly resizes OpenAI's 14x14 positional grid to 24x8, retaining the CLS
position exactly. The method then loads its own portrait E0 without further
interpolation. Text length, normalization, losses and scoring are unchanged.

Keep the two-draw original augmentation pool, scale=(0.9,1), and all original
color/rotation/erasing/flip settings. RandomResizedCrop's W/H ratio changes
from (0.75,4/3) to (0.25,4/9), preserving relative jitter around the new W/H.
Using the old square ratio on an already-portrait image would often fall
back to a central crop. This geometry adaptation is part of the same trial,
not an unreported loss/sampler change. ImageNet mean/std stay unchanged.

Prior RSTP evidence: portrait E0 validation R1/mAP 46.5500/37.4348, versus
square E0 45.2500/36.8525. This motivates a CUHK geometry trial; it is NOT
evidence of CUHK improvement. Fresh CUHK portrait E0 must be measured first.
Official references: [TBPS S config](https://raw.githubusercontent.com/Flame-Chasers/TBPS-CLIP/master/config/s.config.yaml)
and [RDE input defaults](https://raw.githubusercontent.com/QinYang79/RDE/main/2024-CVPR-RDE/utils/options.py).

## Inputs and baseline

Use the existing official TBPS CUHK `train_reid.json`, `val_reid.json`, and
`test_reid.json`, with paired training back-translations. Expected image /
identity / caption counts are 34054/11003/68126, 3078/1000/6158, 3074/1000/6156.
Check unique images, disjoint split identities, orientation-to-image/PID/split
associations and complete captions. Original datasets are never written.
Use only train identities to construct same-person support candidates.

Reuse the existing CUHK seed-1 sampling plans and catalog without modification.
Main batches retain the original expanded-caption-pair DistributedSampler
(sampler seed 0, epoch set to epoch-1). There are 212 full 4x80 steps per
epoch. Existing PK plans are checked only for inherited
input-validation compatibility; they are NOT used by either training stage.

E0 initializes from the OpenAI ViT-B/16 checkpoint, not previously fine-tuned weights.
It has no experts and keeps S N-ITC + R-ITC, original conv1 freeze, transforms,
text augmentation, AdamW groups and 5-epoch LR schedule (peak 1e-4). LR and
soft-label warm-up are scaled using 212 steps/epoch. E0 evaluates only val;
it does not run the old automatic final test. Its completed best-R1 record
supplies the frozen CUHK portrait validation baseline for continuation.

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
retuning relative to V005. Preserve RNG isolation, synchronized AMP skip
decisions, same-input expert diagnostics and transactional best/last saving.

## Commands

Pinned server environment: Python3.8, torch1.13.0+cu117, torchvision0.14.0+cu117.
Use explicit absolute output directories, each previously nonexistent.

```bash
python -m unittest discover -s tests_research -v
# Reuse the existing image/catalog/sampling plans; do NOT overwrite prepared inputs.
python launch_research.py --stage e0-audit --config configs/cuhk_portrait_e0.yaml \
  --record-dir /absolute/e0_audit_launch --output-dir /absolute/e0_audit
python launch_research.py --stage e0-train --config configs/cuhk_portrait_e0.yaml \
  --record-dir /absolute/e0_launch --output-dir /dev/shm/TBPS_ViewResearch_v009_20260907/e0_validation_run
python run_cuhk_e0.py verify-result --result /absolute/e0_validation_run/result.json
# Persist completed E0 first, then log its measured validation baseline.
python persist_completed_run.py --phase e0_validation_run \
  --output-dir /root/autodl-tmp/TBPS_ViewResearch_v009_20260907/e0_validation_run
python launch_research.py --stage audit --config configs/research_v009.json \
  --record-dir /absolute/research_audit_launch --output-dir /absolute/research_audit
python launch_research.py --stage train --config configs/research_v009.json \
  --record-dir /absolute/research_launch --output-dir /dev/shm/TBPS_ViewResearch_v009_20260907/validation_run
python run_research.py verify-result --result /absolute/validation_run/result.json
```

E0 audit uses eight real steps; continuation audit uses four. Both require at
least one non-skipped optimizer update. The original four-step E0 audit had
four synchronized AMP skips and is preserved, not accepted as an update test.
Audit length does not change full-training AMP settings or losses. Log E0 result
and initialize a CUHK-portrait validation ledger before continuation. The
research JSON resolves baseline metrics from its own completed E0 result,
then writes concrete values in the run config. Missing/mismatched E0 fails.

## Storage-only engineering change

The existing persistent server disk has insufficient space for a new-run
transaction guard. This NEW trial writes to the available 124GiB RAM filesystem
under `/dev/shm/TBPS_ViewResearch_v009_20260907`; no guard is lowered. Both
audits and formal runs require the existing full atomic-save safety checks.
Old experiment checkpoints are neither moved nor deleted.

RAM output is volatile on server shutdown. Therefore after EACH formal stage:
verify completion, persist all logs/configs plus compact best on the server
with `persist_completed_run.py`, and immediately download the completed full
last plus best to `D:/004SSH/TBPS_ViewResearch_v009_archives_20260907/<phase>/`.
Check transfer sizes and load checkpoint metadata locally. Do not count an
experiment as archived until these checks pass. Do not shut down the server
while training or before this archive completes. Leave RAM sources intact.
The local disk was checked to have over200GiB available before launch.

The persistent copy helper accepts only this version's two exact completed
phase paths, never overwrites a previous destination, and copies without
changing model/optimizer contents. Server best is selection-only, local
last is the full resumable state. Completed result JSON is published separately
from smoke/failure logs, which stay local.

`compact_best=true`: best retains model, config, selection metadata, scaler,
all-rank RNG and provenance but omits optimizer moments. It is explicitly
selection-only; resume requires the full last checkpoint. Last still stores
the exact AdamW moments and all resume state. Versioned best/last transaction
ordering and alias recovery are unchanged. No old experiment is modified.

For new compact runs the startup guard budgets 2.8 times the conservative
full-checkpoint estimate plus 2GiB reserve, covering two full last files and
two compact best files during an interrupted/ongoing transaction. Each actual
write additionally checks tensor bytes + 2% + 16MiB metadata + 1GiB headroom.
Insufficient space fails before writing; there is no automatic deletion of
other runs. Unit tests cover interrupted commit recovery and bit-identical
next AdamW update after loading full last. This storage change does not alter
the model's forward pass, gradient, optimizer updates or RNG sequence.

Max val R1 selects best, earliest tie, with mAP from that same checkpoint.
Test remains unopened during search. Separate frozen final-test selection
requires the selected checkpoint to have actually received decorrelation
updates. RDE CUHK target is R1 75.94 AND mAP 67.56 on the SAME test checkpoint,
both strictly exceeded at two decimals. A baseline-only win cannot satisfy
the innovation goal. Code/results are published; raw smoke/failed logs stay
local, and checkpoints/data/credentials are never pushed to GitHub.
