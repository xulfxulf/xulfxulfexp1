# V017: five-epoch ICFG transfer of retained V015

This independent snapshot transfers V015's mechanism/settings to ICFG-PEDES.
It excludes V016's unsuccessful epsilon0.0001. First measure a matching
five-epoch S E0,then train the method independently from OpenAI for five
epochs. E0 weights are a reference,not the method initializer. CUHK/RSTP
scores cannot substitute for the new ICFG baseline.

## Fixed data protocol

Original images/OEFormer/annotations stay read-only. Derived inputs live in
`/root/autodl-tmp/TBPS_ViewResearch_ICFG_inputs_20260907`. No official val:
sort original train PIDs,permute with numpy RandomState(1),reserve first310
IDs for val. All images/captions of onePID stay together; testIDs are excluded.

Quarantine BOTH conflicting PID1/PID5 records of this official train path:
`test/2868/2868_005_05_0114afternoon_0833_1_ex.jpg`.
Do not guess its true identity or treat its folder name as the dataset split.
The remaining34672unique train images have3102IDs; derived counts are:

| Split | Images | IDs | Captions |
|---|---:|---:|---:|
|train|31289|2792|31289|
|val|3383|310|3383|
|test|19848|1000|19848|

Train images have one original caption and one supplied backtranslation.
Derived val keeps source_split=train for OEFormer association. Derived test
preserves official test content/order and source_split=test. Runtime checks
deterministic partition semantics,identity/path disjointness,source metadata,
expected counts,orientation association and image existence. Any changed
conflict/partition is rejected. No image/weight/batch/code hashes are computed.
Expected train cross-view eligibility29221/31289=93.39065% motivates transfer;
it is not a claimed retrieval gain.

## Baseline and training

Both runs:OpenAI ViT-B/16,384x128,4GPUs,global320=4x80,seed1,5epochs,
97batches/epoch. E0 expands caption pairs with DistributedSampler and keeps
S N-ITC+R-ITC epsilon0.01 and original conv1 freezing. Method uses V015's
view-aware P160K2,epsilon0.001,and unfreezes ALL original parameters. This
is a full-system comparison,not an isolated expert ablation.

PK batch:160uniqueIDs,twoimages each,40wholePID pairs/rank. Prefer different
raw four-views and angle gap>=60,then balance raw four-view counts. Draw
caption per image; reject repeated images when multiple images exist.
Freeze all five plans before training;97x320=31040samples/epoch.

Keep the two-draw augmentation pool,portrait crop aspect(.25,4/9),scale(.9,1),
flips/rotation/color/erasing,ImageNet normalization,BT0.1,random deletion0.05,
text dropout0.05. No test query's paired-image view enters text or scores.

## Unchanged V015 mechanism

h=projected image CLS; X=projected non-CLS patches. OEFormer72angle masses
give F=[150,210],B=[330,360)union[0,30],S=rest. Uncalibrated confidence
c=clamp(1-H(p3)/log3,0,1); manual calibration waived. Horizontal flip preserves
these merged F/S/B probabilities.

```
attention_g = softmax(LayerNorm_no_affine(X) @ query_g)
z_g = sum_n attention_g[n] * X[n]
r_g = Head_g(h + z_g)
r = c * sum_g p3[g] * r_g
v = normalize(h + 0.2*r)
t = normalize(TextEncoder(caption))
score = dot(t,v)
```

Queries start zero. FP32 heads:LN512-Linear128-GELU-Linear512,zero final layer.
FP32 fusion casts to h.dtype before normalize. Main,fresh no-grad target and
gallery share the image route. Text representation and cosine scoring stay unified.

Up to16extra supports/rank,train only:samePID,different image,different peak
F/S/B,circular angle gap>=60 before/after augmentation. Isolate support RNG;
supports do not enter the main negative bank. Detached weight
w=c_a*c_b*(1-dot(p_a,p_b)) defines global weighted means:

- Shared:1-cos(h_a,h_b),encoder gradient.
- Decorrelation:cos(r_a,r_b)^2,encoder CLS AND patches detached; heads/queries only.
- Total=N-ITC+R-ITC+lambda_shared*shared+lambda_dec*decorrelation.

N-ITC mixes normalized samePID q and fresh no-grad soft targets; beta0->0.5
in epoch1,then0.5. R-ITC uses log(q+0.001),without target renormalization.
Shared weight0->0.4 in epoch1; decorrelation0 in epoch1,0->0.0003 in epoch2,
then0.0003. Empty support losses stay connected to the graph; DDP uses
world-scaled local numerator/global mass.

AdamW betas(.9,.98),eps1e-8,wd0.02 except bias/norm0. Base peak1e-4,
heads/queries peak1e-3. Oneepoch warmup from1%peak,cosine toward5%peak.
Unfreeze before DDP/optimizer; resetTorch/CUDA RNG after construction;
globally synchronize AMP overflow decisions. Visual checkpointing is
grad-training only. Retain residual-size/same-input expert diagnostics.

## Execution

Server:Python3.8.20,torch1.13.0+cu117,torchvision0.14.0+cu117,4xRTX4080.
Never overwrite an existing source,input,launch or output directory.

```bash
python -m unittest discover -s tests_research -v
python prepare_icfg_inputs.py --config configs/icfg_portrait_e0.yaml --output-dir /root/autodl-tmp/TBPS_ViewResearch_ICFG_inputs_20260907
python launch_research.py --stage e0-audit --config configs/icfg_portrait_e0.yaml --record-dir /root/autodl-tmp/TBPS_ViewResearch_v017_20260907/e0_audit_launch --output-dir /dev/shm/TBPS_ViewResearch_v017_20260907/e0_audit_run
python launch_research.py --stage e0-train --config configs/icfg_portrait_e0.yaml --record-dir /root/autodl-tmp/TBPS_ViewResearch_v017_20260907/e0_validation_launch --output-dir /dev/shm/TBPS_ViewResearch_v017_20260907/e0_validation_run
python persist_completed_run.py --phase e0_validation_run --output-dir /root/autodl-tmp/TBPS_ViewResearch_v017_20260907/e0_validation_run
```

Audit must show actual updates/rank agreement. Verify E0 completed485batches,
5validation epochs,no test,and correct best selection. Only then initialize
the separate ICFG ledger and run the method:

```bash
python launch_research.py --stage audit --config configs/research_v017.json --record-dir /root/autodl-tmp/TBPS_ViewResearch_v017_20260907/audit_launch --output-dir /dev/shm/TBPS_ViewResearch_v017_20260907/audit_run
python launch_research.py --stage train --config configs/research_v017.json --record-dir /root/autodl-tmp/TBPS_ViewResearch_v017_20260907/validation_launch --output-dir /dev/shm/TBPS_ViewResearch_v017_20260907/validation_run
python run_research.py verify-result --result /dev/shm/TBPS_ViewResearch_v017_20260907/validation_run/result.json
python persist_completed_run.py --phase validation_run --output-dir /root/autodl-tmp/TBPS_ViewResearch_v017_20260907/validation_run
```

Selection=max valR1,earliest tie; paired mAP is the search metric. Test opens
only after explicit frozen selection,never per epoch/trial. Selected weights
must have actual decorrelation updates. Working final ICFG target:mAP>=41.06
(RDE40.06+1pp),with R1 always reported. Holdout val scores are not published
test scores. No automatic final test is queued.

Persist compact best/records and protect full last before local archival.
Full last includes optimizer and4rank RNG; best is selection-only. Publish
source/compact completed records,not checkpoints,images or raw smoke logs.
