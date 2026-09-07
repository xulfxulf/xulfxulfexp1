# Train-only R-ITC target-floor diagnosis

Completed2026-09-07 on4xRTX4080,torch1.13.0+cu117. Ten fixed train PK batches
at V014 selected epoch5; identical weights,inputs,dropout and beta0.5. No
optimizer updates,validation selection or test access. Source:e988fc9.
Compare epsilon0.01 with0.001 in the existing log(q+epsilon) expression;
do not renormalize targets or alter N-ITC. N-ITC was numerically identical.

| Gradient group | Mean lower/original norm | Mean cosine |
|---|---:|---:|
| Visual last block |6.162012|0.340801|
| Visual projection |2.730590|0.448919|
| Text last block |4.954501|0.393487|
| Text projection |1.889568|0.554186|
| Residual heads |2.177578|0.571936|
| Patch queries |2.511533|0.537580|

All measured gradients finite. Logit-scale norm ratio is ill-conditioned
when its original norm is near zero: mean83.18x,max358.92x; cosine is negative
in one of10 batches. Do not interpret this as uniformly stronger useful signal.
The lower epsilon changes gradient direction appreciably and is a controlled
hypothesis,not evidence of higher retrieval scores. Run an independent
five-epoch V015 from OpenAI after a full-batch four-GPU update audit; keep
all other V014 settings. Do not resume V014 or tune against test.

This is a parameter experiment within TBPS-S,not a claim of RDE reproduction.
Official TBPS-S uses0.01; RDE has a related SDM function but defaults to TAL.
Raw numerical evidence and exact batch indices are in the adjacent JSON;
runtime metadata is in ritc_floor_diagnosis_manifest_20260907.json.
