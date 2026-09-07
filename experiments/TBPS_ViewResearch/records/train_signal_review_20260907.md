# Train-signal review before the next five-epoch trial

This is read-only diagnosis, not a new retrieval result or a chosen test
configuration. Existing seed1 CUHK prepared plans,epochs1-5,1060 batches:

| Statistic | E0 caption-pair plan | Existing PK plan |
| --- | ---: | ---: |
| Mean distinct PIDs per global320 | 315.4915 | 160 |
| Same-PID candidates per anchor,including self | 1.028455 | 2 |
| Different-image same-PID candidates per anchor | 0.023703 | 0.999735 |
| Repeated-same-image candidates per anchor,excluding self | 0.004752 | 0.000265 |
| Train images encountered over5epochs | 34054 | 32637 |
| Train caption pairs encountered over5epochs | 68126 | 63513 |
| Train PIDs encountered | 11003 | 11003 |

Numbers come from train_plan_diagnosis_20260907.json. Both plans have identical
batch size and step count. PK allocates whole PID pairs to ranks,chooses two
different images when available,prioritizes distinct raw four-views with
angle gap>=60,then balances raw four-view counts; captions are randomly drawn
per selected image. Single-image identities necessarily repeat the image.
F/S/B auxiliary support eligibility remains a separate stricter condition.
PK gives stronger repeated-ID supervision but reduces five-epoch image and
caption coverage; that tradeoff must be measured,not called an improvement.

Current N-ITC and R-ITC already build normalized same-PID targets. Thus the
sampling question is whether useful off-diagonal positive targets occur in
practice,not whether the code supports them. The loss is not diagonal-only.
[Upstream TBPS implementation](https://raw.githubusercontent.com/Flame-Chasers/TBPS-CLIP/master/model/tbps_model.py).

A separate possible later diagnostic is R-ITC target-floor smoothing. TBPS
uses1e-2; the RDE repository's SDM helper exposes1e-8. This observation is
about two particular functions,not a claim that RDE's default training is
SDM or that changing epsilon reproduces RDE. No epsilon or loss has been
changed in V012/V013. Avoid combining sampling and loss changes in one trial.
[RDE loss helpers](https://raw.githubusercontent.com/QinYang79/RDE/main/2024-CVPR-RDE/model/objectives.py).

Finish and log V013 first. Then evaluate the next focused hypothesis under
the prospective mAP-focused five-epoch protocol. Keep maximum-validation-R1
checkpoint selection and report its paired mAP. No test retrieval occurred.
