# V007 portrait S reference

Source 262a2f8. RSTPReid, OpenAI CLIP ViT-B/16 initialization, seed 1,
four ranks x 80 caption pairs, five epochs. Train/eval geometry 384x128,
relative crop-aspect jitter preserved. Original S losses and training rates.
This is a new geometry reference, not the official square-input E0 and not
the proposed soft-expert mechanism.

| Epoch | Validation R1 | Validation mAP |
| --- | ---: | ---: |
| 1 | 39.300003 | 31.185215 |
| 2 | 41.300003 | 33.666904 |
| 3 | 45.150002 | 35.485378 |
| 4, selected | 46.550003 | 37.434807 |
| 5 | 45.100002 | 37.359146 |

Max validation R1 selects epoch4. Both reported metrics come from that
same checkpoint. Relative to measured 224x224 S reference, R1 is +1.3000pp
and mAP +0.5823pp in this seed. This does not establish a general effect.
The subsequent method must be evaluated against this portrait reference.

All 575 steps visited, five initial synchronized AMP skips and none later.
Exit code0, result guard passed, launch elapsed551.91 seconds. No test
evaluation. Best is selection-only; full last retains exact resume state.
No earlier experiment checkpoint or original dataset was modified.
