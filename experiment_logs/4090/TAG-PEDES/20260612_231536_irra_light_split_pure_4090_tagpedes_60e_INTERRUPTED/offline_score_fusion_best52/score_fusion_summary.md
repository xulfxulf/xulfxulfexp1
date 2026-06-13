# IRRA-light Split Pure Offline Score Fusion Diagnostic

This file records offline evaluation results only. It does not rank or compare modes.

- Config: `/root/autodl-tmp/IRRA_light_baseline/logs_4090/TAG-PEDES/20260612_231536_irra_light_split_pure_4090_tagpedes_60e/configs.yaml`
- Checkpoint: `/root/autodl-tmp/IRRA_light_baseline/logs_4090/TAG-PEDES/20260612_231536_irra_light_split_pure_4090_tagpedes_60e/best.pth`
- Dataset split: TAG-PEDES test split from the saved config
- Scores: `S_id`, `S_state`, and `S_id + lambda * S_state`
- Headline result field: best/offline R1 for each score variant

| score | R1 | R5 | R10 | mAP | mINP |
|---|---:|---:|---:|---:|---:|
| S_id | 57.241 | 76.478 | 82.689 | 43.596 | 23.321 |
| S_state | 57.338 | 76.472 | 82.810 | 43.628 | 23.349 |
| S_id + 0.05*S_state | 57.247 | 76.448 | 82.683 | 43.603 | 23.329 |
| S_id + 0.1*S_state | 57.271 | 76.448 | 82.695 | 43.610 | 23.327 |
| S_id + 0.2*S_state | 57.235 | 76.418 | 82.744 | 43.609 | 23.328 |
