# TAG-PEDES IRRA-light Three-Mode Result Log

All runs use TAG-PEDES, ViT-B/16, batch size 64, seed 1, random sampler, `img_aug=False`, `identity_loss=sdm`, and test-split evaluation.

This file only records results. It does not rank or compare modes.

| mode | status | best epoch | best R1 | result source |
|---|---|---:|---:|---|
| single_pure | completed | 52 | 57.719 | `20260612_181706_irra_light_single_pure_4090_tagpedes_60e/summary.md` |
| single_proj_pure | completed | 45 | 57.538 | `20260612_210439_irra_light_single_proj_pure_4090_tagpedes_60e/summary.md` |
| split_pure | interrupted, accepted best checkpoint | 52 | 57.235 | `20260612_231536_irra_light_split_pure_4090_tagpedes_60e_INTERRUPTED/summary.md` |

The interrupted `split_pure` run is recorded because the user explicitly chose the epoch-52 `best.pth` checkpoint as the result instead of resuming.
