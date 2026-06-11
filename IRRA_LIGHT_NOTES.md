# IRRA-light clean two-head baseline

Source: official IRRA code from `https://github.com/anosorae/IRRA`.

## Goal

Build a clean IRRA-light baseline that keeps the CLIP image and text encoder
backbones unchanged while removing the heavy IRRA relation machinery.

## Enabled

- CLIP image encoder and CLIP text encoder from the official IRRA repository.
- Random batch sampling with `--sampler random`.
- Fixed seed through `--seed`.
- Optional two fp32 projection heads initialized as identity matrices:
  - `identity_head`: used for retrieval and same-identity alignment.
  - `state_head`: used only for original paired image-text alignment during training.
- Training losses with fixed weight 1:
  - `identity_sdm_loss`: SDM on identity head features, using same PID positives.
  - `state_itc_loss`: ITC on state head features, using only original image-text pairs.
- Per-batch identity statistics are printed as `BatchIdentityStats`.
- Inference uses only the identity head via `encode_image` and `encode_text`.

## Disabled

- MLM dataset path and MLM loss.
- Cross-modal attention and cross-modal transformer.
- Classification identity head from the original IRRA `id` loss in pure modes.
- Random-initialized module 5x learning-rate treatment for the new heads.
- Identity sampler / full identity package sampling.

## First-Round Modes

Use `--irra_light --irra_light_mode <mode>`.

| mode | design | losses | test feature |
|---|---|---|---|
| `single_pure` | A: single embedding clean baseline | SDM + one-to-one ITC on CLIP global features | CLIP global feature |
| `split_pure` | B: clean dual-head method | identity head SDM + state head one-to-one ITC | identity head |
| `single_id` | C: single embedding stable baseline | SDM + one-to-one ITC + ID classification | CLIP global feature |
| `split_id` | D: stable dual-head method | identity head SDM + ID classification, state head one-to-one ITC | identity head |

The default launch script uses `split_pure`, the clean B version.

## Main Files

- `model/build.py`: IRRA-light heads and forward path.
- `processor/processor.py`: per-batch identity statistics and new loss meters.
- `utils/options.py`: `--irra_light`, seed, and forced random sampling.
- `solver/build.py`: projection heads use base learning rate.
- `run_irra_light.sh`: 3090-ready launch script.
- `run_irra_light_ablation.sh`: runs A/B/C/D modes serially.

## 3090 Example

```bash
cd /root/shared-nvme/zixiangwang/yxyx/IRRA_light_baseline
conda activate rde_official_legacy
bash run_irra_light.sh
```

For smoke:

```bash
NUM_EPOCH=2 EXP_NAME=irra_light_smoke bash run_irra_light.sh
```

For the first-round four-mode comparison:

```bash
NUM_EPOCH=60 BASE_EXP_NAME=irra_light_first_round bash run_irra_light_ablation.sh
```
