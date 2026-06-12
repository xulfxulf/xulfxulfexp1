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
- Optional single fp32 projection head initialized as an identity matrix:
  - `single_head`: a fair capacity control where both identity and state losses
    act on the same projected feature.
- Training losses with fixed weight 1:
  - `identity_sdm_loss`: SDM on identity head features, using same PID positives.
  - `state_itc_loss`: ITC on state head features, using only original image-text pairs.
- Batch identity statistics are printed as `BatchIdentityStats` for the first
  batch of each epoch and every `--light_stat_period` batches. The log includes
  duplicate identities, negative ordered pairs, duplicate images, and
  same-image ordered pairs.
- In split modes, inference uses only the identity head via `encode_image` and
  `encode_text`.

## Disabled

- MLM dataset path and MLM loss.
- Cross-modal attention and cross-modal transformer.
- Classification identity head from the original IRRA `id` loss in pure modes.
- Random-initialized module 5x learning-rate treatment for the new heads.
- Identity sampler / full identity package sampling.

## Available Modes

Use `--irra_light --irra_light_mode <mode>`.

| mode | design | losses | test feature |
|---|---|---|---|
| `single_pure` | A: single embedding clean baseline | SDM + one-to-one ITC on CLIP global features | CLIP global feature |
| `single_proj_pure` | B: single projection capacity control | SDM + one-to-one ITC on the same projected feature | single projection head |
| `split_pure` | C: clean dual-head method | identity head SDM + state head one-to-one ITC | identity head |
| `single_id` | D: single embedding stable baseline | SDM + one-to-one ITC + ID classification | CLIP global feature |
| `single_proj_id` | E: single projection stable capacity control | SDM + one-to-one ITC + ID classification on the same projected feature | single projection head |
| `split_id` | F: stable dual-head method | identity head SDM + ID classification, state head one-to-one ITC | identity head |

The default launch script uses `single_pure`, the actual no-extra-projection
baseline. Run `IRRA_LIGHT_MODE=split_pure bash run_irra_light.sh` explicitly for
the clean dual-head method.

For the first-round main table, keep `--irra_light_identity_loss sdm`. The
one-to-one identity-loss option is only an extra sanity check because it changes
the identity head from identity-level relation learning to original-pair
alignment.

## First-Round Mechanism Check

The default ablation script runs only three modes:

1. `single_pure`: raw CLIP capacity baseline.
2. `single_proj_pure`: single-head projection capacity control.
3. `split_pure`: identity/state split-head method.

Decision rule: first compare `single_proj_pure` with `single_pure` to measure
projection-capacity gain; then compare `split_pure` with `single_proj_pure` to
judge whether split-head assignment itself helps.

ID-classification modes remain implemented, but they are second-stage stability
checks. The first round should run only one dataset first, normally
`CUHK-PEDES`.

## Main Files

- `model/build.py`: IRRA-light heads and forward path.
- `processor/processor.py`: per-batch identity statistics and new loss meters.
- `utils/options.py`: `--irra_light`, seed, and forced random sampling.
- `solver/build.py`: projection heads use base learning rate.
- `run_irra_light.sh`: 3090-ready launch script.
- `run_irra_light_ablation.sh`: runs the three-mode mechanism check by default.

## 3090 Example

```bash
cd /root/shared-nvme/zixiangwang/yxyx/IRRA_light_baseline
conda activate rde_official_legacy
bash run_irra_light.sh
```

For smoke:

```bash
NUM_EPOCH=2 IMG_AUG=0 EXP_NAME=irra_light_smoke bash run_irra_light.sh
```

For the first-round mechanism check:

```bash
NUM_EPOCH=60 BASE_EXP_NAME=irra_light_mechanism_check bash run_irra_light_ablation.sh
```
