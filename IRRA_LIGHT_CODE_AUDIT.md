# IRRA-light Code Audit

Audit date: 2026-06-11

Scope: local code in `D:\004SSH\IRRA_light_baseline`, derived from official `anosorae/IRRA`.

## Findings

### P1: Head output is not explicitly normalized in `forward`

Requirement says each head should be one linear layer followed by normalization. The current implementation applies the linear heads in `model/build.py`, but does not normalize the head outputs before returning or before optional ID classification:

- `identity_head/state_head` are applied at `model/build.py:128-131`.
- SDM and ITC normalize internally through `objectives.compute_sdm` / `objectives.compute_itc`, so the training losses are mathematically using normalized features.
- Evaluation also normalizes in `utils/metrics.py:79-82`.

This means retrieval/loss behavior is mostly correct, but the module itself is not literally `linear + normalize`. Any later diagnostic that reads `identity_i_feats`, `state_i_feats`, or checkpoint features before the loss/evaluator will see unnormalized outputs. For strict alignment with the design note, add a small helper that applies `F.normalize(head(x), dim=-1)` in both training and `encode_image/encode_text`, and avoid double-normalization concerns because re-normalizing unit vectors is harmless.

### P2: State ITC treats duplicate-image alternate captions as negatives

`ImageTextDataset` creates one training row per caption, with the same `image_id` reused for multiple captions of the same image (`datasets/bases.py:75-88`). The state branch uses ordinary one-to-one ITC (`model/build.py:151-154`), and `compute_itc` uses only the diagonal as positives.

This matches the narrow "current row only" interpretation, but it can conflict with CUHK-style data where one image has multiple valid captions. If two captions of the same image appear in the same random batch, the off-diagonal image-caption pair is a true original image-caption relation but is treated as a negative by ITC. This is inherited from standard ITC, but it matters more for the stated "original-pair detail branch" interpretation.

Recommended: log duplicate `image_ids` per batch together with PID stats. If duplicate image rows are common enough, consider a later `state_multi_positive_itc` variant using `image_ids`; do not change the first pure version unless you want the first version to define "original pair" as "same image can have multiple original captions."

### P2: No server smoke has run yet

Local checks passed, but the code has not been uploaded or executed on the 3090 because the Paratera SSH gateway currently authenticates through Paramiko but closes session/SFTP channels, while OpenSSH askpass returns password denied. Therefore server-specific issues remain untested:

- CLIP cache availability.
- PyTorch/TorchVision version compatibility in the 3090 environment.
- CUDA memory and actual batch throughput.
- Whether `tensorboard`, `prettytable`, `easydict`, and `regex` are present in the selected conda env.

This is not a code logic failure, but it is a deployment risk. A smoke run to epoch 2 is still required before full training.

### P3: Four-mode expansion is implemented, but default remains split-pure

The code now supports:

- `single_pure`: single embedding, SDM + ITC.
- `split_pure`: identity head SDM + state head ITC.
- `single_id`: single embedding, SDM + ITC + ID classification.
- `split_id`: identity head SDM + ID classification, state head ITC.

This matches the design note. However, the original user request asked first for the clean IRRA-light split baseline. The default `run_irra_light.sh` uses `split_pure`, which is correct. The ablation script is optional and should not be treated as the default experiment.

### P3: Batch identity logging is intentionally heavy

`processor/processor.py:71-75` logs identity statistics every batch. This satisfies the requirement, but a 60-epoch CUHK run will produce roughly 60k+ extra log lines. That is acceptable for diagnosis and small compared with checkpoints, but it should be expected.

## Verified Correct

### IRRA-light disables MLM and cross-modal attention

- `utils/options.py:84-87` forces `MLM=False`, `sampler=random`, and `loss_names=irra_light` whenever `--irra_light` is used.
- `datasets/build.py:85-92` therefore selects `ImageTextDataset`, not `ImageTextMLMDataset`.
- `model/build.py:36` prevents MLM/cross-modal modules from being created in IRRA-light.
- `model/build.py:191-207` is unreachable in IRRA-light because the light branch returns at `model/build.py:167`.

### Random batch sampling is enforced

- `utils/options.py:84-87` overrides any accidental `--sampler identity`.
- `datasets/build.py:116-123` uses a shuffled random `DataLoader`.
- `run_irra_light.sh:21-24` explicitly passes `--sampler random` and the 3090 dataset root.

### Split-pure loss assignment matches the design

For `split_pure`:

- `identity_head` and `state_head` are created only for split modes at `model/build.py:25-29`.
- Identity features are trained with SDM at `model/build.py:138-142`.
- State features are trained with one-to-one ITC at `model/build.py:151-154`.
- No ID classifier is added in pure split mode because `self.irra_light_with_id` is false at `model/build.py:18` and the classifier branch at `model/build.py:31` does not run.
- Total loss is the unweighted sum of loss entries in `processor/processor.py:77-78`.

### Evaluation uses only identity head for split modes

- `utils/metrics.py:56-67` calls `model.encode_text` and `model.encode_image`.
- In split modes, `encode_image` and `encode_text` return `identity_head(...)` only (`model/build.py:98-111`).
- State head is not used in evaluation, and there is no score fusion.

### New heads remain single precision and do not get 5x learning rate

- `build_model` calls `convert_weights(model)` at `model/build.py:215`, then restores projection heads to fp32 through `float_projection_heads()` at `model/build.py:216`.
- `float_projection_heads` restores `identity_head`, `state_head`, and light-mode classifier if present (`model/build.py:77-82`).
- `solver/build.py:20-22` keeps `identity_head/state_head` at base learning rate.
- `solver/build.py:29-30` prevents `classifier/mlm_head` 5x logic from applying in IRRA-light.

### Local static checks passed

Commands run locally:

```text
python -m py_compile train.py test.py model\build.py model\objectives.py processor\processor.py solver\build.py utils\options.py datasets\build.py
```

Also checked:

- `--irra_light` forces `MLM=False`, `sampler=random`, `loss_names=irra_light`.
- All four `irra_light_mode` values parse.
- SDM/ITC functions run on dummy tensors.
- Batch identity stats format works.

## Recommendation Before Training

1. Fix P1 if strict design fidelity matters: make head outputs explicitly normalized in `model/build.py`.
2. Add `image_id` duplicate stats to batch logging so state-branch false negatives can be measured.
3. Upload after the 3090 SSH gateway recovers.
4. Run smoke first:

```bash
cd /root/shared-nvme/zixiangwang/yxyx/IRRA_light_baseline
conda activate rde_official_legacy
NUM_EPOCH=2 EXP_NAME=irra_light_split_pure_smoke IRRA_LIGHT_MODE=split_pure bash run_irra_light.sh
```

5. Only after smoke reaches epoch 2 and validation, run the four-mode comparison if desired:

```bash
NUM_EPOCH=60 BASE_EXP_NAME=irra_light_first_round bash run_irra_light_ablation.sh
```
