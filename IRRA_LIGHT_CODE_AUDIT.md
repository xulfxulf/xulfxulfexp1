# IRRA-light Code Audit

Audit date: 2026-06-11

Scope: local code in `D:\004SSH\IRRA_light_baseline`, derived from official
`anosorae/IRRA`.

## Current Status

The code now implements a clean first-round IRRA-light matrix with six modes:

- `single_pure`: raw CLIP global feature, identity SDM + original-pair ITC.
- `single_proj_pure`: one trainable projection head, both losses on the same projected feature.
- `split_pure`: identity head gets SDM, state head gets original-pair ITC.
- `single_id`: raw CLIP global feature with identity classification.
- `single_proj_id`: one projection head with identity classification.
- `split_id`: identity head gets SDM + ID classification, state head gets original-pair ITC.

The default launch script now runs the true baseline, `single_pure`, with image
augmentation disabled and validation-set model selection enabled.

## Fixed Issues

### Logged tensors no longer keep computation graphs

`utils/meter.py` detaches tensor inputs and stores Python floats. The training
loop in `processor/processor.py` also explicitly converts all meter inputs and
TensorBoard scalar values through `to_scalar`.

### Projection-head capacity controls are now fair

`single_proj_pure` and `single_proj_id` add one trainable fp32 projection head.
This lets `split_pure` be compared against a single-head capacity control, not
only against the raw CLIP feature baseline.

### Evaluation uses the intended feature

- Split modes return `identity_head(...)` from `encode_image` and `encode_text`.
- Single-projection modes return `single_head(...)`.
- Raw single modes return the CLIP global feature.

The state head is never used for retrieval.

### Batch diagnostics are retained but throttled

`BatchIdentityStats` is logged for the first batch of each epoch and every
`--light_stat_period` batches. It includes duplicate identities, negative
ordered pairs, duplicate images, and `same_image_ordered_pairs`.

### Reproducible dataloader worker seeding was added

Training dataloaders use `worker_init_fn` and a seeded `torch.Generator`, both
derived from `args.seed`.

## Remaining Runtime Gate

The code has passed local static checks, but a 3090 server smoke test is still
required. The current server blocker is the Paratera SSH gateway: OpenSSH
returns password-denied after partial password success, and Paramiko opens a
transport but cannot authenticate a usable session/SFTP channel.

Before formal training, run two-epoch smoke tests for at least:

```bash
CUDA_VISIBLE_DEVICES=0 NUM_EPOCH=2 IMG_AUG=0 IRRA_LIGHT_MODE=single_pure EXP_NAME=smoke_single_pure bash run_irra_light.sh
CUDA_VISIBLE_DEVICES=0 NUM_EPOCH=2 IMG_AUG=0 IRRA_LIGHT_MODE=split_pure EXP_NAME=smoke_split_pure bash run_irra_light.sh
```

Only after smoke reaches validation should the six-mode ablation be launched.
