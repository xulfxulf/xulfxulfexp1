# v16 Fast3 TAG-PEDES Full-Training Results

## Run contract

- Code commit: `4a6318b850a3dafc5278f91cb9232f3e80fd3372`
- Hardware: one NVIDIA GeForce RTX 4090
- Dataset: TAG-PEDES
- Schedule: 60 epochs per mode, serial execution
- Shared settings: seed 1, batch size 64, random sampler, no image augmentation
- Backbone initialization: OpenAI CLIP ViT-B/16
- Support size: 3

## Recorded results

Per the experiment logging policy, this summary records Best R1 without adding a baseline comparison.

| Mode | Status | Best R1 | Best epoch | Started | Finished |
| --- | --- | ---: | ---: | --- | --- |
| `split_bag_safe` | completed 60/60 | 54.453468322753906 | 37 | 2026-07-12 16:21:27 CST | 2026-07-12 20:15:49 CST |
| `split_bag_state` | completed 60/60 | 54.55647277832031 | 45 | 2026-07-12 20:15:51 CST | 2026-07-13 00:04:52 CST |
| `split_bag_state_hn` | completed 60/60 | 54.18686294555664 | 48 | 2026-07-13 00:04:54 CST | 2026-07-13 04:31:50 CST |

## Input isolation

| Mode | Frozen generated inputs used |
| --- | --- |
| `split_bag_safe` | `support_reliability_hard_only.csv` |
| `split_bag_state` | `support_reliability_hard_only.csv`, `support_hard_contradiction.csv` |
| `split_bag_state_hn` | `support_reliability_hard_only.csv`, `support_hard_contradiction.csv`, `hard_negative_pool.csv` |

The exact SHA256 values and original server paths are retained in each mode's `run_manifest.txt`.

## Validation

- All three logs contain `Epoch 60 done` and a final `best R1` record.
- No traceback, runtime error, OOM, NaN, Inf, or exception marker was found in the archived text logs.
- Every mode used an independent immutable code/input/run directory.
- Each server-side `best.pth` was present and 1,057,634,533 bytes after completion.
- Checkpoints and generated input tables are intentionally not stored in GitHub. This directory contains configs, full logs, run manifests, serial-launcher output, and TensorBoard event files.
