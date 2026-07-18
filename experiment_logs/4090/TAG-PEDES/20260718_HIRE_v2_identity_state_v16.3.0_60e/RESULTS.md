# HIRE-v2 Identity-State TAG-PEDES 60-Epoch Results

## Run contract

- Experiment version: `v16.3.0`
- Direct baseline version: `v16.2.1`
- Final code commit: `6fd7272282a7143cff9801318983b21640d1e068`
- Source main commit: `44749815b3b6769071b424472938913f3feb3ec3`
- Source overlay SHA256: `8CDDC2325EDCDAAD311C71F8BFB11D1B5C8F566FE40743EA67A58A15549F343D`
- Source design SHA256: `2EA6DD67287083FE22FA235D77BA7347BDDAAEAB8B5971C50BAA12E4DB90CD19`
- Hardware: one NVIDIA GeForce RTX 4090
- Environment: Python 3.8.20, PyTorch 1.9.0+cu111, torchvision 0.10.0+cu111, CUDA 11.1
- Dataset: TAG-PEDES
- Training: 60 epochs, seed 1, batch size 64, random sampler, no optional image augmentation
- Backbone initialization: OpenAI CLIP ViT-B/16
- Mode: `--hire_v2 --hire_v2_mode identity_state`
- Identity settings: support size 3, auxiliary group weight 0.1, identity/state-final main weights 0.25/0.25
- State settings: text tokens 8, image tokens 16, rerank top-K 50

## Recorded training result

The checkpoint selector used `state_final` R1, as required by the v16.3.0 design.

| Status | Selected task | Best R1 | Best epoch | R5 | R10 | mAP | mINP |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| completed 60/60 | `state_final` | 57.61027526855469 | 47 | 75.866 | 82.332 | 43.893 | 23.788 |

At the selected Epoch 47 checkpoint, `identity_final` was R1 57.877, R5 76.260, R10 82.598, mAP 44.222, and mINP 24.157.

The highest logged `identity_final` R1 was 58.077 at Epoch 52, with R5 76.581, R10 82.689, mAP 44.437, and mINP 24.342. This value did not control checkpoint selection.

The final Epoch 60 evaluation was:

| Task | R1 | R5 | R10 | mAP | mINP |
| --- | ---: | ---: | ---: | ---: | ---: |
| `identity_final` | 58.065 | 76.351 | 82.768 | 44.468 | 24.376 |
| `state_final` | 57.568 | 76.188 | 82.471 | 44.095 | 24.021 |

## Training telemetry

- `state_pair_nce` fell from 3.8692 at Epoch 1 iteration 600 to 0.0930 at Epoch 47 and 0.0885 at Epoch 60.
- `state_positive_negative_margin` rose from 0.0096 to 0.2968 at Epoch 47 and 0.2998 at Epoch 60.
- `state_positive_coverage` remained 1.0.
- The learned state gate moved from -0.0015 to -0.2231 at Epoch 47 and -0.226714 at the final evaluation.
- The final valid support-group ratio was 98.64%, and the final mean support count was 2.7002.
- The final identity gate was 0.1018.
- The selected state-text/state-image token counts remained 8/16.

## Design acceptance

The state relation branch learned a separable same-image signal: its NCE decreased sharply, its positive-negative margin became positive, and positive coverage was complete. The resulting state reranking did not satisfy the method's minimum acceptance conditions:

- the final state gate was negative rather than greater than zero;
- `state_final` R1 and mAP were lower than `identity_final` at both the selected checkpoint and Epoch 60;
- the selected `state_final` R1 did not exceed the v16.2.1 best-checkpoint final R1 of 58.034;
- the run therefore does not qualify for additional random seeds under the design contract.

This is a method-level negative result, not a runtime failure. The evidence indicates that the local state branch learned trainable pair evidence, but its signed score contribution reduced the identity-based retrieval result.

## Pending best-checkpoint component evaluation

The required six-component and fix/break evaluation has not been fabricated or inferred from the training log. The 4090 was unmounted after training (`torch.cuda.is_available() == False`), so `tools/hire_v2/eval_identity_state_components.py` remains pending. Its future output must report `global`, `local`, `observation`, `identity`, `identity_final`, `state_final`, and state fix/break/net counts from the retained server-side best checkpoint.

## Validation

- The formal run started at 2026-07-18 02:42:12 CST and finished at 2026-07-18 07:31:45 CST with exit code 0.
- Runtime was 4 hours, 49 minutes, and 33 seconds.
- The log contains 60 validation blocks, `Epoch 60 done`, and the final best-R1 record.
- The server-side `best.pth` was present and 1,099,504,253 bytes.
- No traceback, runtime error, CUDA OOM, CUDA error, NaN, or Inf marker was found.
- The one-epoch smoke test and all eight static mathematical checks passed before the formal run.
- No model function, formula, loss, data flow, training flow, or evaluation calculation was changed after the supplied overlay was frozen.
- The checkpoint is intentionally not stored in GitHub.

## Metadata correction

The raw `run_record/run_manifest.txt` is retained unchanged. It mistakenly records the v16.2.1 design SHA256 (`44D776...`) in `source_design_sha256`. The supplied and committed v16.3.0 design file both hash to `2EA6DD67287083FE22FA235D77BA7347BDDAAEAB8B5971C50BAA12E4DB90CD19`. See `metadata_correction.md`.

## Archived-file hashes

| File | SHA256 |
| --- | --- |
| `train_log.txt` | `E635B2D77AC14322CCA545BB4FAE3A1ACC71D15C0E2EE85893768611FDDE3EEF` |
| `configs.yaml` | `FBEECD9D0CB7DB74BC367AEAD810F934A53C2419283074A1F52E1321FA170DE6` |
| `audit/static_audit.json` | `85513E50B0421D2463943A924489A3FA71D6F22EB6BBCC7702A0107C6B39FC5C` |
| `audit/smoke_acceptance.txt` | `DA71C4BAAE6A8589FAAF735410D3D06CC4E95F32F55DD2E9B603B9CD31FED725` |
| `run_record/nohup.log` | `459CFE477844081220715509C3776AC95EBB8D46F9A69E31682A408B26D8B617` |
| `run_record/run_manifest.txt` | `020F2283774F9B60616A8A1CF16EACAA5E9DED2073BCF5088CB24B4C26F9B9D2` |
| `run_record/command.sh` | `960BC657332306AECD2F662E9E11740528E1DDB25F0C4A54FEB1E899C4E4BA38` |

