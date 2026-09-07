# V008: gallery flip averaging, fixed CUHK checkpoints

Source `4eeaeb0`; no optimizer update, no checkpoint write, validation only.
Each image uses its original trained route and a horizontally mirrored route;
normalize the sum of their unit embeddings. Text and cosine scoring unchanged.

| Source | Validation R1 | Validation mAP |
| --- | ---: | ---: |
| E0 plain, epoch5 | 72.637215 | 65.507744 |
| E0 flip average | 73.156868 | 66.068443 |
| V004 plain, epoch4 | 72.815849 | 65.565125 |
| V004 flip average | 72.750893 | 65.993622 |

Both plain references reproduced exactly, not merely within the 0.0001pp
tolerance. All 3078 validation images / 6158 captions were covered in order;
finite unit embeddings and unchanged source checkpoint metadata passed.
Four focused tests passed both locally and in the pinned server environment.

V004 flip gains +0.113678pp R1 / +0.485878pp mAP against original E0, but
loses -0.405975pp / -0.074821pp against identically flip-averaged E0. Against
its own plain score, R1 decreases -0.064957pp. Joint gain +0.113678pp is below
the existing CUHK numeric incumbent +0.129913pp. Discard this inference change.
Its generic augmentation gain must not be attributed to the expert design.

V004's fixed selected epoch4 has actual decorrelation updates; this is not
the earlier pre-decorrelation epoch1 selection. No test was evaluated.
E0 and V004 inference took about24.2 and23.3s respectively, exit0. Previous
training code and all checkpoint files remain untouched.
