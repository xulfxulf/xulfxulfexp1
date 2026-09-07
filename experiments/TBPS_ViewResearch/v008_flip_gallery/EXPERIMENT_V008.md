# V008: fixed-checkpoint gallery flip averaging

This directory independently contains the complete V004 training/inference
implementation. V008 changes only the standalone evaluation path. Do not run
the inherited training launcher for this experiment.

- Dataset/split: CUHK-PEDES validation only. Seed 1.
- Fixed sources: five-epoch E0 best epoch 5 and V004 best epoch 4. V004 has
  actual shared-consistency and residual-decorrelation updates.
- Encode each gallery image at the original 224x224 evaluation geometry and
  encode its horizontal mirror with the SAME model and F/S/B probabilities.
  The merged side probability is invariant to a horizontal mirror.
- The original and mirror each run the complete trained residual route.
  Normalize their sum in FP32. Text uses its unchanged single normalized
  representation. Scores remain text-gallery dot products.
- No training, new losses, new weights, checkpoint writes, query-side view
  information, support/gallery identity propagation, or test evaluation.
- Reproduce both recorded plain validation R1 and mAP within 0.0001 percentage
  point before accepting each flip result. Preserve gallery alignment and
  require finite, unit-length embeddings.
- Report the method against BOTH plain E0 and flip-averaged E0, as well as
  its original V004 score. This prevents crediting a generic augmentation
  effect exclusively to the residual mechanism.
- OEFormer probabilities are accepted without manual calibration, per user;
  entropy confidence must continue to be called uncalibrated.
- Use the pinned torch 1.13.0/cu117 environment and original numeric backends.
  A global GPU lock prevents overlapping this inference with research jobs.

```bash
python run_flip_evaluation.py \
  --config /absolute/code/root/configs/evaluation_v008.json \
  --output-dir /absolute/new/output/directory
```

Only compact metrics/configs/manifests are written. Every source checkpoint
remains read-only. A failed reproduction check invalidates the experiment;
do not silently widen the tolerance or replace the checkpoint.
