# V013 matched ten-epoch portrait E0

Source e45c55e. CUHK-PEDES, seed1, OpenAI initialization,384x128,
four GPUs,80 samples per rank, ten epochs and ten-epoch cosine schedule.
Maximum validation R1 selects epoch9: R1=72.00389862060547,
mAP=64.98136901855469. Final epoch10:71.79278564453125/64.91203308105469.
All2120 batches completed; five initial globally synchronized AMP skips,
none after epoch1. The eight-step four-GPU audit made three actual updates.
All27 source unit tests passed. The inherited implementation_version text
still says v009-cuhk-portrait-e0-5e; effective config, ten epoch records and
resumable checkpoint next_epoch=11 determine the actual executed horizon.

This measured baseline initializes a separate cuhk_portrait10 ledger. It
does not replace the five-epoch reference and is not a test result. The
V013 method starts independently from OpenAI with the same horizon, not
from this E0 checkpoint. No test selection or evaluation has occurred.

Compact best and completed records are persisted. Full archive transfer
and local restore verification are pending; no full last may be removed
until a corresponding verification receipt exists.
