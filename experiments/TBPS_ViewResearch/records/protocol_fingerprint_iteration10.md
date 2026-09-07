# Protocol fingerprint at primary iteration10

Checked 2026-09-07 before V012; resume helper returned `full_resume` and
`json_matches_tsv`. CUHK portrait ledger is at iteration3. V010/V011 are
already recorded, not pending results and not logged a second time.

- Baseline was measured before each dataset/geometry ledger was initialized.
- Every completed trial is verified/guarded and logged before its successor.
- Bundled helpers own TSV/state mutations and keep/stop decisions.
- Preserve immutable source/run history; discard is ledger-only, not a reset.
- Active foreground loop has no iteration limit. Stop requires the configured
  fixed-checkpoint R1 AND mAP test goal, user stop, or a genuine blocker.
- Two consecutive discards follow marginal V009; no escalation threshold is
  bypassed. Next hypothesis changes training stage, not another beta variant.

Checkpoint cleanup has separate explicit user approval. It does not erase
experiment history: local restore verification precedes exact server-file
deletion, and selected bests, source, logs and datasets are preserved.
