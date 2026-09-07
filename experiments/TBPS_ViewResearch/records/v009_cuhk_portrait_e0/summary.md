# V009 CUHK portrait E0: measured validation baseline

Source `449b3ac`; own OpenAI initialization, 384x128 image input, seed1,
4x80 caption pairs, five epochs. No expert heads in this E0 stage. This is
a portrait S reference, not a claim of official 224x224 E0 equivalence.

| Epoch | Validation R1 | Validation mAP |
| --- | ---: | ---: |
| 1 | 59.142578 | 52.605011 |
| 2 | 64.777519 | 58.140625 |
| 3 | 68.918480 | 61.925274 |
| 4 | 71.451767 | 64.486267 |
| 5, selected | 72.848328 | 65.622398 |

Best validation R1 selects epoch5; mAP is from the same checkpoint. Relative
to original CUHK square E0(72.637215/65.507744), gains are +0.211113pp R1 and
+0.114655pp mAP. These are validation, not test metrics or an innovation win.
Continuation must use this measured portrait baseline, not the square score.

All1060 batches completed, five initial synchronized AMP skips, no later
skips, exit0, elapsed677.52s. The 25 targeted server tests passed; E0 startup
audit had3 actual updates out of8 batches. No original dataset files changed.

Full outputs initially used the guarded124GiB RAM filesystem. The complete
best/model and all logs/configs were then persisted on the server under
`/root/autodl-tmp/TBPS_ViewResearch_v009_20260907/e0_validation_run`.
The full archive, including last and best, was downloaded to
`D:/004SSH/TBPS_ViewResearch_v009_archives_20260907/e0_validation_run`.
Local checkpoint loading validated all302 model keys/shapes/dtypes,
selection consistency, AdamW moments, four-rank RNG, and transfer sizes.
Full last is1791177176 bytes; compact best598667789 bytes. No source deleted.

The separate CUHK-portrait ledger was initialized only AFTER this baseline
was measured. Test remains unopened. Full checkpoint binaries and smoke logs
are not included in the GitHub records.
