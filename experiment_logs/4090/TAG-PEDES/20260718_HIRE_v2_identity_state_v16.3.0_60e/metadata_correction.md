# Run Manifest Metadata Correction

The raw `run_record/run_manifest.txt` is retained byte-for-byte as produced before the formal run.

Its `source_design_sha256` field contains:

```text
44D776E37D8A94DB7D26434B96A7A56E6438FE6C01A6F1F3E956F73C03CAEF15
```

That value belongs to `HIRE_v2_identity_balanced_design_20260717.md` (v16.2.1), not the v16.3.0 identity-state design.

Both the supplied file `HIRE_v2_identity_state_design_20260717.md` and the committed file `HIRE_V2_IDENTITY_STATE_DESIGN.md` hash to:

```text
2EA6DD67287083FE22FA235D77BA7347BDDAAEAB8B5971C50BAA12E4DB90CD19
```

The result summary uses the corrected v16.3.0 hash. No source code, model state, training configuration, metric, or raw run record was changed.

