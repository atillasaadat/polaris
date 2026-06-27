# `tools/` — Dev, CI & data-generation tooling

- `configc/` — the **config compiler**: resolves hardware model-IDs, validates
  against schema, emits F´ params + sim setup + analysis inputs from the single
  source-of-truth config (design doc §19.3).
- `gmat/` — **GMAT golden-data harness**: pinned installer, script generator, and
  runner that produce versioned reference fixtures in `tests/golden/` (design doc
  §23.1). Uses NASA GMAT R2026a, headless.
- `dev/` — developer/CI helper scripts (docs build, traceability collectors).
