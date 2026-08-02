# `tools/` — Dev, CI & data-generation tooling

- `configc/` — the **config compiler**: resolves hardware model-IDs, validates
  against schema, emits F´ params + sim setup + analysis inputs from the single
  source-of-truth config (design doc §19.3).
- `gmat/` — **GMAT golden-data harness**: pinned installer, script generator, and
  runner that produce versioned reference fixtures in `tests/golden/` (design doc
  §23.1). Uses NASA GMAT R2026a, headless.
- `dev/` — developer/CI helper scripts (docs build, traceability collectors).

The rest are the ground-side fetch/derive tools for the committed reference data
(design doc §3.7) — each is a `python -m <pkg>` CLI, run by hand, never in CI, and
each writes its upstream file **verbatim** into `tests/golden/`:

- `eop/` — IERS `finals.all.iau2000` Earth-orientation download (mirror fallback)
  with a parse gate.
- `ephem/` — JPL **DE440** `.bsp` → geocentric Sun/Moon **Chebyshev** fit
  (`de440_bodies.cheb`), the fixture `sim/world/ephemeris_file` parses. The kernel
  is a fetch input, never committed.
- `gravity/` — EGM2008 `.gfc` download and truncation to the committed
  degree-200 window, in the native format.
- `igrf/` — IGRF-14 IAGA coefficient download, plus `golden` regeneration of the
  reference field fixture with the IAGA implementation.
- `spaceweather/` — CelesTrak `SW-All.csv` download with a parse gate.
