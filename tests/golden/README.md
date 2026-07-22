# `tests/golden/` — GMAT golden fixtures

Versioned reference datasets that Polaris numerical functions are compared
against within documented per-quantity tolerance bands (design doc §23.1,
REQ-VV-002 / REQ-SYS-010).

These `*.json` files are **committed data**. The test suite (and CI) reads them
directly and never runs GMAT; GMAT is only used to *regenerate* a fixture, via
[`tools/gmat/`](../../tools/gmat/). Comparison harnesses live beside the fixture
they consume (e.g. `time_scales_golden_test.cpp` ↔ `time_scales.json`).

## Fixture schema (`schema_version: "1.0"`)

```jsonc
{
  "schema_version": "1.0",
  "case": "time_scales",              // fixture identifier
  "category": "time_conversion",      // GMAT V&V category
  "description": "...",
  "provenance": {
    "reference": "...",               // authoritative source of the values
    "gmat_regeneratable": true,       // can GMAT reproduce these?
    "gmat_script": "tools/gmat/...",  // the regenerating script
    "generated_utc": "YYYY-MM-DD"
  },
  "cases": [
    {
      "name": "gps_epoch_1980",
      "utc": { "year": 1980, "month": 1, ... },   // case-specific inputs
      "quantities": {
        "tai_minus_utc_s": { "expected": 19.0, "tol_abs": 1.0e-9, "unit": "s" }
      }
    }
  ]
}
```

A comparison test loads the fixture, computes each named quantity from the
Polaris function under test for each case's inputs, and asserts
`|actual − expected| ≤ tol_abs`.

## Fixtures

- `time_scales.json` — TAI/GPS/TT offsets vs UTC at canonical epochs; validates
  `lib/time`. Seeded from published constants (IERS leap seconds; IAU
  TT−TAI = 32.184 s, TAI−GPS = 19 s), GMAT-regeneratable.
- `gmat_propagation.json` — GMAT RungeKutta89 state histories for three force
  models (`two_body`, `zonal_j2`, `third_body`); cross-validates the Polaris
  propagator and force composite. Uses the state-history shape below rather than
  `quantities`.

### State-history shape (`gmat_propagation.json`)

Propagation cases carry `environment`, `spacecraft`, and `initial_state` blocks
plus a `samples` array, with one `position_tolerance_m` /
`velocity_tolerance_m_s` band per case and a `tolerance_rationale` explaining it:

```jsonc
{
  "name": "two_body",
  "epoch_utc": "2026-01-01T00:00:00Z",
  "environment": { "gravity_degree": 0, "third_bodies": [], "drag_enabled": false, ... },
  "initial_state": { "position_m": [...], "velocity_m_s": [...], ... },
  "position_tolerance_m": 0.05,
  "samples": [ { "t_s": 600.0000003841706, "position_m": [...], "velocity_m_s": [...] } ]
}
```

`t_s` is GMAT's *actual* reported `ElapsedSecs`, which overshoots the nominal
600 s grid by ~0.4 µs. Propagate to exactly these times — comparing at the round
number instead reintroduces the overshoot as apparent position error.
