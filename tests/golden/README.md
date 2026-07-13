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
