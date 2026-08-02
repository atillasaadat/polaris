# `tests/` — Verification & validation

Test pyramid (design doc §23.1): `unit/` (541 GoogleTest cases over `lib/` +
`sim/`) → F´ component tests, which live beside their component under
`flight/PolarisFsw/<Component>/test/ut/` rather than here (today: the
`AttitudeEstimator` harness, 30 cases) → `integration/` (29 cases: closed-loop
SITL, incl. FDIR fault-injection) → `golden/` (4 cases against GMAT/reference
fixtures — golden-fixture comparison *is* the regression tier).

`tools/` holds the Python suite for the ground-side tooling — 7 modules
(`test_config_compiler.py`, `test_gmat_drift.py`, `test_gmat_golden.py`,
`test_gmat_propagation.py`, `test_orbit.py`, `test_prmdb.py`,
`test_spaceweather.py`), 126 collected: 124 pass and 2 skip without a local
GMAT install.

Tests annotate the requirement(s) they verify so traceability flows into the
Sphinx-Needs matrix:
- **Python:** `@pytest.mark.verifies("REQ-…")` + `record_property("margin_pct", …)`.
- **C++ (GoogleTest):** `RecordProperty("verifies", "REQ-…")` + `RecordProperty("margin_pct", …)`.

See `tests/conftest.py` and `tools/dev/collect_gtest_trace.py` for the collectors.
