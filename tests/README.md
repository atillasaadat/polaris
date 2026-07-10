# `tests/` — Verification & validation

Test pyramid (design doc §23.1): `unit/` (math/library) → `component/` (F´
components) → `integration/` (closed-loop SITL, incl. FDIR fault-injection) →
`regression/` → `golden/` (GMAT reference fixtures).

Tests annotate the requirement(s) they verify so traceability flows into the
Sphinx-Needs matrix:
- **Python:** `@pytest.mark.verifies("REQ-…")` + `record_property("margin_pct", …)`.
- **C++ (GoogleTest):** `RecordProperty("verifies", "REQ-…")` + `RecordProperty("margin_pct", …)`.

See `tests/conftest.py` and `tools/dev/collect_gtest_trace.py` for the collectors.
