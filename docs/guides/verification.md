# Verification & Golden Data

Polaris is verified at four levels — unit → integration → golden cross-validation
→ (future) Monte Carlo — with the numerical core checked against **GMAT** golden
fixtures within documented per-quantity tolerance bands (REQ-VV-002).

## The test pyramid

- **Unit** (`tests/unit/`) — each model in isolation, run under
  AddressSanitizer/UBSan. Conversions, physics, the RNG determinism contract.
- **Integration** (`tests/integration/`) — the assembled truth sim in one
  process: the composed force model + RK89 plant, and the {doc}`closed_loop`
  driving sensors and actuators around a full orbit with attitude.
- **Golden** (`tests/golden/`) — Polaris vs GMAT reference states within
  tolerance bands that are *model-difference budgets*, not propagator error.
- **Python** (`tests/tools/`) — the config compiler and the fixture generators.

## GMAT cross-validation

GMAT (NASA GSFC) generates versioned reference fixtures, committed under
`tests/golden/`. CI never runs GMAT — it replays the committed fixtures; GMAT is
only used to *regenerate* them (`tools/gmat/`, {doc}`/api/python`). The orbit
regime matrix spans two-body through GEO through a Molniya HEO, plus an attitude
spinner cross-checking the quaternion kinematics. The per-case bands and their
rationale live in `tools/gmat/README.md`.

## The documentation gate

This site is itself a CI gate: `sphinx-build -W` fails on a broken docstring, a
citation with no `refs.bib` entry, or a **baselined requirement with no verifying
test** (sphinx-needs traceability, design doc §22.2). Documentation drift is a
build failure, not an afterthought.
