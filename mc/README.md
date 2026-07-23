# `mc/` — Monte Carlo framework

> **Status: planned, not yet implemented** — this directory is a placeholder; the
> framework below is the design intent (§13), built once the SITL loop (Phase 3)
> gives it something to disperse.

Dispersion-driven verification across the full stack (design doc §13): seeded and
reproducible from `{config, seed}`, parallel execution, per-run config capture.
**Pass/fail with margin** (pointing percentiles, settling time, momentum/power
margins) reported against requirement thresholds, and estimator-consistency
(NEES/NIS) aggregated across runs. Runs as a CI smoke subset and as full off-line
campaigns.
