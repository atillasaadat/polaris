# `mc/` — Monte Carlo framework

Dispersion-driven verification across the full stack (design doc §13): seeded and
reproducible from `{config, seed}`, parallel execution, per-run config capture.
**Pass/fail with margin** (pointing percentiles, settling time, momentum/power
margins) reported against requirement thresholds, and estimator-consistency
(NEES/NIS) aggregated across runs. Runs as a CI smoke subset and as full off-line
campaigns.
