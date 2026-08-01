# `lib/gnc/` — GNC Algorithms

Estimation, guidance, and control math (design doc §8), shared by the FSW and
the analysis side. Everything here is pure math over the canonical
`EstimatedState` and frame-tagged vectors — no F´ types, no I/O, no globals —
so the F´ components that ship these algorithms are thin wrappers and the same
code is testable off-target.

| File | Role |
|---|---|
| `triad.{hpp,cpp}` | TRIAD deterministic two-vector attitude initializer + Shuster's covariance of the solution [black1964][markley2014][shuster1981]; the covariance is also exposed on its own (`triadCovariance`), since it is linear in the two variances and callers split their error budget across it; REQ-ADET-003 |
| `coarse_attitude.{hpp,cpp}` | Coarse SS+MAG+IMU attitude estimator: gyro propagation, TRIAD acquisition/update, fixed-gain complementary blend over a systematic covariance floor, eclipse coasting with a validity horizon [markley2014][wertz1978]; REQ-ADET-002 |

The coarse estimator is the **Safe-mode floor** (§10): with the analytic Sun
ephemeris fallback (§11.3) it needs neither a star tracker nor an uploaded
table, so it can always be run.

Tuning values (`min_sin_angle`, `triad_gain`, the per-source σ split) have **no
in-code defaults** — they are mission configuration (§19.3), and a
default-constructed config deliberately fails validation.

Flight-safe: fixed-size Eigen, no heap, no exceptions, no recursion, bounded
loops, return codes checked, finiteness checks on every published output.
