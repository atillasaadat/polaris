# `lib/gnc/` — GNC Algorithms

Estimation, guidance, and control math (design doc §8), shared by the FSW and
the analysis side. Everything here is pure math over the canonical
`EstimatedState` and frame-tagged vectors — no F´ types, no I/O, no globals —
so the F´ components that ship these algorithms are thin wrappers and the same
code is testable off-target.

| File | Role |
|---|---|
| `triad.{hpp,cpp}` | TRIAD deterministic two-vector attitude initializer + Shuster's covariance of the solution [black1964][markley2014][shuster1981]; the covariance is also exposed on its own (`triadCovariance`), since it is linear in the two variances and callers split their error budget across it; REQ-ADET-003 |
| `davenport.{hpp,cpp}` | Davenport's q-method: the optimal attitude from N weighted vector pairs (Wahba's problem via the 4×4 K matrix) with the inverse-Fisher covariance, gated on observability in both frames [davenport1968][markley2014][shuster1981]; MEKF cold start/re-init only; REQ-ADET-003 |
| `coarse_attitude.{hpp,cpp}` | Coarse SS+MAG+IMU attitude estimator: gyro propagation, TRIAD acquisition/update, fixed-gain complementary blend over a systematic covariance floor, eclipse coasting with a validity horizon [markley2014][wertz1978]; REQ-ADET-002 |
| `mekf.{hpp,cpp}` | 6-state multiplicative EKF on `[δθ; δb]`: closed-form propagation with the exact `Φ₁₂`, Farrenkopf discrete `Qd`, vector measurements one at a time with per-update NIS gating, Joseph-form update, exact multiplicative reset [lefferts1982][markley2014][farrenkopf1978][barshalom2001]; REQ-ADET-001 |
| `albedo_correction.{hpp,cpp}` | Closed-form inverse of the §6.2 Earth-albedo pull on a measured sun vector: the modelled directed offset toward the sunlit Earth rotated back out, credited only the fraction of itself the centroid model can honestly claim (`albedo_dispersion_fraction`, §19.2) [bhanderi2005][wertz1978][markley2014]; refuses — leaving the measurement untouched, for the caller to weight at the *uncorrected* σ — on missing or degenerate geometry rather than correcting on an invented nadir; `tests/unit/albedo_correction_test.cpp` sweeps the geometry |
| `mag_calibration.{hpp,cpp}` | Attitude-free magnetometer hard/soft-iron calibration: streaming O(1) ten-parameter ellipsoid fit of the raw field against the onboard IGRF **magnitude**, symmetric PD square root for the soft-iron correction, orientation-coverage/conditioning/positive-definiteness/residual-improvement gates [alonso2002complete][alonso2002twostep][vasconcelos2011][higham2008], each refusal naming its gate through the `MagCalibrationRefusal` overload of `solve`; the §8.1 `MAG_CAL_START` flow on `flight::AttitudeEstimator` runs it |

The coarse estimator is the **Safe-mode floor** (§10): with the analytic Sun
ephemeris fallback (§11.3) it needs neither a star tracker nor an uploaded
table, so it can always be run.

Tuning values (`min_sin_angle`, `triad_gain`, the per-source σ split) have **no
in-code defaults** — they are mission configuration (§19.3), and a
default-constructed config deliberately fails validation.

Flight-safe: fixed-size Eigen, no heap, no exceptions, no recursion, bounded
loops, return codes checked, finiteness checks on every published output.
