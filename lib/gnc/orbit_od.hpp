#ifndef POLARIS_GNC_ORBIT_OD_HPP
#define POLARIS_GNC_ORBIT_OD_HPP

/// @file
/// @brief Onboard orbit determination — a 6-state EKF on ECI position/velocity
/// driven by GNSS fixes, with self-covariance propagation (design doc §8.3).
///
/// The orbit half of the canonical `EstimatedState` (§8.0). Where the attitude
/// MEKF (@ref Mekf) carries the vehicle's orientation, this carries where it is:
/// a **GNSS-aided** filter that propagates position and velocity on an onboard
/// force model between fixes and corrects them when a fix arrives.
///
/// **Error state** (6): `x = [δr; δv]`, ECI position [m] and velocity [m/s],
/// laid out in the same order as `state::ErrorState::kPosition` / `kVelocity`,
/// so the 15×15 canonical covariance takes this filter's blocks verbatim. The
/// error is **additive** — a translational state has no group structure and no
/// unit-norm constraint, so there is nothing for the attitude filter's
/// *multiplicative* reset to do here, and the "MEKF" in §8.3's heading refers to
/// this being one block of the same 15-state error-state filter, not to a
/// multiplicative parameterisation of position.
///
/// ## The force model is coarse, but no longer J2-only
///
/// **A degree/order 8 EGM2008 truncation + exponential-density drag.** This is
/// still far below the truth sim's 200×200 EGM2008 / NRLMSIS / third-body / SRP
/// stack (§5.2), and that gap is the design, not an omission (§18,
/// `sim/CLAUDE.md`: truth must differ from onboard, deliberately).
///
/// The filter originally flew closed-form two-body + J2, on the argument that
/// GNSS dominates the solution and a better field buys nothing. That argument
/// holds *for the estimate* and is unchanged: with 1 Hz fixes at ~1 m the
/// steady-state error is receiver-bound, and no force model moves it. It does
/// not hold for the two things the dynamics is actually load-bearing for —
/// **coasting through an outage**, and **onboard ephemeris prediction** (§20
/// pass planning, eclipse prediction, maneuver targeting) — where the J2-only
/// truncation was the whole error budget: **3.17 m** over the 300 s coast
/// horizon against a 32x32 truth, and **14.3 m** over 699 s against GMAT, led by
/// the degree-3+ zonals and the tesserals. The same arcs with this field:
/// **1.28 m** and **1.2 cm**.
///
/// The objection the original argument raised against a table was cost, and the
/// measured cost is small enough to overrule it: 8×8 is 45 coefficient pairs
/// compiled in (`gnc/egm2008_low_degree.hpp`, ~1 kB of `.rodata`, no upload and
/// so no staleness or CRC story), evaluated by the unnormalized Cunningham V/W
/// recursion with `(N+2)²` doubles of stack scratch and no precomputed tables
/// (`gnc/geopotential.hpp`). It is not evaluated in the 10 Hz GNC cycle — the
/// propagation runs at the fix rate and its sub-steps.
///
/// What survives from the original argument is the third point, and it still
/// does the real work: **the remaining truncation is absorbed as process
/// noise**, which is what an error budget is for.
/// `OrbitOdConfig::accel_psd_m2_per_s3` is sized *from the measured* divergence
/// between this model and the full-fidelity one — see "Process noise" below —
/// rather than tuned by feel, so improving the field re-derives `q_a` rather
/// than leaving a now-conservative value in place.
///
/// Degree 0 restores the two-body path and `zonal_j2` the closed-form J2 one;
/// that is how the golden test compares this propagator against GMAT **at
/// matched fidelity**, and the two paths are mutually exclusive by config
/// validation.
///
/// **The field is evaluated in ECEF, per integration sub-step.** A geopotential
/// is defined in the Earth-fixed frame, and ECI (J2000 mean equator) is not it.
/// For the *zonal* fallback path only the pole direction matters, since a zonal
/// field is axisymmetric — but even there ECI must not be used naively:
/// precession and nutation separate the two poles by ~0.36° by 2026, tilting the
/// J2 bulge by that angle at ~100 m per revolution in LEO. That exact defect was
/// found in the truth sim by GMAT cross-validation and is documented in
/// `sim/world/gravity_field.hpp`; it is not repeated here or there.
///
/// The harmonic path cannot take that shortcut at all. A **tesseral** term is
/// fixed to the rotating Earth, so the full ECI↔ECEF rotation is resolved from
/// the same `frames::eci_ecef` reduction every other transform in the repo uses
/// (REQ-CONV-002) — and resolved **per sub-step**, not once per propagation:
/// Earth turns 0.004°/s, so holding one rotation across a 60 s step would smear
/// the tesserals by 0.25° of longitude. Within a sub-step it is held across the
/// four RK4 stages, where the 6e-5 rad of rotation over `h ≤ 1 s` mis-orients a
/// ~1e-5 m/s² term by ~6e-10 m/s², orders below the truncation itself.
///
/// **Drag is a static exponential atmosphere** — `ρ = ρ₀ exp(−(h−h₀)/H)` on a
/// spherical altitude, against the atmosphere-relative velocity `v − ω⊕ × r`
/// (Vallado §8.6.2 / §8.1 [vallado2013]; Montenbruck & Gill §3.5
/// [montenbruck2000]). It carries no solar or geomagnetic activity, so it is
/// wrong by a factor of a few at solar maximum. That is affordable precisely
/// because drag is ~1e-7 m/s² at 400 km: a factor-two density error over the
/// coast horizon is millimetres, three orders below the geopotential truncation
/// it sits beside. The term is here for the *secular* along-track effect over
/// long outages, not for accuracy between fixes, and it is the hook the §8.5
/// tier-3 drag scale factor will eventually estimate.
///
/// Any of the three terms is disabled by setting its coefficient to zero
/// (`zonal_j2`, `drag_ballistic_coeff_m2_per_kg`), which is how the golden test
/// compares this propagator against GMAT **at matched fidelity**.
///
/// ## Propagation and the state transition
///
/// The state is integrated with classical **RK4** over sub-steps of at most
/// `OrbitOdConfig::max_step_s`, bounded by @ref OrbitOd::kMaxSubsteps so the
/// loop is bounded at compile time (§3.6). The covariance is propagated
/// `P = ΦPΦᵀ + Q` per sub-step, with
/// \f[
///   \Phi(h) = I + F h + \tfrac{1}{2} F^2 h^2, \qquad
///   F = \begin{bmatrix} 0 & I
///     \\ \partial a/\partial r & \partial a/\partial v \end{bmatrix},
/// \f]
/// the second-order truncation of `exp(Fh)` (Montenbruck & Gill §7.1
/// [montenbruck2000]). The next term is `O((‖F‖h)³)`; `‖F‖ ≈ n = 1.1e-3 rad/s`
/// in LEO, so at `h ≤ 1 s` it is ~2e-10 relative — below every other error in
/// the covariance by orders.
///
/// **The Jacobian is numerical, by central differences, and that is a choice.**
/// The upper two blocks of `F` are exact (`∂ṙ/∂r = 0`, `∂ṙ/∂v = I`); only
/// `∂a/∂r` and `∂a/∂v` are differenced, at twelve acceleration evaluations per
/// sub-step. The analytic alternative — the two-body gravity gradient is one
/// line, but the J2 gradient is seven terms of product rule and the drag
/// gradient carries the density derivative — is a **second implementation of the
/// force model** that has to be kept in step with the first by hand. A
/// numerical Jacobian differentiates whatever `onboardAcceleration` actually
/// computes, so it cannot drift from the flown model when a term is added or a
/// coefficient changes; and the accuracy question it raises is answered by test
/// rather than by argument (`tests/unit/orbit_od_test.cpp` pins it against the
/// closed-form two-body gradient `−μ/r³(I − 3r̂r̂ᵀ)` to 1e-5 of the gradient's own
/// scale — a bound set by the `O(h)` truncation of `Φ ≈ I + Fh` in the test's
/// extraction, not by the difference stencil, which is finer still). The cost
/// is ~350 flops per sub-step, against a 10 Hz budget. Where a numerical
/// derivative usually loses — noisy or discontinuous functions, ill-chosen steps
/// — none applies: the acceleration is smooth and analytic, and the step is
/// scaled to the argument.
///
/// ## Process noise
///
/// `Q` is the standard **discretised continuous white-noise acceleration**
/// (CWNA) model (Bar-Shalom §6.2.2 [barshalom2001]; Montenbruck & Gill §8.2
/// [montenbruck2000]):
/// \f[
///   Q_d(h) = q_a \begin{bmatrix} \tfrac{1}{3}h^3 I & \tfrac{1}{2}h^2 I
///     \\ \tfrac{1}{2}h^2 I & h\,I \end{bmatrix},
/// \f]
/// with `q_a = accel_psd_m2_per_s3` [m²/s³]. The off-diagonal blocks are kept
/// for the same reason the attitude filter keeps its cross terms: the position
/// and velocity errors produced by one unmodelled acceleration are the *same*
/// error seen twice, and a diagonal shortcut makes the filter mis-weight the
/// velocity measurement against a position it thinks is independent of it.
///
/// **How `q_a` is set.** Not by feel: from the measured truncation. The CWNA
/// model gives a position spread `σ_r(T) = √(q_a T³/3)` over a coast of length
/// `T`, so matching it to the divergence `δr(T)` between this force model and
/// the full-fidelity one at the **coast horizon** gives
/// \f[
///   q_a = \frac{3\,\delta r(T)^2}{T^3} .
/// \f]
/// `δr(T)` is measured, not assumed — `tests/unit/orbit_od_test.cpp` runs this
/// propagator against the GMAT-validated truth sim over the horizon and asserts
/// the result stays under the value `q_a` was sized from, so a force-model
/// regression fails CI rather than quietly invalidating the tuning. For the
/// reference vehicle (12 kg, 400 km, 51.6°) that measurement is **1.28 m over
/// 300 s**, carried at 1.8 m with ~40% margin, giving
/// `q_a = 3.6e-7 m²/s³`. Design doc §8.3 records both.
///
/// **The reference has to stay above the model, and that is a live constraint,
/// not a footnote.** This measurement used to read 3.59 m against an *8x8*
/// truth — which the onboard model now matches, and an 8x8-vs-8x8 comparison
/// reports 0.045 m: a passing test that asserts nothing, because it is measuring
/// two implementations of one field rather than a truncation. The truth degree
/// was raised to 32x32 rather than the smaller number banked. Any future
/// increase in `geopotential_degree` has to check the same thing before trusting
/// the number this paragraph quotes.
///
/// State the approximation honestly: the truncation is a *systematic*, not white
/// noise — and the same measurement says so, since it grows as `t²` (0.0138 m at
/// 30 s against 1.280 m at 300 s) where white noise would grow as `t^{3/2}`.
/// Matching a white model to it **at** `T` therefore makes the filter
/// conservative for `t < T` and optimistic beyond `T`, which is exactly why the
/// solution is declared invalid at `T` rather than left to coast on a covariance
/// that has stopped covering its own error.
///
/// ## Measurements
///
/// A GNSS receiver reports a complete PVT solution in **ECEF and GPS time**, so
/// @ref OrbitOd::ingest owns the §3.2/§6.2 conversion before anything inertial
/// runs (REQ-CONV-001): `TAI = GPS + 19 s` through `time::toTai`, and the ECEF
/// **state** (not just the position — velocity picks up the `ω⊕ × r` transport
/// term, a ~465 m/s error if skipped) through `frames::eciStateFromEcef`. A fix
/// whose epoch lies outside the uploaded EOP table's span cannot be converted
/// and is **refused**, not extrapolated.
///
/// ### Fix latency
///
/// A receiver's solution is valid at its **measurement epoch** but reaches the
/// FSW a fix latency later — tens of milliseconds of internal processing, plus
/// bus transport and scheduling. At 7.6 km/s that interval is ~7.6 m of
/// along-track position per millisecond, which makes it the largest single error
/// this filter can carry: an order above the receiver's own ~1 m noise, and two
/// above the geopotential truncation the harmonic field was added to remove.
///
/// It is corrected on the **measurement**, not the filter. When a fix arrives
/// tagged behind the filter's current epoch — the normal case, since the GNC
/// cycle has propagated in the meantime — the reported PVT is advanced to the
/// filter's epoch on the receiver's *own reported velocity* plus the onboard
/// acceleration,
/// \f[
///   r \mathrel{+}= v\,\tau + \tfrac12 a\,\tau^2, \qquad v \mathrel{+}= a\,\tau,
/// \f]
/// and `R` is inflated by `(σ_v τ)²` for the velocity error that integrates into
/// position across the advance. This is the benign corner of Bar-Shalom §5.6's
/// out-of-sequence-measurement problem: the measurement determines the whole
/// state and carries its own velocity, so the linear term is *measured* rather
/// than modelled and the residual is `O(τ³·jerk)` — micrometres at `τ ≤ 0.2 s`.
/// It is the solution-domain analogue of the signal-transit-time correction Kim
/// et al. apply to navigation-solution measurements [kim2025].
///
/// Retrodicting the *filter* instead (rolling the state back to the fix epoch,
/// updating, re-propagating) is the textbook-exact alternative and is not worth
/// it here: it costs a stored prior and two extra propagations per fix to fix an
/// error the measurement's own velocity already resolves to micrometres.
///
/// The correction is bounded by `OrbitOdConfig::max_fix_latency_s`. Past that a
/// fix is refused as before, because a tag arbitrarily far in the past is a
/// clock fault rather than a latency, and a fix without a velocity is refused
/// too — the filter's own velocity is not a substitute, since using it would
/// fold the filter's error into a measurement that has to stay independent of
/// it. The realised latency is reported on every fix through
/// @ref OrbitOdResult::fix_latency_s, so a latency that climbs is visible in
/// telemetry rather than silently absorbed.
///
/// Position and velocity are folded in as **two sequential 3-row updates**
/// rather than one 6-row batch. The receiver draws them independently, so `R` is
/// block-diagonal and sequential processing is algebraically identical to the
/// batch form — but it keeps every matrix a fixed 3×3 or 6×3, gives each its own
/// NIS and its own gate, and lets a fix without a velocity report take the
/// position path unchanged. Both use the **Joseph form**, which stays symmetric
/// positive-definite under round-off and a suboptimal gain.
///
/// **`R` is the receiver's own reported accuracy, per fix, and it is
/// anisotropic.** `sim/sensors/gnss` realises position error split horizontal vs
/// vertical in the local geodetic frame and carries the realised σ on every
/// measurement, so the filter reconstructs
/// `R = A_eci←ecef · E diag(σ_h², σ_h², σ_v²) Eᵀ · A_eci←ecefᵀ` with
/// `E = [ê n̂ û]` at the fix — the receiver's own covariance rotated into ECI —
/// rather than flattening it to a scalar. Velocity error is per-axis in ECEF, so
/// `σ_v²I` is rotation-invariant and passes through.
///
/// As in the attitude filter, `R` is treated as **white**, and real GNSS error
/// is not: ionosphere, broadcast-orbit and clock error are common-mode across
/// the visible constellation and correlated over minutes (§6.2). A filter
/// averaging successive fixes against a white `R` therefore reports a covariance
/// below its true error. The truth model is white too today, so the NEES
/// campaign that validates this filter validates it against the model it was
/// given; when the correlated component lands in the sensor model (§6.2's stated
/// upgrade path) this filter needs either an inflated `R` or a state-augmented
/// bias, and neither is pretended here.
///
/// ## Consistency, refusal, and the coast horizon
///
/// Every update reports its innovation, `S = HPHᵀ + R`, and the **NIS** `yᵀS⁻¹y`
/// (~χ²₃ when consistent). A measurement outside the gate is **rejected** —
/// counted in @ref OrbitOd::rejectedCount, not applied. The gate is written as
/// an accept range `0 ≤ NIS ≤ gate`: a one-sided "too large?" test waves a
/// *negative* NIS straight through, and a negative NIS means the covariance has
/// gone indefinite, which is the one thing a divergence guard must not miss.
/// @ref OrbitOd::nees is the analysis-side counterpart against a known truth
/// (Bar-Shalom §5.4 [barshalom2001]); there is no truth onboard.
///
/// Every refusal names itself through @ref OrbitOdRefusal rather than returning
/// a bare false. Unconfigured, uninitialised, a non-finite fix, a fix outside
/// the configured geocentric-radius band (§9.1 — wire data is not trusted), a
/// non-positive σ, a **non-increasing fix epoch** (backwards is obvious; a
/// *stuck* clock is the dangerous one, since re-running the update at the same
/// epoch folds the same measurement in twice and shrinks the covariance on
/// information already used), a step longer than `max_dt_s`, a failed frame
/// conversion: all refuse without touching the solution. A non-finite internal
/// result is unrecoverable and drops the filter to cold start, leaving
/// @ref OrbitOd::rejectedCount standing — a filter that has just diverged is
/// when FDIR most needs to see what it had been rejecting on the way there.
///
/// Past `max_coast_s` without an accepted fix the solution is declared invalid
/// **and dropped**, and the next fix re-acquires whole. This is the coarse
/// attitude estimator's discipline, not the attitude MEKF's: the MEKF retains
/// its state past the horizon because a Kalman gain against a grown covariance
/// takes the returning measurement almost whole and there is a converged *gyro
/// bias* worth keeping. Here there is no such parameter — the whole state is
/// position and velocity, a GNSS fix determines both outright and better than
/// any coasted prior, and the coarse force model's error past the horizon is
/// systematic, so blending against a stale prior would drag the fresh fix
/// toward a known-wrong trajectory for no gain.
///
/// **Sizing the horizon.** It covers the outage modes the receiver model already
/// has (§6.2): the cold-start time-to-first-fix, the post-outage reacquisition
/// delay, and brief dropouts — the same graceful-coasting case the §9.2 FDIR
/// suite drives. It is not sized to survive a long GNSS loss, because this force
/// model cannot: see the `q_a` derivation above for why the honest end of a
/// coast is an invalidity flag rather than a slowly-worsening answer. The
/// configured value and the receiver figures it was set from are in design doc
/// §8.3.
///
/// **Frames, units, conventions.** Position/velocity `Vec3<ECI>` [m], [m/s];
/// fixes arrive as `Vec3<ECEF>` in GPS time; times TAI (§3.2); covariance m² /
/// m²s⁻² blocks; all SI. Flight path: fixed-size Eigen, no heap, no exceptions,
/// no recursion, bounded loops, return codes checked, finiteness-guarded output.
/// No F´ types and no I/O.
///
/// References:
///  - Tapley, Schutz & Born, *Statistical Orbit Determination*, 2004, §4
///    (sequential/extended filter for orbit determination; state transition of
///    the linearised two-body dynamics). [tapley2004]
///  - Montenbruck & Gill, *Satellite Orbits*, 2000, §3.2 (zonal geopotential),
///    §3.5 (atmospheric drag), §7.1 (variational equations and Φ), §8.2
///    (dynamic model compensation / process noise for onboard OD).
///    [montenbruck2000]
///  - Vallado, *Fundamentals of Astrodynamics and Applications*, 4th ed., §8.6.2
///    (exponential atmosphere model and its scale-height table), §3.7 (ECEF↔ECI
///    state transformation). [vallado2013]
///  - Bar-Shalom, Li & Kirubarajan, *Estimation with Applications to Tracking
///    and Navigation*, 2001, §5.4 (NEES/NIS consistency), §5.6
///    (out-of-sequence measurements), §6.2.2 (the discretised continuous
///    white-noise acceleration model). [barshalom2001]
///  - Kim, Lee & Park, "Enhanced Real-Time Onboard Orbit Determination of LEO
///    Satellites Using GPS Navigation Solutions with Signal Transit Time
///    Correction", Aerospace 12(6), 2025 (correcting a navigation-solution
///    measurement to the epoch it is applied at). [kim2025]
///  - Cunningham, "On the computation of the spherical harmonic terms needed
///    during the numerical integration of the orbital motion of an artificial
///    satellite", Celestial Mechanics 2, 1970 (the V/W recursion the onboard
///    field is evaluated by). [cunningham1970]

#include <cstdint>
#include <Eigen/Core>

#include "frames/eci_ecef.hpp"
#include "frames/eop.hpp"
#include "gnc/geopotential.hpp"
#include "math/frames.hpp"
#include "math/typed_vector.hpp"
#include "state/estimated_state.hpp"
#include "time/timescales.hpp"

namespace polaris::gnc {

/// Tuning and force-model coefficients for @ref OrbitOd. Every value is mission
/// configuration (§19.3) — there are no flight defaults worth trusting, so
/// @ref isValid gates the constructor and an invalid config leaves the filter
/// permanently inert.
struct OrbitOdConfig {
  /// @name Force model
  /// @{

  /// Gravitational parameter GM [m³/s²]. Must be positive. Set it from
  /// `constants::wgs84::kGM`; it is a config field rather than a hard-coded
  /// constant only so the golden test can pin the propagator against a reference
  /// tool's own μ.
  double mu_m3_per_s2{0.0};
  /// Harmonic degree of the compiled-in EGM2008 truncation to evaluate
  /// (`lib/gnc/geopotential.hpp`). **Zero disables the harmonic field** and falls
  /// back to the closed-form two-body + @ref zonal_j2 path below, which is what
  /// the GMAT matched-fidelity golden cases use. Must be in
  /// `[0, kGeopotentialMaxDegree]`.
  ///
  /// When non-zero the field supplies its own point-mass term and its own GM and
  /// reference radius, @ref zonal_j2 must be zero (the field already contains the
  /// degree-2 zonal), and @ref mu_m3_per_s2 / @ref reference_radius_m must agree
  /// with the table's constants to 1e-6 relative — see `isValid`.
  int geopotential_degree{0};
  /// Harmonic order to evaluate. Must be in `[0, geopotential_degree]`. Order 0
  /// is a purely zonal field; the tesserals are what require the full ECEF
  /// rotation per sub-step.
  int geopotential_order{0};

  /// Unnormalized zonal harmonic J2 [-]. Set it from `constants::gravity::kJ2`.
  /// **Zero disables the term** (pure two-body), which is what the golden
  /// two-body case uses. Must be finite and non-negative, and must be zero when
  /// @ref geopotential_degree is non-zero.
  double zonal_j2{0.0};
  /// Reference radius the zonal coefficient is scaled to [m]. Must be positive
  /// when `zonal_j2 > 0`; `constants::wgs84::kSemiMajorAxis` is the pairing the
  /// coefficient above was solved with.
  double reference_radius_m{0.0};
  /// Ballistic coefficient `C_d·A/m` [m²/kg]. **Zero disables drag.** Must be
  /// finite and non-negative.
  double drag_ballistic_coeff_m2_per_kg{0.0};
  /// Reference density ρ₀ [kg/m³] of the exponential atmosphere, at
  /// @ref drag_ref_altitude_m. Must be positive when drag is enabled.
  double drag_ref_density_kg_m3{0.0};
  /// Reference altitude h₀ [m] above the spherical Earth radius
  /// @ref reference_radius_m, at which the density is ρ₀.
  double drag_ref_altitude_m{0.0};
  /// Scale height H [m]. Must be positive when drag is enabled.
  double drag_scale_height_m{0.0};

  /// @}
  /// @name Filter
  /// @{

  /// Process-noise acceleration PSD `q_a` [m²/s³]. Sized from the measured
  /// force-model truncation over the coast horizon, `q_a = 3·δr(T)²/T³` — see
  /// the file header. Must be positive.
  double accel_psd_m2_per_s3{0.0};
  /// NIS rejection threshold for the 3-row **position** update [-]. A GNSS
  /// position innovation has no degenerate direction, so this is χ²₃ (99.9% ≈
  /// 16.27), unlike the attitude filter's transverse-by-construction vector
  /// gate. Must be positive.
  double position_nis_gate{0.0};
  /// NIS rejection threshold for the 3-row **velocity** update [-]. Also χ²₃,
  /// but carried separately: the two measurements have different error
  /// structures and a receiver can degrade in one without the other. Must be
  /// positive.
  double velocity_nis_gate{0.0};
  /// Longest interval without an accepted fix before the solution is declared
  /// invalid **and dropped** [s]. See the file header for how it is sized and
  /// why this filter drops where the attitude MEKF retains. Must be positive.
  double max_coast_s{0.0};
  /// Largest accepted propagation gap [s]. A longer step is refused
  /// (@ref OrbitOdRefusal::kStepTooLong) rather than integrated, since a clock
  /// glitch must not be absorbed as a legitimate coast. Must be positive and no
  /// more than `kMaxSubsteps · max_step_s`, so the sub-step loop's compile-time
  /// bound can never be the thing that truncates a propagation.
  double max_dt_s{0.0};
  /// Largest RK4 sub-step [s]. Must be positive.
  double max_step_s{0.0};
  /// Largest fix latency the filter will absorb [s] — how far *behind* its own
  /// epoch a fix may be tagged and still be used. See "Fix latency" in the file
  /// header. Zero restores strictly-forward ingest (a fix behind the filter is
  /// refused); it must cover the receiver's `fix_latency_s` plus the bus and
  /// scheduling transport, and should not be set far beyond it, since the
  /// measurement is forward-propagated across this interval on its own reported
  /// velocity. Must be finite and non-negative.
  double max_fix_latency_s{0.0};

  /// @}
  /// @name Measurement plausibility (§9.1)
  /// @{

  /// Smallest / largest plausible geocentric radius of a fix [m]. Wire data from
  /// the receiver is not trusted: an unchecked position does not fail loudly
  /// downstream, it poisons the propagation while every validity flag still
  /// reads true. Must satisfy `0 < min < max`.
  ///
  /// **Size this to the vehicle's own orbit regime, not to "an Earth orbit".**
  /// The band is the *only* check a fix passes on the seed path: a cold filter,
  /// or one whose solution the coast horizon has just dropped, has no prior, so
  /// it has no innovation and the NIS gate does not exist. A fix that clears the
  /// band there becomes the state the vehicle flies on, outright. On the update
  /// path the gate is a second line of defence; on the seed path there is no
  /// second line. A band wide enough to admit geostationary radius on a 400 km
  /// vehicle therefore admits a GEO fix as a seed — measured, and pinned by
  /// `OrbitOdRefusals.AGeoRadiusFixCannotSeedTheFilterOnALeoVehicle`.
  ///
  /// The reference LEO vehicle uses 6.5e6 m to 8.0e6 m (roughly 120 km to
  /// 1600 km altitude): wide enough for the whole LEO band and any dispersion or
  /// decay within it, narrow enough that MEO, GTO and GEO are all outside.
  double min_radius_m{0.0};
  double max_radius_m{0.0};

  /// @}

  /// True when every field is finite and in range. Checked once at construction.
  bool isValid() const;
};

/// Why an entry point declined to act. `kNone` is success; every other value
/// names a specific, testable condition rather than a bare `false`, so FDIR and
/// telemetry can distinguish "the clock stopped" from "the fix was absurd" from
/// "the filter diverged".
enum class OrbitOdRefusal : std::uint8_t {
  kNone = 0,
  kUnconfigured,         ///< the config failed @ref OrbitOdConfig::isValid
  kUninitialised,        ///< no solution yet, and this entry point cannot seed one
  kNonMonotonicEpoch,    ///< epoch is not strictly after the last (backwards *or* stuck)
  kStepTooLong,          ///< gap exceeds `max_dt_s`
  kCoastExpired,         ///< past `max_coast_s`; the solution was dropped
  kFixNotFinite,         ///< a non-finite component in the fix
  kFixImplausible,       ///< fix radius outside the configured band (§9.1)
  kFixSigmaInvalid,      ///< a reported σ that is not positive and finite
  kFrameConversion,      ///< ECEF→ECI failed, or the epoch is outside EOP coverage
  kNoVelocityForSeed,    ///< a seed needs a full PVT; position alone cannot start a 6-state
  kMeasurementRejected,  ///< the NIS gate refused the fix
  kFilterFault,          ///< non-finite internal result; the solution was dropped
};

/// The refusal's own name, for a log line, a report or a test failure message.
///
/// Beside the enum on purpose: a mapping kept anywhere else is the same list
/// written twice, and the two drift the first time a value is added. Here the
/// switch is exhaustive and `-Wswitch` fails the build when it stops being so,
/// which is the property that makes the mapping worth having at all. No storage,
/// no allocation, nothing that would keep it off the flight path.
[[nodiscard]] constexpr const char* refusalName(OrbitOdRefusal refusal) {
  switch (refusal) {
    case OrbitOdRefusal::kNone:
      return "none";
    case OrbitOdRefusal::kUnconfigured:
      return "unconfigured";
    case OrbitOdRefusal::kUninitialised:
      return "uninitialised";
    case OrbitOdRefusal::kNonMonotonicEpoch:
      return "non_monotonic_epoch";
    case OrbitOdRefusal::kStepTooLong:
      return "step_too_long";
    case OrbitOdRefusal::kCoastExpired:
      return "coast_expired";
    case OrbitOdRefusal::kFixNotFinite:
      return "fix_not_finite";
    case OrbitOdRefusal::kFixImplausible:
      return "fix_implausible";
    case OrbitOdRefusal::kFixSigmaInvalid:
      return "fix_sigma_invalid";
    case OrbitOdRefusal::kFrameConversion:
      return "frame_conversion";
    case OrbitOdRefusal::kNoVelocityForSeed:
      return "no_velocity_for_seed";
    case OrbitOdRefusal::kMeasurementRejected:
      return "measurement_rejected";
    case OrbitOdRefusal::kFilterFault:
      return "filter_fault";
  }
  return "unknown";
}

/// Diagnostics from one 3-row measurement update. Populated whether or not the
/// measurement was accepted, so the NIS of a *rejected* fix reaches telemetry
/// and FDIR.
struct OrbitOdUpdate {
  /// Innovation `y = z − Hx̂` in ECI: [m] for position, [m/s] for velocity.
  Eigen::Vector3d innovation{Eigen::Vector3d::Zero()};
  /// Innovation covariance `S = HPHᵀ + R`, ECI; units follow @ref innovation.
  Eigen::Matrix3d innovation_cov{Eigen::Matrix3d::Identity()};
  /// Normalised innovation squared `yᵀS⁻¹y` [-]; ~χ²₃ when consistent.
  double nis{0.0};
  /// The measurement passed its gate and was applied.
  bool accepted{false};
};

/// Everything @ref OrbitOd::ingest needs from one GNSS fix, in the frame and on
/// the timescale the receiver actually reports (§6.2) — deliberately *not* a
/// convenient ECI/TAI form, because converting it is this filter's job and a
/// caller that pre-converted would be doing it outside the one transform
/// library. Mirrors the fields `sim::sensors::GnssMeasurement` carries; the
/// caller applies the receiver's own `valid`/`fresh` flags before offering a fix
/// (§9.1) — this filter never sees a fix the receiver disowned.
struct GnssFix {
  time::Gps time_tag{};                           ///< receiver time tag, GPS scale
  math::Vec3<math::frames::ECEF> position_m{};    ///< reported position [m]
  math::Vec3<math::frames::ECEF> velocity_m_s{};  ///< reported velocity [m/s]
  double position_sigma_h_m{0.0};                 ///< per-axis horizontal 1σ [m]
  double position_sigma_v_m{0.0};                 ///< vertical (up) 1σ [m]
  double velocity_sigma_m_s{0.0};                 ///< per-axis velocity 1σ [m/s]
  /// The receiver reported a velocity. False takes the position-only path; a
  /// *seed* still needs one (@ref OrbitOdRefusal::kNoVelocityForSeed).
  bool velocity_valid{false};
};

/// Outcome of one @ref OrbitOd::ingest call.
struct OrbitOdResult {
  /// `kNone` when the fix was fully processed. A NIS rejection reports
  /// `kMeasurementRejected` with the diagnostics below populated — a normal
  /// outcome, not an error.
  OrbitOdRefusal refusal{OrbitOdRefusal::kNone};
  /// This fix cold-started or re-acquired the filter, so no update was run and
  /// the covariance is the fix's own.
  bool seeded{false};
  /// How far behind the filter's epoch the fix was tagged [s], and therefore how
  /// far its PVT was forward-propagated before the update. Zero for a fix at or
  /// ahead of the filter. Telemetered rather than merely handled: a latency that
  /// climbs is a receiver or scheduling problem, and it is invisible in the
  /// residuals precisely because this filter corrects for it.
  double fix_latency_s{0.0};
  OrbitOdUpdate position{};  ///< position update diagnostics
  OrbitOdUpdate velocity{};  ///< velocity update diagnostics; untouched when the
                             ///< fix carried none
};

/// Earth orientation at one instant, as the force model needs it.
///
/// Carries the full ECI←ECEF rotation rather than just the pole, because the
/// harmonic field is not axisymmetric: a tesseral term is fixed to the rotating
/// Earth and evaluating it in ECI is wrong in longitude by the whole
/// Earth-rotation angle. The zonal and drag terms still want only the pole, and
/// @ref pole reads it off as the third column — the ECEF Z axis expressed in
/// ECI, by definition of the rotation.
struct EarthOrientation {
  /// Rotation taking an ECEF vector to ECI. Identity is a *test* value, not a
  /// physical one — see @ref earthOrientationAt for the real reduction.
  Eigen::Matrix3d eci_from_ecef{Eigen::Matrix3d::Identity()};

  /// The true (CIP) pole expressed in ECI, unit length.
  Eigen::Vector3d pole() const { return eci_from_ecef.col(2); }
};

/// Acceleration of the onboard force model [m/s²], ECI (design doc §8.3).
///
/// Exposed on its own so the propagator's model can be evaluated, differenced
/// and cross-validated without standing up a filter — the golden test compares
/// it against GMAT at matched fidelity that way. Returns zero for a non-finite
/// or degenerate (sub-surface) position rather than a NaN or an assert; the
/// caller's finiteness guard is what turns that into a refusal.
///
/// @param cfg      force-model coefficients (the filter half is ignored)
/// @param position ECI position [m]
/// @param velocity ECI velocity [m/s]
/// @param earth    Earth orientation at the evaluation epoch — the harmonic
///                 field is evaluated through it, the zonal and drag terms use
///                 its pole
math::Vec3<math::frames::ECI> onboardAcceleration(const OrbitOdConfig& cfg,
                                                  const math::Vec3<math::frames::ECI>& position,
                                                  const math::Vec3<math::frames::ECI>& velocity,
                                                  const EarthOrientation& earth);

/// The Earth orientation at @p t, for @ref onboardAcceleration. Thin wrapper
/// over the one ECI↔ECEF reduction (REQ-CONV-002); returns false, leaving
/// @p out untouched, when the reduction fails.
[[nodiscard]] bool earthOrientationAt(const time::Tai& t, const frames::EopValue& eop,
                                      EarthOrientation& out);

/// 6-state GNSS-aided orbit determination filter (design doc §8.3).
///
/// One instance per vehicle. Hand it fixes with @ref ingest — which seeds itself
/// from the first one — and call @ref propagate once per GNC cycle in between.
/// Plain value with fixed-size storage; allocates nothing.
class OrbitOd {
 public:
  /// Error-state dimension.
  static constexpr int kDim = 6;
  /// Row/column of the position error `δr` within @ref Covariance.
  static constexpr int kPosition = 0;
  /// Row/column of the velocity error `δv` within @ref Covariance.
  static constexpr int kVelocity = 3;
  /// Hard bound on the RK4 sub-step loop, so it is bounded at compile time
  /// (§3.6) rather than by a config value. `OrbitOdConfig::isValid` requires
  /// `max_dt_s ≤ kMaxSubsteps · max_step_s`, so this bound can never be the
  /// thing that silently truncates a legitimate propagation.
  static constexpr int kMaxSubsteps = 64;
  /// Error-state covariance, blocked `[δr; δv]` — the same order and meaning as
  /// the corresponding blocks of `state::Covariance`.
  using Covariance = Eigen::Matrix<double, kDim, kDim>;

  /// Construct with @p config. If the config is invalid the filter is inert:
  /// every entry point refuses with `kUnconfigured` forever.
  explicit OrbitOd(const OrbitOdConfig& config);

  /// True when the configuration passed @ref OrbitOdConfig::isValid.
  bool isConfigured() const { return configured_; }

  /// True once a solution exists and no fault or coast expiry has dropped it.
  bool isInitialised() const { return initialised_; }

  /// Drop the solution (cold start) on command. Configuration is retained and
  /// the rejected count **is** cleared — this is the operator saying "start
  /// over". A coast expiry or internal fault drops the solution without clearing
  /// that count.
  void reset();

  /// Seed the filter directly. @ref ingest does this for itself from a fix, so
  /// this exists for a ground-uploaded state vector, for replay, and for tests.
  ///
  /// @param epoch    TAI time tag of the state (§3.2)
  /// @param position ECI position [m]
  /// @param velocity ECI velocity [m/s]
  /// @param cov      `E[xxᵀ]` blocked `[δr; δv]`; must be finite and
  ///                 **positive-definite** — an indefinite seed makes `S`
  ///                 indefinite and the NIS gate meaningless
  /// @return `kNone` on success; the previous state is untouched on refusal.
  OrbitOdRefusal initialize(const time::Tai& epoch, const math::Vec3<math::frames::ECI>& position,
                            const math::Vec3<math::frames::ECI>& velocity, const Covariance& cov);

  /// Propagate the state and covariance to @p epoch on the onboard force model.
  ///
  /// @param epoch TAI time tag to propagate to; must be **strictly** after the
  ///              last one
  /// @param eop   Earth orientation at @p epoch, for the J2 pole
  /// @return `kNone` on a completed step. `kCoastExpired` when the step carried
  ///         the solution past `max_coast_s` — the solution is **dropped**, so
  ///         the next fix re-acquires whole. `kFilterFault` on a non-finite
  ///         internal result (also dropped). Everything else leaves the state
  ///         untouched.
  OrbitOdRefusal propagate(const time::Tai& epoch, const frames::EopValue& eop);

  /// Ingest one GNSS fix: GPS→TAI and ECEF→ECI (REQ-CONV-001), propagate to the
  /// fix epoch, then the position and (when reported) velocity updates.
  ///
  /// Seeds itself when there is no solution — on the first fix, and after a
  /// coast expiry or a fault — so re-acquisition is whole rather than a blend
  /// against a prior that no longer means anything. @ref OrbitOdResult::seeded
  /// says which happened.
  ///
  /// @param fix the receiver's report, in its own frame and timescale
  /// @param eop Earth orientation at the fix epoch
  /// @param out diagnostics and the refusal reason; always fully written
  /// @return `true` iff the fix left the filter with a valid solution, i.e. it
  ///         seeded or at least the position update was accepted
  bool ingest(const GnssFix& fix, const frames::EopValue& eop, OrbitOdResult& out);

  /// Table-driven propagation. Resolves the EOP for @p epoch from the uploaded
  /// table first, refusing with `kFrameConversion` when the epoch lies outside
  /// its span (the table does not extrapolate), then applies the overload above.
  template <std::size_t Capacity>
  OrbitOdRefusal propagate(const time::Tai& epoch, const frames::EopTable<Capacity>& eop,
                           const time::LeapSecondTable& leap) {
    frames::EopValue e;
    if (!eop.lookup(epoch, leap, e)) {
      return OrbitOdRefusal::kFrameConversion;
    }
    return propagate(epoch, e);
  }

  /// Table-driven @ref ingest. A fix whose epoch the uploaded EOP table does not
  /// cover cannot be brought into ECI at all, so it is refused rather than
  /// converted on an extrapolated Earth orientation.
  template <std::size_t Capacity>
  bool ingest(const GnssFix& fix, const frames::EopTable<Capacity>& eop,
              const time::LeapSecondTable& leap, OrbitOdResult& out) {
    frames::EopValue e;
    if (!eop.lookup(time::toTai(fix.time_tag), leap, e)) {
      out = OrbitOdResult{};
      out.refusal = OrbitOdRefusal::kFrameConversion;
      return false;
    }
    return ingest(fix, e, out);
  }

  /// Estimated ECI position [m].
  math::Vec3<math::frames::ECI> position() const {
    return math::Vec3<math::frames::ECI>(position_);
  }

  /// Estimated ECI velocity [m/s].
  math::Vec3<math::frames::ECI> velocity() const {
    return math::Vec3<math::frames::ECI>(velocity_);
  }

  /// 6×6 error-state covariance `[δr; δv]`, symmetric.
  const Covariance& covariance() const { return p_; }

  /// TAI epoch the state is valid at.
  const time::Tai& epoch() const { return last_epoch_; }

  /// Time since the last **accepted** fix [s]; grows through outage.
  double ageSeconds() const { return age_s_; }

  /// The solution exists and is inside the coast horizon. Because expiry drops
  /// the solution, this is equivalent to @ref isInitialised — both are kept
  /// because they answer different questions and a future change to the
  /// retention policy would separate them.
  bool solutionValid() const { return initialised_ && age_s_ <= cfg_.max_coast_s; }

  /// Fixes rejected by a NIS gate since construction or @ref reset. A rising
  /// count is the FDIR signal, not a single rejection.
  std::uint32_t rejectedCount() const { return rejected_; }

  /// Analysis-only 6-state NEES `eᵀP⁻¹e` against a known truth, with
  /// `e = [r_true − r̂; v_true − v̂]` (Bar-Shalom §5.4 [barshalom2001]). Averaged
  /// over Monte-Carlo runs it should sit inside the χ²₆ bounds; systematically
  /// above means overconfident, below means conservative. There is no truth
  /// onboard, so this exists for tests, replay, and covariance validation.
  ///
  /// @return `false` (leaving @p out untouched) when uninitialised, on
  ///         non-finite inputs, or if `P` cannot be inverted.
  bool nees(const math::Vec3<math::frames::ECI>& position_true,
            const math::Vec3<math::frames::ECI>& velocity_true, double& out) const;

 private:
  /// Drop the solution while leaving the rejection count standing — the coast-
  /// expiry and internal-fault paths, for the reason @ref reset documents.
  void dropSolution();

  /// One 3-row update against a measurement of `H = [I 0]` (position, @p offset
  /// = kPosition) or `H = [0 I]` (velocity, @p offset = kVelocity).
  bool applyUpdate(int offset, const Eigen::Vector3d& measured, const Eigen::Matrix3d& r_cov,
                   double gate, OrbitOdUpdate& out);

  /// Seed from an already-converted inertial fix. Shared by @ref ingest's
  /// cold-start and re-acquisition paths.
  OrbitOdRefusal seedFrom(const time::Tai& epoch, const Eigen::Vector3d& r_eci,
                          const Eigen::Vector3d& v_eci, const Eigen::Matrix3d& r_pos_cov,
                          double velocity_sigma);

  OrbitOdConfig cfg_{};
  Eigen::Vector3d position_{Eigen::Vector3d::Zero()};  ///< ECI position estimate [m]
  Eigen::Vector3d velocity_{Eigen::Vector3d::Zero()};  ///< ECI velocity estimate [m/s]
  Covariance p_{Covariance::Zero()};                   ///< error-state covariance
  time::Tai last_epoch_{};                             ///< epoch of the state
  time::Tai last_fix_epoch_{};                         ///< epoch of the last ingested fix
  double age_s_{0.0};                                  ///< since the last accepted fix [s]
  std::uint32_t rejected_{0};                          ///< NIS-gate rejections
  bool configured_{false};
  bool initialised_{false};
  bool have_fix_{false};  ///< `last_fix_epoch_` is meaningful
};

/// Copy an orbit solution into the canonical onboard state (§8.0). Writes
/// position, velocity, their validity flags, and the **position and velocity
/// blocks of the 15×15 error-state covariance including their cross terms** (the
/// correlation is what a consumer propagating the state forward needs).
/// Attitude, rates and biases are untouched: the §8.1 filters own those, and the
/// `mode` field is theirs too — it names the *attitude* mode, and an orbit
/// filter has no business overwriting it.
///
/// As in the attitude path, the covariance blocks are written **only** when the
/// solution is valid, so an invalid solution cannot stamp a meaningless
/// covariance over the last good one. `valid.covariance` is left **set** by this
/// function only when it writes; it never clears a flag the attitude filter set,
/// since the two estimators write disjoint blocks behind one flag (§8.0 requires
/// a consumer to read the block it needs and check that field's own flag).
///
/// @param filter converged (or coasting) filter
/// @param epoch  TAI time tag to stamp on the state (§3.2)
/// @param state  canonical state to update in place
void writeToEstimatedState(const OrbitOd& filter, const time::Tai& epoch,
                           state::EstimatedState& state);

}  // namespace polaris::gnc

#endif  // POLARIS_GNC_ORBIT_OD_HPP
