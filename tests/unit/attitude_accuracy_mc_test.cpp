/// @file Monte Carlo attitude-knowledge accuracy campaign
/// (REQ-ADET-005, REQ-ADET-006; design doc §8.1, §22.3).
///
/// The two **vehicle-level** requirements are stated on the **error norm** — the
/// total eigenaxis rotation angle between the estimated and the true attitude,
/// `θ_err = 2·atan2(‖q_err vec‖, |q_err scalar|)` (`errorNormDeg` below; the
/// atan2 form per lib/README.md, since the errors being measured are small
/// enough that the scalar part alone would have lost half its digits) — at the
/// 3σ (99.73rd-percentile) point, on
/// the reference vehicle's own sensor budget
/// (`config/spacecraft/leo_smallsat.yaml`, `flight.attitudeEstimator.*`).
///
/// **Two budgets, and only one of them verifies.** `requirementRuns()` is the
/// **post-calibration** configuration — the magnetometer fit applied and its
/// tightened `SigmaMagSysRad`/`SeedMinObservability` uplinked, the albedo
/// correction applied, DE440 sun tables at grade `kPrecise` — and it is what
/// REQ-ADET-005 (≤ 5°) and REQ-ADET-006 (≤ 3°) are measured against.
///
/// It is deliberately **not** what `config/spacecraft/leo_smallsat.yaml` ships:
/// that file carries the uncalibrated `SigmaMagSysRad = 0.0337` and
/// `SeedMinObservability = 0.0076`, and they change only by ground-commanded
/// parameter uplink after a successful `MAG_CAL_START` fit. Both requirements
/// name that parameter set as a stated condition, so this campaign verifies the
/// configuration the requirements are written against rather than quietly
/// assuming the vehicle boots into it.
///
/// `campaignRuns()` is the **uncalibrated** budget those thresholds used to be
/// stated on; it is kept as the baseline the informative projections measure
/// their improvement from, and because the coarse-vs-fine estimator comparison
/// is cleanest where the systematic floor dominates.
///
/// Alongside them, and **informative rather than verifying**, the campaign
/// projects the same fine-mode error onto a payload boresight: REQ-PAY-001 is
/// stated in fine+star-tracker mode, which this campaign cannot reach until the
/// §8.2 fusion layer exists, so the projection here is evidence about the metric
/// (and about this campaign having no preferred body axis), not a verification
/// of the requirement.
///
/// **One draw feeds both estimators.** Each run generates its geometry, truth
/// motion, systematic biases and white-noise sequences once and hands the
/// identical measurement stream to the coarse chain and to the MEKF, so
/// coarse-vs-fine is a *paired* comparison — any difference is the estimator,
/// not the sample. Each budget's campaign is built once on first use and shared
/// by every case that reads it.
///
/// **The method behind these numbers is documented once, in
/// `docs/requirements/adcs_determination.rst`** — the metric definition
/// (`adet-knowledge-metric`) and the four choices that decide what the measured
/// thresholds mean (`adet-campaign-method`): systematics drawn per run as biases
/// rather than as noise, geometry swept over the well-conditioned band, the
/// distribution-free sample-maximum bound, and the sensitivity floor on the
/// median. Read that section before changing anything here; the comments below
/// give only the local reason for each choice.
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <vector>

#include "gnc/coarse_attitude.hpp"
#include "gnc/davenport.hpp"
#include "gnc/mekf.hpp"
#include "gnc/st_alignment.hpp"
#include "math/frames.hpp"
#include "math/quaternion.hpp"
#include "math/typed_vector.hpp"
#include "random/rng.hpp"
#include "sensors/payload_sensor.hpp"
#include "time/timescales.hpp"

namespace {

namespace gnc = polaris::gnc;
namespace pm = polaris::math;
namespace frames = polaris::math::frames;
namespace ptime = polaris::time;

constexpr double kDeg = M_PI / 180.0;
constexpr double kDt = 0.1;  ///< estimation cycle [s] (10 Hz, §8.1 rate group)

/// Monte Carlo size: 800 runs put the sample maximum at 88% confidence as an
/// upper bound on the 3σ quantile (`adet-campaign-method`). Larger N is better
/// statistics and a linearly longer test; this is where the campaign still fits
/// a unit-test binary CI runs on every push.
constexpr int kRuns = 800;
/// Cycles per run: 1.5 s at 10 Hz. Both estimators are past their transient by
/// then — the coarse blend settles in ~1 s at the configured gain, the filter is
/// seeded at measurement accuracy — so the sampled error is the steady-state
/// one. Checked against 3 s and 40 s runs: the medians agree to 0.1° and the 3σ
/// bound to 0.8°, with the short run on the conservative side.
constexpr int kSteps = 15;

// ── The reference vehicle's error budget ────────────────────────────────────
// Mirrors `flight.attitudeEstimator.*` in config/spacecraft/leo_smallsat.yaml,
// which derives every value from the units that vehicle carries (GomSpace
// NanoSense FSS, MAG-GENERIC, STIM300). Keep the two in sync: these are the
// numbers the requirement thresholds were measured against, so a budget change
// in the YAML has to be re-measured here, not silently diverge.
constexpr double kSigmaSunWhite = 0.0116;    ///< [rad] FSS noise at the FOV edge
constexpr double kSunSysUncal = 0.0356;      ///< [rad] albedo ⊕ analytic ephemeris
constexpr double kSigmaMagWhite = 0.0017;    ///< [rad] 0.05 µT rms on a 30 µT field
constexpr double kSigmaMagSys = 0.0337;      ///< [rad] hard/soft-iron residual
constexpr double kGyroArw = 4.363e-5;        ///< [rad·s^(-1/2)] STIM300 0.15°/√h
constexpr double kGyroRrw = 4.7e-7;          ///< [rad·s^(-3/2)] STIM300 bias instability
constexpr double kBiasSigmaInit = 4.848e-5;  ///< [rad/s] 10°/h turn-on repeatability
constexpr double kMinSinAngle = 0.17;        ///< sin(10°) TRIAD geometry gate
constexpr double kTriadGain = 0.3;
constexpr double kNisGate = 13.82;          ///< χ²₂ at 99.9% (vector updates)
constexpr double kAttitudeNisGate = 16.27;  ///< χ²₃ at 99.9% (star-tracker updates)
constexpr double kSeedMinObservability = 0.0076;

/// The magnetic systematic **after** the on-orbit hard/soft-iron calibration of
/// §8.1 [rad]. This is a **verifying** constant: it is part of the
/// post-calibration budget REQ-ADET-005/006 are stated against.
///
/// It is *not* what `config/spacecraft/leo_smallsat.yaml` ships. The vehicle
/// flies `SigmaMagSysRad = 0.0337` until a `MAG_CAL_START` window fits and the
/// ground uplinks the tightened value — which is exactly why both requirements
/// are conditioned on the post-calibration parameter set being in force.
///
/// Measured by `mag_calibration_test.cpp`
/// (`PostCalibrationSystematicMeetsTheHalfDegreeTarget`) on the same budget the
/// constants above describe: 0.087° rms residual direction error in the worst of
/// 32 seeds. That figure is the *total* angle, so using it as the per-axis σ
/// here is conservative by √2 — the projection is meant to under-promise.
constexpr double kSigmaMagSysPostCal = 0.087 * kDeg;

/// The sun systematic **after** the onboard Earth-albedo correction of §8.1
/// [rad], on the analytic sun ephemeris. This is the **degraded floor** of
/// REQ-ADET-006 — what the vehicle falls back to with no ephemeris upload in
/// place or past the end of the uploaded span.
///
/// Derived exactly as the uncalibrated 0.0356 in
/// `config/spacecraft/leo_smallsat.yaml` is, with one term replaced. The albedo
/// contribution was the 2.0° (34.9 mrad) histogram bulk; `lib/gnc/albedo_correction`
/// removes the deterministic part of it and leaves the truth model's dispersion,
/// whose *total* 1σ is `f = albedo_dispersion_fraction = 0.30` of the pull (the
/// catalog value; see `config/hardware/sun_sensor/gomspace_nanosense_fss.yaml`
/// for why 0.30, and `albedo_correction_test.cpp`, which measures the ratio
/// rather than assuming it). The analytic-ephemeris term (0.4° = 7.0 mrad) is
/// untouched — the correction does not know where the Sun is any better than the
/// ephemeris does: `sqrt((0.30·34.9)² + 7.0²) = 12.6 mrad`.
///
/// **Not** included: the correction's own sensitivity to the attitude that
/// placed the Earth in the sensor's field, `A·σ_att/2` with `A` the 12° peak
/// scale. That term is a function of how well the attitude is known each cycle,
/// so `AttitudeEstimator` carries it dynamically (in quadrature with this
/// constant) rather than budgeting it here. At the converged fine-mode error
/// this projection measures (~0.6° median) it is ~1 mrad, which moves 12.6 to
/// 12.64 — below the rounding of the constant itself, so folding it in would be
/// false precision. It is *not* negligible right after acquisition, and that is
/// exactly the regime the dynamic term exists for and this steady-state campaign
/// does not sample.
///
/// **This constant, not the algorithm, is what the projection turns on.** The
/// deterministic part of the albedo is removed exactly; everything the vehicle
/// gains or fails to gain is decided by how well the real Earth is modelled by a
/// uniform Lambertian sphere. Read the projection below as a statement about
/// `f`, and treat 0.30 as the honest mid-range figure it is rather than a
/// measured property of any orbit.
constexpr double kSunSysPostAlbedo = 0.0126;

/// The same budget with the **onboard DE440 Chebyshev tables** answering the sun
/// query at grade PRECISE, rather than the analytic fallback [rad].
///
/// The two terms the component composes per cycle are independent — the albedo
/// residual is the sensor's, the ephemeris error is the reference's — so this is
/// the same 10.5 mrad albedo residual against an ephemeris term that all but
/// vanishes: `sqrt(10.5² + 0.05²) = 10.5 mrad`. The tables give an
/// arcsecond-class sun direction from *time alone*, and the ~10 arcsec LEO
/// parallax the component already subtracts when it has a position fix.
///
/// This is the budget the vehicle flies with an ephemeris upload in place, and
/// it — not the analytic-fallback figure above — is what REQ-ADET-006's 3° is
/// conditioned on. Unlike the two calibration constants, this one needs **no
/// parameter uplink**: the vehicle already carries both grades and the
/// estimator selects per cycle on the served `TableGrade`.
constexpr double kSunSysPostAlbedoTables = 0.0105;

/// The Davenport seed's observability gate under the post-calibration budget
/// [dimensionless] — part of the verifying parameter set, and the second value
/// the ground must uplink alongside the tightened magnetometer sigma.
///
/// It has to be re-derived rather than reused, and the reason is worth stating
/// because it is a **flight-parameter finding, not a test detail**. The gated
/// ratio `λ_min/λ_max` of `M = Σ σᵢ⁻²(I − b̂ᵢb̂ᵢᵀ)` is scale-free under a
/// *common* rescaling of the σ's, which is what makes it a geometry gate — but
/// it is not invariant to changing the σ's *relative* to each other. Splitting
/// `M` into its out-of-plane direction (eigenvalue `w_s + w_m`) and its 2×2
/// in-plane block (trace `w_s + w_m`, determinant `w_s w_m sin²θ`) gives
///
///     λ_min/λ_max = ½[1 − √(1 − 4 w_s w_m sin²θ / (w_s + w_m)²)],
///
/// which collapses to `sin²(θ/2)` at equal weights — the shipped 0.0076 at
/// θ = 10°. On the post-calibration budget the weights are far from equal, and
/// the same 10° reads **5.16e-4** on the analytic ephemeris and 6.14e-4 with the
/// DE440 tables. **5.1e-4** is what the ground uplinks: the grade changes per
/// cycle without an uplink, so the gate has to admit 10° in the worse of the two
/// (design doc §8.1, and the derivation block in
/// `config/spacecraft/leo_smallsat.yaml`).
///
/// This campaign runs the gate at **1.0e-4**, deliberately looser than flight.
/// The sweep never goes below 45° of separation, so the gate is not what any run
/// here turns on, and a looser value keeps the projections from being
/// accidentally gated by a flight parameter they are not measuring. Do **not**
/// read this constant as the uplink value; 5.1e-4 is.
constexpr double kSeedMinObservabilityPostCal = 1.0e-4;

/// Sun/field separations sampled: the well-conditioned band the requirements are
/// stated under, not the orthogonal best case (`adet-campaign-method`).
constexpr double kMinSeparationDeg = 45.0;
constexpr double kMaxSeparationDeg = 135.0;

/// Requirement thresholds on the error norm [deg], 3σ — REQ-ADET-005 and
/// REQ-ADET-006, **enacted** at 5°/3° once both calibration items and the DE440
/// ephemeris grading landed (Pushes 46, 47, 48).
///
/// These are requirement values, not measurements: the campaign below is run
/// against them, never fitted to them. The verifying budget is the
/// **post-calibration** configuration — magnetometer calibration applied *and
/// its tightened parameters uplinked*, albedo correction applied, DE440 tables
/// answering the sun query at grade PRECISE — which is what
/// @ref requirementRuns exercises. Both requirements state that parameter set
/// as a condition; as delivered the vehicle boots uncalibrated and sits on the
/// 8.49°/8.07° floor the requirement bodies record.
///
/// The fine threshold is **conditioned on the tables being active**, and the
/// reason is margin rather than threshold: on the analytic-ephemeris fallback
/// the fine bound is 2.69°, which clears 3° but leaves only ~10% margin against
/// the 20% REQ-ADET-006 asks for. Stating one unconditioned number would force a
/// choice between a threshold the vehicle misses whenever an upload lapses and
/// one that gives away the margin an upload buys. The fallback is recorded as
/// the degraded floor by @ref PostBothCorrectionsFallbackIsTheDegradedFloor.
constexpr double kCoarseLimitDeg = 5.0;
constexpr double kFineLimitDeg = 3.0;
/// REQ-ADET-007: fine mode with star tracker(s) fused.
constexpr double kStarLimitDeg = 0.05;

/// Both requirements declare `margin_required: 20 %`, so a passing bound must
/// sit at or below 80% of its threshold. The docs build has no gate for this
/// field (`conf.py` filters on verification coverage only), so the campaign is
/// where it is enforced — otherwise "20% margin" is a number the RVTM prints
/// and nothing checks.
constexpr double kMarginFraction = 0.8;

ptime::Tai epochAt(double t_s) {
  return ptime::Tai::fromNanosecondsSinceEpoch(static_cast<std::int64_t>(t_s * 1.0e9));
}

pm::Quaternion truthAttitude(const Eigen::Vector3d& rate, double t_s,
                             const pm::Quaternion& initial) {
  const double angle = rate.norm() * t_s;
  if (angle <= 0.0) {
    return initial;
  }
  return (pm::Quaternion::FromAxisAngle(rate.normalized(), angle) * initial).canonical();
}

/// The metric the vehicle-level requirements are written on: total eigenaxis
/// angle between two attitudes [deg] (`adet-knowledge-metric`).
double errorNormDeg(const pm::Quaternion& est, const pm::Quaternion& q_true) {
  const pm::Quaternion dq = (est * q_true.inverse()).canonical();
  return 2.0 * std::atan2(dq.vec().norm(), dq.scalar()) / kDeg;
}

Eigen::Vector3d anyPerpendicular(const Eigen::Vector3d& u) {
  const Eigen::Vector3d seed =
      (std::abs(u.x()) < 0.9) ? Eigen::Vector3d::UnitX() : Eigen::Vector3d::UnitY();
  return u.cross(seed).normalized();
}

/// Transverse tilt of a unit vector by @p a1, @p a2 [rad] on its own transverse
/// basis. Used for both error kinds; what separates them is *when the
/// coefficients are drawn* — once per run (systematic) or once per measurement
/// (white).
Eigen::Vector3d tilt(const Eigen::Vector3d& u, double a1, double a2) {
  const Eigen::Vector3d t1 = anyPerpendicular(u);
  const Eigen::Vector3d t2 = u.cross(t1);
  return (u + a1 * t1 + a2 * t2).normalized();
}

/// A run's fixed systematic offsets, drawn once and held: sun pair and magnetic
/// pair, two transverse coefficients each [rad].
struct Systematics {
  double sun1{0.0};
  double sun2{0.0};
  double mag1{0.0};
  double mag2{0.0};

  /// @param sun_sys the sun systematic 1σ [rad] this campaign runs at —
  ///        `kSunSysPostAlbedoTables` for the requirement campaigns,
  ///        `kSunSysPostAlbedo` for the degraded floor, `kSunSysUncal` for the
  ///        uncalibrated baseline the projections measure from.
  /// @param mag_sys the magnetic systematic 1σ [rad], likewise —
  ///        `kSigmaMagSysPostCal` for the requirement campaigns and the degraded
  ///        floor, `kSigmaMagSys` for the uncalibrated baseline.
  ///
  /// Four draws always, in the same order, so two levels give the same
  /// realisations rescaled rather than a different sample: the projections below
  /// are paired against the baseline run for run.
  static Systematics draw(polaris::random::SplitMix64& rng, double sun_sys, double mag_sys) {
    return {sun_sys * rng.gaussian(), sun_sys * rng.gaussian(), mag_sys * rng.gaussian(),
            mag_sys * rng.gaussian()};
  }
};

/// One run's randomised setup: truth attitude, body rate, geometry, gyro bias.
struct RunSetup {
  pm::Quaternion q0{};
  Eigen::Vector3d rate{Eigen::Vector3d::Zero()};
  Eigen::Vector3d sun_eci{Eigen::Vector3d::UnitX()};
  Eigen::Vector3d mag_eci{Eigen::Vector3d::UnitY()};
  Eigen::Vector3d bias{Eigen::Vector3d::Zero()};
  Systematics sys{};
};

RunSetup drawRun(polaris::random::SplitMix64& rng, double sun_sys, double mag_sys) {
  RunSetup s{};
  const Eigen::Vector3d axis =
      Eigen::Vector3d(rng.gaussian(), rng.gaussian(), rng.gaussian()).normalized();
  s.q0 = pm::Quaternion::FromAxisAngle(axis, 180.0 * kDeg * rng.uniform());

  // A slow controlled-vehicle rate, up to ~0.3 deg/s on each axis.
  s.rate = 0.005 * Eigen::Vector3d(rng.gaussian(), rng.gaussian(), rng.gaussian());

  // Sun along a random inertial direction; field at a swept separation from it.
  s.sun_eci = Eigen::Vector3d(rng.gaussian(), rng.gaussian(), rng.gaussian()).normalized();
  const double separation =
      (kMinSeparationDeg + (kMaxSeparationDeg - kMinSeparationDeg) * rng.uniform()) * kDeg;
  const Eigen::Vector3d perp = anyPerpendicular(s.sun_eci);
  s.mag_eci = (Eigen::AngleAxisd(2.0 * M_PI * rng.uniform(), s.sun_eci) *
               (std::cos(separation) * s.sun_eci + std::sin(separation) * perp))
                  .normalized();

  s.bias = kBiasSigmaInit * Eigen::Vector3d(rng.gaussian(), rng.gaussian(), rng.gaussian());
  s.sys = Systematics::draw(rng, sun_sys, mag_sys);
  return s;
}

// ── The sun-sensor suite, and what a handoff costs (§8.2) ───────────────────
//
// The reference vehicle carries six GomSpace FSS units on the six body faces
// (`config/spacecraft/leo_smallsat.yaml`). Two consequences matter to this
// campaign, and the second is the one the requirement rests on.
//
// **Coverage.** Six 60°-half-angle cones on the face normals cover the sphere:
// the worst-placed direction is a body diagonal at arccos(1/√3) = 54.736° from
// each of the three nearest normals. So the Sun is always in *some* unit's
// field, with the best-available incidence never exceeding 54.736°. The flight
// selector does not always pick that unit (see selectSunUnit — σ-ordered, ties
// to the lowest index), so the *selected* incidence is bounded only by the 60°
// field edge; 54.736° is a property of the layout, not of the selection.
//
// **What the vehicle is told, versus what it gets.** The FSS is specified 0.5°
// (3σ) inside 45° incidence and 2.0° out to the 60° edge, and it reports its
// realised σ per sample. The *truth* noise below follows that curve, because
// that is the sensor's physics. The estimators are handed the shipped
// `SigmaSunWhiteRad` — the conservative 60°-edge figure — on every cycle,
// because that is what the flight code does: `AttitudeEstimator` weights with
// the configured constant and does not yet consume the per-sample σ the port
// already carries. Since 54.736° < 60°, the constant **bounds** every geometry
// the suite can hand it, which is exactly the claim the handoff sweep verifies.
// Consuming the realised σ would tighten the fine solution and is deferred: it
// needs a per-cycle white override on `CoarseAttitudeInput` (the MEKF already
// takes σ per update).

constexpr int kSunUnits = 6;

/// Body-frame boresights of the six-face suite, in the vehicle-config order.
const Eigen::Vector3d kSunBoresights[kSunUnits] = {
    Eigen::Vector3d::UnitZ(), Eigen::Vector3d(0.0, 0.0, -1.0),
    Eigen::Vector3d::UnitX(), Eigen::Vector3d(-1.0, 0.0, 0.0),
    Eigen::Vector3d::UnitY(), Eigen::Vector3d(0.0, -1.0, 0.0)};

/// FSS acceptance half-angle and its two accuracy regimes
/// (`config/hardware/sun_sensor/gomspace_nanosense_fss.yaml`), as 1σ.
constexpr double kFssHalfFovRad = 60.0 * kDeg;
constexpr double kFssInnerHalfAngleRad = 45.0 * kDeg;
constexpr double kFssSigmaInner = (0.5 / 3.0) * kDeg;
constexpr double kFssSigmaOuter = (2.0 / 3.0) * kDeg;

/// The realised 1σ a unit at @p incidence_rad reports, from the FSS's two
/// accuracy regimes. This is the step function the *sensor* publishes, and
/// therefore the only thing the flight selector can order on.
double fssSigmaAt(double incidence_rad) {
  return (incidence_rad <= kFssInnerHalfAngleRad) ? kFssSigmaInner : kFssSigmaOuter;
}

/// The unit `AttitudeEstimator::selectSunSensor` would pick for a Sun at
/// @p sun_body, and the σ and incidence that unit realises.
///
/// **This mirrors the flight selector exactly, and the distinction matters.**
/// The flight rule is *smallest reported σ, ties to the lowest index* — not
/// smallest incidence, which the estimator cannot see. Because σ is a two-regime
/// step function, ties are the common case: with several units inside 45° they
/// all report `kFssSigmaInner`, and the vehicle takes the lowest-indexed of
/// them, which is generally **not** the best-aimed one. Selecting on incidence
/// here would flatter the campaign by modelling a selector the vehicle does not
/// have.
///
/// Returns -1 in @p index when no unit sees the Sun, which the six-face suite
/// makes unreachable — asserted rather than assumed by
/// @ref SunSuiteCoversEveryAttitude.
void selectSunUnit(const Eigen::Vector3d& sun_body, int& index, double& sigma_rad,
                   double& incidence_rad) {
  index = -1;
  sigma_rad = kFssSigmaOuter;
  incidence_rad = M_PI;
  for (int i = 0; i < kSunUnits; ++i) {
    // atan2 of the cross-product norm against the dot product (lib/README.md).
    const double angle =
        std::atan2(sun_body.cross(kSunBoresights[i]).norm(), sun_body.dot(kSunBoresights[i]));
    if (angle > kFssHalfFovRad) {
      continue;  // Sun outside this unit's field: it reports no sun in view
    }
    const double sigma = fssSigmaAt(angle);
    // Strictly-less, so a tie keeps the earlier index — the flight rule.
    if (index < 0 || sigma < sigma_rad) {
      index = i;
      sigma_rad = sigma;
      incidence_rad = angle;
    }
  }
}

/// The measured sun direction in body: truth, tilted by the run's fixed
/// systematic offset, then by this cycle's white draw.
///
/// @param white_sigma_rad the 1σ of that white draw. The single-unit campaigns
///        pass the shipped field-edge `kSigmaSunWhite`; the handoff sweep passes
///        the σ the *selected* unit of the six-face suite realises at this
///        attitude, which is the sensor's own accuracy-vs-incidence curve.
Eigen::Vector3d measureSun(const RunSetup& s, const pm::Quaternion& q_true,
                           polaris::random::SplitMix64& rng, double white_sigma_rad) {
  const Eigen::Vector3d truth = q_true.rotate(s.sun_eci);
  const Eigen::Vector3d biased = tilt(truth, s.sys.sun1, s.sys.sun2);
  return tilt(biased, white_sigma_rad * rng.gaussian(), white_sigma_rad * rng.gaussian());
}

Eigen::Vector3d measureSun(const RunSetup& s, const pm::Quaternion& q_true,
                           polaris::random::SplitMix64& rng) {
  return measureSun(s, q_true, rng, kSigmaSunWhite);
}

Eigen::Vector3d measureMag(const RunSetup& s, const pm::Quaternion& q_true,
                           polaris::random::SplitMix64& rng) {
  const Eigen::Vector3d truth = q_true.rotate(s.mag_eci);
  const Eigen::Vector3d biased = tilt(truth, s.sys.mag1, s.sys.mag2);
  return tilt(biased, kSigmaMagWhite * rng.gaussian(), kSigmaMagWhite * rng.gaussian());
}

/// Gyro reading: true rate + run bias (random-walking at the RRW) + ARW white
/// noise discretised as σ_v/√dt.
Eigen::Vector3d measureGyro(const Eigen::Vector3d& rate, Eigen::Vector3d& bias,
                            polaris::random::SplitMix64& rng) {
  const Eigen::Vector3d reading =
      rate + bias +
      (kGyroArw / std::sqrt(kDt)) * Eigen::Vector3d(rng.gaussian(), rng.gaussian(), rng.gaussian());
  bias +=
      (kGyroRrw * std::sqrt(kDt)) * Eigen::Vector3d(rng.gaussian(), rng.gaussian(), rng.gaussian());
  return reading;
}

// ── The star-tracker suite (§8.2) ───────────────────────────────────────────
//
// The reference vehicle carries **two** AURIGA trackers whose boresights are 90°
// apart, both 135° from body +Z so neither ever looks along the payload/array
// face (`config/spacecraft/leo_smallsat.yaml`). Three properties of the model
// below are load-bearing, and each is a place a lazier campaign would flatter the
// result:
//
// **The error is anisotropic, in the unit's own frame.** A tracker barely
// constrains rotation about its own boresight — the identified stars hardly move
// — so the about-boresight σ is several times the cross-boresight one. Modelling
// it isotropically would make the second, non-parallel tracker pointless, which
// is the whole configuration decision under test.
//
// **The per-unit bias does not average down, and the king's is not removable.**
// Each unit carries a fixed bias drawn once per run (the datasheet's `bias_deg` is
// a *bound*: an isotropic direction with magnitude uniform in [0, B]). The
// inter-tracker calibration removes the *difference* between the two — that is
// exactly what it estimates — so after it runs, unit 1 reads in unit 0's frame.
// What nothing removes is unit 0's own bias, because the king's mounting *is* the
// body frame by definition and the payload is mounted against the physical
// structure, not against the king's optical axis. That residual is the floor this
// campaign measures, and it is the honest answer to "what does the king-tracker
// architecture buy?": consistency between the trackers, not absolute truth.
//
// **The calibration is the real library, not a perfect subtraction.** The pairs
// are streamed through `gnc::StAlignmentAccumulator` exactly as the flight
// component streams them, so the residual the campaign carries is the residual the
// fit actually leaves.

constexpr int kStarUnits = 2;

/// Body-frame boresights of the two-tracker suite, in vehicle-config order.
const Eigen::Vector3d kStarBoresights[kStarUnits] = {Eigen::Vector3d(-1.0, 0.0, -1.0).normalized(),
                                                     Eigen::Vector3d(1.0, 0.0, -1.0).normalized()};

/// AURIGA per-axis 1σ white terms, from
/// `config/hardware/star_tracker/sodern_auriga.yaml` (3σ figures / 3):
/// cross-boresight RSS(9.0, 6.6, 11.0) = 15.7 arcsec 3σ; about-boresight
/// RSS(51, 38, 70) = 94.6 arcsec 3σ.
constexpr double kArcsec = M_PI / (180.0 * 3600.0);
constexpr double kStWhiteXy = (15.7 / 3.0) * kArcsec;
constexpr double kStWhiteZ = (94.6 / 3.0) * kArcsec;

/// `bias_deg = 0.017°` = 61.2 arcsec, quoted as a **bound** on the magnitude of an
/// isotropic offset — so the per-axis RMS is `sqrt(E[m²]/3) = (B/√3)/√3 = B/3`.
constexpr double kStBiasBound = 61.2 * kArcsec;
constexpr double kStBiasPerAxis = kStBiasBound / 3.0;

/// The σ pair the *estimator* is told, and therefore what `R` is built from:
/// white ⊕ the per-axis bias, because the filter treats R as white and a bias
/// folded in uninflated would have it converge below its true error (mekf.hpp).
/// These are the shipped `StSigmaXyRad` / `StSigmaZRad` to three figures.
const double kStSigmaXy = std::hypot(kStWhiteXy, kStBiasPerAxis);
const double kStSigmaZ = std::hypot(kStWhiteZ, kStBiasPerAxis);

/// Measurement covariance for a unit with body-frame boresight @p b:
/// `σ_xy²(I − b bᵀ) + σ_z² b bᵀ` — the closed form the flight component uses, and
/// the reason only a boresight is configured and not a full mounting.
Eigen::Matrix3d starNoiseCov(const Eigen::Vector3d& b) {
  const Eigen::Matrix3d bbt = b * b.transpose();
  return (kStSigmaXy * kStSigmaXy) * (Eigen::Matrix3d::Identity() - bbt) +
         (kStSigmaZ * kStSigmaZ) * bbt;
}

/// One run's realised tracker hardware: a fixed bias per unit, drawn once.
struct StarSystematics {
  Eigen::Vector3d bias[kStarUnits]{};

  static StarSystematics draw(polaris::random::SplitMix64& rng) {
    StarSystematics s{};
    for (int i = 0; i < kStarUnits; ++i) {
      const Eigen::Vector3d direction =
          Eigen::Vector3d(rng.gaussian(), rng.gaussian(), rng.gaussian()).normalized();
      s.bias[i] = (kStBiasBound * rng.uniform()) * direction;
    }
    return s;
  }
};

/// Unit @p unit's reported attitude for truth @p q_true: the truth composed with
/// its fixed bias and this cycle's anisotropic white draw, both as body-frame
/// small-angle rotations — `q_meas = δq(θ) ⊗ q_true`, the same composition
/// `sim/sensors/star_tracker.hpp` uses.
pm::Quaternion measureStar(int unit, const pm::Quaternion& q_true, const StarSystematics& sys,
                           polaris::random::SplitMix64& rng) {
  const Eigen::Vector3d& b = kStarBoresights[unit];
  // Two axes across the boresight and the boresight itself; the white draw is
  // tight across and loose about, which is the anisotropy that makes two
  // non-parallel units worth more than two parallel ones.
  const Eigen::Vector3d c1 = anyPerpendicular(b);
  const Eigen::Vector3d c2 = b.cross(c1).normalized();
  const Eigen::Vector3d white =
      kStWhiteXy * (rng.gaussian() * c1 + rng.gaussian() * c2) + kStWhiteZ * rng.gaussian() * b;
  const Eigen::Vector3d theta = sys.bias[unit] + white;
  const double angle = theta.norm();
  if (!(angle > 0.0)) {
    return q_true;
  }
  return pm::Quaternion::FromAxisAngle(theta / angle, angle) * q_true;
}

/// Campaign result: the per-run error samples and the statistics the
/// requirement is judged on.
struct Campaign {
  std::vector<double> samples;  ///< one error norm [deg] per completed run

  double quantile(double p) const {
    std::vector<double> sorted = samples;
    std::sort(sorted.begin(), sorted.end());
    const auto index = static_cast<std::size_t>(p * static_cast<double>(sorted.size() - 1));
    return sorted[index];
  }

  double median() const { return quantile(0.5); }

  double max() const { return *std::max_element(samples.begin(), samples.end()); }
};

/// Report the distribution to the console, so a run shows the margin rather than
/// just pass/fail (REQ-VV-004).
double report(const Campaign& c, double threshold_deg, const char* label) {
  const double bound = c.max();
  const double margin_pct = 100.0 * (threshold_deg - bound) / threshold_deg;
  std::printf(
      "[%s] N=%zu  median=%.3f deg  p95=%.3f deg  3sigma-bound=%.3f deg"
      "  limit=%.3f deg  margin=%.0f%%\n",
      label, c.samples.size(), c.median(), c.quantile(0.95), bound, threshold_deg, margin_pct);
  return margin_pct;
}

gnc::CoarseAttitudeConfig coarseConfig(double sun_sys, double mag_sys) {
  gnc::CoarseAttitudeConfig cfg{};
  cfg.sigma_sun_white_rad = kSigmaSunWhite;
  cfg.sigma_sun_sys_rad = sun_sys;
  cfg.sigma_mag_white_rad = kSigmaMagWhite;
  cfg.sigma_mag_sys_rad = mag_sys;
  cfg.gyro_arw = kGyroArw;
  cfg.min_sin_angle = kMinSinAngle;
  cfg.triad_gain = kTriadGain;
  cfg.max_coast_s = 2400.0;
  cfg.max_dt_s = 0.5;
  return cfg;
}

gnc::MekfConfig mekfConfig() {
  gnc::MekfConfig cfg{};
  cfg.arw_rad_per_sqrt_s = kGyroArw;
  cfg.rrw_rad_per_s_per_sqrt_s = kGyroRrw;
  cfg.nis_gate = kNisGate;
  cfg.attitude_nis_gate = kAttitudeNisGate;
  cfg.max_coast_s = 300.0;
  cfg.max_dt_s = 0.5;
  return cfg;
}

// ── The shared campaign ─────────────────────────────────────────────────────
//
// One draw per run — geometry, truth motion, systematic biases, gyro bias and
// every white-noise sequence — feeding **both** estimators. That is what makes
// the coarse-vs-fine comparison below a paired one: the two chains see the
// identical measurement stream, so any difference between them is the estimator
// and not the sample. It also halves the work, since the payload requirement
// reads the same fine-mode results rather than re-running the campaign.

/// One run's outcome: what each estimator ended at, and the truth both should
/// have matched.
struct PairedRun {
  pm::Quaternion coarse{};
  pm::Quaternion fine{};
  pm::Quaternion truth{};
  bool ok{false};  ///< false if either chain failed to seed or ended invalid

  /// Sun-sensor suite diagnostics, written only by the handoff sweep: which unit
  /// the §8.2 selection picked on the last cycle, the worst incidence at the
  /// selected unit over the run, and how many times the selection changed. Kept
  /// on the run rather than accumulated globally so the assertions can be stated
  /// on the distribution rather than on a side effect.
  int sun_unit{-1};
  double worst_incidence_rad{0.0};
  int handoffs{0};
};

/// Run the campaign at a given magnetic-systematic level [rad]. The seeds do not
/// depend on it, so two levels give the same geometry and the same noise
/// sequences with only the magnetic bias rescaled — the projection below is
/// therefore paired against the baseline as well.
/// @param sweep_handoff model the six-face sun-sensor suite (§8.2): each cycle's
///        white noise is drawn at the σ the *selected* unit realises at that
///        attitude, and the per-run diagnostics above are filled. The estimators
///        are still told the shipped field-edge σ, exactly as the flight code
///        does — see the suite note above. False reproduces the single-unit
///        campaign the earlier pushes measured, bit for bit.
/// @param rate_scale multiplier on the drawn body rate. The default 1 is the
///        slow controlled-vehicle rate; the handoff case raises it so the
///        selected unit changes **within** a run rather than only between runs.
std::vector<PairedRun> runCampaign(double sun_sys, double mag_sys, double seed_min_observability,
                                   bool sweep_handoff = false, double rate_scale = 1.0,
                                   int steps = kSteps) {
  // The 1σ handed to the MEKF and the Davenport seed per source: white ⊕
  // systematic, exactly as `AttitudeEstimator::refreshCoarseConfig` inflates it.
  // The filter has no way to model a systematic term, so the caller pays for it
  // in R (mekf.hpp, "Measurement noise is the caller's, and white").
  const double sigma_sun_total = std::hypot(kSigmaSunWhite, sun_sys);
  const double sigma_mag_total = std::hypot(kSigmaMagWhite, mag_sys);
  {
    std::vector<PairedRun> result;
    result.reserve(kRuns);

    for (int run = 0; run < kRuns; ++run) {
      polaris::random::SplitMix64 rng(polaris::random::streamSeed(0xC0A125Eu, run));
      RunSetup s = drawRun(rng, sun_sys, mag_sys);
      s.rate *= rate_scale;
      Eigen::Vector3d bias = s.bias;

      PairedRun paired{};
      paired.truth = truthAttitude(s.rate, steps * kDt, s.q0);

      gnc::CoarseAttitudeEstimator coarse(coarseConfig(sun_sys, mag_sys));
      gnc::Mekf fine(mekfConfig());
      if (!coarse.isConfigured() || !fine.isConfigured()) {
        result.push_back(paired);
        continue;
      }

      // Fine cold start on the real path: a Davenport seed from one noisy
      // measurement pair, with the turn-on bias uncertainty and no bias
      // estimate. This draw belongs to the filter alone — the coarse chain has
      // no seed step, it acquires on its own first TRIAD — so it is taken
      // before the loop and the per-cycle draws below stay shared.
      gnc::DavenportInput seed_in{};
      seed_in.count = 2;
      seed_in.min_observability = seed_min_observability;
      seed_in.observations[0].body = pm::Vec3<frames::Body>(measureSun(s, s.q0, rng));
      seed_in.observations[0].reference = pm::Vec3<frames::ECI>(s.sun_eci);
      seed_in.observations[0].sigma_rad = sigma_sun_total;
      seed_in.observations[1].body = pm::Vec3<frames::Body>(measureMag(s, s.q0, rng));
      seed_in.observations[1].reference = pm::Vec3<frames::ECI>(s.mag_eci);
      seed_in.observations[1].sigma_rad = sigma_mag_total;

      gnc::DavenportSolution seed{};
      if (!gnc::davenport(seed_in, seed) ||
          !fine.initialize(epochAt(0.0), seed.attitude, seed.covariance,
                           pm::Vec3<frames::Body>(Eigen::Vector3d::Zero()),
                           (kBiasSigmaInit * kBiasSigmaInit) * Eigen::Matrix3d::Identity())) {
        result.push_back(paired);
        continue;
      }

      gnc::CoarseAttitudeOutput coarse_out{};
      bool fine_ok = true;
      int previous_unit = -1;
      for (int step = 1; step <= steps; ++step) {
        const double t = step * kDt;
        const pm::Quaternion q_true = truthAttitude(s.rate, t, s.q0);

        // The §8.2 suite: which unit sees the Sun best at this attitude, and the
        // σ it realises there. Off the handoff sweep this is the shipped
        // field-edge constant on every cycle, which is what the earlier campaigns
        // measured.
        double white_sigma = kSigmaSunWhite;
        if (sweep_handoff) {
          int unit = -1;
          double incidence = 0.0;
          selectSunUnit(q_true.rotate(s.sun_eci), unit, white_sigma, incidence);
          if (unit >= 0) {
            paired.sun_unit = unit;
            paired.worst_incidence_rad = std::max(paired.worst_incidence_rad, incidence);
            if (previous_unit >= 0 && unit != previous_unit) {
              ++paired.handoffs;
            }
            previous_unit = unit;
          }
        }

        // Drawn once, handed to both chains.
        const Eigen::Vector3d gyro = measureGyro(s.rate, bias, rng);
        const Eigen::Vector3d sun_body = measureSun(s, q_true, rng, white_sigma);
        const Eigen::Vector3d mag_body = measureMag(s, q_true, rng);

        gnc::CoarseAttitudeInput in{};
        in.epoch = epochAt(t);
        in.gyro = pm::Vec3<frames::Body>(gyro);
        in.gyro_valid = true;
        in.sun_body = pm::Vec3<frames::Body>(sun_body);
        in.sun_ref = pm::Vec3<frames::ECI>(s.sun_eci);
        in.sun_valid = true;
        in.mag_body = pm::Vec3<frames::Body>(mag_body);
        in.mag_ref = pm::Vec3<frames::ECI>(s.mag_eci);
        in.mag_valid = true;
        coarse.update(in, coarse_out);

        if (!fine.propagate(epochAt(t), pm::Vec3<frames::Body>(gyro), true)) {
          fine_ok = false;
          break;
        }
        gnc::MekfUpdate up{};
        fine.update(pm::Vec3<frames::Body>(sun_body), pm::Vec3<frames::ECI>(s.sun_eci),
                    sigma_sun_total, up);
        fine.update(pm::Vec3<frames::Body>(mag_body), pm::Vec3<frames::ECI>(s.mag_eci),
                    sigma_mag_total, up);
      }

      paired.coarse = coarse_out.attitude.core();
      paired.fine = fine.attitude().core();
      // Every run's geometry is inside the flight gate by construction, so both
      // chains must end usable — a run that quietly failed would otherwise drop
      // out of the statistics and flatter the result.
      paired.ok = coarse_out.attitude_valid && fine_ok && fine.attitudeValid();
      result.push_back(paired);
    }
    return result;
  }
}

/// The **both-corrections** projection campaign, on the analytic-ephemeris
/// fallback. Run once on first use and shared by the two projection cases below
/// — they compare against the same 800 runs, and at ~15 s a campaign that is a
/// meaningful share of this binary's runtime.
const std::vector<PairedRun>& postCorrectionsFallbackRuns() {
  static const std::vector<PairedRun> runs =
      runCampaign(kSunSysPostAlbedo, kSigmaMagSysPostCal, kSeedMinObservabilityPostCal);
  return runs;
}

/// The reference vehicle's **uncalibrated** budget — the baseline the three
/// projections below measure their improvement against, and the budget the
/// head-to-head and cross-boresight cases characterise. Run once on first use.
///
/// This stopped being the requirement campaign when REQ-ADET-005/006 were
/// enacted at 5°/3° against the post-calibration budget; it is kept because a
/// projection needs something to project *from*, and because the estimator
/// comparison is a cleaner measurement on the budget where the systematic floor
/// dominates.
const std::vector<PairedRun>& campaignRuns() {
  static const std::vector<PairedRun> runs =
      runCampaign(kSunSysUncal, kSigmaMagSys, kSeedMinObservability);
  return runs;
}

/// The **verifying** campaign for REQ-ADET-005/006: the **post-calibration**
/// budget — both calibration items applied, the magnetometer fit's tightened
/// parameters uplinked, and the DE440 tables answering the sun query at grade
/// PRECISE. Run once on first use and shared by the two requirement cases and
/// the degraded-floor case below.
const std::vector<PairedRun>& requirementRuns() {
  static const std::vector<PairedRun> runs =
      runCampaign(kSunSysPostAlbedoTables, kSigmaMagSysPostCal, kSeedMinObservabilityPostCal);
  return runs;
}

/// Body rate for the handoff sweep: 0.5 rad/s ≈ 29 °/s, so a 10 s run turns the
/// vehicle through ~290° and the selected sun sensor changes several times
/// *inside* the run. Not a controlled-pointing rate — it is a slew/detumble one,
/// and that is the point: a handoff is a slew-time event, so the sweep has to be
/// flown at slew rates or it only ever samples one unit per run.
constexpr double kHandoffRateScale = 100.0;
/// 5 s at 10 Hz. Longer than the steady-state runs because the transient this
/// case is about — the cycles either side of a handoff — has to be *inside* the
/// sampled window rather than before it, and short enough that the campaign
/// stays a fraction of this binary's runtime. At the slew rate above it turns
/// the vehicle through ~145°, which crosses two or three unit boundaries.
constexpr int kHandoffSteps = 50;

/// The **handoff** campaign: the same post-calibration budget REQ-ADET-005/006
/// are verified on, flown at slew rates through the six-face sun-sensor suite so
/// the active unit changes within each run (§8.2). Run once on first use.
const std::vector<PairedRun>& handoffRuns() {
  static const std::vector<PairedRun> runs =
      runCampaign(kSunSysPostAlbedoTables, kSigmaMagSysPostCal, kSeedMinObservabilityPostCal,
                  /*sweep_handoff=*/true, kHandoffRateScale, kHandoffSteps);
  return runs;
}

// ── The star-tracker campaign (REQ-ADET-007, REQ-PAY-001) ───────────────────
//
// The §8.2 ladder's top rung: with at least one tracker fused, the sun and
// magnetic pairs are **not** folded into the filter at all. So this campaign
// deliberately does not feed them — modelling them as measurements would be
// modelling a mode the vehicle does not fly, and it would flatter nothing but
// would make the measured number un-attributable to the trackers.

/// Pairs collected by the modelled ST_ALIGN_CAL window: the reference vehicle's
/// `StAlignMinSamples`, i.e. 10 s at the 10 Hz rate with both units solving.
constexpr int kAlignSamples = 100;

/// REQ-PAY-001: 10% of the generic imager's smallest **full** field of view.
/// `2 x min(half_x, half_y) = 2 x 4° = 8°`, so the bound is 0.8°.
constexpr double kPayloadLimitDeg = 0.8;

struct StarRun {
  pm::Quaternion fine{};
  pm::Quaternion truth{};
  double align_residual_rad{0.0};  ///< RMS residual the fit reported
  double align_misalign_rad{0.0};  ///< total rotation the correction removed
  bool aligned{false};             ///< the fit was accepted and applied
  bool ok{false};
};

/// Fly the fine mode on @p star_units trackers.
///
/// @param star_units 1 or 2. At 2 the second unit is first calibrated against the
///        king through the **real** `gnc::StAlignmentAccumulator`, exactly as the
///        commanded on-orbit flow does, so the residual this campaign carries is
///        the residual the fit actually leaves rather than a stipulated one.
std::vector<StarRun> runStarCampaign(int star_units) {
  std::vector<StarRun> result;
  result.reserve(kRuns);

  for (int run = 0; run < kRuns; ++run) {
    // A master seed of its own, so this campaign's draws neither disturb nor are
    // disturbed by the SS+MAG ones above — the two are not paired and pretending
    // otherwise by sharing a stream would only couple them.
    polaris::random::SplitMix64 rng(polaris::random::streamSeed(0x57A2C0DEu, run));
    RunSetup s = drawRun(rng, kSunSysPostAlbedoTables, kSigmaMagSysPostCal);
    const StarSystematics stars = StarSystematics::draw(rng);
    Eigen::Vector3d bias = s.bias;

    // **Per-unit substreams, so the campaigns are paired.** Sharing one stream
    // across both trackers makes the 1-unit and 2-unit runs diverge from the first
    // cycle — unit 0 draws different noise depending on whether unit 1 exists —
    // and the "what does the second tracker buy" comparison below then compares
    // two different samples rather than the same run flown twice. The alignment
    // window gets its own pair for the same reason: it runs only in the 2-unit
    // campaign, and drawing from the measurement stream would shift it.
    polaris::random::SplitMix64 meas_rng[kStarUnits] = {
        polaris::random::SplitMix64(polaris::random::streamSeed(0x57A20000u, run)),
        polaris::random::SplitMix64(polaris::random::streamSeed(0x57A20001u, run))};
    polaris::random::SplitMix64 align_rng[kStarUnits] = {
        polaris::random::SplitMix64(polaris::random::streamSeed(0x57A21000u, run)),
        polaris::random::SplitMix64(polaris::random::streamSeed(0x57A21001u, run))};

    StarRun out{};
    out.truth = truthAttitude(s.rate, kSteps * kDt, s.q0);

    gnc::Mekf fine(mekfConfig());
    if (!fine.isConfigured()) {
      result.push_back(out);
      continue;
    }

    // --- The commanded inter-tracker alignment, flown before the run ---------
    gnc::StAlignmentResult alignment{};
    if (star_units >= 2) {
      gnc::StAlignmentConfig cfg{};
      cfg.min_samples = static_cast<std::uint32_t>(kAlignSamples);
      cfg.max_residual_rad = 5.0e-4;  // the shipped StAlignMaxResidualRad
      cfg.min_eigen_gap = 0.9;        // the shipped StAlignMinEigenGap
      gnc::StAlignmentAccumulator accumulator(cfg);
      for (int k = 0; k < kAlignSamples; ++k) {
        const pm::Quaternion q = truthAttitude(s.rate, k * kDt, s.q0);
        // **Uncorrected** readings, as the flight tap feeds them.
        (void)accumulator.addSample(
            pm::Quat<frames::Body, frames::ECI>(measureStar(0, q, stars, align_rng[0])),
            pm::Quat<frames::Body, frames::ECI>(measureStar(1, q, stars, align_rng[1])));
      }
      out.aligned = accumulator.fit(alignment) == gnc::StAlignmentRejection::kNone;
      out.align_residual_rad = alignment.residual_angle_rad;
      out.align_misalign_rad = alignment.misalignment_angle_rad;
    }

    // --- Cold start from the king's own solution and its own R --------------
    // No Davenport solve and no coarse floor: a tracker's solution *is* a seed,
    // which is what makes the top rung reachable in eclipse (the flight component
    // takes the same path).
    const pm::Quaternion seed_meas = measureStar(0, s.q0, stars, meas_rng[0]);
    if (!fine.initialize(epochAt(0.0), pm::Quat<frames::Body, frames::ECI>(seed_meas),
                         starNoiseCov(kStarBoresights[0]),
                         pm::Vec3<frames::Body>(Eigen::Vector3d::Zero()),
                         (kBiasSigmaInit * kBiasSigmaInit) * Eigen::Matrix3d::Identity())) {
      result.push_back(out);
      continue;
    }

    bool fine_ok = true;
    for (int step = 1; step <= kSteps; ++step) {
      const double t = step * kDt;
      const pm::Quaternion q_true = truthAttitude(s.rate, t, s.q0);
      const Eigen::Vector3d gyro = measureGyro(s.rate, bias, rng);
      if (!fine.propagate(epochAt(t), pm::Vec3<frames::Body>(gyro), true)) {
        fine_ok = false;
        break;
      }
      for (int unit = 0; unit < star_units; ++unit) {
        pm::Quat<frames::Body, frames::ECI> measured(
            measureStar(unit, q_true, stars, meas_rng[unit]));
        if (unit != 0) {
          // The one application point: the non-king unit is stated in the king's
          // frame before it reaches the filter. `applyStAlignment` passes it
          // through when nothing was fitted, so an uncalibrated vehicle takes the
          // same path — which is the case this campaign's alignment-refused runs
          // exercise for free.
          measured = gnc::applyStAlignment(alignment, measured);
        }
        gnc::MekfUpdate up{};
        fine.updateAttitude(measured, starNoiseCov(kStarBoresights[unit]), up);
      }
    }

    out.fine = fine.attitude().core();
    out.ok = fine_ok && fine.attitudeValid();
    result.push_back(out);
  }
  return result;
}

/// The single- and dual-tracker campaigns. Run once on first use and shared.
const std::vector<StarRun>& singleStarRuns() {
  static const std::vector<StarRun> runs = runStarCampaign(1);
  return runs;
}

const std::vector<StarRun>& dualStarRuns() {
  static const std::vector<StarRun> runs = runStarCampaign(2);
  return runs;
}

Campaign starErrorNorms(const std::vector<StarRun>& runs) {
  Campaign c;
  c.samples.reserve(runs.size());
  for (const StarRun& r : runs) {
    c.samples.push_back(errorNormDeg(r.fine, r.truth));
  }
  return c;
}

/// The campaign's coarse (or fine) error norms as a @ref Campaign.
Campaign errorNorms(const std::vector<PairedRun>& runs, bool fine) {
  Campaign c;
  c.samples.reserve(runs.size());
  for (const PairedRun& r : runs) {
    c.samples.push_back(errorNormDeg(fine ? r.fine : r.coarse, r.truth));
  }
  return c;
}

/// Fail the calling test if any run did not produce a usable pair.
void requireAllRunsValid(const std::vector<PairedRun>& runs) {
  ASSERT_EQ(runs.size(), static_cast<std::size_t>(kRuns));
  for (std::size_t i = 0; i < runs.size(); ++i) {
    ASSERT_TRUE(runs[i].ok) << "run " << i << " left an estimator invalid";
  }
}

// ── REQ-ADET-005: coarse-mode knowledge accuracy ────────────────────────────

TEST(AttitudeAccuracyMonteCarlo, CoarseModeKnowledgeErrorNorm) {
  RecordProperty("verifies", "REQ-ADET-005");
  requireAllRunsValid(requirementRuns());
  const Campaign campaign = errorNorms(requirementRuns(), false);

  RecordProperty("margin_pct",
                 static_cast<int>(report(campaign, kCoarseLimitDeg, "REQ-ADET-005 coarse")));

  EXPECT_LE(campaign.max(), kCoarseLimitDeg)
      << "3σ knowledge-error bound over " << kRuns << " runs";
  // REQ-ADET-005 carries `margin_required: 20 %`, which was a printed number and
  // nothing more until this gate. Enforcing it here is what makes the margin a
  // property CI defends rather than a claim the RVTM repeats.
  EXPECT_LE(campaign.max(), kMarginFraction * kCoarseLimitDeg)
      << "bound " << campaign.max() << " deg leaves "
      << 100.0 * (kCoarseLimitDeg - campaign.max()) / kCoarseLimitDeg << "% margin against the "
      << kCoarseLimitDeg << " deg threshold; REQ-ADET-005 requires 20%";
  // Sensitivity floor. Calibrated, the remaining floor is the albedo dispersion
  // and the magnetometer's post-fit residual, which put the coarse median near
  // 0.85° — a test bug that zeroed the systematic draws would sail through the
  // bound above and is caught here instead. Set well below the measurement so it
  // guards against a collapse to zero, not against a legitimate improvement.
  EXPECT_GT(campaign.median(), 0.2) << "median error implausibly small — is the noise wired in?";
}

// ── REQ-ADET-006: fine-mode knowledge accuracy, same SS+MAG+IMU suite ───────

TEST(AttitudeAccuracyMonteCarlo, FineModeKnowledgeErrorNorm) {
  RecordProperty("verifies", "REQ-ADET-006");
  requireAllRunsValid(requirementRuns());
  const Campaign campaign = errorNorms(requirementRuns(), true);

  RecordProperty("margin_pct",
                 static_cast<int>(report(campaign, kFineLimitDeg, "REQ-ADET-006 fine MEKF")));

  EXPECT_LE(campaign.max(), kFineLimitDeg) << "3σ knowledge-error bound over " << kRuns << " runs";
  // As REQ-ADET-005: the 20% `margin_required` is enforced, not just reported.
  EXPECT_LE(campaign.max(), kMarginFraction * kFineLimitDeg)
      << "bound " << campaign.max() << " deg leaves "
      << 100.0 * (kFineLimitDeg - campaign.max()) / kFineLimitDeg << "% margin against the "
      << kFineLimitDeg << " deg threshold; REQ-ADET-006 requires 20%";
  // Same sensitivity floor as the coarse campaign; measured median is 0.51°.
  EXPECT_GT(campaign.median(), 0.1) << "median error implausibly small — is the noise wired in?";
}

// ── §8.2: the sun-sensor suite and its handoff geometry ─────────────────────

/// The coverage claim the handoff argument rests on, checked directly rather
/// than inherited from the vehicle-config comment that derives it.
///
/// **Two separate facts, and conflating them is the trap.**
///  - *Coverage geometry:* every direction is inside some unit's 60° cone, and
///    the **best available** unit is never worse than the body-diagonal
///    arccos(1/sqrt(3)) = 54.736°. That is a property of the six-face layout
///    alone, independent of how the vehicle chooses among the units in view.
///  - *Accuracy bound:* the shipped `SigmaSunWhiteRad` is the FSS's **60°
///    field-edge** figure, and that — not 54.736° — is what bounds the error of
///    whatever unit the flight selector actually picks. The selector orders on
///    reported sigma with ties to the lowest index, and sigma is a two-regime
///    step, so under a tie it can take a unit aimed considerably worse than the
///    best available, anywhere out to the 60° edge. The edge figure covers that;
///    the body-diagonal figure would not.
///
/// Both are asserted below, separately and on the right quantity.
TEST(AttitudeAccuracyMonteCarlo, SunSuiteCoversEveryAttitude) {
  RecordProperty("verifies", "REQ-ADET-010");
  polaris::random::SplitMix64 rng(polaris::random::streamSeed(0x5A0E5u, 1));
  double worst_best_available = 0.0;  // coverage geometry
  double worst_selected = 0.0;        // what the flight selector actually takes
  int uncovered = 0;
  int inner_regime = 0;
  constexpr int kDirections = 200000;
  for (int i = 0; i < kDirections; ++i) {
    const Eigen::Vector3d u =
        Eigen::Vector3d(rng.gaussian(), rng.gaussian(), rng.gaussian()).normalized();

    // Best available, over every unit with the Sun in field: the layout property.
    double best = M_PI;
    for (int k = 0; k < kSunUnits; ++k) {
      const double angle = std::atan2(u.cross(kSunBoresights[k]).norm(), u.dot(kSunBoresights[k]));
      if (angle <= kFssHalfFovRad) {
        best = std::min(best, angle);
      }
    }

    int unit = -1;
    double sigma = 0.0;
    double incidence = 0.0;
    selectSunUnit(u, unit, sigma, incidence);
    if (unit < 0) {
      ++uncovered;
      continue;
    }
    worst_best_available = std::max(worst_best_available, best);
    worst_selected = std::max(worst_selected, incidence);
    if (sigma == kFssSigmaInner) {
      ++inner_regime;
    }
  }
  const double best_deg = worst_best_available / kDeg;
  const double selected_deg = worst_selected / kDeg;
  std::printf(
      "[sun suite] %d directions: uncovered=%d  worst best-available=%.3f deg  worst "
      "selected=%.3f deg  %.1f%% in the 45 deg regime\n",
      kDirections, uncovered, best_deg, selected_deg,
      100.0 * static_cast<double>(inner_regime) / static_cast<double>(kDirections));

  EXPECT_EQ(uncovered, 0) << "the six-face suite left a direction with no sun sensor in view";

  // Coverage geometry: arccos(1/sqrt(3)) = 54.7356 deg, the body diagonal. The
  // lower bound is there to prove the sweep reached the worst-case direction.
  EXPECT_LT(best_deg, 54.7357);
  EXPECT_GT(best_deg, 54.0) << "the sweep never reached the worst-case direction";

  // Accuracy bound: the *selected* unit is bounded by the field edge and by
  // nothing tighter, which is precisely why the budget is derived at the edge.
  // Asserting the body-diagonal figure here instead would be asserting a
  // selector the vehicle does not implement.
  EXPECT_LT(worst_selected, kFssHalfFovRad) << "a selected unit was outside its own field of view";
  EXPECT_GT(selected_deg, best_deg)
      << "the tie-breaking selector never took a unit worse than the best available — "
         "either the selector or this model has changed";
}

/// **The handoff sweep** (§8.2): the requirement campaign re-flown at slew rates
/// through the six-face suite, so the active sun sensor changes several times
/// inside every run and the estimators fly across those changes.
///
/// Both REQ-ADET-005 and REQ-ADET-006 must still hold **with their 20% margin**.
/// This is the evidence that the multi-unit suite did not buy coverage at the
/// cost of accuracy: a handoff moves the measurement to a unit at a different
/// incidence, and the estimators are never told which unit they are reading.
///
/// Two things about the setup are worth naming. The rate is a *slew* rate, not a
/// pointing one, because a handoff is a slew-time event; at controlled-pointing
/// rates the suite geometry is swept between runs but never within one. And the
/// run is 10 s rather than 1.5 s so the cycles either side of a handoff are
/// inside the sampled window instead of before it.
TEST(AttitudeAccuracyMonteCarlo, KnowledgeHoldsAcrossSunSensorHandoffs) {
  RecordProperty("verifies", "REQ-ADET-010");
  requireAllRunsValid(handoffRuns());

  // The sweep has to have actually swept, or the thresholds below are being held
  // by a campaign that never handed off.
  int runs_with_handoff = 0;
  int total_handoffs = 0;
  double worst_incidence = 0.0;
  bool unit_selected[kSunUnits] = {};
  for (const PairedRun& r : handoffRuns()) {
    total_handoffs += r.handoffs;
    runs_with_handoff += (r.handoffs > 0) ? 1 : 0;
    worst_incidence = std::max(worst_incidence, r.worst_incidence_rad);
    ASSERT_GE(r.sun_unit, 0) << "a run ended with no sun sensor in view";
    unit_selected[r.sun_unit] = true;
  }
  std::printf(
      "[handoff] %d/%d runs handed off, %d handoffs total, worst selected incidence %.2f deg\n",
      runs_with_handoff, kRuns, total_handoffs, worst_incidence / kDeg);

  EXPECT_GT(runs_with_handoff, kRuns / 2)
      << "most runs never changed sun sensor: the sweep is not sweeping";
  for (int i = 0; i < kSunUnits; ++i) {
    EXPECT_TRUE(unit_selected[i]) << "sun sensor " << i << " was never the selected unit";
  }
  EXPECT_LT(worst_incidence, kFssHalfFovRad) << "a selected unit was outside its own field of view";

  const Campaign coarse = errorNorms(handoffRuns(), false);
  const Campaign fine = errorNorms(handoffRuns(), true);
  report(coarse, kCoarseLimitDeg, "REQ-ADET-005 coarse, handoff sweep");
  report(fine, kFineLimitDeg, "REQ-ADET-006 fine, handoff sweep");

  EXPECT_LE(coarse.max(), kMarginFraction * kCoarseLimitDeg)
      << "coarse bound " << coarse.max() << " deg across sun-sensor handoffs does not keep the "
      << "20% margin against " << kCoarseLimitDeg << " deg";
  EXPECT_LE(fine.max(), kMarginFraction * kFineLimitDeg)
      << "fine bound " << fine.max() << " deg across sun-sensor handoffs does not keep the "
      << "20% margin against " << kFineLimitDeg << " deg";
}

// ── Head to head: does the filter actually earn its keep? ───────────────────

TEST(AttitudeAccuracyMonteCarlo, FineModeBeatsCoarseRunForRun) {
  RecordProperty("verifies", "REQ-ADET-006");
  requireAllRunsValid(campaignRuns());

  // Both chains ran on the identical measurement stream, so this is a *paired*
  // comparison and the right statistic is the **sign test** on the per-run
  // winner: distribution-free, and it uses the pairing, which comparing two
  // medians throws away. Under "the two are equally good" the win count is
  // Binomial(N, 0.5) with σ = √(N/4) = 14 runs, i.e. 1.8% of N — so a win
  // fraction above 0.55 is roughly 28σ from a coin flip and cannot be sampling
  // noise. The threshold is deliberately far above 0.5 for that reason and not
  // because the measured value is marginal (it is not: see below).
  int fine_wins = 0;
  std::vector<double> gaps;
  gaps.reserve(kRuns);
  for (const PairedRun& r : campaignRuns()) {
    const double coarse_deg = errorNormDeg(r.coarse, r.truth);
    const double fine_deg = errorNormDeg(r.fine, r.truth);
    gaps.push_back(coarse_deg - fine_deg);
    if (fine_deg < coarse_deg) {
      ++fine_wins;
    }
  }
  std::sort(gaps.begin(), gaps.end());

  const double win_fraction = static_cast<double>(fine_wins) / static_cast<double>(kRuns);
  const double median_gap = gaps[gaps.size() / 2];
  std::printf(
      "[head-to-head] fine wins %d/%d runs (%.1f%%)  median gap=%.3f deg"
      "  worst case for fine=%.3f deg\n",
      fine_wins, kRuns, 100.0 * win_fraction, median_gap, -gaps.front());
  RecordProperty("fine_win_pct", static_cast<int>(100.0 * win_fraction));

  EXPECT_GT(win_fraction, 0.55) << "the MEKF does not beat the fixed-gain blend run for run";
  EXPECT_GT(median_gap, 0.0) << "median per-run improvement is not positive";
}

// ── Cross-boresight projection of the fine-mode error — informative ─────────
//
// **This does not verify REQ-PAY-001.** That requirement is stated in fine mode
// with star trackers fused (the mode a payload is actually operated in) and at
// 10% of the sensor's own field of view; it is verified per sensor by the §8.2
// fusion push, alongside REQ-ADET-007. What this test does is measure the same
// projection on the SS+MAG+IMU mode this campaign *can* reach, which is worth
// keeping for two reasons that have nothing to do with the threshold:
//
//  - it pins the projection maths against the analytic expectation (the ratio
//    of medians should be the π/4 an isotropically directed error gives), and
//  - it asserts the campaign has **no preferred body axis**, by measuring a
//    canted mounting alongside the reference one. A mounting-dependent answer
//    would invalidate the vehicle-level numbers too, so this is a check on the
//    campaign itself as much as on the metric.
//
// The metric is defined in docs/requirements/payload.rst
// (`pay-cross-boresight-metric`).

/// The mounted boresight in body axes, taken from the payload model itself
/// rather than hand-written, so the projection uses the same +Z convention the
/// sim flies (`sim/sensors/payload_sensor.hpp`).
Eigen::Vector3d payloadBoresightBody(const Eigen::Matrix3d& mounting_dcm) {
  polaris::sim::sensors::PayloadSensorSpec spec{};
  spec.half_fov_x_rad = 5.0 * kDeg;
  spec.half_fov_y_rad = 4.0 * kDeg;
  return polaris::sim::sensors::PayloadSensor(spec, mounting_dcm).boresightBody();
}

/// Angle [deg] between the boresight as the estimate places it and as the truth
/// places it, both in ECI — the metric REQ-PAY-001 is written on, measured here
/// on a mode that requirement is not stated in.
double boresightErrorDeg(const Eigen::Vector3d& boresight_body, const pm::Quaternion& est,
                         const pm::Quaternion& q_true) {
  const Eigen::Vector3d estimated = est.inverse().rotate(boresight_body).normalized();
  const Eigen::Vector3d actual = q_true.inverse().rotate(boresight_body).normalized();
  return std::atan2(estimated.cross(actual).norm(), estimated.dot(actual)) / kDeg;
}

TEST(AttitudeAccuracyMonteCarlo, CrossBoresightProjectionOfFineModeError) {
  // Two mountings: the reference vehicle's (identity — boresight along body +Z,
  // nadir in the nominal Earth-pointing attitude) and a deliberately canted one.
  // The second is the mount-independence check described above.
  const Eigen::Vector3d nadir_mount = payloadBoresightBody(Eigen::Matrix3d::Identity());
  const Eigen::Vector3d canted_mount = payloadBoresightBody(
      pm::Quaternion::FromAxisAngle(Eigen::Vector3d(0.577, 0.577, 0.577).normalized(), 35.0 * kDeg)
          .toRotationMatrix());

  Campaign campaign;
  Campaign canted;
  Campaign norms;  ///< the same runs' total error norms, for the bound below
  campaign.samples.reserve(kRuns);
  canted.samples.reserve(kRuns);
  norms.samples.reserve(kRuns);

  requireAllRunsValid(campaignRuns());
  for (const PairedRun& r : campaignRuns()) {
    campaign.samples.push_back(boresightErrorDeg(nadir_mount, r.fine, r.truth));
    canted.samples.push_back(boresightErrorDeg(canted_mount, r.fine, r.truth));
    norms.samples.push_back(errorNormDeg(r.fine, r.truth));
  }

  ASSERT_EQ(campaign.samples.size(), static_cast<std::size_t>(kRuns));
  // No threshold: reported for the record, not judged. The absolute bound this
  // used to assert belonged to the withdrawn SS+MAG-anchored form of
  // REQ-PAY-001.
  std::printf(
      "[cross-boresight, informative] N=%d  median=%.3f deg  p95=%.3f deg  max=%.3f deg"
      "  (total norm median=%.3f deg)\n",
      kRuns, campaign.median(), campaign.quantile(0.95), campaign.max(), norms.median());
  std::printf("[cross-boresight, canted mount] median=%.3f deg  p95=%.3f deg  max=%.3f deg\n",
              canted.median(), canted.quantile(0.95), canted.max());

  // The cross-boresight error is a component of the total, so it can never
  // exceed it — run by run, not just in the aggregate. A metric that came out
  // *larger* would mean the projection is wrong, which no aggregate bound would
  // catch.
  for (std::size_t i = 0; i < campaign.samples.size(); ++i) {
    ASSERT_LE(campaign.samples[i], norms.samples[i] + 1.0e-9)
        << "run " << i << ": cross-boresight error exceeds the total error norm";
  }

  // Mount independence: the two bounds are the same distribution sampled with a
  // different fixed axis, so they agree to within sampling noise on a tail of
  // 800 draws. A wide tolerance on purpose — this asserts "no preferred body
  // axis", not "identical", and a tight bound here would be a flaky test.
  EXPECT_NEAR(canted.max(), campaign.max(), 0.25 * campaign.max())
      << "cross-boresight bound depends on the mounting — the campaign has a "
         "preferred body axis";

  // Sensitivity floor, as in the two vehicle-level campaigns: this budget cannot
  // place a boresight to a fraction of a degree.
  EXPECT_GT(campaign.median(), 1.5) << "median error implausibly small — is the noise wired in?";
}

// ── REQ-ADET-007: fine-mode knowledge accuracy with star trackers ───────────

TEST(AttitudeAccuracyMonteCarlo, StarTrackerFineModeKnowledgeErrorNorm) {
  RecordProperty("verifies", "REQ-ADET-007");
  const std::vector<StarRun>& runs = dualStarRuns();
  ASSERT_EQ(runs.size(), static_cast<std::size_t>(kRuns));
  for (std::size_t i = 0; i < runs.size(); ++i) {
    ASSERT_TRUE(runs[i].ok) << "run " << i << " left the filter invalid";
  }

  const Campaign campaign = starErrorNorms(runs);
  RecordProperty("margin_pct",
                 static_cast<int>(report(campaign, kStarLimitDeg, "REQ-ADET-007 fine + 2 ST")));

  EXPECT_LE(campaign.max(), kStarLimitDeg)
      << "3-sigma attitude-knowledge error norm exceeds REQ-ADET-007";
  // The same CI-enforced margin every other accuracy requirement carries: the
  // requirement declares 20%, and a margin nothing checks is a number rather than
  // a property (conf.py has no margin_achieved gate).
  EXPECT_LE(campaign.max(), kMarginFraction * kStarLimitDeg)
      << "REQ-ADET-007 holds but with less than the declared 20% margin";

  // Sensitivity floor. If the tracker noise were not wired in, every run would
  // return the truth to round-off and the bound above would pass vacuously — which
  // is the failure mode a threshold test cannot see on its own.
  EXPECT_GT(campaign.median(), 1.0e-4)
      << "median error implausibly small — is the tracker noise wired in?";
}

TEST(AttitudeAccuracyMonteCarlo, SecondStarTrackerCoversTheFirstsWeakAxis) {
  // The configuration decision, measured rather than asserted from the datasheet:
  // what does the second, non-parallel tracker actually buy?
  //
  // **Paired, run for run.** Both campaigns draw unit 0's measurement noise from
  // its own substream, so the single-tracker run and the dual-tracker run see the
  // *identical* truth, systematics and unit-0 sequence — the only difference is
  // whether unit 1 is fused. That makes the comparison a paired one, and a paired
  // win fraction is a real statistic where a 3% gap between two independent
  // medians is sampling noise wearing a number.
  const std::vector<StarRun>& single = singleStarRuns();
  const std::vector<StarRun>& dual = dualStarRuns();
  ASSERT_EQ(single.size(), dual.size());

  const Campaign single_norms = starErrorNorms(single);
  const Campaign dual_norms = starErrorNorms(dual);
  (void)report(single_norms, kStarLimitDeg, "single ST (informative)");
  (void)report(dual_norms, kStarLimitDeg, "dual ST (REQ-ADET-007)");

  int wins = 0;
  std::vector<double> improvement;
  improvement.reserve(single.size());
  for (std::size_t i = 0; i < single.size(); ++i) {
    ASSERT_TRUE(single[i].ok && dual[i].ok) << "run " << i;
    const double a = errorNormDeg(single[i].fine, single[i].truth);
    const double b = errorNormDeg(dual[i].fine, dual[i].truth);
    wins += (b < a) ? 1 : 0;
    improvement.push_back(a - b);
  }
  Campaign gains;
  gains.samples = improvement;
  const double win_fraction = static_cast<double>(wins) / static_cast<double>(single.size());
  std::printf(
      "[dual vs single ST, paired] dual wins %d/%zu = %.1f%%  median gain=%.4f deg"
      "  bound %.4f -> %.4f deg\n",
      wins, single.size(), 100.0 * win_fraction, gains.median(), single_norms.max(),
      dual_norms.max());

  // Under "the second tracker changes nothing" the win count is Binomial(N, 1/2),
  // i.e. 50% ± 1.8% at N = 800. Anything at or above 60% is decisively better and
  // leaves room for the fact that the two units share a floor neither can remove
  // (the king's bias), which is what keeps this well short of 100%.
  EXPECT_GT(win_fraction, 0.60)
      << "the second tracker wins only " << 100.0 * win_fraction
      << "% of paired runs — check the boresight geometry and that R is anisotropic";
  EXPECT_GT(gains.median(), 0.0) << "the median paired improvement is not positive";

  // **The floor is the king's own bias, and no configuration removes it.** The
  // king's mounting *is* the body frame, and the payload is mounted against the
  // physical structure rather than the king's optical axis, so that bias is a real
  // knowledge error. Its per-axis 1σ is bias_bound/3, and the norm of an
  // isotropic 3-vector error sits near 1.6σ at the median — so a dual-tracker
  // median far *below* that would mean the campaign is averaging down a systematic
  // that does not average down, which is the one way this result could be wrong in
  // the flattering direction.
  const double king_bias_floor_deg = kStBiasPerAxis / kDeg;
  EXPECT_GT(dual_norms.median(), king_bias_floor_deg)
      << "dual-tracker median is below the king's own bias — the campaign is "
         "averaging down a systematic that does not average down";
}

TEST(AttitudeAccuracyMonteCarlo, InterTrackerAlignmentFitsOnEveryRun) {
  // The calibration the dual-tracker result rests on. If it were quietly refusing,
  // the campaign above would be flying two *uncalibrated* trackers whose fixed
  // biases differ by ~60 arcsec, and the filter would spend every cycle splitting
  // the difference between two units that disagree — a worse answer than one
  // tracker, and one no threshold test on the norm would attribute correctly.
  const std::vector<StarRun>& runs = dualStarRuns();
  Campaign residual;
  Campaign misalignment;
  int fitted = 0;
  for (const StarRun& r : runs) {
    if (r.aligned) {
      ++fitted;
      residual.samples.push_back(r.align_residual_rad / kArcsec);
      misalignment.samples.push_back(r.align_misalign_rad / kArcsec);
    }
  }
  ASSERT_GT(fitted, 0);
  std::printf(
      "[inter-tracker alignment] fitted %d/%d runs  residual median=%.1f arcsec max=%.1f"
      "  misalignment median=%.1f arcsec max=%.1f\n",
      fitted, kRuns, residual.median(), residual.max(), misalignment.median(), misalignment.max());

  EXPECT_EQ(fitted, kRuns) << "the alignment fit was refused on some runs — a window of "
                           << kAlignSamples << " healthy pairs must always fit";
  // The fit is an average of N samples of a constant, so its residual is the
  // per-sample dispersion and its *error* falls as 1/sqrt(N). What is asserted is
  // the residual against the gate that ships, which is what the ground grades on.
  EXPECT_LT(residual.max() * kArcsec, 5.0e-4)
      << "fitted residual exceeds the shipped StAlignMaxResidualRad";
  // And it must actually have something to find: the two units' biases differ by
  // tens of arcseconds by construction, so a misalignment near zero would mean the
  // campaign is not injecting per-unit biases at all.
  EXPECT_GT(misalignment.median(), 5.0)
      << "estimated misalignment implausibly small — are the per-unit biases wired in?";
}

// ── REQ-PAY-001: payload cross-boresight knowledge, in the mode it images in ──
//
// The requirement is stated on the **cross-boresight** error — the part of the
// knowledge error that displaces the scene on the focal plane — and is evaluated
// in the mode a payload is actually operated in, which is fine mode with star
// trackers fused. Push 45 could only measure the projection on SS+MAG+IMU and
// recorded it as informative evidence; this is the verification.

TEST(AttitudeAccuracyMonteCarlo, PayloadCrossBoresightWithStarTrackers) {
  RecordProperty("verifies", "REQ-PAY-001");
  const Eigen::Vector3d nadir_mount = payloadBoresightBody(Eigen::Matrix3d::Identity());
  const Eigen::Vector3d canted_mount = payloadBoresightBody(
      pm::Quaternion::FromAxisAngle(Eigen::Vector3d(0.577, 0.577, 0.577).normalized(), 35.0 * kDeg)
          .toRotationMatrix());

  const std::vector<StarRun>& runs = dualStarRuns();
  Campaign campaign;
  Campaign canted;
  Campaign norms;
  for (const StarRun& r : runs) {
    ASSERT_TRUE(r.ok);
    campaign.samples.push_back(boresightErrorDeg(nadir_mount, r.fine, r.truth));
    canted.samples.push_back(boresightErrorDeg(canted_mount, r.fine, r.truth));
    norms.samples.push_back(errorNormDeg(r.fine, r.truth));
  }

  RecordProperty("margin_pct", static_cast<int>(report(campaign, kPayloadLimitDeg,
                                                       "REQ-PAY-001 cross-boresight")));
  EXPECT_LE(campaign.max(), kPayloadLimitDeg)
      << "3-sigma cross-boresight knowledge error exceeds 10% of the imager's "
         "smallest full field of view";
  EXPECT_LE(campaign.max(), kMarginFraction * kPayloadLimitDeg)
      << "REQ-PAY-001 holds but with less than the declared 20% margin";

  // The cross-boresight error is a *component* of the total norm, so it can never
  // exceed it — run by run, not merely in the aggregate. A metric that came out
  // larger would mean the projection is wrong, which no aggregate bound catches.
  for (std::size_t i = 0; i < campaign.samples.size(); ++i) {
    ASSERT_LE(campaign.samples[i], norms.samples[i] + 1.0e-12)
        << "run " << i << ": cross-boresight error exceeds the total error norm";
  }

  // Mount independence, as in the Push 45 projection: a mounting-dependent result
  // would mean the campaign has a preferred body axis, which would invalidate the
  // vehicle-level numbers too. Wide tolerance on purpose — this asserts "no
  // preferred axis", not "identical".
  EXPECT_NEAR(canted.max(), campaign.max(), 0.35 * campaign.max())
      << "cross-boresight bound depends on the mounting — the campaign has a "
         "preferred body axis";
  std::printf("[cross-boresight, canted mount] median=%.4f deg  max=%.4f deg\n", canted.median(),
              canted.max());
}

// ── Post-magnetometer-calibration projection — informative ──────────────────
//
// **This verifies nothing**, and it is kept for what it explains rather than for
// what it guards. REQ-ADET-005/006 are enacted at 5°/3° and verified above, on
// the post-calibration budget. What this case answers is why the tightening needed
// *both* calibration items rather than one: *how far toward 5°/3° does the
// magnetometer calibration alone get us?* The answer — the medians move a long
// way and the tails barely at all — is the systematic-floor argument, and it is
// the evidence that sequenced the albedo correction next.
//
// The same 800 runs with one substitution — the magnetic systematic replaced by
// the residual `lib/gnc/mag_calibration` measures on the reference budget
// (`kSigmaMagSysPostCal`). Everything else, including the 2.04° sun systematic,
// is untouched, which is the point: §8.1 commits the 5°/3° tightening to items
// (1) **and** (2), and this case shows what item (1) buys on its own.

TEST(AttitudeAccuracyMonteCarlo, PostMagCalibrationProjection) {
  const std::vector<PairedRun> runs =
      runCampaign(kSunSysUncal, kSigmaMagSysPostCal, kSeedMinObservabilityPostCal);
  ASSERT_EQ(runs.size(), static_cast<std::size_t>(kRuns));
  for (std::size_t i = 0; i < runs.size(); ++i) {
    ASSERT_TRUE(runs[i].ok) << "run " << i << " left an estimator invalid";
  }

  const Campaign coarse = errorNorms(runs, false);
  const Campaign fine = errorNorms(runs, true);
  const Campaign coarse_now = errorNorms(campaignRuns(), false);
  const Campaign fine_now = errorNorms(campaignRuns(), true);

  // The committed post-calibration thresholds of §8.1, printed alongside the
  // projection so the report shows the remaining gap rather than only the gain.
  constexpr double kCommittedCoarseDeg = 5.0;
  constexpr double kCommittedFineDeg = 3.0;
  std::printf(
      "[post-mag-cal, informative] mag systematic %.3f deg -> %.3f deg\n"
      "  coarse: median %.3f -> %.3f deg   3sigma-bound %.3f -> %.3f deg  (committed %.1f deg)\n"
      "  fine:   median %.3f -> %.3f deg   3sigma-bound %.3f -> %.3f deg  (committed %.1f deg)\n",
      kSigmaMagSys / kDeg, kSigmaMagSysPostCal / kDeg, coarse_now.median(), coarse.median(),
      coarse_now.max(), coarse.max(), kCommittedCoarseDeg, fine_now.median(), fine.median(),
      fine_now.max(), fine.max(), kCommittedFineDeg);

  // Two assertions, both about the projection being *real* rather than about
  // where it lands. First: removing a systematic term cannot make either chain
  // worse, and if it did the campaign would be wired wrong.
  EXPECT_LT(coarse.max(), coarse_now.max()) << "calibration did not improve the coarse bound";
  EXPECT_LT(fine.max(), fine_now.max()) << "calibration did not improve the fine bound";
  // Second: the remaining error is the **sun** budget, which the magnetometer
  // calibration does not touch. The 2.04° per-axis sun systematic alone has a
  // 3.44σ Rayleigh tail at ~7.0°, so a projected bound far below that would mean
  // the substitution had leaked into the sun terms.
  EXPECT_GT(coarse.max(), 4.0) << "projected bound is below the sun-only floor — did the "
                                  "substitution touch the sun budget?";
}

// ── Both-corrections projection — informative ───────────────────────────────
//
// **This verifies nothing either**, for the same reason: REQ-ADET-005/006 move
// when the flight chain runs both corrections end to end, not when the libraries
// exist. What it answers is the question §8.1 left open — *the committed 5°/3°
// tightening names items (1) and (2) together; do the two together actually
// reach it?* — and it is the evidence the threshold decision is taken on.
//
// The same 800 runs with **both** systematics replaced: the magnetic term by the
// post-calibration residual of `lib/gnc/mag_calibration`, and the sun term by
// the post-correction residual of `lib/gnc/albedo_correction`. Paired against
// the baseline draw for draw, as the single-item projection above is.

TEST(AttitudeAccuracyMonteCarlo, PostBothCorrectionsProjection) {
  const std::vector<PairedRun>& runs = postCorrectionsFallbackRuns();
  ASSERT_EQ(runs.size(), static_cast<std::size_t>(kRuns));
  for (std::size_t i = 0; i < runs.size(); ++i) {
    ASSERT_TRUE(runs[i].ok) << "run " << i << " left an estimator invalid";
  }

  const Campaign coarse = errorNorms(runs, false);
  const Campaign fine = errorNorms(runs, true);
  const Campaign coarse_now = errorNorms(campaignRuns(), false);
  const Campaign fine_now = errorNorms(campaignRuns(), true);

  constexpr double kCommittedCoarseDeg = 5.0;
  constexpr double kCommittedFineDeg = 3.0;
  std::printf(
      "[post-both-corrections, informative] sun systematic %.3f -> %.3f deg, "
      "mag %.3f -> %.3f deg\n"
      "  coarse: median %.3f -> %.3f deg   3sigma-bound %.3f -> %.3f deg  (committed %.1f deg)\n"
      "  fine:   median %.3f -> %.3f deg   3sigma-bound %.3f -> %.3f deg  (committed %.1f deg)\n",
      kSunSysUncal / kDeg, kSunSysPostAlbedo / kDeg, kSigmaMagSys / kDeg,
      kSigmaMagSysPostCal / kDeg, coarse_now.median(), coarse.median(), coarse_now.max(),
      coarse.max(), kCommittedCoarseDeg, fine_now.median(), fine.median(), fine_now.max(),
      fine.max(), kCommittedFineDeg);
  RecordProperty("coarse_bound_millideg", static_cast<int>(1000.0 * coarse.max()));
  RecordProperty("fine_bound_millideg", static_cast<int>(1000.0 * fine.max()));

  // As above, the assertions are about the projection being real rather than
  // about where it lands — the thresholds move by a design decision on this
  // evidence, not by a test asserting the answer it wants.
  EXPECT_LT(coarse.max(), coarse_now.max()) << "the corrections did not improve the coarse bound";
  EXPECT_LT(fine.max(), fine_now.max()) << "the corrections did not improve the fine bound";
  // With the magnetic term calibrated *and* the sun term corrected, the tail must
  // fall below the sun-only floor the single-item projection is pinned above —
  // that floor was the whole argument for sequencing this item, so a projection
  // that did not clear it would mean the sun substitution never took.
  EXPECT_LT(coarse.max(), 4.0) << "the sun systematic was not actually reduced";
  // Neither chain can do better than the white noise it is left with: a bound
  // near zero would mean the systematics were zeroed rather than reduced.
  EXPECT_GT(fine.max(), 0.2) << "bound implausibly small — is the noise wired in?";
}

// ── The analytic-ephemeris fallback: the degraded floor ─────────────────────
//
// REQ-ADET-006's 3° is conditioned on the DE440 tables being active, which is
// the configuration the vehicle nominally flies and the one the two requirement
// cases above verify. This case records what it falls back to when no upload is
// in place or the epoch runs past the uploaded span — the same budget with the
// analytic sun ephemeris, which is 7.0 mrad of the 12.6 mrad post-albedo sun
// systematic, over half of it and the largest single term the albedo correction
// leaves behind.
//
// **This is the degraded floor, not a requirement.** The fallback still clears
// both thresholds, and that is worth guarding — a regression that pushed it past
// 3° would change the conditioning story rather than merely cost margin. What it
// does *not* clear is REQ-ADET-006's 20% margin requirement (measured ~10%),
// which is precisely why the requirement is conditioned rather than stated flat.

TEST(AttitudeAccuracyMonteCarlo, PostBothCorrectionsFallbackIsTheDegradedFloor) {
  requireAllRunsValid(postCorrectionsFallbackRuns());

  const Campaign coarse_fallback = errorNorms(postCorrectionsFallbackRuns(), false);
  const Campaign fine_fallback = errorNorms(postCorrectionsFallbackRuns(), true);
  const Campaign coarse = errorNorms(requirementRuns(), false);
  const Campaign fine = errorNorms(requirementRuns(), true);

  std::printf(
      "[degraded floor: analytic ephemeris] sun systematic %.3f deg (tables) "
      "-> %.3f deg (analytic)\n"
      "  coarse: median %.3f -> %.3f deg   3sigma-bound %.3f -> %.3f deg  (limit %.1f deg)\n"
      "  fine:   median %.3f -> %.3f deg   3sigma-bound %.3f -> %.3f deg  (limit %.1f deg)\n",
      kSunSysPostAlbedoTables / kDeg, kSunSysPostAlbedo / kDeg, coarse.median(),
      coarse_fallback.median(), coarse.max(), coarse_fallback.max(), kCoarseLimitDeg, fine.median(),
      fine_fallback.median(), fine.max(), fine_fallback.max(), kFineLimitDeg);
  RecordProperty("coarse_fallback_bound_millideg",
                 static_cast<int>(1000.0 * coarse_fallback.max()));
  RecordProperty("fine_fallback_bound_millideg", static_cast<int>(1000.0 * fine_fallback.max()));

  // Losing the tables cannot make either chain better.
  EXPECT_GE(coarse_fallback.max(), coarse.max()) << "the fallback beat the DE440 tables";
  EXPECT_GE(fine_fallback.max(), fine.max()) << "the fallback beat the DE440 tables";
  // The degraded configuration still clears both thresholds — on margin alone,
  // not on the requirement's 20%.
  EXPECT_LE(coarse_fallback.max(), kCoarseLimitDeg) << "degraded floor no longer clears 5 deg";
  EXPECT_LE(fine_fallback.max(), kFineLimitDeg) << "degraded floor no longer clears 3 deg";
  // What is left is the albedo dispersion, which the ephemeris does not touch. A
  // bound far below that floor would mean the substitution had leaked into the
  // sensor terms: the 10.5 mrad per-axis residual alone has a 3.44 sigma Rayleigh
  // tail at ~2.1 deg.
  EXPECT_GT(fine_fallback.max(), 1.5)
      << "fallback bound is below the albedo-only floor — did the substitution touch the albedo "
         "budget?";
}

}  // namespace
