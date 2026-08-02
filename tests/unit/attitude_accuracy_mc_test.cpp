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
/// (`config/spacecraft/leo_smallsat.yaml`, `flight.attitudeEstimator.*`). The
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
/// not the sample — and the campaign runs once for all four tests below.
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
constexpr double kNisGate = 13.82;
constexpr double kSeedMinObservability = 0.0076;

/// The magnetic systematic **after** the on-orbit hard/soft-iron calibration of
/// §8.1 [rad], for the informative projection at the bottom of this file.
///
/// Measured by `mag_calibration_test.cpp`
/// (`PostCalibrationSystematicMeetsTheHalfDegreeTarget`) on the same budget the
/// constants above describe: 0.087° rms residual direction error in the worst of
/// 32 seeds. That figure is the *total* angle, so using it as the per-axis σ
/// here is conservative by √2 — the projection is meant to under-promise.
constexpr double kSigmaMagSysPostCal = 0.087 * kDeg;

/// The sun systematic **after** the onboard Earth-albedo correction of §8.1
/// [rad], for the informative projection at the bottom of this file.
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
/// This is the budget the vehicle actually flies with an ephemeris upload in
/// place, so it — not the analytic-fallback figure above — is what the eventual
/// REQ-ADET-006 tightening should be conditioned on. `kSunSysPostAlbedo` is
/// then the *degraded* floor: what the vehicle falls back to with no upload or
/// past the end of the uploaded span.
constexpr double kSunSysPostAlbedoTables = 0.0105;

/// The Davenport seed's observability gate for the post-calibration projection
/// [dimensionless].
///
/// It has to be re-derived rather than reused, and the reason is worth stating
/// because it is a **flight-parameter finding, not a test detail**. The gated
/// ratio `λ_min/λ_max` of `M = Σ σᵢ⁻²(I − b̂ᵢb̂ᵢᵀ)` is scale-free under a
/// *common* rescaling of the σ's, which is what makes it a geometry gate — but
/// it is not invariant to changing the σ's *relative* to each other. With the
/// uncalibrated budget the two sources are within 10% of each other in weight
/// and a 10° separation gives 0.0076, the shipped value. Calibrated, the
/// magnetic pair is ~16× tighter than the sun pair, so its weight is ~270×
/// larger, `λ_max ≈ w_mag` while `λ_min ≈ w_sun·sin²θ`, and the same 10°
/// geometry now reads ~1.1e-4. The shipped 0.0076 would refuse **every**
/// geometry in the band. So `fine.seedMinObservability` in
/// `config/spacecraft/leo_smallsat.yaml` must be re-derived when the
/// calibration is actually enabled on the vehicle; this constant is that
/// re-derivation for the projection.
constexpr double kSeedMinObservabilityPostCal = 1.0e-4;

/// Sun/field separations sampled: the well-conditioned band the requirements are
/// stated under, not the orthogonal best case (`adet-campaign-method`).
constexpr double kMinSeparationDeg = 45.0;
constexpr double kMaxSeparationDeg = 135.0;

/// Requirement thresholds on the error norm [deg], 3σ — REQ-ADET-005 and
/// REQ-ADET-006. Both were **set from this campaign**, not the other way round:
/// the measured bound is 8.5° coarse and 8.1° fine at the fixed seed, and
/// 8.5–11.6° / 8.1–11.7° across the three other master seeds tried during
/// development, so 15° leaves 43% and 46% margin at the shipped seed and never
/// less than 22% at any seed tried. The sample maximum is a tail statistic and
/// moves with the seed by design; the thresholds are set to clear the worst of
/// them, not the prettiest.
constexpr double kCoarseLimitDeg = 15.0;
constexpr double kFineLimitDeg = 15.0;

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
  ///        `kSunSysUncal` uncalibrated, the post-albedo-correction residual for
  ///        the informative projection.
  /// @param mag_sys the magnetic systematic 1σ [rad], likewise —
  ///        `kSigmaMagSys` for the requirement campaigns, the post-calibration
  ///        residual for the projections.
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

/// The measured sun direction in body: truth, tilted by the run's fixed
/// systematic offset, then by this cycle's white draw.
Eigen::Vector3d measureSun(const RunSetup& s, const pm::Quaternion& q_true,
                           polaris::random::SplitMix64& rng) {
  const Eigen::Vector3d truth = q_true.rotate(s.sun_eci);
  const Eigen::Vector3d biased = tilt(truth, s.sys.sun1, s.sys.sun2);
  return tilt(biased, kSigmaSunWhite * rng.gaussian(), kSigmaSunWhite * rng.gaussian());
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
};

/// Run the campaign at a given magnetic-systematic level [rad]. The seeds do not
/// depend on it, so two levels give the same geometry and the same noise
/// sequences with only the magnetic bias rescaled — the projection below is
/// therefore paired against the baseline as well.
std::vector<PairedRun> runCampaign(double sun_sys, double mag_sys, double seed_min_observability) {
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
      const RunSetup s = drawRun(rng, sun_sys, mag_sys);
      Eigen::Vector3d bias = s.bias;

      PairedRun paired{};
      paired.truth = truthAttitude(s.rate, kSteps * kDt, s.q0);

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
      for (int step = 1; step <= kSteps; ++step) {
        const double t = step * kDt;
        const pm::Quaternion q_true = truthAttitude(s.rate, t, s.q0);

        // Drawn once, handed to both chains.
        const Eigen::Vector3d gyro = measureGyro(s.rate, bias, rng);
        const Eigen::Vector3d sun_body = measureSun(s, q_true, rng);
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

/// The requirement campaign — the reference vehicle's **uncalibrated** budget —
/// run once on first use and shared by every requirement test in this file.
const std::vector<PairedRun>& campaignRuns() {
  static const std::vector<PairedRun> runs =
      runCampaign(kSunSysUncal, kSigmaMagSys, kSeedMinObservability);
  return runs;
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
void requireAllRunsValid() {
  ASSERT_EQ(campaignRuns().size(), static_cast<std::size_t>(kRuns));
  for (std::size_t i = 0; i < campaignRuns().size(); ++i) {
    ASSERT_TRUE(campaignRuns()[i].ok) << "run " << i << " left an estimator invalid";
  }
}

// ── REQ-ADET-005: coarse-mode knowledge accuracy ────────────────────────────

TEST(AttitudeAccuracyMonteCarlo, CoarseModeKnowledgeErrorNorm) {
  RecordProperty("verifies", "REQ-ADET-005");
  requireAllRunsValid();
  const Campaign campaign = errorNorms(campaignRuns(), false);

  RecordProperty("margin_pct",
                 static_cast<int>(report(campaign, kCoarseLimitDeg, "REQ-ADET-005 coarse")));

  EXPECT_LE(campaign.max(), kCoarseLimitDeg)
      << "3σ knowledge-error bound over " << kRuns << " runs";
  // Sensitivity floor: this budget cannot produce a sub-degree coarse solution.
  // The measured median is 3.3° and moves by under 0.1° across seeds, so a test
  // bug that zeroed the systematic draws — which would sail through the bound
  // above — is caught here instead.
  EXPECT_GT(campaign.median(), 2.0) << "median error implausibly small — is the noise wired in?";
}

// ── REQ-ADET-006: fine-mode knowledge accuracy, same SS+MAG+IMU suite ───────

TEST(AttitudeAccuracyMonteCarlo, FineModeKnowledgeErrorNorm) {
  RecordProperty("verifies", "REQ-ADET-006");
  requireAllRunsValid();
  const Campaign campaign = errorNorms(campaignRuns(), true);

  RecordProperty("margin_pct",
                 static_cast<int>(report(campaign, kFineLimitDeg, "REQ-ADET-006 fine MEKF")));

  EXPECT_LE(campaign.max(), kFineLimitDeg) << "3σ knowledge-error bound over " << kRuns << " runs";
  // Same sensitivity floor as the coarse campaign; measured median is 2.9°.
  EXPECT_GT(campaign.median(), 2.0) << "median error implausibly small — is the noise wired in?";
}

// ── Head to head: does the filter actually earn its keep? ───────────────────

TEST(AttitudeAccuracyMonteCarlo, FineModeBeatsCoarseRunForRun) {
  RecordProperty("verifies", "REQ-ADET-006");
  requireAllRunsValid();

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

  requireAllRunsValid();
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

// ── Post-magnetometer-calibration projection — informative ──────────────────
//
// **This verifies nothing.** REQ-ADET-005 and REQ-ADET-006 stand at 15° on the
// uncalibrated budget above, and they move only when the flight chain actually
// runs the calibration end to end (§8.1, "Calibration is commanded, executed and
// assessed on orbit") — not when a library can fit an ellipsoid. What this case
// answers is the question that decides whether the rest of that work is worth
// doing: *how far toward the committed 5°/3° does the magnetometer calibration
// alone get us?*
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

// ── Both corrections **and** the onboard ephemeris tables — informative ──────
//
// **Verifies nothing**, like the two above. What it answers is which budget the
// eventual REQ-ADET-006 tightening should be *conditioned* on. The analytic
// ephemeris fallback is 7.0 mrad of the 12.6 mrad post-albedo sun systematic —
// over half of it, and the largest single term left once the albedo correction
// has run. But the vehicle does not have to fly on the fallback: the Push 37
// onboard DE440 Chebyshev tables give an arcsecond-class sun direction from time
// alone, and the estimator now follows the served grade per cycle.
//
// So there are two honest numbers, not one: what the vehicle achieves with an
// ephemeris upload in place, and what it falls back to without one.

TEST(AttitudeAccuracyMonteCarlo, PostBothCorrectionsWithEphemerisTablesProjection) {
  const std::vector<PairedRun> runs =
      runCampaign(kSunSysPostAlbedoTables, kSigmaMagSysPostCal, kSeedMinObservabilityPostCal);
  ASSERT_EQ(runs.size(), static_cast<std::size_t>(kRuns));
  for (std::size_t i = 0; i < runs.size(); ++i) {
    ASSERT_TRUE(runs[i].ok) << "run " << i << " left an estimator invalid";
  }

  const Campaign coarse = errorNorms(runs, false);
  const Campaign fine = errorNorms(runs, true);
  const Campaign coarse_fallback = errorNorms(postCorrectionsFallbackRuns(), false);
  const Campaign fine_fallback = errorNorms(postCorrectionsFallbackRuns(), true);

  constexpr double kCommittedCoarseDeg = 5.0;
  constexpr double kCommittedFineDeg = 3.0;
  std::printf(
      "[post-both + DE440 tables, informative] sun systematic %.3f deg (analytic) "
      "-> %.3f deg (tables)\n"
      "  coarse: median %.3f -> %.3f deg   3sigma-bound %.3f -> %.3f deg  (committed %.1f deg)\n"
      "  fine:   median %.3f -> %.3f deg   3sigma-bound %.3f -> %.3f deg  (committed %.1f deg)\n",
      kSunSysPostAlbedo / kDeg, kSunSysPostAlbedoTables / kDeg, coarse_fallback.median(),
      coarse.median(), coarse_fallback.max(), coarse.max(), kCommittedCoarseDeg,
      fine_fallback.median(), fine.median(), fine_fallback.max(), fine.max(), kCommittedFineDeg);
  RecordProperty("coarse_bound_millideg", static_cast<int>(1000.0 * coarse.max()));
  RecordProperty("fine_bound_millideg", static_cast<int>(1000.0 * fine.max()));

  // Removing a term cannot make either chain worse.
  EXPECT_LE(coarse.max(), coarse_fallback.max()) << "the tables did not improve the coarse bound";
  EXPECT_LE(fine.max(), fine_fallback.max()) << "the tables did not improve the fine bound";
  // What is left is the albedo dispersion, which the ephemeris does not touch. A
  // bound far below that floor would mean the substitution had leaked into the
  // sensor terms: the 10.5 mrad per-axis residual alone has a 3.44 sigma Rayleigh
  // tail at ~2.1 deg.
  EXPECT_GT(fine.max(), 1.5) << "projected bound is below the albedo-only floor — did the "
                                "substitution touch the albedo budget?";
}

}  // namespace
