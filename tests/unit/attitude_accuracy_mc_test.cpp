/// @file Monte Carlo attitude-knowledge accuracy campaign
/// (REQ-ADET-005, REQ-ADET-006, REQ-PAY-001; design doc §8.1, §22.3).
///
/// The two **vehicle-level** requirements are stated on the **error norm** — the
/// total eigenaxis rotation angle between the estimated and the true attitude,
/// `θ_err = 2·acos(|q_err scalar|)` — at the 3σ (99.73rd-percentile) point, on
/// the reference vehicle's own sensor budget
/// (`config/spacecraft/leo_smallsat.yaml`, `flight.attitudeEstimator.*`). The
/// third is the **payload cross-boresight** error (REQ-PAY-001), measured on the
/// same fine-mode runs so the two metrics are directly comparable.
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
constexpr double kSigmaSunSys = 0.0356;      ///< [rad] albedo ⊕ analytic ephemeris
constexpr double kSigmaMagWhite = 0.0017;    ///< [rad] 0.05 µT rms on a 30 µT field
constexpr double kSigmaMagSys = 0.0337;      ///< [rad] hard/soft-iron residual
constexpr double kGyroArw = 4.363e-5;        ///< [rad·s^(-1/2)] STIM300 0.15°/√h
constexpr double kGyroRrw = 4.7e-7;          ///< [rad·s^(-3/2)] STIM300 bias instability
constexpr double kBiasSigmaInit = 4.848e-5;  ///< [rad/s] 10°/h turn-on repeatability
constexpr double kMinSinAngle = 0.17;        ///< sin(10°) TRIAD geometry gate
constexpr double kTriadGain = 0.3;
constexpr double kNisGate = 13.82;
constexpr double kSeedMinObservability = 0.0076;

/// Sun/field separations sampled: the well-conditioned band the requirements are
/// stated under, not the orthogonal best case (`adet-campaign-method`).
constexpr double kMinSeparationDeg = 45.0;
constexpr double kMaxSeparationDeg = 135.0;

/// Requirement thresholds on the error norm [deg], 3σ — REQ-ADET-005 and
/// REQ-ADET-006. Both were **set from this campaign**, not the other way round:
/// the measured bound is 8.9° coarse and 11.4° fine at the fixed seed, and
/// 8.1–9.7° / 8.2–11.4° across the four other master seeds tried during
/// development, so 15° leaves 41% and 24% margin and does not sit on a tail
/// that moves with the seed.
constexpr double kCoarseLimitDeg = 15.0;
constexpr double kFineLimitDeg = 15.0;

/// REQ-PAY-001, the payload cross-boresight threshold [deg], 3σ. Set the same
/// way: the measured bound is 10.6° at the fixed seed and 8.7–10.6° across the
/// three other master seeds tried, so 14° leaves 24.5% margin. Below the 15° of
/// the two norm requirements because the metric is smaller by construction —
/// the about-boresight component of the attitude error drops out (see the test
/// at the bottom of this file).
constexpr double kCrossBoresightLimitDeg = 14.0;

/// The 1σ handed to the MEKF and the Davenport seed per source: white ⊕
/// systematic, exactly as `AttitudeEstimator::refreshCoarseConfig` inflates it. The
/// filter has no way to model a systematic term, so the caller pays for it in
/// R (mekf.hpp, "Measurement noise is the caller's, and white").
const double kSigmaSunTotal = std::hypot(kSigmaSunWhite, kSigmaSunSys);
const double kSigmaMagTotal = std::hypot(kSigmaMagWhite, kSigmaMagSys);

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

  static Systematics draw(polaris::random::SplitMix64& rng) {
    return {kSigmaSunSys * rng.gaussian(), kSigmaSunSys * rng.gaussian(),
            kSigmaMagSys * rng.gaussian(), kSigmaMagSys * rng.gaussian()};
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

RunSetup drawRun(polaris::random::SplitMix64& rng) {
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
  s.sys = Systematics::draw(rng);
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

gnc::CoarseAttitudeConfig coarseConfig() {
  gnc::CoarseAttitudeConfig cfg{};
  cfg.sigma_sun_white_rad = kSigmaSunWhite;
  cfg.sigma_sun_sys_rad = kSigmaSunSys;
  cfg.sigma_mag_white_rad = kSigmaMagWhite;
  cfg.sigma_mag_sys_rad = kSigmaMagSys;
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

// ── REQ-ADET-005: coarse-mode knowledge accuracy ────────────────────────────

TEST(AttitudeAccuracyMonteCarlo, CoarseModeKnowledgeErrorNorm) {
  RecordProperty("verifies", "REQ-ADET-005");
  Campaign campaign;
  campaign.samples.reserve(kRuns);

  for (int run = 0; run < kRuns; ++run) {
    polaris::random::SplitMix64 rng(polaris::random::streamSeed(0xC0A125Eu, run));
    const RunSetup s = drawRun(rng);
    Eigen::Vector3d bias = s.bias;

    gnc::CoarseAttitudeEstimator estimator(coarseConfig());
    ASSERT_TRUE(estimator.isConfigured());

    gnc::CoarseAttitudeOutput out{};
    for (int step = 1; step <= kSteps; ++step) {
      const double t = step * kDt;
      const pm::Quaternion q_true = truthAttitude(s.rate, t, s.q0);

      gnc::CoarseAttitudeInput in{};
      in.epoch = epochAt(t);
      in.gyro = pm::Vec3<frames::Body>(measureGyro(s.rate, bias, rng));
      in.gyro_valid = true;
      in.sun_body = pm::Vec3<frames::Body>(measureSun(s, q_true, rng));
      in.sun_ref = pm::Vec3<frames::ECI>(s.sun_eci);
      in.sun_valid = true;
      in.mag_body = pm::Vec3<frames::Body>(measureMag(s, q_true, rng));
      in.mag_ref = pm::Vec3<frames::ECI>(s.mag_eci);
      in.mag_valid = true;
      estimator.update(in, out);
    }

    // Every run's geometry is inside the flight gate by construction, so every
    // run must end with a usable attitude — a run that quietly failed to
    // produce one would otherwise drop out of the statistics and flatter the
    // result.
    ASSERT_TRUE(out.attitude_valid) << "run " << run << " produced no valid attitude";
    campaign.samples.push_back(
        errorNormDeg(out.attitude.core(), truthAttitude(s.rate, kSteps * kDt, s.q0)));
  }

  ASSERT_EQ(campaign.samples.size(), static_cast<std::size_t>(kRuns));
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

// ── REQ-ADET-006: fine-mode (SS+MAG+IMU) knowledge accuracy ─────────────────

/// One fine-mode run's outcome: the filter's estimate and the truth it should
/// have matched. Factored out of the test body because two requirements are
/// measured on the *same* runs — the total error norm (REQ-ADET-006) and the
/// cross-boresight error of a mounted payload (REQ-PAY-001). Sharing the runs
/// rather than re-drawing them is what makes the two numbers directly
/// comparable: any difference between them is the metric, not the sample.
struct FineRun {
  pm::Quaternion estimate{};
  pm::Quaternion truth{};
  bool ok{false};  ///< false if the run failed to seed or left the filter invalid
};

FineRun fineRun(int run) {
  FineRun result{};
  polaris::random::SplitMix64 rng(polaris::random::streamSeed(0xF14E5Eu, run));
  const RunSetup s = drawRun(rng);
  Eigen::Vector3d bias = s.bias;

  gnc::Mekf filter(mekfConfig());
  if (!filter.isConfigured()) {
    return result;
  }

  // Cold start on the real path: a Davenport seed from the first noisy
  // measurement pair, with the turn-on bias uncertainty and no bias estimate.
  gnc::DavenportInput seed_in{};
  seed_in.count = 2;
  seed_in.min_observability = kSeedMinObservability;
  seed_in.observations[0].body = pm::Vec3<frames::Body>(measureSun(s, s.q0, rng));
  seed_in.observations[0].reference = pm::Vec3<frames::ECI>(s.sun_eci);
  seed_in.observations[0].sigma_rad = kSigmaSunTotal;
  seed_in.observations[1].body = pm::Vec3<frames::Body>(measureMag(s, s.q0, rng));
  seed_in.observations[1].reference = pm::Vec3<frames::ECI>(s.mag_eci);
  seed_in.observations[1].sigma_rad = kSigmaMagTotal;

  gnc::DavenportSolution seed{};
  if (!gnc::davenport(seed_in, seed) ||
      !filter.initialize(epochAt(0.0), seed.attitude, seed.covariance,
                         pm::Vec3<frames::Body>(Eigen::Vector3d::Zero()),
                         (kBiasSigmaInit * kBiasSigmaInit) * Eigen::Matrix3d::Identity())) {
    return result;
  }

  for (int step = 1; step <= kSteps; ++step) {
    const double t = step * kDt;
    const pm::Quaternion q_true = truthAttitude(s.rate, t, s.q0);
    if (!filter.propagate(epochAt(t), pm::Vec3<frames::Body>(measureGyro(s.rate, bias, rng)),
                          true)) {
      return result;
    }

    gnc::MekfUpdate up{};
    filter.update(pm::Vec3<frames::Body>(measureSun(s, q_true, rng)),
                  pm::Vec3<frames::ECI>(s.sun_eci), kSigmaSunTotal, up);
    filter.update(pm::Vec3<frames::Body>(measureMag(s, q_true, rng)),
                  pm::Vec3<frames::ECI>(s.mag_eci), kSigmaMagTotal, up);
  }

  result.estimate = filter.attitude().core();
  result.truth = truthAttitude(s.rate, kSteps * kDt, s.q0);
  result.ok = filter.attitudeValid();
  return result;
}

TEST(AttitudeAccuracyMonteCarlo, FineModeSunMagKnowledgeErrorNorm) {
  RecordProperty("verifies", "REQ-ADET-006");
  Campaign campaign;
  campaign.samples.reserve(kRuns);

  for (int run = 0; run < kRuns; ++run) {
    const FineRun r = fineRun(run);
    ASSERT_TRUE(r.ok) << "run " << run << " failed to seed or left the filter invalid";
    campaign.samples.push_back(errorNormDeg(r.estimate, r.truth));
  }

  ASSERT_EQ(campaign.samples.size(), static_cast<std::size_t>(kRuns));
  RecordProperty("margin_pct",
                 static_cast<int>(report(campaign, kFineLimitDeg, "REQ-ADET-006 fine SS+MAG")));

  EXPECT_LE(campaign.max(), kFineLimitDeg) << "3σ knowledge-error bound over " << kRuns << " runs";
  // Same sensitivity floor as the coarse campaign; measured median is 2.9°.
  EXPECT_GT(campaign.median(), 2.0) << "median error implausibly small — is the noise wired in?";
}

// ── REQ-PAY-001: payload cross-boresight knowledge accuracy ─────────────────
//
// The metric — the angle between where the payload's boresight actually points
// and where the solution says it points, equal to |θ| sin ψ for an attitude
// error θ at angle ψ to the boresight — is defined in
// docs/requirements/payload.rst (`pay-cross-boresight-metric`), along with why
// the about-boresight component drops out and why the answer is independent of
// where the payload is mounted. The test measures the last of those rather than
// assuming it: a mounting-dependent answer would mean the campaign had a
// preferred body axis, which would invalidate the vehicle-level numbers too.

/// The mounted boresight in body axes, taken from the payload model itself
/// rather than hand-written, so the requirement is measured on the same +Z
/// convention the sim flies (`sim/sensors/payload_sensor.hpp`).
Eigen::Vector3d payloadBoresightBody(const Eigen::Matrix3d& mounting_dcm) {
  polaris::sim::sensors::PayloadSensorSpec spec{};
  spec.half_fov_x_rad = 5.0 * kDeg;
  spec.half_fov_y_rad = 4.0 * kDeg;
  return polaris::sim::sensors::PayloadSensor(spec, mounting_dcm).boresightBody();
}

/// Angle [deg] between the boresight as the estimate places it and as the truth
/// places it, both in ECI. This is the metric REQ-PAY-001 is written on.
double boresightErrorDeg(const Eigen::Vector3d& boresight_body, const pm::Quaternion& est,
                         const pm::Quaternion& q_true) {
  const Eigen::Vector3d estimated = est.inverse().rotate(boresight_body).normalized();
  const Eigen::Vector3d actual = q_true.inverse().rotate(boresight_body).normalized();
  return std::atan2(estimated.cross(actual).norm(), estimated.dot(actual)) / kDeg;
}

TEST(AttitudeAccuracyMonteCarlo, PayloadCrossBoresightKnowledgeError) {
  RecordProperty("verifies", "REQ-PAY-001");

  // Two mountings: the reference vehicle's (identity — boresight along body +Z,
  // nadir in the nominal Earth-pointing attitude) and a deliberately canted one.
  // The requirement is judged on the first; the second is the mount-independence
  // check described above.
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

  for (int run = 0; run < kRuns; ++run) {
    const FineRun r = fineRun(run);
    ASSERT_TRUE(r.ok) << "run " << run << " failed to seed or left the filter invalid";
    campaign.samples.push_back(boresightErrorDeg(nadir_mount, r.estimate, r.truth));
    canted.samples.push_back(boresightErrorDeg(canted_mount, r.estimate, r.truth));
    norms.samples.push_back(errorNormDeg(r.estimate, r.truth));
  }

  ASSERT_EQ(campaign.samples.size(), static_cast<std::size_t>(kRuns));
  RecordProperty("margin_pct", static_cast<int>(report(campaign, kCrossBoresightLimitDeg,
                                                       "REQ-PAY-001 payload")));
  report(canted, kCrossBoresightLimitDeg, "REQ-PAY-001 payload (canted mount)");

  EXPECT_LE(campaign.max(), kCrossBoresightLimitDeg)
      << "3σ cross-boresight bound over " << kRuns << " runs";

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
  // point a payload to a fraction of a degree.
  EXPECT_GT(campaign.median(), 1.5) << "median error implausibly small — is the noise wired in?";
}

}  // namespace
