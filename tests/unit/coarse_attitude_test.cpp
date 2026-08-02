/// @file Unit tests for the coarse SS+MAG+IMU attitude estimator
/// (REQ-ADET-002, REQ-ADET-003; design doc §8.1, §10).
///
/// The estimator under the Safe-mode floor, so the tests are written around the
/// conditions that floor exists for rather than around the happy path:
///
///  - **Acquisition.** A cold estimator with one good vector pair must land on
///    the truth attitude immediately — no filter warm-up, no assumed prior.
///  - **Coasting.** Through eclipse there is no sun vector and the solution is
///    whatever the gyro says. Drift must stay bounded by the gyro error over
///    the gap, the covariance must grow, and past the configured coast horizon
///    the attitude must be declared *invalid* rather than quietly wrong.
///  - **Recovery.** Sun-return inside the horizon blends; sun-return after the
///    horizon re-acquires whole. Both must converge on truth.
///  - **Degeneracy and bad data.** Sun-parallel-to-field, a dead gyro, a
///    backwards clock, a garbage measurement — none may produce a confident
///    answer, and none may assert.
///
/// The truth attitude is a constant-rate spin propagated in closed form, which
/// is the same kinematics the estimator integrates; the tests therefore compare
/// against an independent evaluation of it, not against the estimator's own
/// integration of it.

#include "gnc/coarse_attitude.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <Eigen/Core>
#include <Eigen/Geometry>

#include "gnc/triad.hpp"
#include "math/frames.hpp"
#include "math/quaternion.hpp"
#include "math/typed_vector.hpp"
#include "state/estimated_state.hpp"
#include "time/timescales.hpp"

namespace {

namespace gnc = polaris::gnc;
namespace pm = polaris::math;
namespace frames = polaris::math::frames;
namespace ptime = polaris::time;

constexpr double kDeg = M_PI / 180.0;
constexpr double kDt = 0.1;  ///< estimation cycle [s] (10 Hz, §8.1 rate group)

/// Inertial references: sun along ECI +x, field ~55 deg away — a
/// well-conditioned pair, unlike the near-parallel case exercised below.
const Eigen::Vector3d kSunEci(1.0, 0.0, 0.0);
const Eigen::Vector3d kMagEci =
    Eigen::Vector3d(std::cos(55.0 * kDeg), std::sin(55.0 * kDeg), 0.2).normalized();

/// Constant-rate truth spin about @p axis, evaluated independently of the
/// estimator's propagation.
pm::Quaternion truthAttitude(const Eigen::Vector3d& rate, double t_s,
                             const pm::Quaternion& initial) {
  const double angle = rate.norm() * t_s;
  if (angle <= 0.0) {
    return initial;
  }
  return (pm::Quaternion::FromAxisAngle(rate.normalized(), angle) * initial).canonical();
}

ptime::Tai epochAt(double t_s) {
  return ptime::Tai::fromNanosecondsSinceEpoch(static_cast<std::int64_t>(t_s * 1.0e9));
}

gnc::CoarseAttitudeConfig defaultConfig() {
  gnc::CoarseAttitudeConfig cfg{};
  cfg.sigma_sun_white_rad = 1.0 * kDeg;  // coarse sun-sensor noise
  cfg.sigma_sun_sys_rad = 0.5 * kDeg;    // analytic ephemeris + alignment
  cfg.sigma_mag_white_rad = 3.0 * kDeg;  // magnetometer noise
  cfg.sigma_mag_sys_rad = 2.0 * kDeg;    // IGRF model + hard/soft iron residual
  cfg.gyro_arw = 1.0e-4;                 // rad/s^(1/2)
  cfg.min_sin_angle = std::sin(10.0 * kDeg);
  cfg.triad_gain = 0.3;
  cfg.max_coast_s = 60.0;
  cfg.max_dt_s = 1.0;
  return cfg;
}

/// The systematic covariance floor for the geometry seen at attitude @p q, i.e.
/// what the published covariance may never fall below.
Eigen::Matrix3d systematicFloor(const pm::Quaternion& q) {
  const gnc::CoarseAttitudeConfig cfg = defaultConfig();
  Eigen::Matrix3d floor_cov;
  EXPECT_TRUE(gnc::triadCovariance(pm::Vec3<frames::Body>(q.rotate(kSunEci)),
                                   pm::Vec3<frames::Body>(q.rotate(kMagEci)), cfg.sigma_sun_sys_rad,
                                   cfg.sigma_mag_sys_rad, cfg.min_sin_angle, floor_cov));
  return floor_cov;
}

/// Covariance of a single TRIAD fix at attitude @p q on the full error budget
/// (white ⊕ systematic) — the uncertainty of one unfiltered fix.
Eigen::Matrix3d singleFixCovariance(const pm::Quaternion& q) {
  const gnc::CoarseAttitudeConfig cfg = defaultConfig();
  Eigen::Matrix3d cov;
  EXPECT_TRUE(gnc::triadCovariance(
      pm::Vec3<frames::Body>(q.rotate(kSunEci)), pm::Vec3<frames::Body>(q.rotate(kMagEci)),
      std::hypot(cfg.sigma_sun_white_rad, cfg.sigma_sun_sys_rad),
      std::hypot(cfg.sigma_mag_white_rad, cfg.sigma_mag_sys_rad), cfg.min_sin_angle, cov));
  return cov;
}

/// True if @p p is positive definite — the property every downstream Cholesky
/// depends on, and the one an asymmetric or over-shrunk covariance loses first.
bool isPositiveDefinite(const Eigen::Matrix3d& p) {
  return p.llt().info() == Eigen::Success;
}

/// A measurement cycle generated from the truth attitude (noise-free unless the
/// caller perturbs it afterwards).
gnc::CoarseAttitudeInput makeInput(double t_s, const pm::Quaternion& q_true,
                                   const Eigen::Vector3d& rate, bool sun_valid = true,
                                   bool mag_valid = true) {
  gnc::CoarseAttitudeInput in{};
  in.epoch = epochAt(t_s);
  in.gyro = pm::Vec3<frames::Body>(rate);
  in.gyro_valid = true;
  in.sun_body = pm::Vec3<frames::Body>(q_true.rotate(kSunEci));
  in.sun_ref = pm::Vec3<frames::ECI>(kSunEci);
  in.sun_valid = sun_valid;
  in.mag_body = pm::Vec3<frames::Body>(q_true.rotate(kMagEci));
  in.mag_ref = pm::Vec3<frames::ECI>(kMagEci);
  in.mag_valid = mag_valid;
  return in;
}

/// Rotation angle between two attitudes [deg], formed here rather than via
/// `angularDistance` so the propagation checks below measure the estimator
/// against an independently written formula and not against the quaternion
/// library's own method. Same atan2 convention (lib/README.md), so the two
/// agree to round-off.
double errorDeg(const pm::Quaternion& est, const pm::Quaternion& q_true) {
  const pm::Quaternion dq = (est * q_true.inverse()).canonical();
  return 2.0 * std::atan2(dq.vec().norm(), dq.scalar()) / kDeg;
}

double errorDeg(const gnc::CoarseAttitudeOutput& out, const pm::Quaternion& q_true) {
  return errorDeg(out.attitude.core(), q_true);
}

TEST(CoarseAttitude, RejectsInvalidConfiguration) {
  RecordProperty("verifies", "REQ-ADET-002");
  EXPECT_TRUE(defaultConfig().isValid());

  // A default-constructed config is deliberately *not* flyable: the tuning
  // values are mission configuration (§19.3), so there are no in-code defaults
  // to accidentally fly on.
  EXPECT_FALSE(gnc::CoarseAttitudeConfig{}.isValid());

  gnc::CoarseAttitudeConfig bad = defaultConfig();
  bad.sigma_sun_white_rad = 0.0;
  EXPECT_FALSE(bad.isValid()) << "the white part must be positive";

  bad = defaultConfig();
  bad.sigma_mag_sys_rad = -1.0;
  EXPECT_FALSE(bad.isValid()) << "a negative systematic budget is not a budget";

  bad = defaultConfig();
  bad.triad_gain = 1.5;
  EXPECT_FALSE(bad.isValid());

  bad = defaultConfig();
  bad.min_sin_angle = 0.0;
  EXPECT_FALSE(bad.isValid());

  bad = defaultConfig();
  bad.max_coast_s = -1.0;
  EXPECT_FALSE(bad.isValid());

  // A zero systematic budget is legal — a mission may claim its references are
  // perfect, it just does not get a floor.
  gnc::CoarseAttitudeConfig no_floor = defaultConfig();
  no_floor.sigma_sun_sys_rad = 0.0;
  no_floor.sigma_mag_sys_rad = 0.0;
  EXPECT_TRUE(no_floor.isValid());

  gnc::CoarseAttitudeEstimator est(bad);
  EXPECT_FALSE(est.isConfigured());
  gnc::CoarseAttitudeOutput out{};
  EXPECT_FALSE(
      est.update(makeInput(0.0, pm::Quaternion::Identity(), Eigen::Vector3d::Zero()), out));
  EXPECT_FALSE(out.attitude_valid);
}

TEST(CoarseAttitude, AcquiresFromFirstTriad) {
  RecordProperty("verifies", "REQ-ADET-002;REQ-ADET-003");
  gnc::CoarseAttitudeEstimator est(defaultConfig());
  ASSERT_TRUE(est.isConfigured());
  EXPECT_FALSE(est.isInitialised());

  const Eigen::Vector3d rate(0.001, -0.002, 0.0005);
  const pm::Quaternion q0 =
      pm::Quaternion::FromAxisAngle(Eigen::Vector3d(0.2, 0.9, -0.3).normalized(), 100.0 * kDeg);

  gnc::CoarseAttitudeOutput out{};
  ASSERT_TRUE(est.update(makeInput(0.0, q0, rate), out));
  EXPECT_TRUE(est.isInitialised());
  EXPECT_TRUE(out.attitude_valid);
  EXPECT_TRUE(out.triad_applied);
  EXPECT_LT(errorDeg(out, q0), 1e-9) << "cold start must land on truth, not converge to it";
  EXPECT_GE(out.attitude.core().w(), 0.0);
  EXPECT_EQ(out.age_s, 0.0);

  // Rate is the bias-corrected gyro.
  EXPECT_TRUE(out.rate_valid);
  EXPECT_LT((out.body_rate.eigen() - rate).norm(), 1e-15);

  // The seeded covariance is one TRIAD fix on the *full* budget: the white part
  // TRIAD is solved with, plus the systematic floor evaluated on the same
  // geometry. The two are computed independently here.
  // Not bit-exact: combining the budget as hypot(σ_white, σ_sys) and as a sum of
  // two covariances is the same algebra evaluated in a different order.
  EXPECT_TRUE(out.covariance.isApprox(singleFixCovariance(q0), 1e-12));
  EXPECT_TRUE(isPositiveDefinite(out.covariance));
}

TEST(CoarseAttitude, SubtractsGyroBias) {
  RecordProperty("verifies", "REQ-ADET-002");
  gnc::CoarseAttitudeEstimator est(defaultConfig());
  const Eigen::Vector3d rate(0.01, 0.0, 0.0);
  const Eigen::Vector3d bias(0.002, -0.001, 0.0005);
  const pm::Quaternion q0 = pm::Quaternion::Identity();

  gnc::CoarseAttitudeOutput out{};
  gnc::CoarseAttitudeInput in = makeInput(0.0, q0, rate + bias);
  in.gyro_bias = pm::Vec3<frames::Body>(bias);
  ASSERT_TRUE(est.update(in, out));
  EXPECT_LT((out.body_rate.eigen() - rate).norm(), 1e-15);

  // Propagate one cycle on the biased gyro: the bias-corrected rate is the one
  // that must track truth.
  in = makeInput(kDt, truthAttitude(rate, kDt, q0), rate + bias, false, false);
  in.gyro_bias = pm::Vec3<frames::Body>(bias);
  ASSERT_TRUE(est.update(in, out));
  EXPECT_LT(errorDeg(out, truthAttitude(rate, kDt, q0)), 1e-9);
}

TEST(CoarseAttitude, GyroPropagationTracksTruthThroughEclipse) {
  RecordProperty("verifies", "REQ-ADET-002");
  gnc::CoarseAttitudeEstimator est(defaultConfig());
  const Eigen::Vector3d rate(0.02, -0.01, 0.005);  // ~1.3 deg/s
  const pm::Quaternion q0 = pm::Quaternion::Identity();

  gnc::CoarseAttitudeOutput out{};
  ASSERT_TRUE(est.update(makeInput(0.0, q0, rate), out));
  const double initial_trace = out.covariance.trace();

  // 30 s of eclipse: sun invalid, gyro only. A perfect gyro means the only
  // error is the closed-form propagation itself, which is exact at constant
  // rate — so this pins the kinematics, not the noise model.
  for (int step = 1; step <= 300; ++step) {
    const double t = step * kDt;
    ASSERT_TRUE(est.update(makeInput(t, truthAttitude(rate, t, q0), rate, false, true), out));
    EXPECT_FALSE(out.triad_applied);
    EXPECT_TRUE(out.attitude_valid) << "30 s < 60 s coast horizon";
    EXPECT_LT(errorDeg(out, truthAttitude(rate, t, q0)), 1e-9);
    EXPECT_GE(out.attitude.core().w(), 0.0) << "q0 >= 0 maintained under propagation";
  }
  EXPECT_NEAR(out.age_s, 30.0, 1e-9);

  // Uncertainty must grow over the coast by the gyro random walk, stay
  // symmetric under 300 similarity transforms, and stay factorisable — a
  // covariance that has drifted out of symmetry breaks every consumer that
  // Choleskys it.
  const double arw = defaultConfig().gyro_arw;
  EXPECT_NEAR(out.covariance.trace(), initial_trace + 3.0 * arw * arw * 30.0, 1e-12);
  EXPECT_TRUE(out.covariance.isApprox(out.covariance.transpose(), 1e-15)) << "symmetry";
  EXPECT_TRUE(isPositiveDefinite(out.covariance));
  // The floor is expressed in body axes, so it rotates with the body over the
  // coast — evaluated at the *current* attitude, it must still be dominated.
  EXPECT_TRUE(isPositiveDefinite(out.covariance - systematicFloor(truthAttitude(rate, 30.0, q0))))
      << "the systematic floor rides along under propagation, it is not lost";
}

TEST(CoarseAttitude, InvalidatesPastCoastHorizonAndReacquiresWhole) {
  RecordProperty("verifies", "REQ-ADET-002");
  gnc::CoarseAttitudeConfig cfg = defaultConfig();
  cfg.max_coast_s = 5.0;
  gnc::CoarseAttitudeEstimator est(cfg);
  const Eigen::Vector3d rate(0.01, 0.0, 0.0);
  const pm::Quaternion q0 = pm::Quaternion::Identity();

  gnc::CoarseAttitudeOutput out{};
  ASSERT_TRUE(est.update(makeInput(0.0, q0, rate), out));

  // Drift the estimator off truth by feeding a gyro that is wrong during the
  // gap; the estimate then genuinely diverges and must be declared invalid.
  const Eigen::Vector3d wrong_rate = rate + Eigen::Vector3d(0.0, 0.02, 0.0);
  bool went_invalid = false;
  for (int step = 1; step <= 100; ++step) {
    const double t = step * kDt;
    gnc::CoarseAttitudeInput in = makeInput(t, truthAttitude(rate, t, q0), wrong_rate, false, true);
    const bool ok = est.update(in, out);
    if (t > cfg.max_coast_s) {
      EXPECT_FALSE(ok) << "t = " << t;
      EXPECT_FALSE(out.attitude_valid);
      went_invalid = true;
    }
  }
  ASSERT_TRUE(went_invalid);
  EXPECT_FALSE(est.isInitialised()) << "coast timeout drops the solution";
  EXPECT_GT(out.covariance.trace(), 0.0) << "covariance keeps reporting the drift";

  // Sun return: with the solution dropped, TRIAD is taken whole — one cycle
  // back on truth, not a slow blend from a meaningless prior.
  const double t = 10.1;
  ASSERT_TRUE(est.update(makeInput(t, truthAttitude(rate, t, q0), rate), out));
  EXPECT_TRUE(out.attitude_valid);
  EXPECT_TRUE(out.triad_applied);
  EXPECT_LT(errorDeg(out, truthAttitude(rate, t, q0)), 1e-9);
  EXPECT_EQ(out.age_s, 0.0);
}

TEST(CoarseAttitude, BlendPullsADriftedSolutionBackToTruth) {
  RecordProperty("verifies", "REQ-ADET-002");
  gnc::CoarseAttitudeConfig cfg = defaultConfig();
  cfg.triad_gain = 0.25;
  gnc::CoarseAttitudeEstimator est(cfg);
  const Eigen::Vector3d rate = Eigen::Vector3d::Zero();
  const pm::Quaternion q0 = pm::Quaternion::Identity();

  gnc::CoarseAttitudeOutput out{};
  ASSERT_TRUE(est.update(makeInput(0.0, q0, rate), out));
  const double triad_only_trace = out.covariance.trace();  // one raw TRIAD fix

  // Eclipse with a gyro that reads zero while the vehicle actually rotates:
  // the estimate is left behind by a known 6 deg.
  const Eigen::Vector3d truth_rate(0.0, 0.0, 6.0 * kDeg);  // 6 deg/s about +z
  for (int step = 1; step <= 10; ++step) {
    const double t = step * kDt;
    ASSERT_TRUE(est.update(makeInput(t, truthAttitude(truth_rate, t, q0), rate, false, true), out));
  }
  const pm::Quaternion q_drifted = truthAttitude(truth_rate, 1.0, q0);
  const double drift_deg = errorDeg(out, q_drifted);
  ASSERT_NEAR(drift_deg, 6.0, 0.1);

  // Sun returns and the truth attitude holds still. Each blended update must
  // shrink the error by roughly the gain, monotonically, without overshoot.
  double previous = drift_deg;
  for (int step = 11; step <= 60; ++step) {
    const double t = step * kDt;
    ASSERT_TRUE(est.update(makeInput(t, q_drifted, Eigen::Vector3d::Zero(), true, true), out));
    EXPECT_TRUE(out.triad_applied);
    const double err = errorDeg(out, q_drifted);
    EXPECT_LT(err, previous) << "step " << step;
    previous = err;
  }
  EXPECT_LT(previous, 0.01) << "blend converges on the TRIAD solution";

  // Repeated blending settles between two bounds, and both matter. It must beat
  // one raw fix — otherwise filtering bought nothing — but it must NOT fall
  // below the systematic floor, because ephemeris error, IGRF error and sensor
  // alignment are the same offset every cycle and averaging cannot remove them.
  // Without the floor this converges to k/(2−k) ≈ 0.14× the single-fix
  // covariance, which would be a confident lie handed to the MEKF.
  const Eigen::Matrix3d floor_cov = systematicFloor(q_drifted);
  EXPECT_LT(out.covariance.trace(), triad_only_trace);
  EXPECT_GT(out.covariance.trace(), floor_cov.trace());
  EXPECT_TRUE(isPositiveDefinite(out.covariance - floor_cov))
      << "steady-state covariance must dominate the systematic floor";
  EXPECT_TRUE(isPositiveDefinite(out.covariance));
  EXPECT_TRUE(out.covariance.isApprox(out.covariance.transpose(), 1e-15)) << "symmetry";

  // And the floor is the limit, not a formality: many more fixes do not sink
  // the covariance below it.
  for (int step = 61; step <= 400; ++step) {
    ASSERT_TRUE(est.update(makeInput(step * kDt, q_drifted, Eigen::Vector3d::Zero()), out));
  }
  EXPECT_TRUE(isPositiveDefinite(out.covariance - floor_cov));
  EXPECT_LT(out.covariance.trace(), 0.5 * triad_only_trace) << "but it does still filter";
}

TEST(CoarseAttitude, SkipsUpdateOnDegenerateSunFieldGeometry) {
  RecordProperty("verifies", "REQ-ADET-002;REQ-ADET-003");
  gnc::CoarseAttitudeEstimator est(defaultConfig());
  const Eigen::Vector3d rate = Eigen::Vector3d::Zero();
  const pm::Quaternion q0 = pm::Quaternion::Identity();

  gnc::CoarseAttitudeOutput out{};
  ASSERT_TRUE(est.update(makeInput(0.0, q0, rate), out));
  ASSERT_TRUE(out.triad_applied);

  // Field swings to within 2 deg of the sun line: the roll about the sun is
  // unobservable, so no update — but the estimator keeps coasting, it does not
  // fail or reset.
  const Eigen::Vector3d near_sun =
      Eigen::Vector3d(std::cos(2.0 * kDeg), std::sin(2.0 * kDeg), 0.0).normalized();
  for (int step = 1; step <= 5; ++step) {
    const double t = step * kDt;
    gnc::CoarseAttitudeInput in = makeInput(t, q0, rate);
    in.mag_body = pm::Vec3<frames::Body>(q0.rotate(near_sun));
    in.mag_ref = pm::Vec3<frames::ECI>(near_sun);
    ASSERT_TRUE(est.update(in, out));
    EXPECT_FALSE(out.triad_applied) << "step " << step;
    EXPECT_TRUE(out.attitude_valid);
  }
  EXPECT_NEAR(out.age_s, 0.5, 1e-9) << "age grows: no fix was taken";
}

TEST(CoarseAttitude, HoldsAttitudeWhenGyroIsInvalid) {
  RecordProperty("verifies", "REQ-ADET-002");
  gnc::CoarseAttitudeEstimator est(defaultConfig());
  const Eigen::Vector3d rate(0.01, 0.0, 0.0);
  const pm::Quaternion q0 = pm::Quaternion::Identity();

  gnc::CoarseAttitudeOutput out{};
  ASSERT_TRUE(est.update(makeInput(0.0, q0, rate), out));

  gnc::CoarseAttitudeInput in = makeInput(kDt, q0, rate, false, true);
  in.gyro_valid = false;
  ASSERT_TRUE(est.update(in, out));
  EXPECT_FALSE(out.rate_valid);
  EXPECT_LT(errorDeg(out, q0), 1e-12) << "no gyro, no propagation — the attitude is held";
  EXPECT_GT(out.covariance.trace(), 0.0);
}

TEST(CoarseAttitude, RejectsNonMonotonicEpoch) {
  RecordProperty("verifies", "REQ-ADET-002");
  gnc::CoarseAttitudeEstimator est(defaultConfig());
  const Eigen::Vector3d rate = Eigen::Vector3d::Zero();
  const pm::Quaternion q0 = pm::Quaternion::Identity();

  gnc::CoarseAttitudeOutput out{};
  ASSERT_TRUE(est.update(makeInput(1.0, q0, rate), out));
  EXPECT_FALSE(est.update(makeInput(0.5, q0, rate), out));
  EXPECT_FALSE(out.attitude_valid);
  EXPECT_TRUE(est.isInitialised()) << "a backwards clock must not destroy the solution";

  // Forward again from the original epoch: the estimator carries on.
  ASSERT_TRUE(est.update(makeInput(1.1, q0, rate), out));
  EXPECT_LT(errorDeg(out, q0), 1e-9);
}

TEST(CoarseAttitude, RejectsAStuckClock) {
  RecordProperty("verifies", "REQ-ADET-002");
  gnc::CoarseAttitudeEstimator est(defaultConfig());
  const Eigen::Vector3d rate = Eigen::Vector3d::Zero();
  const pm::Quaternion q0 = pm::Quaternion::Identity();

  gnc::CoarseAttitudeOutput out{};
  ASSERT_TRUE(est.update(makeInput(1.0, q0, rate), out));
  const Eigen::Matrix3d after_first_fix = out.covariance;

  // Re-running at the same epoch would fold the same measurement in again and
  // shrink the covariance on information already used — a stuck clock would
  // manufacture confidence out of nothing. Ten repeats must change nothing.
  for (int i = 0; i < 10; ++i) {
    EXPECT_FALSE(est.update(makeInput(1.0, q0, rate), out));
    EXPECT_FALSE(out.attitude_valid);
  }
  EXPECT_TRUE(est.isInitialised()) << "a stuck clock must not destroy the solution";

  // The proof that the repeats were ignored: an estimator that never saw them
  // ends up in exactly the same place after the same two real epochs.
  gnc::CoarseAttitudeEstimator reference(defaultConfig());
  gnc::CoarseAttitudeOutput reference_out{};
  ASSERT_TRUE(reference.update(makeInput(1.0, q0, rate), reference_out));
  ASSERT_TRUE(reference.update(makeInput(1.1, q0, rate), reference_out));

  ASSERT_TRUE(est.update(makeInput(1.1, q0, rate), out));
  EXPECT_TRUE(out.covariance == reference_out.covariance) << "stuck cycles changed nothing";
  EXPECT_LT(out.covariance.trace(), after_first_fix.trace()) << "a real new fix does reduce it";
}

TEST(CoarseAttitude, RefusedCyclesPublishNothing) {
  RecordProperty("verifies", "REQ-ADET-002");
  const pm::Quaternion q0 = pm::Quaternion::Identity();
  const gnc::CoarseAttitudeOutput pristine{};

  // A *refused* cycle — one the estimator declines to run at all — must leave
  // `out` default-constructed rather than half-written, so no consumer can find
  // a stale or non-finite covariance sitting in the struct behind a cleared
  // validity flag.
  const auto expectNothingPublished = [&](const gnc::CoarseAttitudeOutput& out, const char* what) {
    EXPECT_FALSE(out.attitude_valid) << what;
    EXPECT_FALSE(out.rate_valid) << what;
    EXPECT_FALSE(out.triad_applied) << what;
    EXPECT_EQ(out.age_s, 0.0) << what;
    EXPECT_TRUE(out.covariance.allFinite()) << what;
    EXPECT_TRUE(out.covariance == pristine.covariance) << what;
    EXPECT_TRUE(out.attitude.core().isFinite()) << what;
  };

  gnc::CoarseAttitudeOutput out{};

  gnc::CoarseAttitudeEstimator unconfigured{gnc::CoarseAttitudeConfig{}};
  ASSERT_FALSE(unconfigured.update(makeInput(0.0, q0, Eigen::Vector3d::Zero()), out));
  expectNothingPublished(out, "unconfigured");

  gnc::CoarseAttitudeEstimator est(defaultConfig());
  ASSERT_TRUE(est.update(makeInput(1.0, q0, Eigen::Vector3d::Zero()), out));
  ASSERT_FALSE(est.update(makeInput(0.5, q0, Eigen::Vector3d::Zero()), out));
  expectNothingPublished(out, "backwards clock");

  ASSERT_TRUE(est.update(makeInput(1.1, q0, Eigen::Vector3d::Zero()), out));
  ASSERT_FALSE(est.update(makeInput(1.1, q0, Eigen::Vector3d::Zero()), out));
  expectNothingPublished(out, "stuck clock");
}

TEST(CoarseAttitude, PublishesTheRateBeforeAnAttitudeExists) {
  RecordProperty("verifies", "REQ-ADET-002");
  // A cycle that simply has no attitude yet is *not* a refusal: the gyro is
  // independently valid and Safe-mode rate damping needs it before attitude
  // acquisition. The attitude side must still read as unusable.
  gnc::CoarseAttitudeEstimator est(defaultConfig());
  const Eigen::Vector3d rate(0.01, -0.02, 0.003);
  gnc::CoarseAttitudeOutput out{};

  EXPECT_FALSE(est.update(makeInput(0.0, pm::Quaternion::Identity(), rate, false, false), out));
  EXPECT_FALSE(out.attitude_valid);
  EXPECT_FALSE(est.isInitialised());
  EXPECT_TRUE(out.rate_valid);
  EXPECT_LT((out.body_rate.eigen() - rate).norm(), 1e-15);
  EXPECT_TRUE(out.covariance.allFinite());
}

TEST(CoarseAttitude, TreatsAnOverlongGapAsDropoutNotExtrapolation) {
  RecordProperty("verifies", "REQ-ADET-002");
  gnc::CoarseAttitudeConfig cfg = defaultConfig();
  cfg.max_dt_s = 1.0;
  gnc::CoarseAttitudeEstimator est(cfg);
  const Eigen::Vector3d rate(0.05, 0.0, 0.0);
  const pm::Quaternion q0 = pm::Quaternion::Identity();

  gnc::CoarseAttitudeOutput out{};
  ASSERT_TRUE(est.update(makeInput(0.0, q0, rate), out));

  // A 20 s gap with no vectors: extrapolating the stale rate would swing the
  // estimate ~57 deg on an assumption no one checked, so the attitude is held.
  ASSERT_TRUE(est.update(makeInput(20.0, q0, rate, false, true), out));
  EXPECT_LT(errorDeg(out, q0), 1e-12);
  EXPECT_NEAR(out.age_s, 20.0, 1e-9);
}

TEST(CoarseAttitude, RejectsNonFiniteMeasurements) {
  RecordProperty("verifies", "REQ-ADET-002");
  gnc::CoarseAttitudeEstimator est(defaultConfig());
  const Eigen::Vector3d rate = Eigen::Vector3d::Zero();
  const pm::Quaternion q0 = pm::Quaternion::Identity();

  gnc::CoarseAttitudeOutput out{};
  ASSERT_TRUE(est.update(makeInput(0.0, q0, rate), out));

  gnc::CoarseAttitudeInput in = makeInput(kDt, q0, rate, true, true);
  in.gyro = pm::Vec3<frames::Body>(std::nan(""), 0.0, 0.0);
  in.sun_body = pm::Vec3<frames::Body>(std::nan(""), 0.0, 0.0);
  ASSERT_TRUE(est.update(in, out)) << "bad data is excluded, the solution survives";
  EXPECT_FALSE(out.rate_valid);
  EXPECT_FALSE(out.triad_applied);
  EXPECT_TRUE(out.attitude.core().isFinite());
  EXPECT_TRUE(out.covariance.allFinite());
  EXPECT_LT(errorDeg(out, q0), 1e-12);
}

TEST(CoarseAttitude, ResetReturnsToColdStart) {
  RecordProperty("verifies", "REQ-ADET-002");
  gnc::CoarseAttitudeEstimator est(defaultConfig());
  gnc::CoarseAttitudeOutput out{};
  const pm::Quaternion q0 = pm::Quaternion::Identity();
  ASSERT_TRUE(est.update(makeInput(0.0, q0, Eigen::Vector3d::Zero()), out));
  ASSERT_TRUE(est.isInitialised());

  est.reset();
  EXPECT_FALSE(est.isInitialised());
  gnc::CoarseAttitudeInput in = makeInput(kDt, q0, Eigen::Vector3d::Zero(), false, true);
  EXPECT_FALSE(est.update(in, out));
  EXPECT_FALSE(out.attitude_valid);
}

TEST(CoarseAttitude, WritesTheCanonicalEstimatedState) {
  RecordProperty("verifies", "REQ-ADET-002");
  gnc::CoarseAttitudeEstimator est(defaultConfig());
  const Eigen::Vector3d rate(0.003, 0.001, -0.002);
  const pm::Quaternion q0 =
      pm::Quaternion::FromAxisAngle(Eigen::Vector3d(1.0, 1.0, 1.0).normalized(), 30.0 * kDeg);

  gnc::CoarseAttitudeOutput out{};
  ASSERT_TRUE(est.update(makeInput(7.0, q0, rate), out));

  polaris::state::EstimatedState state{};
  state.position = pm::Vec3<frames::ECI>(7.0e6, 0.0, 0.0);  // owned by the orbit filter
  state.valid.position = true;
  gnc::writeToEstimatedState(out, epochAt(7.0), state);

  EXPECT_EQ(state.epoch, epochAt(7.0));
  EXPECT_EQ(state.mode, polaris::state::EstimationMode::Coarse);
  EXPECT_TRUE(state.valid.attitude);
  EXPECT_TRUE(state.valid.body_rate);
  EXPECT_TRUE(state.valid.covariance);
  EXPECT_LT(errorDeg(state.attitude.core(), q0), 1e-9);
  EXPECT_LT((state.body_rate.eigen() - rate).norm(), 1e-15);
  const auto attitudeBlock = [](const polaris::state::EstimatedState& s) {
    return Eigen::Matrix3d(s.covariance.block<3, 3>(polaris::state::ErrorState::kAttitude,
                                                    polaris::state::ErrorState::kAttitude));
  };
  EXPECT_TRUE(attitudeBlock(state) == out.covariance) << "copied verbatim, not recomputed";

  // Coarse mode owns attitude only; the orbit fields are left alone (§8.0).
  EXPECT_TRUE(state.valid.position);
  EXPECT_FALSE(state.valid.gyro_bias);
  EXPECT_EQ(state.covariance(polaris::state::ErrorState::kPosition,
                             polaris::state::ErrorState::kPosition),
            0.0);

  // An invalid solution must not be published as Coarse — and must not stamp a
  // meaningless covariance over the last good one either.
  gnc::CoarseAttitudeOutput invalid{};
  gnc::writeToEstimatedState(invalid, epochAt(8.0), state);
  EXPECT_EQ(state.mode, polaris::state::EstimationMode::Invalid);
  EXPECT_FALSE(state.valid.attitude);
  EXPECT_FALSE(state.valid.covariance);
  EXPECT_TRUE(attitudeBlock(state) == out.covariance)
      << "an invalid solution leaves the previous covariance in place";
}

}  // namespace
