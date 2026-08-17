/// @file Unit tests for the fine-mode attitude MEKF (REQ-ADET-001,
/// REQ-ADET-004; design doc §8.1).
///
/// A filter is easy to make *look* right — plot the attitude error, watch it go
/// down, ship it. What actually matters for a flight estimator is whether the
/// covariance it reports is the covariance it has, because everything
/// downstream (mode arbitration, FDIR gating, the NIS gate inside this very
/// filter) reasons about the estimate through that number. So the tests are
/// built around consistency rather than around convergence:
///
///  - **NEES over Monte Carlo (REQ-ADET-004).** Many runs with independently
///    drawn gyro noise, bias random walk, initial bias, and measurement noise;
///    the mean 6-state NEES must sit inside the χ²₆ interval. Too high means
///    overconfident — the failure that makes an estimator dangerous rather than
///    merely bad — and too low means it is throwing information away.
///  - **NIS per update.** The innovations must be as large as the filter
///    predicts, no more and no less, which is the same statement checked on the
///    measurement side.
///  - **Bias observability.** The gyro bias must actually converge on the
///    injected truth. It only can because of the Φ₁₂ cross block, so that one
///    is pinned by convergence. Qd's cross blocks are orders of magnitude
///    smaller and no statistical test can see them, so they get a direct
///    algebraic check against a hand-computed `ΦPΦᵀ + Qd` instead.
///  - **The refusal paths.** Stuck clock, backwards clock, dead gyro, an
///    outlier measurement, an unconfigured filter — none may produce a
///    confident answer and none may assert.
///
/// Truth is a constant-rate spin evaluated in closed form, independently of the
/// filter's own integration of it.

#include "gnc/mekf.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>

#include "gnc/davenport.hpp"
#include "math/frames.hpp"
#include "math/quaternion.hpp"
#include "math/typed_vector.hpp"
#include "random/rng.hpp"
#include "state/estimated_state.hpp"
#include "time/timescales.hpp"

namespace {

namespace gnc = polaris::gnc;
namespace pm = polaris::math;
namespace frames = polaris::math::frames;
namespace ptime = polaris::time;

constexpr double kDeg = M_PI / 180.0;
constexpr double kDt = 0.1;  ///< estimation cycle [s] (10 Hz, §8.1 rate group)

/// Inertial references: sun along ECI +x, field ~55 deg away — the same
/// well-conditioned pair the coarse-estimator tests use.
const Eigen::Vector3d kSunEci(1.0, 0.0, 0.0);
const Eigen::Vector3d kMagEci =
    Eigen::Vector3d(std::cos(55.0 * kDeg), std::sin(55.0 * kDeg), 0.2).normalized();

ptime::Tai epochAt(double t_s) {
  return ptime::Tai::fromNanosecondsSinceEpoch(static_cast<std::int64_t>(t_s * 1.0e9));
}

/// Constant-rate truth spin, evaluated independently of the filter.
pm::Quaternion truthAttitude(const Eigen::Vector3d& rate, double t_s,
                             const pm::Quaternion& initial) {
  const double angle = rate.norm() * t_s;
  if (angle <= 0.0) {
    return initial;
  }
  return (pm::Quaternion::FromAxisAngle(rate.normalized(), angle) * initial).canonical();
}

gnc::MekfConfig defaultConfig() {
  gnc::MekfConfig cfg{};
  cfg.arw_rad_per_sqrt_s = 1.0e-4;        // ~0.34 deg/sqrt(h), a MEMS-grade gyro
  cfg.rrw_rad_per_s_per_sqrt_s = 1.0e-6;  // bias random walk
  cfg.nis_gate = 13.8;                    // chi-square(2) at 99.9%
  cfg.attitude_nis_gate = 16.27;          // chi-square(3) at 99.9%
  cfg.max_coast_s = 60.0;
  cfg.max_dt_s = 1.0;
  return cfg;
}

double errorDeg(const pm::Quaternion& est, const pm::Quaternion& q_true) {
  const pm::Quaternion dq = (est * q_true.inverse()).canonical();
  return 2.0 * std::atan2(dq.vec().norm(), dq.scalar()) / kDeg;
}

Eigen::Vector3d anyPerpendicular(const Eigen::Vector3d& u) {
  const Eigen::Vector3d seed =
      (std::abs(u.x()) < 0.9) ? Eigen::Vector3d::UnitX() : Eigen::Vector3d::UnitY();
  return u.cross(seed).normalized();
}

/// Transverse Gaussian perturbation of a unit vector, per-axis 1σ @p sigma_rad.
Eigen::Vector3d perturb(const Eigen::Vector3d& u, double sigma_rad,
                        polaris::random::SplitMix64& rng) {
  const Eigen::Vector3d t1 = anyPerpendicular(u);
  const Eigen::Vector3d t2 = u.cross(t1);
  return (u + sigma_rad * (rng.gaussian() * t1 + rng.gaussian() * t2)).normalized();
}

/// Seed a filter from a Davenport solve on the given (possibly noisy) body
/// vectors — the real cold-start path, not a hand-placed truth attitude.
bool seedFromDavenport(gnc::Mekf& filter, double t_s, const Eigen::Vector3d& sun_body,
                       const Eigen::Vector3d& mag_body, double sigma_sun, double sigma_mag,
                       const Eigen::Matrix3d& bias_cov) {
  gnc::DavenportInput in{};
  in.count = 2;
  in.min_observability = 0.05;
  in.observations[0].body = pm::Vec3<frames::Body>(sun_body);
  in.observations[0].reference = pm::Vec3<frames::ECI>(kSunEci);
  in.observations[0].sigma_rad = sigma_sun;
  in.observations[1].body = pm::Vec3<frames::Body>(mag_body);
  in.observations[1].reference = pm::Vec3<frames::ECI>(kMagEci);
  in.observations[1].sigma_rad = sigma_mag;

  gnc::DavenportSolution seed{};
  if (!gnc::davenport(in, seed)) {
    return false;
  }
  return filter.initialize(epochAt(t_s), seed.attitude, seed.covariance,
                           pm::Vec3<frames::Body>(Eigen::Vector3d::Zero()), bias_cov);
}

bool isPositiveDefinite(const gnc::Mekf::Covariance& p) {
  return p.llt().info() == Eigen::Success;
}

TEST(Mekf, RejectsInvalidConfiguration) {
  RecordProperty("verifies", "REQ-ADET-001");
  EXPECT_TRUE(defaultConfig().isValid());
  // No flight defaults: the tuning is mission configuration (§19.3), so a
  // default-constructed config must not be flyable.
  EXPECT_FALSE(gnc::MekfConfig{}.isValid());

  gnc::MekfConfig bad = defaultConfig();
  bad.arw_rad_per_sqrt_s = 0.0;
  EXPECT_FALSE(bad.isValid());

  bad = defaultConfig();
  bad.rrw_rad_per_s_per_sqrt_s = -1.0;
  EXPECT_FALSE(bad.isValid());

  bad = defaultConfig();
  bad.nis_gate = 0.0;
  EXPECT_FALSE(bad.isValid()) << "a zero gate would reject every measurement";

  bad = defaultConfig();
  bad.max_coast_s = -1.0;
  EXPECT_FALSE(bad.isValid());

  // A bias modelled as exactly constant is legal, if rarely what real hardware
  // wants.
  gnc::MekfConfig constant_bias = defaultConfig();
  constant_bias.rrw_rad_per_s_per_sqrt_s = 0.0;
  EXPECT_TRUE(constant_bias.isValid());

  gnc::Mekf inert(bad);
  EXPECT_FALSE(inert.isConfigured());
  EXPECT_FALSE(inert.initialize(
      epochAt(0.0), pm::Quat<frames::Body, frames::ECI>::Identity(), Eigen::Matrix3d::Identity(),
      pm::Vec3<frames::Body>(Eigen::Vector3d::Zero()), Eigen::Matrix3d::Identity()));
  EXPECT_FALSE(inert.isInitialised());
  EXPECT_FALSE(
      inert.propagate(epochAt(kDt), pm::Vec3<frames::Body>(Eigen::Vector3d::Zero()), true));
}

TEST(Mekf, PropagatesExactlyOnAPerfectGyro) {
  RecordProperty("verifies", "REQ-ADET-001");
  gnc::Mekf filter(defaultConfig());
  const Eigen::Vector3d rate(0.02, -0.01, 0.005);  // ~1.3 deg/s tumble
  const pm::Quaternion q0 =
      pm::Quaternion::FromAxisAngle(Eigen::Vector3d(0.2, 0.9, -0.3).normalized(), 100.0 * kDeg);

  ASSERT_TRUE(seedFromDavenport(filter, 0.0, q0.rotate(kSunEci), q0.rotate(kMagEci), 1.0 * kDeg,
                                2.0 * kDeg, 1.0e-8 * Eigen::Matrix3d::Identity()));
  EXPECT_LT(errorDeg(filter.attitude().core(), q0), 1e-9) << "the Davenport seed is exact here";

  // 30 s of pure propagation: a perfect gyro means the only error is the
  // closed-form kinematics itself, which is exact at constant rate.
  for (int step = 1; step <= 300; ++step) {
    const double t = step * kDt;
    ASSERT_TRUE(filter.propagate(epochAt(t), pm::Vec3<frames::Body>(rate), true));
    ASSERT_LT(errorDeg(filter.attitude().core(), truthAttitude(rate, t, q0)), 1e-9)
        << "step " << step;
    ASSERT_GE(filter.attitude().core().w(), 0.0) << "q0 >= 0 maintained";
  }
  EXPECT_NEAR(filter.ageSeconds(), 30.0, 1e-9);
  EXPECT_TRUE(filter.attitudeValid()) << "30 s < 60 s coast horizon";
  EXPECT_LT((filter.bodyRate().eigen() - rate).norm(), 1e-15);

  // The uncertainty must have grown, stayed symmetric under 300 similarity
  // transforms, and stayed factorisable.
  const gnc::Mekf::Covariance& p = filter.covariance();
  const Eigen::Matrix3d attitude_block = p.block<3, 3>(gnc::Mekf::kAttitude, gnc::Mekf::kAttitude);
  const Eigen::Matrix3d cross_block = p.block<3, 3>(gnc::Mekf::kAttitude, gnc::Mekf::kGyroBias);
  EXPECT_GT(attitude_block.trace(), 0.0);
  EXPECT_TRUE(p.isApprox(p.transpose(), 1e-15)) << "symmetry";
  EXPECT_TRUE(isPositiveDefinite(p));

  // Attitude/bias cross-covariance must be non-zero: it is the Φ₁₂ coupling,
  // and it is the only reason the bias is observable at all.
  EXPECT_GT(cross_block.norm(), 0.0);
}

TEST(Mekf, CovariancePropagationMatchesTheClosedFormAlgebra) {
  RecordProperty("verifies", "REQ-ADET-001");
  // A direct algebraic check of one step, because the statistical tests cannot
  // see Qd's cross blocks: at any realistic tuning they are six orders of
  // magnitude below Q11, so zeroing or sign-flipping them changes nothing a
  // Monte Carlo can resolve. Here P is seeded as I₆ and the expected
  // ΦPΦᵀ + Qd is written out by hand, which pins Φ₁₂ and all three Qd blocks —
  // including their signs — in one comparison.
  gnc::MekfConfig cfg = defaultConfig();
  cfg.rrw_rad_per_s_per_sqrt_s = 1.0e-3;  // exaggerated so Qd is unambiguous
  cfg.max_dt_s = 1.0;
  const double dt = 0.5;
  const double sv2 = cfg.arw_rad_per_sqrt_s * cfg.arw_rad_per_sqrt_s;
  const double su2 = cfg.rrw_rad_per_s_per_sqrt_s * cfg.rrw_rad_per_s_per_sqrt_s;
  const Eigen::Matrix3d id = Eigen::Matrix3d::Identity();

  const auto stepFrom = [&](bool gyro_valid, double step_s) {
    gnc::Mekf filter(cfg);
    EXPECT_TRUE(filter.initialize(epochAt(0.0), pm::Quat<frames::Body, frames::ECI>::Identity(), id,
                                  pm::Vec3<frames::Body>(Eigen::Vector3d::Zero()), id));
    EXPECT_TRUE(filter.propagate(epochAt(step_s), pm::Vec3<frames::Body>(Eigen::Vector3d::Zero()),
                                 gyro_valid));
    return filter.covariance();
  };

  // Qd is the same either way; only Φ differs.
  gnc::Mekf::Covariance q_d = gnc::Mekf::Covariance::Zero();
  q_d.block<3, 3>(gnc::Mekf::kAttitude, gnc::Mekf::kAttitude) =
      (sv2 * dt + su2 * dt * dt * dt / 3.0) * id;
  q_d.block<3, 3>(gnc::Mekf::kAttitude, gnc::Mekf::kGyroBias) = (-0.5 * su2 * dt * dt) * id;
  q_d.block<3, 3>(gnc::Mekf::kGyroBias, gnc::Mekf::kAttitude) = (-0.5 * su2 * dt * dt) * id;
  q_d.block<3, 3>(gnc::Mekf::kGyroBias, gnc::Mekf::kGyroBias) = (su2 * dt) * id;

  // Zero rate, so Φ₁₁ = I and Φ₁₂ takes the small-angle −Δt·I branch:
  // ΦPΦᵀ = [[(1+Δt²)I, −Δt·I], [−Δt·I, I]] for P = I₆.
  gnc::Mekf::Covariance expected = q_d;
  expected.block<3, 3>(gnc::Mekf::kAttitude, gnc::Mekf::kAttitude) += (1.0 + dt * dt) * id;
  expected.block<3, 3>(gnc::Mekf::kAttitude, gnc::Mekf::kGyroBias) += -dt * id;
  expected.block<3, 3>(gnc::Mekf::kGyroBias, gnc::Mekf::kAttitude) += -dt * id;
  expected.block<3, 3>(gnc::Mekf::kGyroBias, gnc::Mekf::kGyroBias) += id;
  EXPECT_LT((stepFrom(true, dt) - expected).cwiseAbs().maxCoeff(), 1e-15);

  // Without a usable gyro, Φ = I: the bias took no part in a propagation that
  // did not happen, so Φ₁₂ must be **zero**, not −Δt·I. Carrying the coupling
  // would invent a −Δt·P_bb correlation the next measurement then "corrects"
  // by dragging the bias — and at a 20 s dropout that block is −20·I.
  gnc::Mekf::Covariance held = q_d;
  held += gnc::Mekf::Covariance::Identity();
  EXPECT_LT((stepFrom(false, dt) - held).cwiseAbs().maxCoeff(), 1e-15);

  // Same for a step past max_dt_s, which is the case where the spurious
  // coupling would have been largest.
  gnc::Mekf::Covariance long_gap = gnc::Mekf::Covariance::Identity();
  const double gap = 20.0;
  long_gap.block<3, 3>(gnc::Mekf::kAttitude, gnc::Mekf::kAttitude) +=
      (sv2 * gap + su2 * gap * gap * gap / 3.0) * id;
  long_gap.block<3, 3>(gnc::Mekf::kAttitude, gnc::Mekf::kGyroBias) = (-0.5 * su2 * gap * gap) * id;
  long_gap.block<3, 3>(gnc::Mekf::kGyroBias, gnc::Mekf::kAttitude) = (-0.5 * su2 * gap * gap) * id;
  long_gap.block<3, 3>(gnc::Mekf::kGyroBias, gnc::Mekf::kGyroBias) += (su2 * gap) * id;
  EXPECT_LT((stepFrom(true, gap) - long_gap).cwiseAbs().maxCoeff(), 1e-15);
}

TEST(Mekf, SubtractsAndEstimatesGyroBias) {
  RecordProperty("verifies", "REQ-ADET-001");
  gnc::Mekf filter(defaultConfig());
  const Eigen::Vector3d rate(0.01, 0.0, -0.004);
  const Eigen::Vector3d true_bias(2.0e-4, -3.0e-4, 1.0e-4);  // ~0.06 deg/s, MEMS turn-on
  const pm::Quaternion q0 = pm::Quaternion::Identity();
  const double sigma = 0.5 * kDeg;

  ASSERT_TRUE(seedFromDavenport(filter, 0.0, q0.rotate(kSunEci), q0.rotate(kMagEci), sigma, sigma,
                                (1.0e-3 * 1.0e-3) * Eigen::Matrix3d::Identity()));
  EXPECT_LT(filter.gyroBias().eigen().norm(), 1e-15) << "seeded with no bias knowledge";

  // 120 s of noise-free measurements. The bias is observable only through the
  // Φ₁₂/Qd coupling, so this is the test a diagonal process-noise shortcut
  // fails.
  for (int step = 1; step <= 1200; ++step) {
    const double t = step * kDt;
    const pm::Quaternion q_true = truthAttitude(rate, t, q0);
    ASSERT_TRUE(filter.propagate(epochAt(t), pm::Vec3<frames::Body>(rate + true_bias), true));
    gnc::MekfUpdate up{};
    ASSERT_TRUE(filter.update(pm::Vec3<frames::Body>(q_true.rotate(kSunEci)),
                              pm::Vec3<frames::ECI>(kSunEci), sigma, up));
    ASSERT_TRUE(filter.update(pm::Vec3<frames::Body>(q_true.rotate(kMagEci)),
                              pm::Vec3<frames::ECI>(kMagEci), sigma, up));
  }

  const pm::Quaternion q_end = truthAttitude(rate, 120.0, q0);
  EXPECT_LT(errorDeg(filter.attitude().core(), q_end), 0.02);
  EXPECT_LT((filter.gyroBias().eigen() - true_bias).norm(), 0.1 * true_bias.norm())
      << "bias estimate " << filter.gyroBias().eigen().transpose() << " vs truth "
      << true_bias.transpose();
  // The published rate is bias-corrected, so it tracks the true rate even
  // though the gyro reads the biased one.
  EXPECT_LT((filter.bodyRate().eigen() - rate).norm(), 0.1 * true_bias.norm());
  EXPECT_EQ(filter.rejectedCount(), 0u);
  EXPECT_TRUE(isPositiveDefinite(filter.covariance()));
}

TEST(Mekf, ConvergesFromAColdStartUnderNoise) {
  RecordProperty("verifies", "REQ-ADET-001;REQ-ADET-003");
  // The real acquisition path: a Davenport seed on *noisy* vectors, then the
  // filter. Filtering must beat the single-frame solve it started from — that
  // is the whole reason the fine mode exists.
  gnc::Mekf filter(defaultConfig());
  polaris::random::SplitMix64 rng(0x11FE01u);
  const Eigen::Vector3d rate(0.008, -0.003, 0.002);
  const Eigen::Vector3d true_bias(1.5e-4, 2.0e-4, -1.0e-4);
  const pm::Quaternion q0 =
      pm::Quaternion::FromAxisAngle(Eigen::Vector3d(1.0, -2.0, 0.5).normalized(), 65.0 * kDeg);
  const double sigma = 1.0 * kDeg;

  ASSERT_TRUE(seedFromDavenport(filter, 0.0, perturb(q0.rotate(kSunEci), sigma, rng),
                                perturb(q0.rotate(kMagEci), sigma, rng), sigma, sigma,
                                (5.0e-4 * 5.0e-4) * Eigen::Matrix3d::Identity()));
  const double seed_error_deg = errorDeg(filter.attitude().core(), q0);
  const double seed_sigma_deg =
      std::sqrt(
          filter.covariance().block<3, 3>(gnc::Mekf::kAttitude, gnc::Mekf::kAttitude).trace() /
          3.0) /
      kDeg;

  for (int step = 1; step <= 600; ++step) {  // 60 s
    const double t = step * kDt;
    const pm::Quaternion q_true = truthAttitude(rate, t, q0);
    const Eigen::Vector3d gyro =
        rate + true_bias +
        (1.0e-4 / std::sqrt(kDt)) * Eigen::Vector3d(rng.gaussian(), rng.gaussian(), rng.gaussian());
    ASSERT_TRUE(filter.propagate(epochAt(t), pm::Vec3<frames::Body>(gyro), true));
    // Not asserted individually: at a 99.9% gate a handful of rejections over
    // 1200 updates is the χ² tail doing its job, not a fault. The rate is
    // checked below.
    gnc::MekfUpdate up{};
    filter.update(pm::Vec3<frames::Body>(perturb(q_true.rotate(kSunEci), sigma, rng)),
                  pm::Vec3<frames::ECI>(kSunEci), sigma, up);
    filter.update(pm::Vec3<frames::Body>(perturb(q_true.rotate(kMagEci), sigma, rng)),
                  pm::Vec3<frames::ECI>(kMagEci), sigma, up);
  }
  EXPECT_LT(filter.rejectedCount(), 12u) << "gate rejections must stay near the 0.1% tail";

  const double final_error_deg = errorDeg(filter.attitude().core(), truthAttitude(rate, 60.0, q0));
  const double final_sigma_deg =
      std::sqrt(
          filter.covariance().block<3, 3>(gnc::Mekf::kAttitude, gnc::Mekf::kAttitude).trace() /
          3.0) /
      kDeg;
  EXPECT_LT(final_error_deg, seed_error_deg)
      << "seed " << seed_error_deg << " deg, converged " << final_error_deg << " deg";
  EXPECT_LT(final_sigma_deg, 0.5 * seed_sigma_deg) << "the covariance must shrink too";
  EXPECT_LT(final_error_deg, 3.0 * final_sigma_deg) << "and stay honest about it";
  EXPECT_TRUE(isPositiveDefinite(filter.covariance()));
}

TEST(Mekf, MonteCarloNeesAndNisAreConsistent) {
  RecordProperty("verifies", "REQ-ADET-004");
  // The headline consistency evidence. Every stochastic input is drawn from the
  // distribution the filter assumes — initial bias from the seeded bias
  // covariance, gyro white noise at the configured ARW, bias random walk at the
  // configured RRW, measurements from the QUEST model at the σ handed to
  // `update` — so a filter whose algebra is right must come back with a mean
  // NEES of 6 and a mean NIS of ~2.
  //
  // NIS is ~2, not 3, because the innovation between two unit vectors is
  // transverse by construction while `R = σ²I` budgets for a third component
  // that carries no variance (mekf.hpp). That is the number the `nis_gate`
  // threshold must be chosen against.
  //
  // The NIS gate is opened wide for this test on purpose: gating truncates the
  // innovation sequence at its upper tail, which biases the sample mean NIS
  // low. Consistency has to be measured on the *untruncated* sequence; the gate
  // itself is exercised by NisGateRejectsAnOutlier.
  gnc::MekfConfig cfg = defaultConfig();
  cfg.nis_gate = 1.0e6;
  constexpr int kRuns = 150;
  constexpr int kSteps = 200;  // 20 s at 10 Hz
  const Eigen::Vector3d rate(0.01, -0.005, 0.003);
  const double sigma = 1.0 * kDeg;
  const double bias_sigma = 3.0e-4;  // 1σ of the drawn initial bias [rad/s]

  double nees_sum = 0.0;
  double nis_sum = 0.0;
  int nis_count = 0;
  int completed = 0;

  for (int run = 0; run < kRuns; ++run) {
    polaris::random::SplitMix64 rng(polaris::random::streamSeed(0xB1A5EDu, run));
    const pm::Quaternion q0 =
        pm::Quaternion::FromAxisAngle(Eigen::Vector3d(0.3, 0.4, -0.86).normalized(), 30.0 * kDeg);
    Eigen::Vector3d bias(bias_sigma * rng.gaussian(), bias_sigma * rng.gaussian(),
                         bias_sigma * rng.gaussian());

    gnc::Mekf filter(cfg);
    ASSERT_TRUE(seedFromDavenport(filter, 0.0, perturb(q0.rotate(kSunEci), sigma, rng),
                                  perturb(q0.rotate(kMagEci), sigma, rng), sigma, sigma,
                                  (bias_sigma * bias_sigma) * Eigen::Matrix3d::Identity()));

    for (int step = 1; step <= kSteps; ++step) {
      const double t = step * kDt;
      const pm::Quaternion q_true = truthAttitude(rate, t, q0);
      // Gyro reads the true rate plus the current bias plus ARW white noise
      // discretised as σ_v/√dt; the bias itself random-walks at σ_u√dt.
      const Eigen::Vector3d gyro =
          rate + bias +
          (cfg.arw_rad_per_sqrt_s / std::sqrt(kDt)) *
              Eigen::Vector3d(rng.gaussian(), rng.gaussian(), rng.gaussian());
      bias += (cfg.rrw_rad_per_s_per_sqrt_s * std::sqrt(kDt)) *
              Eigen::Vector3d(rng.gaussian(), rng.gaussian(), rng.gaussian());

      ASSERT_TRUE(filter.propagate(epochAt(t), pm::Vec3<frames::Body>(gyro), true));
      const Eigen::Vector3d bodies[2] = {perturb(q_true.rotate(kSunEci), sigma, rng),
                                         perturb(q_true.rotate(kMagEci), sigma, rng)};
      const Eigen::Vector3d refs[2] = {kSunEci, kMagEci};
      for (int j = 0; j < 2; ++j) {
        gnc::MekfUpdate up{};
        ASSERT_TRUE(filter.update(pm::Vec3<frames::Body>(bodies[j]), pm::Vec3<frames::ECI>(refs[j]),
                                  sigma, up));
        // Skip the first few cycles: NIS is only χ² once the linearisation
        // point has settled, and the seed's own error dominates before that.
        if (step > 20) {
          nis_sum += up.nis;
          ++nis_count;
        }
      }
    }

    double nees = 0.0;
    ASSERT_TRUE(
        filter.nees(pm::Quat<frames::Body, frames::ECI>(truthAttitude(rate, kSteps * kDt, q0)),
                    pm::Vec3<frames::Body>(bias), nees));
    nees_sum += nees;
    ++completed;
    ASSERT_TRUE(isPositiveDefinite(filter.covariance()));
  }

  ASSERT_EQ(completed, kRuns);

  const double mean_nees = nees_sum / static_cast<double>(kRuns);
  const double mean_nis = nis_sum / static_cast<double>(nis_count);
  // 6-state NEES: the 99% two-sided interval on the mean of 150 runs is
  // 6 ± 6·2.576·sqrt(2/900) ≈ 6 ± 0.73, and this filter lands at 6.01. The NIS
  // sits a couple of percent above 2 — the residual along-vector component that
  // `R = σ²I` budgets for and the measurement model does not supply.
  EXPECT_NEAR(mean_nees, 6.0, 0.8) << "mean NEES over " << kRuns << " runs";
  EXPECT_NEAR(mean_nis, 2.0, 0.1) << "mean NIS over " << nis_count << " updates";
}

TEST(Mekf, NisGateRejectsAnOutlier) {
  RecordProperty("verifies", "REQ-ADET-004");
  gnc::Mekf filter(defaultConfig());
  const pm::Quaternion q0 = pm::Quaternion::Identity();
  const double sigma = 1.0 * kDeg;
  ASSERT_TRUE(seedFromDavenport(filter, 0.0, q0.rotate(kSunEci), q0.rotate(kMagEci), sigma, sigma,
                                1.0e-8 * Eigen::Matrix3d::Identity()));

  // Settle first, so the covariance is small enough that a 30 deg error is
  // unambiguously an outlier rather than a legitimate correction.
  for (int step = 1; step <= 100; ++step) {
    ASSERT_TRUE(filter.propagate(epochAt(step * kDt),
                                 pm::Vec3<frames::Body>(Eigen::Vector3d::Zero()), true));
    gnc::MekfUpdate up{};
    ASSERT_TRUE(filter.update(pm::Vec3<frames::Body>(q0.rotate(kSunEci)),
                              pm::Vec3<frames::ECI>(kSunEci), sigma, up));
  }
  const pm::Quaternion before = filter.attitude().core();
  const gnc::Mekf::Covariance p_before = filter.covariance();
  const double age_before = filter.ageSeconds();

  // A sun vector 30 deg off — a glint, a wrong unit, a stale sample.
  const Eigen::Vector3d bad =
      pm::Quaternion::FromAxisAngle(Eigen::Vector3d::UnitZ(), 30.0 * kDeg).rotate(kSunEci);
  gnc::MekfUpdate up{};
  EXPECT_FALSE(
      filter.update(pm::Vec3<frames::Body>(bad), pm::Vec3<frames::ECI>(kSunEci), sigma, up));
  EXPECT_FALSE(up.accepted);
  EXPECT_GT(up.nis, defaultConfig().nis_gate) << "the gate must be what rejected it";
  EXPECT_GT(up.innovation.norm(), 0.4) << "diagnostics are populated on rejection";
  EXPECT_EQ(filter.rejectedCount(), 1u);

  // A rejected measurement changes nothing at all — not the attitude, not the
  // covariance, and not the age (the filter has *not* had a fresh fix).
  EXPECT_EQ(errorDeg(filter.attitude().core(), before), 0.0);
  EXPECT_TRUE(filter.covariance() == p_before);
  EXPECT_EQ(filter.ageSeconds(), age_before);

  // The good measurement that follows is still accepted: the gate rejects
  // outliers, it does not latch the filter shut.
  ASSERT_TRUE(
      filter.propagate(epochAt(10.1), pm::Vec3<frames::Body>(Eigen::Vector3d::Zero()), true));
  EXPECT_TRUE(filter.update(pm::Vec3<frames::Body>(q0.rotate(kSunEci)),
                            pm::Vec3<frames::ECI>(kSunEci), sigma, up));
  EXPECT_TRUE(up.accepted);
  EXPECT_EQ(filter.rejectedCount(), 1u);

  // Only a *commanded* reset clears the FDIR evidence. The internal fault paths
  // drop the solution without touching it, because a filter that has just been
  // dropped is when the count matters most.
  filter.reset();
  EXPECT_EQ(filter.rejectedCount(), 0u);
}

TEST(Mekf, CoastsBoundedByTheHorizonWithoutLosingTheSolution) {
  RecordProperty("verifies", "REQ-ADET-001");
  gnc::MekfConfig cfg = defaultConfig();
  cfg.max_coast_s = 5.0;
  gnc::Mekf filter(cfg);
  const Eigen::Vector3d rate(0.01, 0.0, 0.0);
  const pm::Quaternion q0 = pm::Quaternion::Identity();
  const double sigma = 1.0 * kDeg;
  ASSERT_TRUE(seedFromDavenport(filter, 0.0, q0.rotate(kSunEci), q0.rotate(kMagEci), sigma, sigma,
                                1.0e-8 * Eigen::Matrix3d::Identity()));
  EXPECT_TRUE(filter.attitudeValid());

  // Eclipse: gyro only, no measurements. Inside the horizon the attitude is
  // usable; past it, it is not.
  for (int step = 1; step <= 100; ++step) {
    const double t = step * kDt;
    ASSERT_TRUE(filter.propagate(epochAt(t), pm::Vec3<frames::Body>(rate), true));
    EXPECT_EQ(filter.attitudeValid(), t <= cfg.max_coast_s) << "t = " << t;
  }
  // Unlike the coarse mode, the solution is *not* dropped: a Kalman gain
  // against a grown covariance takes the returning measurement almost whole
  // anyway, so there is nothing to gain by discarding a converged bias.
  EXPECT_TRUE(filter.isInitialised());
  EXPECT_GT(filter.covariance().trace(), 0.0);

  const pm::Quaternion q_end = truthAttitude(rate, 10.0, q0);
  gnc::MekfUpdate up{};
  ASSERT_TRUE(filter.update(pm::Vec3<frames::Body>(q_end.rotate(kSunEci)),
                            pm::Vec3<frames::ECI>(kSunEci), sigma, up));
  ASSERT_TRUE(filter.update(pm::Vec3<frames::Body>(q_end.rotate(kMagEci)),
                            pm::Vec3<frames::ECI>(kMagEci), sigma, up));
  EXPECT_EQ(filter.ageSeconds(), 0.0);
  EXPECT_TRUE(filter.attitudeValid()) << "one good pair re-validates the solution";
  EXPECT_LT(errorDeg(filter.attitude().core(), q_end), 1.0);
}

TEST(Mekf, HoldsTheAttitudeWithoutAUsableGyro) {
  RecordProperty("verifies", "REQ-ADET-001");
  gnc::Mekf filter(defaultConfig());
  const pm::Quaternion q0 = pm::Quaternion::Identity();
  ASSERT_TRUE(seedFromDavenport(filter, 0.0, q0.rotate(kSunEci), q0.rotate(kMagEci), 1.0 * kDeg,
                                1.0 * kDeg, 1.0e-8 * Eigen::Matrix3d::Identity()));
  const double trace_before = filter.covariance().trace();

  // A dead gyro, a non-finite reading, and a step longer than `max_dt_s` are
  // the same case: the attitude is held and only the process noise grows.
  // Extrapolating a stale rate across a 20 s gap would swing the estimate on an
  // assumption nobody checked.
  ASSERT_TRUE(filter.propagate(epochAt(kDt), pm::Vec3<frames::Body>(1.0, 0.0, 0.0), false));
  EXPECT_FALSE(filter.rateValid());
  EXPECT_EQ(filter.bodyRate().eigen().norm(), 0.0);

  ASSERT_TRUE(filter.propagate(epochAt(0.2), pm::Vec3<frames::Body>(std::nan(""), 0.0, 0.0), true));
  EXPECT_FALSE(filter.rateValid());

  ASSERT_TRUE(filter.propagate(epochAt(20.2), pm::Vec3<frames::Body>(1.0, 0.0, 0.0), true));
  EXPECT_FALSE(filter.rateValid()) << "a 20 s step exceeds max_dt_s";

  EXPECT_LT(errorDeg(filter.attitude().core(), q0), 1e-12) << "attitude held, not extrapolated";
  EXPECT_GT(filter.covariance().trace(), trace_before);
  EXPECT_TRUE(filter.covariance().allFinite());
  EXPECT_TRUE(isPositiveDefinite(filter.covariance()));
}

TEST(Mekf, RefusesClockFaultsWithoutTouchingTheSolution) {
  RecordProperty("verifies", "REQ-ADET-001");
  gnc::Mekf filter(defaultConfig());
  gnc::Mekf reference(defaultConfig());
  const Eigen::Vector3d rate(0.01, 0.002, 0.0);
  const pm::Quaternion q0 = pm::Quaternion::Identity();
  const Eigen::Matrix3d bias_cov = 1.0e-8 * Eigen::Matrix3d::Identity();
  ASSERT_TRUE(seedFromDavenport(filter, 1.0, q0.rotate(kSunEci), q0.rotate(kMagEci), 1.0 * kDeg,
                                1.0 * kDeg, bias_cov));
  ASSERT_TRUE(seedFromDavenport(reference, 1.0, q0.rotate(kSunEci), q0.rotate(kMagEci), 1.0 * kDeg,
                                1.0 * kDeg, bias_cov));

  // A stuck clock is the dangerous one: repeating a zero-length step and then
  // updating would fold the same measurement in twice and manufacture
  // confidence out of nothing.
  for (int i = 0; i < 10; ++i) {
    EXPECT_FALSE(filter.propagate(epochAt(1.0), pm::Vec3<frames::Body>(rate), true));
    EXPECT_FALSE(filter.propagate(epochAt(0.5), pm::Vec3<frames::Body>(rate), true));
  }
  EXPECT_TRUE(filter.isInitialised()) << "a clock fault must not destroy the solution";

  // Bit-identity is the proof: a filter that never saw the faulted cycles ends
  // up in exactly the same place after the same two real epochs.
  ASSERT_TRUE(filter.propagate(epochAt(1.1), pm::Vec3<frames::Body>(rate), true));
  ASSERT_TRUE(reference.propagate(epochAt(1.1), pm::Vec3<frames::Body>(rate), true));
  EXPECT_TRUE(filter.covariance() == reference.covariance());
  EXPECT_EQ(filter.attitude().core().coeffs(), reference.attitude().core().coeffs());
}

TEST(Mekf, RefusesMalformedInputsAndUninitialisedUse) {
  RecordProperty("verifies", "REQ-ADET-001");
  gnc::Mekf filter(defaultConfig());
  gnc::MekfUpdate up{};

  // Nothing works before initialisation — no implicit identity attitude.
  EXPECT_FALSE(
      filter.propagate(epochAt(0.0), pm::Vec3<frames::Body>(Eigen::Vector3d::Zero()), true));
  EXPECT_FALSE(filter.update(pm::Vec3<frames::Body>(kSunEci), pm::Vec3<frames::ECI>(kSunEci),
                             1.0 * kDeg, up));
  double nees = -1.0;
  EXPECT_FALSE(filter.nees(pm::Quat<frames::Body, frames::ECI>::Identity(),
                           pm::Vec3<frames::Body>(Eigen::Vector3d::Zero()), nees));
  EXPECT_EQ(nees, -1.0) << "a refused query leaves the output untouched";

  // A seed that is not a rotation, or whose covariance is not a number.
  EXPECT_FALSE(filter.initialize(
      epochAt(0.0), pm::Quat<frames::Body, frames::ECI>(pm::Quaternion(0.0, 0.0, 0.0, 0.0)),
      Eigen::Matrix3d::Identity(), pm::Vec3<frames::Body>(Eigen::Vector3d::Zero()),
      Eigen::Matrix3d::Identity()));
  EXPECT_FALSE(filter.initialize(epochAt(0.0), pm::Quat<frames::Body, frames::ECI>::Identity(),
                                 Eigen::Matrix3d::Constant(std::nan("")),
                                 pm::Vec3<frames::Body>(Eigen::Vector3d::Zero()),
                                 Eigen::Matrix3d::Identity()));

  // A finite but **indefinite** seed covariance. `initialize` is the trust
  // boundary the component layer crosses, and an indefinite P makes S
  // indefinite, which makes the NIS gate meaningless — a negative NIS would
  // pass a one-sided "too large?" test. Both blocks are checked.
  const Eigen::Matrix3d negative_definite = -Eigen::Matrix3d::Identity();
  Eigen::Matrix3d indefinite = Eigen::Matrix3d::Identity();
  indefinite(2, 2) = -1.0e-6;
  EXPECT_FALSE(filter.initialize(epochAt(0.0), pm::Quat<frames::Body, frames::ECI>::Identity(),
                                 negative_definite, pm::Vec3<frames::Body>(Eigen::Vector3d::Zero()),
                                 Eigen::Matrix3d::Identity()));
  EXPECT_FALSE(filter.initialize(epochAt(0.0), pm::Quat<frames::Body, frames::ECI>::Identity(),
                                 indefinite, pm::Vec3<frames::Body>(Eigen::Vector3d::Zero()),
                                 Eigen::Matrix3d::Identity()));
  EXPECT_FALSE(filter.initialize(epochAt(0.0), pm::Quat<frames::Body, frames::ECI>::Identity(),
                                 Eigen::Matrix3d::Identity(),
                                 pm::Vec3<frames::Body>(Eigen::Vector3d::Zero()), indefinite))
      << "the bias block is a covariance too";
  EXPECT_FALSE(filter.isInitialised());

  const pm::Quaternion q0 = pm::Quaternion::Identity();
  ASSERT_TRUE(seedFromDavenport(filter, 0.0, q0.rotate(kSunEci), q0.rotate(kMagEci), 1.0 * kDeg,
                                1.0 * kDeg, 1.0e-8 * Eigen::Matrix3d::Identity()));
  const gnc::Mekf::Covariance p_before = filter.covariance();

  // Malformed measurements are excluded, never used: a zero-length direction, a
  // NaN, a non-positive sigma. The solution survives all three untouched.
  EXPECT_FALSE(filter.update(pm::Vec3<frames::Body>(0.0, 0.0, 0.0), pm::Vec3<frames::ECI>(kSunEci),
                             1.0 * kDeg, up));
  EXPECT_FALSE(filter.update(pm::Vec3<frames::Body>(std::nan(""), 0.0, 0.0),
                             pm::Vec3<frames::ECI>(kSunEci), 1.0 * kDeg, up));
  EXPECT_FALSE(filter.update(pm::Vec3<frames::Body>(kSunEci),
                             pm::Vec3<frames::ECI>(std::nan(""), 0.0, 0.0), 1.0 * kDeg, up));
  EXPECT_FALSE(
      filter.update(pm::Vec3<frames::Body>(kSunEci), pm::Vec3<frames::ECI>(kSunEci), 0.0, up));
  EXPECT_FALSE(up.accepted);
  EXPECT_EQ(up.nis, 0.0) << "a refused update publishes no diagnostics either";
  EXPECT_EQ(filter.rejectedCount(), 0u) << "malformed is not the same as gate-rejected";
  EXPECT_TRUE(filter.covariance() == p_before);
  EXPECT_LT(errorDeg(filter.attitude().core(), q0), 1e-12);

  filter.reset();
  EXPECT_FALSE(filter.isInitialised());
  EXPECT_TRUE(filter.isConfigured()) << "reset keeps the configuration";
}

// ── Attitude measurements: the star-tracker path (§8.2, REQ-ADET-007) ───────

namespace {

/// A star tracker's measurement covariance in body axes:
/// `σ_⊥²(I − b bᵀ) + σ_∥² b bᵀ` for boresight @p b — tight across the boresight,
/// loose about it. Same closed form the flight component builds.
Eigen::Matrix3d starCov(const Eigen::Vector3d& boresight, double sigma_perp, double sigma_par) {
  const Eigen::Vector3d b = boresight.normalized();
  const Eigen::Matrix3d bbt = b * b.transpose();
  return (sigma_perp * sigma_perp) * (Eigen::Matrix3d::Identity() - bbt) +
         (sigma_par * sigma_par) * bbt;
}

/// Seed a filter at @p q with an isotropic covariance, the shortest path to a
/// filter that is ready for an attitude update.
bool seedAt(gnc::Mekf& filter, const pm::Quaternion& q, double sigma_rad) {
  const Eigen::Matrix3d cov = (sigma_rad * sigma_rad) * Eigen::Matrix3d::Identity();
  return filter.initialize(epochAt(0.0), pm::Quat<frames::Body, frames::ECI>(q), cov,
                           pm::Vec3<frames::Body>(Eigen::Vector3d::Zero()),
                           (1.0e-4 * 1.0e-4) * Eigen::Matrix3d::Identity());
}

}  // namespace

TEST(Mekf, AttitudeUpdatePullsTheReferenceOntoTheMeasurement) {
  RecordProperty("verifies", "REQ-ADET-007");
  gnc::Mekf filter(defaultConfig());
  // Seeded 2 degrees away from truth with a *loose* seed covariance, so an
  // arcsecond-class measurement dominates and the reference should land almost on
  // it in one update.
  const pm::Quaternion truth =
      pm::Quaternion::FromAxisAngle(Eigen::Vector3d(0.3, -0.5, 0.8).normalized(), 0.7);
  const pm::Quaternion seed =
      pm::Quaternion::FromAxisAngle(Eigen::Vector3d::UnitY(), 2.0 * kDeg) * truth;
  ASSERT_TRUE(seedAt(filter, seed, 5.0 * kDeg));
  ASSERT_GT(errorDeg(filter.attitude().core(), truth), 1.9);

  gnc::MekfUpdate up{};
  ASSERT_TRUE(filter.updateAttitude(pm::Quat<frames::Body, frames::ECI>(truth),
                                    starCov(Eigen::Vector3d::UnitZ(), 1.0e-4, 2.0e-4), up));
  EXPECT_TRUE(up.accepted);
  // The innovation is the *exact* rotation vector between measurement and
  // reference, not `2·vec(δq)`: at 2 degrees the two differ in the fifth decimal,
  // and at a post-coast re-acquisition of tens of degrees they differ grossly.
  EXPECT_NEAR(up.innovation.norm(), 2.0 * kDeg, 1.0e-12);
  EXPECT_LT(errorDeg(filter.attitude().core(), truth), 0.01)
      << "an arcsecond-class measurement against a 5 degree seed barely moved the reference";
  EXPECT_TRUE(isPositiveDefinite(filter.covariance()));
  EXPECT_EQ(filter.ageSeconds(), 0.0) << "an accepted attitude update must reset the coast age";
  EXPECT_GE(filter.attitude().core().scalar(), 0.0) << "canonical q0 >= 0 (design doc 3.3)";
}

TEST(Mekf, AnisotropicRIsWhatMakesTwoNonParallelTrackersWorthCarrying) {
  // The configuration decision, at the level of the algebra. A tracker constrains
  // rotation about its own boresight ~6x more weakly than across it, so one unit
  // leaves a weak direction; a second unit whose boresight is 90 degrees away has
  // that direction as one of its *tight* ones.
  const Eigen::Vector3d b0 = Eigen::Vector3d(-1.0, 0.0, -1.0).normalized();
  const Eigen::Vector3d b1 = Eigen::Vector3d(1.0, 0.0, -1.0).normalized();
  ASSERT_NEAR(b0.dot(b1), 0.0, 1.0e-15) << "the reference vehicle's boresights are 90 deg apart";
  constexpr double kPerp = 1.0e-4;
  constexpr double kPar = 6.0e-4;

  const pm::Quaternion truth = pm::Quaternion::Identity();

  // One tracker.
  gnc::Mekf single(defaultConfig());
  ASSERT_TRUE(seedAt(single, truth, 1.0 * kDeg));
  gnc::MekfUpdate up{};
  ASSERT_TRUE(single.updateAttitude(pm::Quat<frames::Body, frames::ECI>(truth),
                                    starCov(b0, kPerp, kPar), up));

  // Two, the second at 90 degrees.
  gnc::Mekf dual(defaultConfig());
  ASSERT_TRUE(seedAt(dual, truth, 1.0 * kDeg));
  ASSERT_TRUE(dual.updateAttitude(pm::Quat<frames::Body, frames::ECI>(truth),
                                  starCov(b0, kPerp, kPar), up));
  ASSERT_TRUE(dual.updateAttitude(pm::Quat<frames::Body, frames::ECI>(truth),
                                  starCov(b1, kPerp, kPar), up));

  const Eigen::Matrix3d p_single =
      single.covariance().block<3, 3>(gnc::Mekf::kAttitude, gnc::Mekf::kAttitude);
  const Eigen::Matrix3d p_dual =
      dual.covariance().block<3, 3>(gnc::Mekf::kAttitude, gnc::Mekf::kAttitude);

  // The first unit's weak direction is its own boresight. That is where the second
  // unit buys almost everything — a factor of ~(σ_par/σ_perp)² — and it is the
  // number an isotropic R would silently throw away.
  const double weak_single = b0.transpose() * p_single * b0;
  const double weak_dual = b0.transpose() * p_dual * b0;
  EXPECT_LT(weak_dual, 0.1 * weak_single)
      << "the second tracker did not tighten the first's about-boresight direction";

  // And it must not *loosen* anything: adding information cannot increase variance
  // in any direction. Checked as a matrix inequality rather than on a few axes,
  // because a sign error in the rotated R would show up in an off-diagonal.
  const Eigen::Matrix3d difference = p_single - p_dual;
  const Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(difference);
  ASSERT_EQ(solver.info(), Eigen::Success);
  EXPECT_GE(solver.eigenvalues().minCoeff(), -1.0e-18)
      << "fusing a second tracker made some direction less certain";

  // Sanity on the anisotropy itself: after one unit, the boresight direction is
  // much less certain than the cross-boresight ones.
  const Eigen::Vector3d cross = b0.cross(Eigen::Vector3d::UnitY()).normalized();
  EXPECT_GT(weak_single, 4.0 * static_cast<double>(cross.transpose() * p_single * cross));
}

TEST(Mekf, AttitudeGateIsThreeDegreesOfFreedomAndRejectionsAreCounted) {
  gnc::Mekf filter(defaultConfig());
  const pm::Quaternion truth = pm::Quaternion::Identity();
  ASSERT_TRUE(seedAt(filter, truth, 1.0e-3));

  const Eigen::Matrix3d r = starCov(Eigen::Vector3d::UnitZ(), 1.0e-4, 2.0e-4);
  // A measurement 5 degrees out, against a seed and an R of milliradian class: far
  // past any sane gate.
  const pm::Quaternion outlier =
      pm::Quaternion::FromAxisAngle(Eigen::Vector3d::UnitX(), 5.0 * kDeg) * truth;

  const pm::Quaternion before = filter.attitude().core();
  const gnc::Mekf::Covariance p_before = filter.covariance();
  gnc::MekfUpdate up{};
  EXPECT_FALSE(filter.updateAttitude(pm::Quat<frames::Body, frames::ECI>(outlier), r, up));
  EXPECT_FALSE(up.accepted);
  EXPECT_GT(up.nis, defaultConfig().attitude_nis_gate) << "the gate must be what rejected it";
  EXPECT_EQ(filter.rejectedCount(), 1u);
  // Bit-unchanged: a rejected measurement is not applied at reduced gain, it is
  // not applied.
  EXPECT_TRUE(filter.attitude().core().coeffs() == before.coeffs());
  EXPECT_TRUE(filter.covariance() == p_before);

  // The next good measurement is still accepted — the guard is per measurement,
  // not a latch.
  EXPECT_TRUE(filter.updateAttitude(pm::Quat<frames::Body, frames::ECI>(truth), r, up));
  EXPECT_TRUE(up.accepted);
  EXPECT_EQ(filter.rejectedCount(), 1u);

  // The gate is its own value, on 3 degrees of freedom rather than the vector
  // path's 2. A filter configured with only the vector gate is not configured.
  gnc::MekfConfig missing = defaultConfig();
  missing.attitude_nis_gate = 0.0;
  EXPECT_FALSE(missing.isValid());
}

TEST(Mekf, AttitudeUpdateNisIsChiSquareOnThreeDegreesOfFreedom) {
  // The consistency claim behind the gate's threshold. If the innovation
  // covariance were wrong — a missing R rotation, an H of the wrong sign — the
  // mean NIS would not sit at 3, and no single-shot test would notice.
  polaris::random::SplitMix64 rng(0xA771);
  const Eigen::Vector3d boresight = Eigen::Vector3d(-1.0, 0.0, -1.0).normalized();
  constexpr double kPerp = 1.0e-4;
  constexpr double kPar = 6.0e-4;
  const Eigen::Matrix3d r = starCov(boresight, kPerp, kPar);
  const Eigen::Vector3d c1 = boresight.cross(Eigen::Vector3d::UnitY()).normalized();
  const Eigen::Vector3d c2 = boresight.cross(c1).normalized();

  double sum = 0.0;
  int samples = 0;
  constexpr int kRuns = 400;
  for (int run = 0; run < kRuns; ++run) {
    gnc::Mekf filter(defaultConfig());
    // Seeded exactly at truth with a covariance drawn from the same distribution
    // the measurement noise has, so S = P + R is the honest one.
    const pm::Quaternion truth = pm::Quaternion::FromAxisAngle(
        Eigen::Vector3d(rng.gaussian(), rng.gaussian(), rng.gaussian()).normalized(),
        M_PI * rng.uniform());
    const Eigen::Vector3d seed_error =
        kPerp * (rng.gaussian() * c1 + rng.gaussian() * c2) + kPar * rng.gaussian() * boresight;
    const double seed_angle = seed_error.norm();
    const pm::Quaternion seeded =
        pm::Quaternion::FromAxisAngle(seed_error / seed_angle, seed_angle) * truth;
    ASSERT_TRUE(filter.initialize(epochAt(0.0), pm::Quat<frames::Body, frames::ECI>(seeded), r,
                                  pm::Vec3<frames::Body>(Eigen::Vector3d::Zero()),
                                  (1.0e-6 * 1.0e-6) * Eigen::Matrix3d::Identity()));

    const Eigen::Vector3d noise =
        kPerp * (rng.gaussian() * c1 + rng.gaussian() * c2) + kPar * rng.gaussian() * boresight;
    const double angle = noise.norm();
    const pm::Quaternion measured = pm::Quaternion::FromAxisAngle(noise / angle, angle) * truth;

    gnc::MekfUpdate up{};
    filter.updateAttitude(pm::Quat<frames::Body, frames::ECI>(measured), r, up);
    sum += up.nis;
    ++samples;
  }
  const double mean_nis = sum / samples;
  std::printf("[attitude NIS] mean=%.3f against 3 over %d updates\n", mean_nis, samples);
  // The 99% interval on the mean of N chi-square(3) draws is 3 ± 2.576·sqrt(6/N).
  EXPECT_NEAR(mean_nis, 3.0, 2.576 * std::sqrt(6.0 / kRuns));
}

TEST(Mekf, AttitudeUpdateRefusesMalformedInput) {
  gnc::Mekf filter(defaultConfig());
  gnc::MekfUpdate up{};
  const Eigen::Matrix3d r = starCov(Eigen::Vector3d::UnitZ(), 1.0e-4, 2.0e-4);

  // Uninitialised: a refusal, not an assert.
  EXPECT_FALSE(filter.updateAttitude(pm::Quat<frames::Body, frames::ECI>::Identity(), r, up));

  ASSERT_TRUE(seedAt(filter, pm::Quaternion::Identity(), 1.0e-3));
  const pm::Quaternion before = filter.attitude().core();

  const double nan = std::numeric_limits<double>::quiet_NaN();
  EXPECT_FALSE(filter.updateAttitude(
      pm::Quat<frames::Body, frames::ECI>(pm::Quaternion(nan, 0.0, 0.0, 0.0)), r, up));
  // Finite but unnormalisable — clears a caller's finiteness gate and is refused
  // here, which is a *refusal* rather than a gate rejection and must not be
  // counted as one.
  EXPECT_FALSE(filter.updateAttitude(
      pm::Quat<frames::Body, frames::ECI>(pm::Quaternion(0.0, 0.0, 0.0, 0.0)), r, up));
  EXPECT_FALSE(filter.updateAttitude(pm::Quat<frames::Body, frames::ECI>::Identity(),
                                     Eigen::Matrix3d::Constant(nan), up));

  // **R is a trust boundary.** An indefinite R makes S indefinite, and the NIS
  // could then come back negative and sail through a one-sided gate — the one
  // failure mode a divergence guard must not have. Refused at the door instead.
  Eigen::Matrix3d indefinite = r;
  indefinite(2, 2) = -1.0;
  EXPECT_FALSE(
      filter.updateAttitude(pm::Quat<frames::Body, frames::ECI>::Identity(), indefinite, up));
  // Singular (a rank-2 "transverse" R, the QUEST-style form) is refused too: it
  // makes S singular along the boresight.
  EXPECT_FALSE(filter.updateAttitude(pm::Quat<frames::Body, frames::ECI>::Identity(),
                                     starCov(Eigen::Vector3d::UnitZ(), 1.0e-4, 0.0), up));

  EXPECT_EQ(filter.rejectedCount(), 0u) << "a refusal is not a gate rejection";
  EXPECT_TRUE(filter.attitude().core().coeffs() == before.coeffs());
  EXPECT_TRUE(filter.isInitialised()) << "malformed input must not drop the filter";
}

TEST(Mekf, WritesTheCanonicalEstimatedState) {
  RecordProperty("verifies", "REQ-ADET-001;REQ-ADET-004");
  gnc::MekfConfig cfg = defaultConfig();
  cfg.max_coast_s = 1.0;
  gnc::Mekf filter(cfg);
  const Eigen::Vector3d rate(0.003, 0.001, -0.002);
  const pm::Quaternion q0 =
      pm::Quaternion::FromAxisAngle(Eigen::Vector3d(1.0, 1.0, 1.0).normalized(), 30.0 * kDeg);
  ASSERT_TRUE(seedFromDavenport(filter, 7.0, q0.rotate(kSunEci), q0.rotate(kMagEci), 1.0 * kDeg,
                                2.0 * kDeg, 1.0e-8 * Eigen::Matrix3d::Identity()));
  ASSERT_TRUE(filter.propagate(epochAt(7.1), pm::Vec3<frames::Body>(rate), true));

  polaris::state::EstimatedState state{};
  state.position = pm::Vec3<frames::ECI>(7.0e6, 0.0, 0.0);  // owned by the orbit filter
  state.valid.position = true;
  gnc::writeToEstimatedState(filter, epochAt(7.1), state);

  EXPECT_EQ(state.epoch, epochAt(7.1));
  EXPECT_EQ(state.mode, polaris::state::EstimationMode::Fine);
  EXPECT_TRUE(state.valid.attitude);
  EXPECT_TRUE(state.valid.body_rate);
  EXPECT_TRUE(state.valid.gyro_bias);
  EXPECT_TRUE(state.valid.covariance);
  EXPECT_LT(errorDeg(state.attitude.core(), truthAttitude(rate, 0.1, q0)), 1e-9);
  EXPECT_LT((state.body_rate.eigen() - rate).norm(), 1e-15);

  using ES = polaris::state::ErrorState;
  const gnc::Mekf::Covariance& p = filter.covariance();
  EXPECT_TRUE(Eigen::Matrix3d(state.covariance.block<3, 3>(ES::kAttitude, ES::kAttitude)) ==
              Eigen::Matrix3d(p.block<3, 3>(gnc::Mekf::kAttitude, gnc::Mekf::kAttitude)));
  EXPECT_TRUE(Eigen::Matrix3d(state.covariance.block<3, 3>(ES::kGyroBias, ES::kGyroBias)) ==
              Eigen::Matrix3d(p.block<3, 3>(gnc::Mekf::kGyroBias, gnc::Mekf::kGyroBias)));
  // The cross terms matter: a consumer reasoning about attitude and bias
  // together needs the correlation, not two independent blocks.
  EXPECT_TRUE(Eigen::Matrix3d(state.covariance.block<3, 3>(ES::kAttitude, ES::kGyroBias)) ==
              Eigen::Matrix3d(p.block<3, 3>(gnc::Mekf::kAttitude, gnc::Mekf::kGyroBias)));
  EXPECT_TRUE(Eigen::Matrix3d(state.covariance.block<3, 3>(ES::kGyroBias, ES::kAttitude)) ==
              Eigen::Matrix3d(p.block<3, 3>(gnc::Mekf::kGyroBias, gnc::Mekf::kAttitude)))
      << "and the transpose, so the written 15x15 stays symmetric";

  // Fine mode owns attitude and gyro bias only; the orbit fields are left alone.
  EXPECT_TRUE(state.valid.position);
  EXPECT_FALSE(state.valid.velocity);
  EXPECT_EQ(state.covariance(ES::kPosition, ES::kPosition), 0.0);

  // Coast past the horizon: the mode drops to Invalid and the covariance blocks
  // are left as they were rather than overwritten with a meaningless one.
  const Eigen::Matrix3d attitude_block = state.covariance.block<3, 3>(ES::kAttitude, ES::kAttitude);
  for (int step = 2; step <= 30; ++step) {
    ASSERT_TRUE(filter.propagate(epochAt(7.0 + step * kDt), pm::Vec3<frames::Body>(rate), true));
  }
  ASSERT_FALSE(filter.attitudeValid());
  gnc::writeToEstimatedState(filter, epochAt(10.0), state);
  EXPECT_EQ(state.mode, polaris::state::EstimationMode::Invalid);
  EXPECT_FALSE(state.valid.attitude);
  EXPECT_FALSE(state.valid.gyro_bias);
  EXPECT_FALSE(state.valid.covariance);
  EXPECT_TRUE(Eigen::Matrix3d(state.covariance.block<3, 3>(ES::kAttitude, ES::kAttitude)) ==
              attitude_block);
}

}  // namespace

// ===========================================================================
// NESC navigation-filter usability practices (NASA/TP-2018-219822 Ch. 7, §9;
// NESC TB 20-03 items d, f, g) — Push 71
// ===========================================================================

namespace {

/// A settled filter tracking the identity attitude on a perfect sun vector.
gnc::Mekf settledFilter(double sigma_rad, int steps = 100) {
  gnc::Mekf filter(defaultConfig());
  const pm::Quaternion q0 = pm::Quaternion::Identity();
  EXPECT_TRUE(seedFromDavenport(filter, 0.0, q0.rotate(kSunEci), q0.rotate(kMagEci), sigma_rad,
                                sigma_rad, 1.0e-8 * Eigen::Matrix3d::Identity()));
  for (int step = 1; step <= steps; ++step) {
    EXPECT_TRUE(filter.propagate(epochAt(step * kDt),
                                 pm::Vec3<frames::Body>(Eigen::Vector3d::Zero()), true));
    gnc::MekfUpdate up{};
    EXPECT_TRUE(filter.update(pm::Vec3<frames::Body>(q0.rotate(kSunEci)),
                              pm::Vec3<frames::ECI>(kSunEci), sigma_rad, up));
  }
  return filter;
}

}  // namespace

/// TB 20-03 item (g) / TP §9.3: a tuning change under a running solution keeps
/// the solution. Before Push 71 the component rebuilt the filter and threw the
/// converged attitude and bias away to change a gate.
TEST(Mekf, RetuneKeepsTheSolutionAndRefusesABadConfig) {
  RecordProperty("verifies", "REQ-ADET-014");
  const double sigma = 1.0 * kDeg;
  gnc::Mekf filter = settledFilter(sigma);
  const pm::Quaternion before = filter.attitude().core();
  const gnc::Mekf::Covariance p_before = filter.covariance();
  const double age_before = filter.ageSeconds();

  gnc::MekfConfig tighter = defaultConfig();
  tighter.nis_gate = 5.99;  // chi-square(2) at 95%: a real change of behaviour
  tighter.max_coast_s = 120.0;
  ASSERT_TRUE(filter.retune(tighter));
  EXPECT_TRUE(filter.isInitialised());
  EXPECT_EQ(errorDeg(filter.attitude().core(), before), 0.0);
  EXPECT_TRUE(filter.covariance() == p_before);
  EXPECT_EQ(filter.ageSeconds(), age_before);

  // The new tuning is the one that governs: a 30 deg outlier is rejected against
  // the tighter gate, and the count is where it was (retune is not a reset).
  const Eigen::Vector3d bad =
      pm::Quaternion::FromAxisAngle(Eigen::Vector3d::UnitZ(), 30.0 * kDeg).rotate(kSunEci);
  gnc::MekfUpdate up{};
  EXPECT_FALSE(
      filter.update(pm::Vec3<frames::Body>(bad), pm::Vec3<frames::ECI>(kSunEci), sigma, up));
  EXPECT_EQ(filter.rejectedCount(), 1u);

  // An invalid upload changes nothing — not the state, not the configuration
  // in force: a running filter is never made inert by a bad table.
  gnc::MekfConfig bad_cfg = tighter;
  bad_cfg.arw_rad_per_sqrt_s = -1.0;
  EXPECT_FALSE(filter.retune(bad_cfg));
  EXPECT_TRUE(filter.isConfigured());
  EXPECT_TRUE(filter.isInitialised());
  EXPECT_FALSE(
      filter.update(pm::Vec3<frames::Body>(bad), pm::Vec3<frames::ECI>(kSunEci), sigma, up));
  EXPECT_EQ(filter.rejectedCount(), 2u) << "the tighter gate is still the one in force";
}

/// TB 20-03 item (f) / TP §9.2: the covariance is re-opened, the attitude and
/// the converged bias are untouched, and the next good measurement is taken
/// almost whole (the gain against a wide P is ~1) rather than edited.
TEST(Mekf, CovarianceReinitialisationKeepsTheState) {
  RecordProperty("verifies", "REQ-ADET-014");
  const double sigma = 1.0 * kDeg;
  gnc::Mekf filter = settledFilter(sigma);
  const pm::Quaternion before = filter.attitude().core();
  const Eigen::Vector3d bias_before = filter.gyroBias().eigen();

  EXPECT_FALSE(filter.reinitializeCovariance(0.0, 1.0e-4));
  EXPECT_FALSE(filter.reinitializeCovariance(0.1, -1.0));
  ASSERT_TRUE(filter.reinitializeCovariance(10.0 * kDeg, 1.0e-4));
  EXPECT_EQ(errorDeg(filter.attitude().core(), before), 0.0);
  EXPECT_TRUE(filter.gyroBias().eigen() == bias_before);
  const gnc::Mekf::Covariance p = filter.covariance();
  EXPECT_NEAR(p(0, 0), (10.0 * kDeg) * (10.0 * kDeg), 1e-12);
  EXPECT_NEAR(p(3, 3), 1.0e-8, 1e-20);
  EXPECT_EQ(p(0, 3), 0.0);
  EXPECT_TRUE(filter.covarianceHealthy());

  // A measurement that the settled covariance would have gated is now taken —
  // that is the point of re-opening P: the filter can be pulled back without a
  // reset and without losing the bias.
  const Eigen::Vector3d off =
      pm::Quaternion::FromAxisAngle(Eigen::Vector3d::UnitZ(), 5.0 * kDeg).rotate(kSunEci);
  gnc::MekfUpdate up{};
  ASSERT_TRUE(
      filter.propagate(epochAt(10.1), pm::Vec3<frames::Body>(Eigen::Vector3d::Zero()), true));
  EXPECT_TRUE(
      filter.update(pm::Vec3<frames::Body>(off), pm::Vec3<frames::ECI>(kSunEci), sigma, up));
  EXPECT_TRUE(up.accepted);
  EXPECT_FALSE(up.forced);
  EXPECT_GT(errorDeg(filter.attitude().core(), before), 3.0);

  gnc::Mekf uninitialised(defaultConfig());
  EXPECT_FALSE(uninitialised.reinitializeCovariance(0.1, 1e-4));
  EXPECT_TRUE(uninitialised.covarianceHealthy()) << "no solution: nothing to be indefinite";
}

/// TB 20-03 item (d) / TP §9.1: the "force" editing flag applies a measurement
/// the gate would reject, and says so — counted apart from both the rejections
/// and the consistent acceptances.
TEST(Mekf, ForceOverridesTheGateAndIsCountedApart) {
  RecordProperty("verifies", "REQ-ADET-014");
  const double sigma = 1.0 * kDeg;
  gnc::Mekf filter = settledFilter(sigma);
  const pm::Quaternion before = filter.attitude().core();

  const Eigen::Vector3d bad =
      pm::Quaternion::FromAxisAngle(Eigen::Vector3d::UnitZ(), 30.0 * kDeg).rotate(kSunEci);
  gnc::MekfUpdate up{};
  EXPECT_TRUE(filter.update(pm::Vec3<frames::Body>(bad), pm::Vec3<frames::ECI>(kSunEci), sigma, up,
                            /*force=*/true));
  EXPECT_TRUE(up.accepted);
  EXPECT_TRUE(up.forced);
  EXPECT_GT(up.nis, defaultConfig().nis_gate);
  EXPECT_EQ(filter.forcedCount(), 1u);
  EXPECT_EQ(filter.rejectedCount(), 0u);
  // Applied at the settled gain — a fraction of the 30 deg, but not zero, which
  // is what a rejection would leave.
  EXPECT_GT(errorDeg(filter.attitude().core(), before), 0.1) << "the update was applied";

  // The attitude path has the same flag.
  const pm::Quaternion far_off =
      pm::Quaternion::FromAxisAngle(Eigen::Vector3d::UnitY(), 40.0 * kDeg);
  const Eigen::Matrix3d r = (0.01 * kDeg) * (0.01 * kDeg) * Eigen::Matrix3d::Identity();
  gnc::MekfUpdate att{};
  EXPECT_FALSE(filter.updateAttitude(pm::Quat<frames::Body, frames::ECI>(far_off), r, att));
  EXPECT_EQ(filter.rejectedCount(), 1u);
  EXPECT_TRUE(filter.updateAttitude(pm::Quat<frames::Body, frames::ECI>(far_off), r, att,
                                    /*force=*/true));
  EXPECT_TRUE(att.forced);
  EXPECT_EQ(filter.forcedCount(), 2u);
  EXPECT_LT(errorDeg(filter.attitude().core(), far_off), 1.0);

  filter.reset();
  EXPECT_EQ(filter.forcedCount(), 0u);
}
