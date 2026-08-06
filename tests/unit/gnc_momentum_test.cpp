/// @file Unit tests for the §8.5 momentum management, magnetic desaturation and
/// disturbance feedforward (lib/gnc/momentum, lib/gnc/disturbance).
/// REQ-ACTL-009, REQ-ACTL-010, REQ-ACTL-011.
///
/// These pin the *math*, off-target and without an F´ topology, so the component
/// test only has to cover what the component adds. Three things are worth stating
/// about what is asserted here rather than measured in SITL:
///
///  * **Dissipativity is a property, not a trend.** The desaturation law is
///    checked term by term against its own Lyapunov argument, including under the
///    per-rod clamp — because a clamp that broke the argument would still *look*
///    like a working desaturation for as long as it happened not to saturate.
///  * **The hysteresis is tested at its boundary**, in both directions, and for
///    the absence of chatter exactly at the thresholds.
///  * **The observer is tested for what it must *not* flag** as much as for what
///    it must: an orbital-frequency gravity-gradient signature that tier 1 models
///    is not an anomaly, and a monitor that fires on it would alarm twice an orbit
///    on a healthy vehicle.

#include <gtest/gtest.h>

#include <cmath>
#include <Eigen/Geometry>

#include "gnc/disturbance.hpp"
#include "gnc/momentum.hpp"
#include "math/frames.hpp"
#include "math/typed_vector.hpp"

namespace {

namespace pm = polaris::math;
namespace gnc = polaris::gnc;
using Body = pm::frames::Body;
using Vec3B = pm::Vec3<Body>;

constexpr std::int64_t kNsPerSecond = 1000000000LL;

/// The reference vehicle's four-wheel pyramid, spin axes on the body diagonals,
/// and its RW-X rotor inertia (0.5 N·m·s / 628.3 rad/s).
gnc::MomentumConfig pyramidMomentum() {
  gnc::MomentumConfig c;
  c.wheel_count = 4;
  const double s = 1.0 / std::sqrt(3.0);
  const Eigen::Vector3d axes[4] = {{s, s, s}, {-s, s, s}, {-s, -s, s}, {s, -s, s}};
  for (int i = 0; i < 4; ++i) {
    c.spin_axes.col(i) = axes[i];
  }
  c.rotor_inertia_kgm2 = 7.9577e-4;
  c.target_nms = Eigen::Vector3d::Zero();
  c.desat_enter_nms = 1.0e-3;
  c.desat_exit_nms = 3.0e-4;
  c.desat_confirm_cycles = 5;
  c.envelope_nms = 2.0e-3;
  return c;
}

gnc::MtqDesatConfig desatConfig(double duty = 0.5) {
  gnc::MtqDesatConfig c;
  c.gain_per_s = 0.2;
  c.duty_factor = duty;
  return c;
}

gnc::DisturbanceObserverConfig observerConfig(double tau = 20.0) {
  gnc::DisturbanceObserverConfig c;
  c.tau_s = tau;
  c.max_dt_s = 0.5;
  c.anomaly_torque_nm = 1.0e-5;
  c.anomaly_clear_nm = 0.8e-5;
  c.anomaly_cycles = 10;
  return c;
}

/// Feed one set of wheel speeds through the manager, all units valid.
gnc::MomentumState fold(gnc::MomentumManager& m, const double (&speeds)[4]) {
  const bool valid[4] = {true, true, true, true};
  gnc::MomentumState out;
  m.update(speeds, valid, out);
  return out;
}

// ======================================================================
// Momentum accounting
// ======================================================================

TEST(MomentumManager, StoredMomentumIsTheWeightedAxisSum) {
  gnc::MomentumManager manager(pyramidMomentum());
  ASSERT_TRUE(manager.isConfigured());

  // All four wheels at the same speed: the pyramid's X and Y contributions
  // cancel and only the common +Z component survives — 4 * I * w / sqrt(3).
  const double speeds[4] = {10.0, 10.0, 10.0, 10.0};
  const gnc::MomentumState state = fold(manager, speeds);
  ASSERT_TRUE(state.valid);
  const double expected_z = 4.0 * 7.9577e-4 * 10.0 / std::sqrt(3.0);
  EXPECT_NEAR(state.stored_nms.eigen().x(), 0.0, 1e-15);
  EXPECT_NEAR(state.stored_nms.eigen().y(), 0.0, 1e-15);
  EXPECT_NEAR(state.stored_nms.eigen().z(), expected_z, 1e-12);
  EXPECT_NEAR(state.stored_norm_nms, expected_z, 1e-12);
  // Target is zero, so the error is the stored momentum itself.
  EXPECT_NEAR(state.error_norm_nms, expected_z, 1e-12);
}

TEST(MomentumManager, TargetOffsetsTheErrorButNotTheStoredMomentum) {
  gnc::MomentumConfig config = pyramidMomentum();
  config.target_nms = Eigen::Vector3d(0.0, 0.0, 1.0e-3);
  gnc::MomentumManager manager(config);
  const double speeds[4] = {1.0, 1.0, 1.0, 1.0};
  const gnc::MomentumState state = fold(manager, speeds);
  ASSERT_TRUE(state.valid);
  const double stored_z = 4.0 * 7.9577e-4 / std::sqrt(3.0);
  EXPECT_NEAR(state.stored_nms.eigen().z(), stored_z, 1e-12);
  EXPECT_NEAR(state.error_nms.eigen().z(), stored_z - 1.0e-3, 1e-12);
}

TEST(MomentumManager, OneInvalidWheelRefusesTheWholeCycle) {
  // A missing term is not a small error in the sum — it is an unknown vector of
  // that wheel's full magnitude, and treating it as zero would understate the
  // momentum by exactly what a runaway wheel is contributing.
  gnc::MomentumManager manager(pyramidMomentum());
  const double speeds[4] = {100.0, 100.0, 100.0, 100.0};
  const bool valid[4] = {true, true, false, true};
  gnc::MomentumState out;
  EXPECT_FALSE(manager.update(speeds, valid, out));
  EXPECT_EQ(out.refusal, gnc::MomentumRefusal::kWheelInvalid);
  // The refusal names the wheel, so the EVR can name the unit instead of the
  // operator diffing four tachometer channels.
  EXPECT_EQ(out.refused_wheel, 2);
  EXPECT_FALSE(out.valid);

  const double nan_speeds[4] = {100.0, std::nan(""), 100.0, 100.0};
  const bool all_valid[4] = {true, true, true, true};
  EXPECT_FALSE(manager.update(nan_speeds, all_valid, out));
  EXPECT_EQ(out.refusal, gnc::MomentumRefusal::kBadInput);
  EXPECT_EQ(out.refused_wheel, 1);
}

TEST(MomentumManager, UnconfiguredIsInert) {
  gnc::MomentumConfig bad = pyramidMomentum();
  bad.desat_exit_nms = bad.desat_enter_nms;  // empty deadband
  gnc::MomentumManager manager(bad);
  EXPECT_FALSE(manager.isConfigured());
  const double speeds[4] = {1.0, 1.0, 1.0, 1.0};
  const bool valid[4] = {true, true, true, true};
  gnc::MomentumState out;
  EXPECT_FALSE(manager.update(speeds, valid, out));
  EXPECT_EQ(out.refusal, gnc::MomentumRefusal::kUnconfigured);

  // And the envelope may not sit below the threshold that prevents it.
  gnc::MomentumConfig inverted = pyramidMomentum();
  inverted.envelope_nms = 0.5 * inverted.desat_enter_nms;
  EXPECT_FALSE(inverted.isValid());
}

// ======================================================================
// The desaturation predicate: hysteresis at its boundary
// ======================================================================

/// Speed, all four wheels together, that stores @p momentum about +Z.
double speedFor(double momentum_nms) {
  return momentum_nms * std::sqrt(3.0) / (4.0 * 7.9577e-4);
}

TEST(MomentumManager, DesatEngagesAboveEnterAndClearsOnlyAfterConfirmation) {
  gnc::MomentumManager manager(pyramidMomentum());

  // Starts *not* desaturating: an empty array must not begin by driving rods.
  double s = speedFor(5.0e-4);
  double speeds[4] = {s, s, s, s};
  EXPECT_FALSE(fold(manager, speeds).desat_required);

  // Above the enter threshold: engaged on the first cycle. Waiting for a streak
  // to *engage* would let the momentum keep growing while the evidence
  // accumulated, and the evidence is a directly measured quantity.
  s = speedFor(1.2e-3);
  double loaded[4] = {s, s, s, s};
  EXPECT_TRUE(fold(manager, loaded).desat_required);

  // Between the two thresholds: the verdict is held, because the deadband is
  // exactly the region in which no new decision is warranted.
  s = speedFor(6.0e-4);
  double between[4] = {s, s, s, s};
  for (int i = 0; i < 20; ++i) {
    EXPECT_TRUE(fold(manager, between).desat_required) << "cycle " << i;
  }

  // Under the exit threshold: still engaged until the confirmation count is met,
  // then clear. Both edges asserted, so "it clears" is not read off a latch that
  // was never set.
  s = speedFor(1.0e-4);
  double empty[4] = {s, s, s, s};
  for (std::uint32_t i = 0; i < 4; ++i) {
    EXPECT_TRUE(fold(manager, empty).desat_required) << "confirmation cycle " << i;
  }
  EXPECT_FALSE(fold(manager, empty).desat_required);
}

TEST(MomentumManager, DoesNotChatterAtEitherThreshold) {
  // The boundary itself, which is where a strict-vs-non-strict comparison bug
  // lives. Sitting exactly on the enter threshold must not engage (the law is
  // "above"), and sitting exactly on the exit threshold must not clear.
  gnc::MomentumManager manager(pyramidMomentum());
  double s = speedFor(1.0e-3);
  double at_enter[4] = {s, s, s, s};
  for (int i = 0; i < 50; ++i) {
    EXPECT_FALSE(fold(manager, at_enter).desat_required) << "cycle " << i;
  }

  s = speedFor(1.5e-3);
  double above[4] = {s, s, s, s};
  ASSERT_TRUE(fold(manager, above).desat_required);
  s = speedFor(3.0e-4);
  double at_exit[4] = {s, s, s, s};
  for (int i = 0; i < 50; ++i) {
    EXPECT_TRUE(fold(manager, at_exit).desat_required) << "cycle " << i;
  }
}

TEST(MomentumManager, EnvelopeIsOnTheStoredMomentumAndIsIndependentOfTheLatch) {
  gnc::MomentumManager manager(pyramidMomentum());
  double s = speedFor(1.5e-3);
  double inside[4] = {s, s, s, s};
  gnc::MomentumState state = fold(manager, inside);
  EXPECT_TRUE(state.desat_required);
  EXPECT_FALSE(state.envelope_exceeded);

  s = speedFor(2.5e-3);
  double outside[4] = {s, s, s, s};
  state = fold(manager, outside);
  EXPECT_TRUE(state.envelope_exceeded);

  // ...and it recovers on the same comparison, with no confirmation count: the
  // envelope is a statement about the present state, not a fault identification.
  state = fold(manager, inside);
  EXPECT_FALSE(state.envelope_exceeded);
}

// ======================================================================
// The cross-product desaturation law
// ======================================================================

TEST(MtqDesaturation, TorqueOpposesThePerpendicularMomentumError) {
  const Vec3B field(Eigen::Vector3d(3.0e-5, 0.0, 0.0));
  const Vec3B error(Eigen::Vector3d(0.0, 1.0e-3, 0.0));  // fully perpendicular
  gnc::MtqDesatResult out;
  ASSERT_TRUE(gnc::mtqDesaturation(desatConfig(1.0), error, field, out));

  // tau = -k_d * dh_perp exactly, when dh has no component along B.
  const Eigen::Vector3d expected = -0.2 * error.eigen();
  EXPECT_NEAR((out.torque_nm.eigen() - expected).norm(), 0.0, 1e-15);
  // ...and the momentum norm is therefore strictly decreasing.
  EXPECT_LT(error.eigen().dot(out.torque_nm.eigen()), 0.0);
}

TEST(MtqDesaturation, MomentumAlongTheFieldProducesNoDipoleAtAll) {
  // The physical ceiling of magnetic unloading, stated as a test rather than as
  // a comment: m x B has no component along B, so the momentum parallel to the
  // field is untouchable this instant and comes off only as the field turns.
  const Vec3B field(Eigen::Vector3d(0.0, 0.0, 3.0e-5));
  const Vec3B error(Eigen::Vector3d(0.0, 0.0, 2.0e-3));
  gnc::MtqDesatResult out;
  ASSERT_TRUE(gnc::mtqDesaturation(desatConfig(), error, field, out));
  EXPECT_NEAR(out.dipole_am2.eigen().norm(), 0.0, 1e-18);
  EXPECT_NEAR(out.torque_nm.eigen().norm(), 0.0, 1e-18);
}

TEST(MtqDesaturation, DutyFactorScalesTheDemandSoTheAverageIsTheGain) {
  const Vec3B field(Eigen::Vector3d(0.0, 3.0e-5, 0.0));
  const Vec3B error(Eigen::Vector3d(1.0e-3, 0.0, 0.0));
  gnc::MtqDesatResult full;
  gnc::MtqDesatResult half;
  ASSERT_TRUE(gnc::mtqDesaturation(desatConfig(1.0), error, field, full));
  ASSERT_TRUE(gnc::mtqDesaturation(desatConfig(0.5), error, field, half));
  EXPECT_NEAR(half.dipole_am2.eigen().norm(), 2.0 * full.dipole_am2.eigen().norm(), 1e-12);
}

TEST(MtqDesaturation, FieldNormalisationMakesTheTorqueFieldStrengthIndependent) {
  // The reason for the 1/|B|^2: the *torque*, and hence the unloading rate, must
  // not vary by 4x between the equator and the poles.
  const Vec3B error(Eigen::Vector3d(1.0e-3, 5.0e-4, -2.0e-4));
  gnc::MtqDesatResult weak;
  gnc::MtqDesatResult strong;
  ASSERT_TRUE(
      gnc::mtqDesaturation(desatConfig(), error, Vec3B(Eigen::Vector3d(0.0, 2.0e-5, 0.0)), weak));
  ASSERT_TRUE(
      gnc::mtqDesaturation(desatConfig(), error, Vec3B(Eigen::Vector3d(0.0, 6.0e-5, 0.0)), strong));
  EXPECT_NEAR((weak.torque_nm.eigen() - strong.torque_nm.eigen()).norm(), 0.0, 1e-18);
}

/// Componentwise clamp to @p limit in the body-aligned rod triad — the flight
/// component's `clampDipoleToRods` for an orthonormal triad equal to the body
/// axes, which is what the reference vehicle flies.
Eigen::Vector3d clampPerRod(const Eigen::Vector3d& m, double limit) {
  Eigen::Vector3d out = m;
  for (int i = 0; i < 3; ++i) {
    out[i] = std::max(-limit, std::min(limit, out[i]));
  }
  return out;
}

TEST(MtqDesaturation, StaysDissipativeUnderPerRodClampingTermByTerm) {
  // The clamp's own Lyapunov argument, checked as an identity rather than as a
  // trend: d/dt |dh|^2 = 2 m . (B x dh), and every *term* of that dot product is
  // non-positive because componentwise clamping preserves each component's sign.
  // A direction-preserving scale would also pass the aggregate test below while
  // a sign-breaking clamp (say, wrapping instead of saturating) would not — which
  // is why the per-term assertion is here.
  std::srand(20260804);
  for (int trial = 0; trial < 200; ++trial) {
    const Eigen::Vector3d dh = Eigen::Vector3d::Random() * 5.0e-3;
    const Eigen::Vector3d b = Eigen::Vector3d::Random().normalized() * 4.0e-5;
    gnc::MtqDesatResult out;
    ASSERT_TRUE(gnc::mtqDesaturation(desatConfig(0.5), Vec3B(dh), Vec3B(b), out));

    const Eigen::Vector3d v = b.cross(dh);
    const Eigen::Vector3d clamped = clampPerRod(out.dipole_am2.eigen(), 15.0);
    for (int i = 0; i < 3; ++i) {
      EXPECT_LE(clamped[i] * v[i], 0.0) << "trial " << trial << " component " << i;
    }
    // The aggregate the components imply: the momentum norm strictly decreases
    // whenever any perpendicular component remains.
    const Eigen::Vector3d torque = clamped.cross(b);
    EXPECT_LT(dh.dot(torque), 0.0) << "trial " << trial;
  }
}

TEST(MtqDesaturation, ClampedCommandsAtSaturationStillReduceTheMomentum) {
  // A saturating case specifically: a momentum error deliberately far past this
  // vehicle's envelope, so the demand exceeds the rods' rating on *all three*
  // axes at once and every component clamps. The law is generic in the momentum,
  // so an out-of-envelope number here tests the clamp rather than the vehicle.
  const Vec3B error(Eigen::Vector3d(4.0e-2, -3.0e-2, 2.0e-2));
  const Vec3B field(Eigen::Vector3d(1.0e-5, 2.0e-5, -1.5e-5));
  gnc::MtqDesatResult out;
  ASSERT_TRUE(gnc::mtqDesaturation(desatConfig(0.5), error, field, out));
  ASSERT_GT(out.dipole_am2.eigen().cwiseAbs().minCoeff(), 15.0) << "test needs a saturating case";
  const Eigen::Vector3d clamped = clampPerRod(out.dipole_am2.eigen(), 15.0);
  EXPECT_LT(error.eigen().dot(clamped.cross(field.eigen())), 0.0);
}

TEST(MtqDesaturation, RefusesUnusableInputs) {
  gnc::MtqDesatResult out;
  const Vec3B error(Eigen::Vector3d(1.0e-3, 0.0, 0.0));
  EXPECT_FALSE(gnc::mtqDesaturation(gnc::MtqDesatConfig{}, error,
                                    Vec3B(Eigen::Vector3d(3.0e-5, 0.0, 0.0)), out));
  EXPECT_EQ(out.refusal, gnc::DesatRefusal::kUnconfigured);
  EXPECT_FALSE(gnc::mtqDesaturation(desatConfig(), error, Vec3B(Eigen::Vector3d::Zero()), out));
  EXPECT_EQ(out.refusal, gnc::DesatRefusal::kBadInput);
  EXPECT_FALSE(gnc::mtqDesaturation(desatConfig(), Vec3B(Eigen::Vector3d(std::nan(""), 0, 0)),
                                    Vec3B(Eigen::Vector3d(3.0e-5, 0.0, 0.0)), out));
  EXPECT_EQ(out.refusal, gnc::DesatRefusal::kBadInput);
  EXPECT_NEAR(out.dipole_am2.eigen().norm(), 0.0, 0.0);  // zero is the safe command
}

// ======================================================================
// Tier 1: model-based feedforward
// ======================================================================

TEST(DisturbanceModel, GravityGradientMatchesTheAnalyticTorque) {
  // Closed form for a diagonal inertia and a nadir at 45 deg in the body X-Z
  // plane: tau = 3 n^2 (n_hat x J n_hat) has a single Y component
  // 3 n^2 (Jxx - Jzz) * nx * nz — restoring toward the minimum-inertia axis
  // along nadir, which is the gravity-gradient stability condition.
  const Eigen::Matrix3d j = Eigen::Vector3d(0.12, 0.12, 0.10).asDiagonal();
  const double n = 1.107e-3;
  const double c = 1.0 / std::sqrt(2.0);
  const Vec3B nadir(Eigen::Vector3d(c, 0.0, c));
  Vec3B torque;
  ASSERT_TRUE(gnc::gravityGradientTorque(j, nadir, n, torque));

  const double expected_y = 3.0 * n * n * (0.12 - 0.10) * c * c;
  EXPECT_NEAR(torque.eigen().x(), 0.0, 1e-20);
  EXPECT_NEAR(torque.eigen().y(), expected_y, 1e-18);
  EXPECT_NEAR(torque.eigen().z(), 0.0, 1e-20);

  // Nadir along a principal axis is an equilibrium: no torque at all.
  ASSERT_TRUE(gnc::gravityGradientTorque(j, Vec3B(Eigen::Vector3d::UnitZ()), n, torque));
  EXPECT_NEAR(torque.eigen().norm(), 0.0, 1e-20);
  // ...and an inertially symmetric body has none in any orientation.
  const Eigen::Matrix3d sphere = Eigen::Vector3d(0.11, 0.11, 0.11).asDiagonal();
  ASSERT_TRUE(gnc::gravityGradientTorque(sphere, nadir, n, torque));
  EXPECT_NEAR(torque.eigen().norm(), 0.0, 1e-20);
}

TEST(DisturbanceModel, GravityGradientRefusesGeometryItCannotUse) {
  const Eigen::Matrix3d j = Eigen::Vector3d(0.12, 0.12, 0.10).asDiagonal();
  Vec3B torque(Eigen::Vector3d(1.0, 2.0, 3.0));
  const Vec3B keep = torque;
  // Not a unit vector: its length would scale the torque silently.
  EXPECT_FALSE(
      gnc::gravityGradientTorque(j, Vec3B(Eigen::Vector3d(0.0, 0.0, 2.0)), 1.0e-3, torque));
  EXPECT_FALSE(gnc::gravityGradientTorque(j, Vec3B(Eigen::Vector3d::UnitZ()), 0.0, torque));
  EXPECT_FALSE(gnc::gravityGradientTorque(j, Vec3B(Eigen::Vector3d(std::nan(""), 0.0, 0.0)), 1.0e-3,
                                          torque));
  // Refusals leave the output untouched rather than zeroing it: the caller's
  // running sum must not be silently zeroed by a term that could not be computed.
  EXPECT_EQ(torque.eigen(), keep.eigen());
}

TEST(DisturbanceModel, ResidualDipoleTorqueIsTheCrossProduct) {
  const Vec3B m(Eigen::Vector3d(0.002, -0.001, 0.0015));
  const Vec3B b(Eigen::Vector3d(1.0e-5, 2.0e-5, -3.0e-5));
  Vec3B torque;
  ASSERT_TRUE(gnc::residualDipoleTorque(m, b, torque));
  EXPECT_NEAR((torque.eigen() - m.eigen().cross(b.eigen())).norm(), 0.0, 1e-24);
  EXPECT_FALSE(gnc::residualDipoleTorque(Vec3B(Eigen::Vector3d(std::nan(""), 0, 0)), b, torque));
}

// ======================================================================
// Tier 2: the momentum-based observer, and the §9 anomaly monitor
// ======================================================================

/// Drive the observer with a vehicle whose total momentum grows at exactly
/// @p torque (the signature of a constant external torque absorbed by the
/// wheels), at rest, for @p cycles of 0.1 s, and return the final result.
gnc::DisturbanceResult driveConstantTorque(
    gnc::DisturbanceObserver& observer, const Eigen::Vector3d& torque, int cycles,
    const Eigen::Vector3d& modelled = Eigen::Vector3d::Zero(), std::int64_t start_ns = 0) {
  gnc::DisturbanceResult out;
  Eigen::Vector3d momentum = Eigen::Vector3d::Zero();
  for (int i = 0; i <= cycles; ++i) {
    const std::int64_t t = start_ns + static_cast<std::int64_t>(i) * (kNsPerSecond / 10);
    observer.update(Vec3B(momentum), Vec3B(Eigen::Vector3d::Zero()), Vec3B(modelled), t, out);
    momentum += torque * 0.1;
  }
  return out;
}

TEST(DisturbanceObserver, ConvergesToAnInjectedConstantTorque) {
  gnc::DisturbanceObserver observer(observerConfig(20.0));
  const Eigen::Vector3d truth(1.5e-4, -0.8e-4, 0.4e-4);

  // One time constant: a first-order filter is at 1 - 1/e = 63% of its input.
  gnc::DisturbanceResult at_tau = driveConstantTorque(observer, truth, 200);
  ASSERT_TRUE(at_tau.valid);
  EXPECT_NEAR(at_tau.torque_nm.eigen().norm() / truth.norm(), 1.0 - std::exp(-1.0), 0.02);

  // Five: 99.3%, i.e. converged for any practical purpose, and pointing the same
  // way — a filter that converged to the right magnitude on the wrong axis would
  // feed forward a torque that makes the pointing worse.
  gnc::DisturbanceObserver converged(observerConfig(20.0));
  const gnc::DisturbanceResult at_5tau = driveConstantTorque(converged, truth, 1000);
  ASSERT_TRUE(at_5tau.valid);
  EXPECT_NEAR((at_5tau.torque_nm.eigen() - truth).norm() / truth.norm(), 0.0, 0.01);
}

TEST(DisturbanceObserver, SubtractsTheModelledTorqueSoOnlyTheUnmodelledPartSurvives) {
  // The vehicle sees `truth`; tier 1 models `modelled` of it. What the observer
  // must report is the difference, because that is what is left to feed forward
  // and what the anomaly monitor must judge.
  gnc::DisturbanceObserver observer(observerConfig(20.0));
  const Eigen::Vector3d truth(2.0e-4, 0.0, 0.0);
  const Eigen::Vector3d modelled(1.4e-4, 0.0, 0.0);
  const gnc::DisturbanceResult out = driveConstantTorque(observer, truth, 1000, modelled);
  ASSERT_TRUE(out.valid);
  EXPECT_NEAR(out.torque_nm.eigen().x(), 0.6e-4, 0.05 * 0.6e-4);
}

TEST(DisturbanceObserver, AnomalyLatchesAboveTheBudgetAndClearsBelowIt) {
  gnc::DisturbanceObserver observer(observerConfig(2.0));  // short tau: converges fast
  const Eigen::Vector3d big(1.0e-4, 0.0, 0.0);             // 10x the 1e-5 budget
  driveConstantTorque(observer, big, 400);
  EXPECT_TRUE(observer.anomaly());

  // ...and it clears below the clear threshold, which is what stops the
  // exclusion being a life sentence. The torque stops; the filter decays down
  // through the deadband and the below-streak completes.
  const gnc::DisturbanceResult quiet = driveConstantTorque(
      observer, Eigen::Vector3d::Zero(), 400, Eigen::Vector3d::Zero(), 401 * (kNsPerSecond / 10));
  EXPECT_TRUE(quiet.valid);
  EXPECT_FALSE(observer.anomaly());
}

TEST(DisturbanceObserver, AnomalyHoldsInTheDeadbandInsteadOfChattering) {
  // An estimate parked *between* the clear and latch thresholds — which is what
  // a real fault at the margin looks like through the low-pass — must hold the
  // latch in whichever state it was in, not cycle it once per confirmation
  // count. Both directions, on the same parked torque. The two phases share one
  // continuous momentum trajectory (unlike `driveConstantTorque`, which starts
  // each call from zero): a discontinuity would spike the estimate through zero
  // and hand the below-streak a crossing the physics never produced.
  const Eigen::Vector3d parked(0.9e-5, 0.0, 0.0);  // inside (0.8e-5, 1.0e-5)
  const auto drive = [](gnc::DisturbanceObserver& observer, const Eigen::Vector3d& torque,
                        int cycles, Eigen::Vector3d& momentum, std::int64_t& t_ns) {
    gnc::DisturbanceResult out;
    for (int i = 0; i < cycles; ++i) {
      observer.update(Vec3B(momentum), Vec3B(Eigen::Vector3d::Zero()),
                      Vec3B(Eigen::Vector3d::Zero()), t_ns, out);
      momentum += torque * 0.1;
      t_ns += kNsPerSecond / 10;
    }
  };

  // Latched first, then the torque drops into the deadband: the estimate decays
  // monotonically from 1e-4 toward 0.9e-5 — never under the 0.8e-5 clear
  // threshold — and the latch must hold however long it sits there.
  gnc::DisturbanceObserver latched(observerConfig(2.0));
  Eigen::Vector3d momentum = Eigen::Vector3d::Zero();
  std::int64_t t_ns = 0;
  drive(latched, Eigen::Vector3d(1.0e-4, 0.0, 0.0), 400, momentum, t_ns);
  ASSERT_TRUE(latched.anomaly());
  drive(latched, parked, 2000, momentum, t_ns);
  EXPECT_NEAR(latched.estimate().eigen().norm(), parked.norm(), 0.1 * parked.norm());
  EXPECT_TRUE(latched.anomaly());

  // Never latched, then parked: stays clear — the deadband is not a back door
  // into the anomaly either.
  gnc::DisturbanceObserver clear(observerConfig(2.0));
  momentum = Eigen::Vector3d::Zero();
  t_ns = 0;
  drive(clear, parked, 2000, momentum, t_ns);
  EXPECT_NEAR(clear.estimate().eigen().norm(), parked.norm(), 0.1 * parked.norm());
  EXPECT_FALSE(clear.anomaly());
}

TEST(DisturbanceObserver, RefusesAClearThresholdAboveTheBudget) {
  gnc::DisturbanceObserverConfig bad = observerConfig();
  bad.anomaly_clear_nm = 2.0 * bad.anomaly_torque_nm;
  EXPECT_FALSE(bad.isValid());
  gnc::DisturbanceObserver observer(bad);
  EXPECT_FALSE(observer.isConfigured());
}

TEST(DisturbanceObserver, DoesNotFlagTheModelledGravityGradientSignature) {
  // **The negative that makes the monitor worth having.** An Earth-pointing
  // vehicle sees a large gravity-gradient torque at twice the orbital frequency.
  // Tier 1 models it and the observer is handed that model, so the anomaly must
  // stay silent — a monitor that fired here would alarm twice an orbit on a
  // perfectly healthy vehicle, which is how a real alert gets ignored.
  gnc::DisturbanceObserver observer(observerConfig(20.0));
  const Eigen::Matrix3d j = Eigen::Vector3d(0.12, 0.12, 0.10).asDiagonal();
  const double n = 1.107e-3;  // 500 km SSO

  Eigen::Vector3d momentum = Eigen::Vector3d::Zero();
  gnc::DisturbanceResult out;
  double worst = 0.0;
  // A quarter of an orbit at 10 Hz — 1400 s, through a full sign reversal of the
  // gravity-gradient torque, which is the part a lagging filter would smear.
  const int cycles = 14000;
  for (int i = 0; i <= cycles; ++i) {
    const double t = 0.1 * i;
    // Nadir sweeping the body X-Z plane at the orbital rate: the Earth-pointing
    // vehicle's own geometry, and the largest signature it produces.
    const Vec3B nadir(Eigen::Vector3d(std::sin(n * t), 0.0, std::cos(n * t)));
    Vec3B gg;
    ASSERT_TRUE(gnc::gravityGradientTorque(j, nadir, n, gg));
    observer.update(Vec3B(momentum), Vec3B(Eigen::Vector3d::Zero()), gg,
                    static_cast<std::int64_t>(i) * (kNsPerSecond / 10), out);
    if (out.valid) {
      worst = std::max(worst, out.torque_nm.eigen().norm());
    }
    momentum += gg.eigen() * 0.1;  // the wheels absorb exactly the modelled torque
  }
  EXPECT_FALSE(observer.anomaly());
  // Not merely under the latch count: the estimate itself stays far below the
  // budget throughout, so the margin is asserted rather than the outcome.
  EXPECT_LT(worst, 0.1 * observerConfig().anomaly_torque_nm);
}

TEST(DisturbanceObserver, RefusesUnusableCyclesAndHoldsTheEstimate) {
  gnc::DisturbanceObserver observer(observerConfig(20.0));
  const Eigen::Vector3d truth(1.0e-4, 0.0, 0.0);
  driveConstantTorque(observer, truth, 500);
  const Eigen::Vector3d converged = observer.estimate().eigen();
  ASSERT_GT(converged.norm(), 0.0);

  gnc::DisturbanceResult out;
  const std::int64_t last = 500 * (kNsPerSecond / 10);
  // A stuck clock: refused, estimate held. Re-differencing at the same epoch
  // divides by zero; a backwards tag inverts the sign of the torque.
  EXPECT_FALSE(observer.update(Vec3B(Eigen::Vector3d::Zero()), Vec3B(Eigen::Vector3d::Zero()),
                               Vec3B(Eigen::Vector3d::Zero()), last, out));
  EXPECT_EQ(out.refusal, gnc::DisturbanceRefusal::kNonMonotonicTime);
  EXPECT_EQ(observer.estimate().eigen(), converged);

  // A gap past max_dt_s: refused, anchor re-taken, estimate held — a missed
  // cycle is not evidence the disturbance stopped.
  EXPECT_FALSE(observer.update(Vec3B(Eigen::Vector3d::Zero()), Vec3B(Eigen::Vector3d::Zero()),
                               Vec3B(Eigen::Vector3d::Zero()), last + 10 * kNsPerSecond, out));
  EXPECT_EQ(out.refusal, gnc::DisturbanceRefusal::kStepTooLong);
  EXPECT_EQ(observer.estimate().eigen(), converged);

  // Non-finite input: refused before it can poison the running estimate.
  EXPECT_FALSE(observer.update(Vec3B(Eigen::Vector3d(std::nan(""), 0.0, 0.0)),
                               Vec3B(Eigen::Vector3d::Zero()), Vec3B(Eigen::Vector3d::Zero()),
                               last + 11 * kNsPerSecond, out));
  EXPECT_EQ(out.refusal, gnc::DisturbanceRefusal::kBadInput);
  EXPECT_EQ(observer.estimate().eigen(), converged);

  // Unconfigured is inert.
  gnc::DisturbanceObserver inert{};
  EXPECT_FALSE(inert.update(Vec3B(Eigen::Vector3d::Zero()), Vec3B(Eigen::Vector3d::Zero()),
                            Vec3B(Eigen::Vector3d::Zero()), 1, out));
  EXPECT_EQ(out.refusal, gnc::DisturbanceRefusal::kUnconfigured);
}

TEST(DisturbanceObserver, IncludesTheGyroscopicTermSoARotatingVehicleIsNotAnAnomaly) {
  // Euler's equation carries omega x H, and dropping it would make any spinning
  // vehicle look like it was under a large external torque. The check: a body
  // rotating with constant total momentum in *inertial* axes sees that momentum
  // rotate in body axes at -omega x H, and the two terms must cancel to zero
  // external torque.
  gnc::DisturbanceObserver observer(observerConfig(1.0));
  const Eigen::Vector3d rate(0.0, 0.0, 0.02);
  const double h = 1.0e-3;
  gnc::DisturbanceResult out;
  double worst = 0.0;
  for (int i = 0; i <= 2000; ++i) {
    const double t = 0.1 * i;
    // H expressed in body axes for an inertially fixed momentum vector.
    const Eigen::Vector3d momentum(h * std::cos(rate.z() * t), -h * std::sin(rate.z() * t), 0.0);
    observer.update(Vec3B(momentum), Vec3B(rate), Vec3B(Eigen::Vector3d::Zero()),
                    static_cast<std::int64_t>(i) * (kNsPerSecond / 10), out);
    if (out.valid && i > 200) {
      worst = std::max(worst, out.torque_nm.eigen().norm());
    }
  }
  // Residual is the finite-difference error of a 0.02 rad/s rotation over a
  // 0.1 s step, not a torque: several orders below the anomaly budget.
  EXPECT_LT(worst, 0.01 * observerConfig().anomaly_torque_nm);
  EXPECT_FALSE(observer.anomaly());
}

}  // namespace
