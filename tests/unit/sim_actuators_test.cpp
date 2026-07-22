/// @file Unit tests for the reaction-wheel and magnetorquer truth models (§7).
///
/// The wheel tests pin the physics an attitude controller must live with: an
/// ideal wheel delivers exactly the reaction torque of its command and builds
/// momentum ∫τdt; the torque box and speed ceiling saturate; friction is
/// dissipative; power regenerates on braking; and static/dynamic imbalance emit a
/// once-per-rev disturbance sized ∝ω² — the seed of the jitter analysis this model
/// exists to feed. The magnetorquer tests pin the rod's dipole limit, linearity,
/// and the residual/hysteresis remanence, plus the fault hooks on both.

#include <gtest/gtest.h>

#include <cmath>
#include <Eigen/Core>

#include "actuators/magnetorquer.hpp"
#include "actuators/reaction_wheel.hpp"
#include "math/frames.hpp"
#include "math/typed_vector.hpp"

namespace act = polaris::sim::actuators;
namespace pm = polaris::math;
namespace pmf = polaris::math::frames;

namespace {
using Vec3B = pm::Vec3<pmf::Body>;

/// An ideal wheel: known inertia, no friction/quantization/imbalance.
act::ReactionWheelSpec idealWheel() {
  act::ReactionWheelSpec s;
  s.max_torque_nm = 1.0;
  s.max_momentum_nms = 1.0;
  s.max_speed_rad_s = 1000.0;
  s.rotor_inertia_kg_m2 = 1.0e-3;
  return s;
}
}  // namespace

// --- Reaction wheel ----------------------------------------------------------

TEST(ReactionWheel, IdealWheelDeliversCommandedReactionAndBuildsMomentum) {
  act::ReactionWheel rw(idealWheel());
  rw.commandTorque(0.01);
  double momentum = 0.0;
  const double dt = 0.01;
  for (int i = 0; i < 100; ++i) {
    const auto out = rw.step(dt);
    // Newton's third law: the body sees the negative of the motor torque.
    EXPECT_NEAR(out.reaction_torque_nm, -0.01, 1e-9) << "step " << i;
    momentum = out.momentum_nms;
  }
  // Momentum built is ∫τ dt = 0.01 N·m · 1 s = 0.01 N·m·s.
  EXPECT_NEAR(momentum, 0.01, 1e-9);
}

TEST(ReactionWheel, TorqueBoxClampsCommand) {
  act::ReactionWheel rw(idealWheel());
  rw.commandTorque(5.0);  // far beyond the 1 N·m peak
  const auto out = rw.step(0.001);
  EXPECT_NEAR(out.reaction_torque_nm, -1.0, 1e-9);
}

TEST(ReactionWheel, SaturatesAtSpeedCeilingAndDeliversLessTorque) {
  act::ReactionWheelSpec spec = idealWheel();
  spec.max_speed_rad_s = 5.0;  // low ceiling
  act::ReactionWheel rw(spec);
  rw.commandTorque(1.0);
  double last_reaction = 0.0;
  for (int i = 0; i < 100; ++i) {
    last_reaction = rw.step(0.01).reaction_torque_nm;
  }
  EXPECT_NEAR(rw.speed(), 5.0, 1e-9) << "must clamp at the ceiling";
  // Once pinned at the ceiling the wheel can no longer accelerate, so it delivers
  // essentially no reaction torque despite the full command.
  EXPECT_NEAR(last_reaction, 0.0, 1e-6);
}

TEST(ReactionWheel, FrictionDissipatesAndBrakingRegenerates) {
  act::ReactionWheelSpec spec = idealWheel();
  spec.dry_friction_nm = 1.0e-3;
  spec.motor_kt_nm_a = 0.1;  // high Kt -> low current, so mechanical regen dominates I²R
  spec.motor_resistance_ohm = 2.0;
  act::ReactionWheel rw(spec);

  // Spin up, then coast: friction alone must slow it down.
  rw.commandTorque(0.05);
  for (int i = 0; i < 200; ++i)
    rw.step(0.01);
  const double spun = rw.speed();
  EXPECT_GT(spun, 0.0);
  rw.commandTorque(0.0);
  for (int i = 0; i < 50; ++i)
    rw.step(0.01);
  EXPECT_LT(rw.speed(), spun) << "friction must decelerate a coasting wheel";

  // Braking a spinning wheel returns power to the bus (negative bus power).
  rw.commandTorque(-0.05);
  const auto braking = rw.step(0.01);
  EXPECT_LT(braking.bus_power_w, 0.0) << "regenerative braking";
}

TEST(ReactionWheel, ImbalanceEmitsSpeedSquaredOncePerRevDisturbance) {
  act::ReactionWheelSpec spec = idealWheel();
  spec.static_imbalance_kg_m = 1.0e-6;    // Us
  spec.dynamic_imbalance_kg_m2 = 2.0e-9;  // Ud
  act::ReactionWheel rw(spec);
  rw.commandTorque(0.01);
  for (int i = 0; i < 100; ++i)
    rw.step(0.01);  // spin up to a steady speed

  rw.commandTorque(0.0);  // coast at constant speed (no friction)
  const auto a = rw.step(0.001);
  const double w2 = rw.speed() * rw.speed();
  EXPECT_NEAR(a.jitter_force_n.norm(), spec.static_imbalance_kg_m * w2, 1e-15);
  EXPECT_NEAR(a.jitter_torque_nm.norm(), spec.dynamic_imbalance_kg_m2 * w2, 1e-18);
  EXPECT_EQ(a.jitter_force_n.z(), 0.0) << "disturbance is radial (spin = +z)";

  // The disturbance rotates with the rotor: its direction changes step to step.
  const auto b = rw.step(0.001);
  EXPECT_FALSE(a.jitter_force_n.isApprox(b.jitter_force_n))
      << "jitter must rotate at the wheel speed";
}

TEST(ReactionWheel, StuckAndRunawayFaults) {
  act::ReactionWheel stuck(idealWheel());
  stuck.commandTorque(0.05);
  stuck.setStuck(true);
  EXPECT_NEAR(stuck.step(0.01).reaction_torque_nm, 0.0, 1e-12) << "stuck: no drive torque";

  act::ReactionWheel runaway(idealWheel());
  runaway.commandTorque(0.0);
  runaway.setRunaway(true, +1.0);
  EXPECT_NEAR(runaway.step(0.001).reaction_torque_nm, -1.0, 1e-9) << "runaway: peak torque";
}

TEST(ReactionWheel, NonPositiveDtIsANoOp) {
  act::ReactionWheel rw(idealWheel());
  rw.commandTorque(0.05);
  const auto out = rw.step(0.0);
  EXPECT_EQ(out.reaction_torque_nm, 0.0);
  EXPECT_EQ(rw.speed(), 0.0);
}

TEST(ReactionWheel, CatalogRocketLabRw04MatchesDatasheet) {
  const auto s = act::catalog::rocketLabRw04();
  EXPECT_DOUBLE_EQ(s.max_torque_nm, 0.1);
  EXPECT_DOUBLE_EQ(s.max_momentum_nms, 0.4);
  EXPECT_GT(s.inertia(), 0.0);
}

// --- Magnetorquer ------------------------------------------------------------

TEST(Magnetorquer, IdealRodPassesTheCommandThrough) {
  act::MagnetorquerSpec spec;
  spec.max_dipole_am2 = 100.0;
  act::Magnetorquer mtq(spec);
  const Eigen::Vector3d cmd(1.0, -2.0, 3.0);
  EXPECT_TRUE(mtq.commandDipole(Vec3B(cmd)).eigen().isApprox(cmd));
}

TEST(Magnetorquer, DipoleSaturatesAtTheRatedMoment) {
  act::MagnetorquerSpec spec;
  spec.max_dipole_am2 = 30.0;
  act::Magnetorquer mtq(spec);
  const auto out = mtq.commandDipole(Vec3B(Eigen::Vector3d(100.0, -100.0, 0.0)));
  EXPECT_DOUBLE_EQ(out.eigen().x(), 30.0);
  EXPECT_DOUBLE_EQ(out.eigen().y(), -30.0);
}

TEST(Magnetorquer, LinearityScalesTheOutput) {
  act::MagnetorquerSpec spec;
  spec.max_dipole_am2 = 100.0;
  spec.linearity = 0.05;
  act::Magnetorquer mtq(spec);
  EXPECT_NEAR(mtq.commandDipole(Vec3B(Eigen::Vector3d(10.0, 0.0, 0.0))).eigen().x(), 10.5, 1e-12);
}

TEST(Magnetorquer, ResidualMomentRemainsAfterCommandingZero) {
  act::MagnetorquerSpec spec;
  spec.max_dipole_am2 = 100.0;
  spec.residual_dipole_am2 = 1.0;
  act::Magnetorquer mtq(spec);
  mtq.commandDipole(Vec3B(Eigen::Vector3d(50.0, 0.0, 0.0)));  // saturate positive
  const auto off = mtq.commandDipole(Vec3B(Eigen::Vector3d::Zero()));
  // De-energized, the core retains the remanent moment (= hysteresis half-width).
  EXPECT_NEAR(off.eigen().x(), 1.0, 1e-12);
}

TEST(Magnetorquer, HysteresisMakesTheMomentLagWithinTheLoop) {
  act::MagnetorquerSpec spec;
  spec.max_dipole_am2 = 100.0;
  spec.residual_dipole_am2 = 2.0;  // ±2 A·m² play band
  act::Magnetorquer mtq(spec);
  mtq.commandDipole(Vec3B(Eigen::Vector3d(50.0, 0.0, 0.0)));  // state pushed to 48
  // A small command move (within the 2 A·m² band of the current state ~48) sticks.
  const auto stuck = mtq.commandDipole(Vec3B(Eigen::Vector3d(49.0, 0.0, 0.0)));
  EXPECT_NEAR(stuck.eigen().x(), 48.0, 1e-12) << "output lags inside the loop";
}

TEST(Magnetorquer, StuckOnAndDropoutFaults) {
  act::MagnetorquerSpec spec;
  spec.max_dipole_am2 = 100.0;
  spec.residual_dipole_am2 = 1.0;
  act::Magnetorquer stuck(spec);
  const double last = stuck.commandDipole(Vec3B(Eigen::Vector3d(20.0, 0.0, 0.0))).eigen().x();
  stuck.setStuckOn(true);
  const auto held = stuck.commandDipole(Vec3B(Eigen::Vector3d::Zero()));
  EXPECT_NEAR(held.eigen().x(), last, 1e-12) << "stuck-on holds the last dipole";

  act::Magnetorquer dropped(spec);
  dropped.commandDipole(Vec3B(Eigen::Vector3d(20.0, 0.0, 0.0)));
  dropped.setDropout(true);
  const auto out = dropped.commandDipole(Vec3B(Eigen::Vector3d(20.0, 0.0, 0.0)));
  // No drive: only the residual remains.
  EXPECT_NEAR(out.eigen().x(), 1.0, 1e-12);
}

TEST(Magnetorquer, PowerSumsAllThreeRods) {
  act::MagnetorquerSpec spec;
  spec.max_dipole_am2 = 30.0;
  spec.power_max_w = 1.2;
  act::Magnetorquer mtq(spec);
  // One rod at full dipole draws the single-rod rating.
  mtq.commandDipole(Vec3B(Eigen::Vector3d(30.0, 0.0, 0.0)));
  EXPECT_NEAR(mtq.busPower(), 1.2, 1e-12);
  // All three at full dipole draw three times as much (I²R per winding).
  mtq.commandDipole(Vec3B(Eigen::Vector3d(30.0, 30.0, 30.0)));
  EXPECT_NEAR(mtq.busPower(), 3.6, 1e-12);
}

TEST(Magnetorquer, CatalogEntriesCarryTheDatasheetBounds) {
  const auto nss = act::catalog::nssTaurus(30.0);
  EXPECT_DOUBLE_EQ(nss.max_dipole_am2, 30.0);
  EXPECT_DOUBLE_EQ(nss.residual_dipole_am2, 1.0);  // < 1.5 A·m² bound
  EXPECT_DOUBLE_EQ(nss.linearity, 0.05);           // ±5%
  EXPECT_GT(act::catalog::genericMagnetorquer().max_dipole_am2, 0.0);
}
