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

TEST(ReactionWheel, SpeedModeReachesAndHoldsTheTarget) {
  // Ideal inner loop (no gain): the drive asks for whatever torque the box allows
  // to reach the target, then holds it. With inertia 1e-3 and a 1 N·m box it slews
  // 10 rad/s per 10 ms step, so 50 rad/s is reached in ~5 steps and held after.
  act::ReactionWheel rw(idealWheel());
  rw.commandSpeed(50.0);
  const double dt = 0.01;
  for (int i = 0; i < 20; ++i) {
    rw.step(dt);
  }
  EXPECT_NEAR(rw.speed(), 50.0, 1e-9);
  // Held: at target the reaction torque is zero (no friction to fight here).
  const auto out = rw.step(dt);
  EXPECT_NEAR(rw.speed(), 50.0, 1e-9);
  EXPECT_NEAR(out.reaction_torque_nm, 0.0, 1e-9);
}

TEST(ReactionWheel, SpeedModeHoldsAgainstFriction) {
  // With bearing friction the ideal loop feed-forwards it, so the wheel still
  // holds the exact target and delivers a steady holding torque, not droop.
  act::ReactionWheelSpec spec = idealWheel();
  spec.dry_friction_nm = 1.0e-4;
  act::ReactionWheel rw(spec);
  rw.commandSpeed(50.0);
  const double dt = 0.01;
  for (int i = 0; i < 40; ++i) {
    rw.step(dt);
  }
  EXPECT_NEAR(rw.speed(), 50.0, 1e-6);
}

TEST(ReactionWheel, FiniteBandwidthSpeedLoopHasDroop) {
  // A configured gain models a real finite-bandwidth loop: against friction it
  // settles just below the target (steady-state droop = friction / gain).
  act::ReactionWheelSpec spec = idealWheel();
  spec.dry_friction_nm = 1.0e-4;
  spec.speed_loop_gain_nm_per_rad_s = 1.0e-3;
  act::ReactionWheel rw(spec);
  rw.commandSpeed(50.0);
  const double dt = 0.01;
  for (int i = 0; i < 2000; ++i) {
    rw.step(dt);
  }
  EXPECT_LT(rw.speed(), 50.0);  // droops below target
  EXPECT_GT(rw.speed(), 49.5);  // but tracks closely
  EXPECT_NEAR(rw.speed(), 50.0 - spec.dry_friction_nm / spec.speed_loop_gain_nm_per_rad_s, 1e-3);
}

TEST(ReactionWheel, CommandingTorqueLeavesSpeedMode) {
  act::ReactionWheel rw(idealWheel());
  rw.commandSpeed(50.0);
  for (int i = 0; i < 20; ++i) {
    rw.step(0.01);
  }
  ASSERT_NEAR(rw.speed(), 50.0, 1e-9);
  // Back to torque mode: a zero torque command coasts (no speed regulation).
  rw.commandTorque(0.0);
  const auto out = rw.step(0.01);
  EXPECT_NEAR(out.reaction_torque_nm, 0.0, 1e-9);
  EXPECT_NEAR(rw.speed(), 50.0, 1e-9);  // coasts at speed, not held by a loop
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

TEST(ReactionWheelSpec, LibraryParamsConvertToSi) {
  // The hardware library speaks the datasheet's units (rpm); the spec is SI.
  // These values mirror config/hardware/reaction_wheel/rocketlab_rw04.yaml — a
  // test fixture, not a catalog (design doc §19.4). What is pinned here is the
  // conversion; the datasheet numbers themselves are pinned against the YAML in
  // tests/tools/test_config_compiler.py.
  const auto s = act::ReactionWheelSpec::fromParams({
      {"max_torque_nm", 0.1},
      {"max_momentum_nms", 0.4},
      {"max_speed_rpm", 6000.0},
      {"dry_friction_nm", 2.0e-4},
  });
  EXPECT_DOUBLE_EQ(s.max_torque_nm, 0.1);
  EXPECT_DOUBLE_EQ(s.max_momentum_nms, 0.4);
  EXPECT_NEAR(s.max_speed_rad_s, 6000.0 * 2.0 * M_PI / 60.0, 1e-12);
  // No explicit rotor inertia: it falls out of momentum / speed.
  EXPECT_NEAR(s.inertia(), 0.4 / s.max_speed_rad_s, 1e-15);
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

TEST(MagnetorquerSpec, LibraryParamsBuildTheSpecAndPowerCurve) {
  // Mirrors config/hardware/magnetorquer/aac_mtq800.yaml as a test fixture (§19.4).
  const auto mtq800 = act::MagnetorquerSpec::fromParams({
      {"max_dipole_am2", 30.0},  // boost limit
      {"residual_dipole_am2", 0.1},
      {"linearity", 0.02},  // ±2% design accuracy
      {"power_max_w", 13.2},
  });
  EXPECT_DOUBLE_EQ(mtq800.max_dipole_am2, 30.0);
  EXPECT_DOUBLE_EQ(mtq800.linearity, 0.02);
  // Peak power is the datasheet 13.2 W, and the dipole² law reproduces the low-end
  // point (~1.54 W at 10 A·m² typ). The datasheet is deliberately sub-quadratic in
  // the boost region (efficiency traded for peak moment), so the mid/high points
  // are approximate — not pinned tightly here.
  act::Magnetorquer m(mtq800);
  m.commandDipole(Vec3B(Eigen::Vector3d(30.0, 0.0, 0.0)));
  EXPECT_NEAR(m.busPower(), 13.2, 1e-9);
  m.commandDipole(Vec3B(Eigen::Vector3d(10.0, 0.0, 0.0)));
  EXPECT_NEAR(m.busPower(), 1.536, 0.15);
}

// ----------------------------------------------------------------------
// §7 MTQ/MAG duty-cycle interlock fidelity (Push 54)
// ----------------------------------------------------------------------

TEST(MagnetorquerNearField, MatchesTheAnalyticDipoleField) {
  // Jackson §5.6: B = (mu0/4 pi r^3)(3 (m.rhat) rhat - m). Checked in the two
  // geometries where the closed form is unambiguous — on the dipole axis, where
  // the field is +2 mu0 m / 4 pi r^3, and transverse to it, where it is
  // -mu0 m / 4 pi r^3.
  const Eigen::Vector3d source = Eigen::Vector3d::Zero();
  const Vec3B dipole(Eigen::Vector3d(5.0, 0.0, 0.0));  // 5 A·m² along +X
  const double r = 0.2;
  constexpr double kMu0Over4Pi = 1.0e-7;

  const Eigen::Vector3d on_axis =
      act::dipoleNearField(dipole, source, Eigen::Vector3d(r, 0.0, 0.0)).eigen();
  EXPECT_NEAR(on_axis.x(), 2.0 * kMu0Over4Pi * 5.0 / (r * r * r), 1e-15);
  EXPECT_NEAR(on_axis.y(), 0.0, 1e-18);
  EXPECT_NEAR(on_axis.z(), 0.0, 1e-18);

  const Eigen::Vector3d transverse =
      act::dipoleNearField(dipole, source, Eigen::Vector3d(0.0, r, 0.0)).eigen();
  EXPECT_NEAR(transverse.x(), -kMu0Over4Pi * 5.0 / (r * r * r), 1e-15);
  EXPECT_NEAR(transverse.y(), 0.0, 1e-18);

  // 125 uT at 20 cm from a 5 A·m² rod — several times the ~30 uT ambient, from a
  // rod at a third of the reference vehicle's rating. That is the whole reason
  // the §7 interlock exists rather than a correction model.
  EXPECT_GT(on_axis.norm(), 3.0 * 3.0e-5);

  // Co-located source and observer return zero rather than diverging: the dipole
  // approximation has no meaning inside the source.
  EXPECT_EQ(act::dipoleNearField(dipole, source, source).eigen(), Eigen::Vector3d::Zero());
}

TEST(Magnetorquer, SettleTransientDecaysFromTheDrivenMomentToTheRemanentOne) {
  act::MagnetorquerSpec spec;
  spec.max_dipole_am2 = 15.0;
  spec.residual_dipole_am2 = 0.0;  // isolate the transient from the remanence
  spec.settle_time_s = 0.01;
  act::Magnetorquer m(spec);

  m.commandDipole(Vec3B(Eigen::Vector3d(10.0, 0.0, 0.0)));
  const double driven = m.dipole().eigen().x();
  ASSERT_NEAR(driven, 10.0, 1e-12);
  m.deenergize();

  // At t = 0 the rod still carries what it was driven at; the field does not
  // vanish the instant the drive does.
  EXPECT_NEAR(m.settlingDipole(0.0).eigen().x(), driven, 1e-12);
  // tau = settle/3, so ~95% is gone at the catalog settle time...
  EXPECT_NEAR(m.settlingDipole(spec.settle_time_s).eigen().x(), driven * std::exp(-3.0), 1e-12);
  // ...and the flight quiet window, opened at three settle times
  // (MtqSettleSec = 3 x the catalog value), sees exp(-9) = 1.2e-4 of the driven
  // moment. That margin is what makes a duty-cycled magnetometer sample a
  // measurement of the geomagnetic field rather than of a torque rod.
  EXPECT_LT(m.settlingDipole(3.0 * spec.settle_time_s).eigen().x(), 2.0e-4 * driven);
  // Monotone all the way down.
  double previous = driven;
  for (int k = 1; k <= 60; ++k) {
    const double now = m.settlingDipole(0.001 * k).eigen().x();
    EXPECT_LE(now, previous + 1e-15);
    previous = now;
  }
}

TEST(Magnetorquer, TheSettleTransientsMeanCarriesTheImpulseNotItsLeadingEdge) {
  // The plant takes one held wrench per integration span, and the span covering
  // the quiet window is a whole macro period (100 ms) while the transient decays
  // in ~10 ms. Holding the value at the span's *start* would apply the full
  // driven moment for the whole period — tens of times the real post-off
  // impulse — so the torque is built from the span mean instead.
  act::MagnetorquerSpec spec;
  spec.max_dipole_am2 = 15.0;
  spec.residual_dipole_am2 = 0.0;
  spec.settle_time_s = 0.01;
  act::Magnetorquer m(spec);

  m.commandDipole(Vec3B(Eigen::Vector3d(10.0, 0.0, 0.0)));
  const double driven = m.dipole().eigen().x();
  m.deenergize();

  // The mean over the quiet window against the same integral by fine quadrature
  // — this is the statement that matters, that the impulse is right.
  const double window = 0.1;
  const int n = 100000;
  double integral = 0.0;
  for (int k = 0; k < n; ++k) {
    integral += m.settlingDipole(window * (static_cast<double>(k) + 0.5) / n).eigen().x();
  }
  const double mean = m.settlingDipoleMean(0.0, window).eigen().x();
  // The closed form is exact; the tolerance is the midpoint rule's own residual.
  EXPECT_NEAR(mean, integral / n, 1e-8);
  // ...and it is ~30x below the leading-edge value the naive hold would use.
  EXPECT_LT(mean, driven / 25.0);

  // Inside the on-window the moment is constant, so a mean must change nothing:
  // an interval starting long after the transient sees only the remanent value.
  EXPECT_NEAR(m.settlingDipoleMean(10.0 * spec.settle_time_s, 0.1).eigen().x(), 0.0, 1e-9);
  // Degenerate spans fall back to the instantaneous value rather than dividing
  // by zero.
  EXPECT_NEAR(m.settlingDipoleMean(0.0, 0.0).eigen().x(), driven, 1e-12);
}

TEST(Magnetorquer, SettleTransientDecaysTowardTheRemanentMomentNotZero) {
  // The remanence is what the core keeps indefinitely, so it is the value the
  // transient decays *toward* — folding it into the transient would model a rod
  // that demagnetises itself.
  act::MagnetorquerSpec spec;
  spec.max_dipole_am2 = 15.0;
  spec.residual_dipole_am2 = 0.5;
  spec.settle_time_s = 0.01;
  act::Magnetorquer m(spec);

  m.commandDipole(Vec3B(Eigen::Vector3d(10.0, 0.0, 0.0)));
  // The play operator's half-width is the remanence, so a rod driven from rest
  // reaches command minus residual — the driven moment is 9.5, not 10.
  const double driven = m.dipole().eigen().x();
  EXPECT_NEAR(driven, 9.5, 1e-12);
  m.deenergize();
  const double remanent = m.dipole().eigen().x();
  EXPECT_NEAR(remanent, 0.5, 1e-12);
  EXPECT_NEAR(m.settlingDipole(0.0).eigen().x(), driven, 1e-12);
  EXPECT_NEAR(m.settlingDipole(100.0 * spec.settle_time_s).eigen().x(), remanent, 1e-12);
}

TEST(Magnetorquer, StuckOnRodIgnoresDeenergiseAndKeepsItsMoment) {
  // The fault the §9 interlock monitor exists to catch: the drive is removed and
  // the moment does not go away, so the "quiet" window is not quiet.
  act::MagnetorquerSpec spec;
  spec.max_dipole_am2 = 15.0;
  spec.settle_time_s = 0.01;
  act::Magnetorquer m(spec);

  m.commandDipole(Vec3B(Eigen::Vector3d(8.0, 0.0, 0.0)));
  m.setStuckOn(true);
  m.deenergize();
  EXPECT_NEAR(m.dipole().eigen().x(), 8.0, 1e-12);
  EXPECT_NEAR(m.settlingDipole(10.0 * spec.settle_time_s).eigen().x(), 8.0, 1e-12);
}

TEST(MagnetorquerSpec, SettleTimeComesFromTheCatalog) {
  const auto spec = act::MagnetorquerSpec::fromParams({
      {"max_dipole_am2", 15.0},
      {"settle_time_s", 0.01},
  });
  EXPECT_DOUBLE_EQ(spec.settle_time_s, 0.01);
  // Absent, the transient is a step — the pre-Push-54 behaviour, which a catalog
  // entry that has not been updated must degrade to rather than to garbage.
  const auto legacy = act::MagnetorquerSpec::fromParams({{"max_dipole_am2", 15.0}});
  EXPECT_DOUBLE_EQ(legacy.settle_time_s, 0.0);
  act::Magnetorquer m(legacy);
  m.commandDipole(Vec3B(Eigen::Vector3d(5.0, 0.0, 0.0)));
  m.deenergize();
  EXPECT_NEAR(m.settlingDipole(1.0e-9).eigen().x(), 0.0, 1e-12);
}
