/// @file Unit tests for the §8.5 control laws (lib/gnc/bdot, attitude_pid,
/// rw_allocation, rw_friction). REQ-ACTL-001, REQ-ACTL-002, REQ-ACTL-010.
///
/// These pin the *math*, off-target and without an F´ topology: the component
/// tests (flight/PolarisFsw/AttitudeController/test/ut) then only have to cover
/// what the component adds. Every law here is stateful, so the tests are written
/// as short closed-loop or multi-cycle sequences rather than single calls —
/// a B-dot command formed from one sample is meaningless by construction, and a
/// PID integrator that is never filled cannot be shown to unwind.

#include <gtest/gtest.h>

#include <cmath>
#include <Eigen/Geometry>
#include <Eigen/SVD>

#include "gnc/attitude_pid.hpp"
#include "gnc/bdot.hpp"
#include "gnc/rw_allocation.hpp"
#include "gnc/rw_friction.hpp"
#include "math/frames.hpp"
#include "math/quaternion.hpp"
#include "math/typed_vector.hpp"
#include "random/rng.hpp"

namespace {

namespace pm = polaris::math;
namespace gnc = polaris::gnc;
using Body = pm::frames::Body;
using ECI = pm::frames::ECI;
using QuatBI = pm::Quat<Body, ECI>;

constexpr std::int64_t kNsPerSecond = 1000000000LL;

/// Reference-vehicle-shaped tuning. Round numbers, but the same scales the
/// config flies, so a failure here is about the law and not about units.
gnc::BdotConfig bdotConfig(double duty = 0.5) {
  gnc::BdotConfig c;
  c.gain_nms = 4.0e-3;
  c.duty_factor = duty;
  c.min_sample_dt_s = 0.05;
  c.max_sample_dt_s = 1.0;
  return c;
}

gnc::AttitudePidConfig pidConfig(double ki = 2.0e-4) {
  gnc::AttitudePidConfig c;
  c.kp_nm_per_rad = 4.4e-3;
  c.ki_nm_per_rad_s = ki;
  c.kd_nm_per_radps = 3.1e-2;
  c.max_integral_rad_s = 0.5;
  c.max_torque_nm = 0.02;
  c.max_dt_s = 0.5;
  return c;
}

/// The reference vehicle's four-wheel pyramid: spin axes on the body diagonals,
/// negated into torque-authority columns exactly as the flight component does.
gnc::RwAllocationConfig pyramidConfig(double limit = 0.025) {
  gnc::RwAllocationConfig c;
  c.wheel_count = 4;
  c.min_conditioning = 0.05;
  const double s = 1.0 / std::sqrt(3.0);
  const double signs[4][3] = {{1, 1, 1}, {-1, 1, 1}, {-1, -1, 1}, {1, -1, 1}};
  for (int i = 0; i < 4; ++i) {
    c.axes.col(i) = -Eigen::Vector3d(signs[i][0] * s, signs[i][1] * s, signs[i][2] * s);
    c.max_torque_nm[i] = limit;
  }
  return c;
}

Eigen::Vector3d deliveredTorque(const gnc::RwAllocationConfig& c,
                                const gnc::RwAllocationResult& r) {
  Eigen::Vector3d out = Eigen::Vector3d::Zero();
  for (int i = 0; i < c.wheel_count; ++i) {
    out += c.axes.col(i) * r.torque_nm[i];
  }
  return out;
}

// ======================================================================
// B-dot
// ======================================================================

TEST(Bdot, UnconfiguredControllerRefusesEveryCycle) {
  gnc::BdotController bdot;
  EXPECT_FALSE(bdot.isConfigured());
  gnc::BdotResult out;
  EXPECT_FALSE(bdot.update(pm::Vec3<Body>(Eigen::Vector3d(3.0e-5, 0, 0)), 0, out));
  EXPECT_EQ(out.refusal, gnc::BdotRefusal::kUnconfigured);
  EXPECT_EQ(out.dipole_am2.eigen(), Eigen::Vector3d::Zero());
}

TEST(Bdot, FirstSampleIsAnAnchorNotACommand) {
  gnc::BdotController bdot(bdotConfig());
  gnc::BdotResult out;
  EXPECT_FALSE(bdot.update(pm::Vec3<Body>(Eigen::Vector3d(3.0e-5, 0, 0)), 0, out));
  EXPECT_EQ(out.refusal, gnc::BdotRefusal::kNoPreviousSample);
  EXPECT_TRUE(bdot.hasPreviousSample());
}

TEST(Bdot, DerivativeUsesTheSampleTimeTagsNotTheControlPeriod) {
  // The same field change over two different intervals must give two different
  // derivatives. This is the whole reason the law takes a time tag: under the §7
  // interlock a usable sample does not arrive every control cycle, and dividing
  // by the nominal period would scale the command by whatever the real gap was.
  const Eigen::Vector3d b0(3.0e-5, 0.0, 0.0);
  const Eigen::Vector3d b1(3.0e-5, 2.0e-9, 0.0);

  gnc::BdotResult fast;
  gnc::BdotController a(bdotConfig());
  gnc::BdotResult ignored;
  ASSERT_FALSE(a.update(pm::Vec3<Body>(b0), 0, ignored));
  ASSERT_TRUE(a.update(pm::Vec3<Body>(b1), kNsPerSecond / 10, fast));  // 0.1 s

  gnc::BdotResult slow;
  gnc::BdotController b(bdotConfig());
  ASSERT_FALSE(b.update(pm::Vec3<Body>(b0), 0, ignored));
  ASSERT_TRUE(b.update(pm::Vec3<Body>(b1), kNsPerSecond / 5, slow));  // 0.2 s

  EXPECT_NEAR(fast.sample_dt_s, 0.1, 1e-12);
  EXPECT_NEAR(slow.sample_dt_s, 0.2, 1e-12);
  EXPECT_NEAR(fast.field_rate_tps.eigen().norm(), 2.0 * slow.field_rate_tps.eigen().norm(), 1e-18);
  EXPECT_NEAR(fast.dipole_am2.eigen().norm(), 2.0 * slow.dipole_am2.eigen().norm(), 1e-12);
}

TEST(Bdot, DipoleScalesInverselyWithTheDutyFactor) {
  // The rods are energised for `duty` of each period, so the *average* dipole is
  // duty x commanded. The law divides the demand by the duty factor, which is
  // what makes the average what the gain asked for — halving the duty must
  // double the commanded peak, leaving the average unchanged.
  const Eigen::Vector3d b0(3.0e-5, 0.0, 0.0);
  const Eigen::Vector3d b1(3.0e-5, 2.0e-9, 0.0);
  gnc::BdotResult ignored;

  gnc::BdotController full(bdotConfig(1.0));
  gnc::BdotResult at_full;
  ASSERT_FALSE(full.update(pm::Vec3<Body>(b0), 0, ignored));
  ASSERT_TRUE(full.update(pm::Vec3<Body>(b1), kNsPerSecond / 10, at_full));

  gnc::BdotController half(bdotConfig(0.5));
  gnc::BdotResult at_half;
  ASSERT_FALSE(half.update(pm::Vec3<Body>(b0), 0, ignored));
  ASSERT_TRUE(half.update(pm::Vec3<Body>(b1), kNsPerSecond / 10, at_half));

  EXPECT_NEAR(at_half.dipole_am2.eigen().norm(), 2.0 * at_full.dipole_am2.eigen().norm(), 1e-12);
  // The averages agree, which is the property that matters physically.
  EXPECT_NEAR(0.5 * at_half.dipole_am2.eigen().norm(), 1.0 * at_full.dipole_am2.eigen().norm(),
              1e-12);
}

TEST(Bdot, DemandOpposesTheDerivativeComponentwiseAndSurvivesClamping) {
  // The law returns an **unclamped** demand: a rated moment is a limit in the
  // rod basis, and the caller clamps there (lib/gnc/bdot.hpp). What the caller
  // relies on is asserted here — the demand opposes the field derivative
  // *componentwise*, so clamping each component independently cannot flip a
  // sign, and every term of the energy rate stays non-positive whether the rods
  // saturate or not.
  gnc::BdotController bdot(bdotConfig());
  gnc::BdotResult out;
  gnc::BdotResult ignored;
  const Eigen::Vector3d b0(3.0e-5, 0.0, 0.0);
  const Eigen::Vector3d b1(3.0e-5, 5.0e-6, -3.0e-6);  // a large, fast change
  ASSERT_FALSE(bdot.update(pm::Vec3<Body>(b0), 0, ignored));
  ASSERT_TRUE(bdot.update(pm::Vec3<Body>(b1), kNsPerSecond / 10, out));

  const Eigen::Vector3d demand = out.dipole_am2.eigen();
  const Eigen::Vector3d rate = out.field_rate_tps.eigen();
  // Big enough that a per-rod clamp really bites, so the property below is not
  // vacuously true of an unsaturated command.
  ASSERT_GT(demand.cwiseAbs().maxCoeff(), 15.0);
  for (int i = 0; i < 3; ++i) {
    EXPECT_LE(demand[i] * rate[i], 0.0);
    const double clamped = std::clamp(demand[i], -15.0, 15.0);
    EXPECT_LE(clamped * rate[i], 0.0) << "componentwise clamping flipped axis " << i;
  }
}

TEST(Bdot, StuckAndBackwardsClocksAreRefused) {
  gnc::BdotController bdot(bdotConfig());
  gnc::BdotResult out;
  ASSERT_FALSE(bdot.update(pm::Vec3<Body>(Eigen::Vector3d(3.0e-5, 0, 0)), 1000, out));
  EXPECT_FALSE(bdot.update(pm::Vec3<Body>(Eigen::Vector3d(3.0e-5, 1e-9, 0)), 1000, out));
  EXPECT_EQ(out.refusal, gnc::BdotRefusal::kNonMonotonicTime);
  EXPECT_FALSE(bdot.update(pm::Vec3<Body>(Eigen::Vector3d(3.0e-5, 1e-9, 0)), 500, out));
  EXPECT_EQ(out.refusal, gnc::BdotRefusal::kNonMonotonicTime);
}

TEST(Bdot, IntervalsOutsideTheConfiguredBandAreRefused) {
  gnc::BdotController bdot(bdotConfig());
  gnc::BdotResult out;
  ASSERT_FALSE(bdot.update(pm::Vec3<Body>(Eigen::Vector3d(3.0e-5, 0, 0)), 0, out));
  // 10 ms, below min_sample_dt_s: mostly magnetometer noise.
  EXPECT_FALSE(
      bdot.update(pm::Vec3<Body>(Eigen::Vector3d(3.0e-5, 1e-9, 0)), kNsPerSecond / 100, out));
  EXPECT_EQ(out.refusal, gnc::BdotRefusal::kIntervalTooShort);
  // The anchor is kept, so the next well-separated sample still works.
  EXPECT_TRUE(
      bdot.update(pm::Vec3<Body>(Eigen::Vector3d(3.0e-5, 1e-9, 0)), kNsPerSecond / 10, out));
  // 2 s, above max_sample_dt_s: the secant is not the tangent, so the pair is
  // dropped and the law re-acquires from the next one.
  EXPECT_FALSE(
      bdot.update(pm::Vec3<Body>(Eigen::Vector3d(3.0e-5, 2e-9, 0)), 21 * kNsPerSecond / 10, out));
  EXPECT_EQ(out.refusal, gnc::BdotRefusal::kIntervalTooLong);
}

TEST(Bdot, RateEnergyDecreasesMonotonicallyOverADetumbleSnippet) {
  RecordProperty("verifies", "REQ-ACTL-001");
  // A minimal closed loop: rigid body in a fixed inertial field, B-dot on the
  // measured body-frame field, forward Euler at the 10 Hz GNC rate. The claim
  // under test is the Lyapunov one — rotational kinetic energy strictly
  // decreases every cycle the law commands — not a settling time, which belongs
  // to the SITL rows where the plant is the real one.
  const Eigen::Vector3d inertia(0.12, 0.12, 0.10);
  const Eigen::Vector3d b_eci(2.0e-5, 1.0e-5, 1.5e-5);  // ~28 uT, fixed
  const double dt = 0.1;
  const double duty = 0.5;

  gnc::BdotController bdot(bdotConfig(duty));
  pm::Quaternion attitude = pm::Quaternion::Identity();
  Eigen::Vector3d rate(0.06, -0.04, 0.05);  // ~5 deg/s tumble

  auto energy = [&inertia](const Eigen::Vector3d& w) {
    return 0.5 * (inertia[0] * w[0] * w[0] + inertia[1] * w[1] * w[1] + inertia[2] * w[2] * w[2]);
  };

  double previous = energy(rate);
  const double initial = previous;
  int commanded_cycles = 0;
  for (int k = 0; k < 6000; ++k) {
    const Eigen::Vector3d b_body = attitude.rotate(b_eci);
    gnc::BdotResult out;
    const std::int64_t tag = static_cast<std::int64_t>(k) * (kNsPerSecond / 10);
    Eigen::Vector3d torque = Eigen::Vector3d::Zero();
    if (bdot.update(pm::Vec3<Body>(b_body), tag, out)) {
      // The average dipole over the period is duty x commanded (§7).
      torque = (duty * out.dipole_am2.eigen()).cross(b_body);
      ++commanded_cycles;
    }

    // Euler's equations, forward Euler. Coarse, but the plant is not what is
    // under test and the step is two orders below the rate time constant.
    Eigen::Vector3d h(inertia[0] * rate[0], inertia[1] * rate[1], inertia[2] * rate[2]);
    const Eigen::Vector3d rate_dot((torque[0] - (rate[1] * h[2] - rate[2] * h[1])) / inertia[0],
                                   (torque[1] - (rate[2] * h[0] - rate[0] * h[2])) / inertia[1],
                                   (torque[2] - (rate[0] * h[1] - rate[1] * h[0])) / inertia[2]);
    rate += rate_dot * dt;

    const double omega = rate.norm();
    if (omega > 0.0) {
      // Body <- ECI kinematics: the body-frame increment is a rotation of
      // |omega|*dt about omega-hat, left-multiplying (lib/gnc/coarse_attitude).
      attitude = pm::Quaternion::FromAxisAngle(rate / omega, omega * dt) * attitude;
      ASSERT_TRUE(attitude.normalize());
    }

    const double now = energy(rate);
    if (commanded_cycles > 0) {
      EXPECT_LE(now, previous + 1e-15) << "rate energy rose at cycle " << k;
    }
    previous = now;
  }

  EXPECT_GT(commanded_cycles, 5900);
  // Margin asserted, not printed. The bound is deliberately loose against what
  // this snippet achieves: the plant here is a fixed inertial field and a
  // forward-Euler integrator, so the *number* is a property of the fixture, not
  // of the vehicle — the flight-relevant time-to-rate is asserted in the SITL
  // row against the real plant (REQ-ACTL-001). What is load-bearing here is the
  // monotonicity above; this is the sanity check that it is monotone *downwards*
  // by a useful amount rather than by an epsilon.
  EXPECT_LT(previous, 0.5 * initial);
}

TEST(RateHysteresis, DeadbandStopsTheVerdictChattering) {
  gnc::RateHysteresisConfig c;
  c.enter_radps = 0.0349;
  c.exit_radps = 0.0087;
  c.confirm_cycles = 5;
  ASSERT_TRUE(c.isValid());
  gnc::RateHysteresis h(c);

  EXPECT_TRUE(h.update(0.05));  // tumbling
  EXPECT_FALSE(h.complete());
  // Four cycles under the exit threshold is not enough.
  for (int i = 0; i < 4; ++i) {
    EXPECT_TRUE(h.update(0.005));
  }
  EXPECT_FALSE(h.complete());
  EXPECT_FALSE(h.update(0.005));  // the fifth confirms
  EXPECT_TRUE(h.complete());

  // Inside the deadband the verdict is held and the streak is broken — no new
  // evidence either way.
  EXPECT_FALSE(h.update(0.02));
  EXPECT_TRUE(h.complete());
  EXPECT_EQ(h.belowStreak(), 0u);

  // Above the entry threshold it flips back in one cycle.
  EXPECT_TRUE(h.update(0.04));
  EXPECT_FALSE(h.complete());

  // A dropped rate is not evidence the vehicle stopped tumbling.
  EXPECT_TRUE(h.update(std::nan("")));
  EXPECT_FALSE(h.complete());
}

// ======================================================================
// Attitude PID
// ======================================================================

TEST(AttitudePid, ErrorRotationTakesTheShortWayRound) {
  // A 181 degree error must be driven the 179 degree way. Without the sgn(dq0)
  // factor the law unwinds the long way round (Wie 1989 §III).
  const double angle = 181.0 * M_PI / 180.0;
  const QuatBI est(pm::Quaternion::FromAxisAngle(Eigen::Vector3d::UnitZ(), angle));
  const QuatBI ref(pm::Quaternion::Identity());
  pm::Vec3<Body> error;
  double error_angle = 0.0;
  ASSERT_TRUE(gnc::attitudeError(est, ref, error, error_angle));

  EXPECT_NEAR(error_angle, (360.0 - 181.0) * M_PI / 180.0, 1e-12);
  EXPECT_LE(error_angle, M_PI);
  // The body sits 181 degrees round +Z from the reference, so the short way home
  // is a further +179 degrees about +Z, not -181. The rotation vector says so:
  // positive about +Z, with magnitude 2*sin(179/2 deg).
  EXPECT_GT(error.eigen()[2], 0.0);
  EXPECT_NEAR(error.eigen()[2], 2.0 * std::sin(0.5 * error_angle), 1e-12);
  EXPECT_NEAR(error.eigen()[0], 0.0, 1e-12);
  EXPECT_NEAR(error.eigen()[1], 0.0, 1e-12);
}

TEST(AttitudePid, ProportionalTermOpposesTheError) {
  gnc::AttitudePid pid(pidConfig());
  const double angle = 10.0 * M_PI / 180.0;
  const QuatBI est(pm::Quaternion::FromAxisAngle(Eigen::Vector3d::UnitX(), angle).canonical());
  const QuatBI ref(pm::Quaternion::Identity());

  gnc::AttitudePidResult out;
  ASSERT_TRUE(pid.update(est, pm::Vec3<Body>(Eigen::Vector3d::Zero()), ref,
                         pm::Vec3<Body>(Eigen::Vector3d::Zero()), 0.1, out));
  // The body must rotate by -angle about +X, so the torque points along -X.
  EXPECT_LT(out.torque_nm.eigen()[0], 0.0);
  EXPECT_NEAR(out.error_angle_rad, angle, 1e-12);
  EXPECT_NEAR(std::abs(out.torque_nm.eigen()[0]),
              pidConfig().kp_nm_per_rad * 2.0 * std::sin(0.5 * angle), 1e-12);
}

TEST(AttitudePid, RateErrorDampsAnInertialHold) {
  gnc::AttitudePid pid(pidConfig(0.0));  // PD
  const QuatBI at_target(pm::Quaternion::Identity());
  gnc::AttitudePidResult out;
  ASSERT_TRUE(pid.update(at_target, pm::Vec3<Body>(Eigen::Vector3d(0.01, 0.0, 0.0)), at_target,
                         pm::Vec3<Body>(Eigen::Vector3d::Zero()), 0.1, out));
  // Zero attitude error, so the command is pure rate damping.
  EXPECT_NEAR(out.torque_nm.eigen()[0], -pidConfig().kd_nm_per_radps * 0.01, 1e-15);
  EXPECT_NEAR(out.torque_nm.eigen()[1], 0.0, 1e-15);
}

TEST(AttitudePid, RegulatesAConstantDisturbanceToZeroWithTheIntegrator) {
  RecordProperty("verifies", "REQ-ACTL-002");
  // A one-axis closed loop against a constant disturbance torque. A PD
  // controller settles at a nonzero offset (torque balance); the PI term is what
  // removes it, which is the property under test.
  const double inertia = 0.12;
  const double disturbance = 2.0e-5;  // N*m, the scale of the §5.3 terms
  const double dt = 0.1;

  auto settle = [&](double ki) {
    gnc::AttitudePid pid(pidConfig(ki));
    double theta = 0.0;
    double omega = 0.0;
    for (int k = 0; k < 20000; ++k) {
      const QuatBI est(pm::Quaternion::FromAxisAngle(Eigen::Vector3d::UnitX(), theta).canonical());
      gnc::AttitudePidResult out;
      EXPECT_TRUE(pid.update(est, pm::Vec3<Body>(Eigen::Vector3d(omega, 0, 0)),
                             QuatBI(pm::Quaternion::Identity()),
                             pm::Vec3<Body>(Eigen::Vector3d::Zero()), dt, out));
      omega += (out.torque_nm.eigen()[0] + disturbance) / inertia * dt;
      theta += omega * dt;
    }
    return std::abs(theta);
  };

  const double pd_offset = settle(0.0);
  const double pi_offset = settle(2.0e-4);
  // The PD offset is the torque balance: disturbance / kp.
  EXPECT_NEAR(pd_offset, disturbance / pidConfig().kp_nm_per_rad, 1e-4);
  // The integral term removes it, with margin asserted rather than printed.
  EXPECT_LT(pi_offset, 0.05 * pd_offset);
}

TEST(AttitudePid, IntegratorIsClampedAndFrozenWhileSaturated) {
  RecordProperty("verifies", "REQ-ACTL-002");
  // A permanent 90 degree error saturates the torque command from cycle one.
  // Conditional integration must leave the integrator at zero, so there is
  // nothing to unwind when the error finally closes.
  gnc::AttitudePidConfig c = pidConfig();
  c.max_torque_nm = 1.0e-4;  // tiny, so even a small error saturates
  gnc::AttitudePid pid(c);
  const QuatBI est(pm::Quaternion::FromAxisAngle(Eigen::Vector3d::UnitY(), 0.5 * M_PI).canonical());
  const QuatBI ref(pm::Quaternion::Identity());

  for (int k = 0; k < 500; ++k) {
    gnc::AttitudePidResult out;
    ASSERT_TRUE(pid.update(est, pm::Vec3<Body>(Eigen::Vector3d::Zero()), ref,
                           pm::Vec3<Body>(Eigen::Vector3d::Zero()), 0.1, out));
    EXPECT_TRUE(out.saturated);
    EXPECT_NEAR(out.torque_nm.eigen().norm(), c.max_torque_nm, 1e-15);
  }
  EXPECT_EQ(pid.integral().eigen(), Eigen::Vector3d::Zero());

  // Unsaturated, the integrator fills — and stops at the clamp rather than
  // running away.
  gnc::AttitudePid clamped(pidConfig());
  for (int k = 0; k < 100000; ++k) {
    gnc::AttitudePidResult out;
    ASSERT_TRUE(clamped.update(
        QuatBI(pm::Quaternion::FromAxisAngle(Eigen::Vector3d::UnitX(), 0.001).canonical()),
        pm::Vec3<Body>(Eigen::Vector3d::Zero()), ref, pm::Vec3<Body>(Eigen::Vector3d::Zero()), 0.1,
        out));
  }
  // The attitude sits at +0.001 rad about +X, so the error rotation — and hence
  // the integral of it — is negative; the clamp bounds its magnitude.
  EXPECT_NEAR(clamped.integral().eigen()[0], -pidConfig().max_integral_rad_s, 1e-12);
}

TEST(AttitudePid, TorqueSaturationPreservesTheCommandedDirection) {
  gnc::AttitudePidConfig c = pidConfig();
  c.max_torque_nm = 1.0e-5;
  gnc::AttitudePid pid(c);
  // An error about a skew axis, so a componentwise clip would be visible as a
  // direction change.
  const Eigen::Vector3d axis = Eigen::Vector3d(1.0, 2.0, -3.0).normalized();
  const QuatBI est(pm::Quaternion::FromAxisAngle(axis, 0.4).canonical());
  gnc::AttitudePidResult out;
  ASSERT_TRUE(pid.update(est, pm::Vec3<Body>(Eigen::Vector3d::Zero()),
                         QuatBI(pm::Quaternion::Identity()),
                         pm::Vec3<Body>(Eigen::Vector3d::Zero()), 0.1, out));
  ASSERT_TRUE(out.saturated);
  const Eigen::Vector3d commanded = out.torque_nm.eigen().normalized();
  const Eigen::Vector3d wanted = out.attitude_error_rad.eigen().normalized();
  EXPECT_NEAR(commanded.dot(wanted), 1.0, 1e-12);
}

TEST(AttitudePid, BadInputsAreRefusedRatherThanCommanded) {
  gnc::AttitudePid pid(pidConfig());
  gnc::AttitudePidResult out;
  const QuatBI ref(pm::Quaternion::Identity());
  // A non-unit quaternion is not a rotation, so there is no error to feed back.
  EXPECT_FALSE(pid.update(QuatBI(pm::Quaternion(0.5, 0.0, 0.0, 0.0)),
                          pm::Vec3<Body>(Eigen::Vector3d::Zero()), ref,
                          pm::Vec3<Body>(Eigen::Vector3d::Zero()), 0.1, out));
  EXPECT_EQ(out.refusal, gnc::AttitudePidRefusal::kBadInput);
  EXPECT_EQ(out.torque_nm.eigen(), Eigen::Vector3d::Zero());

  EXPECT_FALSE(pid.update(ref, pm::Vec3<Body>(Eigen::Vector3d(std::nan(""), 0, 0)), ref,
                          pm::Vec3<Body>(Eigen::Vector3d::Zero()), 0.1, out));
  EXPECT_EQ(out.refusal, gnc::AttitudePidRefusal::kBadInput);
}

// ======================================================================
// Reaction-wheel allocation
// ======================================================================

TEST(RwAllocation, DegenerateArraysLeaveTheAllocatorInert) {
  gnc::RwAllocationConfig c = pyramidConfig();
  // Collapse every wheel onto one axis: no three-axis span.
  for (int i = 0; i < 4; ++i) {
    c.axes.col(i) = Eigen::Vector3d::UnitZ();
  }
  EXPECT_FALSE(c.isValid());
  gnc::RwAllocator a(c);
  EXPECT_FALSE(a.isConfigured());
  gnc::RwAllocationResult r;
  EXPECT_FALSE(a.allocate(pm::Vec3<Body>(Eigen::Vector3d(1e-3, 0, 0)),
                          gnc::RwAllocationMethod::kMinNorm, r));
  EXPECT_EQ(r.refusal, gnc::RwAllocationRefusal::kUnconfigured);
}

TEST(RwAllocation, BothMethodsDeliverTheCommandedTorqueExactly) {
  RecordProperty("verifies", "REQ-ACTL-002");
  const gnc::RwAllocationConfig c = pyramidConfig();
  const gnc::RwAllocator a(c);
  ASSERT_TRUE(a.isConfigured());

  const Eigen::Vector3d commands[] = {
      {1.0e-3, 0.0, 0.0},        {0.0, -5.0e-4, 0.0},       {0.0, 0.0, 8.0e-4},
      {1.0e-3, -5.0e-4, 8.0e-4}, {-2.0e-3, 1.5e-3, 4.0e-4}, {0.0, 0.0, 0.0},
  };
  for (const Eigen::Vector3d& tau : commands) {
    for (const auto method :
         {gnc::RwAllocationMethod::kMinNorm, gnc::RwAllocationMethod::kMinMax}) {
      gnc::RwAllocationResult r;
      ASSERT_TRUE(a.allocate(pm::Vec3<Body>(tau), method, r));
      ASSERT_FALSE(r.saturated) << "test commands must stay inside the torque box";
      const Eigen::Vector3d delivered = deliveredTorque(c, r);
      EXPECT_NEAR((delivered - tau).norm(), 0.0, 1e-15);
      EXPECT_NEAR((r.achieved_torque_nm.eigen() - tau).norm(), 0.0, 1e-15);
    }
  }
}

TEST(RwAllocation, MinMaxNeverExceedsMinNormsLargestWheelTorque) {
  RecordProperty("verifies", "REQ-ACTL-002");
  // The defining property of the L-infinity allocation, and the reason it is
  // worth the search: it saturates the array later than L2 does. alpha = 0 is in
  // the candidate set, so it can never be *worse*.
  const gnc::RwAllocationConfig c = pyramidConfig();
  const gnc::RwAllocator a(c);
  bool ever_strictly_better = false;

  for (int i = 0; i < 200; ++i) {
    // A deterministic sweep over directions and magnitudes; no RNG, so a failure
    // reproduces exactly.
    const double u = static_cast<double>(i) / 200.0;
    const Eigen::Vector3d tau =
        1.0e-3 * Eigen::Vector3d(std::cos(7.0 * u), std::sin(11.0 * u), std::cos(3.0 * u));
    gnc::RwAllocationResult l2;
    gnc::RwAllocationResult linf;
    ASSERT_TRUE(a.allocate(pm::Vec3<Body>(tau), gnc::RwAllocationMethod::kMinNorm, l2));
    ASSERT_TRUE(a.allocate(pm::Vec3<Body>(tau), gnc::RwAllocationMethod::kMinMax, linf));
    EXPECT_LE(linf.max_wheel_torque_nm, l2.max_wheel_torque_nm + 1e-15);
    // ...and still delivers the same torque.
    EXPECT_NEAR((deliveredTorque(c, linf) - tau).norm(), 0.0, 1e-15);
    if (linf.max_wheel_torque_nm < l2.max_wheel_torque_nm - 1e-12) {
      ever_strictly_better = true;
    }
  }
  // A property that only ever holds with equality would be vacuous.
  EXPECT_TRUE(ever_strictly_better);
}

TEST(RwAllocation, SaturationScalesTheWholeVectorAndKeepsTheDirection) {
  RecordProperty("verifies", "REQ-ACTL-002");
  const gnc::RwAllocationConfig c = pyramidConfig();
  const gnc::RwAllocator a(c);
  // Ten times the array's authority about a skew axis: a componentwise clip
  // would deliver a torque pointing somewhere nobody asked for.
  const Eigen::Vector3d tau = 0.5 * Eigen::Vector3d(1.0, -2.0, 0.5).normalized();

  gnc::RwAllocationResult r;
  ASSERT_TRUE(a.allocate(pm::Vec3<Body>(tau), gnc::RwAllocationMethod::kMinNorm, r));
  EXPECT_TRUE(r.saturated);
  EXPECT_GT(r.scale, 0.0);
  EXPECT_LT(r.scale, 1.0);
  const Eigen::Vector3d delivered = deliveredTorque(c, r);
  EXPECT_NEAR(delivered.normalized().dot(tau.normalized()), 1.0, 1e-12);
  EXPECT_NEAR(delivered.norm(), r.scale * tau.norm(), 1e-15);
  for (int i = 0; i < c.wheel_count; ++i) {
    EXPECT_LE(std::abs(r.torque_nm[i]), c.max_torque_nm[i] + 1e-15);
  }
}

TEST(RwAllocation, MinMaxRefusesAnArrayWhoseNullSpaceIsTooLarge) {
  // Five wheels leave a two-dimensional null space, whose exact min-max is a
  // linear program this module does not solve. Refusing is the honest answer;
  // returning the L2 result under an L-infinity label would not be.
  gnc::RwAllocationConfig c = pyramidConfig();
  c.wheel_count = 5;
  c.axes.col(4) = Eigen::Vector3d::UnitZ();
  c.max_torque_nm[4] = 0.025;
  ASSERT_TRUE(c.isValid());
  const gnc::RwAllocator a(c);
  ASSERT_TRUE(a.isConfigured());
  EXPECT_FALSE(a.supportsMinMax());

  gnc::RwAllocationResult r;
  EXPECT_FALSE(
      a.allocate(pm::Vec3<Body>(Eigen::Vector3d(1e-3, 0, 0)), gnc::RwAllocationMethod::kMinMax, r));
  EXPECT_EQ(r.refusal, gnc::RwAllocationRefusal::kMinMaxUnsupported);
  // L2 still works on the same array.
  EXPECT_TRUE(a.allocate(pm::Vec3<Body>(Eigen::Vector3d(1e-3, 0, 0)),
                         gnc::RwAllocationMethod::kMinNorm, r));
  EXPECT_NEAR((deliveredTorque(c, r) - Eigen::Vector3d(1e-3, 0, 0)).norm(), 0.0, 1e-15);
}

TEST(RwAllocation, ThreeWheelArrayHasNoNullSpaceAndBothMethodsAgree) {
  gnc::RwAllocationConfig c;
  c.wheel_count = 3;
  c.min_conditioning = 0.05;
  for (int i = 0; i < 3; ++i) {
    c.axes.col(i) = -Eigen::Vector3d::Unit(i);
    c.max_torque_nm[i] = 0.025;
  }
  ASSERT_TRUE(c.isValid());
  const gnc::RwAllocator a(c);
  ASSERT_TRUE(a.supportsMinMax());

  const Eigen::Vector3d tau(1.0e-3, -2.0e-3, 5.0e-4);
  gnc::RwAllocationResult l2;
  gnc::RwAllocationResult linf;
  ASSERT_TRUE(a.allocate(pm::Vec3<Body>(tau), gnc::RwAllocationMethod::kMinNorm, l2));
  ASSERT_TRUE(a.allocate(pm::Vec3<Body>(tau), gnc::RwAllocationMethod::kMinMax, linf));
  for (int i = 0; i < 3; ++i) {
    EXPECT_NEAR(l2.torque_nm[i], linf.torque_nm[i], 1e-15);
  }
}

TEST(RwAllocation, NonFiniteCommandsAreRefused) {
  const gnc::RwAllocator a(pyramidConfig());
  gnc::RwAllocationResult r;
  EXPECT_FALSE(a.allocate(pm::Vec3<Body>(Eigen::Vector3d(std::nan(""), 0, 0)),
                          gnc::RwAllocationMethod::kMinNorm, r));
  EXPECT_EQ(r.refusal, gnc::RwAllocationRefusal::kBadInput);
  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ(r.torque_nm[i], 0.0);
  }
}

/// Brute force for the four-wheel array: the L-infinity optimum over the 1-D
/// null space `u_p + alpha n`, found by a fine scan and a local refinement, in
/// units of each wheel's limit. Independent of the library's breakpoint search
/// so it can convict it.
double bruteForceMinMax(const gnc::RwAllocationConfig& c, const Eigen::Vector3d& tau) {
  // Pseudo-inverse and null direction recomputed here, not borrowed.
  Eigen::Matrix<double, 3, 4> a = c.axes.leftCols<4>();
  const Eigen::Matrix<double, 4, 3> pinv = a.transpose() * (a * a.transpose()).inverse();
  const Eigen::Vector4d up = pinv * tau;
  const Eigen::JacobiSVD<Eigen::Matrix<double, 3, 4>> svd(a, Eigen::ComputeFullV);
  const Eigen::Vector4d n = svd.matrixV().col(3);
  auto f = [&](double alpha) {
    double worst = 0.0;
    for (int i = 0; i < 4; ++i) {
      worst = std::max(worst, std::abs(up[i] + alpha * n[i]) / c.max_torque_nm[i]);
    }
    return worst;
  };
  double best_alpha = 0.0;
  double best = f(0.0);
  const double span = 4.0 * up.norm() / n.cwiseAbs().minCoeff();
  for (int k = -20000; k <= 20000; ++k) {
    const double alpha = span * static_cast<double>(k) / 20000.0;
    const double v = f(alpha);
    if (v < best) {
      best = v;
      best_alpha = alpha;
    }
  }
  // Ternary refinement on the convex function around the scan minimum.
  double lo = best_alpha - span / 20000.0;
  double hi = best_alpha + span / 20000.0;
  for (int it = 0; it < 200; ++it) {
    const double m1 = lo + (hi - lo) / 3.0;
    const double m2 = hi - (hi - lo) / 3.0;
    if (f(m1) < f(m2)) {
      hi = m2;
    } else {
      lo = m1;
    }
  }
  return std::min(best, f(0.5 * (lo + hi)));
}

TEST(RwAllocation, MinMaxIsTheTrueOptimumOverTheNullSpace) {
  RecordProperty("verifies", "REQ-ACTL-002");
  // The breakpoint search against an independent brute force, on random torques
  // and on a box whose limits differ per wheel — so it is the *weighted* optimum
  // (largest wheel torque in units of its own limit) that is checked.
  gnc::RwAllocationConfig c = pyramidConfig();
  const double limits[4] = {0.025, 0.010, 0.040, 0.015};
  for (int i = 0; i < 4; ++i) {
    c.max_torque_nm[i] = limits[i];
  }
  const gnc::RwAllocator a(c);
  polaris::random::SplitMix64 rng(0x1A11u);
  for (int k = 0; k < 300; ++k) {
    const Eigen::Vector3d tau =
        5.0e-3 * Eigen::Vector3d(rng.gaussian(), rng.gaussian(), rng.gaussian());
    gnc::RwAllocationResult r;
    ASSERT_TRUE(a.allocate(pm::Vec3<Body>(tau), gnc::RwAllocationMethod::kMinMax, r));
    ASSERT_FALSE(r.saturated);
    double utilisation = 0.0;
    for (int i = 0; i < 4; ++i) {
      utilisation = std::max(utilisation, std::abs(r.torque_nm[i]) / limits[i]);
    }
    EXPECT_NEAR(utilisation, bruteForceMinMax(c, tau), 1e-9) << "case " << k;
    EXPECT_NEAR((deliveredTorque(c, r) - tau).norm(), 0.0, 1e-15);
  }
}

TEST(RwAllocation, WeightedMinMaxDelaysSaturationOnAnUnequalBox) {
  RecordProperty("verifies", "REQ-ACTL-002");
  // The whole point of the L-infinity choice, stated on the box it is meant for:
  // with unequal limits the delivered fraction under min-max is never below the
  // L2 one, and is strictly above it somewhere.
  gnc::RwAllocationConfig c = pyramidConfig();
  const double limits[4] = {0.025, 0.008, 0.025, 0.025};
  for (int i = 0; i < 4; ++i) {
    c.max_torque_nm[i] = limits[i];
  }
  const gnc::RwAllocator a(c);
  bool ever_strictly_better = false;
  for (int i = 0; i < 200; ++i) {
    const double u = static_cast<double>(i) / 200.0;
    const Eigen::Vector3d tau =
        4.0e-2 * Eigen::Vector3d(std::cos(7.0 * u), std::sin(11.0 * u), std::cos(3.0 * u));
    gnc::RwAllocationResult l2;
    gnc::RwAllocationResult linf;
    ASSERT_TRUE(a.allocate(pm::Vec3<Body>(tau), gnc::RwAllocationMethod::kMinNorm, l2));
    ASSERT_TRUE(a.allocate(pm::Vec3<Body>(tau), gnc::RwAllocationMethod::kMinMax, linf));
    EXPECT_GE(linf.scale, l2.scale - 1e-15);
    if (linf.scale > l2.scale + 1e-9) {
      ever_strictly_better = true;
    }
    // Every wheel inside its own box, to rounding.
    for (int w = 0; w < 4; ++w) {
      EXPECT_LE(std::abs(linf.torque_nm[w]), limits[w] * (1.0 + 1e-12));
      EXPECT_LE(std::abs(l2.torque_nm[w]), limits[w] * (1.0 + 1e-12));
    }
    // Delivered torque is exactly scale * commanded, both methods.
    EXPECT_NEAR((deliveredTorque(c, linf) - linf.scale * tau).norm(), 0.0, 1e-15);
    EXPECT_NEAR((deliveredTorque(c, l2) - l2.scale * tau).norm(), 0.0, 1e-15);
  }
  EXPECT_TRUE(ever_strictly_better);
}

TEST(RwAllocation, EveryThreeOfFourSubsetIsExactAndTheTwoMethodsAgree) {
  RecordProperty("verifies", "REQ-ACTL-002");
  // A wheel failure hands the allocator the three survivors as a compacted
  // 3-column array: no null space, one exact solution, both methods identical,
  // for every choice of the failed wheel.
  const gnc::RwAllocationConfig full = pyramidConfig();
  const Eigen::Vector3d tau(6.0e-4, -3.0e-4, 2.0e-4);
  for (int failed = 0; failed < 4; ++failed) {
    gnc::RwAllocationConfig c = pyramidConfig();
    c.wheel_count = 3;
    int col = 0;
    for (int i = 0; i < 4; ++i) {
      if (i == failed) {
        continue;
      }
      c.axes.col(col) = full.axes.col(i);
      c.max_torque_nm[col] = full.max_torque_nm[i];
      ++col;
    }
    c.axes.col(3).setZero();
    const gnc::RwAllocator a(c);
    ASSERT_TRUE(a.isConfigured()) << "failed wheel " << failed;
    gnc::RwAllocationResult l2;
    gnc::RwAllocationResult linf;
    ASSERT_TRUE(a.allocate(pm::Vec3<Body>(tau), gnc::RwAllocationMethod::kMinNorm, l2));
    ASSERT_TRUE(a.allocate(pm::Vec3<Body>(tau), gnc::RwAllocationMethod::kMinMax, linf));
    for (int i = 0; i < 3; ++i) {
      EXPECT_NEAR(l2.torque_nm[i], linf.torque_nm[i], 1e-15);
    }
    EXPECT_EQ(l2.torque_nm[3], 0.0);
    EXPECT_NEAR((deliveredTorque(c, l2) - tau).norm(), 0.0, 1e-15);
  }
}

// ======================================================================
// Wheel-drive friction feedforward (REQ-ACTL-010; tightens REQ-ACTL-002)
// ======================================================================

/// The reference vehicle's RW-X coefficients, straight from
/// config/hardware/reaction_wheel/rwx.yaml, with the deadband and trim the
/// spacecraft config flies.
constexpr double kDryNm = 1.0e-4;
constexpr double kViscousNmS = 5.0e-6;
constexpr double kDeadbandRadps = 0.1;

gnc::RwFrictionConfig frictionConfig(double scale = 1.0, double limit = 0.025) {
  gnc::RwFrictionConfig c;
  c.wheel_count = 4;
  c.dry_friction_nm = kDryNm;
  c.viscous_friction_nm_s = kViscousNmS;
  c.deadband_radps = kDeadbandRadps;
  for (int i = 0; i < 4; ++i) {
    c.max_torque_nm[i] = limit;
    c.scale[i] = scale;
  }
  return c;
}

/// The plant's rundown model, `sim/actuators/reaction_wheel.cpp`
/// `ReactionWheel::frictionTorque` with the catalog's zero aero coefficient —
/// signed, opposing the spin. This is the truth the feedforward is inverting, so
/// it is written out here rather than reused: a test that shares the model with
/// the thing under test proves only that they agree with each other.
double truthFriction(double omega) {
  const double sgn = (omega > 0.0) - (omega < 0.0);
  return -sgn * (kDryNm + kViscousNmS * std::abs(omega));
}

TEST(RwFriction, BlendIsBoundedOddAndContinuousThroughZero) {
  const gnc::RwFrictionCompensator f(frictionConfig());
  ASSERT_TRUE(f.isConfigured());

  EXPECT_EQ(f.blend(0.0), 0.0);
  // Bounded and odd everywhere, including far outside the band.
  for (double w = -5.0; w <= 5.0; w += 0.01) {
    EXPECT_LE(std::abs(f.blend(w)), 1.0);
    EXPECT_NEAR(f.blend(w), -f.blend(-w), 1e-15);
  }
  // Saturated to the full sign outside the band — the compensation is complete
  // at every speed the vehicle actually operates the wheels at.
  EXPECT_DOUBLE_EQ(f.blend(kDeadbandRadps), 1.0);
  EXPECT_DOUBLE_EQ(f.blend(50.0), 1.0);
  EXPECT_DOUBLE_EQ(f.blend(-50.0), -1.0);

  // Continuity, stated as the property that matters: no step anywhere, and in
  // particular none at zero. A sign() feedforward would jump by 2 here.
  const double step = 1.0e-4;
  double previous = f.blend(-1.0);
  for (double w = -1.0 + step; w <= 1.0; w += step) {
    const double now = f.blend(w);
    EXPECT_LE(std::abs(now - previous), step / kDeadbandRadps + 1e-12);
    previous = now;
  }
}

TEST(RwFriction, CompensationCancelsTheRotorFrictionOutsideTheBand) {
  const gnc::RwFrictionCompensator f(frictionConfig());
  const double demand[4] = {1.0e-3, -2.0e-3, 5.0e-4, 0.0};
  const double speed[4] = {12.0, -30.0, 0.5, -0.2};
  const bool valid[4] = {true, true, true, true};

  gnc::RwFrictionResult r;
  ASSERT_TRUE(f.compensate(demand, speed, valid, r));
  ASSERT_TRUE(r.valid);
  EXPECT_FALSE(r.saturated);
  for (int i = 0; i < 4; ++i) {
    EXPECT_TRUE(r.compensated[i]);
    // The whole point: net rotor torque (command + friction) is the torque the
    // allocation asked for, so the body reaction -I*omega_dot is the commanded
    // one and the pointing loop is not fighting the bearings.
    EXPECT_NEAR(r.torque_nm[i] + truthFriction(speed[i]), demand[i], 1e-15);
  }
}

TEST(RwFriction, CompensationNeverExceedsTheModelledFriction) {
  const gnc::RwFrictionCompensator f(frictionConfig());
  const bool valid[4] = {true, true, true, true};
  // Sweep across the zero crossing and out to a wheel near its rated speed.
  for (double w = -700.0; w <= 700.0; w += 0.37) {
    const double demand[4] = {0.0, 0.0, 0.0, 0.0};
    const double speed[4] = {w, w, w, w};
    gnc::RwFrictionResult r;
    ASSERT_TRUE(f.compensate(demand, speed, valid, r));
    // At scale <= 1 the bound is unconditional: |c| <= |tau_f|, which is what
    // makes partial compensation monotonically helpful (armstrong1994 5.1).
    EXPECT_LE(std::abs(r.compensation_nm[0]), std::abs(truthFriction(w)) + 1e-18);
    // ... and it opposes the friction rather than adding to it.
    EXPECT_LE(r.compensation_nm[0] * truthFriction(w), 0.0);
  }
}

TEST(RwFriction, InsideTheDeadbandCompensationIsPartialNotChattering) {
  const gnc::RwFrictionCompensator f(frictionConfig());
  const bool valid[4] = {true, true, true, true};
  const double demand[4] = {0.0, 0.0, 0.0, 0.0};

  // A wheel dithering about zero: alternating tachometer signs at a speed far
  // below the deadband. A sign() feedforward would swing the full 2*tau_c
  // (2.0e-4 N.m) every sample; the blend keeps the swing proportional to the
  // dither, which is the trade the deadband buys.
  const double dither = 1.0e-3 * kDeadbandRadps;
  double previous = 0.0;
  double worst_swing = 0.0;
  for (int k = 0; k < 20; ++k) {
    const double w = (k % 2 == 0) ? dither : -dither;
    const double speed[4] = {w, w, w, w};
    gnc::RwFrictionResult r;
    ASSERT_TRUE(f.compensate(demand, speed, valid, r));
    if (k > 0) {
      worst_swing = std::max(worst_swing, std::abs(r.compensation_nm[0] - previous));
    }
    previous = r.compensation_nm[0];
  }
  // Three orders below the 2.0e-4 N.m a sign() feedforward would inject.
  EXPECT_LT(worst_swing, 1.0e-6);

  // The honest cost, asserted rather than only claimed: inside the band the
  // Coulomb friction is deliberately *under*-compensated, linearly in speed.
  const double half = 0.5 * kDeadbandRadps;
  const double speed[4] = {half, half, half, half};
  gnc::RwFrictionResult r;
  ASSERT_TRUE(f.compensate(demand, speed, valid, r));
  EXPECT_NEAR(r.compensation_nm[0], 0.5 * kDryNm + kViscousNmS * half, 1e-18);
  EXPECT_LT(r.compensation_nm[0], std::abs(truthFriction(half)));
}

TEST(RwFriction, SaturationTruncatesTheCompensationAndNotTheDemand) {
  const gnc::RwFrictionCompensator f(frictionConfig());
  const double limit = f.config().max_torque_nm[0];
  const bool valid[4] = {true, true, true, true};
  // Wheel 0 is commanded at its box while spinning the way that needs help;
  // wheel 1 is at its box in the direction where the compensation fits.
  const double demand[4] = {limit, -limit, 0.0, 0.0};
  const double speed[4] = {20.0, 20.0, 20.0, 20.0};

  gnc::RwFrictionResult r;
  ASSERT_TRUE(f.compensate(demand, speed, valid, r));
  EXPECT_TRUE(r.saturated);
  // The demand survives intact on every wheel; only the open-loop refinement is
  // what the box removed.
  EXPECT_DOUBLE_EQ(r.torque_nm[0], limit);
  EXPECT_EQ(r.compensation_nm[0], 0.0);
  EXPECT_NEAR(r.torque_nm[1], -limit - truthFriction(speed[1]), 1e-15);
  for (int i = 0; i < 4; ++i) {
    EXPECT_LE(std::abs(r.torque_nm[i]), limit + 1e-18);
    // Never further from the demand than the compensation asked for, and never
    // on the other side of it.
    EXPECT_LE(std::abs(r.torque_nm[i] - demand[i]), std::abs(truthFriction(speed[i])) + 1e-18);
  }
}

TEST(RwFriction, ScaleTrimsProportionallyAndPartialCompensationIsMonotone) {
  const bool valid[4] = {true, true, true, true};
  const double demand[4] = {0.0, 0.0, 0.0, 0.0};
  const double speed[4] = {40.0, 40.0, 40.0, 40.0};
  const double friction = truthFriction(speed[0]);

  double previous_residual = std::abs(friction);  // k = 0 is the uncompensated vehicle
  for (const double k : {0.25, 0.5, 0.8, 1.0}) {
    const gnc::RwFrictionCompensator f(frictionConfig(k));
    gnc::RwFrictionResult r;
    ASSERT_TRUE(f.compensate(demand, speed, valid, r));
    EXPECT_NEAR(r.compensation_nm[0], -k * friction, 1e-18);
    // What the rotor is left with: the residual shrinks with k and never
    // changes sign, which is the property that lets a flight campaign trim the
    // model up from a conservative start without ever making the vehicle worse.
    const double residual = friction + r.compensation_nm[0];
    EXPECT_LT(std::abs(residual), previous_residual);
    EXPECT_GE(residual * friction, 0.0);
    previous_residual = std::abs(residual);
  }
  // A zero trim is the documented way to disable one wheel's compensation, and
  // it must leave the demand exactly alone rather than refuse.
  const gnc::RwFrictionCompensator off(frictionConfig(0.0));
  gnc::RwFrictionResult r;
  ASSERT_TRUE(off.compensate(demand, speed, valid, r));
  EXPECT_EQ(r.compensation_nm[0], 0.0);
  EXPECT_TRUE(r.compensated[0]);
}

TEST(RwFriction, AWheelWithoutATachometerPassesItsDemandThrough) {
  const gnc::RwFrictionCompensator f(frictionConfig());
  const double demand[4] = {1.0e-3, 1.0e-3, 1.0e-3, 1.0e-3};
  const double speed[4] = {25.0, 25.0, 25.0, 25.0};
  const bool valid[4] = {true, false, true, true};

  gnc::RwFrictionResult r;
  ASSERT_TRUE(f.compensate(demand, speed, valid, r));
  // Not a refusal: the other three wheels are still compensated, and the one
  // without a speed keeps the uncompensated behaviour rather than being handed a
  // guessed sign.
  EXPECT_FALSE(r.compensated[1]);
  EXPECT_EQ(r.compensation_nm[1], 0.0);
  EXPECT_DOUBLE_EQ(r.torque_nm[1], demand[1]);
  EXPECT_TRUE(r.compensated[0]);
  // Spinning positive, so the bearings drag negative and the motor is asked for
  // more positive torque than the allocation demanded.
  EXPECT_GT(r.compensation_nm[0], 0.0);
  EXPECT_NEAR(r.compensation_nm[0], -truthFriction(speed[0]), 1e-18);
}

TEST(RwFriction, BadConfigsAndBadInputsAreRefused) {
  gnc::RwFrictionResult r;
  const double demand[4] = {0.0, 0.0, 0.0, 0.0};
  const double speed[4] = {1.0, 1.0, 1.0, 1.0};
  const bool valid[4] = {true, true, true, true};

  // A zero deadband is the discontinuous sign() the design refuses, not "no
  // blending"; an inert compensator refuses every call so the caller falls back
  // to the uncompensated demand.
  gnc::RwFrictionConfig no_band = frictionConfig();
  no_band.deadband_radps = 0.0;
  EXPECT_FALSE(no_band.isValid());
  const gnc::RwFrictionCompensator inert(no_band);
  EXPECT_FALSE(inert.isConfigured());
  EXPECT_FALSE(inert.compensate(demand, speed, valid, r));
  EXPECT_EQ(r.refusal, gnc::RwFrictionRefusal::kUnconfigured);

  gnc::RwFrictionConfig negative = frictionConfig();
  negative.scale[2] = -1.0;
  EXPECT_FALSE(negative.isValid());

  // A trim above one is *not* refused: it is a decision an operator may have
  // flight data for, and the <= 1 policy is documented rather than enforced.
  EXPECT_TRUE(frictionConfig(1.5).isValid());

  const gnc::RwFrictionCompensator f(frictionConfig());
  const double bad_demand[4] = {std::nan(""), 0.0, 0.0, 0.0};
  EXPECT_FALSE(f.compensate(bad_demand, speed, valid, r));
  EXPECT_EQ(r.refusal, gnc::RwFrictionRefusal::kBadInput);
  EXPECT_EQ(r.torque_nm[0], 0.0);

  // A speed the caller flagged usable that is not finite is a broken gate, not a
  // wheel to reason about.
  const double bad_speed[4] = {1.0, std::nan(""), 1.0, 1.0};
  EXPECT_FALSE(f.compensate(demand, bad_speed, valid, r));
  EXPECT_EQ(r.refusal, gnc::RwFrictionRefusal::kBadInput);
  // The same non-finite speed, correctly flagged unusable, is simply not read.
  const bool gated[4] = {true, false, true, true};
  EXPECT_TRUE(f.compensate(demand, bad_speed, gated, r));
  EXPECT_FALSE(r.compensated[1]);
}

}  // namespace
