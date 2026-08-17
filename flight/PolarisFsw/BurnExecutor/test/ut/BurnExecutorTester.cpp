// ======================================================================
// \title  BurnExecutorTester.cpp
// \brief  Component unit tests for BurnExecutor (design doc §23.1)
// ======================================================================

#include "BurnExecutorTester.hpp"

#include <cmath>

#include "math/quaternion.hpp"

namespace flight {

namespace {
constexpr I64 kNsPerSecond = 1000000000LL;
constexpr I64 kPeriodNs = kNsPerSecond / 10;
constexpr I64 kStartTaiNs = 1767225637000000000LL;
constexpr F64 kThrustN = 0.5;
constexpr F64 kIspS = 220.0;
constexpr F64 kMassKg = 12.0;
constexpr F64 kKnowledge = 0.03;
constexpr F64 kMaxDurationS = 600.0;
constexpr F64 kMaxAttAgeS = 0.35;
constexpr F64 kG0 = 9.80665;

QuatF64 toQuat(const polaris::math::Quaternion& q) {
  QuatF64 out;
  out[0] = q.w();
  out[1] = q.x();
  out[2] = q.y();
  out[3] = q.z();
  return out;
}

Vec3F64 toVec3(const Eigen::Vector3d& v) {
  Vec3F64 out;
  out[0] = v.x();
  out[1] = v.y();
  out[2] = v.z();
  return out;
}

Eigen::Vector3d toEigen(const Vec3F64& v) {
  return Eigen::Vector3d(v[0], v[1], v[2]);
}
}  // namespace

BurnExecutorTester ::BurnExecutorTester()
    : BurnExecutorGTestBase("Tester", MAX_HISTORY_SIZE), component("BurnExecutor") {
  this->initComponents();
  this->connectPorts();
}

BurnExecutorTester ::~BurnExecutorTester() {}

void BurnExecutorTester ::from_thrusterCmdOut_handler(FwIndexType portNum,
                                                      const ThrusterThrottleSet& cmds) {
  static_cast<void>(portNum);
  this->last_cmds_ = cmds;
}

void BurnExecutorTester ::from_accelOut_handler(FwIndexType portNum, const NonGravAccel& accel) {
  static_cast<void>(portNum);
  this->last_accel_ = accel;
  ++this->accel_count_;
}

void BurnExecutorTester ::setValidParameters() {
  this->paramSet_ThrusterCount(1, Fw::ParamValid::VALID);
  Vec3F64PerUnit axes;
  F64PerUnit thrust;
  F64PerUnit isp;
  for (U32 i = 0; i < Vec3F64PerUnit::SIZE; ++i) {
    axes[i] = 0.0;
  }
  for (U32 i = 0; i < F64PerUnit::SIZE; ++i) {
    thrust[i] = 0.0;
    isp[i] = 0.0;
  }
  axes[0] = 1.0;  // body +X
  thrust[0] = kThrustN;
  isp[0] = kIspS;
  this->paramSet_ThrusterAxesBody(axes, Fw::ParamValid::VALID);
  this->paramSet_ThrusterThrustN(thrust, Fw::ParamValid::VALID);
  this->paramSet_ThrusterIspS(isp, Fw::ParamValid::VALID);
  this->paramSet_VehicleMassKg(kMassKg, Fw::ParamValid::VALID);
  this->paramSet_ThrustKnowledgeFrac(kKnowledge, Fw::ParamValid::VALID);
  this->paramSet_MaxBurnDurationS(kMaxDurationS, Fw::ParamValid::VALID);
  this->paramSet_MaxAttitudeAgeS(kMaxAttAgeS, Fw::ParamValid::VALID);
  this->component.loadParameters();
}

void BurnExecutorTester ::setAttitude(const polaris::math::Quaternion& qBodyEci, I64 taiNs,
                                      bool valid) {
  AttitudeEstimate est;
  est.set_epochTaiNs(taiNs);
  est.set_qBodyEci(toQuat(qBodyEci));
  est.set_bodyRateRadps(toVec3(Eigen::Vector3d::Zero()));
  est.set_attitudeValid(valid);
  this->invoke_to_attitudeIn(0, est);
}

void BurnExecutorTester ::runCycleAt(I64 taiNs) {
  const U32 seconds = static_cast<U32>(taiNs / kNsPerSecond);
  const U32 useconds = static_cast<U32>((taiNs % kNsPerSecond) / 1000);
  this->setTestTime(Fw::Time(seconds, useconds));
  this->invoke_to_run(0, 0);
}

// ----------------------------------------------------------------------
// Tests
// ----------------------------------------------------------------------

void BurnExecutorTester ::testRefusesWithoutParameters() {
  this->runCycleAt(kStartTaiNs);
  ASSERT_EVENTS_ConfigInvalid_SIZE(1);
  ASSERT_EQ(this->accel_count_, 1u);
  EXPECT_FALSE(this->last_accel_.get_valid());
  EXPECT_EQ(this->last_cmds_[0], 0.0);
  this->runCycleAt(kStartTaiNs + kPeriodNs);
  ASSERT_EVENTS_ConfigInvalid_SIZE(1);  // once
  this->sendCmd_BURN_START(0, 0, 10.0, 1.0);
  ASSERT_CMD_RESPONSE(0, BurnExecutor::OPCODE_BURN_START, 0, Fw::CmdResponse::VALIDATION_ERROR);
  ASSERT_EVENTS_BurnRefused_SIZE(1);
  ASSERT_EVENTS_BurnRefused(0, BurnExecutor::BurnRefusal::UNCONFIGURED);
}

void BurnExecutorTester ::testRefusalPaths() {
  this->setValidParameters();
  // No attitude yet.
  this->sendCmd_BURN_START(0, 0, 10.0, 1.0);
  ASSERT_EVENTS_BurnRefused(0, BurnExecutor::BurnRefusal::ATTITUDE);
  this->setAttitude(polaris::math::Quaternion::Identity(), kStartTaiNs);
  this->runCycleAt(kStartTaiNs);
  this->sendCmd_BURN_START(0, 0, kMaxDurationS + 1.0, 1.0);
  ASSERT_EVENTS_BurnRefused(1, BurnExecutor::BurnRefusal::DURATION);
  this->sendCmd_BURN_START(0, 0, 0.0, 1.0);
  ASSERT_EVENTS_BurnRefused(2, BurnExecutor::BurnRefusal::DURATION);
  this->sendCmd_BURN_START(0, 0, 10.0, 1.5);
  ASSERT_EVENTS_BurnRefused(3, BurnExecutor::BurnRefusal::THROTTLE);
  this->sendCmd_BURN_START(0, 0, 10.0, 0.0);
  ASSERT_EVENTS_BurnRefused(4, BurnExecutor::BurnRefusal::THROTTLE);
  // Accepted, then a second start while burning is refused.
  this->sendCmd_BURN_START(0, 0, 10.0, 0.5);
  ASSERT_EVENTS_BurnStarted_SIZE(1);
  this->sendCmd_BURN_START(0, 0, 10.0, 0.5);
  ASSERT_EVENTS_BurnRefused(5, BurnExecutor::BurnRefusal::ALREADY_BURNING);
  ASSERT_EVENTS_BurnRefused_SIZE(6);
}

void BurnExecutorTester ::testBurnAccelerationAndDepletion() {
  this->setValidParameters();
  // q (Body <- ECI) is a +90 deg rotation about Z: the body frame is the ECI
  // frame turned +90 deg, so body +X is ECI +Y. The thrust (body +X) is ECI +Y.
  const polaris::math::Quaternion q_bi =
      polaris::math::Quaternion::FromAxisAngle(Eigen::Vector3d::UnitZ(), M_PI / 2).canonical();
  I64 t = kStartTaiNs;
  this->setAttitude(q_bi, t);
  this->runCycleAt(t);
  ASSERT_FALSE(this->last_accel_.get_valid());  // idle: explicit "no thrust"

  const F64 duration = 2.0;
  const F64 throttle = 0.5;
  this->sendCmd_BURN_START(0, 0, duration, throttle);
  ASSERT_CMD_RESPONSE(0, BurnExecutor::OPCODE_BURN_START, 0, Fw::CmdResponse::OK);
  ASSERT_EVENTS_BurnStarted_SIZE(1);

  const F64 a_mag = throttle * kThrustN / kMassKg;  // 0.020833 m/s^2
  const Eigen::Vector3d a_expected = q_bi.inverse().rotate(Eigen::Vector3d::UnitX()) * a_mag;
  EXPECT_NEAR(a_expected.x(), 0.0, 1e-12);
  EXPECT_NEAR(a_expected.y(), a_mag, 1e-12);  // the hand computation
  int burning_cycles = 0;
  for (int k = 1; k <= 25; ++k) {
    t += kPeriodNs;
    this->setAttitude(q_bi, t);
    this->runCycleAt(t);
    if (this->last_accel_.get_valid()) {
      ++burning_cycles;
      const Eigen::Vector3d a = toEigen(this->last_accel_.get_accelEciMps2());
      // Direction exact; the magnitude grows as the mass estimate depletes
      // (0.23 g/s on 12 kg: 2e-6 relative per second), which the last-cycle
      // check below pins exactly.
      EXPECT_NEAR(a.normalized().dot(a_expected.normalized()), 1.0, 1e-12) << "cycle " << k;
      EXPECT_NEAR(a.norm(), a_mag, 1e-6) << "cycle " << k;
      EXPECT_NEAR(this->last_accel_.get_sigmaMps2(), kKnowledge * a.norm(), 1e-12);
      EXPECT_EQ(this->last_cmds_[0], throttle);
      EXPECT_EQ(this->last_cmds_[1], 0.0);
    } else {
      EXPECT_EQ(this->last_cmds_[0], 0.0);
    }
  }
  // 2 s at 10 Hz = 20 burning cycles (dt accounted from the previous cycle),
  // then idle. Mass depleted at F/(Isp g0) over the burn time; delta-v a*t.
  EXPECT_EQ(burning_cycles, 20);
  ASSERT_EVENTS_BurnCompleted_SIZE(1);
  ASSERT_EVENTS_BurnAborted_SIZE(0);
  ASSERT_FALSE(this->last_accel_.get_valid());
  const F64 mdot = throttle * kThrustN / (kIspS * kG0);
  ASSERT_TLM_MassEstimateKg_SIZE(26);
  EXPECT_NEAR(this->tlmHistory_MassEstimateKg->at(25).arg, kMassKg - mdot * duration, 1e-9);
  // Delta-v is a*t on the depleting mass: within 1e-5 relative of the constant-
  // mass figure over 28 mg of 12 kg.
  EXPECT_NEAR(this->tlmHistory_BurnDeltaVMps->at(25).arg, a_mag * duration, 1e-6);
}

void BurnExecutorTester ::testAbortMidBurn() {
  this->setValidParameters();
  I64 t = kStartTaiNs;
  this->setAttitude(polaris::math::Quaternion::Identity(), t);
  this->runCycleAt(t);
  this->sendCmd_BURN_START(0, 0, 10.0, 1.0);
  t += kPeriodNs;
  this->setAttitude(polaris::math::Quaternion::Identity(), t);
  this->runCycleAt(t);
  ASSERT_TRUE(this->last_accel_.get_valid());
  EXPECT_EQ(this->last_cmds_[0], 1.0);
  this->sendCmd_BURN_ABORT(0, 0);
  ASSERT_EVENTS_BurnAborted_SIZE(1);
  t += kPeriodNs;
  this->setAttitude(polaris::math::Quaternion::Identity(), t);
  this->runCycleAt(t);
  EXPECT_FALSE(this->last_accel_.get_valid());
  EXPECT_EQ(this->last_cmds_[0], 0.0);
  ASSERT_TLM_BurnStateTlm(this->tlmHistory_BurnStateTlm->size() - 1,
                          BurnExecutor::BurnState::ABORTED);
  // A new burn clears ABORTED.
  this->sendCmd_BURN_START(0, 0, 1.0, 1.0);
  ASSERT_EVENTS_BurnStarted_SIZE(2);
}

void BurnExecutorTester ::testStaleAttitudeAbortsTheBurn() {
  this->setValidParameters();
  I64 t = kStartTaiNs;
  this->setAttitude(polaris::math::Quaternion::Identity(), t);
  this->runCycleAt(t);
  this->sendCmd_BURN_START(0, 0, 10.0, 1.0);
  // The attitude stops arriving: after MaxAttitudeAgeS the burn is aborted.
  for (int k = 0; k < 6; ++k) {
    t += kPeriodNs;
    this->runCycleAt(t);
  }
  ASSERT_EVENTS_BurnAborted_SIZE(1);
  ASSERT_EVENTS_BurnAborted(0, BurnExecutor::BurnRefusal::ATTITUDE);
  EXPECT_FALSE(this->last_accel_.get_valid());
  EXPECT_EQ(this->last_cmds_[0], 0.0);
}

void BurnExecutorTester ::testArmedBurnFiresOnItsCycle() {
  this->setValidParameters();
  this->component.commandBurnAtCycle(3, 1.0, 0.8);
  I64 t = kStartTaiNs;
  for (int k = 1; k <= 3; ++k) {
    this->setAttitude(polaris::math::Quaternion::Identity(), t);
    this->runCycleAt(t);
    if (k < 3) {
      ASSERT_EVENTS_BurnStarted_SIZE(0);
    }
    t += kPeriodNs;
  }
  ASSERT_EVENTS_BurnStarted_SIZE(1);
  ASSERT_EVENTS_BurnStarted(0, 1.0, 0.8);
}

}  // namespace flight
