// ======================================================================
// \title  AttitudeControllerTester.cpp
// \brief  Test harness for the AttitudeController component (§8.5, §7, §23.1)
// ======================================================================

#include "AttitudeControllerTester.hpp"

#include <cmath>

namespace flight {

namespace {

namespace pm = polaris::math;
using Body = pm::frames::Body;

constexpr I64 kNsPerSecond = 1000000000LL;
constexpr I64 kPeriodNs = 100000000LL;  // 10 Hz
constexpr F64 kPeriodSec = 0.1;

//! A 2026 TAI epoch, the same era the rest of the flight test suite uses.
constexpr I64 kStartTaiNs = 1'770'000'000LL * kNsPerSecond;

//! Tuning. Deliberately the reference vehicle's shape (four body-diagonal
//! wheels, an orthogonal rod triad) with round numbers, so a failure points at
//! the component rather than at a tuning subtlety.
constexpr F64 kMaxEstimateAgeSec = 0.5;
constexpr F64 kMaxAttSigmaRad = 0.02;
constexpr F64 kBdotGainNms = 4.0e-3;
constexpr F64 kBdotMaxDipoleAm2 = 15.0;
constexpr F64 kBdotMinSampleDtSec = 0.05;
constexpr F64 kBdotMaxSampleDtSec = 1.0;
constexpr F64 kDetumbleEnterRadps = 0.0349;
constexpr F64 kDetumbleExitRadps = 0.0087;
constexpr U32 kDetumbleConfirmCycles = 5;
constexpr F64 kPidKp = 4.4e-3;
constexpr F64 kPidKi = 2.0e-4;
constexpr F64 kPidKd = 3.1e-2;
constexpr F64 kPidMaxIntegral = 0.5;
constexpr F64 kPidMaxTorqueNm = 0.02;
constexpr F64 kPidMaxDtSec = 0.5;
constexpr F64 kPidMaxSlewRateRadps = 1.0;  // wide open: the component tests exercise the PID
constexpr U32 kWheelCount = 4;
constexpr F64 kWheelMaxTorqueNm = 0.025;
constexpr F64 kAllocMinConditioning = 0.05;
constexpr U32 kMtqCount = 3;
//! Wheel-drive friction feedforward (§8.5, REQ-ACTL-010) — the RW-X catalog
//! rundown coefficients and the deadband the reference vehicle flies.
constexpr F64 kWheelDryFrictionNm = 1.0e-4;
constexpr F64 kWheelViscousFrictionNmS = 5.0e-6;
constexpr F64 kFrictionDeadbandRadps = 0.1;
constexpr F64 kStuckResidualT = 8.0e-6;
constexpr F64 kWindowToleranceSec = 0.001;
constexpr U32 kStuckConfirmCycles = 5;
constexpr U32 kStuckClearCycles = 20;
constexpr U32 kAlertCycles = 100;
//! Momentum management (§8.5). RW-X rotor inertia; thresholds shaped like the
//! reference vehicle's but with a short confirmation count, so the disengage
//! edge is reachable inside a component test.
constexpr F64 kWheelInertiaKgm2 = 7.9577e-4;
constexpr F64 kMomentumEnterNms = 1.0e-3;
constexpr F64 kMomentumExitNms = 3.0e-4;
constexpr U32 kMomentumConfirmCycles = 5;
constexpr F64 kMomentumEnvelopeNms = 2.0e-3;
constexpr F64 kWheelCapacityNms = 0.030;
constexpr F64 kDesatGainPerSec = 0.2;
constexpr F64 kObserverTauSec = 200.0;
constexpr F64 kDisturbanceBudgetNm = 2.0e-5;
constexpr F64 kDisturbanceClearNm = 1.6e-5;
//! Default duty factor `setValidParameters` loads, named because the feedforward
//! check below is written on it.
constexpr F64 kDutyFactorDefault = 0.5;
constexpr U32 kDisturbanceAnomalyCycles = 100;

//! Nominal field magnitude the monitor tests compare against [T].
constexpr F64 kNominalFieldT = 3.0e-5;

Vec3F64 toVec3(const Eigen::Vector3d& v) {
  Vec3F64 out;
  out[0] = v[0];
  out[1] = v[1];
  out[2] = v[2];
  return out;
}

}  // namespace

// ----------------------------------------------------------------------
// Construction
// ----------------------------------------------------------------------

AttitudeControllerTester ::AttitudeControllerTester()
    : AttitudeControllerGTestBase("Tester", MAX_HISTORY_SIZE), component("AttitudeController") {
  this->initComponents();
  this->connectPorts();
}

AttitudeControllerTester ::~AttitudeControllerTester() {}

// ----------------------------------------------------------------------
// Captured outputs
// ----------------------------------------------------------------------

void AttitudeControllerTester ::from_wheelCmdOut_handler(FwIndexType portNum,
                                                         const WheelTorqueSet& cmds) {
  this->last_wheels_ = cmds;
  ++this->wheel_cmd_count_;
}

void AttitudeControllerTester ::from_mtqCmdOut_handler(FwIndexType portNum,
                                                       const MtqDipoleSet& cmds, F64 onWindowSec) {
  this->last_dipoles_ = cmds;
  this->last_on_window_s_ = onWindowSec;
  ++this->mtq_cmd_count_;
}

void AttitudeControllerTester ::from_mtqActuationOut_handler(FwIndexType portNum,
                                                             const MtqActuation& state) {
  this->last_schedule_ = state;
  ++this->schedule_count_;
}

// ----------------------------------------------------------------------
// Helpers
// ----------------------------------------------------------------------

void AttitudeControllerTester ::setValidParameters(F64 dutyFactor, F64 settleSec,
                                                   bool withFriction) {
  this->paramSet_ControlPeriodSec(kPeriodSec, Fw::ParamValid::VALID);
  this->paramSet_MaxEstimateAgeSec(kMaxEstimateAgeSec, Fw::ParamValid::VALID);
  this->paramSet_MaxAttSigmaRad(kMaxAttSigmaRad, Fw::ParamValid::VALID);
  this->paramSet_BdotGainNms(kBdotGainNms, Fw::ParamValid::VALID);
  this->paramSet_BdotMaxDipoleAm2(kBdotMaxDipoleAm2, Fw::ParamValid::VALID);
  this->paramSet_BdotMinSampleDtSec(kBdotMinSampleDtSec, Fw::ParamValid::VALID);
  this->paramSet_BdotMaxSampleDtSec(kBdotMaxSampleDtSec, Fw::ParamValid::VALID);
  this->paramSet_DetumbleEnterRadps(kDetumbleEnterRadps, Fw::ParamValid::VALID);
  this->paramSet_DetumbleExitRadps(kDetumbleExitRadps, Fw::ParamValid::VALID);
  this->paramSet_DetumbleConfirmCycles(kDetumbleConfirmCycles, Fw::ParamValid::VALID);
  this->paramSet_PidKpNmPerRad(kPidKp, Fw::ParamValid::VALID);
  this->paramSet_PidKiNmPerRadS(kPidKi, Fw::ParamValid::VALID);
  this->paramSet_PidKdNmPerRadps(kPidKd, Fw::ParamValid::VALID);
  this->paramSet_PidMaxIntegralRadS(kPidMaxIntegral, Fw::ParamValid::VALID);
  this->paramSet_PidMaxTorqueNm(kPidMaxTorqueNm, Fw::ParamValid::VALID);
  this->paramSet_PidMaxDtSec(kPidMaxDtSec, Fw::ParamValid::VALID);
  this->paramSet_PidMaxSlewRateRadps(kPidMaxSlewRateRadps, Fw::ParamValid::VALID);
  this->paramSet_WheelCount(kWheelCount, Fw::ParamValid::VALID);
  this->paramSet_WheelMaxTorqueNm(kWheelMaxTorqueNm, Fw::ParamValid::VALID);
  this->paramSet_AllocMinConditioning(kAllocMinConditioning, Fw::ParamValid::VALID);
  this->paramSet_AllocMethodSel(static_cast<U8>(AttitudeController::AllocMethod::MIN_MAX),
                                Fw::ParamValid::VALID);

  Vec3F64PerUnit wheel_axes;
  for (U32 i = 0; i < Vec3F64PerUnit::SIZE; ++i) {
    wheel_axes[i] = 0.0;
  }
  const double s = 1.0 / std::sqrt(3.0);
  const double signs[4][3] = {{1, 1, 1}, {-1, 1, 1}, {-1, -1, 1}, {1, -1, 1}};
  for (U32 i = 0; i < kWheelCount; ++i) {
    for (U32 k = 0; k < 3; ++k) {
      wheel_axes[3 * i + k] = signs[i][k] * s;
    }
  }
  this->paramSet_WheelAxesBody(wheel_axes, Fw::ParamValid::VALID);

  // Wheel-drive friction feedforward (§8.5). Enabled by default: it is what the
  // vehicle flies, so a test that silently ran without it would be testing a
  // different vehicle from the one the SITL rows measure.
  const Fw::ParamValid::T friction_valid =
      withFriction ? Fw::ParamValid::VALID : Fw::ParamValid::INVALID;
  this->paramSet_WheelFrictionEnable(1, friction_valid);
  this->paramSet_WheelDryFrictionNm(kWheelDryFrictionNm, friction_valid);
  this->paramSet_WheelViscousFrictionNmS(kWheelViscousFrictionNmS, friction_valid);
  this->paramSet_WheelFrictionDeadbandRadps(kFrictionDeadbandRadps, friction_valid);
  F64PerUnit friction_scale;
  for (U32 i = 0; i < F64PerUnit::SIZE; ++i) {
    friction_scale[i] = i < kWheelCount ? 1.0 : 0.0;
  }
  this->paramSet_WheelFrictionScale(friction_scale, friction_valid);
  F64PerUnit bias_pattern;
  for (U32 i = 0; i < F64PerUnit::SIZE; ++i) {
    bias_pattern[i] = 0.0;  // off: the component tests measure the loop without a bias
  }
  this->paramSet_WheelBiasNms(bias_pattern, Fw::ParamValid::VALID);
  this->paramSet_WheelBiasGainPerS(0.0, Fw::ParamValid::VALID);
  this->paramSet_WheelBiasMaxTorqueNm(0.0, Fw::ParamValid::VALID);

  Vec3F64PerUnit rod_axes;
  for (U32 i = 0; i < Vec3F64PerUnit::SIZE; ++i) {
    rod_axes[i] = 0.0;
  }
  for (U32 i = 0; i < kMtqCount; ++i) {
    rod_axes[3 * i + i] = 1.0;
  }
  this->paramSet_MtqCount(kMtqCount, Fw::ParamValid::VALID);
  this->paramSet_MtqAxesBody(rod_axes, Fw::ParamValid::VALID);
  this->paramSet_MtqDutyFactor(dutyFactor, Fw::ParamValid::VALID);
  this->paramSet_MtqSettleSec(settleSec, Fw::ParamValid::VALID);
  this->paramSet_MtqWindowToleranceSec(kWindowToleranceSec, Fw::ParamValid::VALID);
  this->paramSet_MtqStuckResidualT(kStuckResidualT, Fw::ParamValid::VALID);
  this->paramSet_MtqStuckConfirmCycles(kStuckConfirmCycles, Fw::ParamValid::VALID);
  this->paramSet_MtqStuckClearCycles(kStuckClearCycles, Fw::ParamValid::VALID);
  this->paramSet_AlertCycles(kAlertCycles, Fw::ParamValid::VALID);

  // Momentum management and disturbance feedforward (§8.5).
  this->paramSet_WheelInertiaKgm2(kWheelInertiaKgm2, Fw::ParamValid::VALID);
  this->paramSet_InertiaBodyKgm2(toVec3(Eigen::Vector3d(0.12, 0.12, 0.10)), Fw::ParamValid::VALID);
  this->paramSet_MomentumTargetBody(toVec3(Eigen::Vector3d::Zero()), Fw::ParamValid::VALID);
  this->paramSet_MomentumDesatEnterNms(kMomentumEnterNms, Fw::ParamValid::VALID);
  this->paramSet_MomentumDesatExitNms(kMomentumExitNms, Fw::ParamValid::VALID);
  this->paramSet_MomentumDesatConfirmCycles(kMomentumConfirmCycles, Fw::ParamValid::VALID);
  this->paramSet_MomentumEnvelopeNms(kMomentumEnvelopeNms, Fw::ParamValid::VALID);
  this->paramSet_WheelCapacityNms(kWheelCapacityNms, Fw::ParamValid::VALID);
  this->paramSet_DesatGainPerSec(kDesatGainPerSec, Fw::ParamValid::VALID);
  this->paramSet_FeedforwardModelEnable(1, Fw::ParamValid::VALID);
  this->paramSet_FeedforwardObserverEnable(1, Fw::ParamValid::VALID);
  this->paramSet_ObserverTauSec(kObserverTauSec, Fw::ParamValid::VALID);
  this->paramSet_ResidualDipoleAm2(toVec3(Eigen::Vector3d(0.002, -0.001, 0.0015)),
                                   Fw::ParamValid::VALID);
  this->paramSet_DisturbanceBudgetNm(kDisturbanceBudgetNm, Fw::ParamValid::VALID);
  this->paramSet_DisturbanceClearNm(kDisturbanceClearNm, Fw::ParamValid::VALID);
  this->paramSet_DisturbanceAnomalyCycles(kDisturbanceAnomalyCycles, Fw::ParamValid::VALID);
  // paramSet_* only stages values in the harness's table; the component's base
  // caches them at load, exactly as the topology does once ParameterDb is up.
  this->component.loadParameters();
}

void AttitudeControllerTester ::setEstimate(const pm::Quaternion& q, const Eigen::Vector3d& rate,
                                            double sigmaRad, I64 epochNs) {
  QuatF64 quat;
  quat[0] = q.w();
  quat[1] = q.x();
  quat[2] = q.y();
  quat[3] = q.z();
  this->estimate_.set_epochTaiNs(epochNs);
  this->estimate_.set_qBodyEci(quat);
  this->estimate_.set_bodyRateRadps(toVec3(rate));
  this->estimate_.set_attCovDiagRad2(
      toVec3(Eigen::Vector3d(sigmaRad * sigmaRad, sigmaRad * sigmaRad, sigmaRad * sigmaRad)));
  this->estimate_.set_ageSec(0.0);
  this->estimate_.set_mode(EstimationMode::FINE);
  this->estimate_.set_attitudeValid(true);
  this->estimate_.set_rateValid(true);
}

void AttitudeControllerTester ::setMagnetic(const Eigen::Vector3d& fieldT, I64 tagNs,
                                            double modelMagnitudeT, bool fieldValid,
                                            bool modelValid) {
  this->estimate_.set_magFieldBody(toVec3(fieldT));
  this->estimate_.set_magFieldTimeTagNs(tagNs);
  this->estimate_.set_magModelMagnitudeT(modelMagnitudeT);
  this->estimate_.set_magFieldValid(fieldValid);
  this->estimate_.set_magModelValid(modelValid);
  // The §9 stuck-on monitor reads the **raw** magnitude, not the voted field: on
  // a real vehicle the estimator's plausibility band would have rejected a
  // rod-scale disturbance before the vote. The harness therefore has to supply
  // both, and here they agree — the fault-sized cases are what the SITL rows
  // cover, where the two genuinely diverge.
  this->estimate_.set_magRawMagnitudeT(fieldT.norm());
  this->estimate_.set_magRawValid(fieldValid);
}

void AttitudeControllerTester ::clearMagnetic() {
  this->estimate_.set_magFieldValid(false);
  this->estimate_.set_magModelValid(false);
  this->estimate_.set_magRawValid(false);
}

double AttitudeControllerTester ::speedForMomentum(double momentumNms) {
  // Four wheels at a common speed on the body diagonals: the X and Y
  // contributions cancel and h_z = 4 * I * w / sqrt(3).
  return momentumNms * std::sqrt(3.0) / (4.0 * kWheelInertiaKgm2);
}

void AttitudeControllerTester ::setWheelSpeeds(double speedRadps, bool valid) {
  for (U32 i = 0; i < kWheelCount; ++i) {
    this->wheel_speed_radps_[i] = speedRadps;
    this->wheel_speed_valid_[i] = valid;
  }
}

void AttitudeControllerTester ::runCycleAt(I64 taiNs) {
  const U32 seconds = static_cast<U32>(taiNs / kNsPerSecond);
  const U32 useconds = static_cast<U32>((taiNs % kNsPerSecond) / 1000);
  this->setTestTime(Fw::Time(seconds, useconds));
  this->invoke_to_estimateIn(0, this->estimate_);
  for (U32 i = 0; i < kWheelCount; ++i) {
    WheelSpeedMeas meas;
    meas.set_speedRadps(this->wheel_speed_radps_[i]);
    meas.set_timeTagNs(taiNs);
    meas.set_valid(this->wheel_speed_valid_[i]);
    this->invoke_to_wheelSpeedIn(static_cast<FwIndexType>(i), meas);
  }
  this->invoke_to_run(0, 0);
}

// ----------------------------------------------------------------------
// Tests
// ----------------------------------------------------------------------

void AttitudeControllerTester ::testRefusesWithoutParameters() {
  // No paramSet_* at all: every read comes back INVALID.
  this->setEstimate(pm::Quaternion::Identity(), Eigen::Vector3d::Zero(), 1.0e-4, kStartTaiNs);
  this->runCycleAt(kStartTaiNs);

  ASSERT_EVENTS_ConfigInvalid_SIZE(1);
  ASSERT_TLM_CtrlModeTlm_SIZE(1);
  ASSERT_TLM_CtrlModeTlm(0, AttitudeController::CtrlMode::IDLE);

  // Zero on every actuator — not silence. An actuator nobody re-commands keeps
  // driving whatever it was last told.
  ASSERT_EQ(this->wheel_cmd_count_, 1u);
  ASSERT_EQ(this->mtq_cmd_count_, 1u);
  for (U32 i = 0; i < WheelTorqueSet::SIZE; ++i) {
    EXPECT_EQ(this->last_wheels_[i], 0.0);
  }
  for (U32 i = 0; i < MtqDipoleSet::SIZE; ++i) {
    for (U32 k = 0; k < 3; ++k) {
      EXPECT_EQ(this->last_dipoles_[i][k], 0.0);
    }
  }
  EXPECT_EQ(this->last_on_window_s_, 0.0);

  // The alert is edge-gated: a second inert cycle costs no second event.
  this->runCycleAt(kStartTaiNs + kPeriodNs);
  ASSERT_EVENTS_ConfigInvalid_SIZE(1);

  // POINT is refused outright while unconfigured.
  this->sendCmd_CTRL_MODE_SET(0, 0, AttitudeController::CtrlMode::POINT);
  ASSERT_EVENTS_ModeRefused_SIZE(1);
  ASSERT_EVENTS_ModeRefused(0, AttitudeController::CtrlMode::POINT,
                            AttitudeController::CtrlRefusal::NOT_CONFIGURED);
}

void AttitudeControllerTester ::testPointRefusalPaths() {
  this->setValidParameters();

  // (1) No estimate has ever arrived. The command runs before any cycle, so the
  // controller has nothing to judge.
  this->sendCmd_CTRL_MODE_SET(0, 0, AttitudeController::CtrlMode::POINT);
  ASSERT_EVENTS_ModeRefused_SIZE(1);
  ASSERT_EVENTS_ModeRefused(0, AttitudeController::CtrlMode::POINT,
                            AttitudeController::CtrlRefusal::NOT_CONFIGURED);

  // Configure the component by running one cycle, then supply a coarse-quality
  // estimate: valid, but well above the quality floor.
  this->setEstimate(pm::Quaternion::Identity(), Eigen::Vector3d::Zero(), 0.05, kStartTaiNs);
  this->runCycleAt(kStartTaiNs);
  this->clearHistory();

  // (2) Quality floor.
  this->sendCmd_CTRL_MODE_SET(0, 0, AttitudeController::CtrlMode::POINT);
  ASSERT_EVENTS_ModeRefused_SIZE(1);
  ASSERT_EVENTS_ModeRefused(0, AttitudeController::CtrlMode::POINT,
                            AttitudeController::CtrlRefusal::QUALITY_FLOOR);
  this->clearHistory();

  // (3) Fine quality but no target.
  this->setEstimate(pm::Quaternion::Identity(), Eigen::Vector3d::Zero(), 1.0e-4,
                    kStartTaiNs + kPeriodNs);
  this->runCycleAt(kStartTaiNs + kPeriodNs);
  this->clearHistory();
  this->sendCmd_CTRL_MODE_SET(0, 0, AttitudeController::CtrlMode::POINT);
  ASSERT_EVENTS_ModeRefused_SIZE(1);
  ASSERT_EVENTS_ModeRefused(0, AttitudeController::CtrlMode::POINT,
                            AttitudeController::CtrlRefusal::NO_TARGET);
  this->clearHistory();

  // (4) A non-finite target is rejected and the previous (absent) one stands.
  this->sendCmd_CTRL_SET_TARGET_Q(0, 0, 0.0, 0.0, 0.0, 0.0);
  ASSERT_EVENTS_TargetRejected_SIZE(1);
  this->clearHistory();

  // (5) With a target and a fine estimate, POINT engages.
  this->sendCmd_CTRL_SET_TARGET_Q(0, 0, 1.0, 0.0, 0.0, 0.0);
  this->sendCmd_CTRL_MODE_SET(0, 0, AttitudeController::CtrlMode::POINT);
  ASSERT_EVENTS_ModeRefused_SIZE(0);
  ASSERT_EVENTS_ModeChanged_SIZE(1);
  ASSERT_EVENTS_ModeChanged(0, AttitudeController::CtrlMode::IDLE,
                            AttitudeController::CtrlMode::POINT);

  // (6) A stale estimate refuses the *cycle*, and the refusal commands zero
  // rather than holding the last torque.
  this->clearHistory();
  this->runCycleAt(kStartTaiNs + 100 * kPeriodNs);  // estimate now 10 s old
  ASSERT_EVENTS_ControlRefused_SIZE(1);
  for (U32 i = 0; i < WheelTorqueSet::SIZE; ++i) {
    EXPECT_EQ(this->last_wheels_[i], 0.0);
  }

  // (7) IDLE is accepted unconditionally, even with the estimate stale.
  this->clearHistory();
  this->sendCmd_CTRL_MODE_SET(0, 0, AttitudeController::CtrlMode::IDLE);
  ASSERT_EVENTS_ModeRefused_SIZE(0);
  ASSERT_EVENTS_ModeChanged_SIZE(1);
}

void AttitudeControllerTester ::testSaturationIsCountedAndReportedUnclipped() {
  this->setValidParameters();
  // Squeeze the torque limit rather than inflate the demand. The alternatives
  // both change what is being tested: a rate large enough to saturate the kd
  // term is a rate that demotes POINT to DETUMBLE, and the largest attainable
  // angle error (pi) still falls short of the flown limit against the flown kp.
  // Saturation is a demand/limit comparison, so moving either side of it is the
  // same experiment.
  constexpr F64 kTightLimitNm = 1.0e-4;
  this->paramSet_PidMaxTorqueNm(kTightLimitNm, Fw::ParamValid::VALID);
  this->paramSet_AlertCycles(1, Fw::ParamValid::VALID);  // every cycle, not one in 100
  this->component.loadParameters();

  const double angle = 10.0 * M_PI / 180.0;
  const pm::Quaternion attitude =
      pm::Quaternion::FromAxisAngle(Eigen::Vector3d::UnitX(), angle).canonical();
  this->setEstimate(attitude, Eigen::Vector3d::Zero(), 1.0e-4, kStartTaiNs);
  this->runCycleAt(kStartTaiNs);
  this->sendCmd_CTRL_SET_TARGET_Q(0, 0, 1.0, 0.0, 0.0, 0.0);
  this->sendCmd_CTRL_MODE_SET(0, 0, AttitudeController::CtrlMode::POINT);
  this->clearHistory();

  // Three saturated cycles. The estimate is re-stamped each time and the
  // attitude held, so every cycle sees the same over-demand.
  for (U32 i = 1; i <= 3; ++i) {
    const I64 when = kStartTaiNs + static_cast<I64>(i) * kPeriodNs;
    this->setEstimate(attitude, Eigen::Vector3d::Zero(), 1.0e-4, when);
    this->runCycleAt(when);
  }

  // The count rises once per saturated cycle. This is the half the event stream
  // cannot carry: the alert is cadence-throttled, so sustained saturation --
  // exactly the case worth knowing about -- is the case the throttle hides.
  ASSERT_TLM_CyclesSaturated_SIZE(3);
  for (U32 i = 0; i < 3; ++i) {
    ASSERT_TLM_CyclesSaturated(i, i + 1);
  }

  // And the event reports the demand *before* the clip. Reporting the command
  // instead -- which is what this used to do -- makes demandNm equal limitNm by
  // construction on every saturated cycle, so the operator learns that the limit
  // was reached and nothing about how far over the vehicle was asked to go.
  ASSERT_EVENTS_TorqueSaturated_SIZE(3);
  const F64 reported = this->eventHistory_TorqueSaturated->at(0).demandNm;
  EXPECT_GT(reported, kTightLimitNm);
  EXPECT_NEAR(reported, kPidKp * 2.0 * std::sin(0.5 * angle), 1.0e-12);

  // A mode change zeroes the count *on the downlink*, not merely in memory:
  // only the POINT path writes this channel, so a reset that stayed internal
  // would leave the previous mode's total standing for the whole of a DETUMBLE.
  this->clearHistory();
  this->sendCmd_CTRL_MODE_SET(0, 0, AttitudeController::CtrlMode::IDLE);
  ASSERT_TLM_CyclesSaturated_SIZE(1);
  ASSERT_TLM_CyclesSaturated(0, 0);
}

void AttitudeControllerTester ::testPointEngagesAndReducesError() {
  this->setValidParameters();

  // A 10 degree error about body +X, at rest. The target is identity.
  const double angle = 10.0 * M_PI / 180.0;
  const pm::Quaternion attitude =
      pm::Quaternion::FromAxisAngle(Eigen::Vector3d::UnitX(), angle).canonical();
  this->setEstimate(attitude, Eigen::Vector3d::Zero(), 1.0e-4, kStartTaiNs);
  this->runCycleAt(kStartTaiNs);

  this->sendCmd_CTRL_SET_TARGET_Q(0, 0, 1.0, 0.0, 0.0, 0.0);
  this->sendCmd_CTRL_MODE_SET(0, 0, AttitudeController::CtrlMode::POINT);
  this->clearHistory();

  this->setEstimate(attitude, Eigen::Vector3d::Zero(), 1.0e-4, kStartTaiNs + kPeriodNs);
  this->runCycleAt(kStartTaiNs + kPeriodNs);

  // The commanded torque opposes the error: the body must rotate by -angle about
  // +X to reach identity, so the torque points along -X.
  ASSERT_TLM_TorqueCmd_SIZE(1);
  Vec3F64 torque;
  torque = this->tlmHistory_TorqueCmd->at(0).arg;
  EXPECT_LT(torque[0], 0.0);
  EXPECT_NEAR(torque[1], 0.0, 1.0e-12);
  EXPECT_NEAR(torque[2], 0.0, 1.0e-12);
  // kp * |dtheta|, where the error rotation vector is 2*vec(dq) = 2*sin(theta/2)
  // — the exact quantity the law feeds back, not its small-angle limit theta.
  // At 10 degrees the two differ by 0.13%, which is why the expectation is
  // written on the former.
  EXPECT_NEAR(std::abs(torque[0]), kPidKp * 2.0 * std::sin(0.5 * angle), 1.0e-12);

  // The allocation reproduces it: A u = tau with A the negated spin axes.
  const double s = 1.0 / std::sqrt(3.0);
  const double signs[4][3] = {{1, 1, 1}, {-1, 1, 1}, {-1, -1, 1}, {1, -1, 1}};
  Eigen::Vector3d delivered = Eigen::Vector3d::Zero();
  for (U32 i = 0; i < kWheelCount; ++i) {
    const Eigen::Vector3d axis(-signs[i][0] * s, -signs[i][1] * s, -signs[i][2] * s);
    delivered += axis * this->last_wheels_[i];
  }
  EXPECT_NEAR(delivered.x(), torque[0], 1.0e-12);
  EXPECT_NEAR(delivered.y(), torque[1], 1.0e-12);
  EXPECT_NEAR(delivered.z(), torque[2], 1.0e-12);

  // POINT drives wheels, not rods: the schedule leaves the whole period quiet.
  EXPECT_EQ(this->last_on_window_s_, 0.0);
  EXPECT_EQ(this->last_schedule_.get_commandedMask(), 0u);
}

void AttitudeControllerTester ::testDetumbleCommandsOpposingDipole() {
  this->setValidParameters();

  // A field walking along +Y in body axes: successive quiet-window samples one
  // control period apart give a clean derivative along +Y. The step is small
  // enough that the demanded dipole stays inside the per-rod limit, so this test
  // measures the law rather than the clamp.
  const double step_t = 2.0e-9;
  const Eigen::Vector3d rate(0.05, 0.0, 0.0);

  auto fieldAt = [&](int k) {
    return Eigen::Vector3d(kNominalFieldT, step_t * static_cast<double>(k), 0.0);
  };

  // Cycle 0 enters DETUMBLE; the law sees its first sample on cycle 1 and can
  // only difference from cycle 2 — the first sample is an anchor, not a command.
  this->setEstimate(pm::Quaternion::Identity(), rate, 0.05, kStartTaiNs);
  this->setMagnetic(fieldAt(0), kStartTaiNs, fieldAt(0).norm());
  this->runCycleAt(kStartTaiNs);
  this->sendCmd_CTRL_MODE_SET(0, 0, AttitudeController::CtrlMode::DETUMBLE);

  const I64 t1 = kStartTaiNs + kPeriodNs;
  this->setEstimate(pm::Quaternion::Identity(), rate, 0.05, t1);
  this->setMagnetic(fieldAt(1), t1, fieldAt(1).norm());
  this->runCycleAt(t1);
  // The anchor cycle commands nothing, and says so rather than guessing.
  EXPECT_EQ(this->last_on_window_s_, 0.0);

  this->clearHistory();
  const I64 t2 = t1 + kPeriodNs;
  this->setEstimate(pm::Quaternion::Identity(), rate, 0.05, t2);
  this->setMagnetic(fieldAt(2), t2, fieldAt(2).norm());
  this->runCycleAt(t2);

  // dB/dt is +Y, so the dipole is -Y: rod 1 carries it, rods 0 and 2 do not.
  EXPECT_LT(this->last_dipoles_[1][1], 0.0);
  EXPECT_NEAR(this->last_dipoles_[0][0], 0.0, 1.0e-15);
  EXPECT_NEAR(this->last_dipoles_[2][2], 0.0, 1.0e-15);
  // Magnitude: k * |dB/dt| / (|B|^2 * duty) — the duty division is what makes
  // the *average* dipole over the period the one the gain asked for.
  const double dbdt = step_t / kPeriodSec;
  const double expected = kBdotGainNms * dbdt / (fieldAt(2).squaredNorm() * 0.5);
  ASSERT_LT(expected, kBdotMaxDipoleAm2) << "test field step saturates the rods";
  EXPECT_NEAR(std::abs(this->last_dipoles_[1][1]), expected, 1.0e-12);

  // The rods are driven, so the schedule opens an on-window and the rod that
  // carried the command is the only candidate the interlock names.
  EXPECT_NEAR(this->last_on_window_s_, 0.5 * kPeriodSec, 1.0e-12);
  EXPECT_EQ(this->last_schedule_.get_commandedMask(), 0x2u);
}

void AttitudeControllerTester ::testDutyCycleScheduleInvariants() {
  const F64 duty = 0.4;
  const F64 settle = 0.02;
  this->setValidParameters(duty, settle);

  const double step_t = 2.0e-9;
  const Eigen::Vector3d rate(0.05, 0.0, 0.0);
  auto fieldAt = [&](int k) {
    return Eigen::Vector3d(kNominalFieldT, step_t * static_cast<double>(k), 0.0);
  };

  this->setEstimate(pm::Quaternion::Identity(), rate, 0.05, kStartTaiNs);
  this->setMagnetic(fieldAt(0), kStartTaiNs, fieldAt(0).norm());
  this->runCycleAt(kStartTaiNs);
  this->sendCmd_CTRL_MODE_SET(0, 0, AttitudeController::CtrlMode::DETUMBLE);

  const I64 t1 = kStartTaiNs + kPeriodNs;
  this->setEstimate(pm::Quaternion::Identity(), rate, 0.05, t1);
  this->setMagnetic(fieldAt(1), t1, fieldAt(1).norm());
  this->runCycleAt(t1);

  const I64 t2 = t1 + kPeriodNs;
  this->setEstimate(pm::Quaternion::Identity(), rate, 0.05, t2);
  this->setMagnetic(fieldAt(2), t2, fieldAt(2).norm());
  this->runCycleAt(t2);

  const MtqActuation& s = this->last_schedule_;
  EXPECT_EQ(s.get_periodStartTaiNs(), t2);
  EXPECT_EQ(s.get_onWindowEndTaiNs(), t2 + static_cast<I64>(duty * kPeriodSec * 1.0e9));
  EXPECT_EQ(s.get_quietStartTaiNs(), s.get_onWindowEndTaiNs() + static_cast<I64>(settle * 1.0e9));
  // The late-sample tolerance extends the window's **end** and nothing else: the
  // rods are off from the on-window's end until the next period's on-window, so
  // a late sample is still quiet, while a tolerance on the *start* would admit a
  // dirty one.
  EXPECT_EQ(s.get_quietEndTaiNs(), t2 + kPeriodNs + static_cast<I64>(kWindowToleranceSec * 1.0e9));
  EXPECT_EQ(s.get_quietStartTaiNs(), s.get_onWindowEndTaiNs() + static_cast<I64>(settle * 1.0e9));
  EXPECT_TRUE(s.get_interlockHealthy());
  // **The invariant that matters:** the on-window and the quiet window are
  // disjoint, and no sample time-tagged inside the on-window can ever be
  // admitted. This is what the estimator's gate relies on.
  EXPECT_GT(s.get_quietStartTaiNs(), s.get_onWindowEndTaiNs());
  EXPECT_GT(s.get_quietEndTaiNs(), s.get_quietStartTaiNs());

  // A cycle that commands nothing leaves the whole period quiet, which is what
  // lets the stuck-on monitor see an undisturbed field.
  this->sendCmd_CTRL_MODE_SET(0, 0, AttitudeController::CtrlMode::IDLE);
  const I64 t3 = t2 + kPeriodNs;
  this->setEstimate(pm::Quaternion::Identity(), rate, 0.05, t3);
  this->runCycleAt(t3);
  EXPECT_EQ(this->last_schedule_.get_onWindowEndTaiNs(), t3);
  EXPECT_EQ(this->last_schedule_.get_quietStartTaiNs(), t3);
  EXPECT_EQ(this->last_schedule_.get_quietEndTaiNs(),
            t3 + kPeriodNs + static_cast<I64>(kWindowToleranceSec * 1.0e9));
  EXPECT_EQ(this->last_on_window_s_, 0.0);
}

void AttitudeControllerTester ::testStuckOnMonitorLatchesAndClears() {
  this->setValidParameters();

  // Rods commanded off (IDLE) but the field magnitude is 50 uT against a 30 uT
  // model: a disturbance nothing on this vehicle asked for.
  const Eigen::Vector3d disturbed(5.0e-5, 0.0, 0.0);
  I64 t = kStartTaiNs;
  for (U32 i = 0; i < kStuckConfirmCycles; ++i) {
    this->setEstimate(pm::Quaternion::Identity(), Eigen::Vector3d::Zero(), 1.0e-4, t);
    this->setMagnetic(disturbed, t, kNominalFieldT);
    this->runCycleAt(t);
    t += kPeriodNs;
  }
  ASSERT_EVENTS_MtqStuckOn_SIZE(1);
  EXPECT_FALSE(this->last_schedule_.get_interlockHealthy());

  // One more disturbed cycle costs no second event: the latch is an edge.
  this->setEstimate(pm::Quaternion::Identity(), Eigen::Vector3d::Zero(), 1.0e-4, t);
  this->setMagnetic(disturbed, t, kNominalFieldT);
  this->runCycleAt(t);
  t += kPeriodNs;
  ASSERT_EVENTS_MtqStuckOn_SIZE(1);

  // Re-admission on the criterion that excluded it: the same residual test back
  // under the same threshold for the clear count.
  const Eigen::Vector3d clean(kNominalFieldT, 0.0, 0.0);
  for (U32 i = 0; i < kStuckClearCycles; ++i) {
    this->setEstimate(pm::Quaternion::Identity(), Eigen::Vector3d::Zero(), 1.0e-4, t);
    this->setMagnetic(clean, t, kNominalFieldT);
    this->runCycleAt(t);
    t += kPeriodNs;
  }
  ASSERT_EVENTS_MtqStuckCleared_SIZE(1);
  EXPECT_TRUE(this->last_schedule_.get_interlockHealthy());
}

void AttitudeControllerTester ::testStuckOnAttribution() {
  this->setValidParameters();

  // Detumble with a derivative along +Y only: exactly one rod carries a command,
  // so the candidate set is a singleton and the attribution is decisive.
  const Eigen::Vector3d field0(kNominalFieldT, 0.0, 0.0);
  this->setEstimate(pm::Quaternion::Identity(), Eigen::Vector3d(0.05, 0.0, 0.0), 0.05, kStartTaiNs);
  this->setMagnetic(field0, kStartTaiNs, kNominalFieldT);
  this->runCycleAt(kStartTaiNs);
  this->sendCmd_CTRL_MODE_SET(0, 0, AttitudeController::CtrlMode::DETUMBLE);

  I64 t = kStartTaiNs + kPeriodNs;
  for (U32 i = 0; i < kStuckConfirmCycles; ++i) {
    // A field that walks along +Y (giving a +Y derivative, hence a -Y dipole on
    // rod 1 alone) *and* is 20 uT too strong in magnitude.
    const Eigen::Vector3d field(6.0e-5, 2.0e-6 * static_cast<double>(i + 1), 0.0);
    this->setEstimate(pm::Quaternion::Identity(), Eigen::Vector3d(0.05, 0.0, 0.0), 0.05, t);
    this->setMagnetic(field, t, kNominalFieldT);
    this->runCycleAt(t);
    t += kPeriodNs;
  }
  ASSERT_EVENTS_MtqStuckOn_SIZE(1);
  ASSERT_EVENTS_MtqStuckOn(0, 1, AttitudeController::StuckAttribution::DECISIVE, 0x2u,
                           this->eventHistory_MtqStuckOn->at(0).residualT, kStuckConfirmCycles);
}

void AttitudeControllerTester ::testResetClearsState() {
  this->setValidParameters();

  const Eigen::Vector3d disturbed(5.0e-5, 0.0, 0.0);
  I64 t = kStartTaiNs;
  for (U32 i = 0; i < kStuckConfirmCycles; ++i) {
    this->setEstimate(pm::Quaternion::Identity(), Eigen::Vector3d::Zero(), 1.0e-4, t);
    this->setMagnetic(disturbed, t, kNominalFieldT);
    this->runCycleAt(t);
    t += kPeriodNs;
  }
  ASSERT_EVENTS_MtqStuckOn_SIZE(1);
  EXPECT_FALSE(this->last_schedule_.get_interlockHealthy());

  this->sendCmd_CTRL_RESET(0, 0);
  ASSERT_EVENTS_ControllerReset_SIZE(1);

  // The latch is gone, so the next cycle publishes a healthy interlock — the
  // commanded re-admission path.
  this->setEstimate(pm::Quaternion::Identity(), Eigen::Vector3d::Zero(), 1.0e-4, t);
  this->clearMagnetic();
  this->runCycleAt(t);
  EXPECT_TRUE(this->last_schedule_.get_interlockHealthy());
}

// ----------------------------------------------------------------------
// Momentum management (§8.5)
// ----------------------------------------------------------------------

void AttitudeControllerTester ::testDesatEngagesAndDisengagesInPoint() {
  this->setValidParameters();

  // A field along +X and wheels loaded along +Z: the momentum error is fully
  // perpendicular to the field, so the unloading is at full effect and the
  // dipole is along +Y (dh x B).
  const Eigen::Vector3d field(kNominalFieldT, 0.0, 0.0);
  const pm::Quaternion attitude = pm::Quaternion::Identity();
  I64 t = kStartTaiNs;

  this->setEstimate(attitude, Eigen::Vector3d::Zero(), 1.0e-4, t);
  this->setMagnetic(field, t, kNominalFieldT);
  this->setWheelSpeeds(0.0);
  this->runCycleAt(t);
  this->sendCmd_CTRL_SET_TARGET_Q(0, 0, 1.0, 0.0, 0.0, 0.0);
  this->sendCmd_CTRL_MODE_SET(0, 0, AttitudeController::CtrlMode::POINT);
  this->clearHistory();

  // Empty wheels: no desaturation, and the rods stay off.
  t += kPeriodNs;
  this->setEstimate(attitude, Eigen::Vector3d::Zero(), 1.0e-4, t);
  this->setMagnetic(field, t, kNominalFieldT);
  this->runCycleAt(t);
  ASSERT_EVENTS_DesatEngaged_SIZE(0);
  EXPECT_EQ(this->last_on_window_s_, 0.0);

  // Past the threshold: engaged on the first cycle, with the rods commanded
  // *and* the wheels still holding the attitude — the concurrency claim.
  this->clearHistory();
  t += kPeriodNs;
  this->setWheelSpeeds(speedForMomentum(1.5e-3));
  this->setEstimate(attitude, Eigen::Vector3d::Zero(), 1.0e-4, t);
  this->setMagnetic(field, t, kNominalFieldT);
  this->runCycleAt(t);
  ASSERT_EVENTS_DesatEngaged_SIZE(1);
  ASSERT_TLM_DesatActive_SIZE(1);
  ASSERT_TLM_DesatActive(0, true);
  EXPECT_NEAR(this->last_on_window_s_, 0.5 * kPeriodSec, 1.0e-12);
  // dh is +Z and B is +X, so dh x B is along +Y: rod 1 carries the command and
  // the other two do not, which is also the ordering claim.
  EXPECT_GT(this->last_dipoles_[1][1], 0.0);
  EXPECT_NEAR(this->last_dipoles_[0][0], 0.0, 1.0e-15);
  EXPECT_NEAR(this->last_dipoles_[2][2], 0.0, 1.0e-15);
  EXPECT_EQ(this->last_schedule_.get_commandedMask(), 0x2u);
  // The wheels are still being driven by the pointing law: desaturation is
  // concurrent with POINT, not a mode that replaces it.
  ASSERT_TLM_TorqueCmd_SIZE(1);

  // **The rods' torque is fed forward into the wheel demand.** At a zero
  // attitude error and zero rate the PID's own terms are zero, so the whole
  // commanded torque is the feedforward — which must be exactly minus the
  // average magnetic torque the rods are about to apply, or the wheels would
  // discover it as pointing error instead (measured at 2.8 deg in SITL before
  // this term existed, against a 1.0 deg requirement).
  ASSERT_TLM_FeedforwardTorque_SIZE(1);
  const Vec3F64 ff = this->tlmHistory_FeedforwardTorque->at(0).arg;
  Eigen::Vector3d applied = Eigen::Vector3d::Zero();
  for (U32 i = 0; i < kMtqCount; ++i) {
    applied += Eigen::Vector3d(this->last_dipoles_[i][0], this->last_dipoles_[i][1],
                               this->last_dipoles_[i][2]);
  }
  // Minus the *average* magnetic torque: the rods carry the dipole only through
  // the on-window, so the duty factor is part of the model and not a detail.
  const Eigen::Vector3d expected = -kDutyFactorDefault * applied.cross(field);
  // The transverse axes carry the tier-1 residual-dipole model (m_res x B, ~5e-8
  // N.m here) and the axis of the desaturation torque carries the tier-2
  // observer's contribution as well — the harness stepped the wheel speeds,
  // which is a real momentum jump and which the observer honestly reports. Both
  // are orders below the term under test, which is what the tolerances say.
  EXPECT_NEAR(ff[0], expected.x(), 1.0e-7);
  EXPECT_NEAR(ff[1], expected.y(), 1.0e-7);
  EXPECT_NEAR(ff[2], expected.z(), 0.05 * std::abs(expected.z()));
  // ...and it reaches the *demand*, which is the claim that matters: with a zero
  // attitude and rate error the PID's own terms vanish, so the commanded torque
  // is the feedforward exactly. (The two differ from `expected` by the tier-2
  // observer's contribution — the harness stepped the wheel speeds, which is a
  // real momentum jump and which the observer honestly reports.)
  const Vec3F64 torque = this->tlmHistory_TorqueCmd->at(0).arg;
  EXPECT_NEAR(torque[0], ff[0], 1.0e-15) << "the feedforward did not reach the demand";
  EXPECT_NEAR(torque[1], ff[1], 1.0e-15);
  EXPECT_NEAR(torque[2], ff[2], 1.0e-15);

  // Momentum comes down under the exit threshold: still engaged through the
  // confirmation count, then disengaged. Both edges asserted.
  this->clearHistory();
  this->setWheelSpeeds(speedForMomentum(1.0e-4));
  for (U32 i = 0; i < kMomentumConfirmCycles; ++i) {
    t += kPeriodNs;
    this->setEstimate(attitude, Eigen::Vector3d::Zero(), 1.0e-4, t);
    this->setMagnetic(field, t, kNominalFieldT);
    this->runCycleAt(t);
  }
  ASSERT_EVENTS_DesatDisengaged_SIZE(1);
  EXPECT_EQ(this->last_on_window_s_, 0.0);
  for (U32 i = 0; i < MtqDipoleSet::SIZE; ++i) {
    for (U32 k = 0; k < 3; ++k) {
      EXPECT_EQ(this->last_dipoles_[i][k], 0.0);
    }
  }
}

void AttitudeControllerTester ::testDesatExcludedFromDetumbleAndIdle() {
  this->setValidParameters();

  const Eigen::Vector3d field(kNominalFieldT, 0.0, 0.0);
  // Wheels loaded well past the threshold for the whole test: if the mode gate
  // were missing, every cycle below would desaturate.
  this->setWheelSpeeds(speedForMomentum(1.8e-3));

  // IDLE.
  I64 t = kStartTaiNs;
  this->setEstimate(pm::Quaternion::Identity(), Eigen::Vector3d::Zero(), 1.0e-4, t);
  this->setMagnetic(field, t, kNominalFieldT);
  this->runCycleAt(t);
  t += kPeriodNs;
  this->setEstimate(pm::Quaternion::Identity(), Eigen::Vector3d::Zero(), 1.0e-4, t);
  this->setMagnetic(field, t, kNominalFieldT);
  this->runCycleAt(t);
  ASSERT_EVENTS_DesatEngaged_SIZE(0);
  EXPECT_EQ(this->last_on_window_s_, 0.0);

  // DETUMBLE: the rods *are* driven, but by B-dot — the dipole opposes the field
  // derivative rather than following dh x B, and no desaturation was engaged.
  // A field walking along +Y gives a B-dot dipole along -Y; the desaturation
  // demand for a +Z momentum error in a +X field would be along **+Y**, so the
  // sign of rod 1 is what tells the two laws apart.
  this->sendCmd_CTRL_MODE_SET(0, 0, AttitudeController::CtrlMode::DETUMBLE);
  this->clearHistory();
  const double step_t = 2.0e-9;
  for (int k = 1; k <= 3; ++k) {
    t += kPeriodNs;
    const Eigen::Vector3d walking(kNominalFieldT, step_t * k, 0.0);
    this->setEstimate(pm::Quaternion::Identity(), Eigen::Vector3d(0.05, 0.0, 0.0), 0.05, t);
    this->setMagnetic(walking, t, walking.norm());
    this->runCycleAt(t);
  }
  ASSERT_EVENTS_DesatEngaged_SIZE(0);
  EXPECT_LT(this->last_dipoles_[1][1], 0.0) << "the rods are B-dot's in DETUMBLE";
  ASSERT_TLM_DesatActive_SIZE(3);
  ASSERT_TLM_DesatActive(2, false);
}

void AttitudeControllerTester ::testDesatGroundOverride() {
  this->setValidParameters();

  const Eigen::Vector3d field(kNominalFieldT, 0.0, 0.0);
  I64 t = kStartTaiNs;
  this->setWheelSpeeds(speedForMomentum(1.5e-3));
  this->setEstimate(pm::Quaternion::Identity(), Eigen::Vector3d::Zero(), 1.0e-4, t);
  this->setMagnetic(field, t, kNominalFieldT);
  this->runCycleAt(t);
  this->sendCmd_CTRL_SET_TARGET_Q(0, 0, 1.0, 0.0, 0.0, 0.0);
  this->sendCmd_CTRL_MODE_SET(0, 0, AttitudeController::CtrlMode::POINT);

  // INHIBIT stops a desaturation the predicate is asking for, immediately.
  this->sendCmd_CTRL_DESAT(0, 0, AttitudeController::DesatOverride::INHIBIT);
  ASSERT_EVENTS_DesatOverrideChanged_SIZE(1);
  this->clearHistory();
  t += kPeriodNs;
  this->setEstimate(pm::Quaternion::Identity(), Eigen::Vector3d::Zero(), 1.0e-4, t);
  this->setMagnetic(field, t, kNominalFieldT);
  this->runCycleAt(t);
  ASSERT_EVENTS_DesatEngaged_SIZE(0);
  EXPECT_EQ(this->last_on_window_s_, 0.0);

  // AUTO hands the decision back, and the predicate is still asking.
  this->sendCmd_CTRL_DESAT(0, 0, AttitudeController::DesatOverride::AUTO);
  this->clearHistory();
  t += kPeriodNs;
  this->setEstimate(pm::Quaternion::Identity(), Eigen::Vector3d::Zero(), 1.0e-4, t);
  this->setMagnetic(field, t, kNominalFieldT);
  this->runCycleAt(t);
  ASSERT_EVENTS_DesatEngaged_SIZE(1);

  // FORCE desaturates momentum the predicate would leave alone...
  this->setWheelSpeeds(speedForMomentum(5.0e-5));
  this->sendCmd_CTRL_DESAT(0, 0, AttitudeController::DesatOverride::FORCE);
  this->clearHistory();
  t += kPeriodNs;
  this->setEstimate(pm::Quaternion::Identity(), Eigen::Vector3d::Zero(), 1.0e-4, t);
  this->setMagnetic(field, t, kNominalFieldT);
  this->runCycleAt(t);
  EXPECT_GT(this->last_on_window_s_, 0.0);
  EXPECT_GT(this->last_dipoles_[1][1], 0.0);

  // ...but it is a *permission*, not an instruction to drive a rod blind: with no
  // admissible field sample there is no law to run, and the rods stay off.
  this->clearHistory();
  t += kPeriodNs;
  this->setEstimate(pm::Quaternion::Identity(), Eigen::Vector3d::Zero(), 1.0e-4, t);
  this->clearMagnetic();
  this->runCycleAt(t);
  EXPECT_EQ(this->last_on_window_s_, 0.0);
  ASSERT_EVENTS_DesatDisengaged_SIZE(1);
}

void AttitudeControllerTester ::testWheelFrictionFeedforward() {
  const double angle = 10.0 * M_PI / 180.0;
  const double speed = 20.0;
  // Spinning positive, so the bearings drag negative and the drive is asked for
  // more positive torque than the allocation demanded.
  const double expected = kWheelDryFrictionNm + kWheelViscousFrictionNmS * speed;
  const pm::Quaternion attitude =
      pm::Quaternion::FromAxisAngle(Eigen::Vector3d::UnitX(), angle).canonical();
  const double s = 1.0 / std::sqrt(3.0);
  const double signs[4][3] = {{1, 1, 1}, {-1, 1, 1}, {-1, -1, 1}, {1, -1, 1}};

  I64 t = kStartTaiNs;

  // (1) A vehicle whose friction model is missing is refused outright, before
  //     anything else — it is a vehicle nobody has characterised, and flying it
  //     with the feedforward quietly off would hide that behind pointing error
  //     nobody could attribute. The coefficients are therefore read and validated
  //     whether or not the feedforward is enabled.
  this->setValidParameters(0.5, 0.03, /*withFriction=*/false);
  this->setEstimate(attitude, Eigen::Vector3d::Zero(), 1.0e-4, t);
  this->runCycleAt(t);
  ASSERT_EVENTS_ConfigInvalid_SIZE(1);
  ASSERT_TLM_CtrlModeTlm(0, AttitudeController::CtrlMode::IDLE);
  for (U32 i = 0; i < WheelTorqueSet::SIZE; ++i) {
    EXPECT_EQ(this->last_wheels_[i], 0.0);
  }

  // A 10 degree error about body +X on a **spinning** array — the operating
  // point the whole feature is about. 20 rad/s is two hundred times the
  // deadband, so the Coulomb term is at full magnitude and the blend is not what
  // is under test here.
  this->clearHistory();
  this->setValidParameters();
  this->setWheelSpeeds(speed);
  t += kPeriodNs;
  this->setEstimate(attitude, Eigen::Vector3d::Zero(), 1.0e-4, t);
  this->runCycleAt(t);
  this->sendCmd_CTRL_SET_TARGET_Q(0, 0, 1.0, 0.0, 0.0, 0.0);
  this->sendCmd_CTRL_MODE_SET(0, 0, AttitudeController::CtrlMode::POINT);
  this->clearHistory();

  t += kPeriodNs;
  this->setEstimate(attitude, Eigen::Vector3d::Zero(), 1.0e-4, t);
  this->runCycleAt(t);

  // (2) Each wheel carries the allocation's demand *plus* the modelled friction,
  //     and the two are separable from telemetry alone — which is what makes the
  //     WheelTorque/WheelFrictionNm pair an ablation an operator can read without
  //     a ground model.
  ASSERT_TLM_WheelFrictionNm_SIZE(1);
  F64PerUnit friction;
  friction = this->tlmHistory_WheelFrictionNm->at(0).arg;
  for (U32 i = 0; i < kWheelCount; ++i) {
    EXPECT_NEAR(friction[i], expected, 1.0e-15);
  }

  // (3) Subtracting it recovers the commanded body torque exactly: the friction
  //     term is spent inside the bearings and must not appear in the delivered
  //     body torque the allocation solved for.
  ASSERT_TLM_TorqueCmd_SIZE(1);
  Vec3F64 torque;
  torque = this->tlmHistory_TorqueCmd->at(0).arg;
  Eigen::Vector3d delivered = Eigen::Vector3d::Zero();
  for (U32 i = 0; i < kWheelCount; ++i) {
    const Eigen::Vector3d axis(-signs[i][0] * s, -signs[i][1] * s, -signs[i][2] * s);
    delivered += axis * (this->last_wheels_[i] - friction[i]);
  }
  EXPECT_NEAR(delivered.x(), torque[0], 1.0e-12);
  EXPECT_NEAR(delivered.y(), torque[1], 1.0e-12);
  EXPECT_NEAR(delivered.z(), torque[2], 1.0e-12);

  // (4) A wheel whose tachometer is not usable is passed through uncompensated
  //     rather than handed a guessed sign — and the others are still helped, so
  //     one dead tach costs one wheel's compensation and not the feature.
  this->wheel_speed_valid_[2] = false;
  this->clearHistory();
  t += kPeriodNs;
  this->setEstimate(attitude, Eigen::Vector3d::Zero(), 1.0e-4, t);
  this->runCycleAt(t);
  friction = this->tlmHistory_WheelFrictionNm->at(0).arg;
  EXPECT_TRUE(std::isnan(friction[2]));
  EXPECT_NEAR(friction[0], expected, 1.0e-15);
  this->wheel_speed_valid_[2] = true;

  // (5) Disabling the feedforward puts the uncompensated vehicle back exactly —
  //     the ablation the SITL rows fly to measure what it buys. Sent through the
  //     parameter *port* rather than staged in the harness table, because that is
  //     the path an uplink takes and the only one that re-reads the tuning.
  this->paramSet_WheelFrictionEnable(0, Fw::ParamValid::VALID);
  this->paramSend_WheelFrictionEnable(0, 0);
  this->sendCmd_CTRL_SET_TARGET_Q(0, 0, 1.0, 0.0, 0.0, 0.0);
  this->sendCmd_CTRL_MODE_SET(0, 0, AttitudeController::CtrlMode::POINT);
  this->clearHistory();
  t += kPeriodNs;
  this->setEstimate(attitude, Eigen::Vector3d::Zero(), 1.0e-4, t);
  this->runCycleAt(t);
  friction = this->tlmHistory_WheelFrictionNm->at(0).arg;
  torque = this->tlmHistory_TorqueCmd->at(0).arg;
  delivered = Eigen::Vector3d::Zero();
  for (U32 i = 0; i < kWheelCount; ++i) {
    EXPECT_TRUE(std::isnan(friction[i]));
    const Eigen::Vector3d axis(-signs[i][0] * s, -signs[i][1] * s, -signs[i][2] * s);
    delivered += axis * this->last_wheels_[i];
  }
  EXPECT_NEAR(delivered.x(), torque[0], 1.0e-12);

  // (6) A zero deadband is the discontinuous sign() the design rejects, not "no
  //     blending wanted", and it takes the whole configuration down rather than
  //     being silently reinterpreted.
  this->paramSet_WheelFrictionDeadbandRadps(0.0, Fw::ParamValid::VALID);
  this->clearHistory();
  this->paramSend_WheelFrictionDeadbandRadps(0, 0);
  t += kPeriodNs;
  this->setEstimate(attitude, Eigen::Vector3d::Zero(), 1.0e-4, t);
  this->runCycleAt(t);
  ASSERT_EVENTS_ConfigInvalid_SIZE(1);
}

void AttitudeControllerTester ::testWheelCapacityMonitorSeesNullSpaceMomentum() {
  this->setValidParameters();
  // Wheels spinning against each other in the pyramid's null pattern
  // [+,-,+,-]: the body momentum sums to zero, so the envelope monitor, the
  // desaturation threshold and the observer are all quiet — while every wheel
  // holds 95 % of its capacity. This is the case the per-wheel monitor exists for.
  const double w = 0.95 * kWheelCapacityNms / kWheelInertiaKgm2;
  const double pattern[4] = {w, -w, w, -w};
  for (U32 i = 0; i < kWheelCount; ++i) {
    this->wheel_speed_radps_[i] = pattern[i];
    this->wheel_speed_valid_[i] = true;
  }
  I64 t = kStartTaiNs;
  this->setEstimate(pm::Quaternion::Identity(), Eigen::Vector3d::Zero(), 1.0e-4, t);
  this->runCycleAt(t);
  ASSERT_EVENTS_MomentumEnvelopeExceeded_SIZE(0);
  ASSERT_TLM_StoredMomentumNms_SIZE(1);
  EXPECT_NEAR(this->tlmHistory_StoredMomentumNms->at(0).arg, 0.0, 1.0e-9);
  ASSERT_TLM_MaxWheelMomentumNms_SIZE(1);
  EXPECT_NEAR(this->tlmHistory_MaxWheelMomentumNms->at(0).arg, 0.95 * kWheelCapacityNms, 1.0e-9);
  ASSERT_TLM_NullSpaceMomentumNms_SIZE(1);
  EXPECT_NEAR(this->tlmHistory_NullSpaceMomentumNms->at(0).arg, 2.0 * 0.95 * kWheelCapacityNms,
              1.0e-9);  // ||[+,-,+,-]|| * h_i
  ASSERT_EVENTS_WheelNearCapacity_SIZE(1);
  ASSERT_EVENTS_WheelCapacityRecovered_SIZE(0);

  // Persisting: one event, however long.
  for (int i = 0; i < 3; ++i) {
    t += kPeriodNs;
    this->setEstimate(pm::Quaternion::Identity(), Eigen::Vector3d::Zero(), 1.0e-4, t);
    this->runCycleAt(t);
  }
  ASSERT_EVENTS_WheelNearCapacity_SIZE(1);

  // Back under 90 %: the recovery edge, once.
  this->clearHistory();
  this->setWheelSpeeds(0.5 * kWheelCapacityNms / kWheelInertiaKgm2);
  t += kPeriodNs;
  this->setEstimate(pm::Quaternion::Identity(), Eigen::Vector3d::Zero(), 1.0e-4, t);
  this->runCycleAt(t);
  ASSERT_EVENTS_WheelCapacityRecovered_SIZE(1);
  ASSERT_EVENTS_WheelNearCapacity_SIZE(0);
}

void AttitudeControllerTester ::testWheelBiasServoAddsNullSpaceTorqueOnly() {
  this->setValidParameters();
  // Bias on: [+b,-b,+b,-b] at 10 % of capacity, a slow trim.
  const double b = 0.1 * kWheelCapacityNms;
  F64PerUnit pattern;
  for (U32 i = 0; i < F64PerUnit::SIZE; ++i) {
    pattern[i] = i < kWheelCount ? ((i % 2 == 0) ? b : -b) : 0.0;
  }
  this->paramSet_WheelBiasNms(pattern, Fw::ParamValid::VALID);
  this->paramSet_WheelBiasGainPerS(0.02, Fw::ParamValid::VALID);
  this->paramSet_WheelBiasMaxTorqueNm(1.0e-3, Fw::ParamValid::VALID);
  this->component.loadParameters();
  this->setWheelSpeeds(0.0);  // wheels at rest: the whole pattern is the error

  // Ten degrees off, POINT: pointing torque plus the bias trim.
  const double angle = 10.0 * M_PI / 180.0;
  const pm::Quaternion attitude =
      pm::Quaternion::FromAxisAngle(Eigen::Vector3d::UnitX(), angle).canonical();
  this->setEstimate(attitude, Eigen::Vector3d::Zero(), 1.0e-4, kStartTaiNs);
  this->runCycleAt(kStartTaiNs);
  this->sendCmd_CTRL_SET_TARGET_Q(0, 0, 1.0, 0.0, 0.0, 0.0);
  this->sendCmd_CTRL_MODE_SET(0, 0, AttitudeController::CtrlMode::POINT);
  this->clearHistory();
  this->setEstimate(attitude, Eigen::Vector3d::Zero(), 1.0e-4, kStartTaiNs + kPeriodNs);
  this->runCycleAt(kStartTaiNs + kPeriodNs);

  ASSERT_EVENTS_WheelBiasEngaged_SIZE(0);  // engaged at configuration, before clearHistory
  ASSERT_TLM_TorqueCmd_SIZE(1);
  const Vec3F64 torque = this->tlmHistory_TorqueCmd->at(0).arg;
  // The wheels deliver exactly the PID torque — the bias trim is invisible to
  // the body — and every wheel carries the trim on top: gain * b in the pattern.
  const double s = 1.0 / std::sqrt(3.0);
  const double signs[4][3] = {{1, 1, 1}, {-1, 1, 1}, {-1, -1, 1}, {1, -1, 1}};
  Eigen::Vector3d delivered = Eigen::Vector3d::Zero();
  double null_content = 0.0;
  for (U32 i = 0; i < kWheelCount; ++i) {
    const Eigen::Vector3d axis(-signs[i][0] * s, -signs[i][1] * s, -signs[i][2] * s);
    delivered += axis * this->last_wheels_[i];
    null_content += this->last_wheels_[i] * ((i % 2 == 0) ? 0.5 : -0.5);
  }
  EXPECT_NEAR(delivered.x(), torque[0], 1.0e-12);
  EXPECT_NEAR(delivered.y(), torque[1], 1.0e-12);
  EXPECT_NEAR(delivered.z(), torque[2], 1.0e-12);
  // Projection of the command onto the null vector [+,-,+,-]/2: gain*b*|pattern|
  // = 0.02 * b * 2 (the min-norm allocation contributes nothing there).
  EXPECT_NEAR(null_content, 0.02 * b * 2.0, 1.0e-12);
  ASSERT_TLM_WheelBiasTorqueNm_SIZE(1);
  EXPECT_NEAR(this->tlmHistory_WheelBiasTorqueNm->at(0).arg, 0.02 * b, 1.0e-12);

  // Wheels already at the pattern: the trim rests and only pointing remains.
  this->clearHistory();
  for (U32 i = 0; i < kWheelCount; ++i) {
    this->wheel_speed_radps_[i] = pattern[i] / kWheelInertiaKgm2;
    this->wheel_speed_valid_[i] = true;
  }
  this->setEstimate(attitude, Eigen::Vector3d::Zero(), 1.0e-4, kStartTaiNs + 2 * kPeriodNs);
  this->runCycleAt(kStartTaiNs + 2 * kPeriodNs);
  ASSERT_TLM_WheelBiasTorqueNm_SIZE(1);
  EXPECT_NEAR(this->tlmHistory_WheelBiasTorqueNm->at(0).arg, 0.0, 1.0e-12);
}

void AttitudeControllerTester ::testWheelBiasPastCapacityIsRefused() {
  this->setValidParameters();
  F64PerUnit pattern;
  for (U32 i = 0; i < F64PerUnit::SIZE; ++i) {
    pattern[i] = 0.0;
  }
  pattern[0] = kWheelCapacityNms;  // a "bias" that is the whole wheel
  this->paramSet_WheelBiasNms(pattern, Fw::ParamValid::VALID);
  this->paramSet_WheelBiasGainPerS(0.02, Fw::ParamValid::VALID);
  this->paramSet_WheelBiasMaxTorqueNm(1.0e-3, Fw::ParamValid::VALID);
  this->component.loadParameters();
  this->setEstimate(pm::Quaternion::Identity(), Eigen::Vector3d::Zero(), 1.0e-4, kStartTaiNs);
  this->runCycleAt(kStartTaiNs);
  ASSERT_EVENTS_ConfigInvalid_SIZE(1);
  ASSERT_EVENTS_WheelBiasEngaged_SIZE(0);
}

void AttitudeControllerTester ::testMomentumEnvelopeAndWheelDropout() {
  this->setValidParameters();

  I64 t = kStartTaiNs;
  this->setWheelSpeeds(speedForMomentum(1.0e-3));
  this->setEstimate(pm::Quaternion::Identity(), Eigen::Vector3d::Zero(), 1.0e-4, t);
  this->runCycleAt(t);
  ASSERT_EVENTS_MomentumEnvelopeExceeded_SIZE(0);
  ASSERT_TLM_StoredMomentumNms_SIZE(1);
  EXPECT_NEAR(this->tlmHistory_StoredMomentumNms->at(0).arg, 1.0e-3, 1.0e-9);

  // Past the envelope: one event, and only one however long it persists.
  this->setWheelSpeeds(speedForMomentum(2.5e-3));
  for (int i = 0; i < 3; ++i) {
    t += kPeriodNs;
    this->setEstimate(pm::Quaternion::Identity(), Eigen::Vector3d::Zero(), 1.0e-4, t);
    this->runCycleAt(t);
  }
  ASSERT_EVENTS_MomentumEnvelopeExceeded_SIZE(1);

  // A wheel with no usable speed refuses the sum rather than understating it —
  // and holds the latch, because a missing tachometer is not evidence that the
  // wheels emptied. The telemetry says "no value" rather than zero — on the
  // vector channel too, whose zero would draw as a perfectly empty array — and
  // the refusal is FDIR-visible: one dead tachometer blinds desaturation and
  // both §9 monitors at once, so the event names the reason and the wheel
  // instead of leaving a silent NaN as the only witness.
  this->clearHistory();
  this->wheel_speed_valid_[2] = false;
  t += kPeriodNs;
  this->setEstimate(pm::Quaternion::Identity(), Eigen::Vector3d::Zero(), 1.0e-4, t);
  this->runCycleAt(t);
  ASSERT_TLM_StoredMomentumNms_SIZE(1);
  EXPECT_TRUE(std::isnan(this->tlmHistory_StoredMomentumNms->at(0).arg));
  ASSERT_TLM_StoredMomentum_SIZE(1);
  EXPECT_TRUE(std::isnan(this->tlmHistory_StoredMomentum->at(0).arg[0]));
  ASSERT_TLM_MomentumValid_SIZE(1);
  ASSERT_TLM_MomentumValid(0, false);
  ASSERT_EVENTS_MomentumUnavailable_SIZE(1);
  ASSERT_EVENTS_MomentumUnavailable(0, AttitudeController::MomentumRefusalEv::WHEEL_INVALID, 2, 1);
  ASSERT_EVENTS_MomentumEnvelopeRecovered_SIZE(0);

  // The wheel comes back inside the envelope: recovery on the same comparison,
  // the validity channel back to true, and no further refusal events.
  this->clearHistory();
  this->setWheelSpeeds(speedForMomentum(5.0e-4));
  t += kPeriodNs;
  this->setEstimate(pm::Quaternion::Identity(), Eigen::Vector3d::Zero(), 1.0e-4, t);
  this->runCycleAt(t);
  ASSERT_EVENTS_MomentumEnvelopeRecovered_SIZE(1);
  ASSERT_EVENTS_MomentumUnavailable_SIZE(0);
  ASSERT_TLM_MomentumValid_SIZE(1);
  ASSERT_TLM_MomentumValid(0, true);
}

}  // namespace flight
