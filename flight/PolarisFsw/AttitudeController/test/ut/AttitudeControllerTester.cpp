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
constexpr U32 kWheelCount = 4;
constexpr F64 kWheelMaxTorqueNm = 0.025;
constexpr F64 kAllocMinConditioning = 0.05;
constexpr U32 kMtqCount = 3;
constexpr F64 kStuckResidualT = 8.0e-6;
constexpr F64 kWindowToleranceSec = 0.001;
constexpr U32 kStuckConfirmCycles = 5;
constexpr U32 kStuckClearCycles = 20;
constexpr U32 kAlertCycles = 100;

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

void AttitudeControllerTester ::setValidParameters(F64 dutyFactor, F64 settleSec) {
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

void AttitudeControllerTester ::runCycleAt(I64 taiNs) {
  const U32 seconds = static_cast<U32>(taiNs / kNsPerSecond);
  const U32 useconds = static_cast<U32>((taiNs % kNsPerSecond) / 1000);
  this->setTestTime(Fw::Time(seconds, useconds));
  this->invoke_to_estimateIn(0, this->estimate_);
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

}  // namespace flight
