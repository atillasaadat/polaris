// ======================================================================
// \title  AttitudeController.cpp
// \brief  Attitude control on the 10 Hz GNC cycle (§8.5, §7, §9)
// ======================================================================

#include "flight/PolarisFsw/AttitudeController/AttitudeController.hpp"

#include <cmath>
#include <limits>

#include "constants/constants.hpp"
#include "Fw/Log/LogString.hpp"

namespace flight {

namespace pm = polaris::math;
using Body = polaris::math::frames::Body;
using ECI = polaris::math::frames::ECI;

// The wheel command array (SitlPorts) and the wheel telemetry array (GncPorts)
// are declared in different modules and sized independently. `applyParameters`
// bounds WheelCount by one of them and `commandActuators` writes the other, so a
// divergence would silently leave the highest-indexed wheels uncommanded.
static_assert(static_cast<U32>(WheelTorqueSet::SIZE) == static_cast<U32>(F64PerUnit::SIZE),
              "WheelTorqueSet and F64PerUnit must be the same width");

namespace {

constexpr I64 kNsPerSecond = 1000000000LL;
constexpr I64 kNsPerMicrosecond = 1000LL;

//! Telemetry sentinel for "no value this cycle" — NaN draws as a gap on a strip
//! chart, where a zero would draw as a perfect measurement. Same convention the
//! estimator uses.
const F64 kNoValue = std::numeric_limits<F64>::quiet_NaN();

Vec3F64 toVec3F64(const Eigen::Vector3d& v) {
  Vec3F64 out;
  out[0] = v[0];
  out[1] = v[1];
  out[2] = v[2];
  return out;
}

pm::Vec3<Body> fromVec3F64(const Vec3F64& v) {
  return pm::Vec3<Body>(Eigen::Vector3d(v[0], v[1], v[2]));
}

}  // namespace

// ----------------------------------------------------------------------
// Construction
// ----------------------------------------------------------------------

AttitudeController ::AttitudeController(const char* const compName)
    : AttitudeControllerComponentBase(compName) {}

AttitudeController ::~AttitudeController() {}

void AttitudeController ::guidanceIn_handler(FwIndexType, const AttitudeTarget& target) {
  if (!target.get_valid()) {
    this->guidance_valid_ = false;
    return;
  }
  const QuatF64& q = target.get_qBodyFromEci();
  const polaris::math::Quaternion core(q[0], q[1], q[2], q[3]);
  const Vec3F64& w = target.get_rateBodyRadS();
  const pm::Vec3<Body> rate(w[0], w[1], w[2]);
  // A non-finite or null-norm target is dropped rather than normalised into
  // something plausible: the guidance is the only thing that knows what it
  // meant, and a controller that repairs its input hides the fault.
  if (!core.isFinite() || !rate.isFinite() || !(core.coeffs().norm() > 0.0)) {
    this->guidance_valid_ = false;
    return;
  }
  this->guidance_target_ = polaris::math::Quat<Body, ECI>(core.canonical());
  this->guidance_rate_ = rate;
  this->guidance_valid_ = true;
}

void AttitudeController ::commandModeAtStartup(U32 mode, const F64 q[4]) {
  const double norm = std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
  if (norm > 0.0) {
    Fw::CmdArgBuffer args;
    if (args.serializeFrom(q[0]) == Fw::FW_SERIALIZE_OK &&
        args.serializeFrom(q[1]) == Fw::FW_SERIALIZE_OK &&
        args.serializeFrom(q[2]) == Fw::FW_SERIALIZE_OK &&
        args.serializeFrom(q[3]) == Fw::FW_SERIALIZE_OK) {
      this->get_cmdIn_InputPort(0)->invoke(this->getIdBase() + OPCODE_CTRL_SET_TARGET_Q, 0, args);
    }
  }
  if (mode == 0) {
    return;
  }
  // The mode request is issued at setup, before the first rate-group cycle, so
  // no estimate has arrived and POINT would be refused on NO_ESTIMATE. DETUMBLE
  // is refused for the same reason. Both are therefore *latched* here rather than
  // commanded: `pending_mode_` is retried each cycle until the estimate can
  // support it, which is what a ground operator would do and what the Phase-7
  // mode manager will do.
  this->pending_mode_ = mode;
}

void AttitudeController ::setWheelBiasAtStartup(bool enable) {
  this->wheel_bias_override_ = true;
  this->wheel_bias_override_enable_ = enable;
  (void)this->applyParameters();
}

void AttitudeController ::setFeedforwardAtStartup(bool model, bool observer) {
  this->feedforward_override_ = true;
  this->feedforward_model_override_ = model;
  this->feedforward_observer_override_ = observer;
  (void)this->applyParameters();
}

// ----------------------------------------------------------------------
// Inputs
// ----------------------------------------------------------------------

void AttitudeController ::estimateIn_handler(FwIndexType portNum,
                                             const AttitudeEstimate& estimate) {
  static_cast<void>(portNum);
  this->estimate_ = estimate;
  this->have_estimate_ = true;
}

void AttitudeController ::wheelSpeedIn_handler(FwIndexType portNum, const WheelSpeedMeas& meas) {
  const U32 index = static_cast<U32>(portNum);
  if (index >= static_cast<U32>(polaris::gnc::kMaxWheels)) {
    return;
  }
  this->wheel_speed_radps_[index] = meas.get_speedRadps();
  this->wheel_speed_time_ns_[index] = meas.get_timeTagNs();
  // The unit's own validity flag is necessary but not sufficient: a finite-looking
  // reading is what the momentum sum needs, and an infinite one flagged valid
  // would poison every term (§9.1).
  this->wheel_speed_valid_[index] = meas.get_valid() && std::isfinite(meas.get_speedRadps());
}

I64 AttitudeController ::currentTaiNs() const {
  const Fw::Time now = this->getTime();
  return static_cast<I64>(now.getSeconds()) * kNsPerSecond +
         static_cast<I64>(now.getUSeconds()) * kNsPerMicrosecond;
}

// ----------------------------------------------------------------------
// Configuration (§19.3 — no defaults)
// ----------------------------------------------------------------------

bool AttitudeController ::readAxes(const Vec3F64PerUnit& axes, U32 count, Eigen::Vector3d* out) {
  for (U32 i = 0; i < count; ++i) {
    const Eigen::Vector3d raw(axes[3 * i], axes[3 * i + 1], axes[3 * i + 2]);
    const double norm = raw.norm();
    if (!raw.allFinite() || !(norm > 0.0)) {
      return false;
    }
    out[i] = raw / norm;
  }
  return true;
}

bool AttitudeController ::applyParameters() {
  auto fail = [this](const char* detail) {
    if (!this->config_alerted_) {
      Fw::LogStringArg arg(detail);
      this->log_WARNING_HI_ConfigInvalid(arg);
      this->config_alerted_ = true;
    }
    this->configured_ = false;
    return false;
  };

  // Each parameter is read with its own validity flag: a value that is not VALID
  // has no value at all, and a controller that substituted one would be flying a
  // gain nobody chose. The macro exists so that cannot be forgotten on the
  // twenty-odd reads below.
#define POLARIS_GET(dest, getter, name)         \
  do {                                          \
    Fw::ParamValid v = Fw::ParamValid::INVALID; \
    (dest) = this->getter(v);                   \
    if (v != Fw::ParamValid::VALID) {           \
      return fail(name);                        \
    }                                           \
  } while (0)

  POLARIS_GET(this->control_period_s_, paramGet_ControlPeriodSec, "ControlPeriodSec");
  POLARIS_GET(this->max_estimate_age_s_, paramGet_MaxEstimateAgeSec, "MaxEstimateAgeSec");
  POLARIS_GET(this->max_att_sigma_rad_, paramGet_MaxAttSigmaRad, "MaxAttSigmaRad");

  polaris::gnc::BdotConfig bdot;
  POLARIS_GET(bdot.gain_nms, paramGet_BdotGainNms, "BdotGainNms");
  POLARIS_GET(this->bdot_max_dipole_am2_, paramGet_BdotMaxDipoleAm2, "BdotMaxDipoleAm2");
  POLARIS_GET(bdot.min_sample_dt_s, paramGet_BdotMinSampleDtSec, "BdotMinSampleDtSec");
  POLARIS_GET(bdot.max_sample_dt_s, paramGet_BdotMaxSampleDtSec, "BdotMaxSampleDtSec");
  POLARIS_GET(bdot.duty_factor, paramGet_MtqDutyFactor, "MtqDutyFactor");

  polaris::gnc::RateHysteresisConfig hysteresis;
  POLARIS_GET(hysteresis.enter_radps, paramGet_DetumbleEnterRadps, "DetumbleEnterRadps");
  POLARIS_GET(hysteresis.exit_radps, paramGet_DetumbleExitRadps, "DetumbleExitRadps");
  {
    Fw::ParamValid v = Fw::ParamValid::INVALID;
    hysteresis.confirm_cycles = this->paramGet_DetumbleConfirmCycles(v);
    if (v != Fw::ParamValid::VALID) {
      return fail("DetumbleConfirmCycles");
    }
  }

  polaris::gnc::AttitudePidConfig pid;
  POLARIS_GET(pid.kp_nm_per_rad, paramGet_PidKpNmPerRad, "PidKpNmPerRad");
  POLARIS_GET(pid.ki_nm_per_rad_s, paramGet_PidKiNmPerRadS, "PidKiNmPerRadS");
  POLARIS_GET(pid.kd_nm_per_radps, paramGet_PidKdNmPerRadps, "PidKdNmPerRadps");
  POLARIS_GET(pid.max_integral_rad_s, paramGet_PidMaxIntegralRadS, "PidMaxIntegralRadS");
  POLARIS_GET(pid.max_torque_nm, paramGet_PidMaxTorqueNm, "PidMaxTorqueNm");
  POLARIS_GET(pid.max_dt_s, paramGet_PidMaxDtSec, "PidMaxDtSec");
  POLARIS_GET(pid.max_slew_rate_radps, paramGet_PidMaxSlewRateRadps, "PidMaxSlewRateRadps");

  polaris::gnc::RwAllocationConfig alloc;
  F64 wheel_max_torque = 0.0;
  {
    Fw::ParamValid v = Fw::ParamValid::INVALID;
    const U32 count = this->paramGet_WheelCount(v);
    if (v != Fw::ParamValid::VALID) {
      return fail("WheelCount");
    }
    if (count < 3 || count > static_cast<U32>(polaris::gnc::kMaxWheels) ||
        count > static_cast<U32>(F64PerUnit::SIZE)) {
      return fail("WheelCount out of range");
    }
    alloc.wheel_count = static_cast<int>(count);
    this->wheel_count_ = count;
  }
  POLARIS_GET(wheel_max_torque, paramGet_WheelMaxTorqueNm, "WheelMaxTorqueNm");
  POLARIS_GET(alloc.min_conditioning, paramGet_AllocMinConditioning, "AllocMinConditioning");
  {
    Fw::ParamValid v = Fw::ParamValid::INVALID;
    const Vec3F64PerUnit axes = this->paramGet_WheelAxesBody(v);
    if (v != Fw::ParamValid::VALID) {
      return fail("WheelAxesBody");
    }
    Eigen::Vector3d spin[polaris::gnc::kMaxWheels];
    if (!readAxes(axes, this->wheel_count_, spin)) {
      return fail("WheelAxesBody has a null or non-finite slot");
    }
    for (U32 i = 0; i < this->wheel_count_; ++i) {
      // **The sign lives here, once.** The config carries each wheel's physical
      // spin axis; a wheel's reaction on the body is -I*omega_dot, so the body
      // torque per unit of commanded motor torque is the *negated* axis. Building
      // the allocator's columns from the spin axes unnegated gives a
      // sign-inverted, perfectly plausible controller.
      alloc.axes.col(static_cast<Eigen::Index>(i)) = -spin[i];
      alloc.max_torque_nm[i] = wheel_max_torque;
      // The momentum side keeps the axes **unnegated**: a wheel spinning along
      // +a stores momentum along +a while its motor torque reacts on the body
      // along -a. Two signs, one config read, both stated where they are applied.
      this->wheel_axes_[i] = spin[i];
    }
  }
  {
    Fw::ParamValid v = Fw::ParamValid::INVALID;
    const U8 method = this->paramGet_AllocMethodSel(v);
    if (v != Fw::ParamValid::VALID || method > static_cast<U8>(AllocMethod::MIN_MAX)) {
      return fail("AllocMethodSel");
    }
    this->alloc_method_ = method == static_cast<U8>(AllocMethod::MIN_MAX)
                              ? polaris::gnc::RwAllocationMethod::kMinMax
                              : polaris::gnc::RwAllocationMethod::kMinNorm;
  }

  // --- Wheel-drive friction feedforward (§8.5; REQ-ACTL-010) ---------------
  // Read and validated whether or not it is enabled: a vehicle whose friction
  // model is missing is a vehicle nobody has characterised, and flying it with
  // the feedforward silently off would hide that behind 3 deg of pointing error
  // nobody could attribute.
  polaris::gnc::RwFrictionConfig friction;
  friction.wheel_count = alloc.wheel_count;
  POLARIS_GET(friction.dry_friction_nm, paramGet_WheelDryFrictionNm, "WheelDryFrictionNm");
  POLARIS_GET(friction.viscous_friction_nm_s, paramGet_WheelViscousFrictionNmS,
              "WheelViscousFrictionNmS");
  POLARIS_GET(friction.deadband_radps, paramGet_WheelFrictionDeadbandRadps,
              "WheelFrictionDeadbandRadps");
  {
    Fw::ParamValid v = Fw::ParamValid::INVALID;
    const U8 enable = this->paramGet_WheelFrictionEnable(v);
    if (v != Fw::ParamValid::VALID || enable > 1) {
      return fail("WheelFrictionEnable");
    }
    this->friction_enabled_ = enable != 0;
    const F64PerUnit scale = this->paramGet_WheelFrictionScale(v);
    if (v != Fw::ParamValid::VALID) {
      return fail("WheelFrictionScale");
    }
    for (U32 i = 0; i < this->wheel_count_; ++i) {
      friction.scale[i] = scale[i];
      // The torque box the compensated command is clamped to is the allocation's
      // own, read once above — two limits for one drive is two ways to describe
      // the same hardware and one of them would eventually be wrong.
      friction.max_torque_nm[i] = wheel_max_torque;
    }
  }

  {
    Fw::ParamValid v = Fw::ParamValid::INVALID;
    const U32 count = this->paramGet_MtqCount(v);
    if (v != Fw::ParamValid::VALID) {
      return fail("MtqCount");
    }
    if (count != kRodCount) {
      return fail("MtqCount must be 3 (orthogonal rod triad)");
    }
    const Vec3F64PerUnit axes = this->paramGet_MtqAxesBody(v);
    if (v != Fw::ParamValid::VALID) {
      return fail("MtqAxesBody");
    }
    if (!readAxes(axes, kRodCount, this->rod_axes_)) {
      return fail("MtqAxesBody has a null or non-finite slot");
    }
    // The commanded body dipole is resolved onto the rods by projection and
    // clamped per rod, which reconstructs the demand exactly only for a mutually
    // orthogonal set. A skewed set needs the allocation layer the wheels use, so
    // it is refused rather than silently approximated.
    for (U32 i = 0; i < kRodCount; ++i) {
      for (U32 j = i + 1; j < kRodCount; ++j) {
        if (std::abs(this->rod_axes_[i].dot(this->rod_axes_[j])) > kRodOrthogonalityTol) {
          return fail("MtqAxesBody is not an orthogonal triad");
        }
      }
    }
  }
  POLARIS_GET(this->mtq_duty_factor_, paramGet_MtqDutyFactor, "MtqDutyFactor");
  POLARIS_GET(this->mtq_settle_s_, paramGet_MtqSettleSec, "MtqSettleSec");
  POLARIS_GET(this->mtq_window_tolerance_s_, paramGet_MtqWindowToleranceSec,
              "MtqWindowToleranceSec");
  POLARIS_GET(this->mtq_stuck_residual_t_, paramGet_MtqStuckResidualT, "MtqStuckResidualT");
  {
    Fw::ParamValid v = Fw::ParamValid::INVALID;
    this->mtq_stuck_confirm_cycles_ = this->paramGet_MtqStuckConfirmCycles(v);
    if (v != Fw::ParamValid::VALID || this->mtq_stuck_confirm_cycles_ == 0) {
      return fail("MtqStuckConfirmCycles");
    }
    this->mtq_stuck_clear_cycles_ = this->paramGet_MtqStuckClearCycles(v);
    if (v != Fw::ParamValid::VALID || this->mtq_stuck_clear_cycles_ == 0) {
      return fail("MtqStuckClearCycles");
    }
    this->alert_cycles_ = this->paramGet_AlertCycles(v);
    if (v != Fw::ParamValid::VALID || this->alert_cycles_ == 0) {
      return fail("AlertCycles");
    }
  }

  // --- Momentum management and disturbance feedforward (§8.5) --------------
  polaris::gnc::MomentumConfig momentum;
  polaris::gnc::MtqDesatConfig desat;
  polaris::gnc::DisturbanceObserverConfig observer;
  POLARIS_GET(this->wheel_inertia_kgm2_, paramGet_WheelInertiaKgm2, "WheelInertiaKgm2");
  POLARIS_GET(momentum.desat_enter_nms, paramGet_MomentumDesatEnterNms, "MomentumDesatEnterNms");
  POLARIS_GET(momentum.desat_exit_nms, paramGet_MomentumDesatExitNms, "MomentumDesatExitNms");
  POLARIS_GET(momentum.envelope_nms, paramGet_MomentumEnvelopeNms, "MomentumEnvelopeNms");
  POLARIS_GET(this->wheel_capacity_nms_, paramGet_WheelCapacityNms, "WheelCapacityNms");
  if (!(this->wheel_capacity_nms_ > 0.0)) {
    return fail("WheelCapacityNms must be positive");
  }
  POLARIS_GET(desat.gain_per_s, paramGet_DesatGainPerSec, "DesatGainPerSec");
  POLARIS_GET(observer.tau_s, paramGet_ObserverTauSec, "ObserverTauSec");
  POLARIS_GET(this->disturbance_budget_nm_, paramGet_DisturbanceBudgetNm, "DisturbanceBudgetNm");
  POLARIS_GET(observer.anomaly_clear_nm, paramGet_DisturbanceClearNm, "DisturbanceClearNm");
  {
    Fw::ParamValid v = Fw::ParamValid::INVALID;
    momentum.desat_confirm_cycles = this->paramGet_MomentumDesatConfirmCycles(v);
    if (v != Fw::ParamValid::VALID) {
      return fail("MomentumDesatConfirmCycles");
    }
    observer.anomaly_cycles = this->paramGet_DisturbanceAnomalyCycles(v);
    if (v != Fw::ParamValid::VALID || observer.anomaly_cycles == 0) {
      return fail("DisturbanceAnomalyCycles");
    }
    const U8 model_enable = this->paramGet_FeedforwardModelEnable(v);
    if (v != Fw::ParamValid::VALID || model_enable > 1) {
      return fail("FeedforwardModelEnable");
    }
    this->feedforward_model_ = model_enable != 0;
    const U8 observer_enable = this->paramGet_FeedforwardObserverEnable(v);
    if (v != Fw::ParamValid::VALID || observer_enable > 1) {
      return fail("FeedforwardObserverEnable");
    }
    this->feedforward_observer_ = observer_enable != 0;
    // The SITL/bench override (see setFeedforwardAtStartup) is applied *after*
    // the parameters are read and validated, so a run with feedforward forced off
    // still refuses a configuration whose feedforward tuning is missing — the
    // experiment must fly the same vehicle, not a differently-validated one.
    if (this->feedforward_override_) {
      this->feedforward_model_ = this->feedforward_model_override_;
      this->feedforward_observer_ = this->feedforward_observer_override_;
    }

    const Vec3F64 target = this->paramGet_MomentumTargetBody(v);
    if (v != Fw::ParamValid::VALID) {
      return fail("MomentumTargetBody");
    }
    momentum.target_nms = Eigen::Vector3d(target[0], target[1], target[2]);
    const Vec3F64 inertia = this->paramGet_InertiaBodyKgm2(v);
    if (v != Fw::ParamValid::VALID) {
      return fail("InertiaBodyKgm2");
    }
    this->inertia_diag_kgm2_ = Eigen::Vector3d(inertia[0], inertia[1], inertia[2]);
    if (!this->inertia_diag_kgm2_.allFinite() || this->inertia_diag_kgm2_.minCoeff() <= 0.0) {
      return fail("InertiaBodyKgm2 must be positive");
    }
    const Vec3F64 residual = this->paramGet_ResidualDipoleAm2(v);
    if (v != Fw::ParamValid::VALID) {
      return fail("ResidualDipoleAm2");
    }
    this->residual_dipole_am2_ = Eigen::Vector3d(residual[0], residual[1], residual[2]);
    if (!this->residual_dipole_am2_.allFinite()) {
      return fail("ResidualDipoleAm2 must be finite");
    }
  }
  momentum.wheel_count = alloc.wheel_count;
  momentum.rotor_inertia_kgm2 = this->wheel_inertia_kgm2_;
  for (U32 i = 0; i < this->wheel_count_; ++i) {
    momentum.spin_axes.col(static_cast<Eigen::Index>(i)) = this->wheel_axes_[i];
  }
  // The desaturation runs on the same duty cycle B-dot does — one schedule, one
  // divisor — and the observer's largest step is the PID's, because both are
  // asking "how long since the last cycle of this same rate group".
  desat.duty_factor = this->mtq_duty_factor_;
  observer.max_dt_s = pid.max_dt_s;
  observer.anomaly_torque_nm = this->disturbance_budget_nm_;
  this->momentum_envelope_nms_ = momentum.envelope_nms;

  // --- Wheel-speed bias servo (§8.5; REQ-ACTL-012) ---------------------------
  polaris::gnc::RwBiasConfig bias;
  bias.wheel_count = alloc.wheel_count;
  bias.axes = alloc.axes;
  POLARIS_GET(bias.gain_per_s, paramGet_WheelBiasGainPerS, "WheelBiasGainPerS");
  POLARIS_GET(bias.max_torque_nm, paramGet_WheelBiasMaxTorqueNm, "WheelBiasMaxTorqueNm");
  {
    Fw::ParamValid v = Fw::ParamValid::INVALID;
    const F64PerUnit pattern = this->paramGet_WheelBiasNms(v);
    if (v != Fw::ParamValid::VALID) {
      return fail("WheelBiasNms");
    }
    for (U32 i = 0; i < this->wheel_count_; ++i) {
      bias.bias_nms[i] = pattern[i];
      // A bias past the wheel's own capacity is a mis-set pattern, not a trim.
      if (std::abs(pattern[i]) >= this->wheel_capacity_nms_) {
        return fail("WheelBiasNms exceeds WheelCapacityNms");
      }
    }
    // The SITL/bench override (setWheelBiasAtStartup) is applied after the
    // validation, so a bias-off experiment still flies a validated vehicle.
    if (this->wheel_bias_override_ && !this->wheel_bias_override_enable_) {
      for (U32 i = 0; i < this->wheel_count_; ++i) {
        bias.bias_nms[i] = 0.0;
      }
    }
  }
  if (!bias.isValid()) {
    return fail("WheelBias servo config");
  }

#undef POLARIS_GET

  // Range gates that span parameters, and therefore belong here rather than in
  // any one library config.
  if (!(this->control_period_s_ > 0.0) || !(this->max_estimate_age_s_ > 0.0) ||
      !(this->max_att_sigma_rad_ > 0.0)) {
    return fail("ControlPeriodSec/MaxEstimateAgeSec/MaxAttSigmaRad must be positive");
  }
  if (!(this->mtq_settle_s_ >= 0.0) || !(this->mtq_stuck_residual_t_ > 0.0)) {
    return fail("MtqSettleSec/MtqStuckResidualT out of range");
  }
  if (!(this->mtq_window_tolerance_s_ >= 0.0) ||
      this->mtq_window_tolerance_s_ >= this->control_period_s_) {
    return fail("MtqWindowToleranceSec out of range");
  }
  // The quiet window must exist: on-window plus settle time has to leave room
  // inside the control period, or there is no instant at which a magnetometer
  // sample is a measurement of the geomagnetic field.
  if (this->mtq_duty_factor_ * this->control_period_s_ + this->mtq_settle_s_ >=
      this->control_period_s_) {
    return fail("MtqDutyFactor and MtqSettleSec leave no quiet window");
  }
  if (!bdot.isValid() || !std::isfinite(this->bdot_max_dipole_am2_) ||
      !(this->bdot_max_dipole_am2_ > 0.0)) {
    return fail("B-dot tuning out of range");
  }
  if (!hysteresis.isValid()) {
    return fail("Detumble rate hysteresis out of range");
  }
  if (!pid.isValid()) {
    return fail("PID tuning out of range");
  }
  if (!alloc.isValid()) {
    return fail("Wheel array is degenerate or its limits are out of range");
  }
  if (!friction.isValid()) {
    return fail("Wheel friction model out of range (coefficients, deadband or scale)");
  }
  if (!momentum.isValid()) {
    return fail("Momentum tuning out of range (thresholds, envelope or wheel inertia)");
  }
  if (!desat.isValid()) {
    return fail("DesatGainPerSec out of range");
  }
  if (!observer.isValid()) {
    return fail("Disturbance observer tuning out of range");
  }

  this->pid_max_torque_nm_ = pid.max_torque_nm;
  this->bdot_ = polaris::gnc::BdotController(bdot);
  this->pid_ = polaris::gnc::AttitudePid(pid);
  this->allocator_ = polaris::gnc::RwAllocator(alloc);
  this->friction_ = polaris::gnc::RwFrictionCompensator(friction);

  this->bias_ = polaris::gnc::RwBiasServo(bias);
  {
    bool engaged = this->bias_.nullDimension() > 0 && bias.gain_per_s > 0.0;
    double effective0 = this->bias_.effectiveBiasNms(0);
    bool any = false;
    for (U32 i = 0; i < this->wheel_count_; ++i) {
      any = any || this->bias_.effectiveBiasNms(static_cast<int>(i)) != 0.0;
    }
    if (engaged && any) {
      this->log_ACTIVITY_HI_WheelBiasEngaged(effective0,
                                             static_cast<U8>(this->bias_.nullDimension()));
    }
  }
  this->rate_hysteresis_ = polaris::gnc::RateHysteresis(hysteresis);
  this->momentum_ = polaris::gnc::MomentumManager(momentum);
  this->observer_ = polaris::gnc::DisturbanceObserver(observer);
  this->desat_config_ = desat;

  if (this->alloc_method_ == polaris::gnc::RwAllocationMethod::kMinMax &&
      !this->allocator_.supportsMinMax()) {
    if (!this->alloc_fallback_alerted_) {
      this->log_WARNING_LO_AllocationFallback(this->wheel_count_);
      this->alloc_fallback_alerted_ = true;
    }
    this->alloc_method_ = polaris::gnc::RwAllocationMethod::kMinNorm;
  } else {
    this->alloc_fallback_alerted_ = false;
  }

  this->configured_ = true;
  this->config_alerted_ = false;
  return true;
}

void AttitudeController ::parameterUpdated(FwPrmIdType id) {
  static_cast<void>(id);
  // Any tuning change rebuilds all three laws and drops the accumulated state:
  // an integrator filled under the old gains means nothing under the new ones.
  (void)this->applyParameters();
  this->pid_.reset();
  this->bdot_.reset();
  this->rate_hysteresis_.reset();
  this->momentum_.reset();
  // The observer's history is a filtered torque built under the old inertia and
  // the old model; a tuning change makes it a number about a different vehicle.
  this->observer_.reset();
}

// ----------------------------------------------------------------------
// Commands
// ----------------------------------------------------------------------

void AttitudeController ::setMode(CtrlMode::T next) {
  if (next == this->mode_) {
    return;
  }
  this->log_ACTIVITY_HI_ModeChanged(CtrlMode(this->mode_), CtrlMode(next));
  this->mode_ = next;
  // Every mode entry starts from a clean law: an integrator or a stored field
  // sample from the last time this mode ran describes a vehicle that has since
  // moved. The momentum manager and the disturbance observer are deliberately
  // **not** reset: stored wheel momentum and the ambient disturbance torque are
  // properties of the vehicle and its orbit, not of the mode, and throwing away a
  // converged torque estimate at every mode change would leave the observer
  // permanently re-converging on a vehicle that changes mode.
  this->pid_.reset();
  this->bdot_.reset();
  this->rate_hysteresis_.reset();
  this->refusal_streak_ = 0;
  this->tlmWrite_CtrlModeTlm(CtrlMode(this->mode_));
}

bool AttitudeController ::tryEnterMode(CtrlMode::T requested, CtrlRefusal::T& reason) {
  // IDLE is unconditional on purpose: the way out of a bad state must never
  // itself have a precondition.
  if (requested == CtrlMode::IDLE) {
    this->setMode(requested);
    return true;
  }
  if (!this->configured_) {
    reason = CtrlRefusal::NOT_CONFIGURED;
    return false;
  }
  if (!this->estimateUsable(this->currentTaiNs(), requested, reason)) {
    return false;
  }
  if (requested == CtrlMode::POINT && !this->have_target_) {
    reason = CtrlRefusal::NO_TARGET;
    return false;
  }
  // TRACK is refused unless the guidance is solving *now*. Entering it on the
  // strength of a command alone would put the vehicle in a mode whose first act
  // is to refuse every cycle, which reads to an operator as a controller fault
  // rather than as a pointing command that cannot be met.
  if (requested == CtrlMode::TRACK && !this->guidance_valid_) {
    reason = CtrlRefusal::NO_GUIDANCE;
    return false;
  }
  if (requested == CtrlMode::DETUMBLE && !this->estimate_.get_magFieldValid()) {
    reason = CtrlRefusal::NO_FIELD;
    return false;
  }
  this->setMode(requested);
  return true;
}

void AttitudeController ::CTRL_MODE_SET_cmdHandler(FwOpcodeType opCode, U32 cmdSeq, CtrlMode mode) {
  CtrlRefusal::T reason = CtrlRefusal::NOT_CONFIGURED;
  if (!this->tryEnterMode(mode.e, reason)) {
    this->log_WARNING_LO_ModeRefused(CtrlMode(mode.e), CtrlRefusal(reason));
    this->cmdResponse_out(opCode, cmdSeq, Fw::CmdResponse::EXECUTION_ERROR);
    return;
  }
  this->cmdResponse_out(opCode, cmdSeq, Fw::CmdResponse::OK);
}

void AttitudeController ::CTRL_SET_TARGET_Q_cmdHandler(FwOpcodeType opCode, U32 cmdSeq, F64 q0,
                                                       F64 q1, F64 q2, F64 q3) {
  polaris::math::Quaternion q(q0, q1, q2, q3);
  if (!q.isFinite() || !q.normalize()) {
    this->log_WARNING_LO_TargetRejected();
    this->cmdResponse_out(opCode, cmdSeq, Fw::CmdResponse::VALIDATION_ERROR);
    return;
  }
  q = q.canonical();
  this->target_ = polaris::math::Quat<Body, ECI>(q);
  this->have_target_ = true;
  // A new target is a new error to accumulate against; carrying the integrator
  // across would apply history built against an attitude nobody is asking for.
  this->pid_.reset();
  this->log_ACTIVITY_HI_TargetSet(q.w(), q.x(), q.y(), q.z());
  this->cmdResponse_out(opCode, cmdSeq, Fw::CmdResponse::OK);
}

void AttitudeController ::CTRL_RESET_cmdHandler(FwOpcodeType opCode, U32 cmdSeq) {
  this->pid_.reset();
  this->bdot_.reset();
  this->rate_hysteresis_.reset();
  this->momentum_.reset();
  this->observer_.reset();
  this->desat_active_ = false;
  this->envelope_alerted_ = false;
  this->anomaly_alerted_ = false;
  this->stuck_mask_ = 0;
  this->stuck_candidate_mask_ = 0;
  this->stuck_streak_ = 0;
  this->clear_streak_ = 0;
  this->stuck_confirmed_ = false;
  this->refusal_streak_ = 0;
  this->momentum_refusal_streak_ = 0;
  this->have_last_cycle_ = false;
  this->setMode(CtrlMode::IDLE);
  (void)this->applyParameters();
  this->log_ACTIVITY_HI_ControllerReset();
  this->cmdResponse_out(opCode, cmdSeq, Fw::CmdResponse::OK);
}

void AttitudeController ::CTRL_DESAT_cmdHandler(FwOpcodeType opCode, U32 cmdSeq,
                                                DesatOverride action) {
  this->desat_override_ = action.e;
  this->log_ACTIVITY_HI_DesatOverrideChanged(DesatOverride(this->desat_override_));
  this->tlmWrite_DesatOverrideTlm(DesatOverride(this->desat_override_));
  this->cmdResponse_out(opCode, cmdSeq, Fw::CmdResponse::OK);
}

// ----------------------------------------------------------------------
// Cycle
// ----------------------------------------------------------------------

bool AttitudeController ::alertDue(U32 streak) const {
  return this->alert_cycles_ > 0 && (streak == 1 || (streak % this->alert_cycles_) == 0);
}

bool AttitudeController ::estimateUsable(I64 nowNs, CtrlMode::T mode,
                                         CtrlRefusal::T& reason) const {
  if (!this->have_estimate_) {
    reason = CtrlRefusal::NO_ESTIMATE;
    return false;
  }
  const I64 age_ns = nowNs - this->estimate_.get_epochTaiNs();
  const double age_s = static_cast<double>(age_ns < 0 ? -age_ns : age_ns) / 1.0e9;
  if (age_s > this->max_estimate_age_s_) {
    reason = CtrlRefusal::NO_ESTIMATE;
    return false;
  }
  // DETUMBLE needs no attitude at all — that is the whole point of B-dot, and
  // requiring one would make the recovery mode depend on the thing that is
  // broken. It needs the rate only for the completion predicate, which is
  // reported rather than acted on.
  if (mode == CtrlMode::DETUMBLE) {
    return true;
  }
  if (!this->estimate_.get_attitudeValid()) {
    reason = CtrlRefusal::ATTITUDE_INVALID;
    return false;
  }
  if (!this->estimate_.get_rateValid()) {
    reason = CtrlRefusal::RATE_INVALID;
    return false;
  }
  const Vec3F64& cov = this->estimate_.get_attCovDiagRad2();
  double worst = 0.0;
  for (U32 i = 0; i < 3; ++i) {
    if (!std::isfinite(cov[i]) || cov[i] < 0.0) {
      reason = CtrlRefusal::QUALITY_FLOOR;
      return false;
    }
    worst = cov[i] > worst ? cov[i] : worst;
  }
  if (std::sqrt(worst) > this->max_att_sigma_rad_) {
    reason = CtrlRefusal::QUALITY_FLOOR;
    return false;
  }
  return true;
}

bool AttitudeController ::runDetumble(pm::Vec3<Body>& dipole, CtrlRefusal::T& reason) {
  this->commanded_mask_ = 0;
  if (!this->estimate_.get_magFieldValid()) {
    // No admissible quiet-window sample this cycle. The stored sample is kept:
    // the B-dot law's own interval gate decides whether the next one is still
    // comparable, which is the one place that judgement belongs.
    reason = CtrlRefusal::NO_FIELD;
    return false;
  }
  polaris::gnc::BdotResult result;
  if (!this->bdot_.update(fromVec3F64(this->estimate_.get_magFieldBody()),
                          this->estimate_.get_magFieldTimeTagNs(), result)) {
    // A sample arrived but no derivative could be formed from it. Reported
    // separately from NO_FIELD: an operator seeing a persistent refusal needs to
    // know whether the interlock/sensor is withholding samples or whether the
    // samples are arriving at a spacing the law cannot use.
    reason = CtrlRefusal::FIELD_INTERVAL;
    return false;
  }

  if (!this->clampDipoleToRods(result.dipole_am2, dipole)) {
    // Distinct from NO_FIELD: a sample arrived and a derivative was formed, and
    // the arithmetic still produced something uncommandable. That is a numerics
    // fault, not a sensing one, and the two want different responses.
    reason = CtrlRefusal::BAD_COMMAND;
    return false;
  }
  this->noteMagneticTorque(dipole, fromVec3F64(this->estimate_.get_magFieldBody()));
  return true;
}

bool AttitudeController ::clampDipoleToRods(const pm::Vec3<Body>& demand,
                                            pm::Vec3<Body>& applied_out) {
  // Resolve the body dipole onto the rod triad, clamp each rod, and re-expand.
  // **The only clamp on the magnetic path**, and shared by B-dot and
  // desaturation so the two cannot saturate differently: the laws return an
  // unclamped demand because a rated moment is a limit in the *rod* basis, and
  // clamping in body axes too would discard authority the rods still have
  // whenever the triad is not body-aligned. Componentwise (rather than a
  // direction-preserving scale) is what keeps both laws dissipative — the sign of
  // every term of the energy/momentum rate survives it, which is the argument
  // lib/gnc/bdot.hpp and lib/gnc/momentum.hpp each carry for their own law, and
  // which holds only because the triad is orthonormal (applyParameters refuses
  // any other).
  this->commanded_mask_ = 0;
  Eigen::Vector3d applied = Eigen::Vector3d::Zero();
  bool saturated = false;
  double worst = 0.0;
  for (U32 i = 0; i < kRodCount; ++i) {
    double strength = demand.eigen().dot(this->rod_axes_[i]);
    if (strength > this->bdot_max_dipole_am2_) {
      strength = this->bdot_max_dipole_am2_;
      saturated = true;
    } else if (strength < -this->bdot_max_dipole_am2_) {
      strength = -this->bdot_max_dipole_am2_;
      saturated = true;
    }
    if (std::abs(strength) > 0.0) {
      this->commanded_mask_ |= (1u << i);
    }
    worst = std::abs(strength) > worst ? std::abs(strength) : worst;
    applied += strength * this->rod_axes_[i];
  }
  if (!applied.allFinite()) {
    this->commanded_mask_ = 0;
    return false;
  }

  if (saturated) {
    ++this->dipole_saturation_streak_;
    if (this->alertDue(this->dipole_saturation_streak_)) {
      this->log_WARNING_LO_DipoleSaturated(worst, this->bdot_max_dipole_am2_);
    }
  } else {
    this->dipole_saturation_streak_ = 0;
  }

  applied_out = pm::Vec3<Body>(applied);
  return true;
}

bool AttitudeController ::runPoint(double dtSec, double* wheelTorque, CtrlRefusal::T& reason) {
  const QuatF64& q = this->estimate_.get_qBodyEci();
  polaris::math::Quaternion estimate(q[0], q[1], q[2], q[3]);
  if (!estimate.isFinite() || !estimate.normalize()) {
    reason = CtrlRefusal::ATTITUDE_INVALID;
    return false;
  }

  // Disturbance feedforward (§8.5 tiers 1-2), computed earlier this cycle by
  // `updateDisturbance`, enters the demand *inside* the PID — ahead of the
  // saturation test and the anti-windup decision. Added afterwards it could
  // command torque the vehicle does not have; hidden from the integrator it
  // would fight the integrator that is cancelling the same disturbance.
  const pm::Vec3<Body>& feedforward = this->feedforward_nm_;

  // POINT holds a fixed inertial attitude at zero commanded rate; TRACK follows
  // whatever the guidance streams, including its feedforward rate. One control
  // law, two sources — the PID already takes a target rate, so tracking is a
  // change of argument rather than a second controller.
  const bool tracking = this->mode_ == CtrlMode::TRACK;
  if (tracking && !this->guidance_valid_) {
    reason = CtrlRefusal::NO_GUIDANCE;
    return false;
  }
  const polaris::math::Quat<Body, ECI>& commanded_att =
      tracking ? this->guidance_target_ : this->target_;
  const pm::Vec3<Body> commanded_rate =
      tracking ? this->guidance_rate_ : pm::Vec3<Body>(Eigen::Vector3d::Zero());

  polaris::gnc::AttitudePidResult pid;
  if (!this->pid_.update(polaris::math::Quat<Body, ECI>(estimate),
                         fromVec3F64(this->estimate_.get_bodyRateRadps()), commanded_att,
                         commanded_rate, feedforward, dtSec, pid)) {
    reason = CtrlRefusal::ATTITUDE_INVALID;
    return false;
  }

  polaris::gnc::RwAllocationResult alloc;
  if (!this->allocator_.allocate(pid.torque_nm, this->alloc_method_, alloc)) {
    reason = CtrlRefusal::ALLOCATION;
    return false;
  }
  for (U32 i = 0; i < this->wheel_count_; ++i) {
    wheelTorque[i] = alloc.torque_nm[i];
  }

  // Wheel-speed bias servo (§8.5; REQ-ACTL-012): a null-space torque added on
  // top of the allocation, so it changes wheel speeds and no body torque. It
  // needs this cycle's tachometers; a stale wheel skips the trim rather than
  // steering on a guessed momentum.
  {
    double bias_peak = 0.0;
    bool fresh = true;
    double h_w[polaris::gnc::kMaxWheels] = {};
    for (U32 i = 0; i < this->wheel_count_; ++i) {
      fresh = fresh && this->wheel_speed_fresh_[i];
      h_w[i] = this->wheel_inertia_kgm2_ * this->wheel_speed_radps_[i];
    }
    polaris::gnc::RwBiasResult bias;
    if (fresh && this->bias_.update(h_w, bias) && bias.active) {
      for (U32 i = 0; i < this->wheel_count_; ++i) {
        wheelTorque[i] += bias.torque_nm[i];
        bias_peak = std::max(bias_peak, std::abs(bias.torque_nm[i]));
      }
    }
    this->tlmWrite_WheelBiasTorqueNm(bias_peak);
  }

  // Drive friction feedforward (§8.5; REQ-ACTL-010), applied **after** the
  // allocation and **before** the command so the *net* rotor torque is the one
  // the allocation asked for. It belongs here and not inside the allocation
  // because friction is a property of a drive, not of the array geometry:
  // folding it in would make the pseudo-inverse a function of wheel speed.
  //
  // It is also not the disturbance feedforward that enters the PID upstream, and
  // deliberately so. That term is a *body* torque the control law must supply;
  // this one is a *per-wheel* torque that never reaches the body at all — it is
  // spent inside the bearings — so adding it to the demand the PID saturates and
  // anti-windup against would charge the control law for authority it never got.
  if (this->friction_enabled_) {
    polaris::gnc::RwFrictionResult ff;
    // The tachometer verdict is this cycle's, from `updateMomentum`: valid *and*
    // fresh. A wheel without one is passed through uncompensated rather than
    // given a guessed sign — half the time that guess is a torque pushing the
    // wrong way, which is worse than the vehicle this feature exists to improve.
    if (this->friction_.compensate(wheelTorque, this->wheel_speed_radps_, this->wheel_speed_fresh_,
                                   ff)) {
      for (U32 i = 0; i < this->wheel_count_; ++i) {
        wheelTorque[i] = ff.torque_nm[i];
        if (ff.compensated[i]) {
          this->friction_nm_[i] = ff.compensation_nm[i];
        }
      }
    }
    // A refused compensation is not a refused control cycle: the uncompensated
    // demand is exactly what the vehicle flew before this feature existed, so
    // falling through with `wheelTorque` untouched degrades to that rather than
    // dropping the pointing command. The NaN telemetry says it happened.
  }

  if (pid.saturated || alloc.saturated) {
    ++this->saturation_streak_;
    if (this->alertDue(this->saturation_streak_)) {
      this->log_WARNING_LO_TorqueSaturated(pid.torque_nm.eigen().norm(), this->pid_max_torque_nm_,
                                           alloc.scale);
    }
  } else {
    this->saturation_streak_ = 0;
  }

  this->tlmWrite_TorqueCmd(toVec3F64(pid.torque_nm.eigen()));
  this->tlmWrite_PointingErrorRad(pid.error_angle_rad);
  this->tlmWrite_MaxWheelTorque(alloc.max_wheel_torque_nm);
  this->tlmWrite_AllocScale(alloc.scale);
  return true;
}

void AttitudeController ::runStuckOnMonitor() {
  // The comparison is on field **magnitudes** against the onboard IGRF, which is
  // attitude-free. Judging a magnetometer through an attitude that magnetometer
  // helped build is the circularity that latches out the healthy unit (P52), and
  // the near-field geometry of a rod is not knowledge the flight software has —
  // so magnitude is both the honest signal and the only one available.
  if (!this->estimate_.get_magModelValid() || !this->estimate_.get_magRawValid()) {
    // No comparison this cycle. Neither streak advances: absence of evidence is
    // not evidence of a stuck rod, and it is not evidence of a healthy one
    // either, so the latch is held exactly where it is.
    this->tlmWrite_MagResidualT(kNoValue);
    return;
  }
  // The **raw** magnitude, not the voted field: a rod stuck on puts hundreds of
  // microtesla on the sensor, which the estimator's §8.2 plausibility band
  // rejects before the vote ever runs — so a monitor reading the voted field
  // would go blind at exactly the disturbance it exists to name. Being out of
  // band is evidence here, not a reason to look away.
  const double measured = this->estimate_.get_magRawMagnitudeT();
  const double modelled = this->estimate_.get_magModelMagnitudeT();
  const double residual = std::abs(measured - modelled);
  this->tlmWrite_MagResidualT(residual);
  if (!std::isfinite(residual)) {
    return;
  }

  if (residual > this->mtq_stuck_residual_t_) {
    this->clear_streak_ = 0;
    ++this->stuck_streak_;
    // The candidate set is the rods that carried a command in the period this
    // sample was taken in. It is accumulated rather than replaced so a rod that
    // was driven earlier in the confirmation run stays a candidate.
    this->stuck_candidate_mask_ |= this->commanded_mask_;
    if (!this->stuck_confirmed_ && this->stuck_streak_ >= this->mtq_stuck_confirm_cycles_) {
      this->stuck_confirmed_ = true;
      // Decisive only when exactly one rod is a candidate. Three rods driven
      // together are three equally good explanations of one residual, and naming
      // one of them would be a guess; resolving that needs a commanded isolation
      // sweep, which is the Phase-7 state machine's recovery action.
      U8 unit = 255;
      StuckAttribution::T attribution = StuckAttribution::AMBIGUOUS;
      U32 candidates = 0;
      for (U32 i = 0; i < kRodCount; ++i) {
        if ((this->stuck_candidate_mask_ & (1u << i)) != 0u) {
          ++candidates;
          unit = static_cast<U8>(i);
        }
      }
      if (candidates == 1) {
        attribution = StuckAttribution::DECISIVE;
        this->stuck_mask_ = this->stuck_candidate_mask_;
      } else {
        unit = 255;
        this->stuck_mask_ = 0;
      }
      this->log_WARNING_HI_MtqStuckOn(unit, StuckAttribution(attribution),
                                      this->stuck_candidate_mask_, residual, this->stuck_streak_);
    }
    return;
  }

  this->stuck_streak_ = 0;
  if (!this->stuck_confirmed_) {
    this->stuck_candidate_mask_ = 0;
    return;
  }
  // Re-admission on **the criterion that excluded it**: the same residual test,
  // back under the same threshold, for MtqStuckClearCycles consecutive quiet
  // windows. An exclusion whose release test differs from its trigger is a life
  // sentence dressed as a policy.
  ++this->clear_streak_;
  if (this->clear_streak_ >= this->mtq_stuck_clear_cycles_) {
    this->log_ACTIVITY_HI_MtqStuckCleared(this->stuck_candidate_mask_, this->clear_streak_);
    this->stuck_confirmed_ = false;
    this->stuck_mask_ = 0;
    this->stuck_candidate_mask_ = 0;
    this->clear_streak_ = 0;
  }
}

void AttitudeController ::commandActuators(I64 nowNs, const pm::Vec3<Body>& dipole,
                                           const double* wheelTorque, bool rodsActive) {
  WheelTorqueSet wheels;  // default-constructed: all zero
  for (U32 i = 0; i < this->wheel_count_ && i < WheelTorqueSet::SIZE; ++i) {
    wheels[i] = std::isfinite(wheelTorque[i]) ? wheelTorque[i] : 0.0;
  }

  MtqDipoleSet rods;  // default-constructed: all zero
  const Eigen::Vector3d m = dipole.eigen();
  if (rodsActive && m.allFinite()) {
    for (U32 i = 0; i < kRodCount && i < MtqDipoleSet::SIZE; ++i) {
      // Rod i carries only its own axis's share; the three together reproduce
      // the commanded body dipole exactly, because the triad is orthogonal.
      const Eigen::Vector3d rod = m.dot(this->rod_axes_[i]) * this->rod_axes_[i];
      rods[i][0] = rod[0];
      rods[i][1] = rod[1];
      rods[i][2] = rod[2];
    }
  }

  // §7 duty-cycle schedule for the period these commands apply over. A cycle
  // that commands no dipole schedules a zero-length on-window, which makes the
  // whole period quiet — the honest schedule, and the one that lets the stuck-on
  // monitor see an undisturbed field.
  //
  // **A schedule is published only when there is one to publish.** With no valid
  // control period there is no window at all, and emitting `quietStart ==
  // quietEnd == now` would be a window no sample can be inside: a magnetometer
  // tag is never bit-exactly the cycle epoch, so an unconfigured controller —
  // one commanding nothing — would silently reject every sample the estimator
  // takes. Staying silent is the permissive state by construction: the
  // estimator's `have_mtq_schedule_` then still means "nothing has ever driven a
  // rod", which is exactly true of a controller that cannot command one.
  const double period_s = this->control_period_s_;
  const bool driving = rodsActive && m.squaredNorm() > 0.0;
  const double on_window_s = driving ? this->mtq_duty_factor_ * period_s : 0.0;
  const I64 period_ns = static_cast<I64>(period_s * 1.0e9);
  const I64 on_ns = static_cast<I64>(on_window_s * 1.0e9);
  const I64 settle_ns = static_cast<I64>(this->mtq_settle_s_ * 1.0e9);
  // Late-sample tolerance, applied to the window's **end** only. From the
  // on-window's end until the *next* period's on-window begins the rods are off,
  // so a sample arriving a little after the nominal boundary is still a
  // measurement of the geomagnetic field; extending the *start* would do the
  // opposite and admit a dirty one.
  const I64 tolerance_ns = static_cast<I64>(this->mtq_window_tolerance_s_ * 1.0e9);
  const bool publishable = this->configured_ && period_ns > 0;

  if (this->isConnected_wheelCmdOut_OutputPort(0)) {
    this->wheelCmdOut_out(0, wheels);
  }
  if (this->isConnected_mtqCmdOut_OutputPort(0)) {
    this->mtqCmdOut_out(0, rods, on_window_s);
  }
  if (publishable && this->isConnected_mtqActuationOut_OutputPort(0)) {
    MtqActuation schedule;
    schedule.set_periodStartTaiNs(nowNs);
    schedule.set_onWindowEndTaiNs(nowNs + on_ns);
    schedule.set_quietStartTaiNs(nowNs + on_ns + (driving ? settle_ns : 0));
    schedule.set_quietEndTaiNs(nowNs + period_ns + tolerance_ns);
    schedule.set_commandedMask(driving ? this->commanded_mask_ : 0u);
    schedule.set_interlockHealthy(!this->stuck_confirmed_);
    this->mtqActuationOut_out(0, schedule);
  }

  this->tlmWrite_DipoleCmd(toVec3F64(driving ? m : Eigen::Vector3d::Zero()));
  F64PerUnit wheel_tlm;
  for (U32 i = 0; i < F64PerUnit::SIZE; ++i) {
    wheel_tlm[i] = i < WheelTorqueSet::SIZE ? wheels[i] : 0.0;
  }
  this->tlmWrite_WheelTorque(wheel_tlm);
  F64PerUnit friction_tlm;
  for (U32 i = 0; i < F64PerUnit::SIZE; ++i) {
    friction_tlm[i] = i < static_cast<U32>(polaris::gnc::kMaxWheels)
                          ? this->friction_nm_[i]
                          : std::numeric_limits<F64>::quiet_NaN();
  }
  this->tlmWrite_WheelFrictionNm(friction_tlm);
  this->tlmWrite_MtqInterlockHealthy(!this->stuck_confirmed_);
  this->tlmWrite_MtqStuckMask(this->stuck_mask_);
}

void AttitudeController ::run_handler(FwIndexType portNum, U32 context) {
  static_cast<void>(portNum);
  static_cast<void>(context);
  const I64 nowNs = this->currentTaiNs();
  ++this->cycles_run_;

  // Nothing has been friction-compensated yet this cycle. Reset before the first
  // path that can publish, so a cycle that commands no wheel torque reports "no
  // compensation" rather than repeating the last cycle's number.
  for (U32 i = 0; i < polaris::gnc::kMaxWheels; ++i) {
    this->friction_nm_[i] = kNoValue;
  }

  // §7 timing check. The quiet window is computed from the **declared** control
  // period, so a real period that differs from it puts the window somewhere no
  // magnetometer sample lands — and the symptom on a vehicle is the
  // magnetometers appearing to fail, with nothing saying why. Compared against
  // the same tolerance the window's end carries, since that is the slack the
  // schedule was sized with.
  if (this->configured_ && this->have_last_run_) {
    const double measured_s = static_cast<double>(nowNs - this->last_run_ns_) / 1.0e9;
    if (std::abs(measured_s - this->control_period_s_) > this->mtq_window_tolerance_s_) {
      ++this->period_mismatch_streak_;
      if (this->alertDue(this->period_mismatch_streak_)) {
        this->log_WARNING_HI_CyclePeriodMismatch(measured_s, this->control_period_s_,
                                                 this->period_mismatch_streak_);
      }
    } else {
      this->period_mismatch_streak_ = 0;
    }
  }
  this->last_run_ns_ = nowNs;
  this->have_last_run_ = true;

  // A mode latched at startup (SITL/bench, see commandModeAtStartup) is retried
  // once per cycle until the estimate can support it, then dropped. Retrying is
  // what makes "command DETUMBLE at boot" mean "as soon as there is a field",
  // which is what the operator asked for; commanding once at setup would always
  // be refused, because no measurement has arrived yet. The *guards* are shared
  // with the command handler (tryEnterMode) rather than the command being
  // re-dispatched through cmdIn: `run` and the command handler are both guarded
  // ports on this passive component, so invoking one from inside the other would
  // re-enter the component mutex.
  if (this->pending_mode_ != 0) {
    CtrlRefusal::T pending_reason = CtrlRefusal::NOT_CONFIGURED;
    if (this->tryEnterMode(static_cast<CtrlMode::T>(this->pending_mode_), pending_reason)) {
      this->pending_mode_ = 0;
    }
  }

  if (!this->configured_ && !this->applyParameters()) {
    // Inert: IDLE, zero on every actuator, and a schedule that says the whole
    // period is quiet — a controller with no tuning must not leave the rods'
    // last command latched in the plant.
    this->setMode(CtrlMode::IDLE);
    const double zero_torque[polaris::gnc::kMaxWheels] = {};
    this->commandActuators(nowNs, pm::Vec3<Body>(Eigen::Vector3d::Zero()), zero_torque, false);
    ++this->cycles_refused_;
    this->tlmWrite_CtrlModeTlm(CtrlMode(this->mode_));
    this->tlmWrite_CyclesRefused(this->cycles_refused_);
    this->tlmWrite_CyclesRun(this->cycles_run_);
    return;
  }

  // The completion predicate the Phase-7 mode manager will read. Fed every cycle
  // an estimate is available, in every mode, so the answer does not depend on
  // which law happens to be running.
  if (this->have_estimate_ && this->estimate_.get_rateValid()) {
    const double rate_norm = fromVec3F64(this->estimate_.get_bodyRateRadps()).eigen().norm();
    (void)this->rate_hysteresis_.update(rate_norm);
    this->tlmWrite_RateNorm(rate_norm);
  } else {
    this->tlmWrite_RateNorm(kNoValue);
  }
  this->tlmWrite_DetumbleComplete(this->rate_hysteresis_.complete());

  // The stuck-on monitor runs in every mode, including IDLE — a rod stuck on
  // while nothing is commanding it is exactly the case worth catching, and it is
  // the one in which the residual is unambiguous.
  this->runStuckOnMonitor();

  // Momentum accounting and the disturbance observer likewise run in every mode:
  // the §9 envelope and momentum-anomaly monitors are vehicle-level fault
  // conditions, not POINT diagnostics, and an observer that only ran while
  // pointing would have to re-converge every time the vehicle did anything else.
  (void)this->updateMomentum(nowNs);

  double wheel_torque[polaris::gnc::kMaxWheels] = {};
  pm::Vec3<Body> dipole(Eigen::Vector3d::Zero());
  bool rods_active = false;
  bool ok = true;
  CtrlRefusal::T reason = CtrlRefusal::NOT_CONFIGURED;

  if (this->mode_ != CtrlMode::IDLE) {
    ok = this->estimateUsable(nowNs, this->mode_, reason);
  }

  // **Desaturation runs concurrently with POINT**: the wheels hold the attitude
  // and the rods dump what the wheels are holding, in the same cycle on
  // different actuators. `desatDue` excludes every mode but POINT, so B-dot's
  // ownership of the rods in DETUMBLE is never contested.
  //
  // It is decided **before** the pointing law, and that ordering is load-bearing
  // rather than tidy: the magnetic torque the rods are about to apply is a
  // disturbance to the pointing loop, it is known exactly and in advance, and
  // handing it to the PID as feedforward is the difference between the wheels
  // being told about it and the wheels discovering it as attitude error. A
  // desaturation cycle that produces no dipole (a refused law, a clamp that went
  // non-finite) simply leaves the rods off; it does not refuse the pointing
  // cycle, because pointing is not what failed.
  const bool desat_now = ok && this->desatDue();
  this->magnetic_torque_prev_nm_ = this->magnetic_torque_nm_;
  this->magnetic_torque_nm_ = pm::Vec3<Body>(Eigen::Vector3d::Zero());
  if (desat_now && this->runDesat(dipole)) {
    rods_active = true;
  } else if (!desat_now && this->mode_ == CtrlMode::POINT) {
    this->commanded_mask_ = 0;
  }

  // The disturbance chain, which now has this cycle's magnetic torque to model,
  // and which runs in every mode: the §9 envelope and momentum-anomaly monitors
  // are vehicle-level fault conditions, not POINT diagnostics, and an observer
  // that only ran while pointing would re-converge every time the vehicle did
  // anything else.
  this->updateDisturbance(nowNs);

  if (this->mode_ != CtrlMode::IDLE && ok) {
    if (this->mode_ == CtrlMode::DETUMBLE) {
      ok = this->runDetumble(dipole, reason);
      rods_active = ok;
    } else if (this->mode_ == CtrlMode::TRACK && !this->guidance_valid_) {
      // TRACK's precondition is a guidance solution *this cycle*, not a target
      // ever having been commanded. The two refusals are kept apart because the
      // operator's fix differs: NO_TARGET means uplink one, NO_GUIDANCE means
      // the one you uplinked cannot be met right now.
      ok = false;
      reason = CtrlRefusal::NO_GUIDANCE;
    } else if (this->mode_ == CtrlMode::POINT && !this->have_target_) {
      ok = false;
      reason = CtrlRefusal::NO_TARGET;
    } else {
      const double dt_s =
          this->have_last_cycle_ ? static_cast<double>(nowNs - this->last_cycle_ns_) / 1.0e9 : 0.0;
      ok = this->runPoint(dt_s, wheel_torque, reason);
    }
  }

  if (this->mode_ != CtrlMode::IDLE && !ok) {
    // A refused cycle commands zero rather than holding the last command: an
    // actuator nobody re-commands keeps driving, which is how a controller that
    // lost its estimate keeps torquing toward where the vehicle used to be.
    for (U32 i = 0; i < polaris::gnc::kMaxWheels; ++i) {
      wheel_torque[i] = 0.0;
      // The friction record follows the command it accompanied (the same rule
      // the magnetic-torque record below is written on): a cycle that commands
      // zero torque compensated nothing, whatever was computed before it failed.
      this->friction_nm_[i] = kNoValue;
    }
    dipole = pm::Vec3<Body>(Eigen::Vector3d::Zero());
    rods_active = false;
    this->commanded_mask_ = 0;
    // The recorded magnetic torque must describe what `commandActuators`
    // actually publishes, and this path publishes nothing: a desaturation
    // decided above and withdrawn here would otherwise still be subtracted from
    // the observer next cycle, and the §9 anomaly would fire on a torque the
    // vehicle never applied — its own inaction, sign-reversed.
    this->magnetic_torque_nm_ = pm::Vec3<Body>(Eigen::Vector3d::Zero());
    ++this->cycles_refused_;
    if (reason == this->refusal_reason_) {
      ++this->refusal_streak_;
    } else {
      this->refusal_reason_ = reason;
      this->refusal_streak_ = 1;
    }
    if (this->alertDue(this->refusal_streak_)) {
      this->log_WARNING_LO_ControlRefused(CtrlMode(this->mode_), CtrlRefusal(reason),
                                          this->refusal_streak_);
    }
  } else {
    this->refusal_streak_ = 0;
    this->last_cycle_ns_ = nowNs;
    this->have_last_cycle_ = true;
  }

  // The desat edge is reported from what is **commanded**, not from what was
  // decided, which is why it sits below the refusal path: `desat_now` upstream
  // is a decision, and both a refused desaturation law and a pointing refusal
  // that zeroes the dipole withdraw the command after the decision. An event
  // whose purpose is to timestamp magnetic activity for an operator correlating
  // a payload anomaly must never report a window in which no rod was driven.
  const bool desat_commanded = rods_active && this->mode_ == CtrlMode::POINT;
  if (desat_commanded != this->desat_active_) {
    const double momentum_nms =
        this->momentum_state_.valid ? this->momentum_state_.error_norm_nms : kNoValue;
    if (desat_commanded) {
      this->log_ACTIVITY_HI_DesatEngaged(momentum_nms, this->momentum_.config().desat_enter_nms,
                                         this->desat_override_ == DesatOverride::FORCE);
    } else {
      this->log_ACTIVITY_HI_DesatDisengaged(momentum_nms);
    }
    this->desat_active_ = desat_commanded;
  }
  this->tlmWrite_DesatActive(this->desat_active_);
  this->tlmWrite_DesatOverrideTlm(DesatOverride(this->desat_override_));

  if (this->mode_ == CtrlMode::IDLE) {
    this->tlmWrite_TorqueCmd(toVec3F64(Eigen::Vector3d::Zero()));
    this->tlmWrite_PointingErrorRad(kNoValue);
    this->tlmWrite_MaxWheelTorque(0.0);
    this->tlmWrite_AllocScale(1.0);
    this->commanded_mask_ = 0;
  }

  this->commandActuators(nowNs, dipole, wheel_torque, rods_active);
  this->tlmWrite_CtrlModeTlm(CtrlMode(this->mode_));
  this->tlmWrite_CyclesRefused(this->cycles_refused_);
  this->tlmWrite_CyclesRun(this->cycles_run_);
}

}  // namespace flight
