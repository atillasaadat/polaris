// ======================================================================
// \title  BurnExecutor.cpp
// \brief  Finite-burn executor implementation (design doc §17, §8.3)
// ======================================================================

#include "flight/PolarisFsw/BurnExecutor/BurnExecutor.hpp"

#include <algorithm>
#include <cmath>

#include "Fw/Log/LogString.hpp"
#include "math/quaternion.hpp"

namespace flight {

namespace {
constexpr I64 kNsPerSecond = 1000000000LL;
constexpr I64 kNsPerMicrosecond = 1000LL;
constexpr F64 kG0 = 9.80665;  // standard gravity [m/s^2], the Isp convention

Vec3F64 toVec3F64(const Eigen::Vector3d& v) {
  Vec3F64 out;
  out[0] = v.x();
  out[1] = v.y();
  out[2] = v.z();
  return out;
}
}  // namespace

BurnExecutor ::BurnExecutor(const char* compName) : BurnExecutorComponentBase(compName) {}

void BurnExecutor ::commandBurnAtCycle(U32 cycle, F64 durationS, F64 throttle) {
  this->armed_cycle_ = cycle;
  this->armed_duration_s_ = durationS;
  this->armed_throttle_ = throttle;
}

// ----------------------------------------------------------------------
// Handlers
// ----------------------------------------------------------------------

void BurnExecutor ::attitudeIn_handler(FwIndexType portNum, const AttitudeEstimate& estimate) {
  static_cast<void>(portNum);
  this->attitude_ = estimate;
  this->attitude_seen_ = true;
}

I64 BurnExecutor ::nowTaiNs() const {
  const Fw::Time now = this->getTime();
  return static_cast<I64>(now.getSeconds()) * kNsPerSecond +
         static_cast<I64>(now.getUSeconds()) * kNsPerMicrosecond;
}

bool BurnExecutor ::attitudeUsable(I64 nowNs) const {
  if (!this->attitude_seen_ || !this->attitude_.get_attitudeValid()) {
    return false;
  }
  const F64 age_s = static_cast<F64>(nowNs - this->attitude_.get_epochTaiNs()) / 1.0e9;
  return age_s >= 0.0 && age_s <= this->max_att_age_s_;
}

void BurnExecutor ::run_handler(FwIndexType portNum, U32 context) {
  static_cast<void>(portNum);
  static_cast<void>(context);
  const I64 nowNs = this->nowTaiNs();
  ++this->cycle_;

  if (!this->configured_ && !this->applyParameters()) {
    this->publishIdle(nowNs);
    return;
  }

  // The bench hook: the command body, run here because the command port
  // shares this handler's mutex.
  if (this->armed_cycle_ != 0 && this->cycle_ == this->armed_cycle_) {
    BurnRefusal::T reason = BurnRefusal::UNCONFIGURED;
    if (!this->startBurn(this->armed_duration_s_, this->armed_throttle_, reason)) {
      this->log_WARNING_LO_BurnRefused(BurnRefusal(reason));
    }
  }

  const F64 dt_s =
      this->last_run_ns_ > 0 ? static_cast<F64>(nowNs - this->last_run_ns_) / 1.0e9 : 0.0;
  this->last_run_ns_ = nowNs;

  if (this->state_ != BurnState::BURNING) {
    this->publishIdle(nowNs);
    return;
  }

  // Burning: the attitude must still be usable to say where the thrust points.
  if (!this->attitudeUsable(nowNs)) {
    this->endBurn(false, BurnRefusal::ATTITUDE);
    this->publishIdle(nowNs);
    return;
  }

  // Commanded acceleration over the vehicle's own mass estimate, rotated to
  // ECI: the estimate is Body <- ECI, so its inverse carries a body vector out.
  const QuatF64& q = this->attitude_.get_qBodyEci();
  polaris::math::Quaternion q_bi(q[0], q[1], q[2], q[3]);
  if (!q_bi.isFinite() || !q_bi.normalize()) {
    this->endBurn(false, BurnRefusal::ATTITUDE);
    this->publishIdle(nowNs);
    return;
  }
  const polaris::math::Quaternion q_ib = q_bi.inverse();
  Eigen::Vector3d accel_eci = Eigen::Vector3d::Zero();
  F64 mass_flow = 0.0;
  ThrusterThrottleSet cmds;
  for (U32 i = 0; i < ThrusterThrottleSet::SIZE; ++i) {
    cmds[i] = 0.0;
  }
  for (U32 i = 0; i < this->count_; ++i) {
    const F64 f = this->throttle_ * this->thrust_n_[i];
    accel_eci += q_ib.rotate(this->axis_[i]) * (f / this->mass_kg_);
    mass_flow += f / (this->isp_s_[i] * kG0);
    cmds[i] = this->throttle_;
  }
  const F64 a_mag = accel_eci.norm();
  if (!accel_eci.allFinite() || !std::isfinite(mass_flow)) {
    this->endBurn(false, BurnRefusal::UNCONFIGURED);
    this->publishIdle(nowNs);
    return;
  }

  // Bookkeeping over the elapsed step (the first burning cycle has dt of one
  // period behind it too: the throttle went out last cycle).
  this->delta_v_mps_ += a_mag * dt_s;
  this->mass_kg_ = std::max(this->mass_kg_ - mass_flow * dt_s, 1.0e-3);
  this->remaining_s_ -= dt_s;

  if (this->isConnected_thrusterCmdOut_OutputPort(0)) {
    this->thrusterCmdOut_out(0, cmds);
  }
  NonGravAccel accel;
  accel.set_epochTaiNs(nowNs);
  accel.set_accelEciMps2(toVec3F64(accel_eci));
  accel.set_sigmaMps2(this->knowledge_frac_ * a_mag);
  accel.set_valid(true);
  if (this->isConnected_accelOut_OutputPort(0)) {
    this->accelOut_out(0, accel);
  }
  this->tlmWrite_BurnStateTlm(BurnState(this->state_));
  this->tlmWrite_BurnRemainingS(this->remaining_s_);
  this->tlmWrite_BurnDeltaVMps(this->delta_v_mps_);
  this->tlmWrite_MassEstimateKg(this->mass_kg_);
  this->tlmWrite_ThrottleCmd(this->throttle_);

  if (this->remaining_s_ <= 0.0) {
    this->endBurn(true, BurnRefusal::UNCONFIGURED);
  }
}

void BurnExecutor ::publishIdle(I64 nowNs) {
  ThrusterThrottleSet cmds;
  for (U32 i = 0; i < ThrusterThrottleSet::SIZE; ++i) {
    cmds[i] = 0.0;
  }
  if (this->isConnected_thrusterCmdOut_OutputPort(0)) {
    this->thrusterCmdOut_out(0, cmds);
  }
  NonGravAccel accel;
  accel.set_epochTaiNs(nowNs);
  accel.set_accelEciMps2(toVec3F64(Eigen::Vector3d::Zero()));
  accel.set_sigmaMps2(0.0);
  accel.set_valid(false);
  if (this->isConnected_accelOut_OutputPort(0)) {
    this->accelOut_out(0, accel);
  }
  this->tlmWrite_BurnStateTlm(BurnState(this->state_));
  this->tlmWrite_BurnRemainingS(0.0);
  this->tlmWrite_BurnDeltaVMps(this->delta_v_mps_);
  this->tlmWrite_MassEstimateKg(this->mass_kg_);
  this->tlmWrite_ThrottleCmd(0.0);
}

bool BurnExecutor ::startBurn(F64 durationS, F64 throttle, BurnRefusal::T& reason) {
  // A command may arrive before the first cycle: read the tuning now rather
  // than refuse a configured vehicle for being early.
  if (!this->configured_ && !this->applyParameters()) {
    reason = BurnRefusal::UNCONFIGURED;
    return false;
  }
  if (this->state_ == BurnState::BURNING) {
    reason = BurnRefusal::ALREADY_BURNING;
    return false;
  }
  if (!std::isfinite(durationS) || durationS <= 0.0 || durationS > this->max_duration_s_) {
    reason = BurnRefusal::DURATION;
    return false;
  }
  if (!std::isfinite(throttle) || throttle <= 0.0 || throttle > 1.0) {
    reason = BurnRefusal::THROTTLE;
    return false;
  }
  if (!this->attitudeUsable(this->nowTaiNs())) {
    reason = BurnRefusal::ATTITUDE;
    return false;
  }
  this->state_ = BurnState::BURNING;
  this->throttle_ = throttle;
  this->remaining_s_ = durationS;
  this->delta_v_mps_ = 0.0;
  this->log_ACTIVITY_HI_BurnStarted(durationS, throttle);
  return true;
}

void BurnExecutor ::endBurn(bool completed, BurnRefusal::T reason) {
  this->throttle_ = 0.0;
  this->remaining_s_ = 0.0;
  if (completed) {
    this->state_ = BurnState::IDLE;
    this->log_ACTIVITY_HI_BurnCompleted(this->delta_v_mps_);
  } else {
    this->state_ = BurnState::ABORTED;
    this->log_WARNING_HI_BurnAborted(BurnRefusal(reason));
  }
}

// ----------------------------------------------------------------------
// Commands
// ----------------------------------------------------------------------

void BurnExecutor ::BURN_START_cmdHandler(FwOpcodeType opCode, U32 cmdSeq, F64 durationS,
                                          F64 throttleFrac) {
  BurnRefusal::T reason = BurnRefusal::UNCONFIGURED;
  if (this->startBurn(durationS, throttleFrac, reason)) {
    this->cmdResponse_out(opCode, cmdSeq, Fw::CmdResponse::OK);
  } else {
    this->log_WARNING_LO_BurnRefused(BurnRefusal(reason));
    this->cmdResponse_out(opCode, cmdSeq, Fw::CmdResponse::VALIDATION_ERROR);
  }
}

void BurnExecutor ::BURN_ABORT_cmdHandler(FwOpcodeType opCode, U32 cmdSeq) {
  if (this->state_ == BurnState::BURNING) {
    // Operator abort: reported as an abort with no fault reason of its own.
    this->throttle_ = 0.0;
    this->remaining_s_ = 0.0;
    this->state_ = BurnState::ABORTED;
    this->log_WARNING_HI_BurnAborted(BurnRefusal(BurnRefusal::ALREADY_BURNING));
  }
  this->cmdResponse_out(opCode, cmdSeq, Fw::CmdResponse::OK);
}

// ----------------------------------------------------------------------
// Parameters
// ----------------------------------------------------------------------

void BurnExecutor ::parameterUpdated(FwPrmIdType id) {
  static_cast<void>(id);
  (void)this->applyParameters();
}

bool BurnExecutor ::applyParameters() {
  auto fail = [this](const char* detail) {
    if (!this->config_alerted_) {
      Fw::LogStringArg arg(detail);
      this->log_WARNING_HI_ConfigInvalid(arg);
      this->config_alerted_ = true;
    }
    this->configured_ = false;
    return false;
  };
#define POLARIS_GET(dest, getter, name)         \
  do {                                          \
    Fw::ParamValid v = Fw::ParamValid::INVALID; \
    (dest) = this->getter(v);                   \
    if (v != Fw::ParamValid::VALID) {           \
      return fail(name);                        \
    }                                           \
  } while (0)
  U32 count = 0;
  POLARIS_GET(count, paramGet_ThrusterCount, "ThrusterCount");
  if (count < 1 || count > kMaxThrusters) {
    return fail("ThrusterCount out of range");
  }
  Vec3F64PerUnit axes;
  F64PerUnit thrust;
  F64PerUnit isp;
  POLARIS_GET(axes, paramGet_ThrusterAxesBody, "ThrusterAxesBody");
  POLARIS_GET(thrust, paramGet_ThrusterThrustN, "ThrusterThrustN");
  POLARIS_GET(isp, paramGet_ThrusterIspS, "ThrusterIspS");
  for (U32 i = 0; i < count; ++i) {
    Eigen::Vector3d a(axes[3 * i], axes[3 * i + 1], axes[3 * i + 2]);
    if (!a.allFinite() || a.norm() < 1.0e-9) {
      return fail("ThrusterAxesBody has a zero or non-finite axis");
    }
    this->axis_[i] = a.normalized();
    if (!std::isfinite(thrust[i]) || thrust[i] <= 0.0) {
      return fail("ThrusterThrustN must be positive");
    }
    if (!std::isfinite(isp[i]) || isp[i] <= 0.0) {
      return fail("ThrusterIspS must be positive");
    }
    this->thrust_n_[i] = thrust[i];
    this->isp_s_[i] = isp[i];
  }
  F64 mass = 0.0;
  POLARIS_GET(mass, paramGet_VehicleMassKg, "VehicleMassKg");
  POLARIS_GET(this->knowledge_frac_, paramGet_ThrustKnowledgeFrac, "ThrustKnowledgeFrac");
  POLARIS_GET(this->max_duration_s_, paramGet_MaxBurnDurationS, "MaxBurnDurationS");
  POLARIS_GET(this->max_att_age_s_, paramGet_MaxAttitudeAgeS, "MaxAttitudeAgeS");
#undef POLARIS_GET
  if (!std::isfinite(mass) || mass <= 0.0) {
    return fail("VehicleMassKg must be positive");
  }
  if (!std::isfinite(this->knowledge_frac_) || this->knowledge_frac_ < 0.0 ||
      this->knowledge_frac_ > 1.0) {
    return fail("ThrustKnowledgeFrac must be in [0, 1]");
  }
  if (!std::isfinite(this->max_duration_s_) || this->max_duration_s_ <= 0.0 ||
      !std::isfinite(this->max_att_age_s_) || this->max_att_age_s_ <= 0.0) {
    return fail("MaxBurnDurationS/MaxAttitudeAgeS must be positive");
  }
  this->count_ = count;
  // The mass estimate is only re-armed from the parameter when the executor is
  // (re)configured: a burn's depletion is not undone by an unrelated re-read.
  if (!this->configured_) {
    this->mass_kg_ = mass;
  }
  this->configured_ = true;
  this->config_alerted_ = false;
  return true;
}

}  // namespace flight
