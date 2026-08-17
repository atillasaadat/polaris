// ======================================================================
// \title  AttitudeControllerMomentum.cpp
// \brief  Momentum management, magnetic desaturation and the disturbance
//         feedforward (§8.5), plus the §9 envelope and momentum-anomaly
//         monitors they drive
//
// Split out of AttitudeController.cpp for the same reason the estimator's
// sensor voting is (AttitudeEstimatorSensors.cpp): one component and one class,
// several translation units, so the cycle's control ladder stays readable beside
// the momentum accounting rather than buried in it. Everything here is a private
// member of flight::AttitudeController — nothing is a free function with its own
// idea of the vehicle.
// ======================================================================

#include <cmath>
#include <limits>

#include "constants/constants.hpp"
#include "flight/PolarisFsw/AttitudeController/AttitudeController.hpp"

namespace flight {

namespace pm = polaris::math;
using Body = polaris::math::frames::Body;
using ECI = polaris::math::frames::ECI;

namespace {

//! Telemetry sentinel for "no value this cycle" — NaN draws as a gap on a strip
//! chart, where a zero would draw as a perfect measurement. The same convention
//! the rest of the component uses.
const F64 kNoValue = std::numeric_limits<F64>::quiet_NaN();

//! Fraction of WheelCapacityNms at which WheelNearCapacity fires. A fixed
//! policy rather than a parameter: the number that varies per vehicle is the
//! capacity, and 90 % is where a wheel has one desaturation's worth of headroom.
constexpr double kWheelCapacityAlertFraction = 0.9;

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
// Momentum management (§8.5) and its §9 monitors
// ----------------------------------------------------------------------

bool AttitudeController ::updateMomentum(I64 nowNs) {
  this->momentum_state_ = polaris::gnc::MomentumState{};
  // Staleness, on the same gate the estimate is judged by (§9.1): the tachometers
  // ride the same rate group, so a second bound for one physical question would
  // be a second chance to disagree. A stuck driver reports a plausible speed with
  // an old tag, which is exactly what this catches — and a stale wheel refuses
  // the sum rather than contributing a momentum the array no longer has.
  //
  // The verdict is a member rather than a local because the §8.5 friction
  // feedforward asks the same question of the same tachometers later in the same
  // cycle, and two answers to one question is how they come to disagree.
  for (U32 i = 0; i < polaris::gnc::kMaxWheels; ++i) {
    const I64 age_ns = nowNs - this->wheel_speed_time_ns_[i];
    const double age_s = static_cast<double>(age_ns < 0 ? -age_ns : age_ns) / 1.0e9;
    this->wheel_speed_fresh_[i] = this->wheel_speed_valid_[i] && age_s <= this->max_estimate_age_s_;
  }
  if (!this->momentum_.update(this->wheel_speed_radps_, this->wheel_speed_fresh_,
                              this->momentum_state_)) {
    // No usable momentum this cycle. The envelope latch is **held** rather than
    // cleared: a missing tachometer is not evidence the wheels emptied, and
    // clearing on absence is how a monitor un-alarms itself by losing its input.
    this->tlmWrite_StoredMomentum(toVec3F64(Eigen::Vector3d(kNoValue, kNoValue, kNoValue)));
    this->tlmWrite_StoredMomentumNms(kNoValue);
    this->tlmWrite_MaxWheelMomentumNms(kNoValue);
    this->tlmWrite_NullSpaceMomentumNms(kNoValue);
    this->tlmWrite_MomentumValid(false);
    // The refusal itself is FDIR-visible, not just a gap on a strip chart: one
    // dead tachometer takes out desaturation *and* both §9 momentum monitors,
    // and a monitor that loses its input has to say so. Bounded cadence, and
    // only once configured — an unconfigured controller already alarms through
    // ConfigInvalid, and a second stream for the same cause would be noise.
    if (this->configured_) {
      ++this->momentum_refusal_streak_;
      if (this->alertDue(this->momentum_refusal_streak_)) {
        MomentumRefusalEv::T reason = MomentumRefusalEv::UNCONFIGURED;
        switch (this->momentum_state_.refusal) {
          case polaris::gnc::MomentumRefusal::kWheelInvalid:
            reason = MomentumRefusalEv::WHEEL_INVALID;
            break;
          case polaris::gnc::MomentumRefusal::kBadInput:
            reason = MomentumRefusalEv::BAD_INPUT;
            break;
          default:
            break;
        }
        this->log_WARNING_HI_MomentumUnavailable(MomentumRefusalEv(reason),
                                                 this->momentum_state_.refused_wheel,
                                                 this->momentum_refusal_streak_);
      }
    }
    return false;
  }

  this->momentum_refusal_streak_ = 0;
  this->tlmWrite_StoredMomentum(toVec3F64(this->momentum_state_.stored_nms.eigen()));
  this->tlmWrite_StoredMomentumNms(this->momentum_state_.stored_norm_nms);
  this->tlmWrite_MaxWheelMomentumNms(this->momentum_state_.max_wheel_nms);
  this->tlmWrite_NullSpaceMomentumNms(this->momentum_state_.null_space_nms);
  this->tlmWrite_MomentumValid(true);

  // §9 per-wheel capacity monitor, edge-gated both ways on the one comparison.
  // Watches the largest single wheel, not the body sum: the envelope below is
  // blind to null-space momentum, and this is the alarm for what it cannot see.
  const bool near_capacity =
      this->momentum_state_.max_wheel_nms > kWheelCapacityAlertFraction * this->wheel_capacity_nms_;
  if (near_capacity && !this->wheel_capacity_alerted_) {
    this->log_WARNING_HI_WheelNearCapacity(this->momentum_state_.max_wheel_nms,
                                           this->wheel_capacity_nms_,
                                           this->momentum_state_.null_space_nms);
    this->wheel_capacity_alerted_ = true;
  } else if (!near_capacity && this->wheel_capacity_alerted_) {
    this->log_ACTIVITY_HI_WheelCapacityRecovered(this->momentum_state_.max_wheel_nms);
    this->wheel_capacity_alerted_ = false;
  }

  // §9 envelope monitor, edge-gated both ways on the one comparison.
  if (this->momentum_state_.envelope_exceeded && !this->envelope_alerted_) {
    this->log_WARNING_HI_MomentumEnvelopeExceeded(this->momentum_state_.stored_norm_nms,
                                                  this->momentum_envelope_nms_);
    this->envelope_alerted_ = true;
  } else if (!this->momentum_state_.envelope_exceeded && this->envelope_alerted_) {
    this->log_ACTIVITY_HI_MomentumEnvelopeRecovered(this->momentum_state_.stored_norm_nms);
    this->envelope_alerted_ = false;
  }
  return true;
}

void AttitudeController ::updateDisturbance(I64 nowNs) {
  this->feedforward_nm_ = pm::Vec3<Body>(Eigen::Vector3d::Zero());

  // --- Tier 1: model-based (§8.5) -----------------------------------------
  // Each term is taken only when its own inputs are there. A missing position
  // costs the gravity-gradient term and nothing else; a missing field costs the
  // residual-dipole term and nothing else. Guessing either would feed forward a
  // torque pointing somewhere the vehicle never was.
  Eigen::Vector3d modelled = Eigen::Vector3d::Zero();
  const Eigen::Matrix3d inertia = this->inertia_diag_kgm2_.asDiagonal();
  if (this->have_estimate_ && this->estimate_.get_posValid() &&
      this->estimate_.get_attitudeValid()) {
    const Vec3F64& p = this->estimate_.get_posEciM();
    const Eigen::Vector3d r_eci(p[0], p[1], p[2]);
    const double radius = r_eci.norm();
    if (std::isfinite(radius) && radius > 0.0) {
      const QuatF64& q = this->estimate_.get_qBodyEci();
      polaris::math::Quaternion attitude(q[0], q[1], q[2], q[3]);
      if (attitude.isFinite() && attitude.normalize()) {
        // Nadir in body axes, and the Keplerian mean motion at this radius. The
        // circular-orbit n is the standard gravity-gradient form and is exact for
        // this vehicle's near-circular orbit; the error it carries at small
        // eccentricity is second order in e and far below the term's own model
        // error.
        const Eigen::Vector3d nadir_eci = -r_eci / radius;
        const pm::Vec3<Body> nadir_body =
            pm::Quat<Body, ECI>(attitude).rotate(pm::Vec3<ECI>(nadir_eci));
        const double n_rad_s =
            std::sqrt(polaris::constants::wgs84::kGM / (radius * radius * radius));
        pm::Vec3<Body> gg;
        if (polaris::gnc::gravityGradientTorque(inertia, nadir_body, n_rad_s, gg)) {
          modelled += gg.eigen();
        }
      }
    }
  }
  if (this->have_estimate_ && this->estimate_.get_magFieldValid()) {
    pm::Vec3<Body> dipole_torque;
    if (polaris::gnc::residualDipoleTorque(pm::Vec3<Body>(this->residual_dipole_am2_),
                                           fromVec3F64(this->estimate_.get_magFieldBody()),
                                           dipole_torque)) {
      modelled += dipole_torque.eigen();
    }
  }

  // The magnetic torque this cycle's own desaturation is about to apply (zero
  // when it is not desaturating). It is **not** part of the tier-1 environment
  // model and is not switchable with it: it is the vehicle's own commanded
  // actuation, known exactly rather than modelled, and the two reasons to carry
  // it are independent of any tuning choice. Feeding it forward is what lets the
  // wheels be *told* about the rods instead of discovering them as attitude
  // error; subtracting it from the observer's input is what stops a
  // desaturation from looking like an unmodelled external torque and tripping
  // the §9 anomaly on the vehicle's own action.
  const Eigen::Vector3d commanded = this->magnetic_torque_nm_.eigen();
  // What the observer must subtract is the environment model **plus the previous
  // cycle's** magnetic command, since the momentum change it differences happened
  // over the interval that cycle drove. What the feedforward carries is *this*
  // cycle's, which applies over the interval both it and the wheel command cover.
  // The two are deliberately different vectors. One known approximation: when the
  // observer skipped a cycle and the next difference still fits inside
  // `max_dt_s`, the momentum change spans two commanded intervals while only the
  // immediately-previous magnetic torque is subtracted — an error of at most one
  // rod-window of impulse, far below the 200 s low-pass's resolution.
  const Eigen::Vector3d observer_model = modelled + this->magnetic_torque_prev_nm_.eigen();

  // --- Tier 2: the momentum observer, which is also the §9 monitor ---------
  // It runs whenever the inputs exist, whatever `FeedforwardObserverEnable`
  // says: the anomaly monitor is not a control feature, and a monitor a control
  // parameter can switch off is not a monitor.
  if (this->momentum_state_.valid && this->have_estimate_ && this->estimate_.get_rateValid()) {
    const Eigen::Vector3d rate = fromVec3F64(this->estimate_.get_bodyRateRadps()).eigen();
    // H = J*omega + h_wheels, the total system momentum: internal torques cancel
    // in it identically, so what the observer differences is external by
    // construction.
    const Eigen::Vector3d total = inertia * rate + this->momentum_state_.stored_nms.eigen();
    polaris::gnc::DisturbanceResult result;
    // A refusal holds the running estimate and is visible through
    // `hasEstimate()` and the ResidualTorque channel below; there is no
    // per-refusal action to take here.
    (void)this->observer_.update(pm::Vec3<Body>(total), pm::Vec3<Body>(rate),
                                 pm::Vec3<Body>(observer_model), nowNs, result);
  }

  const bool anomaly = this->observer_.anomaly();
  if (anomaly && !this->anomaly_alerted_) {
    this->log_WARNING_HI_MomentumAnomaly(this->observer_.estimate().eigen().norm(),
                                         this->disturbance_budget_nm_);
    this->anomaly_alerted_ = true;
  } else if (!anomaly && this->anomaly_alerted_) {
    this->log_ACTIVITY_HI_MomentumAnomalyCleared(this->observer_.estimate().eigen().norm());
    this->anomaly_alerted_ = false;
  }

  if (this->observer_.hasEstimate()) {
    this->tlmWrite_ResidualTorque(toVec3F64(this->observer_.estimate().eigen()));
  } else {
    this->tlmWrite_ResidualTorque(toVec3F64(Eigen::Vector3d(kNoValue, kNoValue, kNoValue)));
  }

  // The feedforward is **minus** the disturbance: the demand is the torque the
  // actuators must supply to cancel it.
  Eigen::Vector3d feedforward = -commanded;
  if (this->feedforward_model_) {
    feedforward -= modelled;
  }
  if (this->feedforward_observer_ && this->observer_.hasEstimate()) {
    feedforward -= this->observer_.estimate().eigen();
  }
  if (feedforward.allFinite()) {
    this->feedforward_nm_ = pm::Vec3<Body>(feedforward);
  }
  this->tlmWrite_FeedforwardTorque(toVec3F64(this->feedforward_nm_.eigen()));
}

bool AttitudeController ::desatDue() const {
  // Concurrent with POINT, excluded everywhere else. DETUMBLE is the exclusion
  // that matters: B-dot owns the rods there, and the two laws would otherwise
  // sum onto one set of coils with no schedule that describes either.
  if (!this->configured_ || this->mode_ != CtrlMode::POINT) {
    return false;
  }
  if (this->desat_override_ == DesatOverride::INHIBIT) {
    return false;
  }
  // The law needs a momentum error and a field, whatever the override says:
  // FORCE is a permission, never an instruction to drive a rod on no data.
  if (!this->momentum_state_.valid || !this->estimate_.get_magFieldValid()) {
    return false;
  }
  if (this->desat_override_ == DesatOverride::FORCE) {
    return true;
  }
  return this->momentum_state_.desat_required;
}

bool AttitudeController ::runDesat(pm::Vec3<Body>& dipole) {
  // The candidate set for the stuck-on attribution starts empty each cycle: a
  // refusal below would otherwise leave the previous cycle's bits accumulating
  // as candidates for a period in which no rod was commanded. `runDetumble`
  // clears at its own top for the same reason.
  this->commanded_mask_ = 0;
  const pm::Vec3<Body> field = fromVec3F64(this->estimate_.get_magFieldBody());
  polaris::gnc::MtqDesatResult result;
  if (!polaris::gnc::mtqDesaturation(this->desat_config_, this->momentum_state_.error_nms, field,
                                     result)) {
    return false;
  }
  if (!this->clampDipoleToRods(result.dipole_am2, dipole)) {
    return false;
  }
  // **The torque this cycle's rods will apply**, fed forward into the wheel
  // demand rather than left for the wheels to discover: the magnetic torque is
  // the one disturbance this vehicle knows *exactly and in advance*, and a
  // pointing loop that only reacts to it carries an error of the disturbance
  // over the proportional gain for as long as the desaturation lasts. Measured
  // on the reference vehicle that error was 2.9 deg — nearly three times
  // REQ-ACTL-002 — which is what turned this from a refinement into the thing
  // that makes concurrent desaturation and pointing work at all. The momentum
  // still leaves the wheels: they are being *told* to produce the counter-torque
  // instead of being dragged into it.
  this->noteMagneticTorque(dipole, field);
  return true;
}

void AttitudeController ::noteMagneticTorque(const pm::Vec3<Body>& dipole,
                                             const pm::Vec3<Body>& field) {
  // The torque the rods will apply over the interval this command covers, from
  // the **clamped** dipole and averaged over the period by the duty factor (the
  // rods carry it only through the on-window). Both magnetic laws record it, for
  // two consumers that need it for opposite reasons: the pointing demand feeds
  // *this* cycle's forward, and the disturbance observer subtracts the
  // *previous* cycle's, because the momentum change it differences happened over
  // the interval that cycle commanded. Without the second, B-dot's own torque —
  // more than ten times the §9 anomaly budget — would make every detumble look
  // like an unmodelled fault.
  const Eigen::Vector3d torque = this->mtq_duty_factor_ * dipole.eigen().cross(field.eigen());
  if (torque.allFinite()) {
    this->magnetic_torque_nm_ = pm::Vec3<Body>(torque);
  }
}

}  // namespace flight
