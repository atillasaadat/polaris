#include "gnc/rw_friction.hpp"

#include <cmath>

namespace polaris::gnc {

namespace {

double clampTo(double x, double limit) {
  if (x > limit) {
    return limit;
  }
  if (x < -limit) {
    return -limit;
  }
  return x;
}

}  // namespace

bool RwFrictionConfig::isValid() const {
  if (wheel_count < 1 || wheel_count > kMaxWheels) {
    return false;
  }
  if (!std::isfinite(dry_friction_nm) || dry_friction_nm < 0.0) {
    return false;
  }
  if (!std::isfinite(viscous_friction_nm_s) || viscous_friction_nm_s < 0.0) {
    return false;
  }
  // A zero deadband is the discontinuous sign() this module exists to avoid, so
  // it is refused rather than read as "no blending wanted".
  if (!std::isfinite(deadband_radps) || !(deadband_radps > 0.0)) {
    return false;
  }
  for (int i = 0; i < wheel_count; ++i) {
    if (!std::isfinite(max_torque_nm[i]) || !(max_torque_nm[i] > 0.0)) {
      return false;
    }
    // Deliberately no upper bound on the trim: over-compensation is a decision an
    // operator may have flight data for, and the policy that it stays at or below
    // one is documented rather than silently enforced (rw_friction.hpp).
    if (!std::isfinite(scale[i]) || scale[i] < 0.0) {
      return false;
    }
  }
  return true;
}

RwFrictionCompensator ::RwFrictionCompensator(const RwFrictionConfig& config) {
  if (!config.isValid()) {
    return;
  }
  config_ = config;
  configured_ = true;
}

double RwFrictionCompensator ::blend(double speed_radps) const {
  if (!configured_ || !std::isfinite(speed_radps)) {
    return 0.0;
  }
  return clampTo(speed_radps / config_.deadband_radps, 1.0);
}

bool RwFrictionCompensator ::compensate(const double* demand_nm, const double* speed_radps,
                                        const bool* speed_valid, RwFrictionResult& out) const {
  out = RwFrictionResult{};
  if (!configured_) {
    out.refusal = RwFrictionRefusal::kUnconfigured;
    return false;
  }
  if (demand_nm == nullptr || speed_radps == nullptr || speed_valid == nullptr) {
    out.refusal = RwFrictionRefusal::kBadInput;
    return false;
  }

  for (int i = 0; i < config_.wheel_count; ++i) {
    if (!std::isfinite(demand_nm[i])) {
      out = RwFrictionResult{};
      out.refusal = RwFrictionRefusal::kBadInput;
      return false;
    }
    // A speed flagged usable that is not finite is a fault in the caller's gate,
    // not a wheel this cycle can be reasoned about — refuse the whole set rather
    // than quietly dropping one wheel's compensation, because the two cases mean
    // different things to whoever reads `compensated`.
    if (speed_valid[i] && !std::isfinite(speed_radps[i])) {
      out = RwFrictionResult{};
      out.refusal = RwFrictionRefusal::kBadInput;
      return false;
    }

    const double limit = config_.max_torque_nm[i];
    // The allocation already clamped to this box; re-asserting it here is what
    // makes the "the clamp eats only the compensation" statement below true
    // without trusting the caller for it.
    const double demand = clampTo(demand_nm[i], limit);

    double applied = 0.0;
    if (speed_valid[i]) {
      const double omega = speed_radps[i];
      // -tau_f: the motor must supply what the bearings take away, so the sign
      // follows the speed. The Coulomb term goes through the bounded blend; the
      // viscous term is already continuous at zero and needs none.
      const double wanted = config_.scale[i] * (config_.dry_friction_nm * blend(omega) +
                                                config_.viscous_friction_nm_s * omega);
      // The clamp removes the *compensation*, never the demand: `demand` is
      // inside the box, so the clamped sum is at least as far along the demand's
      // direction as the demand itself, and the difference has `wanted`'s sign
      // with at most its magnitude.
      const double sum = demand + wanted;
      const double clamped = clampTo(sum, limit);
      // Saturation is "the clamp fired", not "applied differs from wanted":
      // `(d + w) - d` is not bit-exactly `w` in floating point even when nothing
      // was truncated, and a saturation flag that trips on round-off would be a
      // permanent alert on a wheel doing exactly what it was asked.
      if (clamped != sum) {
        out.saturated = true;
      }
      applied = clamped - demand;
      out.compensated[i] = true;
    }

    out.compensation_nm[i] = applied;
    out.torque_nm[i] = demand + applied;
    const double magnitude = std::abs(applied);
    out.max_compensation_nm =
        magnitude > out.max_compensation_nm ? magnitude : out.max_compensation_nm;

    if (!std::isfinite(out.torque_nm[i])) {
      out = RwFrictionResult{};
      out.refusal = RwFrictionRefusal::kNonFiniteOutput;
      return false;
    }
  }

  out.valid = true;
  out.refusal = RwFrictionRefusal::kNone;
  return true;
}

}  // namespace polaris::gnc
