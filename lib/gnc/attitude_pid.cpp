#include "gnc/attitude_pid.hpp"

#include <algorithm>
#include <cmath>

namespace polaris::gnc {
namespace {

/// Unit-norm tolerance on an input quaternion. Loose enough for a filter's
/// working normalisation, tight enough that a garbage value is caught.
constexpr double kUnitTol = 1.0e-6;

bool quaternionUsable(const math::Quaternion& q) {
  return q.isFinite() && std::abs(q.norm() - 1.0) <= kUnitTol;
}

bool refuse(AttitudePidRefusal reason, AttitudePidResult& out) {
  out = AttitudePidResult{};
  out.refusal = reason;
  return false;
}

}  // namespace

bool attitudeError(const math::Quat<math::frames::Body, math::frames::ECI>& q_est,
                   const math::Quat<math::frames::Body, math::frames::ECI>& q_ref,
                   math::Vec3<math::frames::Body>& out, double& angle_rad) {
  if (!quaternionUsable(q_est.core()) || !quaternionUsable(q_ref.core())) {
    return false;
  }
  // dq = q_ref (x) q_est^-1 : the rotation carrying the current body frame onto
  // the reference one. Composed through the *typed* operator so the frame legs
  // are checked — q_est.inverse() is ECI<-Body, and the ECI cancels against
  // q_ref's — rather than dropping to the untagged core and taking it on trust.
  const math::Quaternion dq = (q_ref * q_est.inverse()).core();
  const Eigen::Vector3d vec = dq.vec();
  const double scalar = dq.scalar();

  // Short way round: dq and -dq are the same rotation, and the sign of the
  // scalar part is what picks the <= 180 deg branch.
  const double sign = scalar < 0.0 ? -1.0 : 1.0;
  out = math::Vec3<math::frames::Body>(2.0 * sign * vec);
  // atan2 form: exact at both ends, unlike 2*acos(scalar) (lib/README.md).
  angle_rad = 2.0 * std::atan2(vec.norm(), std::abs(scalar));
  return true;
}

bool AttitudePidConfig::isValid() const {
  return std::isfinite(kp_nm_per_rad) && kp_nm_per_rad > 0.0 && std::isfinite(ki_nm_per_rad_s) &&
         ki_nm_per_rad_s >= 0.0 && std::isfinite(kd_nm_per_radps) && kd_nm_per_radps > 0.0 &&
         std::isfinite(max_integral_rad_s) && max_integral_rad_s >= 0.0 &&
         std::isfinite(max_torque_nm) && max_torque_nm > 0.0 && std::isfinite(max_dt_s) &&
         max_dt_s > 0.0;
}

AttitudePid::AttitudePid(const AttitudePidConfig& config)
    : config_(config), configured_(config.isValid()) {}

void AttitudePid::reset() {
  integral_ = math::Vec3<math::frames::Body>(Eigen::Vector3d::Zero());
}

bool AttitudePid::update(const math::Quat<math::frames::Body, math::frames::ECI>& q_est,
                         const math::Vec3<math::frames::Body>& rate_est,
                         const math::Quat<math::frames::Body, math::frames::ECI>& q_ref,
                         const math::Vec3<math::frames::Body>& rate_ref, double dt_s,
                         AttitudePidResult& out) {
  if (!configured_) {
    return refuse(AttitudePidRefusal::kUnconfigured, out);
  }
  if (!rate_est.eigen().allFinite() || !rate_ref.eigen().allFinite() || !std::isfinite(dt_s)) {
    return refuse(AttitudePidRefusal::kBadInput, out);
  }

  math::Vec3<math::frames::Body> error{};
  double angle_rad = 0.0;
  if (!attitudeError(q_est, q_ref, error, angle_rad)) {
    return refuse(AttitudePidRefusal::kBadInput, out);
  }

  const Eigen::Vector3d rate_error = rate_ref.eigen() - rate_est.eigen();

  // Unsaturated demand with the integrator as it stands. The integrator is
  // advanced only after the saturation test, so a saturated cycle does not fill
  // it (conditional integration).
  const Eigen::Vector3d proportional = config_.kp_nm_per_rad * error.eigen();
  const Eigen::Vector3d derivative = config_.kd_nm_per_radps * rate_error;
  const Eigen::Vector3d integral_term = config_.ki_nm_per_rad_s * integral_.eigen();
  Eigen::Vector3d demand = proportional + integral_term + derivative;

  const double demand_norm = demand.norm();
  const bool saturated = demand_norm > config_.max_torque_nm;
  if (saturated) {
    // One scale factor for the whole vector: an over-demand becomes a slower
    // correction about the right axis, never a faster one about the wrong axis.
    demand *= config_.max_torque_nm / demand_norm;
  }

  const bool integrate =
      config_.ki_nm_per_rad_s > 0.0 && !saturated && dt_s > 0.0 && dt_s <= config_.max_dt_s;
  if (integrate) {
    Eigen::Vector3d next = integral_.eigen() + error.eigen() * dt_s;
    for (int i = 0; i < 3; ++i) {
      next[i] = std::clamp(next[i], -config_.max_integral_rad_s, config_.max_integral_rad_s);
    }
    integral_ = math::Vec3<math::frames::Body>(next);
  }

  if (!demand.allFinite()) {
    return refuse(AttitudePidRefusal::kNonFiniteOutput, out);
  }

  out.torque_nm = math::Vec3<math::frames::Body>(demand);
  out.attitude_error_rad = error;
  out.rate_error_radps = math::Vec3<math::frames::Body>(rate_error);
  out.error_angle_rad = angle_rad;
  out.saturated = saturated;
  out.valid = true;
  out.refusal = AttitudePidRefusal::kNone;
  return true;
}

}  // namespace polaris::gnc
