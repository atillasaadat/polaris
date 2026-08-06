#include "gnc/disturbance.hpp"

#include <cmath>

namespace polaris::gnc {
namespace {

constexpr double kNsPerSecond = 1.0e9;
/// Unit-norm tolerance on the nadir direction. Loose enough for a direction
/// rotated out of an estimated attitude, tight enough that an unnormalised vector
/// (whose length would scale the torque) is caught.
constexpr double kUnitTol = 1.0e-6;

bool refuse(DisturbanceRefusal reason, DisturbanceResult& out) {
  out = DisturbanceResult{};
  out.refusal = reason;
  return false;
}

}  // namespace

bool gravityGradientTorque(const Eigen::Matrix3d& inertia_kgm2,
                           const math::Vec3<math::frames::Body>& nadir_unit_body,
                           double orbit_rate_rad_s, math::Vec3<math::frames::Body>& out) {
  const Eigen::Vector3d n = nadir_unit_body.eigen();
  if (!inertia_kgm2.allFinite() || !n.allFinite() || !std::isfinite(orbit_rate_rad_s) ||
      !(orbit_rate_rad_s > 0.0)) {
    return false;
  }
  if (std::abs(n.norm() - 1.0) > kUnitTol) {
    return false;
  }
  const Eigen::Vector3d torque =
      3.0 * orbit_rate_rad_s * orbit_rate_rad_s * n.cross(inertia_kgm2 * n);
  if (!torque.allFinite()) {
    return false;
  }
  out = math::Vec3<math::frames::Body>(torque);
  return true;
}

bool residualDipoleTorque(const math::Vec3<math::frames::Body>& residual_dipole_am2,
                          const math::Vec3<math::frames::Body>& field_tesla,
                          math::Vec3<math::frames::Body>& out) {
  const Eigen::Vector3d m = residual_dipole_am2.eigen();
  const Eigen::Vector3d b = field_tesla.eigen();
  if (!m.allFinite() || !b.allFinite()) {
    return false;
  }
  const Eigen::Vector3d torque = m.cross(b);
  if (!torque.allFinite()) {
    return false;
  }
  out = math::Vec3<math::frames::Body>(torque);
  return true;
}

bool DisturbanceObserverConfig::isValid() const {
  return std::isfinite(tau_s) && tau_s > 0.0 && std::isfinite(max_dt_s) && max_dt_s > 0.0 &&
         std::isfinite(anomaly_torque_nm) && anomaly_torque_nm > 0.0 &&
         std::isfinite(anomaly_clear_nm) && anomaly_clear_nm > 0.0 &&
         anomaly_clear_nm <= anomaly_torque_nm && anomaly_cycles > 0;
}

DisturbanceObserver::DisturbanceObserver(const DisturbanceObserverConfig& config)
    : config_(config), configured_(config.isValid()) {}

void DisturbanceObserver::reset() {
  have_previous_ = false;
  have_estimate_ = false;
  anomaly_ = false;
  estimate_ = math::Vec3<math::frames::Body>(Eigen::Vector3d::Zero());
  previous_momentum_ = math::Vec3<math::frames::Body>(Eigen::Vector3d::Zero());
  previous_time_ns_ = 0;
  above_streak_ = 0;
  below_streak_ = 0;
}

bool DisturbanceObserver::update(const math::Vec3<math::frames::Body>& total_momentum_nms,
                                 const math::Vec3<math::frames::Body>& body_rate_radps,
                                 const math::Vec3<math::frames::Body>& modelled_torque_nm,
                                 std::int64_t time_tag_tai_ns, DisturbanceResult& out) {
  if (!configured_) {
    return refuse(DisturbanceRefusal::kUnconfigured, out);
  }
  const Eigen::Vector3d h = total_momentum_nms.eigen();
  const Eigen::Vector3d w = body_rate_radps.eigen();
  const Eigen::Vector3d model = modelled_torque_nm.eigen();
  if (!h.allFinite() || !w.allFinite() || !model.allFinite()) {
    return refuse(DisturbanceRefusal::kBadInput, out);
  }

  if (!have_previous_) {
    previous_momentum_ = total_momentum_nms;
    previous_time_ns_ = time_tag_tai_ns;
    have_previous_ = true;
    return refuse(DisturbanceRefusal::kNoPreviousSample, out);
  }
  if (time_tag_tai_ns <= previous_time_ns_) {
    // A stuck clock divides by zero and a backwards one inverts the sign of the
    // torque; neither is recoverable by using the sample, so the newer one
    // becomes the anchor and this cycle is refused.
    previous_momentum_ = total_momentum_nms;
    previous_time_ns_ = time_tag_tai_ns;
    return refuse(DisturbanceRefusal::kNonMonotonicTime, out);
  }
  const double dt_s = static_cast<double>(time_tag_tai_ns - previous_time_ns_) / kNsPerSecond;
  if (dt_s > config_.max_dt_s) {
    previous_momentum_ = total_momentum_nms;
    previous_time_ns_ = time_tag_tai_ns;
    return refuse(DisturbanceRefusal::kStepTooLong, out);
  }

  // dH/dt + w x H is the total external torque; subtracting the modelled part
  // leaves the unmodelled one, which is what is worth feeding forward and what
  // the §9 monitor gates.
  const Eigen::Vector3d raw = (h - previous_momentum_.eigen()) / dt_s + w.cross(h) - model;
  previous_momentum_ = total_momentum_nms;
  previous_time_ns_ = time_tag_tai_ns;
  if (!raw.allFinite()) {
    return refuse(DisturbanceRefusal::kNonFiniteOutput, out);
  }

  // First-order low pass. alpha = dt/(tau+dt) is the exact pole of the
  // continuous filter under a forward-Euler step, and is bounded in (0, 1) for
  // every dt this far, so no step can make the filter overshoot its input.
  const double alpha = dt_s / (config_.tau_s + dt_s);
  const Eigen::Vector3d next = estimate_.eigen() + alpha * (raw - estimate_.eigen());
  if (!next.allFinite()) {
    // Do not poison the running estimate with a non-finite value: hold the last
    // good one and report the refusal.
    return refuse(DisturbanceRefusal::kNonFiniteOutput, out);
  }
  estimate_ = math::Vec3<math::frames::Body>(next);
  have_estimate_ = true;

  // §9 anomaly: persistence-counted both ways, with a deadband between the latch
  // and clear thresholds in which the latch holds its state — an estimate parked
  // at the budget must not cycle the anomaly once per confirmation count.
  const double norm = next.norm();
  if (norm > config_.anomaly_torque_nm) {
    below_streak_ = 0;
    if (above_streak_ < config_.anomaly_cycles) {
      ++above_streak_;
    }
    if (above_streak_ >= config_.anomaly_cycles) {
      anomaly_ = true;
    }
  } else if (norm < config_.anomaly_clear_nm) {
    above_streak_ = 0;
    if (below_streak_ < config_.anomaly_cycles) {
      ++below_streak_;
    }
    if (below_streak_ >= config_.anomaly_cycles) {
      anomaly_ = false;
    }
  } else {
    above_streak_ = 0;
    below_streak_ = 0;
  }

  out.torque_nm = estimate_;
  out.raw_torque_nm = math::Vec3<math::frames::Body>(raw);
  out.dt_s = dt_s;
  out.anomaly = anomaly_;
  out.valid = true;
  out.refusal = DisturbanceRefusal::kNone;
  return true;
}

}  // namespace polaris::gnc
