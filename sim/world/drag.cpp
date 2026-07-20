#include "world/drag.hpp"

#include <Eigen/Geometry>

#include "constants/constants.hpp"

namespace polaris::sim::world {

Eigen::Vector3d AtmosphericDrag::relativeVelocity(const state::TruthState& s) {
  // The atmosphere co-rotates with the Earth: omega = omega_e * z_hat, so the
  // wind velocity at r is omega x r. Only the ECI z-axis is involved, which is
  // why this needs no ECI->ECEF reduction.
  const Eigen::Vector3d omega(0.0, 0.0, constants::wgs84::kEarthRate);
  return s.velocity.eigen() - omega.cross(s.position.eigen());
}

bool AtmosphericDrag::dragAcceleration(const state::TruthState& s, Eigen::Vector3d& a_eci) const {
  if (!density_ || !(mass_ > 0.0)) {  // the !(> 0) form also rejects NaN mass
    return false;
  }
  const double rho = density_(s.epoch, s.position);
  if (!(rho > 0.0)) {  // vacuum, or a resolver with no answer here
    return false;
  }
  const Eigen::Vector3d v_rel = relativeVelocity(s);
  const double v = v_rel.norm();
  if (v < kMinSpeed_) {
    return false;  // no relative motion -> no drag, and no defined direction
  }
  a_eci = (-0.5 * rho * cd_ * area_ / mass_ * v) * v_rel;
  return true;
}

math::Vec3<math::frames::ECI> AtmosphericDrag::acceleration(const state::TruthState& s) const {
  Eigen::Vector3d a = Eigen::Vector3d::Zero();
  if (!dragAcceleration(s, a)) {
    return math::Vec3<math::frames::ECI>::Zero();
  }
  return math::Vec3<math::frames::ECI>(a);
}

math::Vec3<math::frames::Body> AtmosphericDrag::torque(const state::TruthState& s) const {
  // Pure cannon-ball (c.p. at c.m.) is torque-free; skip the density lookup.
  if (r_cp_.eigen().isZero()) {
    return math::Vec3<math::frames::Body>::Zero();
  }
  Eigen::Vector3d a = Eigen::Vector3d::Zero();
  if (!dragAcceleration(s, a)) {
    return math::Vec3<math::frames::Body>::Zero();
  }
  const Eigen::Matrix3d A = s.attitude.core().toRotationMatrix();  // Body <- ECI
  const Eigen::Vector3d f_body = A * (mass_ * a);
  return math::Vec3<math::frames::Body>(Eigen::Vector3d(r_cp_.eigen().cross(f_body)));
}

}  // namespace polaris::sim::world
