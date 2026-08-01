#include "world/gravity_gradient.hpp"

namespace polaris::sim::world {

math::Vec3<math::frames::Body> GravityGradientTorque::torque(const state::TruthState& s) const {
  const Eigen::Vector3d r_eci = s.position.eigen();
  const double rn = r_eci.norm();
  // A singular radius yields no finite gradient; a real orbit never reaches
  // r = 0, so return zero rather than a NaN (§3.6 boundary guard).
  if (rn < kMinRadius_) {
    return math::Vec3<math::frames::Body>::Zero();
  }

  // The inertia tensor is expressed in Body, so the direction must come to Body
  // rather than the tensor going to ECI.
  const Eigen::Vector3d r_hat_body = s.attitude.core().rotate(r_eci) / rn;
  const Eigen::Vector3d t = (3.0 * mu_ / (rn * rn * rn)) * r_hat_body.cross(*inertia_ * r_hat_body);
  return math::Vec3<math::frames::Body>(t);
}

}  // namespace polaris::sim::world
