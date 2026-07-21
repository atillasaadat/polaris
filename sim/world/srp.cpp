/// @file
/// @brief Cannon-ball solar radiation pressure (see srp.hpp for the model).

#include "world/srp.hpp"

#include <cmath>

#include "constants/constants.hpp"
#include "time/tdb.hpp"
#include "time/timescales.hpp"
#include "world/eclipse.hpp"

namespace polaris::sim::world {

bool SolarRadiationPressure::solarAcceleration(const state::TruthState& s,
                                               Eigen::Vector3d& a_eci) const {
  if (!sun_ || !(mass_ > 0.0)) {
    return false;
  }

  // The ephemeris argument is TDB; the truth state carries TAI (§3.2).
  const time::Tdb tdb = time::toTdb(time::toTt(s.epoch));
  math::Vec3<math::frames::ECI> sun_pos;
  if (!sun_(tdb, sun_pos)) {
    return false;  // outside ephemeris coverage — no SRP this epoch
  }

  const Eigen::Vector3d r = s.position.eigen();
  const Eigen::Vector3d r_sun = sun_pos.eigen();

  // Sun -> spacecraft: the direction photons push, i.e. anti-sunward.
  const Eigen::Vector3d d = r - r_sun;
  const double d_n = d.norm();
  if (d_n < kMinDistance_) {
    return false;  // singular range guard (§3.6)
  }

  const double nu = eclipse_enabled_ ? shadowFactor(r, r_sun) : 1.0;
  if (nu <= 0.0) {
    return false;  // umbra — nothing to add
  }

  // Inverse-square falloff of the 1 AU reference pressure over the true range.
  const double au_over_d = constants::bodies::kAstronomicalUnit / d_n;
  const double pressure = constants::srp::kPressureAt1Au * au_over_d * au_over_d;
  a_eci = (nu * pressure * cr_ * area_ / mass_) * (d / d_n);
  return true;
}

math::Vec3<math::frames::ECI> SolarRadiationPressure::acceleration(
    const state::TruthState& s) const {
  Eigen::Vector3d a = Eigen::Vector3d::Zero();
  if (!solarAcceleration(s, a)) {
    return math::Vec3<math::frames::ECI>::Zero();
  }
  return math::Vec3<math::frames::ECI>(a);
}

math::Vec3<math::frames::Body> SolarRadiationPressure::torque(const state::TruthState& s) const {
  // Pure cannon-ball (center of pressure at the center of mass) is torque-free;
  // skip the ephemeris and shadow work entirely in that common case.
  if (r_cp_.eigen().isZero()) {
    return math::Vec3<math::frames::Body>::Zero();
  }
  Eigen::Vector3d a = Eigen::Vector3d::Zero();
  if (!solarAcceleration(s, a)) {
    return math::Vec3<math::frames::Body>::Zero();
  }
  // Force is ECI; the lever arm is Body-fixed, so rotate the force into Body
  // (A = Body <- ECI) before crossing. tau = r_cp x F.
  const Eigen::Matrix3d A = s.attitude.core().toRotationMatrix();
  const Eigen::Vector3d f_body = A * (mass_ * a);
  return math::Vec3<math::frames::Body>(Eigen::Vector3d(r_cp_.eigen().cross(f_body)));
}

}  // namespace polaris::sim::world
