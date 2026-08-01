#include "world/magnetic_field.hpp"

#include "time/utc.hpp"

namespace polaris::sim::world {

bool decimalYear(const time::Tai& epoch, const time::LeapSecondTable& leap, double& out) {
  // One conversion for the whole repo: the FSW's onboard IGRF reference (§8.1)
  // needs the same decimal year, so it lives in lib/time rather than here.
  return time::decimalYear(epoch, leap, out);
}

bool EarthMagneticField::field(const time::Tai& epoch, const math::Vec3<math::frames::ECI>& r_eci,
                               math::Vec3<math::frames::ECI>& out) const {
  if (!good()) {
    return false;
  }

  math::Quat<math::frames::ECEF, math::frames::ECI> q;
  if (!eci_to_ecef_(epoch, q)) {
    return false;
  }

  double year = 0.0;
  if (!time::decimalYear(epoch, *leap_, year)) {
    return false;
  }

  const math::Vec3<math::frames::ECEF> r_ecef(q.core().rotate(r_eci.eigen()));
  math::Vec3<math::frames::ECEF> b_ecef;
  if (!field_.field(r_ecef, year, b_ecef)) {
    return false;
  }

  // B is a vector field, so it rotates back with the inverse of the same
  // rotation that took the position out — no translation, the frames share an
  // origin.
  out = math::Vec3<math::frames::ECI>(q.core().inverse().rotate(b_ecef.eigen()));
  return true;
}

MagneticFieldFn EarthMagneticField::fieldFn() const {
  return [this](const time::Tai& epoch, const math::Vec3<math::frames::ECI>& r_eci,
                math::Vec3<math::frames::ECI>& out) { return field(epoch, r_eci, out); };
}

math::Vec3<math::frames::Body> ResidualDipoleTorque::torque(const state::TruthState& s) const {
  if (!field_) {
    return math::Vec3<math::frames::Body>::Zero();
  }
  math::Vec3<math::frames::ECI> b_eci;
  if (!field_(s.epoch, s.position, b_eci)) {
    return math::Vec3<math::frames::Body>::Zero();
  }
  // The dipole is fixed in the Body frame, so bring the field to Body rather
  // than the dipole to ECI — the cross product has to happen in one frame and
  // the torque is wanted in Body.
  const Eigen::Vector3d b_body = s.attitude.core().rotate(b_eci.eigen());
  return math::Vec3<math::frames::Body>(Eigen::Vector3d(dipole_body_.eigen().cross(b_body)));
}

}  // namespace polaris::sim::world
