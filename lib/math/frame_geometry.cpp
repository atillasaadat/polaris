/// @file
/// @brief Orbit-relative frame construction (design doc §3.1). See frame_geometry.hpp.

#include "math/frame_geometry.hpp"

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace polaris::math {

namespace {

/// Assemble a passive DCM whose rows are an orthonormal target-frame triad
/// (@p axis0, @p axis1, @p axis2, each expressed in ECI). `v_target = A v_eci`.
Eigen::Matrix3d dcmFromRows(const Eigen::Vector3d& axis0, const Eigen::Vector3d& axis1,
                            const Eigen::Vector3d& axis2) {
  Eigen::Matrix3d a;
  a.row(0) = axis0.transpose();
  a.row(1) = axis1.transpose();
  a.row(2) = axis2.transpose();
  return a;
}

/// Radial and orbit-normal unit vectors shared by both frames. Returns false if
/// the state is non-finite, the radius is zero, or r∥v (orbit normal vanishes).
bool orbitBasis(const Eigen::Vector3d& r, const Eigen::Vector3d& v, Eigen::Vector3d& r_hat,
                Eigen::Vector3d& h_hat) {
  if (!r.allFinite() || !v.allFinite()) {
    return false;
  }
  const double rn = r.norm();
  if (!(rn > 0.0)) {
    return false;
  }
  const Eigen::Vector3d h = r.cross(v);
  const double hn = h.norm();
  if (!(hn > 0.0)) {
    return false;
  }
  r_hat = r / rn;
  h_hat = h / hn;
  return true;
}

}  // namespace

bool ricFromEci(const Vec3<frames::ECI>& r_eci, const Vec3<frames::ECI>& v_eci,
                Quat<frames::RIC, frames::ECI>& out) {
  Eigen::Vector3d r_hat;
  Eigen::Vector3d c_hat;  // = ĥ
  if (!orbitBasis(r_eci.eigen(), v_eci.eigen(), r_hat, c_hat)) {
    return false;
  }
  const Eigen::Vector3d i_hat = c_hat.cross(r_hat);  // unit: ĥ ⟂ r̂
  const Eigen::Matrix3d dcm = dcmFromRows(r_hat, i_hat, c_hat);
  const Quaternion q = Quaternion::FromRotationMatrix(dcm).canonical();
  if (!q.isFinite()) {  // finite-but-huge inputs could overflow the triad (§3.6)
    return false;
  }
  out = Quat<frames::RIC, frames::ECI>(q);
  return true;
}

bool lvlhFromEci(const Vec3<frames::ECI>& r_eci, const Vec3<frames::ECI>& v_eci,
                 Quat<frames::LVLH, frames::ECI>& out) {
  Eigen::Vector3d r_hat;
  Eigen::Vector3d h_hat;
  if (!orbitBasis(r_eci.eigen(), v_eci.eigen(), r_hat, h_hat)) {
    return false;
  }
  const Eigen::Vector3d z_hat = -r_hat;              // nadir
  const Eigen::Vector3d y_hat = -h_hat;              // −orbit-normal
  const Eigen::Vector3d x_hat = y_hat.cross(z_hat);  // unit: ŷ ⟂ ẑ, ≈ +velocity
  const Eigen::Matrix3d dcm = dcmFromRows(x_hat, y_hat, z_hat);
  const Quaternion q = Quaternion::FromRotationMatrix(dcm).canonical();
  if (!q.isFinite()) {  // finite-but-huge inputs could overflow the triad (§3.6)
    return false;
  }
  out = Quat<frames::LVLH, frames::ECI>(q);
  return true;
}

}  // namespace polaris::math
