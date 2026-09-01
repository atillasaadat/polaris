#include "gnc/target_pointing.hpp"

#include <cmath>

namespace polaris::gnc {
namespace {

/// Below this, two unit vectors are treated as parallel and the cross product
/// that would resolve roll is numerically meaningless.
///
/// sin(0.5 deg) — chosen at the scale where the *result* stops being usable
/// rather than where the arithmetic stops working. At half a degree of
/// separation the roll solution is already swinging by tens of degrees for
/// arc-minute changes in the inputs, so a caller that got an answer here would
/// be handed a number far less certain than it looks.
constexpr double kParallelSin = 8.7e-3;

/// A body-axis pair separated by less than this cannot define a frame.
constexpr double kAxisParallelSin = 1.0e-6;

}  // namespace

const char* toString(TargetPointingStatus status) {
  switch (status) {
    case TargetPointingStatus::kOk:
      return "OK";
    case TargetPointingStatus::kBadInput:
      return "BAD_INPUT";
    case TargetPointingStatus::kCoincident:
      return "COINCIDENT";
    case TargetPointingStatus::kBadAxes:
      return "BAD_AXES";
    case TargetPointingStatus::kSecondaryDegenerate:
      return "SECONDARY_DEGENERATE";
  }
  return "UNKNOWN";
}

TargetPointingStatus targetPointingQuaternion(
    const math::Vec3<math::frames::ECI>& observer_eci_m,
    const math::Vec3<math::frames::ECI>& target_eci_m,
    const math::Vec3<math::frames::Body>& boresight_body,
    const math::Vec3<math::frames::Body>& secondary_body,
    const math::Vec3<math::frames::ECI>& secondary_ref_eci,
    math::Quat<math::frames::Body, math::frames::ECI>& out) {
  if (!observer_eci_m.isFinite() || !target_eci_m.isFinite() || !boresight_body.isFinite() ||
      !secondary_body.isFinite() || !secondary_ref_eci.isFinite()) {
    return TargetPointingStatus::kBadInput;
  }

  // ---- The two body-frame axes must actually span a plane.
  const Eigen::Vector3d b1 = boresight_body.eigen();
  const Eigen::Vector3d b2 = secondary_body.eigen();
  if (!(b1.norm() > 0.0) || !(b2.norm() > 0.0)) {
    return TargetPointingStatus::kBadAxes;
  }
  const Eigen::Vector3d b1u = b1.normalized();
  const Eigen::Vector3d b2u = b2.normalized();
  if (b1u.cross(b2u).norm() < kAxisParallelSin) {
    return TargetPointingStatus::kBadAxes;
  }

  // ---- The line of sight, which the boresight must land on exactly.
  const Eigen::Vector3d los = target_eci_m.eigen() - observer_eci_m.eigen();
  const double range = los.norm();
  if (!(range > 0.0)) {
    return TargetPointingStatus::kCoincident;
  }
  const Eigen::Vector3d l1 = los / range;

  // ---- Roll, from the secondary constraint.
  const Eigen::Vector3d ref = secondary_ref_eci.eigen();
  if (!(ref.norm() > 0.0)) {
    return TargetPointingStatus::kBadInput;
  }
  const Eigen::Vector3d refu = ref.normalized();
  // The component of the reference perpendicular to the line of sight is the
  // only part that carries roll information. When it vanishes the constraint is
  // silent and any roll satisfies it equally — refused rather than resolved by
  // an arbitrary choice, since the arbitrary choice is stable and plausible and
  // therefore invisible.
  Eigen::Vector3d perp = refu - l1 * l1.dot(refu);
  if (perp.norm() < kParallelSin) {
    return TargetPointingStatus::kSecondaryDegenerate;
  }
  const Eigen::Vector3d l2 = perp.normalized();
  const Eigen::Vector3d l3 = l1.cross(l2);

  // ---- Build both frames from the same construction and compose.
  //
  // Two orthonormal triads: one from the body axes, one from the inertial
  // directions they must land on. Each is built by the *same* Gram-Schmidt, so
  // the residual non-orthogonality of the inputs is absorbed identically on both
  // sides and the composed rotation is orthonormal by construction rather than
  // by renormalising a nearly-orthonormal product afterwards.
  const Eigen::Vector3d c2 = (b2u - b1u * b1u.dot(b2u)).normalized();
  const Eigen::Vector3d c3 = b1u.cross(c2);

  Eigen::Matrix3d body_axes;  // columns: the body triad
  body_axes.col(0) = b1u;
  body_axes.col(1) = c2;
  body_axes.col(2) = c3;

  Eigen::Matrix3d eci_axes;  // columns: where each must point, in ECI
  eci_axes.col(0) = l1;
  eci_axes.col(1) = l2;
  eci_axes.col(2) = l3;

  // A(Body<-ECI) maps the ECI triad onto the body triad.
  const Eigen::Matrix3d dcm = body_axes * eci_axes.transpose();
  if (!dcm.allFinite()) {
    return TargetPointingStatus::kBadInput;
  }
  out = math::Quat<math::frames::Body, math::frames::ECI>(math::Quaternion::FromRotationMatrix(dcm))
            .canonical();
  return TargetPointingStatus::kOk;
}

TargetPointingStatus targetPointingRate(const math::Vec3<math::frames::ECI>& observer_eci_m,
                                        const math::Vec3<math::frames::ECI>& observer_vel_m_s,
                                        const math::Vec3<math::frames::ECI>& target_eci_m,
                                        const math::Vec3<math::frames::ECI>& target_vel_m_s,
                                        math::Vec3<math::frames::ECI>& rate_eci_rad_s) {
  if (!observer_eci_m.isFinite() || !observer_vel_m_s.isFinite() || !target_eci_m.isFinite() ||
      !target_vel_m_s.isFinite()) {
    return TargetPointingStatus::kBadInput;
  }
  const Eigen::Vector3d r = target_eci_m.eigen() - observer_eci_m.eigen();
  const Eigen::Vector3d v = target_vel_m_s.eigen() - observer_vel_m_s.eigen();
  const double range2 = r.squaredNorm();
  if (!(range2 > 0.0)) {
    return TargetPointingStatus::kCoincident;
  }
  // omega = (r x v) / |r|^2 — the rotation rate of the unit line-of-sight
  // vector. Range rate contributes nothing, which is exactly right: closing on a
  // target does not rotate the direction to it.
  const Eigen::Vector3d omega = r.cross(v) / range2;
  if (!omega.allFinite()) {
    return TargetPointingStatus::kBadInput;
  }
  rate_eci_rad_s = math::Vec3<math::frames::ECI>(omega);
  return TargetPointingStatus::kOk;
}

}  // namespace polaris::gnc
