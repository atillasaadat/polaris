#include "sensors/star_tracker.hpp"

#include <cmath>

namespace polaris::sim::sensors {
namespace {

constexpr double kDeg2Rad = 0.017453292519943295;
/// Arcseconds to radians: 1″ = (π/180)/3600.
constexpr double kArcsec2Rad = kDeg2Rad / 3600.0;

double get(const std::map<std::string, double>& p, const std::string& key) {
  const auto it = p.find(key);
  return it == p.end() ? 0.0 : it->second;
}

/// Any unit vector perpendicular to @p v. Picks the seed axis least aligned with
/// @p v so the cross product is never near-degenerate.
Eigen::Vector3d anyPerpendicular(const Eigen::Vector3d& v) {
  Eigen::Index smallest = 0;
  v.cwiseAbs().minCoeff(&smallest);
  Eigen::Vector3d seed = Eigen::Vector3d::Zero();
  seed[smallest] = 1.0;
  return v.cross(seed).normalized();
}

}  // namespace

StarTrackerSpec StarTrackerSpec::fromParams(const std::map<std::string, double>& p) {
  StarTrackerSpec s;
  s.cross_axis_sigma = get(p, "cross_axis_arcsec") * kArcsec2Rad;
  s.boresight_sigma = get(p, "boresight_arcsec") * kArcsec2Rad;
  s.update_rate_hz = get(p, "update_rate_hz");
  s.fov_rad = get(p, "fov_deg") * kDeg2Rad;
  s.max_slew_rate = get(p, "max_slew_rate_deg_s") * kDeg2Rad;

  // The Earth constraint is measured from the *edge* of the field of view: a limb
  // just outside the boresight but inside the FOV still floods the detector. The
  // datasheet margin adds to that half-angle rather than replacing it.
  const double earth_margin = get(p, "earth_keepout_deg") * kDeg2Rad;
  const double half_fov = 0.5 * s.fov_rad;
  s.keep_out.earth_rad = (half_fov + earth_margin > 0.0) ? half_fov + earth_margin : 0.0;
  // Sun and Moon exclusion angles are quoted from the boresight and already
  // include the unit's baffle performance, so they stand on their own.
  s.keep_out.sun_rad = get(p, "sun_keepout_deg") * kDeg2Rad;
  s.keep_out.moon_rad = get(p, "moon_keepout_deg") * kDeg2Rad;
  return s;
}

StarTrackerMeasurement StarTracker::sample(
    const time::Tai& epoch, const math::Quat<math::frames::Body, math::frames::ECI>& truth,
    const math::Vec3<math::frames::Body>& body_rate, const SkyGeometry& sky) {
  StarTrackerMeasurement m;
  m.time_tag = epoch;

  // Three draws every call, whether or not the solution ends up valid: an outage
  // must not shift the sensor's stream position, or a scenario that changes only
  // the geometry would silently change the noise on every later sample (§3.6).
  const double g_cross_1 = rng_.gaussian();
  const double g_cross_2 = rng_.gaussian();
  const double g_bore = rng_.gaussian();

  const Eigen::Vector3d boresight_eci =
      truth.inverse().rotate(math::Vec3<math::frames::Body>(boresight_body_)).eigen();

  m.occlusion = evaluateLineOfSight(boresight_eci, 0.5 * spec_.fov_rad, sky, spec_.keep_out);
  m.rate_limited = spec_.max_slew_rate > 0.0 && body_rate.eigen().norm() > spec_.max_slew_rate;
  m.valid = !fault_dropout_ && m.occlusion.occluder == Occluder::kNone && !m.rate_limited;

  // Anisotropic error: σ_cross about the two axes across the boresight, σ_bore
  // about the boresight itself (see the header — roll is the weak axis).
  const Eigen::Vector3d e3 = boresight_body_.normalized();
  const Eigen::Vector3d e1 = anyPerpendicular(e3);
  const Eigen::Vector3d e2 = e3.cross(e1);
  const Eigen::Vector3d error_body = spec_.cross_axis_sigma * (g_cross_1 * e1 + g_cross_2 * e2) +
                                     spec_.boresight_sigma * g_bore * e3 + fault_bias_;

  // Small-angle rotation vector -> quaternion, applied on the body side:
  // q_meas = δq(error) ⊗ q_truth, so the error is expressed in body axes.
  const double angle = error_body.norm();
  math::Quaternion delta = math::Quaternion::Identity();
  if (angle > 0.0) {
    delta = math::Quaternion::FromAxisAngle(error_body / angle, angle);
  }
  m.attitude = math::Quat<math::frames::Body, math::frames::ECI>(delta * truth.core());
  return m;
}

}  // namespace polaris::sim::sensors
