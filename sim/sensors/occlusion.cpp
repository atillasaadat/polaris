#include "sensors/occlusion.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include "constants/constants.hpp"

namespace polaris::sim::sensors {
namespace {

constexpr double kEarthRadius = constants::wgs84::kSemiMajorAxis;
constexpr double kSunRadius = constants::bodies::kSunRadius;
constexpr double kMoonRadius = constants::bodies::kMoonRadius;

}  // namespace

double limbClearance(const Eigen::Vector3d& boresight_eci, const Eigen::Vector3d& to_body,
                     double body_radius_m) {
  const double bore_norm = boresight_eci.norm();
  const double distance = to_body.norm();
  if (!(bore_norm > 0.0) || !(distance > 0.0)) {
    return std::numeric_limits<double>::infinity();
  }

  // Inside the body: no meaningful angular radius, and the line of sight is
  // blocked whichever way it points.
  if (distance <= body_radius_m) {
    return -std::numeric_limits<double>::infinity();
  }

  // acos is clamped because a boresight exactly on the body centre can land a
  // hair outside [-1, 1] after rounding, which would otherwise be a NaN
  // propagating into a validity flag.
  const double cos_sep = std::clamp(boresight_eci.dot(to_body) / (bore_norm * distance), -1.0, 1.0);
  const double separation = std::acos(cos_sep);
  const double apparent_radius = std::asin(body_radius_m / distance);
  return separation - apparent_radius;
}

Occluder checkLineOfSight(const Eigen::Vector3d& boresight_eci, const SkyGeometry& sky,
                          const KeepOutSpec& keep_out) {
  // Earth first: at LEO its apparent radius is ~70°, so it is the constraint that
  // usually decides, and naming it is the more useful diagnosis when a Sun cone
  // happens to overlap the same direction.
  if (keep_out.earth_rad > 0.0 &&
      limbClearance(boresight_eci, -sky.sat, kEarthRadius) < keep_out.earth_rad) {
    return Occluder::kEarth;
  }
  if (keep_out.sun_rad > 0.0 &&
      limbClearance(boresight_eci, sky.sun - sky.sat, kSunRadius) < keep_out.sun_rad) {
    return Occluder::kSun;
  }
  if (keep_out.moon_rad > 0.0 &&
      limbClearance(boresight_eci, sky.moon - sky.sat, kMoonRadius) < keep_out.moon_rad) {
    return Occluder::kMoon;
  }
  return Occluder::kNone;
}

}  // namespace polaris::sim::sensors
