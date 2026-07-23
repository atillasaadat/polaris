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

/// Apparent angular radius [rad] of a sphere of radius @p body_radius_m at
/// distance @p distance_m, or π (the whole sky) when the point is inside it.
double apparentRadius(double body_radius_m, double distance_m) {
  if (!(distance_m > body_radius_m)) {
    return M_PI;
  }
  return std::asin(body_radius_m / distance_m);
}

/// Angle [rad] between two directions, robust to rounding at the endpoints.
double separation(const Eigen::Vector3d& a, const Eigen::Vector3d& b) {
  const double na = a.norm();
  const double nb = b.norm();
  if (!(na > 0.0) || !(nb > 0.0)) {
    return std::numeric_limits<double>::infinity();
  }
  return std::acos(std::clamp(a.dot(b) / (na * nb), -1.0, 1.0));
}

}  // namespace

double OcclusionState::blockedFraction() const {
  return std::max({earth_atmosphere_fraction, sun_fraction, moon_fraction});
}

double limbClearance(const Eigen::Vector3d& boresight_eci, const Eigen::Vector3d& to_body,
                     double body_radius_m) {
  const double distance = to_body.norm();
  const double sep = separation(boresight_eci, to_body);
  if (std::isinf(sep)) {
    return std::numeric_limits<double>::infinity();
  }
  // Inside the body: no meaningful angular radius, and the line of sight is
  // blocked whichever way it points.
  if (distance <= body_radius_m) {
    return -std::numeric_limits<double>::infinity();
  }
  return sep - std::asin(body_radius_m / distance);
}

double fovCoveredFraction(double half_fov_rad, double separation_rad, double body_radius_rad) {
  // A sensor with no field of view has no fraction to report; returning 0 keeps
  // a NaN out of a telemetry channel.
  if (!(half_fov_rad > 0.0) || !std::isfinite(separation_rad)) {
    return 0.0;
  }
  if (!(body_radius_rad > 0.0)) {
    return 0.0;
  }

  const double r_fov = half_fov_rad;
  const double r_body = body_radius_rad;
  const double d = std::abs(separation_rad);

  if (d >= r_fov + r_body) {
    return 0.0;  // disjoint
  }
  if (d <= r_body - r_fov) {
    return 1.0;  // FOV entirely inside the body
  }
  if (d <= r_fov - r_body) {
    // Body entirely inside the FOV: the ratio of the two disk areas.
    const double ratio = r_body / r_fov;
    return ratio * ratio;
  }

  // Partial overlap: the classical circular-lens area, on angular radii (see the
  // header on why the planar formula is used and what it costs).
  const double d2 = d * d;
  const double rf2 = r_fov * r_fov;
  const double rb2 = r_body * r_body;
  const double cos_fov = std::clamp((d2 + rf2 - rb2) / (2.0 * d * r_fov), -1.0, 1.0);
  const double cos_body = std::clamp((d2 + rb2 - rf2) / (2.0 * d * r_body), -1.0, 1.0);
  const double alpha = std::acos(cos_fov);
  const double beta = std::acos(cos_body);
  // Each circular segment is (r² · angle) minus the triangle it contains.
  const double area =
      rf2 * (alpha - std::sin(2.0 * alpha) * 0.5) + rb2 * (beta - std::sin(2.0 * beta) * 0.5);
  return std::clamp(area / (M_PI * rf2), 0.0, 1.0);
}

OcclusionState evaluateLineOfSight(const Eigen::Vector3d& boresight_eci, double half_fov_rad,
                                   const SkyGeometry& sky, const KeepOutSpec& keep_out) {
  OcclusionState state;

  // The Earth is two concentric bodies: the solid globe and the optically
  // obstructing shell above it. Both fractions are reported; the keep-out is
  // judged against the shell, the conservative choice (see the header).
  const Eigen::Vector3d to_earth = -sky.sat;
  const double earth_distance = to_earth.norm();
  const double atmosphere_radius = kEarthRadius + std::max(0.0, sky.atmosphere_height_m);
  const double earth_sep = separation(boresight_eci, to_earth);

  if (std::isfinite(earth_sep)) {
    state.earth_fraction =
        fovCoveredFraction(half_fov_rad, earth_sep, apparentRadius(kEarthRadius, earth_distance));
    state.earth_atmosphere_fraction = fovCoveredFraction(
        half_fov_rad, earth_sep, apparentRadius(atmosphere_radius, earth_distance));
  }

  const Eigen::Vector3d to_sun = sky.sun - sky.sat;
  const double sun_sep = separation(boresight_eci, to_sun);
  if (std::isfinite(sun_sep)) {
    state.sun_fraction =
        fovCoveredFraction(half_fov_rad, sun_sep, apparentRadius(kSunRadius, to_sun.norm()));
  }

  const Eigen::Vector3d to_moon = sky.moon - sky.sat;
  const double moon_sep = separation(boresight_eci, to_moon);
  if (std::isfinite(moon_sep)) {
    state.moon_fraction =
        fovCoveredFraction(half_fov_rad, moon_sep, apparentRadius(kMoonRadius, to_moon.norm()));
  }

  // Earth first: at LEO its apparent radius is ~70°, so it is the constraint that
  // usually decides, and naming it is the more useful diagnosis when a Sun cone
  // happens to overlap the same direction.
  if (keep_out.earth_rad > 0.0 &&
      limbClearance(boresight_eci, to_earth, atmosphere_radius) < keep_out.earth_rad) {
    state.occluder = Occluder::kEarth;
  } else if (keep_out.sun_rad > 0.0 &&
             limbClearance(boresight_eci, to_sun, kSunRadius) < keep_out.sun_rad) {
    state.occluder = Occluder::kSun;
  } else if (keep_out.moon_rad > 0.0 &&
             limbClearance(boresight_eci, to_moon, kMoonRadius) < keep_out.moon_rad) {
    state.occluder = Occluder::kMoon;
  }
  return state;
}

}  // namespace polaris::sim::sensors
