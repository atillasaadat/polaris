#include "gnc/albedo_correction.hpp"

#include <algorithm>
#include <cmath>
#include <Eigen/Geometry>

#include "constants/constants.hpp"
#include "math/fov_overlap.hpp"

namespace polaris::gnc {
namespace {

using Body = math::frames::Body;

/// Widest field a real sun sensor presents. A part quoting more than a full
/// hemisphere of acceptance is a corrupt parameter, not a wide-angle unit.
constexpr double kMaxHalfFov = M_PI * 0.5;

/// Ceiling on the configured peak albedo error. GomSpace quote >10° for the
/// reference part and 12° is the tail of their ISS histogram; a quarter turn is
/// far past any physical sun-sensor albedo error and catches a units mistake
/// (degrees left unconverted) before it reaches a measurement.
constexpr double kMaxAlbedoError = M_PI * 0.25;

}  // namespace

bool AlbedoCorrectionConfig::isValid() const {
  if (!std::isfinite(albedo_error_rad) || albedo_error_rad < 0.0 ||
      albedo_error_rad > kMaxAlbedoError) {
    return false;
  }
  if (!std::isfinite(half_fov_rad) || !(half_fov_rad > 0.0) || half_fov_rad > kMaxHalfFov) {
    return false;
  }
  if (!boresight_body.isFinite()) {
    return false;
  }
  // A boresight is a direction; a zero-length one names no axis at all.
  return boresight_body.eigen().norm() > 0.5;
}

bool albedoCorrection(const AlbedoCorrectionConfig& cfg, const AlbedoCorrectionInput& in,
                      math::Vec3<Body>& sun_corrected, double& applied_rad) {
  applied_rad = 0.0;
  if (!cfg.isValid()) {
    return false;
  }

  // Every input is checked before it is used: this runs on measured geometry,
  // and a NaN reaching the rotation below would leave an unusable direction in
  // the estimator's primary vector source.
  if (!in.sun_meas.isFinite() || !in.nadir_body.isFinite() || !std::isfinite(in.radius_m) ||
      !std::isfinite(in.dayside)) {
    return false;
  }
  // No Earthshine on the night side, and none in eclipse (dayside is zero there
  // by construction). Not a fault — the correct answer is simply zero.
  if (!(in.dayside > 0.0)) {
    return false;
  }
  // Below the surface is not a position this correction can be evaluated at; the
  // apparent radius would be undefined.
  if (!(in.radius_m > constants::wgs84::kSemiMajorAxis)) {
    return false;
  }

  math::Vec3<Body> sun_hat;
  math::Vec3<Body> nadir_hat;
  if (!in.sun_meas.normalized(sun_hat) || !in.nadir_body.normalized(nadir_hat)) {
    return false;
  }

  const Eigen::Vector3d s = sun_hat.eigen();
  const Eigen::Vector3d d = nadir_hat.eigen();
  const Eigen::Vector3d b = cfg.boresight_body.eigen().normalized();

  // How much of the field the Earth fills, from the same overlap the sensor
  // error was generated with (lib/math/fov_overlap.hpp) — one geometry, one
  // answer across the sim/flight seam.
  const double earth_angular_radius = std::asin(constants::wgs84::kSemiMajorAxis / in.radius_m);
  const double boresight_to_nadir = std::acos(std::clamp(b.dot(d), -1.0, 1.0));
  const double fraction =
      math::fovCoveredFraction(cfg.half_fov_rad, boresight_to_nadir, earth_angular_radius);
  if (!(fraction > 0.0)) {
    return false;  // no Earth in the field: nothing to remove
  }

  // The rotation axis. Sun and Earth centre collinear leaves the pull
  // undetermined in direction — and zero in magnitude, since sin(ψ) vanishes
  // there — so refusing costs nothing real and keeps a 0/0 out of the axis.
  const Eigen::Vector3d axis = s.cross(d);
  const double axis_norm = axis.norm();
  if (!(axis_norm > 0.0)) {
    return false;
  }

  const double peak = cfg.albedo_error_rad * fraction * std::min(1.0, in.dayside);
  // φ is defined on the *measured* separation (albedo_correction.hpp), so this
  // is the whole inverse — no iteration, and exact against the truth model when
  // its dispersion is zero.
  const double separation = std::acos(std::clamp(s.dot(d), -1.0, 1.0));
  const double phi = peak * std::sin(separation);
  if (!std::isfinite(phi) || phi <= 0.0) {
    return false;
  }

  const Eigen::Vector3d corrected = Eigen::AngleAxisd(-phi, axis / axis_norm) * s;
  if (!corrected.allFinite()) {
    return false;
  }
  sun_corrected = math::Vec3<Body>(corrected.normalized());
  applied_rad = phi;
  return true;
}

}  // namespace polaris::gnc
