#include "sensors/payload_sensor.hpp"

#include <algorithm>
#include <cmath>

namespace polaris::sim::sensors {
namespace {

constexpr double kDeg2Rad = 0.017453292519943295;

double get(const std::map<std::string, double>& p, const std::string& key) {
  const auto it = p.find(key);
  return it == p.end() ? 0.0 : it->second;
}

bool has(const std::map<std::string, double>& p, const std::string& key) {
  return p.find(key) != p.end();
}

/// A pixel count of zero means the catalog entry did not say, not that the
/// detector has no pixels; 1 is the honest fallback (and the right answer for a
/// single-element instrument).
std::int32_t pixelCount(const std::map<std::string, double>& p, const std::string& key) {
  const double v = get(p, key);
  return v >= 1.0 ? static_cast<std::int32_t>(std::llround(v)) : 1;
}

/// Angular size of one pixel along an axis: the full field divided by the count.
double ifov(double half_fov_rad, std::int32_t pixels) {
  return pixels > 0 ? 2.0 * half_fov_rad / static_cast<double>(pixels) : 0.0;
}

}  // namespace

double PayloadSensorSpec::equivalentHalfFovRad() const {
  if (shape == FovShape::kConic) {
    return half_fov_x_rad;
  }
  if (!(half_fov_x_rad > 0.0) || !(half_fov_y_rad > 0.0)) {
    return 0.0;
  }
  // Solid angle of the rectangular pyramid, then the cone of equal solid angle.
  // The arcsine argument is a product of two sines and so is always in range;
  // the clamp is belt-and-braces against rounding at the endpoints. Half-angles
  // at or beyond π/2 are rejected where a unit is built (scenario/vehicle.cpp),
  // not here — a spec built directly in a test may hold anything.
  const double omega =
      4.0 * std::asin(std::clamp(std::sin(half_fov_x_rad) * std::sin(half_fov_y_rad), -1.0, 1.0));
  // The equal-solid-angle cone has cos θ = 1 - Ω/2π. Keeping u = Ω/2π as the
  // variable and going through atan2(sin θ, cos θ) rather than acos(1 - u)
  // preserves the small-field precision: for a degrees-wide payload u is ~1e-3,
  // where 1 - u is stationary in θ and acos would surrender half its digits.
  // sin θ = √(u(2 - u)) is computed from u itself, so it never sees that loss.
  const double u = omega / (2.0 * M_PI);
  return std::atan2(std::sqrt(std::max(0.0, u * (2.0 - u))), 1.0 - u);
}

double PayloadSensorSpec::ifovXRad() const {
  return ifov(half_fov_x_rad, pixels_x);
}

double PayloadSensorSpec::ifovYRad() const {
  return ifov(shape == FovShape::kConic ? half_fov_x_rad : half_fov_y_rad, pixels_y);
}

PayloadSensorSpec PayloadSensorSpec::fromParams(const std::map<std::string, double>& p) {
  PayloadSensorSpec s;

  // The shape is which half-angles the entry sets, not a separate key (see the
  // header). Per-axis wins if both forms are present — it is the more specific
  // statement, and the alternative is silently ignoring the axis detail.
  if (has(p, "half_fov_x_deg") || has(p, "half_fov_y_deg")) {
    s.half_fov_x_rad = get(p, "half_fov_x_deg") * kDeg2Rad;
    s.half_fov_y_rad = get(p, "half_fov_y_deg") * kDeg2Rad;
    s.shape = (s.half_fov_x_rad == s.half_fov_y_rad) ? FovShape::kSquare : FovShape::kRectangular;
  } else {
    s.half_fov_x_rad = get(p, "half_fov_deg") * kDeg2Rad;
    s.half_fov_y_rad = s.half_fov_x_rad;
    s.shape = FovShape::kConic;
  }

  s.pixels_x = pixelCount(p, "pixels_x");
  s.pixels_y = pixelCount(p, "pixels_y");
  s.update_rate_hz = get(p, "update_rate_hz");

  s.keep_out.sun_rad = get(p, "sun_exclusion_deg") * kDeg2Rad;
  s.keep_out.earth_rad = get(p, "earth_exclusion_deg") * kDeg2Rad;
  s.keep_out.moon_rad = get(p, "moon_exclusion_deg") * kDeg2Rad;
  return s;
}

PayloadSensor::PayloadSensor(const PayloadSensorSpec& spec, const Eigen::Matrix3d& mounting_dcm)
    : spec_(spec), mounting_dcm_(mounting_dcm) {
  updateBoresight();
}

void PayloadSensor::updateBoresight() {
  // Sensor +Z is the boresight, always — the third column of the mounting.
  const Eigen::Vector3d nominal = mounting_dcm_.col(2);
  // Small-angle rotation vector applied in body axes: b' = normalize(b + θ × b).
  const Eigen::Vector3d tilted = nominal + fault_misalignment_.cross(nominal);
  const double norm = tilted.norm();
  boresight_body_ = norm > 0.0 ? Eigen::Vector3d(tilted / norm) : nominal;
}

void PayloadSensor::injectBoresightMisalignment(const math::Vec3<math::frames::Body>& delta_rad) {
  fault_misalignment_ = delta_rad.eigen();
  updateBoresight();
}

void PayloadSensor::clearFaults() {
  fault_misalignment_.setZero();
  fault_dropout_ = false;
  updateBoresight();
}

PayloadSensorSample PayloadSensor::sample(const time::Tai& epoch,
                                          const PayloadSensorInput& input) const {
  PayloadSensorSample out;
  out.time_tag = epoch;

  // Body→ECI: the truth attitude is Body←ECI, so its inverse carries the
  // mounted boresight out into the inertial frame.
  const Eigen::Vector3d boresight_eci =
      input.attitude.inverse().rotate(math::Vec3<math::frames::Body>(boresight_body_)).eigen();
  out.boresight_eci = math::Vec3<math::frames::ECI>(boresight_eci);

  // The shared §6.1 evaluator, on the equal-solid-angle cone (header). The
  // keep-out verdict does not depend on the field shape; the fractions do, in
  // the mean.
  out.occlusion =
      evaluateLineOfSight(boresight_eci, spec_.equivalentHalfFovRad(), input.sky, spec_.keep_out);
  out.valid = !fault_dropout_ && out.occlusion.occluder == Occluder::kNone;
  return out;
}

bool PayloadSensor::inFieldOfView(const Eigen::Vector3d& direction_sensor) const {
  const double norm = direction_sensor.norm();
  if (!(norm > 0.0) || !(direction_sensor.z() > 0.0)) {
    return false;  // rear hemisphere, or nothing to test
  }
  const Eigen::Vector3d d = direction_sensor / norm;
  if (spec_.shape == FovShape::kConic) {
    // Angle off the sensor +Z axis, in the same atan2 form used everywhere else:
    // the in-plane norm against the axial component. Exact on the boresight,
    // where acos(d.z()) would be accurate only to ~1e-8 rad.
    return std::atan2(std::hypot(d.x(), d.y()), d.z()) <= spec_.half_fov_x_rad;
  }
  // Rectangular/square: the two field angles are measured independently in the
  // X–Z and Y–Z planes, which is what a detector's rows and columns subtend.
  return std::abs(std::atan2(d.x(), d.z())) <= spec_.half_fov_x_rad &&
         std::abs(std::atan2(d.y(), d.z())) <= spec_.half_fov_y_rad;
}

}  // namespace polaris::sim::sensors
