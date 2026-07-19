/// @file
/// @brief Conical Earth-shadow function (see eclipse.hpp for the geometry).

#include "world/eclipse.hpp"

#include <algorithm>
#include <cmath>

#include "constants/constants.hpp"

namespace polaris::sim::world {

namespace {

/// asin with the argument clamped to [-1, 1]. The ratios below are radii over
/// distances and so are mathematically <= 1, but a spacecraft grazing the
/// surface can push R_e/|r_sat| a hair past 1 in floating point; clamping keeps
/// the result real instead of NaN (§3.6).
double safeAsin(double x) {
  return std::asin(std::clamp(x, -1.0, 1.0));
}

/// acos(num/den) with the ratio clamped to [-1, 1]. Same rounding guard as
/// `safeAsin`, for the lens-area terms where the ratio is analytically in range
/// only because the caller already established partial overlap.
double safeAcosRatio(double num, double den) {
  return std::acos(std::clamp(num / den, -1.0, 1.0));
}

constexpr double kPi = 3.141'592'653'589'793'238'46;

}  // namespace

double shadowFactor(const Eigen::Vector3d& r_sat, const Eigen::Vector3d& r_sun) {
  constexpr double kRe = constants::wgs84::kSemiMajorAxis;
  constexpr double kRs = constants::bodies::kSunRadius;

  const double r_sat_n = r_sat.norm();
  const Eigen::Vector3d sat_to_sun = r_sun - r_sat;
  const double sat_to_sun_n = sat_to_sun.norm();

  // Degenerate geometry, all of which fail dark rather than returning a
  // plausible-looking number (§3.6): at or inside the Earth's surface nothing is
  // lit; a Sun coincident with the spacecraft has no illumination direction; and
  // a Sun at or inside the Earth is not a Sun at all — that input means the
  // ephemeris resolver handed back garbage (e.g. an uninitialized zero vector),
  // which must not silently produce SRP.
  if (r_sat_n <= kRe || sat_to_sun_n <= 0.0 || r_sun.norm() <= kRe) {
    return 0.0;
  }

  // Apparent radii of the two disks and their apparent separation, as seen from
  // the spacecraft. The separation uses atan2 rather than acos so it stays
  // accurate for the small angles that dominate here.
  const double a = safeAsin(kRs / sat_to_sun_n);
  const double b = safeAsin(kRe / r_sat_n);
  const Eigen::Vector3d sat_to_earth = -r_sat;
  const double c = std::atan2(sat_to_earth.cross(sat_to_sun).norm(), sat_to_earth.dot(sat_to_sun));

  if (c >= a + b) {
    return 1.0;  // disks disjoint — full sunlight
  }
  if (c + a <= b) {
    return 0.0;  // solar disk wholly occulted — umbra
  }
  if (c + b <= a) {
    // Earth's disk wholly inside the solar disk — annular. Unreachable for an
    // Earth orbiter (needs range beyond the umbral cone tip) but keeps the
    // branch structure total rather than falling through to the lens formula,
    // whose acos arguments would leave their domain here.
    return 1.0 - (b * b) / (a * a);
  }

  // Penumbra: area of the circular lens where the two disks overlap, over the
  // area of the solar disk (Montenbruck & Gill §3.4.2, Eq. 3.87).
  const double x = (c * c + a * a - b * b) / (2.0 * c);
  const double y = std::sqrt(std::max(0.0, a * a - x * x));
  const double area = a * a * safeAcosRatio(x, a) + b * b * safeAcosRatio(c - x, b) - c * y;
  return std::clamp(1.0 - area / (kPi * a * a), 0.0, 1.0);
}

}  // namespace polaris::sim::world
