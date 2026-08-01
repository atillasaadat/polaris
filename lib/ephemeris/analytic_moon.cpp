/// @file
/// @brief Low-precision analytic Moon ephemeris (design doc §8.1, §11.3). See
/// analytic_moon.hpp.

#include "ephemeris/analytic_moon.hpp"

#include <cmath>

#include "constants/constants.hpp"

namespace polaris::ephemeris {

namespace {

constexpr double kDegToRad = 3.141'592'653'589'793'238 / 180.0;

/// sin of an argument given in degrees (the series arguments are all in degrees).
double sinDeg(double deg) {
  return std::sin(deg * kDegToRad);
}

double cosDeg(double deg) {
  return std::cos(deg * kDegToRad);
}

}  // namespace

math::Vec3<math::frames::ECI> moonPositionEci(const time::Tdb& tdb) {
  // Julian centuries since J2000 on the TDB scale (see analytic_sun.cpp for why a
  // single TDB argument suffices at this precision).
  const double t = time::daysSinceJ2000(tdb) / constants::time::kDaysPerJulianCentury;

  // Ecliptic longitude, ecliptic latitude, and horizontal parallax [deg].
  // Arguments are passed in degrees→radians unreduced (they run to ~10^6·T);
  // std::sin/std::cos perform the range reduction.
  const double lambda_deg =
      218.32 + 481267.8813 * t + 6.29 * sinDeg(134.9 + 477198.85 * t) -
      1.27 * sinDeg(259.2 - 413335.38 * t) + 0.66 * sinDeg(235.7 + 890534.23 * t) +
      0.21 * sinDeg(269.9 + 954397.70 * t) - 0.19 * sinDeg(357.5 + 35999.05 * t) -
      0.11 * sinDeg(186.6 + 966404.05 * t);
  const double phi_deg = 5.13 * sinDeg(93.3 + 483202.03 * t) +
                         0.28 * sinDeg(228.2 + 960400.87 * t) - 0.28 * sinDeg(318.3 + 6003.18 * t) -
                         0.17 * sinDeg(217.6 - 407332.20 * t);
  const double parallax_deg =
      0.9508 + 0.0518 * cosDeg(134.9 + 477198.85 * t) + 0.0095 * cosDeg(259.2 - 413335.38 * t) +
      0.0078 * cosDeg(235.7 + 890534.23 * t) + 0.0028 * cosDeg(269.9 + 954397.70 * t);

  const double lambda = lambda_deg * kDegToRad;
  const double phi = phi_deg * kDegToRad;
  const double eps = (23.439291 - 0.0130042 * t) * kDegToRad;

  // Geocentric range: r = R_earth / sin(parallax) [m].
  const double r = constants::wgs84::kSemiMajorAxis / sinDeg(parallax_deg);

  const double cos_phi = std::cos(phi);
  const double sin_phi = std::sin(phi);
  const double sin_lambda = std::sin(lambda);
  const double cos_lambda = std::cos(lambda);
  const double cos_eps = std::cos(eps);
  const double sin_eps = std::sin(eps);

  return math::Vec3<math::frames::ECI>(r * cos_phi * cos_lambda,
                                       r * (cos_eps * cos_phi * sin_lambda - sin_eps * sin_phi),
                                       r * (sin_eps * cos_phi * sin_lambda + cos_eps * sin_phi));
}

}  // namespace polaris::ephemeris
