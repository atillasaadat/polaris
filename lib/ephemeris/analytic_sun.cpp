/// @file
/// @brief Low-precision analytic Sun ephemeris (design doc §8.1, §11.3). See
/// analytic_sun.hpp.

#include "ephemeris/analytic_sun.hpp"

#include <cmath>

#include "constants/constants.hpp"

namespace polaris::ephemeris {

namespace {

constexpr double kDegToRad = 3.141'592'653'589'793'238 / 180.0;

}  // namespace

math::Vec3<math::frames::ECI> sunPositionEci(const time::Tdb& tdb) {
  // Julian centuries since J2000 on the TDB scale. Vallado's algorithm nominally
  // takes T_UT1 for the angles and T_TDB for the obliquity; at ~0.01° the two are
  // interchangeable, so a single TDB argument is used throughout.
  const double t = time::daysSinceJ2000(tdb) / constants::time::kDaysPerJulianCentury;

  // Mean longitude and mean anomaly [deg]; reduce the anomaly before the trig so
  // std::sin sees a bounded argument even for large |T|.
  const double lambda_m_deg = 280.460 + 36000.771 * t;
  const double m_deg = std::fmod(357.5291092 + 35999.05034 * t, 360.0);
  const double m = m_deg * kDegToRad;

  // Apparent ecliptic longitude [deg] -> rad.
  const double lambda_ecl_deg =
      lambda_m_deg + 1.914666471 * std::sin(m) + 0.019994643 * std::sin(2.0 * m);
  const double lambda = lambda_ecl_deg * kDegToRad;

  // Geocentric range [AU] -> m and obliquity of the ecliptic [deg] -> rad.
  const double r_au = 1.000140612 - 0.016708617 * std::cos(m) - 0.000139589 * std::cos(2.0 * m);
  const double r = r_au * constants::bodies::kAstronomicalUnit;
  const double eps = (23.439291 - 0.0130042 * t) * kDegToRad;

  const double sin_lambda = std::sin(lambda);
  return math::Vec3<math::frames::ECI>(r * std::cos(lambda), r * std::cos(eps) * sin_lambda,
                                       r * std::sin(eps) * sin_lambda);
}

}  // namespace polaris::ephemeris
