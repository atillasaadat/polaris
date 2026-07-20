#include "world/atmosphere.hpp"

#include <cmath>
#include <cstddef>
#include <Eigen/Core>

#include "constants/constants.hpp"

namespace polaris::sim::world {
namespace {

/// One band of the piecewise-exponential fit: base altitude [m], base density
/// [kg/m^3], scale height [m]. Vallado Table 8-4 (US Standard Atmosphere 1976 /
/// CIRA-72), tabulated there in km and converted here once.
struct Band {
  double base_altitude_m;
  double base_density;
  double scale_height_m;
};

constexpr Band kBands[] = {
    {0.0, 1.225, 7'249.0},
    {25'000.0, 3.899e-2, 6'349.0},
    {30'000.0, 1.774e-2, 6'682.0},
    {40'000.0, 3.972e-3, 7'554.0},
    {50'000.0, 1.057e-3, 8'382.0},
    {60'000.0, 3.206e-4, 7'714.0},
    {70'000.0, 8.770e-5, 6'549.0},
    {80'000.0, 1.905e-5, 5'799.0},
    {90'000.0, 3.396e-6, 5'382.0},
    {100'000.0, 5.297e-7, 5'877.0},
    {110'000.0, 9.661e-8, 7'263.0},
    {120'000.0, 2.438e-8, 9'473.0},
    {130'000.0, 8.484e-9, 12'636.0},
    {140'000.0, 3.845e-9, 16'149.0},
    {150'000.0, 2.070e-9, 22'523.0},
    {180'000.0, 5.464e-10, 29'740.0},
    {200'000.0, 2.789e-10, 37'105.0},
    {250'000.0, 7.248e-11, 45'546.0},
    {300'000.0, 2.418e-11, 53'628.0},
    {350'000.0, 9.518e-12, 53'298.0},
    {400'000.0, 3.725e-12, 58'515.0},
    {450'000.0, 1.585e-12, 60'828.0},
    {500'000.0, 6.967e-13, 63'822.0},
    {600'000.0, 1.454e-13, 71'835.0},
    {700'000.0, 3.614e-14, 88'667.0},
    {800'000.0, 1.170e-14, 124'640.0},
    {900'000.0, 5.245e-15, 181'050.0},
    {1'000'000.0, 3.019e-15, 268'000.0},
};

constexpr std::size_t kBandCount = sizeof(kBands) / sizeof(kBands[0]);

/// Distance from the spin axis [m] below which the closed-form polar answer
/// `|z| - b` is used instead of the geodetic-latitude iteration.
///
/// Set at a metre rather than at the point where the iteration actually divides
/// by zero, because the iteration degrades *before* it breaks: within a few
/// millimetres of the axis it is metres off, which would put a discontinuity in
/// accuracy right at the seam. The closed form is exact on the axis and errs by
/// O(p^2/b) beside it — under a picometre at p = 1 m — so handing it the whole
/// last metre is strictly the more accurate split.
constexpr double kPolarTolerance = 1.0;

/// pi/2 — the geodetic latitude at the poles, where the iteration is skipped.
constexpr double kHalfPi = 1.570'796'326'794'896'619'23;

}  // namespace

Geodetic geodetic(const Eigen::Vector3d& r) {
  constexpr double kA = constants::wgs84::kSemiMajorAxis;
  constexpr double kB = constants::wgs84::kSemiMinorAxis;
  constexpr double kE2 = constants::wgs84::kEccentricitySq;

  const double z = r.z();
  const double p = std::hypot(r.x(), r.y());
  const double longitude = std::atan2(r.y(), r.x());

  // On the spin axis the ellipsoid normal is the axis itself, and the iteration
  // below divides by cos(lat) -> 0. Handle it in closed form instead.
  if (p < kPolarTolerance) {
    const double pole = z >= 0.0 ? kHalfPi : -kHalfPi;
    return Geodetic{pole, longitude, std::abs(z) - kB};
  }

  // Fixed-point iteration on geodetic latitude (Vallado Alg. 12). The update
  // term e^2 N/(N+h) shrinks with cos(lat), so convergence is slowest near the
  // poles: three passes leave ~0.5 m of error inside a tenth of a degree of the
  // axis, four bring that back under a centimetre everywhere out to GEO. Four
  // it is — this is an exported general-purpose function, and a caller doing
  // geolocation should not have to know where the accurate latitudes are.
  double lat = std::atan2(z, p * (1.0 - kE2));
  double altitude = 0.0;
  for (int i = 0; i < 4; ++i) {
    const double sin_lat = std::sin(lat);
    const double n = kA / std::sqrt(1.0 - kE2 * sin_lat * sin_lat);
    altitude = p / std::cos(lat) - n;
    lat = std::atan2(z, p * (1.0 - kE2 * n / (n + altitude)));
  }
  return Geodetic{lat, longitude, altitude};
}

double geodeticAltitude(const math::Vec3<math::frames::ECI>& r_eci) {
  return geodetic(r_eci.eigen()).altitude_m;
}

double exponentialDensity(double altitude_m) {
  // Below the ellipsoid is not a place a spacecraft is; report no atmosphere
  // rather than extrapolating the sea-level band downward (§3.6 boundary guard).
  if (!(altitude_m >= 0.0)) {  // also rejects NaN
    return 0.0;
  }

  // Last band whose base is at or below the altitude. The table is short enough
  // that a linear scan beats anything cleverer, and it is called once per
  // integrator stage.
  std::size_t i = 0;
  while (i + 1 < kBandCount && altitude_m >= kBands[i + 1].base_altitude_m) {
    ++i;
  }

  const Band& b = kBands[i];
  return b.base_density * std::exp(-(altitude_m - b.base_altitude_m) / b.scale_height_m);
}

}  // namespace polaris::sim::world
