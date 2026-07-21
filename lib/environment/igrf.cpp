/// @file
/// @brief IGRF-14 main-field evaluation (see igrf.hpp).

#include "environment/igrf.hpp"

#include <algorithm>
#include <cmath>

namespace polaris::environment {
namespace {

/// nT -> T. IGRF publishes nanotesla; the repo speaks SI everywhere else.
constexpr double kNanoteslaToTesla = 1.0e-9;

/// Below this |sin θ| a point counts as being *at* a pole and the B_φ terms are
/// taken in the limit instead of dividing. Chosen well under the smallest
/// colatitude any caller would plausibly mean as "not the pole" (1e-12 rad is
/// ~6 nm of surface displacement), so ordinary near-polar queries still take the
/// exact ratio and only a true pole hits the limit branch.
constexpr double kPoleSinTolerance = 1.0e-12;

/// Scratch tables for one evaluation, indexed [n][m]. `p` holds the Schmidt
/// semi-normalised associated Legendre functions and `dp` their derivatives with
/// respect to colatitude. Fixed size — no allocation on the flight path.
struct LegendreTable {
  double p[kIgrfMaxDegree + 1][kIgrfMaxDegree + 1]{};
  double dp[kIgrfMaxDegree + 1][kIgrfMaxDegree + 1]{};
};

/// Fill @p out to degree @p nmax at the given colatitude.
///
/// Recursions after Langel (1987) Eq. 27 and Table 2 (p. 256) — the same scheme
/// the IAGA reference implementation uses, so the committed golden values and
/// this code agree to round-off rather than to a "close enough" tolerance.
void legendre(int nmax, double cos_theta, double sin_theta, LegendreTable& out) {
  out.p[0][0] = 1.0;
  if (nmax >= 1) {
    out.p[1][1] = sin_theta;
  }

  // Values. `root` is sqrt of a small integer; computing it inline keeps the
  // table out of the object and costs a handful of sqrt per call.
  for (int m = 0; m < nmax; ++m) {
    const double tmp = std::sqrt(static_cast<double>(2 * m + 1)) * out.p[m][m];
    out.p[m + 1][m] = cos_theta * tmp;
    if (m > 0) {
      out.p[m + 1][m + 1] = sin_theta * tmp / std::sqrt(static_cast<double>(2 * m + 2));
    }
    for (int n = m + 2; n <= nmax; ++n) {
      const double d = static_cast<double>(n * n - m * m);
      const double e = static_cast<double>(2 * n - 1);
      out.p[n][m] =
          (e * cos_theta * out.p[n - 1][m] - std::sqrt(d - e) * out.p[n - 2][m]) / std::sqrt(d);
    }
  }

  // Derivatives.
  if (nmax >= 1) {
    out.dp[1][0] = -out.p[1][1];
    out.dp[1][1] = out.p[1][0];
  }
  for (int n = 2; n <= nmax; ++n) {
    const double nn = static_cast<double>(n * n + n);
    out.dp[n][0] = -std::sqrt(nn / 2.0) * out.p[n][1];
    out.dp[n][1] = (std::sqrt(2.0 * nn) * out.p[n][0] - std::sqrt(nn - 2.0) * out.p[n][2]) / 2.0;
    for (int m = 2; m < n; ++m) {
      out.dp[n][m] =
          0.5 * (std::sqrt(static_cast<double>((n + m) * (n - m + 1))) * out.p[n][m - 1] -
                 std::sqrt(static_cast<double>((n + m + 1) * (n - m))) * out.p[n][m + 1]);
    }
    out.dp[n][n] = std::sqrt(2.0 * static_cast<double>(n)) * out.p[n][n - 1] / 2.0;
  }
}

}  // namespace

bool validIgrfCoefficients(const IgrfCoefficients& c) {
  if (c.degree < 1 || c.degree > kIgrfMaxDegree) {
    return false;
  }
  if (c.sv_degree < 0 || c.sv_degree > c.degree) {
    return false;
  }
  return std::isfinite(c.epoch_year);
}

IgrfField::IgrfField(const IgrfCoefficients& coefficients)
    : coefficients_(coefficients), good_(validIgrfCoefficients(coefficients)) {}

bool IgrfField::fieldSpherical(double radius_m, double colatitude_rad, double longitude_rad,
                               double decimal_year, double& b_r, double& b_theta,
                               double& b_phi) const {
  if (!good_) {
    return false;
  }
  if (!std::isfinite(radius_m) || !std::isfinite(colatitude_rad) || !std::isfinite(longitude_rad) ||
      !std::isfinite(decimal_year) || radius_m <= 0.0) {
    return false;
  }

  const int nmax = coefficients_.degree;
  const double cos_theta = std::cos(colatitude_rad);
  // From cos rather than sin(colatitude) directly so that |sin| is guaranteed
  // non-negative even for a colatitude wrapped outside [0, π].
  const double sin_theta = std::sqrt(std::max(0.0, 1.0 - cos_theta * cos_theta));

  LegendreTable lp;
  legendre(nmax, cos_theta, sin_theta, lp);

  const double dt = decimal_year - coefficients_.epoch_year;
  const double a_over_r = kIgrfReferenceRadius / radius_m;
  const bool at_pole = sin_theta < kPoleSinTolerance;
  // Sign of the L'Hopital limit of P_n^m / sin θ: +dP at the north pole,
  // −dP at the south (cos θ = −1 there).
  const double pole_sign = cos_theta >= 0.0 ? 1.0 : -1.0;

  double sum_r = 0.0;
  double sum_theta = 0.0;
  double sum_phi = 0.0;

  // (a/r)^{n+2}, built up rather than pow()ed per term.
  double radial = a_over_r * a_over_r * a_over_r;
  for (int n = 1; n <= nmax; ++n) {
    for (int m = 0; m <= n; ++m) {
      const double sv_scale = n <= coefficients_.sv_degree ? dt : 0.0;
      const double g = coefficients_.g[n][m] + sv_scale * coefficients_.g_sv[n][m];
      const double h = coefficients_.h[n][m] + sv_scale * coefficients_.h_sv[n][m];

      const double m_phi = static_cast<double>(m) * longitude_rad;
      const double cos_m_phi = std::cos(m_phi);
      const double sin_m_phi = std::sin(m_phi);
      const double gh = g * cos_m_phi + h * sin_m_phi;

      sum_r += static_cast<double>(n + 1) * radial * gh * lp.p[n][m];
      sum_theta -= radial * gh * lp.dp[n][m];

      if (m > 0) {
        const double ratio = at_pole ? pole_sign * lp.dp[n][m] : lp.p[n][m] / sin_theta;
        sum_phi += static_cast<double>(m) * radial * ratio * (g * sin_m_phi - h * cos_m_phi);
      }
    }
    radial *= a_over_r;
  }

  b_r = sum_r * kNanoteslaToTesla;
  b_theta = sum_theta * kNanoteslaToTesla;
  b_phi = sum_phi * kNanoteslaToTesla;
  return true;
}

bool IgrfField::field(const math::Vec3<math::frames::ECEF>& r_ecef, double decimal_year,
                      math::Vec3<math::frames::ECEF>& out) const {
  const Eigen::Vector3d r = r_ecef.eigen();
  const double radius = r.norm();
  if (!(radius > 0.0)) {
    return false;
  }
  const double colatitude = std::acos(std::max(-1.0, std::min(1.0, r.z() / radius)));
  const double longitude = std::atan2(r.y(), r.x());

  double b_r = 0.0;
  double b_theta = 0.0;
  double b_phi = 0.0;
  if (!fieldSpherical(radius, colatitude, longitude, decimal_year, b_r, b_theta, b_phi)) {
    return false;
  }

  // Spherical (r̂, θ̂, φ̂) -> Cartesian ECEF. θ̂ points south, φ̂ east.
  const double sin_theta = std::sin(colatitude);
  const double cos_theta = std::cos(colatitude);
  const double sin_phi = std::sin(longitude);
  const double cos_phi = std::cos(longitude);

  const Eigen::Vector3d b(
      b_r * sin_theta * cos_phi + b_theta * cos_theta * cos_phi - b_phi * sin_phi,
      b_r * sin_theta * sin_phi + b_theta * cos_theta * sin_phi + b_phi * cos_phi,
      b_r * cos_theta - b_theta * sin_theta);
  out = math::Vec3<math::frames::ECEF>(b);
  return true;
}

}  // namespace polaris::environment
