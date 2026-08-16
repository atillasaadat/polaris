#include "gnc/geopotential.hpp"

#include <algorithm>
#include <cmath>

namespace polaris::gnc {
namespace {

/// The V/W tables run one degree past the field being evaluated, because the
/// acceleration at degree n reads row n+1 (Montenbruck & Gill Eq. 3.33). Two
/// extra rows, not one, so the `n+1` row of the highest evaluated degree is
/// itself built by a recursion that reads the row above it.
constexpr int kTableSize = kGeopotentialMaxDegree + 3;

}  // namespace

Eigen::Vector3d geopotentialAcceleration(const Eigen::Vector3d& r_ecef, int degree, int order,
                                         double mu, double reference_radius) {
  const Eigen::Vector3d zero = Eigen::Vector3d::Zero();
  if (!r_ecef.allFinite() || !std::isfinite(mu) || !std::isfinite(reference_radius)) {
    return zero;
  }
  if (!(mu > 0.0) || !(reference_radius > 0.0)) {
    return zero;
  }

  const double r2 = r_ecef.squaredNorm();
  const double r = std::sqrt(r2);
  if (!(r >= kGeopotentialMinRadiusM)) {
    return zero;
  }

  const int n_max = std::clamp(degree, 0, kGeopotentialMaxDegree);
  const int m_max = std::clamp(order, 0, n_max);

  const double x = r_ecef.x();
  const double y = r_ecef.y();
  const double z = r_ecef.z();
  const double re = reference_radius;

  // Recursion arguments (Montenbruck & Gill Eqs. 3.29-3.31): the position scaled
  // by Re/r², which is what makes each step dimensionless.
  const double xs = re * x / r2;
  const double ys = re * y / r2;
  const double zs = re * z / r2;
  const double rs = re * re / r2;

  double v[kTableSize][kTableSize]{};
  double w[kTableSize][kTableSize]{};

  // Seed: V_00 = Re/r, W_00 = 0.
  v[0][0] = re / r;
  w[0][0] = 0.0;

  // The tables must reach degree n_max + 1 for the acceleration assembly, and
  // order m_max + 1 for the same reason on the sectoral side.
  const int n_table = std::min(n_max + 1, kTableSize - 1);
  const int m_table = std::min(m_max + 1, n_table);

  // Sectorals, up the diagonal (Eq. 3.30). Each V_mm/W_mm is built only from
  // V_{m-1,m-1}/W_{m-1,m-1}, so this pass stands alone.
  for (int m = 1; m <= m_table; ++m) {
    const double factor = static_cast<double>(2 * m - 1);
    v[m][m] = factor * (xs * v[m - 1][m - 1] - ys * w[m - 1][m - 1]);
    w[m][m] = factor * (xs * w[m - 1][m - 1] + ys * v[m - 1][m - 1]);
  }

  // Verticals, down each column (Eq. 3.31). The n = m + 1 row drops the
  // second term, since V_{m-1,m} does not exist.
  for (int m = 0; m <= m_table; ++m) {
    for (int n = m + 1; n <= n_table; ++n) {
      const double a = static_cast<double>(2 * n - 1) / static_cast<double>(n - m);
      v[n][m] = a * zs * v[n - 1][m];
      w[n][m] = a * zs * w[n - 1][m];
      if (n - 2 >= m) {
        const double b = static_cast<double>(n + m - 1) / static_cast<double>(n - m);
        v[n][m] -= b * rs * v[n - 2][m];
        w[n][m] -= b * rs * w[n - 2][m];
      }
    }
  }

  // Acceleration assembly (Eq. 3.33). Accumulated in units of mu/Re², applied at
  // the end — one multiply instead of one per term, and it keeps the summands the
  // same order of magnitude as one another.
  double ax = 0.0;
  double ay = 0.0;
  double az = 0.0;

  for (int n = 0; n <= n_max; ++n) {
    const int m_top = std::min(n, m_max);
    for (int m = 0; m <= m_top; ++m) {
      const double c = egm2008::kC[n][m];
      const double s = egm2008::kS[n][m];
      if (c == 0.0 && s == 0.0) {
        continue;
      }
      if (m == 0) {
        // A zonal term contributes to x and y only through the m = 1 column of
        // the row above; there is no S_n0 (the sine of a zero-order term is
        // identically zero) and no second half to the bracket.
        ax -= c * v[n + 1][1];
        ay -= c * w[n + 1][1];
      } else {
        // (n-m+2)!/(n-m)! — the ratio in Eq. 3.33, two factors, not a factorial.
        const double fac = static_cast<double>((n - m + 2) * (n - m + 1));
        ax += 0.5 * ((-c * v[n + 1][m + 1] - s * w[n + 1][m + 1]) +
                     fac * (c * v[n + 1][m - 1] + s * w[n + 1][m - 1]));
        ay += 0.5 * ((-c * w[n + 1][m + 1] + s * v[n + 1][m + 1]) +
                     fac * (-c * w[n + 1][m - 1] + s * v[n + 1][m - 1]));
      }
      az += static_cast<double>(n - m + 1) * (-c * v[n + 1][m] - s * w[n + 1][m]);
    }
  }

  const double scale = mu / (re * re);
  const Eigen::Vector3d a(scale * ax, scale * ay, scale * az);
  return a.allFinite() ? a : zero;
}

}  // namespace polaris::gnc
