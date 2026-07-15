/// @file
/// @brief Cunningham/Montenbruck-Gill V/W recursion for spherical-harmonic gravity.

#include "world/gravity_field.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>

namespace polaris::sim::world {

namespace {

/// Allocate a zero-filled triangular table: row n has n+1 columns (m = 0..n).
std::vector<std::vector<double>> triangular(int nmax) {
  std::vector<std::vector<double>> t(static_cast<std::size_t>(nmax) + 1);
  for (int n = 0; n <= nmax; ++n) {
    t[static_cast<std::size_t>(n)].assign(static_cast<std::size_t>(n) + 1, 0.0);
  }
  return t;
}

/// Fill the M&G auxiliary functions V, W for n,m in [0, N] (M&G Eqs. 3.29-3.31).
/// Square tables of side N+1 (the acceleration recursion reads V[n+1][m+1]).
void computeVW(const Eigen::Vector3d& r, double re, int N, std::vector<std::vector<double>>& V,
               std::vector<std::vector<double>>& W) {
  const std::size_t dim = static_cast<std::size_t>(N) + 1;
  V.assign(dim, std::vector<double>(dim, 0.0));
  W.assign(dim, std::vector<double>(dim, 0.0));

  const double r2 = r.squaredNorm();
  const double rinv2 = 1.0 / r2;
  const double xf = r.x() * re * rinv2;
  const double yf = r.y() * re * rinv2;
  const double zf = r.z() * re * rinv2;
  const double rf = re * re * rinv2;

  V[0][0] = re / std::sqrt(r2);
  W[0][0] = 0.0;

  for (int m = 1; m <= N; ++m) {
    const auto um = static_cast<std::size_t>(m);
    // Sectoral (diagonal) terms from the previous diagonal.
    const double c = 2.0 * m - 1.0;
    V[um][um] = c * (xf * V[um - 1][um - 1] - yf * W[um - 1][um - 1]);
    W[um][um] = c * (xf * W[um - 1][um - 1] + yf * V[um - 1][um - 1]);
  }

  for (int m = 0; m <= N; ++m) {
    const auto um = static_cast<std::size_t>(m);
    for (int n = m + 1; n <= N; ++n) {
      const auto un = static_cast<std::size_t>(n);
      const double a = (2.0 * n - 1.0) / (n - m);
      V[un][um] = a * zf * V[un - 1][um];
      W[un][um] = a * zf * W[un - 1][um];
      if (n - 2 >= m) {  // the two-back term drops when n-2 < m
        const double b = static_cast<double>(n + m - 1) / (n - m);
        V[un][um] -= b * rf * V[un - 2][um];
        W[un][um] -= b * rf * W[un - 2][um];
      }
    }
  }
}

}  // namespace

GravityCoeffs GravityCoeffs::pointMass() {
  GravityCoeffs g;
  g.nmax = 0;
  g.C = {{1.0}};
  g.S = {{0.0}};
  return g;
}

GravityCoeffs GravityCoeffs::earthZonal() {
  GravityCoeffs g;
  g.nmax = 6;
  g.C = triangular(g.nmax);
  g.S = triangular(g.nmax);
  g.C[0][0] = 1.0;   // point mass
  g.C[1][0] = 0.0;   // origin at the geocenter -> no degree-1 term
  g.C[2][0] = -kJ2;  // C_{n,0} = -J_n
  g.C[3][0] = -kJ3;
  g.C[4][0] = -kJ4;
  g.C[5][0] = -kJ5;
  g.C[6][0] = -kJ6;
  return g;
}

SphericalHarmonicGravity::SphericalHarmonicGravity(GravityCoeffs coeffs,
                                                   const Eigen::Matrix3d& inertia, int degree,
                                                   int order, double mu, double ref_radius)
    : coeffs_(std::move(coeffs)),
      inertia_(inertia),
      degree_(std::clamp(degree, 0, coeffs_.nmax)),
      order_(std::clamp(order, 0, std::clamp(degree, 0, coeffs_.nmax))),
      mu_(mu),
      re_(ref_radius) {
  // Validate the coefficient table shape up front (§3.6): the recursion indexes
  // C[n][m]/S[n][m] assuming a full triangular table of side nmax+1. A malformed
  // table (e.g. a default-constructed or half-populated loader result) would read
  // out of bounds — fail fast rather than corrupt memory.
  assert(coeffs_.nmax >= 0);
  assert(static_cast<int>(coeffs_.C.size()) == coeffs_.nmax + 1);
  assert(static_cast<int>(coeffs_.S.size()) == coeffs_.nmax + 1);
  for (int n = 0; n <= coeffs_.nmax; ++n) {
    const auto un = static_cast<std::size_t>(n);
    assert(static_cast<int>(coeffs_.C[un].size()) == n + 1);
    assert(static_cast<int>(coeffs_.S[un].size()) == n + 1);
  }
}

Eigen::Vector3d SphericalHarmonicGravity::gradient(const Eigen::Vector3d& r) const {
  std::vector<std::vector<double>> V, W;
  computeVW(r, re_, degree_ + 1, V, W);

  const double f = mu_ / (re_ * re_);
  double ax = 0.0, ay = 0.0, az = 0.0;
  for (int n = 0; n <= degree_; ++n) {
    const auto un = static_cast<std::size_t>(n);
    const int mmax = std::min(n, order_);
    for (int m = 0; m <= mmax; ++m) {
      const auto um = static_cast<std::size_t>(m);
      const double cnm = coeffs_.C[un][um];
      const double snm = coeffs_.S[un][um];
      if (m == 0) {
        ax += f * (-cnm * V[un + 1][1]);
        ay += f * (-cnm * W[un + 1][1]);
        az += f * ((n + 1) * (-cnm * V[un + 1][0]));
      } else {
        const double fac = static_cast<double>((n - m + 2) * (n - m + 1));
        ax += f * 0.5 *
              ((-cnm * V[un + 1][um + 1] - snm * W[un + 1][um + 1]) +
               fac * (cnm * V[un + 1][um - 1] + snm * W[un + 1][um - 1]));
        ay += f * 0.5 *
              ((-cnm * W[un + 1][um + 1] + snm * V[un + 1][um + 1]) +
               fac * (-cnm * W[un + 1][um - 1] + snm * V[un + 1][um - 1]));
        az += f * (n - m + 1) * (-cnm * V[un + 1][um] - snm * W[un + 1][um]);
      }
    }
  }
  return Eigen::Vector3d(ax, ay, az);
}

double SphericalHarmonicGravity::potential(const Eigen::Vector3d& r) const {
  // Same singular-radius guard as acceleration()/torque(): computeVW divides by
  // |r|^2, so r -> 0 would produce Inf/NaN instead of a finite value (§3.6).
  if (r.norm() < kMinRadius_) {
    return 0.0;
  }
  std::vector<std::vector<double>> V, W;
  computeVW(r, re_, degree_, V, W);
  double u = 0.0;
  for (int n = 0; n <= degree_; ++n) {
    const auto un = static_cast<std::size_t>(n);
    const int mmax = std::min(n, order_);
    for (int m = 0; m <= mmax; ++m) {
      const auto um = static_cast<std::size_t>(m);
      u += coeffs_.C[un][um] * V[un][um] + coeffs_.S[un][um] * W[un][um];
    }
  }
  return (mu_ / re_) * u;
}

math::Vec3<math::frames::ECI> SphericalHarmonicGravity::acceleration(
    const state::TruthState& s) const {
  const Eigen::Vector3d r = s.position.eigen();
  // Singular-radius guard: a real orbit never reaches r = 0 (§3.6).
  if (r.norm() < kMinRadius_) {
    return math::Vec3<math::frames::ECI>::Zero();
  }
  return math::Vec3<math::frames::ECI>(gradient(r));
}

math::Vec3<math::frames::Body> SphericalHarmonicGravity::torque(const state::TruthState& s) const {
  const Eigen::Vector3d r = s.position.eigen();
  const double rn = r.norm();
  if (rn < kMinRadius_) {
    return math::Vec3<math::frames::Body>::Zero();
  }
  // Nadir (spacecraft -> geocenter) unit vector in Body, via A = Body<-ECI.
  const Eigen::Matrix3d A = s.attitude.core().toRotationMatrix();
  const Eigen::Vector3d c_hat = A * (-r / rn);
  const Eigen::Vector3d tau = (3.0 * mu_ / (rn * rn * rn)) * c_hat.cross(inertia_ * c_hat);
  return math::Vec3<math::frames::Body>(tau);
}

}  // namespace polaris::sim::world
