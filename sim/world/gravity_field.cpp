/// @file
/// @brief Fully-normalized, singularity-free spherical-harmonic gravity.
///
/// Two independent engines (see gravity_field.hpp @file):
///  - `gradient()`  ports the normalized Gottlieb algorithm from
///    NASA/TP-2016-218604 Appendix C.9 (`gottliebnorm.m`) [eckman2016;
///    gottlieb1993] — direct Cartesian acceleration, no 1/cos(phi), stable to
///    high degree and at the poles.
///  - `potential()` sums the geopotential (Eq. 1.20) with a normalized
///    associated-Legendre forward-column recursion [holmes2002] — independent
///    code so `a = grad U` cross-validates the two.

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

}  // namespace

GravityCoeffs GravityCoeffs::pointMass() {
  GravityCoeffs g;
  g.nmax = 0;
  g.C = {{1.0}};
  g.S = {{0.0}};
  return g;
}

GravityCoeffs GravityCoeffs::earthZonal() {
  // Fully-normalized zonal coefficients Cbar_{n,0} = -J_n / sqrt(2n+1)
  // (N_{n,0} = sqrt(2n+1); C_{n,0} = -J_n). [eckman2016 §2.1]
  GravityCoeffs g;
  g.nmax = 6;
  g.C = triangular(g.nmax);
  g.S = triangular(g.nmax);
  const double kJ[7] = {0.0, 0.0, kJ2, kJ3, kJ4, kJ5, kJ6};
  g.C[0][0] = 1.0;  // point mass
  g.C[1][0] = 0.0;  // origin at the geocenter -> no degree-1 term
  for (int n = 2; n <= 6; ++n) {
    g.C[static_cast<std::size_t>(n)][0] = -kJ[n] / std::sqrt(2.0 * n + 1.0);
  }
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
  buildNormTables();
}

void SphericalHarmonicGravity::buildNormTables() {
  // NASA/TP-2016-218604 App. C.9 precompute block, indexed by degree n (and order
  // m). Computed for n = 2..degree_+1 (the recursion reads one row ahead).
  const int nax = degree_;
  const auto dim = static_cast<std::size_t>(nax) + 2;
  norm1_.assign(dim, 0.0);
  norm2_.assign(dim, 0.0);
  norm11_.assign(dim, 0.0);
  normn10_.assign(dim, 0.0);
  norm1m_.assign(dim, std::vector<double>(dim, 0.0));
  norm2m_.assign(dim, std::vector<double>(dim, 0.0));
  normn1_.assign(dim, std::vector<double>(dim, 0.0));
  for (int n = 2; n <= nax + 1; ++n) {
    const auto un = static_cast<std::size_t>(n);
    norm1_[un] = std::sqrt((2.0 * n + 1.0) / (2.0 * n - 1.0));
    norm2_[un] = std::sqrt((2.0 * n + 1.0) / (2.0 * n - 3.0));
    norm11_[un] = std::sqrt((2.0 * n + 1.0) / (2.0 * n)) / (2.0 * n - 1.0);
    normn10_[un] = std::sqrt((n + 1.0) * n / 2.0);
    for (int m = 1; m <= n; ++m) {
      const auto um = static_cast<std::size_t>(m);
      norm1m_[un][um] = std::sqrt((n - m) * (2.0 * n + 1.0) / ((n + m) * (2.0 * n - 1.0)));
      norm2m_[un][um] = std::sqrt((n - m) * (n - m - 1.0) * (2.0 * n + 1.0) /
                                  ((n + m) * (n + m - 1.0) * (2.0 * n - 3.0)));
      normn1_[un][um] = std::sqrt((n + m + 1.0) * (n - m));
    }
  }
}

Eigen::Vector3d SphericalHarmonicGravity::gradient(const Eigen::Vector3d& r) const {
  // Faithful port of NASA/TP-2016-218604 App. C.9 `gottliebnorm.m` [eckman2016].
  // 1-based indexing kept to match the MATLAB source (index 0 unused). p(row,col)
  // holds the normalized derived Legendre function of degree row-1, order col-1
  // (the cos^m(phi) sectoral factor is carried by the ctil/stil direction-cosine
  // recursion, which is why there is no 1/cos(phi) anywhere). rnp = I (ECI eval).
  const int nax = degree_;
  const int mmax = order_;

  // Direction cosines (x/r, y/r, z/r) and radial ratios.
  const double rmag = r.norm();
  const double ri = 1.0 / rmag;
  const double xor_ = r.x() * ri;
  const double yor = r.y() * ri;
  const double zor = r.z() * ri;
  const double ep = zor;  // sin(phi)
  const double reor = re_ * ri;
  const double muor2 = mu_ * ri * ri;

  const auto dim = static_cast<std::size_t>(nax) + 5;
  std::vector<std::vector<double>> p(dim, std::vector<double>(dim, 0.0));
  std::vector<double> ctil(dim, 0.0), stil(dim, 0.0);

  // Sectoral (diagonal) normalized ALFs.
  p[1][1] = 1.0;
  p[2][2] = std::sqrt(3.0);  // norm
  for (int n = 2; n <= nax; ++n) {
    const auto ni = static_cast<std::size_t>(n) + 1;
    p[ni][ni] = norm11_[static_cast<std::size_t>(n)] * p[ni - 1][ni - 1] * (2.0 * n - 1.0);
  }

  ctil[1] = 1.0;
  stil[1] = 0.0;
  ctil[2] = xor_;
  stil[2] = yor;

  double sumh = 0.0, sumgm = 1.0, sumj = 0.0, sumk = 0.0;
  double reorn = reor;

  p[2][1] = std::sqrt(3.0) * ep;  // norm

  for (int n = 2; n <= nax; ++n) {
    const auto un = static_cast<std::size_t>(n);
    const auto ni = un + 1;
    reorn *= reor;
    const double n2m1 = 2.0 * n - 1.0;
    const int nm1 = n - 1;
    const int np1 = n + 1;

    p[ni][ni - 1] = normn1_[un][un - 1] * ep * p[ni][ni];  // norm
    p[ni][1] = (n2m1 * ep * norm1_[un] * p[ni - 1][1] - nm1 * norm2_[un] * p[ni - 2][1]) / n;
    p[ni][2] =
        (n2m1 * ep * norm1m_[un][1] * p[ni - 1][2] - n * norm2m_[un][1] * p[ni - 2][2]) / nm1;
    double sumhn = normn10_[un] * p[ni][2] * coeffs_.C[un][0];  // norm
    double sumgmn = p[ni][1] * coeffs_.C[un][0] * np1;

    if (mmax > 0) {
      // Fill tesseral columns only up to the highest order the sum below reads
      // (order mmax+1, from the sumhn m+1 term) — columns beyond that are never
      // used, so a low-order/high-degree field costs O(degree*order), not
      // O(degree^2). Each column's vertical recursion is independent of higher
      // ones, so truncating is exact.
      const int fill_max = std::min(n - 2, mmax + 1);
      for (int m = 2; m <= fill_max; ++m) {
        const auto um = static_cast<std::size_t>(m);
        p[ni][um + 1] = (n2m1 * ep * norm1m_[un][um] * p[ni - 1][um + 1] -
                         (nm1 + m) * norm2m_[un][um] * p[ni - 2][um + 1]) /
                        (n - m);  // norm
      }
      double sumjn = 0.0, sumkn = 0.0;
      ctil[ni] = ctil[2] * ctil[ni - 1] - stil[2] * stil[ni - 1];
      stil[ni] = stil[2] * ctil[ni - 1] + ctil[2] * stil[ni - 1];
      const int lim = (n < mmax) ? n : mmax;
      for (int m = 1; m <= lim; ++m) {
        const auto um = static_cast<std::size_t>(m);
        const auto mi = um + 1;
        const double mxpnm = m * p[ni][mi];
        const double cnm = coeffs_.C[un][um];
        const double snm = coeffs_.S[un][um];
        const double bnmtil = cnm * ctil[mi] + snm * stil[mi];
        sumhn += normn1_[un][um] * p[ni][mi + 1] * bnmtil;  // norm
        sumgmn += (n + m + 1) * p[ni][mi] * bnmtil;
        const double bnmtm1 = cnm * ctil[mi - 1] + snm * stil[mi - 1];
        const double anmtm1 = cnm * stil[mi - 1] - snm * ctil[mi - 1];
        sumjn += mxpnm * bnmtm1;
        sumkn -= mxpnm * anmtm1;
      }
      sumj += reorn * sumjn;
      sumk += reorn * sumkn;
    }
    sumh += reorn * sumhn;
    sumgm += reorn * sumgmn;
  }

  const double lambda = sumgm + ep * sumh;
  return Eigen::Vector3d(-muor2 * (lambda * xor_ - sumj), -muor2 * (lambda * yor - sumk),
                         -muor2 * (lambda * zor - sumh));
}

double SphericalHarmonicGravity::potential(const Eigen::Vector3d& r) const {
  // Independent engine (cross-checks gradient() by a = grad U): geopotential
  // Eq. 1.20 summed with the Holmes-Featherstone normalized-ALF forward-column
  // recursion [holmes2002], stable to very high degree. Sectoral seeds multiply
  // by cos(phi) (never divide), so it is finite at the poles too.
  const double rmag = r.norm();
  if (rmag < kMinRadius_) {
    return 0.0;  // singular-radius guard (§3.6): match gradient()/torque()
  }
  const int nax = degree_;
  const double sphi = r.z() / rmag;  // sin(phi)
  const double rxy = std::hypot(r.x(), r.y());
  const double cphi = rxy / rmag;  // cos(phi) >= 0
  // Longitude direction (cos lambda, sin lambda); at a pole cos(phi)=0 kills every
  // m>=1 term, so the exact value is irrelevant — pick a finite placeholder.
  const double cl = (rxy > 0.0) ? r.x() / rxy : 1.0;
  const double sl = (rxy > 0.0) ? r.y() / rxy : 0.0;

  // cos(m.lambda), sin(m.lambda) by Chebyshev recursion.
  std::vector<double> cml(static_cast<std::size_t>(nax) + 1, 0.0);
  std::vector<double> sml(static_cast<std::size_t>(nax) + 1, 0.0);
  cml[0] = 1.0;
  sml[0] = 0.0;
  if (nax >= 1) {
    cml[1] = cl;
    sml[1] = sl;
  }
  for (int m = 2; m <= nax; ++m) {
    const auto um = static_cast<std::size_t>(m);
    cml[um] = 2.0 * cl * cml[um - 1] - cml[um - 2];
    sml[um] = 2.0 * cl * sml[um - 1] - sml[um - 2];
  }

  // Normalized ALF table Pbar[n][m](sin phi), Holmes-Featherstone forward column.
  std::vector<std::vector<double>> pbar = triangular(nax);
  pbar[0][0] = 1.0;
  if (nax >= 1) {
    pbar[1][1] = std::sqrt(3.0) * cphi;
    pbar[1][0] = std::sqrt(3.0) * sphi;
  }
  for (int m = 2; m <= nax; ++m) {  // sectoral diagonal
    const auto um = static_cast<std::size_t>(m);
    pbar[um][um] = std::sqrt((2.0 * m + 1.0) / (2.0 * m)) * cphi * pbar[um - 1][um - 1];
  }
  for (int m = 0; m <= nax; ++m) {
    const auto um = static_cast<std::size_t>(m);
    if (m + 1 <= nax) {  // first sub-diagonal
      pbar[um + 1][um] = std::sqrt(2.0 * m + 3.0) * sphi * pbar[um][um];
    }
    for (int n = m + 2; n <= nax; ++n) {
      const auto un = static_cast<std::size_t>(n);
      const double anm = std::sqrt((2.0 * n - 1.0) * (2.0 * n + 1.0) / ((n - m) * (n + m)));
      const double bnm = std::sqrt((2.0 * n + 1.0) * (n + m - 1.0) * (n - m - 1.0) /
                                   ((2.0 * n - 3.0) * (n - m) * (n + m)));
      pbar[un][um] = anm * sphi * pbar[un - 1][um] - bnm * pbar[un - 2][um];
    }
  }

  double sum = 0.0;
  double reorn = 1.0;  // (Re/r)^n
  const double reor = re_ / rmag;
  for (int n = 0; n <= nax; ++n) {
    const auto un = static_cast<std::size_t>(n);
    const int lim = std::min(n, order_);
    double inner = 0.0;
    for (int m = 0; m <= lim; ++m) {
      const auto um = static_cast<std::size_t>(m);
      inner += pbar[un][um] * (coeffs_.C[un][um] * cml[um] + coeffs_.S[un][um] * sml[um]);
    }
    sum += reorn * inner;
    reorn *= reor;
  }
  return (mu_ / rmag) * sum;
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
