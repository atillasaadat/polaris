/// @file Tests for spherical-harmonic gravity + gravity-gradient torque (REQ-SIM-002).
///
/// Layered so a defect localizes:
///  - degree 0 must reproduce point-mass gravity exactly;
///  - the degree-2 zonal term must match the closed-form J2 acceleration;
///  - the full recursion (incl. tesserals) must satisfy a = grad U, checked by
///    finite-differencing the potential — one test covering every order m;
///  - propagated through the plant, J2 must produce the analytic nodal regression;
///  - the gravity-gradient torque must match a hand-worked case and vanish in its
///    two degenerate configurations (nadir on a principal axis; isotropic inertia).

#include <gtest/gtest.h>

#include <cmath>
#include <Eigen/Core>
#include <random>
#include <vector>

#include "constants/constants.hpp"
#include "dynamics/force_torque.hpp"
#include "dynamics/integrator.hpp"
#include "dynamics/rigid_body.hpp"
#include "math/frames.hpp"
#include "math/quaternion.hpp"
#include "state/truth_state.hpp"
#include "world/gravity_field.hpp"

namespace world = polaris::sim::world;
namespace dyn = polaris::sim::dynamics;
namespace pf = polaris::math::frames;
namespace pm = polaris::math;
namespace ps = polaris::state;

namespace {

constexpr double kMu = polaris::constants::wgs84::kGM;
constexpr double kRe = polaris::constants::wgs84::kSemiMajorAxis;

ps::TruthState at(const Eigen::Vector3d& r) {
  ps::TruthState s;
  s.position = pm::Vec3<pf::ECI>(Eigen::Vector3d(r));
  return s;
}

// Independent oracle: the UNNORMALIZED Cunningham/Montenbruck-Gill V/W recursion
// (M&G §3.2.4, the engine this push replaced). It is numerically exact at low
// degree, and — like Gottlieb — Cartesian and singularity-free (it divides only
// by |r|^2, never cos(phi)), so it is also a valid pole oracle. Used to check
// that the new normalized engine reproduces the physical field: feed it
// UNNORMALIZED coefficients (C_{n,0} = -J_n) while the class holds normalized ones.
Eigen::Vector3d unnormAccel(const std::vector<std::vector<double>>& C,
                            const std::vector<std::vector<double>>& S, int degree, int order,
                            const Eigen::Vector3d& r, double mu, double re) {
  const int N = degree + 1;
  const auto dim = static_cast<std::size_t>(N) + 1;
  std::vector<std::vector<double>> V(dim, std::vector<double>(dim, 0.0));
  std::vector<std::vector<double>> W(dim, std::vector<double>(dim, 0.0));
  const double r2 = r.squaredNorm();
  const double rinv2 = 1.0 / r2;
  const double xf = r.x() * re * rinv2, yf = r.y() * re * rinv2, zf = r.z() * re * rinv2;
  const double rf = re * re * rinv2;
  V[0][0] = re / std::sqrt(r2);
  for (int m = 1; m <= N; ++m) {
    const auto um = static_cast<std::size_t>(m);
    const double c = 2.0 * m - 1.0;
    V[um][um] = c * (xf * V[um - 1][um - 1] - yf * W[um - 1][um - 1]);
    W[um][um] = c * (xf * W[um - 1][um - 1] + yf * V[um - 1][um - 1]);
  }
  for (int m = 0; m <= N; ++m) {
    const auto um = static_cast<std::size_t>(m);
    for (int n = m + 1; n <= N; ++n) {
      const auto un = static_cast<std::size_t>(n);
      V[un][um] = (2.0 * n - 1.0) / (n - m) * zf * V[un - 1][um];
      W[un][um] = (2.0 * n - 1.0) / (n - m) * zf * W[un - 1][um];
      if (n - 2 >= m) {
        const double b = static_cast<double>(n + m - 1) / (n - m);
        V[un][um] -= b * rf * V[un - 2][um];
        W[un][um] -= b * rf * W[un - 2][um];
      }
    }
  }
  const double f = mu / (re * re);
  double ax = 0.0, ay = 0.0, az = 0.0;
  for (int n = 0; n <= degree; ++n) {
    const auto un = static_cast<std::size_t>(n);
    for (int m = 0; m <= std::min(n, order); ++m) {
      const auto um = static_cast<std::size_t>(m);
      const double cnm = C[un][um], snm = S[un][um];
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

// --- Degree 0 == point mass -------------------------------------------------

TEST(SphericalHarmonicGravity, DegreeZeroReducesToPointMass) {
  RecordProperty("verifies", "REQ-SIM-002");
  const world::SphericalHarmonicGravity g(world::GravityCoeffs::earthZonal(),
                                          Eigen::Matrix3d::Identity(), 0, 0);
  const dyn::TwoBodyGravity two_body(kMu);

  for (const Eigen::Vector3d& r :
       {Eigen::Vector3d(7.0e6, 0.0, 0.0), Eigen::Vector3d(3.0e6, -4.0e6, 5.0e6),
        Eigen::Vector3d(0.0, 0.0, 6.9e6)}) {
    const Eigen::Vector3d a = g.acceleration(at(r)).eigen();
    const Eigen::Vector3d a_ref = two_body.acceleration(at(r)).eigen();
    EXPECT_LT((a - a_ref).norm(), a_ref.norm() * 1.0e-9);
  }
}

// --- Degree 2 zonal == closed-form J2 ---------------------------------------

TEST(SphericalHarmonicGravity, J2TermMatchesClosedForm) {
  RecordProperty("verifies", "REQ-SIM-002");
  const world::SphericalHarmonicGravity g(world::GravityCoeffs::earthZonal(),
                                          Eigen::Matrix3d::Identity(), 2, 0);

  for (const Eigen::Vector3d& r :
       {Eigen::Vector3d(7.0e6, 0.0, 0.0), Eigen::Vector3d(3.0e6, -4.0e6, 5.0e6)}) {
    const double rn = r.norm();
    const Eigen::Vector3d pointmass = -kMu * r / (rn * rn * rn);
    const Eigen::Vector3d a_j2 = g.acceleration(at(r)).eigen() - pointmass;

    // Vallado §8: a_J2 = -3/2 J2 mu Re^2 / r^5 [(1-5z^2/r^2)x, (..)y, (3-5z^2/r^2)z].
    const double zr2 = (r.z() * r.z()) / (rn * rn);
    const double k = -1.5 * world::kJ2 * kMu * kRe * kRe / std::pow(rn, 5.0);
    const Eigen::Vector3d expected(k * (1.0 - 5.0 * zr2) * r.x(), k * (1.0 - 5.0 * zr2) * r.y(),
                                   k * (3.0 - 5.0 * zr2) * r.z());
    EXPECT_LT((a_j2 - expected).norm(), expected.norm() * 1.0e-9);
  }
}

// --- a = grad U for the full recursion, tesserals included ------------------

TEST(SphericalHarmonicGravity, AccelerationIsGradientOfPotential) {
  RecordProperty("verifies", "REQ-SIM-002");
  // A synthetic degree/order-2 field with genuine tesseral (m>0) terms — the only
  // way to exercise the m>0 branch of the M&G acceleration recursion. If a = grad U
  // holds here, every order is wired correctly.
  world::GravityCoeffs c;
  c.nmax = 2;
  c.C = {{1.0}, {0.0, 0.0}, {-1.0e-3, 2.0e-4, 1.5e-4}};
  c.S = {{0.0}, {0.0, 0.0}, {0.0, -1.0e-4, 3.0e-5}};
  const world::SphericalHarmonicGravity g(c, Eigen::Matrix3d::Identity(), 2, 2);

  const Eigen::Vector3d r(4.0e6, -2.0e6, 3.0e6);
  const Eigen::Vector3d a = g.acceleration(at(r)).eigen();

  const double h = 50.0;  // central difference; balances cancellation vs truncation
  Eigen::Vector3d numeric;
  for (int i = 0; i < 3; ++i) {
    Eigen::Vector3d rp = r, rm = r;
    rp[i] += h;
    rm[i] -= h;
    numeric[i] = (g.potential(rp) - g.potential(rm)) / (2.0 * h);
  }
  EXPECT_LT((a - numeric).norm(), a.norm() * 1.0e-6);
}

// --- J2 nodal regression through the plant ----------------------------------

TEST(SphericalHarmonicGravity, J2CausesAnalyticNodalRegression) {
  RecordProperty("verifies", "REQ-SIM-002");
  // The signature of J2: the orbit plane regresses at Omega_dot = -3/2 n J2 (Re/p)^2 cos i.
  // Track the azimuth of the angular-momentum vector (which rotates at Omega_dot).
  const double a = 7.0e6;
  const double inc = M_PI / 4.0;  // 45 deg
  const double v = std::sqrt(kMu / a);
  const double n = std::sqrt(kMu / (a * a * a));
  const double period = 2.0 * M_PI / n;

  ps::TruthState s;
  s.position = pm::Vec3<pf::ECI>(a, 0.0, 0.0);
  s.velocity = pm::Vec3<pf::ECI>(0.0, v * std::cos(inc), v * std::sin(inc));

  const world::SphericalHarmonicGravity gravity(world::GravityCoeffs::earthZonal(),
                                                Eigen::Matrix3d::Identity(), 2, 0);
  const dyn::RigidBody6Dof body(Eigen::Matrix3d::Identity(), gravity);

  auto h_azimuth = [](const ps::TruthState& x) {
    const Eigen::Vector3d h = x.position.eigen().cross(x.velocity.eigen());
    return std::atan2(h.y(), h.x());
  };

  const double az0 = h_azimuth(s);
  dyn::StepControl ctl;
  ctl.rel_tol = 1.0e-10;
  ctl.max_step = 120.0;
  const double span = 10.0 * period;
  const ps::TruthState s1 = body.propagate(s, span, ctl);
  const double az1 = h_azimuth(s1);

  const double p = a;  // circular
  const double omega_dot = -1.5 * n * world::kJ2 * (kRe / p) * (kRe / p) * std::cos(inc);
  const double expected = omega_dot * span;
  const double measured = std::remainder(az1 - az0, 2.0 * M_PI);
  EXPECT_NEAR(measured, expected, std::abs(expected) * 0.06);
}

// --- Gravity-gradient torque ------------------------------------------------

TEST(SphericalHarmonicGravity, GravityGradientTorqueMatchesHandWorkedCase) {
  RecordProperty("verifies", "REQ-SIM-002");
  // Identity attitude, r = -R (1,1,1)/sqrt(3) so the Body nadir is c_hat = (1,1,1)/sqrt(3).
  // With J = diag(10,20,30): tau = 3 mu/R^3 * c_hat x (J c_hat) = mu/R^3 (10,-20,10).
  const double R = 7.0e6;
  const Eigen::Matrix3d J = Eigen::Vector3d(10.0, 20.0, 30.0).asDiagonal();
  const world::SphericalHarmonicGravity g(world::GravityCoeffs::pointMass(), J, 0, 0);

  ps::TruthState s;
  s.position = pm::Vec3<pf::ECI>(Eigen::Vector3d(-R * Eigen::Vector3d(1, 1, 1).normalized()));
  // attitude defaults to identity (Body <- ECI).

  const Eigen::Vector3d tau = g.torque(s).eigen();
  const Eigen::Vector3d expected = (kMu / (R * R * R)) * Eigen::Vector3d(10.0, -20.0, 10.0);
  EXPECT_LT((tau - expected).norm(), expected.norm() * 1.0e-12);
}

TEST(SphericalHarmonicGravity, GravityGradientTorqueUsesAttitudeRotation) {
  RecordProperty("verifies", "REQ-SIM-002");
  // Exercise the A = Body<-ECI rotation with a genuine (non-identity) attitude, so
  // an A <-> A^transpose mix-up cannot pass silently. Attitude = ROT3(30 deg) about
  // the ECI Z axis, r along +X so nadir_ECI = (-1,0,0). With A = ROT3(+30):
  //   c_hat = A(-x_hat) = (-cos30, sin30, 0) = (-sqrt3/2, 1/2, 0).
  // J = diag(10,20,30) -> J c_hat = (-5 sqrt3, 10, 0), and
  //   c_hat x (J c_hat) = (0, 0, -5 sqrt3 / 2)  (sign flips under A^transpose).
  const double R = 7.0e6;
  const Eigen::Matrix3d J = Eigen::Vector3d(10.0, 20.0, 30.0).asDiagonal();
  const world::SphericalHarmonicGravity g(world::GravityCoeffs::pointMass(), J, 0, 0);

  ps::TruthState s;
  s.position = pm::Vec3<pf::ECI>(R, 0.0, 0.0);
  s.attitude = pm::Quat<pf::Body, pf::ECI>(
      pm::Quaternion::FromAxisAngle(Eigen::Vector3d(0.0, 0.0, 1.0), M_PI / 6.0));

  const Eigen::Vector3d tau = g.torque(s).eigen();
  const Eigen::Vector3d expected =
      (3.0 * kMu / (R * R * R)) * Eigen::Vector3d(0.0, 0.0, -5.0 * std::sqrt(3.0) / 2.0);
  EXPECT_LT((tau - expected).norm(), expected.norm() * 1.0e-9);
}

TEST(SphericalHarmonicGravity, GravityGradientTorqueVanishesInDegenerateCases) {
  RecordProperty("verifies", "REQ-SIM-002");
  const double R = 7.0e6;
  ps::TruthState s;
  s.position = pm::Vec3<pf::ECI>(R, 0.0, 0.0);  // nadir along -X (a principal axis)

  // (a) Nadir aligned with a principal axis -> J c_hat parallel to c_hat -> zero.
  const Eigen::Matrix3d J = Eigen::Vector3d(10.0, 20.0, 30.0).asDiagonal();
  const world::SphericalHarmonicGravity aligned(world::GravityCoeffs::pointMass(), J, 0, 0);
  EXPECT_LT(aligned.torque(s).eigen().norm(), 1.0e-20);

  // (b) Isotropic inertia -> J c_hat parallel to c_hat for any attitude -> zero.
  s.position = pm::Vec3<pf::ECI>(Eigen::Vector3d(-R * Eigen::Vector3d(1, 1, 1).normalized()));
  const world::SphericalHarmonicGravity isotropic(world::GravityCoeffs::pointMass(),
                                                  5.0 * Eigen::Matrix3d::Identity(), 0, 0);
  EXPECT_LT(isotropic.torque(s).eigen().norm(), 1.0e-20);
}

// --- Degree/order clamping --------------------------------------------------

TEST(SphericalHarmonicGravity, DegreeAndOrderClampToTable) {
  RecordProperty("verifies", "REQ-SIM-002");
  const world::SphericalHarmonicGravity g(world::GravityCoeffs::earthZonal(),
                                          Eigen::Matrix3d::Identity(), 99, 99);
  EXPECT_EQ(g.degree(), 6);  // clamped to nmax
  EXPECT_EQ(g.order(), 6);   // clamped to degree
}

// --- Normalized low-degree engine matches an independent unnormalized oracle -

TEST(SphericalHarmonicGravity, NormalizedZonalMatchesUnnormalizedOracle) {
  RecordProperty("verifies", "REQ-SIM-002");
  // The whole point of the normalization: the normalized Gottlieb engine must
  // reproduce the SAME physical field as the unnormalized V/W recursion fed the
  // raw C_{n,0} = -J_n. Agreement to ~1e-9 pins the sqrt(2n+1) zonal factors.
  const world::SphericalHarmonicGravity g(world::GravityCoeffs::earthZonal(),
                                          Eigen::Matrix3d::Identity(), 6, 0);
  // Unnormalized zonal table for the oracle.
  std::vector<std::vector<double>> C(7), Sc(7);
  for (int n = 0; n <= 6; ++n) {
    C[static_cast<std::size_t>(n)].assign(static_cast<std::size_t>(n) + 1, 0.0);
    Sc[static_cast<std::size_t>(n)].assign(static_cast<std::size_t>(n) + 1, 0.0);
  }
  C[0][0] = 1.0;
  const double Jn[7] = {0.0, 0.0, world::kJ2, world::kJ3, world::kJ4, world::kJ5, world::kJ6};
  for (int n = 2; n <= 6; ++n)
    C[static_cast<std::size_t>(n)][0] = -Jn[n];

  for (const Eigen::Vector3d& r :
       {Eigen::Vector3d(7.0e6, 0.0, 0.0), Eigen::Vector3d(3.0e6, -4.0e6, 5.0e6),
        Eigen::Vector3d(-2.0e6, 1.0e6, 6.5e6)}) {
    const Eigen::Vector3d a = g.acceleration(at(r)).eigen();
    const Eigen::Vector3d ref = unnormAccel(C, Sc, 6, 0, r, kMu, kRe);
    EXPECT_LT((a - ref).norm(), ref.norm() * 1.0e-9);
  }
}

// --- a = grad U at 200x200 with a full random tesseral field ----------------

TEST(SphericalHarmonicGravity, AccelerationIsGradientAt200x200) {
  RecordProperty("verifies", "REQ-SIM-002");
  // The core high-degree correctness test: build a deterministic (seeded, never
  // wall-clock) full Cbar/Sbar table to 200x200 with small magnitudes, then check
  // the normalized Gottlieb acceleration equals the central finite difference of
  // the independently-coded normalized potential. Only a stable normalized
  // recursion survives this — the old unnormalized V/W overflows well before 200.
  constexpr int kN = 200;
  world::GravityCoeffs c;
  c.nmax = kN;
  c.C.resize(static_cast<std::size_t>(kN) + 1);
  c.S.resize(static_cast<std::size_t>(kN) + 1);
  std::mt19937_64 rng(0xC0FFEEULL);  // fixed seed: bit-reproducible
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  for (int n = 0; n <= kN; ++n) {
    const auto un = static_cast<std::size_t>(n);
    c.C[un].assign(un + 1, 0.0);
    c.S[un].assign(un + 1, 0.0);
    for (int m = 0; m <= n; ++m) {
      const auto um = static_cast<std::size_t>(m);
      c.C[un][um] = 1.0e-6 * dist(rng);
      c.S[un][um] = 1.0e-6 * dist(rng);
    }
  }
  c.C[0][0] = 1.0;                          // point-mass leading term
  c.C[1][0] = c.C[1][1] = c.S[1][1] = 0.0;  // geocenter: no degree-1

  const world::SphericalHarmonicGravity g(c, Eigen::Matrix3d::Identity(), kN, kN);
  EXPECT_EQ(g.degree(), kN);
  EXPECT_EQ(g.order(), kN);

  const Eigen::Vector3d r(4.2e6, -3.1e6, 4.6e6);  // |r| ~ 7.0e6, mid latitude
  const Eigen::Vector3d a = g.acceleration(at(r)).eigen();
  ASSERT_TRUE(a.allFinite());

  const double h = 50.0;
  Eigen::Vector3d numeric;
  for (int i = 0; i < 3; ++i) {
    Eigen::Vector3d rp = r, rm = r;
    rp[i] += h;
    rm[i] -= h;
    numeric[i] = (g.potential(rp) - g.potential(rm)) / (2.0 * h);
  }
  EXPECT_LT((a - numeric).norm(), a.norm() * 1.0e-6);
}

// --- Non-square field: order << degree (tesseral column truncation) ---------

TEST(SphericalHarmonicGravity, AccelerationIsGradientNonSquare200x5) {
  RecordProperty("verifies", "REQ-SIM-002");
  // A high-degree, low-order (200x5) field exercises the order<degree tesseral
  // path where gradient() truncates the Legendre column fill at order mmax+1.
  // a = grad U must still hold: potential() truncates at the same order, so the
  // two independent engines agree only if the truncation is exact.
  constexpr int kN = 200;
  constexpr int kM = 5;
  world::GravityCoeffs c;
  c.nmax = kN;
  c.C.resize(static_cast<std::size_t>(kN) + 1);
  c.S.resize(static_cast<std::size_t>(kN) + 1);
  std::mt19937_64 rng(0xFEEDFACEULL);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  for (int n = 0; n <= kN; ++n) {
    const auto un = static_cast<std::size_t>(n);
    c.C[un].assign(un + 1, 0.0);
    c.S[un].assign(un + 1, 0.0);
    for (int m = 0; m <= n; ++m) {
      c.C[un][static_cast<std::size_t>(m)] = 1.0e-6 * dist(rng);
      c.S[un][static_cast<std::size_t>(m)] = 1.0e-6 * dist(rng);
    }
  }
  c.C[0][0] = 1.0;
  c.C[1][0] = c.C[1][1] = c.S[1][1] = 0.0;

  const world::SphericalHarmonicGravity g(c, Eigen::Matrix3d::Identity(), kN, kM);
  EXPECT_EQ(g.degree(), kN);
  EXPECT_EQ(g.order(), kM);

  const Eigen::Vector3d r(4.2e6, -3.1e6, 4.6e6);
  const Eigen::Vector3d a = g.acceleration(at(r)).eigen();
  ASSERT_TRUE(a.allFinite());
  const double h = 50.0;
  Eigen::Vector3d numeric;
  for (int i = 0; i < 3; ++i) {
    Eigen::Vector3d rp = r, rm = r;
    rp[i] += h;
    rm[i] -= h;
    numeric[i] = (g.potential(rp) - g.potential(rm)) / (2.0 * h);
  }
  EXPECT_LT((a - numeric).norm(), a.norm() * 1.0e-6);
}

// --- Finite (no NaN/Inf) at 200x200, including on the pole axis --------------

TEST(SphericalHarmonicGravity, FiniteAtPoleAndHighDegree) {
  RecordProperty("verifies", "REQ-SIM-002");
  // The singularity-free formulation's payoff: on the spin axis (cos(phi)=0) a
  // 1/cos(phi) formulation would blow up. Build a 200x200 random field and
  // evaluate exactly on +Z — every component must be finite.
  constexpr int kN = 200;
  world::GravityCoeffs c;
  c.nmax = kN;
  c.C.resize(static_cast<std::size_t>(kN) + 1);
  c.S.resize(static_cast<std::size_t>(kN) + 1);
  std::mt19937_64 rng(0xBADC0DEULL);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  for (int n = 0; n <= kN; ++n) {
    const auto un = static_cast<std::size_t>(n);
    c.C[un].assign(un + 1, 0.0);
    c.S[un].assign(un + 1, 0.0);
    for (int m = 0; m <= n; ++m) {
      c.C[un][static_cast<std::size_t>(m)] = 1.0e-6 * dist(rng);
      c.S[un][static_cast<std::size_t>(m)] = 1.0e-6 * dist(rng);
    }
  }
  c.C[0][0] = 1.0;
  const world::SphericalHarmonicGravity g(c, Eigen::Matrix3d::Identity(), kN, kN);

  // On the +Z axis (x=y=0) and a hair off it — both must be finite.
  for (const Eigen::Vector3d& r :
       {Eigen::Vector3d(0.0, 0.0, 7.0e6), Eigen::Vector3d(1.0e-3, 1.0e-3, 7.0e6)}) {
    const Eigen::Vector3d a = g.acceleration(at(r)).eigen();
    EXPECT_TRUE(a.allFinite());
    EXPECT_TRUE(std::isfinite(g.potential(r)));
  }

  // At the exact pole, a zonal-only field must give the closed-form zonal
  // acceleration (matched by the singularity-free unnormalized oracle).
  const world::SphericalHarmonicGravity gz(world::GravityCoeffs::earthZonal(),
                                           Eigen::Matrix3d::Identity(), 6, 0);
  std::vector<std::vector<double>> Cz(7), Sz(7);
  for (int n = 0; n <= 6; ++n) {
    Cz[static_cast<std::size_t>(n)].assign(static_cast<std::size_t>(n) + 1, 0.0);
    Sz[static_cast<std::size_t>(n)].assign(static_cast<std::size_t>(n) + 1, 0.0);
  }
  Cz[0][0] = 1.0;
  const double Jn[7] = {0.0, 0.0, world::kJ2, world::kJ3, world::kJ4, world::kJ5, world::kJ6};
  for (int n = 2; n <= 6; ++n)
    Cz[static_cast<std::size_t>(n)][0] = -Jn[n];

  const Eigen::Vector3d rp(0.0, 0.0, 6.9e6);
  const Eigen::Vector3d a_pole = gz.acceleration(at(rp)).eigen();
  ASSERT_TRUE(a_pole.allFinite());
  const Eigen::Vector3d ref = unnormAccel(Cz, Sz, 6, 0, rp, kMu, kRe);
  EXPECT_LT((a_pole - ref).norm(), ref.norm() * 1.0e-9);
  EXPECT_LT(std::hypot(a_pole.x(), a_pole.y()), std::abs(a_pole.z()) * 1.0e-12);
}

}  // namespace
