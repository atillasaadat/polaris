/// @file Unit tests for the EGM2008 `.gfc` loader and ECEF-frame gravity
/// evaluation (REQ-SIM-002; design doc §5.2, §3.7).
///
/// Two concerns: (1) the ICGEM `.gfc` parser fills the triangular `GravityCoeffs`
/// table correctly — header scalars, Fortran `D` exponents, degree truncation, and
/// the forced `Cbar_00 = 1`; (2) tesseral (m>0) terms are evaluated in the
/// Earth-fixed frame via the injected ECI->ECEF rotation, so the returned
/// acceleration equals `Rᵀ · grad U(R·r)` and differs from the naive ECI eval —
/// while a purely zonal field is left frame-invariant. The real EGM2008 file is
/// checked in the golden test (`tests/golden/egm2008_golden_test.cpp`).

#include <gtest/gtest.h>

#include <cmath>
#include <Eigen/Core>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>

#include "frames/eci_ecef.hpp"
#include "frames/eop.hpp"
#include "math/frames.hpp"
#include "math/quaternion.hpp"
#include "math/typed_vector.hpp"
#include "state/truth_state.hpp"
#include "time/timescales.hpp"
#include "world/egm2008.hpp"
#include "world/gravity_field.hpp"

namespace world = polaris::sim::world;
namespace pm = polaris::math;
namespace pmf = polaris::math::frames;
namespace pf = polaris::frames;
namespace pt = polaris::time;

namespace {

// A minimal but real-shaped ICGEM `.gfc`: GM uses a Fortran `D` exponent (must be
// accepted), radius an `E` exponent, plus a few fully-normalized coefficients.
constexpr const char* kGfc =
    "product_type                gravity_field\n"
    "modelname                   TEST\n"
    "earth_gravity_constant      3.986004415D+14\n"
    "radius                      0.63781363000000E+07\n"
    "max_degree                  4\n"
    "norm                        fully_normalized\n"
    "tide_system                 tide_free\n"
    "end_of_head\n"
    "gfc    0    0    1.000000000000E+00    0.000000000000E+00\n"
    "gfc    2    0   -4.841695000000E-04    0.000000000000E+00\n"
    "gfc    2    2    2.439383573000E-06   -1.400273000000E-06\n"
    "gfc    3    0    9.571612070000E-07    0.000000000000E+00\n"
    "gfc    4    0    5.399658666000E-07    0.000000000000E+00\n";

world::GravityCoeffs parse(const std::string& text, int max_degree, world::Egm2008Header* h) {
  std::istringstream in(text);
  return world::loadEgm2008Gfc(in, max_degree, h);
}

/// A truth state at an ECI position and a fixed epoch inside any EOP span.
polaris::state::TruthState at(const Eigen::Vector3d& r_eci) {
  polaris::state::TruthState s;
  s.position = pm::Vec3<pmf::ECI>(r_eci);
  s.epoch = pt::Tai::fromNanosecondsSinceEpoch(1'900'000'000'000'000'000LL);  // ~2030
  return s;
}

}  // namespace

TEST(Egm2008Loader, ParsesHeaderCoefficientsAndFortranExponent) {
  RecordProperty("verifies", "REQ-SIM-002");
  world::Egm2008Header h;
  const world::GravityCoeffs g = parse(kGfc, 10, &h);

  EXPECT_NEAR(h.gm, 3.986004415e14, 1.0);  // 'D+14' exponent accepted
  EXPECT_NEAR(h.radius, 6378136.3, 1e-3);
  EXPECT_EQ(h.max_degree, 4);
  EXPECT_EQ(g.nmax, 4);  // min(requested 10, file 4)

  EXPECT_DOUBLE_EQ(g.C[0][0], 1.0);
  EXPECT_DOUBLE_EQ(g.C[2][0], -4.841695e-4);
  EXPECT_DOUBLE_EQ(g.C[2][2], 2.439383573e-6);
  EXPECT_DOUBLE_EQ(g.S[2][2], -1.400273e-6);
  EXPECT_DOUBLE_EQ(g.C[3][0], 9.57161207e-7);
  EXPECT_DOUBLE_EQ(g.C[4][0], 5.399658666e-7);
  // Coefficients absent from the file stay zero.
  EXPECT_DOUBLE_EQ(g.C[1][0], 0.0);
  EXPECT_DOUBLE_EQ(g.C[3][1], 0.0);
  EXPECT_DOUBLE_EQ(g.S[4][4], 0.0);
}

TEST(Egm2008Loader, TruncatesToRequestedDegreeAndForcesMonopole) {
  RecordProperty("verifies", "REQ-SIM-002");
  const world::GravityCoeffs g = parse(kGfc, 2, nullptr);
  EXPECT_EQ(g.nmax, 2);                         // truncated below the file's degree 4
  EXPECT_EQ(g.C.size(), 3u);                    // rows 0..2 only
  EXPECT_DOUBLE_EQ(g.C[0][0], 1.0);             // monopole forced
  EXPECT_DOUBLE_EQ(g.C[2][2], 2.439383573e-6);  // kept
  // A degree-3 row does not exist in the truncated table.
  EXPECT_EQ(static_cast<int>(g.C.size()), g.nmax + 1);
}

TEST(Egm2008Loader, ThrowsWhenFileDoesNotCoverRequestedDegree) {
  RecordProperty("verifies", "REQ-SIM-002");
  // The header lies (max_degree 50) but the body only reaches degree 4. Asking
  // for degree 10 must fail loudly rather than silently zero-fill degrees 5..10 —
  // a truth-fidelity hazard (sim/CLAUDE.md).
  const std::string liar =
      "earth_gravity_constant 3.986004415E+14\nradius 6378136.3\n"
      "max_degree 50\nnorm fully_normalized\nend_of_head\n"
      "gfc 0 0 1.0 0.0\ngfc 2 0 -4.84e-4 0.0\ngfc 4 0 5.4e-7 0.0\n";
  EXPECT_THROW(parse(liar, 10, nullptr), std::runtime_error);
  EXPECT_NO_THROW(parse(liar, 4, nullptr));  // within actual coverage: fine
}

TEST(Egm2008Loader, ParsedZonalFieldReproducesEarthZonalAcceleration) {
  RecordProperty("verifies", "REQ-SIM-002");
  // Feed the loader a .gfc holding exactly the embedded J2..J6 (normalized), then
  // check its acceleration matches the earthZonal() field — cross-validates the
  // loader against an independent construction of the same physics.
  const world::GravityCoeffs ref = world::GravityCoeffs::earthZonal();
  std::ostringstream gfc;
  gfc << "earth_gravity_constant 3.986004418E+14\nradius 6378137.0\nmax_degree 6\n"
         "norm fully_normalized\nend_of_head\n";
  gfc << std::scientific << std::setprecision(15);
  for (int n = 0; n <= ref.nmax; ++n) {
    for (int m = 0; m <= n; ++m) {
      gfc << "gfc " << n << " " << m << " " << ref.C[n][m] << " " << ref.S[n][m] << "\n";
    }
  }
  const world::GravityCoeffs loaded = parse(gfc.str(), 6, nullptr);

  const world::SphericalHarmonicGravity g_ref(ref, Eigen::Matrix3d::Identity(), 6, 0);
  const world::SphericalHarmonicGravity g_loaded(loaded, Eigen::Matrix3d::Identity(), 6, 0);
  const Eigen::Vector3d r(6.9e6, 1.1e6, 2.3e6);
  const Eigen::Vector3d a_ref = g_ref.acceleration(at(r)).eigen();
  const Eigen::Vector3d a_loaded = g_loaded.acceleration(at(r)).eigen();
  EXPECT_LT((a_ref - a_loaded).norm(), a_ref.norm() * 1e-12);
}

TEST(EcefGravity, TesseralEvaluatedInEcefMatchesRotatedGradient) {
  RecordProperty("verifies", "REQ-SIM-002");
  // A field with real tesseral (m>0) content.
  world::GravityCoeffs c;
  c.nmax = 4;
  c.C = {{1.0},
         {0.0, 0.0},
         {-4.84e-4, -2.0e-10, 2.44e-6},
         {9.6e-7, 2.0e-6, 0.9e-6, 1.0e-6},
         {5.4e-7, -5.4e-7, 3.5e-7, 9.9e-7, -1.9e-7}};
  c.S = {{0.0},
         {0.0, 0.0},
         {0.0, 1.4e-9, -1.40e-6},
         {0.0, 2.5e-7, -0.6e-6, 1.4e-6},
         {0.0, -4.7e-7, 6.6e-7, -2.0e-7, 3.1e-7}};

  world::SphericalHarmonicGravity g(c, Eigen::Matrix3d::Identity(), 4, 4);
  const world::SphericalHarmonicGravity g_plain(c, Eigen::Matrix3d::Identity(), 4, 4);

  // A fixed, finite EOP; the reduction core always succeeds for finite inputs, so
  // this exercises the gravity rotation wiring, not the table lookup (tested
  // elsewhere).
  pf::EopValue eop;
  eop.ut1_minus_tai = -37.0;
  eop.xp_arcsec = 0.05;
  eop.yp_arcsec = 0.30;
  g.setEciToEcef([&](const pt::Tai& t, pm::Quat<pmf::ECEF, pmf::ECI>& q) {
    return pf::ecefFromEci(t, eop, q);
  });

  const Eigen::Vector3d r_eci(6.6e6, -2.9e6, 3.1e6);
  const polaris::state::TruthState s = at(r_eci);

  // Expected: rotate position into ECEF, take the (plain, ECI-labelled) gradient
  // there, rotate the acceleration back to ECI.
  pm::Quat<pmf::ECEF, pmf::ECI> q;
  ASSERT_TRUE(pf::ecefFromEci(s.epoch, eop, q));
  const pm::Vec3<pmf::ECEF> r_ecef = q.rotate(s.position);
  polaris::state::TruthState s_ecef = s;
  s_ecef.position = pm::Vec3<pmf::ECI>(r_ecef.eigen());  // feed r_ecef as the eval point
  const pm::Vec3<pmf::ECEF> a_ecef(g_plain.acceleration(s_ecef).eigen());
  const Eigen::Vector3d expected = q.inverse().rotate(a_ecef).eigen();

  const Eigen::Vector3d actual = g.acceleration(s).eigen();
  ASSERT_TRUE(actual.allFinite());
  EXPECT_LT((actual - expected).norm(), expected.norm() * 1e-12);

  // And the rotation actually matters: evaluating the tesseral field on the raw
  // ECI position (no provider) gives a materially different vector.
  const Eigen::Vector3d naive = g_plain.acceleration(s).eigen();
  EXPECT_GT((actual - naive).norm(), 1e-6);
}

TEST(EcefGravity, ZonalFieldIsFrameInvariantSoRotationIsSkipped) {
  RecordProperty("verifies", "REQ-SIM-002");
  // order 0: the field is axisymmetric, so installing an ECEF rotation must not
  // change the result (the model skips the rotation for a purely zonal field).
  world::SphericalHarmonicGravity g(world::GravityCoeffs::earthZonal(), Eigen::Matrix3d::Identity(),
                                    6, 0);
  const Eigen::Vector3d r(5.5e6, 4.1e6, -2.7e6);
  const Eigen::Vector3d before = g.acceleration(at(r)).eigen();

  pf::EopValue eop;
  eop.ut1_minus_tai = -37.0;
  eop.xp_arcsec = 0.20;
  eop.yp_arcsec = -0.10;
  g.setEciToEcef([&](const pt::Tai& t, pm::Quat<pmf::ECEF, pmf::ECI>& q) {
    return pf::ecefFromEci(t, eop, q);
  });
  const Eigen::Vector3d after = g.acceleration(at(r)).eigen();
  EXPECT_EQ(before, after);  // bit-identical: rotation path not taken for order 0
}
