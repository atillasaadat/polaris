/// @file
/// @brief The onboard low-degree geopotential evaluator (design doc §8.3).
///
/// `lib/gnc/geopotential.cpp` is a third, flight-side implementation of a field
/// the repo already evaluates two ways in `sim/world/gravity_field` (a
/// normalized-ALF potential sum and a normalized Gottlieb Cartesian assembly,
/// which cross-check each other by finite difference, and which GMAT
/// cross-validation certified). It exists because neither of those is flight
/// code — both carry heap-allocated coefficient and normalization tables sized
/// at construction — and its header states that trade.
///
/// A third implementation is only safe if it cannot drift from the two that were
/// validated, so the load-bearing test here is
/// `MatchesTheTruthSimFieldAtMatchedDegreeAndOrder`: the flight evaluator is
/// pinned against `sim::world::SphericalHarmonicGravity` over a spread of
/// positions, at every degree from 2 to the table's maximum. The analytic tests
/// around it check the two degenerate cases a cross-comparison cannot — degree 0
/// must be *exactly* point-mass and degree 2 order 0 must be *exactly* the
/// textbook closed-form J2 — because those are the forms `orbit_od`'s own
/// analytic tests are written against, and pinning both ends closes the chain.

#include "gnc/geopotential.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <string>
#include <vector>

#include "constants/constants.hpp"
#include "world/egm2008.hpp"
#include "world/gravity_field.hpp"

namespace {

namespace pg = polaris::gnc;
namespace pc = polaris::constants;
namespace pw = polaris::sim::world;

constexpr double kMu = pg::egm2008::kGm;
constexpr double kRe = pg::egm2008::kReferenceRadius;

/// A spread of ECEF positions chosen to exercise the recursion where it is most
/// likely to be wrong rather than where it is easiest: a pole (where a
/// `1/cos φ` formulation would divide by zero), the equator (where the sectoral
/// terms peak), a mid-latitude general point, and one well above LEO so the
/// `(Re/r)^n` attenuation is exercised in both directions.
std::vector<Eigen::Vector3d> testPositions() {
  const double r_leo = 6778137.0;
  const double r_meo = 2.0e7;
  return {
      Eigen::Vector3d(0.0, 0.0, r_leo),                             // north pole
      Eigen::Vector3d(0.0, 0.0, -r_leo),                            // south pole
      Eigen::Vector3d(r_leo, 0.0, 0.0),                             // equator, lambda = 0
      Eigen::Vector3d(0.0, r_leo, 0.0),                             // equator, lambda = 90 deg
      Eigen::Vector3d(-r_leo * 0.6, r_leo * 0.8, 0.0),              // equator, second quadrant
      Eigen::Vector3d(4.0e6, 3.0e6, 4.0e6).normalized() * r_leo,    // mid-latitude
      Eigen::Vector3d(-2.0e6, 5.0e6, -3.5e6).normalized() * r_leo,  // southern mid-latitude
      Eigen::Vector3d(1.0e7, -1.2e7, 8.0e6).normalized() * r_meo,   // above LEO
  };
}

/// Closed-form two-body acceleration.
Eigen::Vector3d pointMass(const Eigen::Vector3d& r) {
  const double rn = r.norm();
  return -(kMu / (rn * rn * rn)) * r;
}

/// The textbook J2 component form (Montenbruck & Gill §3.2; Vallado §8.7),
/// written about the ECEF Z axis because that is the frame the evaluator works
/// in. Deliberately a different expression from the one `orbit_od.cpp` flies —
/// that one is the vector identity about an arbitrary pole — so agreement here
/// is agreement between two independently-written forms.
Eigen::Vector3d closedFormJ2(const Eigen::Vector3d& r) {
  const double j2 = -pg::egm2008::kC[2][0];  // C_20 = -J2 by definition
  const double rn = r.norm();
  const double zr2 = 5.0 * r.z() * r.z() / (rn * rn);
  const double k = 1.5 * j2 * kMu * kRe * kRe / std::pow(rn, 5);
  return pointMass(r) -
         k * Eigen::Vector3d(r.x() * (1.0 - zr2), r.y() * (1.0 - zr2), r.z() * (3.0 - zr2));
}

}  // namespace

/// The stored C_20 must *be* -J2. If the de-normalization in
/// `tools/gravity/cxxtable.py` were wrong by its `sqrt((2-delta)(2n+1)(n-m)!/(n+m)!)`
/// factor, every test below would still be self-consistent — both the evaluator
/// and the closed forms above read the same table — and only this comparison
/// against an independently-published constant would catch it.
TEST(Geopotential, StoredCoefficientsCarryThePublishedZonalHarmonics) {
  RecordProperty("verifies", "REQ-ODP-005");

  // `constants::gravity::kJ_n` are the published unnormalized zonals the truth
  // sim's zonal field and the filter's closed-form J2 path both use. They come
  // from an earlier solution than EGM2008, so the two genuinely differ, and the
  // band has to be wide enough to admit that: measured, the relative differences
  // are 4.7e-7 (J2), 9.7e-5 (J3), 1.7e-4 (J4), 2.1e-3 (J5), 2.8e-5 (J6) — larger
  // for the higher, less well-determined harmonics, exactly as expected.
  //
  // 1% is still diagnostic for what this test is actually for. A wrong
  // de-normalization would be off by `sqrt((2-delta)(2n+1)(n-m)!/(n+m)!)`, which
  // for a zonal is `sqrt(2n+1)` — a factor of 2.2 at n = 2 and 3.6 at n = 6, i.e.
  // 120% to 260% wrong. There is no normalization mistake that lands inside 1%.
  constexpr double kSolutionSpread = 1.0e-2;
  EXPECT_NEAR(-pg::egm2008::kC[2][0], pc::gravity::kJ2, kSolutionSpread * pc::gravity::kJ2);
  EXPECT_NEAR(-pg::egm2008::kC[3][0], pc::gravity::kJ3,
              kSolutionSpread * std::abs(pc::gravity::kJ3));
  EXPECT_NEAR(-pg::egm2008::kC[4][0], pc::gravity::kJ4,
              kSolutionSpread * std::abs(pc::gravity::kJ4));
  EXPECT_NEAR(-pg::egm2008::kC[5][0], pc::gravity::kJ5,
              kSolutionSpread * std::abs(pc::gravity::kJ5));
  EXPECT_NEAR(-pg::egm2008::kC[6][0], pc::gravity::kJ6,
              kSolutionSpread * std::abs(pc::gravity::kJ6));

  // C_00 is the point-mass term and must be exactly one — not "close to".
  EXPECT_EQ(pg::egm2008::kC[0][0], 1.0);
  // Degree 1 is the geocentre offset, which is zero by the choice of origin. A
  // non-zero value here would displace the whole field.
  for (int m = 0; m <= 1; ++m) {
    EXPECT_EQ(pg::egm2008::kC[1][m], 0.0) << "m = " << m;
    EXPECT_EQ(pg::egm2008::kS[1][m], 0.0) << "m = " << m;
  }
  // S_n0 is identically zero: sin(0 * lambda) = 0, so the coefficient is not a
  // free parameter and a non-zero one would mean the parse mis-columned.
  for (int n = 0; n <= pg::kGeopotentialMaxDegree; ++n) {
    EXPECT_EQ(pg::egm2008::kS[n][0], 0.0) << "n = " << n;
  }
}

/// Degree 0 must be point-mass to machine precision — not approximately. The
/// harmonic sum supplies its own `GM/r` through C_00, and `orbit_od`'s two-body
/// tests are written against the closed form, so any discrepancy here would
/// show up there as a mysterious model difference.
TEST(Geopotential, DegreeZeroIsExactlyPointMass) {
  RecordProperty("verifies", "REQ-ODP-005");
  for (const Eigen::Vector3d& r : testPositions()) {
    const Eigen::Vector3d a = pg::geopotentialAcceleration(r, 0, 0, kMu, kRe);
    const Eigen::Vector3d expected = pointMass(r);
    EXPECT_LT((a - expected).norm(), 1.0e-15 * expected.norm()) << "r = " << r.transpose();
  }
}

/// Degree 2 order 0 must be the closed-form J2, likewise to machine precision.
/// This is the pin that lets `orbit_od_test.cpp` keep testing its closed-form J2
/// path analytically while the filter flies the harmonic one.
TEST(Geopotential, DegreeTwoOrderZeroIsExactlyTheClosedFormJ2) {
  RecordProperty("verifies", "REQ-ODP-005");
  for (const Eigen::Vector3d& r : testPositions()) {
    const Eigen::Vector3d a = pg::geopotentialAcceleration(r, 2, 0, kMu, kRe);
    const Eigen::Vector3d expected = closedFormJ2(r);
    EXPECT_LT((a - expected).norm(), 1.0e-13 * expected.norm()) << "r = " << r.transpose();
  }
}

/// The cross-validation that makes a third implementation safe.
///
/// `sim::world::SphericalHarmonicGravity` evaluates the same field by a
/// completely different route — fully-normalized coefficients through the
/// normalized Gottlieb recursion — and is the model GMAT cross-validation
/// certified. Agreement at every degree from 2 to the table maximum, over the
/// position spread, means the flight evaluator's recursion, its de-normalized
/// coefficients, and its acceleration assembly are all right together: a fault
/// in any one of the three moves the answer.
///
/// Note this compares `gradient()` output at the *same* degree AND order, so
/// the tesseral machinery is under test, not just the zonals.
TEST(Geopotential, MatchesTheTruthSimFieldAtMatchedDegreeAndOrder) {
  RecordProperty("verifies", "REQ-ODP-005");
  RecordProperty("verifies", "REQ-VV-002");

  pw::Egm2008Header header;
  const pw::GravityCoeffs coeffs = pw::loadEgm2008Gfc(
      std::string(POLARIS_GOLDEN_DIR) + "/EGM2008_to200.gfc", pg::kGeopotentialMaxDegree, &header);

  // The two must be handed the same constants or the comparison measures the
  // constants rather than the recursions. Both come from the same `.gfc` header,
  // which is exactly the point of the generated table carrying them.
  ASSERT_NEAR(header.gm, kMu, 1.0e-3);
  ASSERT_NEAR(header.radius, kRe, 1.0e-6);

  double worst_relative = 0.0;
  for (int degree = 2; degree <= pg::kGeopotentialMaxDegree; ++degree) {
    // No `setEciToEcef` resolver is installed, so the sim field evaluates on the
    // position as given — which is what is wanted here: both sides are handed an
    // ECEF position and neither rotates. The frame handling is `orbit_od`'s job
    // and is tested there.
    pw::SphericalHarmonicGravity field(coeffs, degree, degree, header.gm, header.radius);

    for (const Eigen::Vector3d& r : testPositions()) {
      const Eigen::Vector3d flight = pg::geopotentialAcceleration(r, degree, degree, kMu, kRe);

      // The sim engine's Cartesian assembly, reached through its potential
      // gradient by central difference — the same route its own self-check uses,
      // and independent of the flight assembly's algebra.
      const double h = 1.0;
      Eigen::Vector3d truth;
      for (int i = 0; i < 3; ++i) {
        Eigen::Vector3d plus = r;
        Eigen::Vector3d minus = r;
        plus(i) += h;
        minus(i) -= h;
        truth(i) = (field.potential(plus) - field.potential(minus)) / (2.0 * h);
      }

      const double relative = (flight - truth).norm() / truth.norm();
      worst_relative = std::max(worst_relative, relative);
      // The bound is set by the central difference in the *reference*, not by
      // either field: at h = 1 m the O(h^2 * U''') truncation against
      // double-precision cancellation in a ~6e7 m^2/s^2 potential lands near
      // 1e-9 relative. The recursions themselves agree far better than this.
      EXPECT_LT(relative, 5.0e-9) << "degree/order " << degree << ", r = " << r.transpose();
    }
  }
  RecordProperty("worst_relative_vs_truth_sim", std::to_string(worst_relative));
}

/// Guards, at the trust boundary. A flight evaluator that returned a NaN or an
/// infinity would defeat the filter's finiteness check by poisoning the state
/// before it runs; returning zero keeps the failure visible and recoverable.
TEST(Geopotential, DegenerateInputsReturnZeroRatherThanNonFinite) {
  RecordProperty("verifies", "REQ-ODP-005");
  const Eigen::Vector3d good(6778137.0, 0.0, 0.0);
  const double nan = std::numeric_limits<double>::quiet_NaN();

  EXPECT_TRUE(pg::geopotentialAcceleration(Eigen::Vector3d::Zero(), 8, 8, kMu, kRe).isZero());
  EXPECT_TRUE(pg::geopotentialAcceleration(Eigen::Vector3d(1.0, 2.0, 3.0), 8, 8, kMu, kRe).isZero())
      << "a position inside the Earth must not be evaluated";
  EXPECT_TRUE(
      pg::geopotentialAcceleration(Eigen::Vector3d(nan, 0.0, 0.0), 8, 8, kMu, kRe).isZero());
  EXPECT_TRUE(pg::geopotentialAcceleration(good, 8, 8, -1.0, kRe).isZero()) << "negative mu";
  EXPECT_TRUE(pg::geopotentialAcceleration(good, 8, 8, kMu, 0.0).isZero()) << "zero radius";

  // Degree and order are clamped rather than trusted: a caller asking for more
  // than the table holds gets the whole table, and a negative one gets the
  // point mass. Neither reads outside the array.
  const Eigen::Vector3d full = pg::geopotentialAcceleration(good, pg::kGeopotentialMaxDegree,
                                                            pg::kGeopotentialMaxDegree, kMu, kRe);
  EXPECT_EQ(pg::geopotentialAcceleration(good, 999, 999, kMu, kRe), full);
  EXPECT_EQ(pg::geopotentialAcceleration(good, -5, -5, kMu, kRe),
            pg::geopotentialAcceleration(good, 0, 0, kMu, kRe));
  // An order above the degree is clamped to the degree, not read past it.
  EXPECT_EQ(pg::geopotentialAcceleration(good, 4, 99, kMu, kRe),
            pg::geopotentialAcceleration(good, 4, 4, kMu, kRe));
}

/// Order truncation must actually do something, and the zonal-only case must be
/// axisymmetric. Without this a bug that silently evaluated full order for every
/// request would pass every test above — they all compare at matched order.
TEST(Geopotential, OrderTruncationIsRealAndZonalFieldsAreAxisymmetric) {
  RecordProperty("verifies", "REQ-ODP-005");
  const double r_leo = 6778137.0;
  const Eigen::Vector3d r(r_leo * 0.6, r_leo * 0.5, r_leo * 0.6244997998398398);

  const Eigen::Vector3d zonal = pg::geopotentialAcceleration(r, 8, 0, kMu, kRe);
  const Eigen::Vector3d full = pg::geopotentialAcceleration(r, 8, 8, kMu, kRe);
  EXPECT_GT((full - zonal).norm(), 1.0e-9)
      << "order 8 and order 0 produced the same field — the order argument is being ignored";

  // A zonal (m = 0) field is invariant under rotation about the Z axis, so
  // rotating the position by an arbitrary longitude must rotate the
  // acceleration by exactly the same angle. This is the property that fails
  // first if a sectoral term leaks into the m = 0 sum.
  const double angle = 0.7;
  Eigen::Matrix3d rz;
  rz << std::cos(angle), -std::sin(angle), 0.0, std::sin(angle), std::cos(angle), 0.0, 0.0, 0.0,
      1.0;
  const Eigen::Vector3d rotated = pg::geopotentialAcceleration(rz * r, 8, 0, kMu, kRe);
  EXPECT_LT((rotated - rz * zonal).norm(), 1.0e-14 * zonal.norm());

  // The full-order field is *not* axisymmetric — the same rotation must break
  // it — or the test above would be vacuous.
  const Eigen::Vector3d rotated_full = pg::geopotentialAcceleration(rz * r, 8, 8, kMu, kRe);
  EXPECT_GT((rotated_full - rz * full).norm(), 1.0e-9);
}
