/// @file Unit tests for the TRIAD single-frame attitude initializer
/// (REQ-ADET-003; design doc §8.1).
///
/// Three independent layers:
///
///  - **Exactness.** With noise-free observations TRIAD is a closed-form
///    identity, not an approximation: over random attitudes and random vector
///    pairs it must return the generating attitude to round-off, in canonical
///    form (`q0 >= 0`).
///
///  - **Covariance.** The reported covariance is checked twice: analytically at
///    orthogonal geometry, where the Shuster result collapses to the familiar
///    `diag(σ₂², σ₁², σ₁²)` in the triad basis, and by Monte Carlo against the
///    QUEST measurement model, where the sample NEES must sit at 3. A wrong
///    sign, a transposed frame, or a missing `1/sin²θ` fails the second even if
///    the first were tuned to pass.
///
///  - **Degeneracy.** Near-parallel observations make the roll about the
///    primary unobservable. That is a normal flight condition (the sun and the
///    field lines do line up), so it must come back as an invalid solution —
///    not an assert, and above all not a confident wrong answer.

#include "gnc/triad.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/LU>

#include "math/frames.hpp"
#include "math/quaternion.hpp"
#include "math/typed_vector.hpp"
#include "random/rng.hpp"

namespace {

namespace gnc = polaris::gnc;
namespace pm = polaris::math;
namespace frames = polaris::math::frames;

constexpr double kDeg = M_PI / 180.0;

/// Uniformly-distributed unit vector from @p rng (Marsaglia's method).
Eigen::Vector3d randomUnit(polaris::random::SplitMix64& rng) {
  const double z = 2.0 * rng.uniform() - 1.0;
  const double phi = 2.0 * M_PI * rng.uniform();
  const double r = std::sqrt(std::max(0.0, 1.0 - z * z));
  return Eigen::Vector3d(r * std::cos(phi), r * std::sin(phi), z);
}

/// Random attitude quaternion (random axis, angle in [0, π]).
pm::Quaternion randomAttitude(polaris::random::SplitMix64& rng) {
  return pm::Quaternion::FromAxisAngle(randomUnit(rng), M_PI * rng.uniform()).canonical();
}

/// Any unit vector orthogonal to @p u.
Eigen::Vector3d anyPerpendicular(const Eigen::Vector3d& u) {
  const Eigen::Vector3d seed =
      (std::abs(u.x()) < 0.9) ? Eigen::Vector3d::UnitX() : Eigen::Vector3d::UnitY();
  return u.cross(seed).normalized();
}

/// Perturb unit @p u by a transverse Gaussian of per-axis 1σ @p sigma_rad —
/// exactly the QUEST measurement model the covariance assumes.
Eigen::Vector3d perturb(const Eigen::Vector3d& u, double sigma_rad,
                        polaris::random::SplitMix64& rng) {
  const Eigen::Vector3d t1 = anyPerpendicular(u);
  const Eigen::Vector3d t2 = u.cross(t1);
  return (u + sigma_rad * (rng.gaussian() * t1 + rng.gaussian() * t2)).normalized();
}

/// Body-frame attitude error δθ [rad] defined by A_est = (I − [δθ×])·A_true,
/// i.e. twice the vector part of q_est ⊗ q_true⁻¹ for a small error.
Eigen::Vector3d attitudeError(const pm::Quaternion& est, const pm::Quaternion& truth) {
  const pm::Quaternion dq = (est * truth.inverse()).canonical();
  return 2.0 * dq.vec();
}

/// Rotation angle between two attitudes [rad], evaluated through the error
/// quaternion rather than `angularDistance`. Both are correct, but `acos` of a
/// dot product near 1 loses half the mantissa (its floor is ~3e-8 rad), which
/// is coarser than the round-off these exactness checks are pinning.
double errorRad(const pm::Quaternion& est, const pm::Quaternion& truth) {
  const pm::Quaternion dq = (est * truth.inverse()).canonical();
  return 2.0 * std::atan2(dq.vec().norm(), dq.scalar());
}

/// A well-conditioned noise-free input at a chosen separation.
gnc::TriadInput makeInput(const pm::Quaternion& q_bi, const Eigen::Vector3d& ref1,
                          const Eigen::Vector3d& ref2, double sigma1, double sigma2) {
  gnc::TriadInput in{};
  in.primary.body = pm::Vec3<frames::Body>(q_bi.rotate(ref1));
  in.primary.reference = pm::Vec3<frames::ECI>(ref1);
  in.primary.sigma_rad = sigma1;
  in.secondary.body = pm::Vec3<frames::Body>(q_bi.rotate(ref2));
  in.secondary.reference = pm::Vec3<frames::ECI>(ref2);
  in.secondary.sigma_rad = sigma2;
  in.min_sin_angle = std::sin(5.0 * kDeg);
  return in;
}

TEST(Triad, RecoversRandomAttitudesExactly) {
  RecordProperty("verifies", "REQ-ADET-003");
  polaris::random::SplitMix64 rng(0xA771ADu);

  // 200 draws, of which the near-parallel ones are skipped as unfair exactness
  // cases; `solved` counts what actually ran so a bad gate cannot silently
  // reduce this to a no-op test.
  int solved = 0;
  for (int trial = 0; trial < 200; ++trial) {
    const pm::Quaternion q_true = randomAttitude(rng);
    Eigen::Vector3d ref1 = randomUnit(rng);
    Eigen::Vector3d ref2 = randomUnit(rng);
    if (std::abs(ref1.dot(ref2)) > std::cos(10.0 * kDeg)) {
      continue;
    }

    gnc::TriadSolution out{};
    ASSERT_TRUE(gnc::triad(makeInput(q_true, ref1, ref2, 0.01, 0.05), out)) << "trial " << trial;
    EXPECT_LT(errorRad(out.attitude.core(), q_true), 1e-13);
    EXPECT_GE(out.attitude.core().w(), 0.0) << "canonical q0 >= 0 (REQ-SYS-003)";
    EXPECT_TRUE(out.attitude.core().isUnit(1e-12));
    EXPECT_NEAR(out.separation_rad, std::acos(ref1.dot(ref2)), 1e-12);
    ++solved;
  }
  EXPECT_GT(solved, 150);
}

TEST(Triad, PrimaryObservationIsFittedExactly) {
  RecordProperty("verifies", "REQ-ADET-003");
  polaris::random::SplitMix64 rng(0x50121u);
  const pm::Quaternion q_true = randomAttitude(rng);
  const Eigen::Vector3d ref1(1.0, 0.0, 0.0);
  const Eigen::Vector3d ref2(0.0, 1.0, 0.0);

  // Corrupt only the secondary: TRIAD must still map the primary reference onto
  // the primary observation exactly — that asymmetry is the algorithm.
  gnc::TriadInput in = makeInput(q_true, ref1, ref2, 0.01, 0.05);
  in.secondary.body =
      pm::Vec3<frames::Body>(perturb(in.secondary.body.eigen().normalized(), 5.0 * kDeg, rng));

  gnc::TriadSolution out{};
  ASSERT_TRUE(gnc::triad(in, out));
  const Eigen::Vector3d mapped = out.attitude.core().rotate(ref1);
  EXPECT_LT((mapped - in.primary.body.eigen().normalized()).norm(), 1e-12);
}

TEST(Triad, CovarianceMatchesShusterAtOrthogonalGeometry) {
  RecordProperty("verifies", "REQ-ADET-003");
  const double s1 = 0.5 * kDeg;
  const double s2 = 2.0 * kDeg;
  const Eigen::Vector3d ref1(1.0, 0.0, 0.0);
  const Eigen::Vector3d ref2(0.0, 1.0, 0.0);

  gnc::TriadSolution out{};
  ASSERT_TRUE(gnc::triad(makeInput(pm::Quaternion::Identity(), ref1, ref2, s1, s2), out));

  // Orthogonal observations: variance σ₂² about the primary (the roll the
  // secondary resolves) and σ₁² about the other two axes, with no coupling.
  Eigen::Matrix3d expected = Eigen::Matrix3d::Identity() * (s1 * s1);
  expected(0, 0) = s2 * s2;
  EXPECT_LT((out.covariance - expected).cwiseAbs().maxCoeff(), 1e-15);
}

TEST(Triad, CovarianceIsDiagonalisedByTheTriadBasis) {
  RecordProperty("verifies", "REQ-ADET-003");
  const double s1 = 0.5 * kDeg;
  const double s2 = 2.0 * kDeg;
  const double sep = 60.0 * kDeg;
  const Eigen::Vector3d ref1(1.0, 0.0, 0.0);
  const Eigen::Vector3d ref2(std::cos(sep), std::sin(sep), 0.0);
  // A non-trivial attitude, so the triad basis is nowhere near the coordinate
  // axes and this is a statement about the covariance rather than about x/y/z.
  const pm::Quaternion q_true =
      pm::Quaternion::FromAxisAngle(Eigen::Vector3d(0.4, -0.7, 0.59).normalized(), 77.0 * kDeg);

  gnc::TriadSolution out{};
  ASSERT_TRUE(gnc::triad(makeInput(q_true, ref1, ref2, s1, s2), out));

  // In the orthonormal body triad {b̂₁, m̂, n̂} the Shuster covariance has a
  // closed form: the roll variance about the primary, σ₁² on the other two
  // axes, and a single off-diagonal coupling between b̂₁ and m̂. Pinning the
  // whole 3×3 here is what makes the coupling term non-optional — the
  // orthogonal-geometry case has zero coupling and cannot see it.
  const Eigen::Vector3d b1 = q_true.rotate(ref1).normalized();
  const Eigen::Vector3d b2 = q_true.rotate(ref2).normalized();
  const double c = b1.dot(b2);
  const double s = b1.cross(b2).norm();
  Eigen::Matrix3d basis;
  basis.col(0) = b1;
  basis.col(1) = (b2 - c * b1) / s;  // m̂
  basis.col(2) = b1.cross(b2) / s;   // n̂

  Eigen::Matrix3d expected_in_basis = Eigen::Matrix3d::Zero();
  expected_in_basis(0, 0) = (s1 * s1 * c * c + s2 * s2) / (s * s);
  expected_in_basis(1, 1) = s1 * s1;
  expected_in_basis(2, 2) = s1 * s1;
  expected_in_basis(0, 1) = s1 * s1 * c / s;
  expected_in_basis(1, 0) = expected_in_basis(0, 1);

  const Eigen::Matrix3d expected = basis * expected_in_basis * basis.transpose();
  EXPECT_LT((out.covariance - expected).cwiseAbs().maxCoeff(), 1e-15);

  // And the coupling is genuinely non-zero at this geometry, so the assertion
  // above has something to fail on.
  EXPECT_GT(std::abs(expected_in_basis(0, 1)), 1e-6);
}

TEST(Triad, CovarianceSplitsLinearlyAcrossIndependentErrorSources) {
  RecordProperty("verifies", "REQ-ADET-003");
  // The coarse estimator relies on this: it evaluates the white and systematic
  // parts of its budget separately on one geometry and adds them, so the two
  // must sum to the covariance of the root-sum-square sigmas.
  const double s1_white = 0.4 * kDeg;
  const double s1_sys = 0.9 * kDeg;
  const double s2_white = 1.2 * kDeg;
  const double s2_sys = 2.5 * kDeg;
  const Eigen::Vector3d ref1(1.0, 0.0, 0.0);
  const Eigen::Vector3d ref2(std::cos(50.0 * kDeg), std::sin(50.0 * kDeg), 0.3);
  const pm::Quaternion q_true =
      pm::Quaternion::FromAxisAngle(Eigen::Vector3d(1.0, 2.0, -1.0).normalized(), 63.0 * kDeg);

  const pm::Vec3<frames::Body> b1(q_true.rotate(ref1));
  const pm::Vec3<frames::Body> b2(q_true.rotate(ref2.normalized()));
  const double gate = std::sin(5.0 * kDeg);

  Eigen::Matrix3d white;
  Eigen::Matrix3d systematic;
  Eigen::Matrix3d total;
  ASSERT_TRUE(gnc::triadCovariance(b1, b2, s1_white, s2_white, gate, white));
  ASSERT_TRUE(gnc::triadCovariance(b1, b2, s1_sys, s2_sys, gate, systematic));
  ASSERT_TRUE(gnc::triadCovariance(b1, b2, std::hypot(s1_white, s1_sys),
                                   std::hypot(s2_white, s2_sys), gate, total));
  // Equal up to round-off, not bitwise: hypot-then-square and square-then-add
  // are the same algebra in a different order.
  EXPECT_TRUE((white + systematic).isApprox(total, 1e-12));

  // Zero sigma is a legal budget split (a source with no systematic part).
  Eigen::Matrix3d zero_cov;
  ASSERT_TRUE(gnc::triadCovariance(b1, b2, 0.0, 0.0, gate, zero_cov));
  EXPECT_TRUE(zero_cov == Eigen::Matrix3d::Zero());

  // ...but a negative one, a degenerate gate, or degenerate geometry is not.
  Eigen::Matrix3d untouched = Eigen::Matrix3d::Constant(42.0);
  EXPECT_FALSE(gnc::triadCovariance(b1, b2, -1.0, s2_sys, gate, untouched));
  EXPECT_FALSE(gnc::triadCovariance(b1, b2, s1_sys, s2_sys, 0.0, untouched));
  EXPECT_FALSE(gnc::triadCovariance(b1, b1, s1_sys, s2_sys, gate, untouched));
  EXPECT_TRUE(untouched == Eigen::Matrix3d::Constant(42.0)) << "output untouched on rejection";
}

TEST(Triad, CovarianceDivergesAsGeometryDegrades) {
  RecordProperty("verifies", "REQ-ADET-003");
  const double s1 = 0.5 * kDeg;
  const double s2 = 2.0 * kDeg;
  const Eigen::Vector3d ref1(1.0, 0.0, 0.0);

  double previous_roll_var = 0.0;
  for (double sep_deg : {90.0, 45.0, 20.0, 10.0}) {
    const double sep = sep_deg * kDeg;
    const Eigen::Vector3d ref2(std::cos(sep), std::sin(sep), 0.0);
    gnc::TriadSolution out{};
    ASSERT_TRUE(gnc::triad(makeInput(pm::Quaternion::Identity(), ref1, ref2, s1, s2), out));

    // The roll about the primary is the direction that loses observability.
    const double roll_var = out.covariance(0, 0);
    EXPECT_GT(roll_var, previous_roll_var) << "separation " << sep_deg << " deg";
    EXPECT_NEAR(
        roll_var,
        (s1 * s1 * std::cos(sep) * std::cos(sep) + s2 * s2) / (std::sin(sep) * std::sin(sep)),
        1e-15);
    previous_roll_var = roll_var;

    // Positive semi-definite at every geometry (Cholesky succeeds).
    Eigen::LLT<Eigen::Matrix3d> llt(out.covariance);
    EXPECT_EQ(llt.info(), Eigen::Success);
  }
}

TEST(Triad, CovarianceIsConsistentWithSampledNoise) {
  RecordProperty("verifies", "REQ-ADET-003");
  const double s1 = 0.4 * kDeg;
  const double s2 = 1.5 * kDeg;
  const double sep = 60.0 * kDeg;
  const Eigen::Vector3d ref1(1.0, 0.0, 0.0);
  const Eigen::Vector3d ref2(std::cos(sep), std::sin(sep), 0.0);
  const pm::Quaternion q_true =
      pm::Quaternion::FromAxisAngle(Eigen::Vector3d(0.3, -0.5, 0.81).normalized(), 41.0 * kDeg);

  gnc::TriadSolution nominal{};
  ASSERT_TRUE(gnc::triad(makeInput(q_true, ref1, ref2, s1, s2), nominal));

  polaris::random::SplitMix64 rng(0xC0FFEEu);
  constexpr int kSamples = 20000;
  Eigen::Matrix3d sample_cov = Eigen::Matrix3d::Zero();
  double nees_sum = 0.0;
  const Eigen::Matrix3d p_inv = nominal.covariance.inverse();

  for (int i = 0; i < kSamples; ++i) {
    gnc::TriadInput in = makeInput(q_true, ref1, ref2, s1, s2);
    in.primary.body = pm::Vec3<frames::Body>(perturb(q_true.rotate(ref1).normalized(), s1, rng));
    in.secondary.body = pm::Vec3<frames::Body>(perturb(q_true.rotate(ref2).normalized(), s2, rng));

    gnc::TriadSolution out{};
    ASSERT_TRUE(gnc::triad(in, out));
    const Eigen::Vector3d err = attitudeError(out.attitude.core(), q_true);
    sample_cov += err * err.transpose();
    nees_sum += err.dot(p_inv * err);
  }
  sample_cov /= static_cast<double>(kSamples);

  // NEES of a consistent 3-DOF estimate is 3; the 99% interval at this sample
  // count is well inside ±5%.
  const double nees = nees_sum / static_cast<double>(kSamples);
  EXPECT_NEAR(nees, 3.0, 0.15) << "sample NEES";

  // Whitened comparison rather than an elementwise absolute one: with
  // P = L·Lᵀ, a consistent estimator has L⁻¹·S·L⁻ᵀ = I, and every entry then
  // carries the same ~sqrt(2/N) ≈ 1 % sampling error regardless of how large or
  // small that direction's variance is. An absolute tolerance loose enough for
  // the roll variance would be far looser than the b̂₁–m̂ coupling term, so
  // deleting that term from the implementation would still pass; here it shifts
  // the whitened off-diagonal by ~0.13, well outside this bound.
  const Eigen::Matrix3d l = nominal.covariance.llt().matrixL();
  const Eigen::Matrix3d whitened = l.triangularView<Eigen::Lower>().solve(
      l.triangularView<Eigen::Lower>().solve(sample_cov).transpose());
  EXPECT_LT((whitened - Eigen::Matrix3d::Identity()).cwiseAbs().maxCoeff(), 0.05);
}

TEST(Triad, RejectsDegenerateGeometry) {
  RecordProperty("verifies", "REQ-ADET-003");
  const Eigen::Vector3d ref1(1.0, 0.0, 0.0);
  const double sep = 1.0 * kDeg;  // below the 5 deg gate set by makeInput
  const Eigen::Vector3d ref2(std::cos(sep), std::sin(sep), 0.0);

  gnc::TriadSolution out{};
  EXPECT_FALSE(gnc::triad(makeInput(pm::Quaternion::Identity(), ref1, ref2, 0.01, 0.05), out));
  EXPECT_FALSE(out.valid);

  // Exactly antiparallel is degenerate too, and must not divide by zero.
  gnc::TriadInput anti = makeInput(pm::Quaternion::Identity(), ref1, ref1, 0.01, 0.05);
  anti.secondary.body = pm::Vec3<frames::Body>(-ref1);
  anti.secondary.reference = pm::Vec3<frames::ECI>(-ref1);
  EXPECT_FALSE(gnc::triad(anti, out));
  EXPECT_FALSE(out.valid);
}

TEST(Triad, RejectsMalformedInput) {
  RecordProperty("verifies", "REQ-ADET-003");
  const Eigen::Vector3d ref1(1.0, 0.0, 0.0);
  const Eigen::Vector3d ref2(0.0, 1.0, 0.0);
  const gnc::TriadInput good = makeInput(pm::Quaternion::Identity(), ref1, ref2, 0.01, 0.05);
  gnc::TriadSolution out{};

  gnc::TriadInput zero_vector = good;
  zero_vector.secondary.body = pm::Vec3<frames::Body>::Zero();
  EXPECT_FALSE(gnc::triad(zero_vector, out));

  gnc::TriadInput not_finite = good;
  not_finite.primary.reference = pm::Vec3<frames::ECI>(std::nan(""), 0.0, 0.0);
  EXPECT_FALSE(gnc::triad(not_finite, out));

  gnc::TriadInput bad_sigma = good;
  bad_sigma.primary.sigma_rad = 0.0;
  EXPECT_FALSE(gnc::triad(bad_sigma, out));

  gnc::TriadInput bad_gate = good;
  bad_gate.min_sin_angle = -1.0;
  EXPECT_FALSE(gnc::triad(bad_gate, out));

  // A body pair whose geometry contradicts the reference pair is rejected too:
  // the references are 90 deg apart, the observations 1 deg.
  gnc::TriadInput inconsistent = good;
  inconsistent.secondary.body =
      pm::Vec3<frames::Body>(std::cos(1.0 * kDeg), std::sin(1.0 * kDeg), 0.0);
  EXPECT_FALSE(gnc::triad(inconsistent, out));
}

}  // namespace
