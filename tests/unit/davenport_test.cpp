/// @file Unit tests for Davenport's q-method (REQ-ADET-003; design doc §8.1).
///
/// The q-method is the MEKF's cold-start path, so what has to be pinned is not
/// "does it rotate things" but the two properties a filter seed lives or dies
/// on:
///
///  - **The attitude is the Wahba optimum.** Noise-free it must land on truth
///    exactly for any number of pairs, and with noise it must be at least as
///    good as TRIAD on the same two-vector data — the whole reason to run an
///    optimal solver instead of the deterministic one.
///  - **The covariance is honest.** The reported `P` is the inverse Fisher
///    information; a Monte Carlo of the actual errors whitened by `P` has to
///    come back as the identity. A seed whose covariance is wrong hands the
///    MEKF a lie it cannot detect.
///
/// Plus the refusal paths: collinear observation sets (the sun and the field
/// lines do line up — normal flight, not a fault), malformed inputs, and counts
/// outside the fixed capacity. None of these may assert.

#include "gnc/davenport.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <Eigen/Core>
#include <Eigen/Geometry>

#include "gnc/triad.hpp"
#include "math/frames.hpp"
#include "math/quaternion.hpp"
#include "math/typed_vector.hpp"
#include "random/rng.hpp"

namespace {

namespace gnc = polaris::gnc;
namespace pm = polaris::math;
namespace frames = polaris::math::frames;

constexpr double kDeg = M_PI / 180.0;
/// Observability gate used throughout: `λ_min/λ_max ≥ 0.05` of the information
/// matrix, roughly an 18° separation for an equally-weighted pair.
constexpr double kMinObservability = 0.05;

Eigen::Vector3d randomUnit(polaris::random::SplitMix64& rng) {
  const double z = 2.0 * rng.uniform() - 1.0;
  const double phi = 2.0 * M_PI * rng.uniform();
  const double r = std::sqrt(std::max(0.0, 1.0 - z * z));
  return Eigen::Vector3d(r * std::cos(phi), r * std::sin(phi), z);
}

pm::Quaternion randomAttitude(polaris::random::SplitMix64& rng) {
  return pm::Quaternion::FromAxisAngle(randomUnit(rng), M_PI * rng.uniform()).canonical();
}

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

/// Body-frame attitude error δθ [rad] with A_est = (I − [δθ×])·A_true, the
/// convention the reported covariance is expressed in (davenport.hpp).
Eigen::Vector3d attitudeError(const pm::Quaternion& est, const pm::Quaternion& truth) {
  const pm::Quaternion dq = (est * truth.inverse()).canonical();
  return 2.0 * dq.vec();
}

double errorRad(const pm::Quaternion& est, const pm::Quaternion& truth) {
  const pm::Quaternion dq = (est * truth.inverse()).canonical();
  return 2.0 * std::atan2(dq.vec().norm(), dq.scalar());
}

/// A noise-free observation set: each reference rotated into the body frame by
/// @p q_bi, with the matching per-pair sigma.
gnc::DavenportInput makeInput(const pm::Quaternion& q_bi, const Eigen::Vector3d* refs,
                              const double* sigmas, int count) {
  gnc::DavenportInput in{};
  in.count = count;
  in.min_observability = kMinObservability;
  for (int i = 0; i < count; ++i) {
    in.observations[i].body = pm::Vec3<frames::Body>(q_bi.rotate(refs[i]));
    in.observations[i].reference = pm::Vec3<frames::ECI>(refs[i]);
    in.observations[i].sigma_rad = sigmas[i];
  }
  return in;
}

TEST(Davenport, RecoversRandomAttitudesExactlyFromNPairs) {
  RecordProperty("verifies", "REQ-ADET-003");
  polaris::random::SplitMix64 rng(0xDA7E17u);

  // `solved` counts the draws that actually ran, so a gate that started
  // rejecting everything cannot quietly turn this into a no-op.
  int solved = 0;
  for (int trial = 0; trial < 200; ++trial) {
    const int count = 2 + (trial % 4);  // 2..5 pairs
    const pm::Quaternion q_true = randomAttitude(rng);

    Eigen::Vector3d refs[gnc::DavenportInput::kMaxObservations];
    double sigmas[gnc::DavenportInput::kMaxObservations];
    for (int i = 0; i < count; ++i) {
      refs[i] = randomUnit(rng);
      sigmas[i] = (0.2 + 2.0 * rng.uniform()) * kDeg;
    }

    gnc::DavenportInput in = makeInput(q_true, refs, sigmas, count);
    gnc::DavenportSolution out{};
    if (!gnc::davenport(in, out)) {
      continue;  // an ill-conditioned random draw is a legitimate refusal
    }
    ++solved;

    EXPECT_LT(errorRad(out.attitude.core(), q_true), 1e-12) << "trial " << trial;
    EXPECT_GE(out.attitude.core().w(), 0.0) << "canonical q0 >= 0";
    EXPECT_TRUE(out.attitude.core().isUnit(1e-12));
    // Noise-free and self-consistent: the Wahba loss at the optimum is zero.
    EXPECT_LT(out.loss, 1e-6) << "trial " << trial;
    EXPECT_GE(out.observability, kMinObservability);
    EXPECT_TRUE(out.covariance.isApprox(out.covariance.transpose(), 1e-15));
    EXPECT_EQ(out.covariance.llt().info(), Eigen::Success) << "covariance must be PD";
  }
  EXPECT_GT(solved, 150) << "most random draws must be well-conditioned enough to solve";
}

TEST(Davenport, AgreesWithTriadOnTwoVectorProblems) {
  RecordProperty("verifies", "REQ-ADET-003");
  // Same two pairs, same noise, two solvers. Noise-free they must agree to
  // round-off; with noise they may differ, but only by an amount the shared
  // covariance accounts for — TRIAD is a *suboptimal* solution of the same
  // problem, not a different one.
  const double s1 = 1.0 * kDeg;
  const double s2 = 2.0 * kDeg;
  const double sep = 55.0 * kDeg;
  const Eigen::Vector3d refs[2] = {Eigen::Vector3d(1.0, 0.0, 0.0),
                                   Eigen::Vector3d(std::cos(sep), std::sin(sep), 0.0)};
  const double sigmas[2] = {s1, s2};
  const pm::Quaternion q_true =
      pm::Quaternion::FromAxisAngle(Eigen::Vector3d(0.2, -0.6, 0.77).normalized(), 73.0 * kDeg);

  gnc::DavenportSolution clean{};
  ASSERT_TRUE(gnc::davenport(makeInput(q_true, refs, sigmas, 2), clean));

  gnc::TriadInput ti{};
  ti.primary.body = pm::Vec3<frames::Body>(q_true.rotate(refs[0]));
  ti.primary.reference = pm::Vec3<frames::ECI>(refs[0]);
  ti.primary.sigma_rad = s1;
  ti.secondary.body = pm::Vec3<frames::Body>(q_true.rotate(refs[1]));
  ti.secondary.reference = pm::Vec3<frames::ECI>(refs[1]);
  ti.secondary.sigma_rad = s2;
  ti.min_sin_angle = std::sin(5.0 * kDeg);
  gnc::TriadSolution ts{};
  ASSERT_TRUE(gnc::triad(ti, ts));
  EXPECT_LT(errorRad(clean.attitude.core(), ts.attitude.core()), 1e-12)
      << "noise-free, both solvers hit the same exact answer";

  // With noise: over many draws the q-method must be at least as accurate as
  // TRIAD in mean-square, because it is the maximum-likelihood solution of the
  // same measurement model. If this ever inverts, the weighting is wrong.
  polaris::random::SplitMix64 rng(0x5EED01u);
  constexpr int kSamples = 2000;
  double dav_mse = 0.0;
  double triad_mse = 0.0;
  double agreement_max = 0.0;
  for (int i = 0; i < kSamples; ++i) {
    const Eigen::Vector3d b1 = perturb(q_true.rotate(refs[0]).normalized(), s1, rng);
    const Eigen::Vector3d b2 = perturb(q_true.rotate(refs[1]).normalized(), s2, rng);

    gnc::DavenportInput di = makeInput(q_true, refs, sigmas, 2);
    di.observations[0].body = pm::Vec3<frames::Body>(b1);
    di.observations[1].body = pm::Vec3<frames::Body>(b2);
    gnc::DavenportSolution ds{};
    ASSERT_TRUE(gnc::davenport(di, ds));

    ti.primary.body = pm::Vec3<frames::Body>(b1);
    ti.secondary.body = pm::Vec3<frames::Body>(b2);
    ASSERT_TRUE(gnc::triad(ti, ts));

    dav_mse += attitudeError(ds.attitude.core(), q_true).squaredNorm();
    triad_mse += attitudeError(ts.attitude.core(), q_true).squaredNorm();
    agreement_max = std::max(agreement_max, errorRad(ds.attitude.core(), ts.attitude.core()));
  }
  dav_mse /= static_cast<double>(kSamples);
  triad_mse /= static_cast<double>(kSamples);
  EXPECT_LT(dav_mse, triad_mse) << "the optimal solver must beat the deterministic one";

  // "To within covariance": the largest disagreement over 2000 draws stays
  // inside a few sigma of the reported uncertainty.
  EXPECT_LT(agreement_max, 5.0 * std::sqrt(clean.covariance.trace()));
}

TEST(Davenport, ThirdObservationTightensTheSolution) {
  RecordProperty("verifies", "REQ-ADET-003");
  // Adding information may only reduce uncertainty — the inverse-Fisher form
  // guarantees it, and a weighting bug is the thing most likely to break it.
  const Eigen::Vector3d refs[3] = {Eigen::Vector3d(1.0, 0.0, 0.0), Eigen::Vector3d(0.0, 1.0, 0.0),
                                   Eigen::Vector3d(0.0, 0.0, 1.0)};
  const double sigmas[3] = {1.0 * kDeg, 1.0 * kDeg, 1.0 * kDeg};
  const pm::Quaternion q_true = pm::Quaternion::Identity();

  gnc::DavenportSolution two{};
  gnc::DavenportSolution three{};
  ASSERT_TRUE(gnc::davenport(makeInput(q_true, refs, sigmas, 2), two));
  ASSERT_TRUE(gnc::davenport(makeInput(q_true, refs, sigmas, 3), three));
  EXPECT_LT(three.covariance.trace(), two.covariance.trace());

  // Three orthogonal equally-weighted unit vectors: M = Σ σ⁻²(I − ûûᵀ) = 2σ⁻²I,
  // so P = (σ²/2)·I exactly. A closed form worth pinning, because it fixes both
  // the scale and the isotropy of the whole covariance path.
  const double sigma_sq = sigmas[0] * sigmas[0];
  EXPECT_TRUE(three.covariance.isApprox(0.5 * sigma_sq * Eigen::Matrix3d::Identity(), 1e-12));
}

TEST(Davenport, CovarianceMatchesMonteCarlo) {
  RecordProperty("verifies", "REQ-ADET-003");
  // The headline covariance check: perturb three pairs with the exact QUEST
  // measurement model the derivation assumes, and whiten the sample covariance
  // of the resulting attitude errors by the reported P. A consistent estimator
  // gives L⁻¹·S·L⁻ᵀ = I.
  const double sigmas[3] = {0.5 * kDeg, 1.5 * kDeg, 3.0 * kDeg};
  const Eigen::Vector3d refs[3] = {
      Eigen::Vector3d(1.0, 0.0, 0.0),
      Eigen::Vector3d(std::cos(50.0 * kDeg), std::sin(50.0 * kDeg), 0.0),
      Eigen::Vector3d(0.1, -0.4, 0.9).normalized()};
  const pm::Quaternion q_true =
      pm::Quaternion::FromAxisAngle(Eigen::Vector3d(0.3, -0.5, 0.81).normalized(), 41.0 * kDeg);

  gnc::DavenportSolution nominal{};
  ASSERT_TRUE(gnc::davenport(makeInput(q_true, refs, sigmas, 3), nominal));

  polaris::random::SplitMix64 rng(0xC0FFEEu);
  constexpr int kSamples = 20000;
  Eigen::Matrix3d sample_cov = Eigen::Matrix3d::Zero();
  double nees_sum = 0.0;
  const Eigen::Matrix3d p_inv = nominal.covariance.inverse();

  for (int i = 0; i < kSamples; ++i) {
    gnc::DavenportInput in = makeInput(q_true, refs, sigmas, 3);
    for (int j = 0; j < 3; ++j) {
      in.observations[j].body =
          pm::Vec3<frames::Body>(perturb(q_true.rotate(refs[j]).normalized(), sigmas[j], rng));
    }
    gnc::DavenportSolution out{};
    ASSERT_TRUE(gnc::davenport(in, out));
    const Eigen::Vector3d err = attitudeError(out.attitude.core(), q_true);
    sample_cov += err * err.transpose();
    nees_sum += err.dot(p_inv * err);
  }
  sample_cov /= static_cast<double>(kSamples);

  // NEES of a consistent 3-DOF estimate is 3.
  EXPECT_NEAR(nees_sum / static_cast<double>(kSamples), 3.0, 0.15) << "sample NEES";

  const Eigen::Matrix3d l = nominal.covariance.llt().matrixL();
  const Eigen::Matrix3d whitened = l.triangularView<Eigen::Lower>().solve(
      l.triangularView<Eigen::Lower>().solve(sample_cov).transpose());
  EXPECT_LT((whitened - Eigen::Matrix3d::Identity()).cwiseAbs().maxCoeff(), 0.05);
}

TEST(Davenport, ReportsLossWhenObservationsDisagree) {
  RecordProperty("verifies", "REQ-ADET-003");
  // The Wahba loss is the residual no rotation can explain away. Zero for a
  // consistent set; large when one observation is inconsistent with the rest —
  // a fault signature (a mis-modelled reference, a wired-wrong sensor) that
  // geometry gating alone would never catch, so it is reported rather than used
  // to reject.
  const Eigen::Vector3d refs[3] = {Eigen::Vector3d(1.0, 0.0, 0.0), Eigen::Vector3d(0.0, 1.0, 0.0),
                                   Eigen::Vector3d(0.0, 0.0, 1.0)};
  const double sigmas[3] = {1.0 * kDeg, 1.0 * kDeg, 1.0 * kDeg};
  const pm::Quaternion q_true = pm::Quaternion::Identity();

  gnc::DavenportSolution consistent{};
  ASSERT_TRUE(gnc::davenport(makeInput(q_true, refs, sigmas, 3), consistent));
  EXPECT_LT(consistent.loss, 1e-6);

  gnc::DavenportInput in = makeInput(q_true, refs, sigmas, 3);
  in.observations[2].body = pm::Vec3<frames::Body>(0.0, 0.0, -1.0);  // 180 deg wrong
  gnc::DavenportSolution broken{};
  ASSERT_TRUE(gnc::davenport(in, broken)) << "an inconsistent set is still solvable, just bad";
  EXPECT_GT(broken.loss, 1.0e3) << "loss ~ 2w for a fully reversed observation";
}

TEST(Davenport, RejectsCollinearObservations) {
  RecordProperty("verifies", "REQ-ADET-003");
  // Two nearly-parallel directions leave the rotation about them unobservable.
  // Normal flight geometry, so it is a refusal, never an assert.
  const double sep = 2.0 * kDeg;
  const Eigen::Vector3d refs[2] = {Eigen::Vector3d(1.0, 0.0, 0.0),
                                   Eigen::Vector3d(std::cos(sep), std::sin(sep), 0.0)};
  const double sigmas[2] = {1.0 * kDeg, 1.0 * kDeg};

  gnc::DavenportSolution out{};
  EXPECT_FALSE(gnc::davenport(makeInput(pm::Quaternion::Identity(), refs, sigmas, 2), out));
  EXPECT_FALSE(out.valid);

  // Exactly parallel, and duplicated observations of the same direction: both
  // are rank-1 information matrices.
  const Eigen::Vector3d same[2] = {Eigen::Vector3d(1.0, 0.0, 0.0), Eigen::Vector3d(2.0, 0.0, 0.0)};
  EXPECT_FALSE(gnc::davenport(makeInput(pm::Quaternion::Identity(), same, sigmas, 2), out));

  // Well-conditioned in the body frame but degenerate in the references: the
  // measurements disagree with the models by more than the geometry, which is
  // not a solution worth publishing (the same both-frames rule as TRIAD).
  gnc::DavenportInput mismatched = makeInput(pm::Quaternion::Identity(), refs, sigmas, 2);
  mismatched.observations[1].body = pm::Vec3<frames::Body>(0.0, 1.0, 0.0);
  EXPECT_FALSE(gnc::davenport(mismatched, out));
}

TEST(Davenport, RejectsMalformedInput) {
  RecordProperty("verifies", "REQ-ADET-003");
  const Eigen::Vector3d refs[3] = {Eigen::Vector3d(1.0, 0.0, 0.0), Eigen::Vector3d(0.0, 1.0, 0.0),
                                   Eigen::Vector3d(0.0, 0.0, 1.0)};
  const double sigmas[3] = {1.0 * kDeg, 1.0 * kDeg, 1.0 * kDeg};
  const pm::Quaternion q = pm::Quaternion::Identity();
  gnc::DavenportSolution out{};

  gnc::DavenportInput too_few = makeInput(q, refs, sigmas, 3);
  too_few.count = 1;
  EXPECT_FALSE(gnc::davenport(too_few, out)) << "one pair cannot fix three axes";

  gnc::DavenportInput too_many = makeInput(q, refs, sigmas, 3);
  too_many.count = gnc::DavenportInput::kMaxObservations + 1;
  EXPECT_FALSE(gnc::davenport(too_many, out)) << "past the fixed capacity";

  // No default observability gate: an unconfigured solve is refused rather than
  // run on an invented threshold (§19.3).
  gnc::DavenportInput ungated = makeInput(q, refs, sigmas, 3);
  ungated.min_observability = 0.0;
  EXPECT_FALSE(gnc::davenport(ungated, out));

  // The gate is a ratio of eigenvalues, so anything at or above 1 is
  // unreachable — refused up front rather than silently rejecting every solve
  // for the rest of the mission.
  gnc::DavenportInput unreachable = makeInput(q, refs, sigmas, 3);
  unreachable.min_observability = 1.0;
  EXPECT_FALSE(gnc::davenport(unreachable, out));

  gnc::DavenportInput bad_sigma = makeInput(q, refs, sigmas, 3);
  bad_sigma.observations[1].sigma_rad = 0.0;
  EXPECT_FALSE(gnc::davenport(bad_sigma, out)) << "a zero sigma is an infinite weight";

  gnc::DavenportInput zero_vector = makeInput(q, refs, sigmas, 3);
  zero_vector.observations[0].body = pm::Vec3<frames::Body>(0.0, 0.0, 0.0);
  EXPECT_FALSE(gnc::davenport(zero_vector, out));

  gnc::DavenportInput nan_vector = makeInput(q, refs, sigmas, 3);
  nan_vector.observations[2].reference = pm::Vec3<frames::ECI>(std::nan(""), 0.0, 0.0);
  EXPECT_FALSE(gnc::davenport(nan_vector, out));

  gnc::DavenportInput nan_sigma = makeInput(q, refs, sigmas, 3);
  nan_sigma.observations[0].sigma_rad = std::nan("");
  EXPECT_FALSE(gnc::davenport(nan_sigma, out));

  EXPECT_FALSE(out.valid) << "no rejection may leave a usable-looking solution";
}

}  // namespace
