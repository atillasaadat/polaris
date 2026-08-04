/// @file
/// @brief Inter-star-tracker alignment calibration (design doc §8.2;
/// REQ-ADET-013). Pins the estimator itself — recovery of a known relative
/// rotation from simultaneous solution pairs — and every refusal, because a
/// calibration that applies a wrong correction is worse than one that refuses.
///
/// What is *not* tested here is the commanded lifecycle (window, abort, clear,
/// telemetry), which belongs to the component and is pinned in
/// `flight/PolarisFsw/AttitudeEstimator/test/ut/`.
///
/// References: Markley, Cheng, Crassidis & Oshman, "Averaging Quaternions", JGCD
/// 30(4), 2007 [markley2007]; design doc §8.2.

#include "gnc/st_alignment.hpp"

#include <gtest/gtest.h>

#include <cmath>

#include "random/rng.hpp"

namespace {

namespace gnc = polaris::gnc;
namespace pm = polaris::math;
namespace frames = polaris::math::frames;

using QuatBI = pm::Quat<frames::Body, frames::ECI>;
using QuatBB = pm::Quat<frames::Body, frames::Body>;

constexpr double kArcsec = M_PI / (180.0 * 3600.0);

gnc::StAlignmentConfig defaultConfig() {
  gnc::StAlignmentConfig cfg{};
  cfg.min_samples = 20;
  cfg.max_residual_rad = 5.0e-4;  // ~103 arcsec, the shipped gate
  cfg.min_eigen_gap = 0.9;
  return cfg;
}

/// A small-angle rotation as a quaternion, the form both trackers' errors take.
pm::Quaternion deltaQ(const Eigen::Vector3d& theta) {
  const double angle = theta.norm();
  if (!(angle > 0.0)) {
    return pm::Quaternion::Identity();
  }
  return pm::Quaternion::FromAxisAngle(theta / angle, angle);
}

/// A deterministic sequence of distinct attitudes — a tumble about two
/// incommensurate axes, so the pairs are not all at one orientation. Nothing in
/// the estimator needs that (an attitude pair has no degenerate geometry), which
/// is itself worth pinning: see `FitsFromASingleAttitude`.
pm::Quaternion attitudeAt(int k) {
  const double a = 0.31 * static_cast<double>(k);
  const double b = 0.17 * static_cast<double>(k);
  return pm::Quaternion::FromAxisAngle(Eigen::Vector3d::UnitZ(), a) *
         pm::Quaternion::FromAxisAngle(Eigen::Vector3d::UnitX(), b);
}

/// Angle between two rotations [rad], `2·atan2(‖v‖, |q₀|)` (§3.3).
double angleBetween(const pm::Quaternion& a, const pm::Quaternion& b) {
  const pm::Quaternion d = (a * b.inverse()).canonical();
  return 2.0 * std::atan2(d.vec().norm(), std::fabs(d.scalar()));
}

}  // namespace

TEST(StAlignment, RecoversAKnownMisalignmentExactly) {
  // Noise-free: the average of N identical rotations is that rotation, and the
  // residual is exactly zero. This is the algebra check — if the composition order
  // of `q_rel = q_2 ⊗ q_k⁻¹` were inverted, this test would recover the inverse
  // rotation and nothing downstream would notice, because an inverse misalignment
  // is still a plausible-looking small rotation.
  const Eigen::Vector3d truth(200.0 * kArcsec, -90.0 * kArcsec, 40.0 * kArcsec);
  const pm::Quaternion misalignment = deltaQ(truth);

  gnc::StAlignmentAccumulator accumulator(defaultConfig());
  ASSERT_TRUE(accumulator.isConfigured());
  for (int k = 0; k < 50; ++k) {
    const pm::Quaternion king = attitudeAt(k);
    const pm::Quaternion second = misalignment * king;
    ASSERT_TRUE(accumulator.addSample(QuatBI(king), QuatBI(second)));
  }
  ASSERT_EQ(accumulator.sampleCount(), 50u);

  gnc::StAlignmentResult fit;
  ASSERT_EQ(accumulator.fit(fit), gnc::StAlignmentRejection::kNone);
  EXPECT_TRUE(fit.valid);
  EXPECT_EQ(fit.samples, 50u);
  // Not zero, and that is a property of the estimator rather than a defect: the
  // residual is `2·sqrt(1 − λ_max/N)`, and the square root amplifies round-off, so
  // on noise-free data it floors at ~2e-8 rad (4 milliarcsec). Six orders below
  // any gate a real tracker pair justifies — but a caller must not read
  // "residual == 0" as the perfect-fit condition (st_alignment.hpp).
  EXPECT_LT(fit.residual_angle_rad, 1.0e-7);
  EXPECT_NEAR(fit.eigen_gap, 1.0, 1.0e-12);
  EXPECT_NEAR(fit.misalignment_angle_rad, truth.norm(), 1.0e-12);

  // The published correction is the *inverse*, so applying it to a second-unit
  // reading returns the king's frame. Checked on a fresh attitude rather than one
  // that was fitted, which is the property that matters: the correction has to
  // work where the calibration was not taken.
  const pm::Quaternion king = attitudeAt(500);
  const QuatBI corrected = gnc::applyStAlignment(fit, QuatBI(pm::Quaternion(misalignment * king)));
  EXPECT_LT(angleBetween(corrected.core(), king), 1.0e-12);
  // And q0 >= 0 on the way out (§3.3).
  EXPECT_GE(fit.correction.core().scalar(), 0.0);
  EXPECT_GE(corrected.core().scalar(), 0.0);
}

TEST(StAlignment, AveragesDownTheTrackersOwnNoise) {
  // The reason the sample floor is above the algebraic minimum: a single pair fits
  // exactly and reports zero residual whatever the truth, so what more pairs buy
  // is 1/sqrt(N) on the *error*, which is the quantity nobody can see from the
  // residual alone. Pinned as a comparison rather than an absolute bound, so it
  // stays a statement about the estimator rather than about a seed.
  const Eigen::Vector3d truth(150.0 * kArcsec, 60.0 * kArcsec, -220.0 * kArcsec);
  const pm::Quaternion misalignment = deltaQ(truth);
  // The AURIGA's about-boresight class, halved so the resulting residual
  // (√6·σ ≈ 49 arcsec) sits comfortably inside the shipped 103 arcsec dispersion
  // gate rather than on it — this case is about the estimator, not the gate, and a
  // fixture parked on a threshold is a flaky test waiting to happen.
  const double noise = 20.0 * kArcsec;

  auto errorAt = [&](int samples, std::uint64_t seed) {
    polaris::random::SplitMix64 rng(seed);
    gnc::StAlignmentConfig cfg = defaultConfig();
    cfg.min_samples = 2;
    gnc::StAlignmentAccumulator accumulator(cfg);
    for (int k = 0; k < samples; ++k) {
      const pm::Quaternion king_error =
          deltaQ(noise * Eigen::Vector3d(rng.gaussian(), rng.gaussian(), rng.gaussian()));
      const pm::Quaternion second_error =
          deltaQ(noise * Eigen::Vector3d(rng.gaussian(), rng.gaussian(), rng.gaussian()));
      const pm::Quaternion king = king_error * attitudeAt(k);
      const pm::Quaternion second = second_error * misalignment * attitudeAt(k);
      EXPECT_TRUE(accumulator.addSample(QuatBI(king), QuatBI(second)));
    }
    gnc::StAlignmentResult fit;
    EXPECT_EQ(accumulator.fit(fit), gnc::StAlignmentRejection::kNone);
    return angleBetween(fit.correction.core().inverse(), misalignment);
  };

  // Averaged over several seeds: one draw of a random variable proves nothing, and
  // a single-seed comparison of two sample sizes is exactly the flaky test this
  // otherwise invites.
  double few = 0.0;
  double many = 0.0;
  constexpr int kSeeds = 24;
  for (int s = 0; s < kSeeds; ++s) {
    few += errorAt(10, polaris::random::streamSeed(0x5A11u, s));
    many += errorAt(1000, polaris::random::streamSeed(0x5A11u, s));
  }
  few /= kSeeds;
  many /= kSeeds;
  // 100x the samples is 10x the precision; 4x is a loose but decisive assertion.
  EXPECT_LT(many * 4.0, few) << "the fit is not averaging down the per-sample noise: "
                             << "10 pairs -> " << few / kArcsec << " arcsec, 1000 -> "
                             << many / kArcsec;

  // The reported residual, by contrast, measures the *per-sample* dispersion and
  // therefore does **not** fall with N. That is the property the ops procedure
  // rests on — the ground grades a calibration on a number that means the same
  // thing whatever the window length — and it is the one an "average down the
  // residual" reading of the fit would get backwards.
  polaris::random::SplitMix64 rng(12345);
  gnc::StAlignmentAccumulator accumulator(defaultConfig());
  for (int k = 0; k < 2000; ++k) {
    const pm::Quaternion e1 =
        deltaQ(noise * Eigen::Vector3d(rng.gaussian(), rng.gaussian(), rng.gaussian()));
    const pm::Quaternion e2 =
        deltaQ(noise * Eigen::Vector3d(rng.gaussian(), rng.gaussian(), rng.gaussian()));
    EXPECT_TRUE(accumulator.addSample(QuatBI(pm::Quaternion(e1 * attitudeAt(k))),
                                      QuatBI(pm::Quaternion(e2 * misalignment * attitudeAt(k)))));
  }
  gnc::StAlignmentResult fit;
  ASSERT_EQ(accumulator.fit(fit), gnc::StAlignmentRejection::kNone);
  // Two independent draws of the same σ compose to σ√2 on the relative rotation,
  // and the residual is the RMS of that over three axes: √3·σ√2 ≈ 2.45σ.
  EXPECT_NEAR(fit.residual_angle_rad, std::sqrt(6.0) * noise, 0.15 * std::sqrt(6.0) * noise);
}

TEST(StAlignment, FitsFromASingleAttitude) {
  // **Attitude pairs have no degenerate geometry**, and that is worth pinning
  // rather than asserting in a comment: one pair determines all three parameters,
  // at any attitude, so a window taken with the vehicle parked is as valid as one
  // taken through a tumble. This is the structural difference from the
  // magnetometer calibration, which needs orientation diversity and gates on it —
  // and the reason there is no coverage gate here to get wrong.
  const pm::Quaternion misalignment = deltaQ(Eigen::Vector3d(1.0e-3, 0.0, 0.0));
  const pm::Quaternion parked = attitudeAt(7);

  gnc::StAlignmentAccumulator accumulator(defaultConfig());
  for (int k = 0; k < 20; ++k) {
    ASSERT_TRUE(
        accumulator.addSample(QuatBI(parked), QuatBI(pm::Quaternion(misalignment * parked))));
  }
  gnc::StAlignmentResult fit;
  ASSERT_EQ(accumulator.fit(fit), gnc::StAlignmentRejection::kNone);
  EXPECT_NEAR(fit.misalignment_angle_rad, 1.0e-3, 1.0e-12);
}

TEST(StAlignment, RefusesTooFewSamples) {
  gnc::StAlignmentAccumulator accumulator(defaultConfig());
  for (int k = 0; k < 19; ++k) {  // one short of min_samples
    ASSERT_TRUE(accumulator.addSample(QuatBI(attitudeAt(k)), QuatBI(attitudeAt(k))));
  }
  gnc::StAlignmentResult fit;
  EXPECT_EQ(accumulator.fit(fit), gnc::StAlignmentRejection::kSamples);
  EXPECT_FALSE(fit.valid);
  // A refused fit leaves nothing half-written the caller could apply.
  EXPECT_EQ(fit.samples, 0u);
  EXPECT_EQ(fit.residual_angle_rad, 0.0);
}

TEST(StAlignment, RefusesPairsThatDoNotShareOneRotation) {
  // The fault the eigen-gap gate exists for, and the one it is easy to mis-describe:
  // this is **not** a degenerate geometry, it is a *unit* delivering solutions that
  // do not sit at a fixed rotation from the king's — a mis-identified star field, a
  // stale or cross-wired solution, a mounting that is moving. Averaging a rotation
  // that does not exist would produce a plausible-looking correction from nothing.
  polaris::random::SplitMix64 rng(999);
  gnc::StAlignmentAccumulator accumulator(defaultConfig());
  for (int k = 0; k < 200; ++k) {
    const pm::Quaternion king = attitudeAt(k);
    // Each pair at its own arbitrary relative rotation.
    const Eigen::Vector3d axis =
        Eigen::Vector3d(rng.gaussian(), rng.gaussian(), rng.gaussian()).normalized();
    const pm::Quaternion second =
        pm::Quaternion::FromAxisAngle(axis, 2.0 * M_PI * rng.uniform()) * king;
    ASSERT_TRUE(accumulator.addSample(QuatBI(king), QuatBI(second)));
  }
  gnc::StAlignmentResult fit;
  EXPECT_EQ(accumulator.fit(fit), gnc::StAlignmentRejection::kDegenerate);
  EXPECT_FALSE(fit.valid);
}

TEST(StAlignment, RefusesExcessiveDispersion) {
  // Pairs that *do* share a rotation, but too loosely to be describing a fixed
  // misalignment. Distinguished from the case above by the eigen gap staying high
  // while the residual crosses the gate, which is why the two reasons are reported
  // separately: "collect longer / re-tune" versus "suspect a unit".
  polaris::random::SplitMix64 rng(4242);
  const pm::Quaternion misalignment = deltaQ(Eigen::Vector3d(1.0e-4, 0.0, 0.0));
  const double noise = 2.0e-3;  // 4x the residual gate
  gnc::StAlignmentAccumulator accumulator(defaultConfig());
  for (int k = 0; k < 300; ++k) {
    const pm::Quaternion e =
        deltaQ(noise * Eigen::Vector3d(rng.gaussian(), rng.gaussian(), rng.gaussian()));
    ASSERT_TRUE(accumulator.addSample(QuatBI(attitudeAt(k)),
                                      QuatBI(pm::Quaternion(e * misalignment * attitudeAt(k)))));
  }
  gnc::StAlignmentResult fit;
  EXPECT_EQ(accumulator.fit(fit), gnc::StAlignmentRejection::kDispersion);
  EXPECT_FALSE(fit.valid);
}

TEST(StAlignment, RefusesMalformedInputAndAnUnconfiguredAccumulator) {
  gnc::StAlignmentAccumulator inert{gnc::StAlignmentConfig{}};
  EXPECT_FALSE(inert.isConfigured());
  EXPECT_FALSE(inert.addSample(QuatBI(attitudeAt(0)), QuatBI(attitudeAt(0))));
  gnc::StAlignmentResult fit;
  EXPECT_EQ(inert.fit(fit), gnc::StAlignmentRejection::kConfig);

  gnc::StAlignmentAccumulator accumulator(defaultConfig());
  const double nan = std::numeric_limits<double>::quiet_NaN();
  // Wire values, so these are refusals rather than asserts.
  EXPECT_FALSE(
      accumulator.addSample(QuatBI(pm::Quaternion(nan, 0.0, 0.0, 0.0)), QuatBI(attitudeAt(0))));
  EXPECT_FALSE(
      accumulator.addSample(QuatBI(attitudeAt(0)), QuatBI(pm::Quaternion(0.0, 0.0, 0.0, 0.0))));
  EXPECT_EQ(accumulator.sampleCount(), 0u) << "a refused sample must not be counted";

  // Config range gates: fewer than two samples is not a window, and the two
  // quality gates must be positive with the eigen gap a ratio in (0, 1).
  gnc::StAlignmentConfig bad = defaultConfig();
  bad.min_samples = 1;
  EXPECT_FALSE(bad.isValid());
  bad = defaultConfig();
  bad.max_residual_rad = 0.0;
  EXPECT_FALSE(bad.isValid());
  bad = defaultConfig();
  bad.min_eigen_gap = 1.0;
  EXPECT_FALSE(bad.isValid());
}

TEST(StAlignment, ResetDropsTheWindowAndApplyPassesThroughUnfitted) {
  gnc::StAlignmentAccumulator accumulator(defaultConfig());
  for (int k = 0; k < 30; ++k) {
    ASSERT_TRUE(accumulator.addSample(QuatBI(attitudeAt(k)), QuatBI(attitudeAt(k))));
  }
  accumulator.reset();
  EXPECT_EQ(accumulator.sampleCount(), 0u);
  gnc::StAlignmentResult fit;
  EXPECT_EQ(accumulator.fit(fit), gnc::StAlignmentRejection::kSamples)
      << "reset must drop the accumulated moment, not just the count";

  // An unfitted (default) alignment passes the reading through **unchanged**,
  // which is what lets the component call applyStAlignment unconditionally at one
  // application point — including for the king, whose slot is never fitted.
  const gnc::StAlignmentResult none{};
  EXPECT_FALSE(none.valid);
  const pm::Quaternion q = attitudeAt(3);
  const QuatBI through = gnc::applyStAlignment(none, QuatBI(q));
  EXPECT_TRUE(through.core().coeffs() == q.coeffs());
}
