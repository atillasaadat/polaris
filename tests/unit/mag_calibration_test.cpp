/// @file Attitude-independent magnetometer calibration (design doc §8.1).
///
/// The fit is the first of the three committed calibration items (§8.1,
/// "Calibration roadmap"): it attacks the 1.9° 1σ magnetic **systematic** on the
/// reference vehicle, which is a floor no filter can remove. These tests pin
/// three things in that order of importance: that the recovered correction is
/// right (exact recovery, then recovery at the reference noise budget), that
/// every way the fit can be unobservable is a **refusal** rather than a bad
/// calibration silently applied, and that the streaming accumulation the F´
/// component uses is order-invariant.
///
/// This file verifies no requirement on its own: calibration is a capability
/// that moves REQ-ADET-005/006, and the requirement evidence is the Monte Carlo
/// campaign in `attitude_accuracy_mc_test.cpp` re-run at the residual measured
/// here.
#include "gnc/mag_calibration.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <vector>

#include "math/frames.hpp"
#include "math/typed_vector.hpp"
#include "random/rng.hpp"

namespace {

namespace gnc = polaris::gnc;
namespace pm = polaris::math;
namespace frames = polaris::math::frames;

constexpr double kDeg = M_PI / 180.0;

// ── The reference vehicle's magnetic budget (design doc §8.1) ───────────────
// A ~30 µT LEO field, a 1 µT hard iron and 5 mrad of soft iron/misalignment —
// which together are the 1.9° systematic the coarse chain carries today — and
// the generic magnetometer's 0.05 µT rms noise.
constexpr double kFieldNominal = 30.0e-6;  ///< [T]
constexpr double kHardIron = 1.0e-6;       ///< [T] per-axis scale of `b`
constexpr double kSoftIron = 5.0e-3;       ///< [dimensionless] scale of `S − I`
constexpr double kSensorNoise = 0.05e-6;   ///< [T] per-axis rms
/// IGRF-14 magnitude error: a common-mode scale plus a per-sample part. The
/// scale term is deliberately included because it is **benign for attitude** —
/// an isotropic scale on the recovered field changes its magnitude, not its
/// direction — and a test that omitted it would not show that.
constexpr double kFieldScaleError = 5.0e-3;  ///< [dimensionless]
constexpr double kFieldNoise = 100.0e-9;     ///< [T] per-sample rms

constexpr int kSamples = 200;

gnc::MagCalibrationConfig referenceConfig() {
  gnc::MagCalibrationConfig cfg{};
  cfg.nominal_field_t = kFieldNominal;
  cfg.min_field_t = 5.0e-6;
  cfg.max_field_t = 100.0e-6;
  cfg.min_samples = 100;
  cfg.min_coverage = 0.35;
  cfg.max_condition = 1.0e6;
  cfg.min_residual_improvement = 2.0;
  return cfg;
}

/// One vehicle's iron: `m = S·B + b`.
struct Iron {
  Eigen::Matrix3d s{Eigen::Matrix3d::Identity()};
  Eigen::Vector3d b{Eigen::Vector3d::Zero()};
};

/// A symmetric soft-iron matrix at the budgeted scale plus a hard iron.
/// Symmetric because a permeability tensor is (file header) — which is also
/// what makes the symmetric square root the *physical* factor here, so this
/// draw is what lets the test assert `M ≈ S⁻¹` rather than only `MᵀM ≈ A`.
Iron drawIron(polaris::random::SplitMix64& rng) {
  Iron iron;
  Eigen::Matrix3d raw;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      raw(i, j) = rng.gaussian();
    }
  }
  iron.s = Eigen::Matrix3d::Identity() + kSoftIron * 0.5 * (raw + raw.transpose());
  iron.b = kHardIron * Eigen::Vector3d(rng.gaussian(), rng.gaussian(), rng.gaussian());
  return iron;
}

/// Field magnitude at sample @p k [T]: a deterministic orbit-driven sweep from
/// ~20 µT to ~40 µT, the LEO variation the fit's inhomogeneous right-hand side
/// depends on for absolute scale.
///
/// Note this magnitude is a function of the sample *index* alone, independent of
/// the sampled direction, which is the **optimistic** observability case: it
/// decouples the two things the fit needs to separate. On a real orbit |B| and
/// the field direction move together (both track magnetic latitude), so the
/// magnitude sweep is partly redundant with the direction sweep and the normal
/// matrix is worse conditioned than it is here. That coupling is exercised for
/// real in tests/integration/sitl_attitude_tuning_test.cpp, which flies the
/// window over an actual arc against the onboard IGRF — and which found exactly
/// this: a window that clears the coverage gate can still be refused on
/// CONDITION. These tests pin the algorithm; that one pins the flight case.
double fieldMagnitude(int k) {
  return kFieldNominal * (1.0 + 0.35 * std::sin(0.031 * static_cast<double>(k)));
}

/// A direction uniformly distributed inside a cone of half-angle @p half_angle
/// about @p axis. `π` gives the whole sphere.
Eigen::Vector3d coneDirection(const Eigen::Vector3d& axis, double half_angle,
                              polaris::random::SplitMix64& rng) {
  const double cos_theta = 1.0 - (1.0 - std::cos(half_angle)) * rng.uniform();
  const double sin_theta = std::sqrt(std::max(0.0, 1.0 - cos_theta * cos_theta));
  const double phi = 2.0 * M_PI * rng.uniform();
  const Eigen::Vector3d t1 =
      axis.cross((std::abs(axis.x()) < 0.9) ? Eigen::Vector3d::UnitX() : Eigen::Vector3d::UnitY())
          .normalized();
  const Eigen::Vector3d t2 = axis.cross(t1);
  return (cos_theta * axis + sin_theta * (std::cos(phi) * t1 + std::sin(phi) * t2)).normalized();
}

/// One collected sample: what the sensor read and what the onboard IGRF said
/// the magnitude was.
struct Sample {
  Eigen::Vector3d m{Eigen::Vector3d::Zero()};
  double field_t{0.0};
  Eigen::Vector3d truth_direction{Eigen::Vector3d::Zero()};  ///< for the accuracy metric
};

/// Collect @p count samples of a tumbling vehicle carrying @p iron.
std::vector<Sample> collect(const Iron& iron, int count, double sensor_noise, double field_noise,
                            double field_scale_error, double cone_half_angle,
                            polaris::random::SplitMix64& rng) {
  std::vector<Sample> out;
  out.reserve(count);
  for (int k = 0; k < count; ++k) {
    Sample s;
    s.truth_direction = coneDirection(Eigen::Vector3d::UnitZ(), cone_half_angle, rng);
    const double f_true = fieldMagnitude(k);
    const Eigen::Vector3d b_true = f_true * s.truth_direction;
    s.m = iron.s * b_true + iron.b +
          sensor_noise * Eigen::Vector3d(rng.gaussian(), rng.gaussian(), rng.gaussian());
    s.field_t = f_true * (1.0 + field_scale_error) + field_noise * rng.gaussian();
    out.push_back(s);
  }
  return out;
}

/// Feed @p samples to @p acc; returns how many were accepted.
int feed(gnc::MagCalibrationAccumulator& acc, const std::vector<Sample>& samples) {
  int accepted = 0;
  for (const Sample& s : samples) {
    if (acc.addSample(pm::Vec3<frames::Body>(s.m), s.field_t)) {
      ++accepted;
    }
  }
  return accepted;
}

/// The headline accuracy metric: the **systematic** direction error the
/// calibration leaves behind [deg, rms]. Evaluated on *noiseless* readings, so
/// what it measures is the calibration residual alone — the sensor's white noise
/// is budgeted separately (`sigma_mag_white`) and would otherwise be
/// double-counted.
double residualDirectionErrorDeg(const gnc::MagCalibrationResult& cal, const Iron& iron, int count,
                                 polaris::random::SplitMix64& rng) {
  double sum_sq = 0.0;
  for (int k = 0; k < count; ++k) {
    const Eigen::Vector3d direction = coneDirection(Eigen::Vector3d::UnitZ(), M_PI, rng);
    const Eigen::Vector3d m = iron.s * (fieldMagnitude(k) * direction) + iron.b;
    const Eigen::Vector3d corrected =
        gnc::applyMagCalibration(cal, pm::Vec3<frames::Body>(m)).eigen().normalized();
    const double angle =
        std::atan2(corrected.cross(direction).norm(), corrected.dot(direction)) / kDeg;
    sum_sq += angle * angle;
  }
  return std::sqrt(sum_sq / static_cast<double>(count));
}

// ── Configuration ───────────────────────────────────────────────────────────

TEST(MagCalibration, DefaultConfigIsRefusedAndLeavesTheAccumulatorInert) {
  EXPECT_FALSE(gnc::MagCalibrationConfig{}.isValid());
  gnc::MagCalibrationAccumulator acc{gnc::MagCalibrationConfig{}};
  EXPECT_FALSE(acc.isConfigured());
  EXPECT_FALSE(acc.addSample(pm::Vec3<frames::Body>(Eigen::Vector3d(3.0e-5, 0.0, 0.0)), 3.0e-5));
  EXPECT_EQ(acc.sampleCount(), 0);
  gnc::MagCalibrationResult cal{};
  EXPECT_FALSE(acc.solve(cal));
  EXPECT_FALSE(cal.valid);
}

TEST(MagCalibration, ConfigRejectsEachOutOfRangeField) {
  EXPECT_TRUE(referenceConfig().isValid());
  // Below the algebraic minimum the fit is exact by construction and every
  // quality gate downstream is meaningless.
  gnc::MagCalibrationConfig too_few = referenceConfig();
  too_few.min_samples = gnc::MagCalibrationAccumulator::kParameters - 1;
  EXPECT_FALSE(too_few.isValid());

  gnc::MagCalibrationConfig inverted_band = referenceConfig();
  inverted_band.max_field_t = inverted_band.min_field_t;
  EXPECT_FALSE(inverted_band.isValid());

  gnc::MagCalibrationConfig no_scale = referenceConfig();
  no_scale.nominal_field_t = 0.0;
  EXPECT_FALSE(no_scale.isValid());

  gnc::MagCalibrationConfig worse_allowed = referenceConfig();
  worse_allowed.min_residual_improvement = 0.5;  // would permit a harmful calibration
  EXPECT_FALSE(worse_allowed.isValid());

  gnc::MagCalibrationConfig loose_condition = referenceConfig();
  loose_condition.max_condition = 1.0;
  EXPECT_FALSE(loose_condition.isValid());
}

// ── Recovery ────────────────────────────────────────────────────────────────

TEST(MagCalibration, RecoversHardAndSoftIronExactlyFromNoiselessSamples) {
  polaris::random::SplitMix64 rng(polaris::random::streamSeed(0x3A9C11u, 1));
  const Iron iron = drawIron(rng);
  const std::vector<Sample> samples = collect(iron, kSamples, 0.0, 0.0, 0.0, M_PI, rng);

  gnc::MagCalibrationAccumulator acc(referenceConfig());
  ASSERT_EQ(feed(acc, samples), kSamples);

  gnc::MagCalibrationResult cal{};
  ASSERT_TRUE(acc.solve(cal));
  EXPECT_EQ(cal.sample_count, kSamples);

  // The hard iron is recovered to round-off on a 1 µT offset.
  EXPECT_LT((cal.hard_iron_offset.eigen() - iron.b).norm(), 1.0e-15);
  // The soft iron is recovered up to the unobservable left rotation; the truth
  // here is symmetric, so the symmetric factor the fit returns *is* S⁻¹.
  const Eigen::Matrix3d expected = iron.s.inverse();
  EXPECT_LT((cal.soft_iron_inverse - expected).norm(), 1.0e-11);
  // MᵀM = A is the identity the fit actually constrains, and holds regardless.
  EXPECT_LT(
      (cal.soft_iron_inverse.transpose() * cal.soft_iron_inverse - expected.transpose() * expected)
          .norm(),
      1.0e-11);
  EXPECT_LT(cal.residual_rms_t, 1.0e-13);
  EXPECT_GT(cal.uncalibrated_residual_rms_t, 0.5 * kHardIron);
}

TEST(MagCalibration, PostCalibrationSystematicMeetsTheHalfDegreeTarget) {
  // The number this whole push exists to produce: what is left of the 1.9°
  // magnetic systematic after an on-orbit fit at the reference budget. Run over
  // seeds, because a single draw of the iron and the noise says nothing about
  // stability.
  constexpr int kSeeds = 32;
  std::vector<double> residuals;
  residuals.reserve(kSeeds);
  double worst_uncalibrated = 0.0;

  for (int seed = 0; seed < kSeeds; ++seed) {
    polaris::random::SplitMix64 rng(polaris::random::streamSeed(0x3A9C11u, seed));
    const Iron iron = drawIron(rng);
    const std::vector<Sample> samples =
        collect(iron, kSamples, kSensorNoise, kFieldNoise, kFieldScaleError, M_PI, rng);

    gnc::MagCalibrationAccumulator acc(referenceConfig());
    ASSERT_EQ(feed(acc, samples), kSamples);
    gnc::MagCalibrationResult cal{};
    ASSERT_TRUE(acc.solve(cal)) << "seed " << seed;

    polaris::random::SplitMix64 eval_rng(polaris::random::streamSeed(0x5EED11u, seed));
    residuals.push_back(residualDirectionErrorDeg(cal, iron, 400, eval_rng));

    // The same metric with no calibration applied, for the before/after.
    gnc::MagCalibrationResult none{};
    polaris::random::SplitMix64 raw_rng(polaris::random::streamSeed(0x5EED11u, seed));
    worst_uncalibrated =
        std::max(worst_uncalibrated, residualDirectionErrorDeg(none, iron, 400, raw_rng));
  }

  std::sort(residuals.begin(), residuals.end());
  const double median = residuals[residuals.size() / 2];
  const double worst = residuals.back();
  std::printf(
      "[mag-cal] N=%d samples x %d seeds  post-cal systematic: median=%.4f deg  worst=%.4f deg"
      "  (uncalibrated worst=%.3f deg)\n",
      kSamples, kSeeds, median, worst, worst_uncalibrated);

  // The design-doc target for this item is the 0.2-0.5° class (§8.1).
  EXPECT_LT(worst, 0.5) << "post-calibration magnetic systematic misses the §8.1 target";
  // Sensitivity floor: the uncalibrated budget must actually be the ~1.9° the
  // requirement thresholds were set against, or the improvement above is
  // measured against nothing.
  EXPECT_GT(worst_uncalibrated, 1.0) << "uncalibrated systematic implausibly small";
}

TEST(MagCalibration, ResultIsInvariantToTheNominalFieldPreconditioner) {
  // nominal_field_t only scales the normal equations, so the calibration must
  // not depend on it. That is the claim the header makes; this pins it.
  polaris::random::SplitMix64 rng(polaris::random::streamSeed(0x3A9C11u, 7));
  const Iron iron = drawIron(rng);
  const std::vector<Sample> samples =
      collect(iron, kSamples, kSensorNoise, kFieldNoise, kFieldScaleError, M_PI, rng);

  gnc::MagCalibrationConfig alternate = referenceConfig();
  alternate.nominal_field_t = 3.0 * kFieldNominal;

  gnc::MagCalibrationAccumulator a(referenceConfig());
  gnc::MagCalibrationAccumulator b(alternate);
  ASSERT_EQ(feed(a, samples), kSamples);
  ASSERT_EQ(feed(b, samples), kSamples);

  gnc::MagCalibrationResult ca{};
  gnc::MagCalibrationResult cb{};
  ASSERT_TRUE(a.solve(ca));
  ASSERT_TRUE(b.solve(cb));
  EXPECT_LT((ca.hard_iron_offset.eigen() - cb.hard_iron_offset.eigen()).norm(),
            1.0e-12 * kHardIron);
  EXPECT_LT((ca.soft_iron_inverse - cb.soft_iron_inverse).norm(), 1.0e-9);
}

TEST(MagCalibration, AccumulationIsOrderInvariant) {
  // The F´ component streams samples as they arrive; nothing about the answer
  // may depend on the arrival order, including the interleaving of two halves
  // of a collection window.
  polaris::random::SplitMix64 rng(polaris::random::streamSeed(0x3A9C11u, 3));
  const Iron iron = drawIron(rng);
  const std::vector<Sample> samples =
      collect(iron, kSamples, kSensorNoise, kFieldNoise, kFieldScaleError, M_PI, rng);

  std::vector<Sample> interleaved;
  interleaved.reserve(samples.size());
  const std::size_t half = samples.size() / 2;
  for (std::size_t i = 0; i < half; ++i) {
    interleaved.push_back(samples[i]);
    interleaved.push_back(samples[half + i]);
  }

  gnc::MagCalibrationAccumulator in_order(referenceConfig());
  gnc::MagCalibrationAccumulator shuffled(referenceConfig());
  ASSERT_EQ(feed(in_order, samples), kSamples);
  ASSERT_EQ(feed(shuffled, interleaved), kSamples);

  gnc::MagCalibrationResult a{};
  gnc::MagCalibrationResult b{};
  ASSERT_TRUE(in_order.solve(a));
  ASSERT_TRUE(shuffled.solve(b));
  // Not bit-identical — summation order changes the last bits — but far below
  // anything the fit resolves.
  EXPECT_LT((a.hard_iron_offset.eigen() - b.hard_iron_offset.eigen()).norm(), 1.0e-10 * kHardIron);
  EXPECT_LT((a.soft_iron_inverse - b.soft_iron_inverse).norm(), 1.0e-10);
}

TEST(MagCalibration, ResetDiscardsTheWindow) {
  polaris::random::SplitMix64 rng(polaris::random::streamSeed(0x3A9C11u, 5));
  const Iron iron = drawIron(rng);
  const std::vector<Sample> samples = collect(iron, kSamples, 0.0, 0.0, 0.0, M_PI, rng);

  gnc::MagCalibrationAccumulator acc(referenceConfig());
  feed(acc, samples);
  ASSERT_EQ(acc.sampleCount(), kSamples);
  acc.reset();
  EXPECT_EQ(acc.sampleCount(), 0);
  gnc::MagCalibrationResult cal{};
  EXPECT_FALSE(acc.solve(cal));
}

// ── Refusals ────────────────────────────────────────────────────────────────

TEST(MagCalibration, RefusesFewerThanTheConfiguredMinimumSamples) {
  polaris::random::SplitMix64 rng(polaris::random::streamSeed(0x3A9C11u, 11));
  const Iron iron = drawIron(rng);
  const gnc::MagCalibrationConfig cfg = referenceConfig();
  const std::vector<Sample> samples =
      collect(iron, static_cast<int>(cfg.min_samples) - 1, 0.0, 0.0, 0.0, M_PI, rng);

  gnc::MagCalibrationAccumulator acc(cfg);
  feed(acc, samples);
  gnc::MagCalibrationResult cal{};
  EXPECT_FALSE(acc.solve(cal));
  EXPECT_FALSE(cal.valid);
}

TEST(MagCalibration, RefusesANarrowConeOfOrientations) {
  // The failure mode this gate exists for: a nearly inertially-fixed vehicle
  // whose field direction wanders over a few tens of degrees. The ellipsoid is
  // then extrapolated over directions never observed, which is worse than no
  // calibration at all.
  polaris::random::SplitMix64 rng(polaris::random::streamSeed(0x3A9C11u, 13));
  const Iron iron = drawIron(rng);
  const std::vector<Sample> narrow =
      collect(iron, kSamples, kSensorNoise, kFieldNoise, kFieldScaleError, 20.0 * kDeg, rng);

  gnc::MagCalibrationAccumulator acc(referenceConfig());
  ASSERT_EQ(feed(acc, narrow), kSamples);
  gnc::MagCalibrationResult cal{};
  EXPECT_FALSE(acc.solve(cal));

  // Same vehicle, same count, full-sphere coverage: accepted. The refusal above
  // is the geometry, not the data volume or the noise.
  polaris::random::SplitMix64 wide_rng(polaris::random::streamSeed(0x3A9C11u, 13));
  const Iron wide_iron = drawIron(wide_rng);
  const std::vector<Sample> wide =
      collect(wide_iron, kSamples, kSensorNoise, kFieldNoise, kFieldScaleError, M_PI, wide_rng);
  gnc::MagCalibrationAccumulator wide_acc(referenceConfig());
  ASSERT_EQ(feed(wide_acc, wide), kSamples);
  EXPECT_TRUE(wide_acc.solve(cal));
}

TEST(MagCalibration, CoverageIsMonotoneInTheSampledConeAndMatchesItsGeometricMeaning) {
  // `coverage = 3·λ_min(D)` equals `(3/2)·⟨sin²θ⟩` for directions uniform in a
  // cone of half-angle α — the closed form the header quotes, which is what
  // gives `min_coverage` a meaning an operator can reason about.
  const double angles_deg[] = {20.0, 30.0, 40.0, 45.0, 60.0, 90.0};
  double previous = -1.0;
  for (double alpha_deg : angles_deg) {
    const double alpha = alpha_deg * kDeg;
    polaris::random::SplitMix64 rng(polaris::random::streamSeed(0x9E11u, 1));
    const Iron clean;  // identity/zero: coverage is a property of the directions
    const std::vector<Sample> samples = collect(clean, 4000, 0.0, 0.0, 0.0, alpha, rng);

    // Read the metric off the accumulator directly: at 20° the fit itself is
    // (correctly) refused, so a solve-based reading could not reach the cone the
    // gate exists to exclude.
    gnc::MagCalibrationAccumulator acc(referenceConfig());
    feed(acc, samples);

    const double c = std::cos(alpha);
    const double mean_sin2 = (2.0 / 3.0 - c + c * c * c / 3.0) / (1.0 - c);
    EXPECT_NEAR(acc.coverage(), 1.5 * mean_sin2, 0.03) << alpha_deg << " deg cone";
    EXPECT_GT(acc.coverage(), previous) << "coverage is not monotone in the cone half-angle";
    if (alpha_deg <= 40.0) {
      EXPECT_LT(acc.coverage(), referenceConfig().min_coverage)
          << alpha_deg << " deg cone passes the reference gate";
    } else {
      EXPECT_GT(acc.coverage(), referenceConfig().min_coverage)
          << alpha_deg << " deg cone fails the reference gate";
    }
    previous = acc.coverage();
  }
  // The reference gate at 0.35 sits between the 40° and 45° cones, and the 20°
  // cone the test above refuses measures ~0.09.
  EXPECT_GT(previous, 0.9) << "a hemispherical sample set should read as full coverage";
}

TEST(MagCalibration, RejectsNonFiniteAndOutOfBandSamplesWithoutDisturbingTheWindow) {
  polaris::random::SplitMix64 rng(polaris::random::streamSeed(0x3A9C11u, 17));
  const Iron iron = drawIron(rng);
  const std::vector<Sample> samples = collect(iron, kSamples, 0.0, 0.0, 0.0, M_PI, rng);

  gnc::MagCalibrationAccumulator acc(referenceConfig());
  ASSERT_EQ(feed(acc, samples), kSamples);
  gnc::MagCalibrationResult before{};
  ASSERT_TRUE(acc.solve(before));

  const double nan = std::nan("");
  EXPECT_FALSE(
      acc.addSample(pm::Vec3<frames::Body>(Eigen::Vector3d(nan, 0.0, 0.0)), kFieldNominal));
  EXPECT_FALSE(
      acc.addSample(pm::Vec3<frames::Body>(Eigen::Vector3d(kFieldNominal, 0.0, 0.0)), nan));
  EXPECT_FALSE(  // saturated reading
      acc.addSample(pm::Vec3<frames::Body>(Eigen::Vector3d(1.0, 0.0, 0.0)), kFieldNominal));
  EXPECT_FALSE(  // unpowered sensor
      acc.addSample(pm::Vec3<frames::Body>(Eigen::Vector3d::Zero()), kFieldNominal));
  EXPECT_FALSE(  // nonsensical reference
      acc.addSample(pm::Vec3<frames::Body>(Eigen::Vector3d(kFieldNominal, 0.0, 0.0)), 0.0));

  EXPECT_EQ(acc.sampleCount(), kSamples);
  gnc::MagCalibrationResult after{};
  ASSERT_TRUE(acc.solve(after));
  EXPECT_EQ(after.hard_iron_offset.eigen(), before.hard_iron_offset.eigen());
  EXPECT_EQ(after.soft_iron_inverse, before.soft_iron_inverse);
}

TEST(MagCalibration, RefusesAMagnitudeStreamInconsistentWithTheMeasurements) {
  // Field magnitudes anti-correlated with the readings — a mis-tagged or
  // mis-positioned IGRF reference. The quadric that would fit this is not
  // positive definite, so there is no ellipsoid and no square root to take.
  polaris::random::SplitMix64 rng(polaris::random::streamSeed(0x3A9C11u, 19));
  const Iron iron = drawIron(rng);
  std::vector<Sample> samples = collect(iron, kSamples, 0.0, 0.0, 0.0, M_PI, rng);
  for (int k = 0; k < kSamples; ++k) {
    samples[static_cast<std::size_t>(k)].field_t = 2.0 * kFieldNominal - fieldMagnitude(k);
  }

  // Coverage and conditioning both depend only on the readings, which are the
  // good ones, so those gates pass; relaxing the residual gate to "must not be
  // worse" leaves the positive-definiteness check as the only thing that can
  // still refuse — and it does.
  gnc::MagCalibrationConfig permissive = referenceConfig();
  permissive.min_residual_improvement = 1.0;
  gnc::MagCalibrationAccumulator acc(permissive);
  ASSERT_EQ(feed(acc, samples), kSamples);
  ASSERT_GT(acc.coverage(), permissive.min_coverage);
  gnc::MagCalibrationResult cal{};
  EXPECT_FALSE(acc.solve(cal));
  EXPECT_FALSE(cal.valid);
}

TEST(MagCalibration, RefusesACalibrationThatDoesNotImproveOnTheRawSensor) {
  // An already-clean magnetometer: there is nothing to recover, so the fit can
  // only chase noise. Applying it would be a net loss of trust for no gain, and
  // `min_residual_improvement` is what says so.
  polaris::random::SplitMix64 rng(polaris::random::streamSeed(0x3A9C11u, 23));
  const Iron clean;
  const std::vector<Sample> samples =
      collect(clean, kSamples, kSensorNoise, kFieldNoise, 0.0, M_PI, rng);

  gnc::MagCalibrationAccumulator acc(referenceConfig());
  ASSERT_EQ(feed(acc, samples), kSamples);
  gnc::MagCalibrationResult cal{};
  EXPECT_FALSE(acc.solve(cal));

  // The same data with the gate relaxed to "must not be worse" is accepted —
  // proving it is the improvement gate that refused above and not a
  // conditioning or coverage problem with clean data.
  gnc::MagCalibrationConfig permissive = referenceConfig();
  permissive.min_residual_improvement = 1.0;
  gnc::MagCalibrationAccumulator lenient(permissive);
  ASSERT_EQ(feed(lenient, samples), kSamples);
  EXPECT_TRUE(lenient.solve(cal));
}

TEST(MagCalibration, ApplyPassesRawThroughWhenTheCalibrationIsNotValid) {
  const Eigen::Vector3d raw(1.0e-5, -2.0e-5, 3.0e-5);
  const gnc::MagCalibrationResult none{};
  EXPECT_EQ(gnc::applyMagCalibration(none, pm::Vec3<frames::Body>(raw)).eigen(), raw);
}

}  // namespace
