/// @file Unit tests for the onboard Earth-albedo correction of the sun vector
/// (`lib/gnc/albedo_correction`; design doc §8.1 calibration item (2)).
///
/// The tests that carry weight here are the ones that run the correction against
/// the **truth model it inverts** (`sim/sensors/sun_sensor`, §6.2) rather than
/// against a hand-written replica of its formula: a correction validated only
/// against its own algebra would pass while removing an error the sensor never
/// had. So the exact-recovery case builds a real `SunSensor`, samples it, and
/// asks whether the corrected vector is the truth direction back.
///
/// The rest are the refusal paths. Every one of them is a *normal* condition —
/// eclipse, night side, no Earth in the field, no position fix — and each must
/// leave the measurement untouched and report the uncorrected state rather than
/// apply a correction on geometry it does not have.

#include "gnc/albedo_correction.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <Eigen/Core>
#include <Eigen/Geometry>

#include "constants/constants.hpp"
#include "math/frames.hpp"
#include "math/typed_vector.hpp"
#include "sensors/sun_sensor.hpp"
#include "time/timescales.hpp"

namespace {

namespace gnc = polaris::gnc;
namespace sensors = polaris::sim::sensors;
namespace pm = polaris::math;
namespace pmf = polaris::math::frames;
namespace pt = polaris::time;

using Vec3B = pm::Vec3<pmf::Body>;

constexpr double kDeg = M_PI / 180.0;
constexpr double kRe = polaris::constants::wgs84::kSemiMajorAxis;
constexpr double kAu = polaris::constants::bodies::kAstronomicalUnit;
constexpr double kOrbitRadius = kRe + 500.0e3;  ///< the reference 500 km SSO

const pt::Tai kEpoch = pt::Tai::fromNanosecondsSinceEpoch(1767225637000000000LL);

pt::Tai epochPlus(double seconds) {
  return pt::Tai::fromNanosecondsSinceEpoch(kEpoch.nanosecondsSinceEpoch() +
                                            static_cast<std::int64_t>(seconds * 1.0e9));
}

/// The GomSpace NanoSense FSS as the reference vehicle carries it, with the
/// truth-side dispersion under the test's control (§19.4 permits a hardcoded
/// spec in tests; this one mirrors gomspace_nanosense_fss.yaml).
sensors::SunSensorSpec fssSpec(double dispersion_fraction) {
  sensors::SunSensorSpec spec = sensors::SunSensorSpec::fromParams({
      {"half_fov_deg", 60.0},
      {"accuracy_inner_half_angle_deg", 45.0},
      {"accuracy_inner_deg_3sigma", 0.5},
      {"accuracy_outer_deg_3sigma", 2.0},
      {"albedo_error_deg", 12.0},
      {"sample_period_ms", 10.0},
  });
  spec.albedo_dispersion_fraction = dispersion_fraction;
  // The white noise is a separate mechanism with its own budget line; isolating
  // the albedo is what these tests are for.
  spec.accuracy_inner_sigma = 0.0;
  spec.accuracy_outer_sigma = 0.0;
  return spec;
}

/// The onboard config matching @ref fssSpec, boresight along body +Z.
gnc::AlbedoCorrectionConfig fssConfig() {
  gnc::AlbedoCorrectionConfig cfg{};
  cfg.albedo_error_rad = 12.0 * kDeg;
  cfg.half_fov_rad = 60.0 * kDeg;
  cfg.boresight_body = Vec3B(Eigen::Vector3d::UnitZ());
  return cfg;
}

/// Sun direction at a given incidence from the boresight (+Z), in the x-z plane.
Eigen::Vector3d sunAtIncidence(double deg) {
  return Eigen::Vector3d(std::sin(deg * kDeg), 0.0, std::cos(deg * kDeg));
}

/// A day-side LEO geometry with the Earth at @p nadir_body and the Sun at the
/// given incidence, fully sunlit below.
sensors::SunSensorInput truthInput(double sun_incidence_deg, const Eigen::Vector3d& nadir_body) {
  sensors::SunSensorInput in;
  in.sun_dir_body = Vec3B(sunAtIncidence(sun_incidence_deg));
  in.nadir_dir_body = Vec3B(nadir_body.normalized());
  in.shadow_factor = 1.0;
  in.sky.sat = Eigen::Vector3d(kOrbitRadius, 0.0, 0.0);
  in.sky.sun = Eigen::Vector3d(kAu, 0.0, 0.0);  // sub-satellite point fully sunlit
  return in;
}

/// The correction input matching @ref truthInput: what the flight software forms
/// from its position fix, its ephemeris query and its attitude estimate.
gnc::AlbedoCorrectionInput onboardInput(const sensors::SunSensorMeasurement& m,
                                        const sensors::SunSensorInput& truth) {
  gnc::AlbedoCorrectionInput in{};
  in.sun_meas = m.sun_dir_body;
  in.nadir_body = truth.nadir_dir_body;
  in.radius_m = truth.sky.sat.norm();
  in.dayside = std::max(0.0, truth.sky.sat.normalized().dot(truth.sky.sun.normalized()));
  return in;
}

double angleBetween(const Eigen::Vector3d& a, const Eigen::Vector3d& b) {
  return std::atan2(a.cross(b).norm(), a.dot(b));
}

}  // namespace

// --- The claim the post-correction budget rests on ---------------------------

TEST(AlbedoCorrection, RemovesTheTruthModelPullExactlyWithoutDispersion) {
  // With the Earth perfectly modelled — dispersion zero — the correction must
  // return the *truth* direction, not merely a better one. Anything short of
  // round-off would mean the flight formula and the truth model disagree about
  // the geometry, and the residual budget of §19.2 would be measuring that
  // disagreement instead of the physics.
  sensors::SunSensor ss(fssSpec(0.0), Eigen::Matrix3d::Identity(), 0xA1BED0, 1);
  const gnc::AlbedoCorrectionConfig cfg = fssConfig();

  // Sweep the geometry rather than assert one case: the pull depends on where
  // the Earth sits in the field and on the Sun-Earth separation, and an inverse
  // that is exact at one point and first-order elsewhere is the likely bug.
  int applied_count = 0;
  double worst_residual = 0.0;
  double largest_pull = 0.0;
  double t = 0.0;
  for (double earth_off_boresight = 0.0; earth_off_boresight <= 90.0; earth_off_boresight += 10.0) {
    for (double sun_incidence = 0.0; sun_incidence <= 55.0; sun_incidence += 5.0) {
      const Eigen::Vector3d nadir(0.0, std::sin(earth_off_boresight * kDeg),
                                  std::cos(earth_off_boresight * kDeg));
      const auto truth_in = truthInput(sun_incidence, nadir);
      t += 1.0;
      const auto m = ss.sample(epochPlus(t), truth_in);
      if (!m.valid) {
        continue;
      }

      pm::Vec3<pmf::Body> corrected;
      double applied = 0.0;
      const bool ok = gnc::albedoCorrection(cfg, onboardInput(m, truth_in), corrected, applied);
      if (!ok) {
        // Refusal is only allowed where there is nothing to remove.
        EXPECT_NEAR(m.albedo_angle_rad, 0.0, 1e-15)
            << "refused a geometry the sensor did apply a pull in";
        continue;
      }
      ++applied_count;
      largest_pull = std::max(largest_pull, applied);
      EXPECT_NEAR(applied, m.albedo_angle_rad, 1e-12) << "the pull removed is not the pull applied";
      worst_residual =
          std::max(worst_residual, angleBetween(corrected.eigen(), truth_in.sun_dir_body.eigen()));
    }
  }

  EXPECT_GT(applied_count, 20) << "the sweep must actually exercise the correction";
  EXPECT_GT(largest_pull, 3.0 * kDeg) << "the sweep must reach a geometry where albedo is large";
  EXPECT_LT(worst_residual, 1.0e-12) << "recovery is not exact";
}

TEST(AlbedoCorrection, LeavesOnlyTheDispersionWhenTheEarthIsNotPerfectlyModelled) {
  // The honest version of the test above, and the number the vehicle budget is
  // derived from: with the truth model dispersed by a fraction f, what survives
  // the correction is that fraction of the pull — and nothing more. This is what
  // makes `albedo_dispersion_fraction`, not `albedo_error_deg`, the driver of
  // the post-correction `SigmaSunSysRad`.
  constexpr double kDispersion = 0.30;
  const gnc::AlbedoCorrectionConfig cfg = fssConfig();
  const auto truth_in = truthInput(20.0, Eigen::Vector3d(0.0, 0.0, 1.0));

  double sum_sq_residual = 0.0;
  double sum_sq_uncorrected = 0.0;
  int units = 0;
  for (std::uint64_t unit = 1; unit <= 64; ++unit) {
    sensors::SunSensor ss(fssSpec(kDispersion), Eigen::Matrix3d::Identity(), 0xA1BED0, unit);
    const auto m = ss.sample(kEpoch, truth_in);
    ASSERT_TRUE(m.valid);

    pm::Vec3<pmf::Body> corrected;
    double applied = 0.0;
    ASSERT_TRUE(gnc::albedoCorrection(cfg, onboardInput(m, truth_in), corrected, applied));

    const double truth_pull = angleBetween(m.sun_dir_body.eigen(), truth_in.sun_dir_body.eigen());
    const double residual = angleBetween(corrected.eigen(), truth_in.sun_dir_body.eigen());
    sum_sq_uncorrected += truth_pull * truth_pull;
    sum_sq_residual += residual * residual;
    ++units;
    // Nothing this test does may depend on the correction knowing the realised
    // dispersion: it removes the *modelled* pull, which is what flight software
    // can compute.
    EXPECT_NEAR(applied,
                gnc::AlbedoCorrectionConfig{cfg}.albedo_error_rad *
                    std::sin(angleBetween(m.sun_dir_body.eigen(), truth_in.nadir_dir_body.eigen())),
                1e-12);
  }

  ASSERT_EQ(units, 64);
  const double rms_uncorrected = std::sqrt(sum_sq_uncorrected / units);
  const double rms_residual = std::sqrt(sum_sq_residual / units);
  std::printf(
      "[albedo residual] f=%.2f  uncorrected rms %.3f deg -> corrected rms %.3f deg (%.2fx)\n",
      kDispersion, rms_uncorrected / kDeg, rms_residual / kDeg, rms_uncorrected / rms_residual);

  // The residual is the dispersion acting on two independent axes — a scale
  // error along the pull and an out-of-plane centroid offset — but the catalog
  // fraction is the **total**, split 1/√2 to each axis, so the quadrature sum
  // comes back to f itself: ≈ 0.30 here, with no hidden √2 (sun_sensor.hpp says
  // the same from the truth side). That ratio, measured rather than assumed, is
  // what the vehicle's post-correction SigmaSunSysRad is derived from (§19.2).
  const double ratio = rms_residual / rms_uncorrected;
  EXPECT_GT(ratio, 0.5 * kDispersion) << "no residual at all — is the truth dispersion wired in?";
  EXPECT_LT(ratio, 2.0 * kDispersion) << "residual far above the configured dispersion";

  // A 1σ fraction of 0.30 over 64 draws: the gate is generous on the upper side
  // and tight on the lower — what is asserted is that the residual is *of the
  // dispersion's order*, not zero (which would mean the dispersion was not
  // applied) and not of the pull's order (which would mean the correction did
  // nothing). And the point of the whole push: correcting beats not correcting,
  // by roughly 1/f, on the same draws.
  EXPECT_LT(rms_residual, 0.7 * rms_uncorrected) << "the correction did not reduce the rms error";
}

TEST(AlbedoCorrection, AttitudeErrorGainIsBoundedByThePeakScale) {
  // **The correction's sensitivity to the caller's own attitude error**, which
  // every other test in this file hides by handing it the truth nadir. The
  // caller has an *estimate*, so this is the term that decides whether the
  // correction is safe to run on a freshly acquired solution.
  //
  // The tempting analysis — an attitude error ε perturbs the pull magnitude φ by
  // a relative ε, therefore negligible — is wrong by about 6× on this part. The
  // dominant term is the rotation **axis**: â = (ŝ × d̂)/|ŝ × d̂| swings by
  // ε/sin ψ, so the error in the correction vector is φ·(ε/sin ψ) = A·Φ·η·ε,
  // set by the **peak** A rather than by the applied φ — the sin ψ cancels, which
  // is exactly why the naive reading misses it. Uncaught, this was a documented
  // claim of "a few degrees of attitude error → ~0.1°" against a true ~0.1° per
  // *degree*.
  const gnc::AlbedoCorrectionConfig cfg = fssConfig();
  sensors::SunSensor ss(fssSpec(0.0), Eigen::Matrix3d::Identity(), 0xA1BED0, 1);

  double worst_gain = 0.0;
  double sum_sq_gain = 0.0;
  int samples = 0;
  double t = 0.0;
  for (double earth_off = 0.0; earth_off <= 60.0; earth_off += 15.0) {
    for (double sun_incidence = 10.0; sun_incidence <= 50.0; sun_incidence += 10.0) {
      const Eigen::Vector3d nadir(0.0, std::sin(earth_off * kDeg), std::cos(earth_off * kDeg));
      const auto truth_in = truthInput(sun_incidence, nadir);
      t += 1.0;
      const auto m = ss.sample(epochPlus(t), truth_in);
      if (!m.valid) {
        continue;
      }

      // The correction with perfect knowledge, as the baseline.
      pm::Vec3<pmf::Body> exact;
      double applied_exact = 0.0;
      if (!gnc::albedoCorrection(cfg, onboardInput(m, truth_in), exact, applied_exact)) {
        continue;
      }

      // Now perturb the *nadir the caller supplies* — which is what an attitude
      // error does — about several axes, and measure how far the corrected
      // vector moves per radian of it.
      for (const double error_deg : {1.0, 3.0, 10.0}) {
        for (int axis = 0; axis < 3; ++axis) {
          Eigen::Vector3d rotation_axis = Eigen::Vector3d::Zero();
          rotation_axis[axis] = 1.0;
          const Eigen::Vector3d perturbed =
              Eigen::AngleAxisd(error_deg * kDeg, rotation_axis) * truth_in.nadir_dir_body.eigen();

          gnc::AlbedoCorrectionInput in = onboardInput(m, truth_in);
          in.nadir_body = Vec3B(perturbed);
          pm::Vec3<pmf::Body> shifted;
          double applied = 0.0;
          if (!gnc::albedoCorrection(cfg, in, shifted, applied)) {
            continue;
          }
          const double gain = angleBetween(shifted.eigen(), exact.eigen()) / (error_deg * kDeg);
          worst_gain = std::max(worst_gain, gain);
          sum_sq_gain += gain * gain;
          ++samples;
        }
      }
    }
  }

  ASSERT_GT(samples, 50) << "the sweep must actually exercise the sensitivity";
  const double rms_gain = std::sqrt(sum_sq_gain / samples);
  std::printf(
      "[albedo attitude sensitivity] rms %.3f deg per deg, worst %.3f deg per deg "
      "(peak scale A = %.1f deg)\n",
      rms_gain, worst_gain, cfg.albedo_error_rad / kDeg);

  // The bound is the mechanism, not a fitted number: the gain cannot exceed the
  // peak scale A (with margin for the geometry factors Φ·η ≤ 1 and for the
  // second-order terms at 10 deg of error). A gain that broke this would mean
  // the axis sensitivity is worse than A·ε, which would put the whole
  // "correct with the previous cycle's attitude" design in question.
  EXPECT_LE(worst_gain, 1.3 * cfg.albedo_error_rad)
      << "attitude-error gain exceeds the peak albedo scale";
  // And it is genuinely of that order rather than negligible — the claim this
  // test exists to have caught. A gain near zero would mean the perturbation
  // never reached the correction.
  EXPECT_GT(rms_gain, 0.1 * cfg.albedo_error_rad) << "sensitivity implausibly small — is the "
                                                     "perturbed nadir actually being used?";
}

// --- Graceful degradation: every refusal is a normal condition ---------------

TEST(AlbedoCorrection, RefusesWithoutUsableGeometryAndLeavesTheMeasurementAlone) {
  const gnc::AlbedoCorrectionConfig cfg = fssConfig();
  const Eigen::Vector3d sun = sunAtIncidence(20.0);

  gnc::AlbedoCorrectionInput good{};
  good.sun_meas = Vec3B(sun);
  good.nadir_body = Vec3B(Eigen::Vector3d::UnitZ());
  good.radius_m = kOrbitRadius;
  good.dayside = 1.0;

  const Vec3B kSentinel(Eigen::Vector3d(9.0, 9.0, 9.0));
  auto refuses = [&](const gnc::AlbedoCorrectionInput& in, const char* why) {
    pm::Vec3<pmf::Body> out = kSentinel;
    double applied = -1.0;
    EXPECT_FALSE(gnc::albedoCorrection(cfg, in, out, applied)) << why;
    EXPECT_DOUBLE_EQ(applied, 0.0) << why;
    EXPECT_EQ(out.eigen(), kSentinel.eigen()) << why << ": output must be left untouched";
  };

  // The correction works on the control case, so each refusal below is about the
  // one thing it changes.
  {
    pm::Vec3<pmf::Body> out;
    double applied = 0.0;
    ASSERT_TRUE(gnc::albedoCorrection(cfg, good, out, applied));
    ASSERT_GT(applied, 0.0);
  }

  // No position fix. The component leaves `radius_m` at zero rather than
  // substituting a nominal orbit — correcting on a guessed altitude would inject
  // a bias the size of the one being removed (§8.1).
  auto no_position = good;
  no_position.radius_m = 0.0;
  refuses(no_position, "no position fix");

  auto subsurface = good;
  subsurface.radius_m = kRe * 0.5;
  refuses(subsurface, "position below the surface");

  // Night side and eclipse: no sunlit ground to reflect. Zero is the right
  // answer, and refusing says so without pretending a correction ran.
  auto night = good;
  night.dayside = 0.0;
  refuses(night, "night side");

  // No Earth in the field: sensor pointed away from nadir past the limb.
  auto looking_away = good;
  looking_away.nadir_body = Vec3B(-Eigen::Vector3d::UnitZ());
  refuses(looking_away, "Earth behind the sensor");

  // Sun and Earth centre collinear: the pull direction is undetermined, and its
  // magnitude is zero there anyway.
  auto collinear = good;
  collinear.sun_meas = Vec3B(Eigen::Vector3d::UnitZ());
  refuses(collinear, "Sun on the Earth's centre");

  // Non-finite inputs must never reach the rotation.
  auto nan_sun = good;
  nan_sun.sun_meas = Vec3B(Eigen::Vector3d(std::nan(""), 0.0, 1.0));
  refuses(nan_sun, "non-finite sun measurement");
  auto nan_nadir = good;
  nan_nadir.nadir_body = Vec3B(Eigen::Vector3d(0.0, std::nan(""), 1.0));
  refuses(nan_nadir, "non-finite nadir");
  auto zero_sun = good;
  zero_sun.sun_meas = Vec3B(Eigen::Vector3d::Zero());
  refuses(zero_sun, "zero-length sun measurement");
}

TEST(AlbedoCorrection, RefusesAnUnconfiguredOrNonsensicalUnit) {
  // Parameters have no flight defaults (§19.3), so an unset config must refuse
  // rather than correct by an implied zero — and an out-of-range one must be
  // caught here, before it reaches the estimator's primary vector source.
  gnc::AlbedoCorrectionInput in{};
  in.sun_meas = Vec3B(sunAtIncidence(20.0));
  in.nadir_body = Vec3B(Eigen::Vector3d::UnitZ());
  in.radius_m = kOrbitRadius;
  in.dayside = 1.0;

  pm::Vec3<pmf::Body> out;
  double applied = 0.0;

  EXPECT_FALSE(gnc::albedoCorrection(gnc::AlbedoCorrectionConfig{}, in, out, applied))
      << "default-constructed config must refuse";

  auto degrees_not_radians = fssConfig();
  degrees_not_radians.albedo_error_rad = 12.0;  // the units mistake this gate is for
  EXPECT_FALSE(gnc::albedoCorrection(degrees_not_radians, in, out, applied));

  auto no_fov = fssConfig();
  no_fov.half_fov_rad = 0.0;
  EXPECT_FALSE(gnc::albedoCorrection(no_fov, in, out, applied));

  auto no_boresight = fssConfig();
  no_boresight.boresight_body = Vec3B(Eigen::Vector3d::Zero());
  EXPECT_FALSE(gnc::albedoCorrection(no_boresight, in, out, applied));

  auto nan_boresight = fssConfig();
  nan_boresight.boresight_body = Vec3B(Eigen::Vector3d(std::nan(""), 0.0, 1.0));
  EXPECT_FALSE(gnc::albedoCorrection(nan_boresight, in, out, applied));
}

TEST(AlbedoCorrection, NeverMovesTheVectorFurtherThanTheModelledPull) {
  // A bounded-output check on a flight path: whatever the geometry, the applied
  // rotation cannot exceed the part's configured peak, so a bad input cannot turn
  // the estimator's primary vector source into a direction pointing somewhere
  // else entirely.
  const gnc::AlbedoCorrectionConfig cfg = fssConfig();
  for (double earth_off = 0.0; earth_off <= 180.0; earth_off += 3.0) {
    for (double sun_incidence = 0.0; sun_incidence <= 60.0; sun_incidence += 3.0) {
      gnc::AlbedoCorrectionInput in{};
      in.sun_meas = Vec3B(sunAtIncidence(sun_incidence));
      in.nadir_body =
          Vec3B(Eigen::Vector3d(0.0, std::sin(earth_off * kDeg), std::cos(earth_off * kDeg)));
      in.radius_m = kOrbitRadius;
      in.dayside = 1.0;

      pm::Vec3<pmf::Body> out;
      double applied = 0.0;
      if (!gnc::albedoCorrection(cfg, in, out, applied)) {
        continue;
      }
      EXPECT_TRUE(out.isFinite());
      EXPECT_NEAR(out.eigen().norm(), 1.0, 1e-12);
      EXPECT_GE(applied, 0.0);
      EXPECT_LE(applied, cfg.albedo_error_rad);
      EXPECT_NEAR(angleBetween(out.eigen(), in.sun_meas.eigen()), applied, 1e-12);
    }
  }
}
