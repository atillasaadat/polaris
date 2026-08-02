/// @file Unit tests for the sun sensor truth model (§6.2).
///
/// Two output contracts are exercised, because they are genuinely different
/// parts. (1) **Analogue**: per-diode counts following the cosine law, cut off
/// at the acceptance cone, with albedo, dark current, quantization and
/// saturation — the FSW reconstructs the direction from these. (2) **Digital**
/// (GomSpace NanoSense FSS): the unit reports a vector whose accuracy depends on
/// the incidence angle, and is sampled no faster than its own period.
///
/// The properties that matter most and are easiest to get quietly wrong: the
/// cosine law (not a constant), the FOV cut-off, albedo scaling with how much
/// sunlit Earth is in view, eclipse invalidating rather than zeroing, and the
/// accuracy regime switching at the datasheet's angle.
///
/// The Auriga-style fixtures mirror config/hardware/sun_sensor/*.yaml (design
/// doc §19.4 — hardcoded specs are permitted in tests, and only there).

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <Eigen/Core>
#include <vector>

#include "constants/constants.hpp"
#include "math/frames.hpp"
#include "math/typed_vector.hpp"
#include "sensors/sun_sensor.hpp"
#include "time/timescales.hpp"

namespace {

namespace sensors = polaris::sim::sensors;
namespace pm = polaris::math;
namespace pmf = polaris::math::frames;
namespace pt = polaris::time;

using Vec3B = pm::Vec3<pmf::Body>;

constexpr double kDeg2Rad = 0.017453292519943295;
constexpr double kRe = polaris::constants::wgs84::kSemiMajorAxis;
constexpr double kAu = polaris::constants::bodies::kAstronomicalUnit;
const pt::Tai kEpoch = pt::Tai::fromNanosecondsSinceEpoch(1767225637000000000LL);

pt::Tai epochPlus(double seconds) {
  return pt::Tai::fromNanosecondsSinceEpoch(kEpoch.nanosecondsSinceEpoch() +
                                            static_cast<std::int64_t>(seconds * 1.0e9));
}

/// Single-cell coarse sun sensor, mirroring coarse_generic.yaml.
sensors::SunSensorSpec coarseSpec() {
  return sensors::SunSensorSpec::fromParams({
      {"diode_count", 1},
      {"diode_cant_deg", 0.0},
      {"half_fov_deg", 60.0},
      {"full_scale_counts", 4095.0},
      {"saturation_counts", 4095.0},
      {"dark_counts", 20.0},
      {"noise_counts_rms", 8.0},
      {"resolution_counts", 1.0},
      {"scale_factor_pct", 3.0},
      {"alignment_mrad", 5.0},
      {"albedo_coefficient", 0.3},
      {"update_rate_hz", 10.0},
  });
}

/// GomSpace NanoSense FSS, mirroring gomspace_nanosense_fss.yaml (datasheet §8).
sensors::SunSensorSpec nanoSenseSpec() {
  return sensors::SunSensorSpec::fromParams({
      {"diode_count", 4},
      {"diode_cant_deg", 0.0},
      {"half_fov_deg", 60.0},
      {"accuracy_inner_half_angle_deg", 45.0},
      {"accuracy_inner_deg_3sigma", 0.5},
      {"accuracy_outer_deg_3sigma", 2.0},
      {"albedo_error_deg", 12.0},
      {"sample_period_ms", 10.0},
      {"update_rate_hz", 100.0},
  });
}

/// Deep space: no Earth anywhere near, so albedo cannot contribute. Isolates the
/// clean-sky behaviour the datasheet's accuracy figures are quoted under.
sensors::SunSensorInput cleanSky(const Eigen::Vector3d& sun_body) {
  sensors::SunSensorInput in;
  in.sun_dir_body = Vec3B(sun_body.normalized());
  in.shadow_factor = 1.0;
  in.sky.sat = Eigen::Vector3d(kAu, 0.0, 0.0);  // far from Earth
  in.sky.sun = Eigen::Vector3d(2.0 * kAu, 0.0, 0.0);
  return in;
}

/// A 500 km LEO day-side geometry with the Earth below (body -z) and the Sun
/// wherever the caller puts it.
sensors::SunSensorInput leoDayside(const Eigen::Vector3d& sun_body) {
  sensors::SunSensorInput in;
  in.sun_dir_body = Vec3B(sun_body.normalized());
  in.nadir_dir_body = Vec3B(Eigen::Vector3d(0.0, 0.0, -1.0));
  in.shadow_factor = 1.0;
  // Sub-satellite point fully sunlit: sat and Sun on the same side of the Earth.
  in.sky.sat = Eigen::Vector3d(kRe + 500e3, 0.0, 0.0);
  in.sky.sun = Eigen::Vector3d(kAu, 0.0, 0.0);
  return in;
}

/// Sun direction at a given incidence angle from the sensor boresight (+z).
Eigen::Vector3d sunAtIncidence(double deg) {
  const double a = deg * kDeg2Rad;
  return Eigen::Vector3d(std::sin(a), 0.0, std::cos(a));
}

}  // namespace

// --- Analogue: per-diode counts ----------------------------------------------

TEST(SunSensor, DiodeFollowsTheCosineLaw) {
  // The defining response: current is proportional to projected area, so a cell
  // at 60° incidence reads half of what it reads at normal incidence. A model
  // that reported full scale across the field would make every reconstruction
  // look perfect and hide the geometry the FSW has to solve.
  sensors::SunSensorSpec spec = coarseSpec();
  spec.noise_counts = 0.0;  // isolate the response curve
  spec.dark_counts = 0.0;
  spec.scale_factor = 0.0;
  spec.alignment_sigma = 0.0;
  spec.albedo_coefficient = 0.0;
  spec.resolution_counts = 0.0;
  sensors::SunSensor ss(spec, Eigen::Matrix3d::Identity(), 1, 1);

  for (const double deg : {0.0, 15.0, 30.0, 45.0, 59.0}) {
    const auto m = ss.sample(epochPlus(deg), cleanSky(sunAtIncidence(deg)));
    ASSERT_TRUE(m.valid) << deg << " deg";
    EXPECT_NEAR(m.counts[0], 4095.0 * std::cos(deg * kDeg2Rad), 1e-9) << deg << " deg";
  }
}

TEST(SunSensor, OutsideTheAcceptanceConeTheCellSeesNothing) {
  sensors::SunSensor ss(coarseSpec(), Eigen::Matrix3d::Identity(), 1, 1);
  const auto inside = ss.sample(epochPlus(0.0), cleanSky(sunAtIncidence(55.0)));
  EXPECT_TRUE(inside.valid);
  EXPECT_TRUE(inside.sun_present);

  // Beyond the 60° half-FOV: no direct signal, so no usable measurement. Not a
  // fault — a sun sensor facing away from the Sun is the normal case.
  const auto outside = ss.sample(epochPlus(1.0), cleanSky(sunAtIncidence(75.0)));
  EXPECT_FALSE(outside.sun_present);
  EXPECT_FALSE(outside.valid);
}

TEST(SunSensor, EclipseInvalidatesRatherThanReadingZero) {
  // In umbra there is no direction information at all. The distinction matters:
  // a zero reading looks like "the Sun is exactly perpendicular", which is a
  // measurement; an invalid reading says the sensor has nothing to contribute.
  sensors::SunSensor ss(coarseSpec(), Eigen::Matrix3d::Identity(), 1, 1);
  auto in = cleanSky(sunAtIncidence(10.0));
  in.shadow_factor = 0.0;
  const auto m = ss.sample(kEpoch, in);
  EXPECT_FALSE(m.sun_present);
  EXPECT_FALSE(m.valid);
  EXPECT_DOUBLE_EQ(m.shadow_factor, 0.0);
}

TEST(SunSensor, PenumbraScalesTheSignalBeforeItInvalidates) {
  sensors::SunSensorSpec spec = coarseSpec();
  spec.noise_counts = 0.0;
  spec.dark_counts = 0.0;
  spec.scale_factor = 0.0;
  spec.alignment_sigma = 0.0;
  spec.albedo_coefficient = 0.0;
  spec.resolution_counts = 0.0;
  sensors::SunSensor ss(spec, Eigen::Matrix3d::Identity(), 1, 1);

  auto in = cleanSky(sunAtIncidence(0.0));
  in.shadow_factor = 0.5;  // half the solar disk occulted
  const auto m = ss.sample(kEpoch, in);
  EXPECT_TRUE(m.valid);
  EXPECT_NEAR(m.counts[0], 4095.0 * 0.5, 1e-9) << "half the disk, half the signal";
}

TEST(SunSensor, AlbedoAddsSignalOnTheDaysideAndNoneAtNight) {
  // Earthshine is the dominant coarse-sensor error, so it must be present with
  // the Earth in view and absent on the night side — a model that applied it
  // uniformly would corrupt eclipse-exit readings that are actually clean.
  sensors::SunSensorSpec spec = coarseSpec();
  spec.noise_counts = 0.0;
  spec.scale_factor = 0.0;
  spec.alignment_sigma = 0.0;
  spec.resolution_counts = 0.0;
  spec.dark_counts = 0.0;
  // Point the cell at nadir so the Earth fills its field; the Sun grazes the
  // edge of the cone so the direct signal stays small next to the albedo.
  Eigen::Matrix3d nadir_mount;
  nadir_mount << 1, 0, 0, 0, -1, 0, 0, 0, -1;  // sensor +z -> body -z
  sensors::SunSensor ss(spec, nadir_mount, 1, 1);

  auto day = leoDayside(sunAtIncidence(0.0));
  const auto lit = ss.sample(kEpoch, day);
  EXPECT_GT(lit.albedo_counts[0], 0.0) << "Earth in view, day side";

  // Night side: the sub-satellite point is unlit, so there is nothing to reflect.
  auto night = day;
  night.sky.sun = Eigen::Vector3d(-kAu, 0.0, 0.0);
  const auto dark = ss.sample(epochPlus(1.0), night);
  EXPECT_DOUBLE_EQ(dark.albedo_counts[0], 0.0);
}

TEST(SunSensor, DarkCurrentAndSaturationBoundTheReading) {
  sensors::SunSensorSpec spec = coarseSpec();
  spec.noise_counts = 0.0;
  spec.scale_factor = 0.0;
  spec.alignment_sigma = 0.0;
  spec.albedo_coefficient = 0.0;
  sensors::SunSensor ss(spec, Eigen::Matrix3d::Identity(), 1, 1);

  // Facing away: only dark current, never negative.
  const auto dark = ss.sample(kEpoch, cleanSky(Eigen::Vector3d(0.0, 0.0, -1.0)));
  EXPECT_GE(dark.counts[0], 0.0);

  // Full scale plus dark current would exceed the ADC range; it clamps.
  const auto bright = ss.sample(epochPlus(1.0), cleanSky(sunAtIncidence(0.0)));
  EXPECT_DOUBLE_EQ(bright.counts[0], 4095.0);
}

TEST(SunSensor, ADeadCellReadsDarkWhileTheOthersKeepWorking) {
  // Partial failure is the interesting FDIR case: the sensor degrades rather
  // than disappearing, and the reconstruction has to cope with one channel gone.
  sensors::SunSensorSpec spec = sensors::SunSensorSpec::fromParams({
      {"diode_count", 4},
      {"diode_cant_deg", 30.0},
      {"half_fov_deg", 60.0},
      {"full_scale_counts", 1000.0},
  });
  sensors::SunSensor ss(spec, Eigen::Matrix3d::Identity(), 1, 1);
  const auto nominal = ss.sample(kEpoch, cleanSky(sunAtIncidence(0.0)));
  ASSERT_EQ(nominal.counts.size(), 4u);
  for (const double c : nominal.counts) {
    EXPECT_GT(c, 0.0);
  }

  ss.failDiode(2);
  const auto degraded = ss.sample(epochPlus(1.0), cleanSky(sunAtIncidence(0.0)));
  EXPECT_DOUBLE_EQ(degraded.counts[2], 0.0);
  EXPECT_GT(degraded.counts[0], 0.0);
  EXPECT_GT(degraded.counts[1], 0.0);
  EXPECT_GT(degraded.counts[3], 0.0);
  EXPECT_TRUE(degraded.valid) << "a dead cell degrades the sensor, it does not remove it";
}

TEST(SunSensor, IsBitReproducibleFromSeedAndUnitsDiffer) {
  const auto in = cleanSky(sunAtIncidence(20.0));
  sensors::SunSensor a(coarseSpec(), Eigen::Matrix3d::Identity(), 0xABCD, 1);
  sensors::SunSensor b(coarseSpec(), Eigen::Matrix3d::Identity(), 0xABCD, 1);
  for (int i = 0; i < 20; ++i) {
    EXPECT_EQ(a.sample(epochPlus(i), in).counts, b.sample(epochPlus(i), in).counts) << i;
  }
  // Different stream ids are different physical units.
  sensors::SunSensor c(coarseSpec(), Eigen::Matrix3d::Identity(), 0xABCD, 2);
  EXPECT_FALSE(c.diodeNormalsBody()[0].isApprox(a.diodeNormalsBody()[0]));
}

TEST(SunSensor, NoiseDisabledAnalogueReportsCleanCosine) {
  // coarseSpec carries dark current, per-sample noise, and a per-diode scale
  // error; an ideal build strips all of them, leaving the exact cosine law.
  sensors::SunSensor ss(coarseSpec(), Eigen::Matrix3d::Identity(), 7, 1, /*noise_enabled=*/false);
  const double deg = 25.0;
  const auto m = ss.sample(kEpoch, cleanSky(sunAtIncidence(deg)));
  ASSERT_TRUE(m.valid);
  EXPECT_NEAR(m.counts[0], 4095.0 * std::cos(deg * kDeg2Rad), 1e-9);
}

// --- Digital: GomSpace NanoSense FSS vector output ----------------------------

TEST(SunSensor, NoiseDisabledVectorReportsTruthDirection) {
  // Ideal digital part: the reported Sun vector is exactly the truth direction,
  // and the realised accuracy is zero.
  const auto in = cleanSky(sunAtIncidence(30.0));
  sensors::SunSensor ss(nanoSenseSpec(), Eigen::Matrix3d::Identity(), 7, 1,
                        /*noise_enabled=*/false);
  const auto m = ss.sample(kEpoch, in);
  ASSERT_TRUE(m.valid);
  EXPECT_DOUBLE_EQ(m.accuracy_sigma_rad, 0.0);
  EXPECT_TRUE(m.sun_dir_body.eigen().isApprox(in.sun_dir_body.eigen(), 1e-12));
}

TEST(SunSensorSpec, NanoSenseDatasheetParamsConvertToSi) {
  const auto s = nanoSenseSpec();
  EXPECT_EQ(s.output, sensors::SunSensorOutput::kSunVector);
  EXPECT_NEAR(s.half_fov_rad, 60.0 * kDeg2Rad, 1e-15);
  EXPECT_NEAR(s.accuracy_inner_half_angle_rad, 45.0 * kDeg2Rad, 1e-15);
  // Vendors quote 3σ; the model works in 1σ.
  EXPECT_NEAR(s.accuracy_inner_sigma, 0.5 * kDeg2Rad / 3.0, 1e-15);
  EXPECT_NEAR(s.accuracy_outer_sigma, 2.0 * kDeg2Rad / 3.0, 1e-15);
  EXPECT_NEAR(s.albedo_error_rad, 12.0 * kDeg2Rad, 1e-15);
  EXPECT_NEAR(s.sample_period_s, 0.010, 1e-15);
}

TEST(SunSensorSpec, ASingleQuotedAccuracyAppliesAcrossTheWholeField) {
  // Guard against a part that quotes one number silently becoming perfect at
  // wide incidence, which is what a zero outer sigma would mean.
  const auto s = sensors::SunSensorSpec::fromParams({
      {"half_fov_deg", 50.0},
      {"accuracy_inner_half_angle_deg", 20.0},
      {"accuracy_inner_deg_3sigma", 1.0},
  });
  EXPECT_NEAR(s.accuracy_outer_sigma, s.accuracy_inner_sigma, 1e-18);
  EXPECT_GT(s.accuracy_outer_sigma, 0.0);
}

TEST(SunSensor, VectorAccuracyDegradesWithIncidenceAngle) {
  // The datasheet's headline behaviour: ±0.5° (3σ) inside 45°, ±2.0° out to the
  // 60° edge — a factor of four across one part's field. Sampled statistically
  // because the error is random; the regime boundary is what is being pinned.
  const auto spec = nanoSenseSpec();
  auto measuredSigma = [&spec](double incidence_deg) {
    sensors::SunSensor ss(spec, Eigen::Matrix3d::Identity(), 0xBEEF, 1);
    const Eigen::Vector3d truth = sunAtIncidence(incidence_deg).normalized();
    constexpr int n = 4000;
    double sum_sq = 0.0;
    for (int i = 0; i < n; ++i) {
      // Step past the sample period each time so every reading is genuinely new.
      const auto m = ss.sample(epochPlus(i * 0.02), cleanSky(truth));
      EXPECT_TRUE(m.valid);
      const double err = std::acos(std::clamp(m.sun_dir_body.eigen().dot(truth), -1.0, 1.0));
      sum_sq += err * err;
    }
    // Two independent perpendicular components, so the per-axis σ is the total
    // angular RMS over √2.
    return std::sqrt(sum_sq / n) / std::sqrt(2.0);
  };

  const double inner = measuredSigma(20.0);
  const double outer = measuredSigma(55.0);
  EXPECT_NEAR(inner, spec.accuracy_inner_sigma, spec.accuracy_inner_sigma * 0.1);
  EXPECT_NEAR(outer, spec.accuracy_outer_sigma, spec.accuracy_outer_sigma * 0.1);
  EXPECT_GT(outer, 3.0 * inner) << "accuracy must degrade toward the field edge";
}

TEST(SunSensor, VectorAccuracyRegimeSwitchesAtTheDatasheetAngle) {
  // The reported σ is carried on the measurement, so an estimator can be handed
  // the noise the sensor actually had rather than a datasheet headline.
  sensors::SunSensor ss(nanoSenseSpec(), Eigen::Matrix3d::Identity(), 1, 1);
  const auto spec = nanoSenseSpec();

  const auto just_inside = ss.sample(epochPlus(0.0), cleanSky(sunAtIncidence(44.0)));
  EXPECT_NEAR(just_inside.accuracy_sigma_rad, spec.accuracy_inner_sigma, 1e-15);
  EXPECT_NEAR(just_inside.incidence_angle_rad, 44.0 * kDeg2Rad, 1e-9);

  const auto just_outside = ss.sample(epochPlus(1.0), cleanSky(sunAtIncidence(46.0)));
  EXPECT_NEAR(just_outside.accuracy_sigma_rad, spec.accuracy_outer_sigma, 1e-15);
}

TEST(SunSensor, VectorOutputReportsNoCountsAtAll) {
  // A digital part has no photocurrents on the bus. Reporting zeros would be
  // indistinguishable from a set of dark cells; the array is empty so a consumer
  // that reaches for counts on the wrong sort of part finds nothing rather than a
  // plausible-looking reading that means nothing.
  sensors::SunSensor ss(nanoSenseSpec(), Eigen::Matrix3d::Identity(), 1, 1);
  const auto m = ss.sample(kEpoch, cleanSky(sunAtIncidence(20.0)));
  ASSERT_TRUE(m.valid);
  EXPECT_TRUE(m.counts.empty());
  EXPECT_TRUE(m.albedo_counts.empty());
  EXPECT_GT(m.sun_dir_body.eigen().norm(), 0.0) << "it reports a vector instead";
}

TEST(SunSensor, VectorOutputIsInvalidBeyondTheFieldOfView) {
  sensors::SunSensor ss(nanoSenseSpec(), Eigen::Matrix3d::Identity(), 1, 1);
  EXPECT_TRUE(ss.sample(epochPlus(0.0), cleanSky(sunAtIncidence(59.0))).valid);
  const auto outside = ss.sample(epochPlus(1.0), cleanSky(sunAtIncidence(61.0)));
  EXPECT_FALSE(outside.valid);
  EXPECT_FALSE(outside.sun_present);
}

namespace {

/// The nanoSense part with albedo dispersion configured, as the reference
/// vehicle flies it (config/hardware/sun_sensor/gomspace_nanosense_fss.yaml).
sensors::SunSensorSpec nanoSenseDispersedSpec() {
  sensors::SunSensorSpec spec = nanoSenseSpec();
  spec.albedo_dispersion_fraction = 0.30;
  return spec;
}

/// The Earth on the boresight and the Sun in the field: the geometry the albedo
/// term is largest in.
sensors::SunSensorInput earthFilledDayside(double sun_incidence_deg) {
  auto in = leoDayside(sunAtIncidence(sun_incidence_deg));
  in.nadir_dir_body = Vec3B(Eigen::Vector3d(0.0, 0.0, 1.0));  // Earth on boresight
  return in;
}

}  // namespace

TEST(SunSensor, AlbedoPullsTheReportedVectorTowardTheEarth) {
  // The property that decides whether the error is correctable at all. Earthshine
  // arrives from the sunlit ground in the field, so it drags the reported vector
  // *toward the Earth* — a deterministic direction, not a random tilt of the same
  // size. Modelling it as noise would make it provably uncorrectable, which is a
  // claim about the physics that is false, and would have hidden the §8.1 sun
  // budget's largest term behind a σ.
  sensors::SunSensorSpec spec = nanoSenseSpec();
  spec.accuracy_inner_sigma = 0.0;  // isolate the albedo from the white draw
  spec.accuracy_outer_sigma = 0.0;
  sensors::SunSensor ss(spec, Eigen::Matrix3d::Identity(), 1, 1);

  const Eigen::Vector3d truth = sunAtIncidence(20.0).normalized();
  const auto in = earthFilledDayside(20.0);
  const Eigen::Vector3d nadir = in.nadir_dir_body.eigen();
  const auto m = ss.sample(kEpoch, in);
  ASSERT_TRUE(m.valid);

  const Eigen::Vector3d reported = m.sun_dir_body.eigen();
  EXPECT_GT(m.albedo_angle_rad, 1.0 * kDeg2Rad) << "the Earth fills the field on the day side";
  // Closer to the Earth than the truth is, and by the reported amount.
  EXPECT_GT(reported.dot(nadir), truth.dot(nadir));
  const double moved = std::acos(std::clamp(reported.dot(truth), -1.0, 1.0));
  EXPECT_NEAR(moved, m.albedo_angle_rad, 1e-12);
  // In the Sun-Earth plane: no component out of it without dispersion.
  EXPECT_NEAR(reported.dot(truth.cross(nadir).normalized()), 0.0, 1e-12);
}

TEST(SunSensor, AlbedoVanishesWithoutSunlitEarthInTheField) {
  // Three ways there is no Earthshine, all of which must give exactly zero rather
  // than a small number or a NaN: no Earth in the field, the night side, and
  // eclipse. The onboard correction refuses in each of them (§8.1), so a truth
  // model that leaked a pull here would leave a bias no correction could see.
  sensors::SunSensorSpec spec = nanoSenseDispersedSpec();
  spec.accuracy_inner_sigma = 0.0;
  spec.accuracy_outer_sigma = 0.0;
  sensors::SunSensor ss(spec, Eigen::Matrix3d::Identity(), 7, 3);

  const auto deep_space = ss.sample(epochPlus(0.0), cleanSky(sunAtIncidence(20.0)));
  EXPECT_DOUBLE_EQ(deep_space.albedo_angle_rad, 0.0) << "no Earth anywhere near";

  // Night side: the sub-satellite point is on the far side of the Earth from the
  // Sun, so the ground below reflects nothing.
  auto night = earthFilledDayside(20.0);
  night.sky.sun = -night.sky.sun;
  const auto dark = ss.sample(epochPlus(1.0), night);
  EXPECT_DOUBLE_EQ(dark.albedo_angle_rad, 0.0);

  // Eclipse: no Sun to reflect, and no measurement either.
  auto eclipsed = earthFilledDayside(20.0);
  eclipsed.shadow_factor = 0.0;
  const auto umbra = ss.sample(epochPlus(2.0), eclipsed);
  EXPECT_DOUBLE_EQ(umbra.albedo_angle_rad, 0.0);
  EXPECT_FALSE(umbra.valid);
}

TEST(SunSensor, AlbedoDispersionIsPerUnitAndScalesTheReportedSigma) {
  // The dispersion is what survives the onboard correction, so two things about
  // it are load-bearing for the §19.2 post-correction sun budget: it is drawn
  // **once per unit** (the surface and cloud field under the orbit changes over
  // minutes, not between 10 Hz samples, so no filter averages it away), and it —
  // not the deterministic pull — is what the reported σ carries.
  const auto spec = nanoSenseDispersedSpec();
  const auto in = earthFilledDayside(20.0);

  sensors::SunSensor a(spec, Eigen::Matrix3d::Identity(), 0xA1BED0, 1);
  sensors::SunSensor b(spec, Eigen::Matrix3d::Identity(), 0xA1BED0, 2);  // different stream = unit

  const auto first = a.sample(epochPlus(0.0), in);
  const auto second = a.sample(epochPlus(1.0), in);
  const auto other_unit = b.sample(epochPlus(0.0), in);

  ASSERT_GT(first.albedo_angle_rad, 0.0);
  EXPECT_DOUBLE_EQ(first.albedo_angle_rad, second.albedo_angle_rad)
      << "the same unit's dispersion must not resample between readings";
  EXPECT_NE(first.albedo_angle_rad, other_unit.albedo_angle_rad)
      << "two units over the same ground must draw different dispersions";

  // The reported σ is the white part plus the dispersion, and excludes the pull:
  // the pull is a bias the correction removes, and reporting it would tell a
  // filter to distrust a reading whose error it can compute.
  const double expected = std::hypot(spec.accuracy_inner_sigma,
                                     spec.albedo_dispersion_fraction * first.albedo_angle_rad);
  EXPECT_NEAR(first.accuracy_sigma_rad, expected, 1e-15);
  EXPECT_LT(first.accuracy_sigma_rad, first.albedo_angle_rad)
      << "the deterministic pull must not be reported as noise";
}

TEST(SunSensor, SamplingFasterThanThePeriodRepeatsTheReading) {
  // The part integrates over 10 ms. Polling faster returns the register contents
  // again — an estimator that treated those as independent would average down
  // noise that never averaged and grow confident on information it never got.
  sensors::SunSensor ss(nanoSenseSpec(), Eigen::Matrix3d::Identity(), 1, 1);
  const auto in = cleanSky(sunAtIncidence(20.0));

  const auto first = ss.sample(kEpoch, in);
  EXPECT_TRUE(first.fresh);

  // 3 ms later: inside the sample period, so the same value comes back.
  const auto repeat = ss.sample(epochPlus(0.003), in);
  EXPECT_FALSE(repeat.fresh);
  EXPECT_EQ(repeat.sun_dir_body.eigen(), first.sun_dir_body.eigen());
  EXPECT_EQ(repeat.time_tag.nanosecondsSinceEpoch(), epochPlus(0.003).nanosecondsSinceEpoch())
      << "the time tag is when it was read, even though the data is stale";

  // 12 ms after the first: the period has elapsed, so this one is genuinely new.
  const auto fresh = ss.sample(epochPlus(0.012), in);
  EXPECT_TRUE(fresh.fresh);
  EXPECT_NE(fresh.sun_dir_body.eigen(), first.sun_dir_body.eigen());
}

TEST(SunSensor, VectorDropoutInvalidatesWithoutDisturbingTheStream) {
  sensors::SunSensor a(nanoSenseSpec(), Eigen::Matrix3d::Identity(), 0x5EED, 4);
  sensors::SunSensor b(nanoSenseSpec(), Eigen::Matrix3d::Identity(), 0x5EED, 4);
  const auto in = cleanSky(sunAtIncidence(20.0));

  a.setDropout(true);
  EXPECT_FALSE(a.sample(kEpoch, in).valid);
  a.clearFaults();
  (void)b.sample(kEpoch, in);

  // Both have drawn the same number of times, so they stay in lockstep.
  for (int i = 1; i < 20; ++i) {
    EXPECT_EQ(a.sample(epochPlus(i * 0.02), in).sun_dir_body.eigen(),
              b.sample(epochPlus(i * 0.02), in).sun_dir_body.eigen())
        << "sample " << i;
  }
}
