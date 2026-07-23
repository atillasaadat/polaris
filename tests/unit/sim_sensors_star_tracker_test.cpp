/// @file Unit tests for the star tracker truth model (§6.2).
///
/// Four concerns. (1) The datasheet→SI conversion, including the 3σ→1σ division
/// and the exclusion-angle convention. (2) The **error decomposition**: bias,
/// thermo-elastic, low- and high-frequency spatial, and temporal noise are
/// distinct mechanisms with different time behaviour, and a test that only
/// checked the total σ would let any of them be silently folded into another.
/// (3) **Availability as a state machine**: acquisition and tracking envelopes
/// are separate, and re-acquisition costs the lost-in-space time. (4) Geometry,
/// reproducibility, and fault hooks.
///
/// The Auriga fixture below mirrors config/hardware/star_tracker/sodern_auriga.yaml
/// (design doc §19.4 — hardcoded specs are permitted in tests, and only there).

#include <gtest/gtest.h>

#include <cmath>
#include <Eigen/Core>
#include <vector>

#include "constants/constants.hpp"
#include "math/frames.hpp"
#include "math/quaternion.hpp"
#include "math/typed_vector.hpp"
#include "sensors/star_tracker.hpp"
#include "time/timescales.hpp"

namespace {

namespace sensors = polaris::sim::sensors;
namespace pm = polaris::math;
namespace pmf = polaris::math::frames;
namespace pt = polaris::time;

using QuatBI = pm::Quat<pmf::Body, pmf::ECI>;
using Vec3B = pm::Vec3<pmf::Body>;
using Mode = sensors::StarTrackerMode;

constexpr double kDeg2Rad = 0.017453292519943295;
constexpr double kArcsec2Rad = kDeg2Rad / 3600.0;
constexpr double kRe = polaris::constants::wgs84::kSemiMajorAxis;
constexpr double kAu = polaris::constants::bodies::kAstronomicalUnit;
const pt::Tai kEpoch = pt::Tai::fromNanosecondsSinceEpoch(1767225637000000000LL);

/// Sodern Auriga, end-of-life worst case (datasheet p.5).
sensors::StarTrackerSpec aurigaSpec() {
  return sensors::StarTrackerSpec::fromParams({
      {"lf_spatial_xy_arcsec_3sigma", 9.0},
      {"lf_spatial_z_arcsec_3sigma", 51.0},
      {"hf_spatial_xy_arcsec_3sigma", 6.6},
      {"hf_spatial_z_arcsec_3sigma", 38.0},
      {"lf_spatial_correlation_s", 300.0},
      {"hf_spatial_correlation_s", 10.0},
      {"temporal_noise_xy_arcsec_3sigma", 11.0},
      {"temporal_noise_z_arcsec_3sigma", 70.0},
      {"bias_deg", 0.017},
      {"thermo_elastic_arcsec_per_c", 1.5},
      {"acquisition_rate_deg_s", 2.0},
      {"tracking_rate_deg_s", 3.0},
      {"acquisition_accel_deg_s2", 1.0},
      {"tracking_accel_deg_s2", 2.5},
      {"lost_in_space_s", 3.8},
      {"update_rate_hz", 10.0},
      {"fov_deg", 20.0},
      {"sun_exclusion_deg", 35.0},
      {"earth_exclusion_deg", 22.0},
      {"moon_exclusion_deg", 0.0},
  });
}

/// Spacecraft on +x at 500 km, Sun on +y, Moon on -y.
///
/// At 500 km the Earth's limb sits 67.6° from nadir, so a boresight perpendicular
/// to nadir clears it by only 22.4° — right at the Auriga's 22° Earth exclusion.
/// Only a near-zenith look direction is comfortably unoccluded.
sensors::SkyGeometry sky() {
  sensors::SkyGeometry s;
  s.sat = Eigen::Vector3d(kRe + 500e3, 0.0, 0.0);
  s.sun = Eigen::Vector3d(0.0, kAu, 0.0);
  s.moon = Eigen::Vector3d(0.0, -3.844e8, 0.0);
  return s;
}

/// Mounting that puts the sensor's +z boresight along body +x — with an identity
/// attitude that is ECI +x, i.e. zenith for @ref sky(). The nominal test posture.
Eigen::Matrix3d zenithMount() {
  Eigen::Matrix3d m;
  m << 0, 0, 1, 0, 1, 0, -1, 0, 0;
  return m;
}

/// A quiescent input: pointing at zenith, not moving, at calibration temperature.
sensors::StarTrackerInput restingInput() {
  sensors::StarTrackerInput in;
  in.attitude = QuatBI::Identity();
  in.sky = sky();
  return in;
}

/// Drive a tracker through acquisition so it is tracking, and return it there.
void bringUp(sensors::StarTracker& st, const sensors::StarTrackerInput& in, double dt = 1.0) {
  for (int i = 0; i < 100 && st.mode() != Mode::kTracking; ++i) {
    (void)st.sample(kEpoch, dt, in);
  }
  ASSERT_EQ(st.mode(), Mode::kTracking);
}

/// Angle [rad] between two attitudes, via the relative rotation.
double attitudeError(const QuatBI& a, const QuatBI& b) {
  const pm::Quaternion delta = a.core() * b.core().conjugate();
  return 2.0 * std::asin(std::min(1.0, delta.vec().norm()));
}

/// Small-angle error vector of a measurement against truth, in body axes.
Eigen::Vector3d errorVector(const sensors::StarTrackerMeasurement& m, const QuatBI& truth) {
  return 2.0 * (m.attitude.core() * truth.core().conjugate()).vec();
}

}  // namespace

// --- Datasheet -> SI ---------------------------------------------------------

TEST(StarTrackerSpec, AurigaDatasheetParamsConvertToSi) {
  const auto s = aurigaSpec();
  // Vendors quote 3σ; the model works in 1σ. Getting this wrong would make the
  // tracker three times better than the unit you bought.
  EXPECT_NEAR(s.low_freq_spatial.cross, 9.0 * kArcsec2Rad / 3.0, 1e-15);
  EXPECT_NEAR(s.low_freq_spatial.boresight, 51.0 * kArcsec2Rad / 3.0, 1e-15);
  EXPECT_NEAR(s.high_freq_spatial.cross, 6.6 * kArcsec2Rad / 3.0, 1e-15);
  EXPECT_NEAR(s.high_freq_spatial.boresight, 38.0 * kArcsec2Rad / 3.0, 1e-15);
  EXPECT_NEAR(s.temporal.cross, 11.0 * kArcsec2Rad / 3.0, 1e-15);
  EXPECT_NEAR(s.temporal.boresight, 70.0 * kArcsec2Rad / 3.0, 1e-15);

  EXPECT_NEAR(s.bias_bound, 0.017 * kDeg2Rad, 1e-15);
  EXPECT_NEAR(s.thermo_elastic_per_k, 1.5 * kArcsec2Rad, 1e-18);

  EXPECT_NEAR(s.acquisition_rate_limit, 2.0 * kDeg2Rad, 1e-15);
  EXPECT_NEAR(s.tracking_rate_limit, 3.0 * kDeg2Rad, 1e-15);
  EXPECT_NEAR(s.acquisition_accel_limit, 1.0 * kDeg2Rad, 1e-15);
  EXPECT_NEAR(s.tracking_accel_limit, 2.5 * kDeg2Rad, 1e-15);
  EXPECT_DOUBLE_EQ(s.lost_in_space_s, 3.8);
  EXPECT_DOUBLE_EQ(s.update_rate_hz, 10.0);

  // Exclusion angles are absolute boresight-to-limb angles, the vendor
  // convention — 22° here, not 22° added to the half-FOV.
  EXPECT_NEAR(s.keep_out.sun_rad, 35.0 * kDeg2Rad, 1e-15);
  EXPECT_NEAR(s.keep_out.earth_rad, 22.0 * kDeg2Rad, 1e-15);
  // "Full Moon in the field of view: no performance degradation" — the baffle
  // handles the Moon outright, so there is no lunar cone at all.
  EXPECT_DOUBLE_EQ(s.keep_out.moon_rad, 0.0);
}

TEST(StarTrackerSpec, EarthExclusionIsNeverLooserThanTheFieldOfView) {
  // A limb inside the FOV floods the detector whatever the baffle is rated for,
  // so the constraint is the larger of the two.
  const auto wide = sensors::StarTrackerSpec::fromParams({
      {"fov_deg", 60.0},
      {"earth_exclusion_deg", 22.0},
  });
  EXPECT_NEAR(wide.keep_out.earth_rad, 30.0 * kDeg2Rad, 1e-15) << "half-FOV wins";

  const auto narrow = sensors::StarTrackerSpec::fromParams({
      {"fov_deg", 20.0},
      {"earth_exclusion_deg", 22.0},
  });
  EXPECT_NEAR(narrow.keep_out.earth_rad, 22.0 * kDeg2Rad, 1e-15) << "exclusion angle wins";
}

// --- Error decomposition -----------------------------------------------------

TEST(StarTracker, PerfectSpecPassesTheTruthAttitudeThrough) {
  sensors::StarTrackerSpec spec;  // no errors, no constraints, no acquisition delay
  sensors::StarTracker st(spec, Eigen::Matrix3d::Identity(), 1, 1);
  auto in = restingInput();
  in.attitude = QuatBI(pm::Quaternion::FromAxisAngle(Eigen::Vector3d(0.0, 0.0, 1.0), 0.3));
  const auto m = st.sample(kEpoch, 0.1, in);
  EXPECT_TRUE(m.valid);
  EXPECT_LT(attitudeError(m.attitude, in.attitude), 1e-15);
}

TEST(StarTracker, NoiseDisabledReportsTruthAttitude) {
  // Ideal build: once tracking, the reported attitude is exactly the truth — no
  // bias, spatial, temporal, or thermo-elastic error. Availability still applies.
  sensors::StarTracker st(aurigaSpec(), zenithMount(), 0xBEEF, 1, /*noise_enabled=*/false);
  auto in = restingInput();
  in.temperature_delta_k = 25.0;  // a large ΔT that would drive thermo-elastic error
  bringUp(st, in);
  const auto m = st.sample(kEpoch, 1.0, in);
  ASSERT_TRUE(m.valid);
  EXPECT_LT(attitudeError(m.attitude, in.attitude), 1e-12);
}

TEST(StarTracker, EveryErrorTermIsAnisotropicAboutTheBoresight) {
  // Roll is the weak axis for every mechanism, not just the noise. Measured along
  // the actual boresight rather than a body axis, so the assertion holds for any
  // mounting.
  sensors::StarTracker st(aurigaSpec(), zenithMount(), 0xBEEF, 1);
  const auto in = restingInput();
  bringUp(st, in);

  const Eigen::Vector3d bore = st.boresightBody().normalized();
  constexpr int n = 20000;
  double sum_sq_bore = 0.0;
  double sum_sq_cross = 0.0;
  for (int i = 0; i < n; ++i) {
    // A long step relative to both correlation times, so the spatial terms are
    // effectively redrawn each sample and the total is their stationary sum.
    const auto m = st.sample(kEpoch, 600.0, in);
    const Eigen::Vector3d err = errorVector(m, in.attitude) - st.unitBias();
    const double along = err.dot(bore);
    sum_sq_bore += along * along;
    sum_sq_cross += (err - along * bore).squaredNorm();
  }
  const double sigma_bore = std::sqrt(sum_sq_bore / n);
  const double sigma_cross = std::sqrt(sum_sq_cross / (2 * n));

  // Total 1σ is the root-sum-square of the three independent mechanisms.
  const auto s = aurigaSpec();
  const double expect_cross = std::sqrt(s.temporal.cross * s.temporal.cross +
                                        s.low_freq_spatial.cross * s.low_freq_spatial.cross +
                                        s.high_freq_spatial.cross * s.high_freq_spatial.cross);
  const double expect_bore =
      std::sqrt(s.temporal.boresight * s.temporal.boresight +
                s.low_freq_spatial.boresight * s.low_freq_spatial.boresight +
                s.high_freq_spatial.boresight * s.high_freq_spatial.boresight);
  EXPECT_NEAR(sigma_cross, expect_cross, expect_cross * 0.05);
  EXPECT_NEAR(sigma_bore, expect_bore, expect_bore * 0.05);
  EXPECT_GT(sigma_bore, 4.0 * sigma_cross) << "about-boresight must be the weak axis";
}

TEST(StarTracker, SpatialErrorsAreCorrelatedInTimeWhileNoiseIsNot) {
  // The defining difference between the spatial terms and the temporal one, and
  // the reason they are separate fields at all: an estimator tuned for white
  // noise expects error to average down as √N, and the spatial terms do not.
  //
  // The comparison is against the model's own white configuration rather than a
  // computed number, because the *magnitude* of the lag-1 autocorrelation
  // depends on the observation window (over 400 s the 300 s term is nearly
  // frozen, so it contributes little sample variance — being frozen is itself
  // the non-whiteness). What must hold regardless of window is that turning the
  // correlation times off collapses the autocorrelation to zero.
  const auto in = restingInput();

  auto lagOneAutocorrelation = [&in](const sensors::StarTrackerSpec& spec) {
    sensors::StarTracker st(spec, zenithMount(), 0x5EED, 2);
    for (int i = 0; i < 100 && st.mode() != Mode::kTracking; ++i) {
      (void)st.sample(kEpoch, 4.0, in);
    }
    // Warm the Gauss-Markov states to their stationary distribution: they start
    // at zero, and a ramping state would read as less correlation, not more.
    for (int i = 0; i < 20; ++i) {
      (void)st.sample(kEpoch, 1000.0, in);
    }
    constexpr int n = 4000;
    std::vector<double> series;
    series.reserve(n);
    for (int i = 0; i < n; ++i) {
      series.push_back(
          errorVector(st.sample(kEpoch, 0.1, in), in.attitude).dot(st.boresightBody()));
    }
    double mean = 0.0;
    for (double v : series) {
      mean += v;
    }
    mean /= static_cast<double>(series.size());
    double var = 0.0;
    double cov = 0.0;
    for (std::size_t i = 0; i < series.size(); ++i) {
      var += (series[i] - mean) * (series[i] - mean);
      if (i > 0) {
        cov += (series[i] - mean) * (series[i - 1] - mean);
      }
    }
    var /= static_cast<double>(series.size());
    cov /= static_cast<double>(series.size() - 1);
    return cov / var;
  };

  const double correlated = lagOneAutocorrelation(aurigaSpec());

  sensors::StarTrackerSpec white = aurigaSpec();
  white.low_freq_correlation_s = 0.0;  // degenerate to white noise
  white.high_freq_correlation_s = 0.0;
  const double uncorrelated = lagOneAutocorrelation(white);

  EXPECT_GT(correlated, 0.15) << "spatial error must persist between samples";
  EXPECT_LT(std::abs(uncorrelated), 0.05) << "with no correlation time it is white";
  EXPECT_GT(correlated, 3.0 * std::abs(uncorrelated));
}

TEST(StarTracker, BiasIsFixedPerUnitAndWithinTheDatasheetBound) {
  // Bias does not average down — it is the term a pointing budget carries in
  // full. Each seeded unit is a distinct device, but none exceeds the worst case.
  const auto spec = aurigaSpec();
  for (std::uint64_t seed = 1; seed <= 40; ++seed) {
    sensors::StarTracker st(spec, zenithMount(), seed, 1);
    EXPECT_LE(st.unitBias().norm(), spec.bias_bound + 1e-15) << "seed " << seed;
  }

  // And it is genuinely constant across samples for one unit: the mean error of
  // a long run converges on this unit's bias, not on zero.
  sensors::StarTracker st(spec, zenithMount(), 7, 1);
  const auto in = restingInput();
  bringUp(st, in);
  Eigen::Vector3d sum = Eigen::Vector3d::Zero();
  constexpr int n = 20000;
  for (int i = 0; i < n; ++i) {
    sum += errorVector(st.sample(kEpoch, 600.0, in), in.attitude);
  }
  const Eigen::Vector3d mean = sum / n;
  EXPECT_LT((mean - st.unitBias()).norm(), 0.1 * spec.bias_bound);
  EXPECT_GT(st.unitBias().norm(), 0.0);
}

TEST(StarTracker, ThermoElasticErrorScalesWithTemperatureDeparture) {
  // 1.5 arcsec/°C: at 20 °C off calibration that is 30 arcsec — comparable with
  // the whole low-frequency spatial budget, so a thermally swinging mount is not
  // a second-order effect.
  sensors::StarTrackerSpec spec = aurigaSpec();
  spec.temporal = {};  // isolate the thermal term
  spec.low_freq_spatial = {};
  spec.high_freq_spatial = {};
  spec.bias_bound = 0.0;
  spec.lost_in_space_s = 0.0;

  sensors::StarTracker st(spec, zenithMount(), 3, 1);
  auto in = restingInput();
  (void)st.sample(kEpoch, 1.0, in);  // acquire

  in.temperature_delta_k = 0.0;
  EXPECT_LT(attitudeError(st.sample(kEpoch, 1.0, in).attitude, in.attitude), 1e-15);

  in.temperature_delta_k = 20.0;
  const double error = attitudeError(st.sample(kEpoch, 1.0, in).attitude, in.attitude);
  EXPECT_NEAR(error, 20.0 * 1.5 * kArcsec2Rad, 1e-9);

  // Linear, and reverses sign with the temperature departure.
  in.temperature_delta_k = -10.0;
  EXPECT_NEAR(attitudeError(st.sample(kEpoch, 1.0, in).attitude, in.attitude),
              10.0 * 1.5 * kArcsec2Rad, 1e-9);
}

// --- Availability state machine ----------------------------------------------

TEST(StarTracker, LostInSpaceDelaysTheFirstFix) {
  // 3.8 s typical: a tracker does not produce an attitude the instant it is
  // switched on or the instant the sky clears.
  sensors::StarTracker st(aurigaSpec(), zenithMount(), 1, 1);
  const auto in = restingInput();
  EXPECT_EQ(st.mode(), Mode::kLost);

  // 3 seconds in 1 s steps: still acquiring.
  for (int i = 0; i < 3; ++i) {
    const auto m = st.sample(kEpoch, 1.0, in);
    EXPECT_FALSE(m.valid) << "second " << i + 1;
    EXPECT_EQ(m.mode, Mode::kAcquiring);
    EXPECT_NEAR(m.acquisition_elapsed_s, i + 1.0, 1e-12);
  }
  // The step that crosses 3.8 s produces the first solution.
  const auto fixed = st.sample(kEpoch, 1.0, in);
  EXPECT_TRUE(fixed.valid);
  EXPECT_EQ(fixed.mode, Mode::kTracking);
}

TEST(StarTracker, AcquisitionAndTrackingRateEnvelopesAreSeparate) {
  // The operationally important asymmetry: the Auriga tracks through 3 °/s but
  // only acquires below 2 °/s. A model with one threshold would resume the
  // instant the vehicle dropped under 3 °/s, which is optimistic by a wide
  // margin — and by exactly the margin that matters during a slew.
  sensors::StarTracker st(aurigaSpec(), zenithMount(), 1, 1);
  auto in = restingInput();
  bringUp(st, in);

  // 2.5 °/s: above the acquisition limit, below the tracking limit. Already
  // tracking, so it keeps tracking.
  in.body_rate = Vec3B(Eigen::Vector3d(2.5 * kDeg2Rad, 0.0, 0.0));
  const auto riding = st.sample(kEpoch, 1.0, in);
  EXPECT_TRUE(riding.valid) << "tracks through a rate it could not acquire at";

  // 4 °/s: beyond the tracking limit too — track is lost.
  in.body_rate = Vec3B(Eigen::Vector3d(4.0 * kDeg2Rad, 0.0, 0.0));
  const auto lost = st.sample(kEpoch, 1.0, in);
  EXPECT_FALSE(lost.valid);
  EXPECT_TRUE(lost.rate_limited);
  EXPECT_EQ(lost.mode, Mode::kLost);

  // Back to 2.5 °/s — inside the tracking envelope, but it cannot *re*-acquire
  // there, so it stays lost however long it waits.
  in.body_rate = Vec3B(Eigen::Vector3d(2.5 * kDeg2Rad, 0.0, 0.0));
  for (int i = 0; i < 20; ++i) {
    const auto m = st.sample(kEpoch, 1.0, in);
    EXPECT_FALSE(m.valid) << "step " << i;
    EXPECT_EQ(m.mode, Mode::kLost);
  }

  // Slow to 1 °/s and it re-acquires, after the lost-in-space delay.
  in.body_rate = Vec3B(Eigen::Vector3d(1.0 * kDeg2Rad, 0.0, 0.0));
  EXPECT_FALSE(st.sample(kEpoch, 1.0, in).valid) << "delay still applies";
  for (int i = 0; i < 5; ++i) {
    (void)st.sample(kEpoch, 1.0, in);
  }
  EXPECT_EQ(st.mode(), Mode::kTracking);
}

TEST(StarTracker, AccelerationEnvelopesAreSeparateFromRate) {
  // A slew transient can sit well inside the rate envelope and still break the
  // solve: 1 °/s² acquiring, 2.5 °/s² tracking.
  sensors::StarTracker st(aurigaSpec(), zenithMount(), 1, 1);
  auto in = restingInput();
  bringUp(st, in);

  // 2 °/s² with negligible rate: fine while tracking.
  in.angular_accel = Vec3B(Eigen::Vector3d(0.0, 2.0 * kDeg2Rad, 0.0));
  EXPECT_TRUE(st.sample(kEpoch, 1.0, in).valid);

  // 3 °/s²: beyond the tracking envelope.
  in.angular_accel = Vec3B(Eigen::Vector3d(0.0, 3.0 * kDeg2Rad, 0.0));
  const auto lost = st.sample(kEpoch, 1.0, in);
  EXPECT_FALSE(lost.valid);
  EXPECT_TRUE(lost.accel_limited);
  EXPECT_FALSE(lost.rate_limited) << "the rate was fine; the acceleration was not";

  // 2 °/s² is inside the tracking envelope but outside the 1 °/s² acquisition
  // one, so it cannot come back until the transient settles.
  in.angular_accel = Vec3B(Eigen::Vector3d(0.0, 2.0 * kDeg2Rad, 0.0));
  for (int i = 0; i < 20; ++i) {
    EXPECT_FALSE(st.sample(kEpoch, 1.0, in).valid) << "step " << i;
  }
  in.angular_accel = Vec3B(Eigen::Vector3d(0.0, 0.5 * kDeg2Rad, 0.0));
  for (int i = 0; i < 6; ++i) {
    (void)st.sample(kEpoch, 1.0, in);
  }
  EXPECT_EQ(st.mode(), Mode::kTracking);
}

TEST(StarTracker, RecoveryFromAnOcclusionPaysTheLostInSpaceTime) {
  // Geometry drives the same state machine: the sky clearing is the beginning of
  // an acquisition, not the end of an outage.
  sensors::StarTracker st(aurigaSpec(), zenithMount(), 1, 1);
  auto in = restingInput();
  bringUp(st, in);

  // Put the Sun on the boresight (zenith), inside the 35° exclusion cone.
  in.sky.sun = in.sky.sat.normalized() * kAu;
  const auto blinded = st.sample(kEpoch, 1.0, in);
  EXPECT_FALSE(blinded.valid);
  EXPECT_EQ(blinded.occlusion.occluder, sensors::Occluder::kSun);

  in.sky = sky();  // Sun back off the boresight
  EXPECT_FALSE(st.sample(kEpoch, 1.0, in).valid) << "no instant recovery";
  for (int i = 0; i < 5; ++i) {
    (void)st.sample(kEpoch, 1.0, in);
  }
  EXPECT_EQ(st.mode(), Mode::kTracking);
}

TEST(StarTracker, FullMoonInTheFieldOfViewIsNotAnOutage) {
  // Datasheet: "Full Moon in the field of view — no performance degradation."
  // The zero lunar exclusion angle is a claim about the baffle, so staring
  // straight at the Moon must still solve.
  sensors::StarTracker st(aurigaSpec(), zenithMount(), 1, 1);
  auto in = restingInput();
  in.sky.moon = in.sky.sat.normalized() * 3.844e8;  // Moon on the boresight
  bringUp(st, in);
  const auto m = st.sample(kEpoch, 1.0, in);
  EXPECT_TRUE(m.valid);
  EXPECT_EQ(m.occlusion.occluder, sensors::Occluder::kNone);
  EXPECT_GT(m.occlusion.moon_fraction, 0.0) << "it is in the field of view, just harmless";
}

TEST(StarTracker, EarthExclusionInvalidatesTheSolution) {
  Eigen::Matrix3d nadir;
  nadir << 0, 0, -1, 0, 1, 0, 1, 0, 0;  // sensor +z -> body -x, straight down
  sensors::StarTracker st(aurigaSpec(), nadir, 1, 1);
  const auto m = st.sample(kEpoch, 1.0, restingInput());
  EXPECT_FALSE(m.valid);
  EXPECT_EQ(m.occlusion.occluder, sensors::Occluder::kEarth);
  EXPECT_DOUBLE_EQ(m.occlusion.earth_fraction, 1.0);
}

TEST(StarTracker, ReportsHowMuchOfTheFieldOfViewEachBodyCovers) {
  // The tracker carries the occlusion state on every sample, so a consumer can
  // watch the Earth march into the field rather than only learn that it arrived.
  sensors::StarTracker st(aurigaSpec(), zenithMount(), 1, 1);
  const auto in = restingInput();
  bringUp(st, in);
  const auto clear = st.sample(kEpoch, 1.0, in);
  EXPECT_DOUBLE_EQ(clear.occlusion.earth_atmosphere_fraction, 0.0) << "zenith stare is clear";

  Eigen::Matrix3d nadir;
  nadir << 0, 0, -1, 0, 1, 0, 1, 0, 0;
  sensors::StarTracker down(aurigaSpec(), nadir, 1, 1);
  const auto blocked = down.sample(kEpoch, 1.0, in);
  EXPECT_DOUBLE_EQ(blocked.occlusion.earth_atmosphere_fraction, 1.0);
  EXPECT_DOUBLE_EQ(blocked.occlusion.blockedFraction(), 1.0);
}

// --- Contract ----------------------------------------------------------------

TEST(StarTracker, MountingRotatesTheBoresightIntoBodyAxes) {
  sensors::StarTracker st(aurigaSpec(), zenithMount(), 1, 1);
  EXPECT_TRUE(st.boresightBody().isApprox(Eigen::Vector3d::UnitX()));
}

TEST(StarTracker, IsBitReproducibleFromSeed) {
  sensors::StarTracker a(aurigaSpec(), zenithMount(), 0xABCD, 3);
  sensors::StarTracker b(aurigaSpec(), zenithMount(), 0xABCD, 3);
  EXPECT_EQ(a.unitBias(), b.unitBias());
  const auto in = restingInput();
  for (int i = 0; i < 50; ++i) {
    EXPECT_EQ(a.sample(kEpoch, 1.0, in).attitude.core().coeffs(),
              b.sample(kEpoch, 1.0, in).attitude.core().coeffs())
        << "sample " << i;
  }
}

TEST(StarTracker, DifferentUnitsGetDifferentBiases) {
  // Two identical part numbers are not the same device.
  sensors::StarTracker a(aurigaSpec(), zenithMount(), 0xABCD, 1);
  sensors::StarTracker b(aurigaSpec(), zenithMount(), 0xABCD, 2);
  EXPECT_NE(a.unitBias(), b.unitBias());
}

TEST(StarTracker, AnOutageDoesNotDisturbTheNoiseStream) {
  // A geometric outage must not shift the stream position, or changing only the
  // orbit would silently change the noise on every later sample (§3.6).
  const auto in = restingInput();
  sensors::StarTracker clear(aurigaSpec(), zenithMount(), 0x5EED, 9);
  sensors::StarTracker blocked(aurigaSpec(), zenithMount(), 0x5EED, 9);

  blocked.setDropout(true);
  (void)blocked.sample(kEpoch, 1.0, in);
  blocked.clearFaults();
  (void)clear.sample(kEpoch, 1.0, in);

  for (int i = 0; i < 20; ++i) {
    EXPECT_EQ(clear.sample(kEpoch, 1.0, in).attitude.core().coeffs(),
              blocked.sample(kEpoch, 1.0, in).attitude.core().coeffs())
        << "sample " << i;
  }
}

TEST(StarTracker, NonPositiveDtIsAnInvalidNoOp) {
  // Consistent with the IMU: a degenerate step must not advance the stream, or
  // the noise would depend on how the caller chose to step.
  sensors::StarTracker a(aurigaSpec(), zenithMount(), 11, 1);
  sensors::StarTracker b(aurigaSpec(), zenithMount(), 11, 1);
  const auto in = restingInput();

  EXPECT_FALSE(a.sample(kEpoch, 0.0, in).valid);
  EXPECT_FALSE(a.sample(kEpoch, -1.0, in).valid);
  for (int i = 0; i < 10; ++i) {
    EXPECT_EQ(a.sample(kEpoch, 1.0, in).attitude.core().coeffs(),
              b.sample(kEpoch, 1.0, in).attitude.core().coeffs())
        << "sample " << i;
  }
}

// --- Fault injection ---------------------------------------------------------

TEST(StarTracker, DropoutAndAttitudeBiasFaults) {
  sensors::StarTracker st(aurigaSpec(), zenithMount(), 7, 1);
  const auto in = restingInput();
  bringUp(st, in);

  st.setDropout(true);
  EXPECT_FALSE(st.sample(kEpoch, 1.0, in).valid);
  st.clearFaults();
  // Recovery from a dropout is a re-acquisition, delay included.
  for (int i = 0; i < 5; ++i) {
    (void)st.sample(kEpoch, 1.0, in);
  }
  EXPECT_EQ(st.mode(), Mode::kTracking);

  // A 0.5° bias jump must show up as ~0.5° of attitude error — orders of
  // magnitude above the arcsecond errors, which is what makes it detectable by
  // FDIR. All stochastic terms are off here so the assertion isolates the fault.
  sensors::StarTrackerSpec quiet = aurigaSpec();
  quiet.temporal = {};
  quiet.low_freq_spatial = {};
  quiet.high_freq_spatial = {};
  quiet.bias_bound = 0.0;
  quiet.lost_in_space_s = 0.0;
  sensors::StarTracker noiseless(quiet, zenithMount(), 7, 1);
  (void)noiseless.sample(kEpoch, 1.0, in);

  const double bias = 0.5 * kDeg2Rad;
  noiseless.injectAttitudeBias(Vec3B(Eigen::Vector3d(bias, 0.0, 0.0)));
  const auto m = noiseless.sample(kEpoch, 1.0, in);
  EXPECT_TRUE(m.valid) << "a bias is a wrong answer, not a lost one";
  EXPECT_NEAR(attitudeError(m.attitude, in.attitude), bias, 1.0e-12);
}
