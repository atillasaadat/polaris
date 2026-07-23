/// @file Unit tests for the star tracker truth model (§6.2).
///
/// Three concerns. (1) The datasheet→SI conversion, including the Earth keep-out
/// being built from the half-FOV plus the datasheet margin. (2) The error model:
/// a perfect spec passes the truth attitude through, and the noise is
/// **anisotropic** — about-boresight error is far larger than cross-boresight,
/// the property an estimator must not be allowed to assume away. (3) Validity:
/// Earth intrusion, Sun keep-out, and slew-rate smear each invalidate a solution
/// and are reported distinctly, and an outage does not disturb the noise stream.

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

constexpr double kDeg2Rad = 0.017453292519943295;
constexpr double kArcsec2Rad = kDeg2Rad / 3600.0;
constexpr double kRe = polaris::constants::wgs84::kSemiMajorAxis;
constexpr double kAu = polaris::constants::bodies::kAstronomicalUnit;
const pt::Tai kEpoch = pt::Tai::fromNanosecondsSinceEpoch(1767225637000000000LL);

/// An ST-16-class **test fixture** in datasheet-native keys, mirroring
/// config/hardware/star_tracker/st16.yaml (design doc §19.4 — hardcoded specs
/// are permitted in tests, and only in tests).
sensors::StarTrackerSpec st16Spec() {
  return sensors::StarTrackerSpec::fromParams({
      {"cross_axis_arcsec", 5.0},
      {"boresight_arcsec", 30.0},
      {"update_rate_hz", 2.0},
      {"fov_deg", 15.0},
      {"max_slew_rate_deg_s", 3.0},
      {"sun_keepout_deg", 45.0},
      {"earth_keepout_deg", 25.0},
      {"moon_keepout_deg", 25.0},
  });
}

/// Spacecraft on +x at 500 km, Sun on +y, Moon on -y.
///
/// Worth stating why the "clear sky" direction is zenith and not something more
/// neutral: at 500 km the Earth's limb sits 67.6° from nadir, so a boresight
/// perpendicular to nadir clears it by only 22.4° — inside the ST-16's 32.5°
/// Earth constraint. A tracker on this orbit genuinely cannot stare at the
/// horizon, so only a near-zenith look direction is unoccluded.
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

const Vec3B kAtRest{Eigen::Vector3d::Zero()};

/// Angle [rad] between two attitudes, via the relative rotation.
double attitudeError(const QuatBI& a, const QuatBI& b) {
  const pm::Quaternion delta = a.core() * b.core().conjugate();
  return 2.0 * std::asin(std::min(1.0, delta.vec().norm()));
}

}  // namespace

// --- Datasheet -> SI ---------------------------------------------------------

TEST(StarTrackerSpec, DatasheetParamsConvertToSi) {
  const auto s = st16Spec();
  EXPECT_NEAR(s.cross_axis_sigma, 5.0 * kArcsec2Rad, 1e-15);
  EXPECT_NEAR(s.boresight_sigma, 30.0 * kArcsec2Rad, 1e-15);
  EXPECT_NEAR(s.fov_rad, 15.0 * kDeg2Rad, 1e-15);
  EXPECT_NEAR(s.max_slew_rate, 3.0 * kDeg2Rad, 1e-15);
  EXPECT_NEAR(s.keep_out.sun_rad, 45.0 * kDeg2Rad, 1e-15);
  // The Earth constraint is the half-FOV plus the datasheet margin: a limb inside
  // the field of view floods the detector even when it misses the boresight.
  EXPECT_NEAR(s.keep_out.earth_rad, (7.5 + 25.0) * kDeg2Rad, 1e-15);
  EXPECT_DOUBLE_EQ(s.update_rate_hz, 2.0);
}

TEST(StarTrackerSpec, NoFovStillYieldsNoEarthKeepOut) {
  // Guards the half-FOV arithmetic: an entry with neither FOV nor margin must
  // leave the constraint disabled rather than land on a tiny non-zero angle.
  const auto s = sensors::StarTrackerSpec::fromParams({{"cross_axis_arcsec", 5.0}});
  EXPECT_DOUBLE_EQ(s.keep_out.earth_rad, 0.0);
}

// --- Error model -------------------------------------------------------------

TEST(StarTracker, PerfectSpecPassesTheTruthAttitudeThrough) {
  sensors::StarTrackerSpec spec;  // all errors zero, no constraints
  sensors::StarTracker st(spec, Eigen::Matrix3d::Identity(), 1, 1);
  const QuatBI truth(pm::Quaternion::FromAxisAngle(Eigen::Vector3d(0.0, 0.0, 1.0), 0.3));
  const auto m = st.sample(kEpoch, truth, kAtRest, sky());
  EXPECT_TRUE(m.valid);
  EXPECT_LT(attitudeError(m.attitude, truth), 1e-15);
}

TEST(StarTracker, NoiseIsAnisotropicAboutTheBoresight) {
  // The defining property of a star tracker: roll (about the boresight) is the
  // weak axis. An isotropic model would make an estimator overconfident in roll.
  sensors::StarTracker st(st16Spec(), zenithMount(), 0xBEEF, 1);
  const QuatBI truth = QuatBI::Identity();

  // Resolve the error along the *boresight* rather than a body axis: the
  // anisotropy is defined by where the unit is looking, not by how it is bolted
  // on, and asserting on body axes would only hold for one particular mounting.
  const Eigen::Vector3d bore = st.boresightBody();
  constexpr int n = 20000;
  double sum_sq_bore = 0.0;
  double sum_sq_cross = 0.0;
  for (int i = 0; i < n; ++i) {
    const auto m = st.sample(kEpoch, truth, kAtRest, sky());
    // Small-angle error vector in body axes: 2·vector part of δq = q_meas ⊗ q_truth*.
    const pm::Quaternion delta = m.attitude.core() * truth.core().conjugate();
    const Eigen::Vector3d err = 2.0 * delta.vec();
    const double along = err.dot(bore);
    sum_sq_bore += along * along;
    sum_sq_cross += (err - along * bore).squaredNorm();  // both cross axes
  }
  const double sigma_bore = std::sqrt(sum_sq_bore / n);
  const double sigma_cross = std::sqrt(sum_sq_cross / (2 * n));  // per cross axis
  const auto spec = st16Spec();
  EXPECT_NEAR(sigma_cross, spec.cross_axis_sigma, spec.cross_axis_sigma * 0.05);
  EXPECT_NEAR(sigma_bore, spec.boresight_sigma, spec.boresight_sigma * 0.05);
  EXPECT_GT(sigma_bore, 4.0 * sigma_cross) << "about-boresight must be the weak axis";
}

TEST(StarTracker, MountingRotatesTheBoresightIntoBodyAxes) {
  // A tracker mounted looking along body +x, not the sensor's own +z.
  Eigen::Matrix3d mount;
  mount << 0, 0, 1, 0, 1, 0, -1, 0, 0;  // sensor +z -> body +x
  sensors::StarTracker st(st16Spec(), mount, 1, 1);
  EXPECT_TRUE(st.boresightBody().isApprox(Eigen::Vector3d::UnitX()));
}

TEST(StarTracker, IsBitReproducibleFromSeed) {
  sensors::StarTracker a(st16Spec(), Eigen::Matrix3d::Identity(), 0xABCD, 3);
  sensors::StarTracker b(st16Spec(), Eigen::Matrix3d::Identity(), 0xABCD, 3);
  const QuatBI truth = QuatBI::Identity();
  for (int i = 0; i < 50; ++i) {
    EXPECT_EQ(a.sample(kEpoch, truth, kAtRest, sky()).attitude.core().coeffs(),
              b.sample(kEpoch, truth, kAtRest, sky()).attitude.core().coeffs())
        << "sample " << i;
  }
}

// --- Validity ----------------------------------------------------------------

TEST(StarTracker, EarthInTheFieldOfViewInvalidatesTheSolution) {
  // Attitude that puts the sensor's +z boresight along ECI -x, i.e. straight
  // down at the Earth from a spacecraft on +x.
  Eigen::Matrix3d mount;
  mount << 0, 0, -1, 0, 1, 0, 1, 0, 0;  // sensor +z -> body -x
  sensors::StarTracker st(st16Spec(), mount, 1, 1);
  const auto m = st.sample(kEpoch, QuatBI::Identity(), kAtRest, sky());
  EXPECT_FALSE(m.valid);
  EXPECT_EQ(m.occluder, sensors::Occluder::kEarth);
  EXPECT_FALSE(m.rate_limited);
}

TEST(StarTracker, SunInTheKeepOutConeInvalidatesTheSolution) {
  // Sun placed at zenith so the Earth constraint is comfortably satisfied and the
  // Sun cone is the only thing that can fail — otherwise the Earth would be
  // reported first and the test would pass for the wrong reason.
  sensors::SkyGeometry s = sky();
  s.sun = s.sat.normalized() * kAu;
  sensors::StarTracker st(st16Spec(), zenithMount(), 1, 1);
  const auto m = st.sample(kEpoch, QuatBI::Identity(), kAtRest, s);
  EXPECT_FALSE(m.valid);
  EXPECT_EQ(m.occluder, sensors::Occluder::kSun);
}

TEST(StarTracker, SlewRateSmearInvalidatesTheSolution) {
  sensors::StarTracker st(st16Spec(), zenithMount(), 1, 1);
  const QuatBI truth = QuatBI::Identity();

  // Just under the 3 °/s limit: still solving.
  const auto slow =
      st.sample(kEpoch, truth, Vec3B(Eigen::Vector3d(2.0 * kDeg2Rad, 0.0, 0.0)), sky());
  EXPECT_TRUE(slow.valid);
  EXPECT_FALSE(slow.rate_limited);

  // Well beyond it: images smear and identification fails. Reported distinctly
  // from an occlusion, because the operational response is different.
  const auto fast =
      st.sample(kEpoch, truth, Vec3B(Eigen::Vector3d(0.0, 10.0 * kDeg2Rad, 0.0)), sky());
  EXPECT_FALSE(fast.valid);
  EXPECT_TRUE(fast.rate_limited);
  EXPECT_EQ(fast.occluder, sensors::Occluder::kNone);
}

TEST(StarTracker, AnOutageDoesNotDisturbTheNoiseStream) {
  // A geometric outage must not shift the stream position, or changing only the
  // orbit geometry would silently change the noise on every later sample (§3.6).
  const QuatBI truth = QuatBI::Identity();
  sensors::StarTracker clear(st16Spec(), Eigen::Matrix3d::Identity(), 0x5EED, 9);
  sensors::StarTracker blocked(st16Spec(), Eigen::Matrix3d::Identity(), 0x5EED, 9);

  // Drive one of them through an outage by dropping it out, then compare the
  // samples that follow: the noise draws must line up.
  blocked.setDropout(true);
  (void)blocked.sample(kEpoch, truth, kAtRest, sky());
  blocked.clearFaults();
  (void)clear.sample(kEpoch, truth, kAtRest, sky());

  for (int i = 0; i < 20; ++i) {
    EXPECT_EQ(clear.sample(kEpoch, truth, kAtRest, sky()).attitude.core().coeffs(),
              blocked.sample(kEpoch, truth, kAtRest, sky()).attitude.core().coeffs())
        << "sample " << i;
  }
}

// --- Fault injection ---------------------------------------------------------

TEST(StarTracker, DropoutAndAttitudeBiasFaults) {
  sensors::StarTracker st(st16Spec(), zenithMount(), 7, 1);
  const QuatBI truth = QuatBI::Identity();

  st.setDropout(true);
  EXPECT_FALSE(st.sample(kEpoch, truth, kAtRest, sky()).valid);
  st.clearFaults();
  EXPECT_TRUE(st.sample(kEpoch, truth, kAtRest, sky()).valid);

  // A 0.5° attitude bias must show up as ~0.5° of attitude error — orders of
  // magnitude above the arcsecond noise, which is what makes it detectable by
  // FDIR. Noise is switched off here so the assertion isolates the fault path.
  sensors::StarTrackerSpec quiet = st16Spec();
  quiet.cross_axis_sigma = 0.0;
  quiet.boresight_sigma = 0.0;
  sensors::StarTracker noiseless(quiet, zenithMount(), 7, 1);
  const double bias = 0.5 * kDeg2Rad;
  noiseless.injectAttitudeBias(Vec3B(Eigen::Vector3d(bias, 0.0, 0.0)));
  const auto m = noiseless.sample(kEpoch, truth, kAtRest, sky());
  EXPECT_TRUE(m.valid) << "a bias is a wrong answer, not a lost one";
  EXPECT_NEAR(attitudeError(m.attitude, truth), bias, 1.0e-12);
}
