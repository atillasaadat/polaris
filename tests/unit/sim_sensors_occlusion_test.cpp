/// @file Unit tests for the shared line-of-sight occlusion model (§6.1).
///
/// Four concerns. (1) The limb geometry, pinned against hand-computable cases: at
/// 500 km the Earth's apparent radius is a known ~68°, so a nadir boresight is
/// deep inside the disk and a zenith boresight is clear by a known margin.
/// (2) The **fraction of field of view** each body covers, checked against a
/// brute-force spherical quadrature — that comparison is what bounds the planar
/// lens approximation, so the documented error budget is a measured claim rather
/// than an assumption. (3) The **atmosphere**: the optical limb sits above the
/// solid one, so the atmospheric fraction leads the solid fraction, and a
/// grazing line of sight is blocked by air before it ever touches ground.
/// (4) Robustness: zero keep-out means unconstrained, and degenerate geometry
/// fails open rather than blinding a sensor with a NaN.

#include <gtest/gtest.h>

#include <cmath>
#include <Eigen/Core>

#include "constants/constants.hpp"
#include "sensors/occlusion.hpp"

namespace {

namespace sensors = polaris::sim::sensors;

constexpr double kDeg2Rad = 0.017453292519943295;
constexpr double kRe = polaris::constants::wgs84::kSemiMajorAxis;
constexpr double kAu = polaris::constants::bodies::kAstronomicalUnit;

/// A 500 km circular orbit position on the +x axis, with the Sun far out on +y
/// and the Moon far out on +z — three mutually perpendicular directions, so each
/// keep-out can be exercised without the others interfering.
sensors::SkyGeometry sky() {
  sensors::SkyGeometry s;
  s.sat = Eigen::Vector3d(kRe + 500e3, 0.0, 0.0);
  s.sun = Eigen::Vector3d(0.0, kAu, 0.0);
  s.moon = Eigen::Vector3d(0.0, 0.0, 3.844e8);
  return s;
}

/// Brute-force fraction of a circular FOV covered by a disk, by quadrature over
/// the FOV cap on the unit sphere.
///
/// This is the independent check on `fovCoveredFraction`: it makes no
/// small-angle assumption, integrating solid angle directly. Equal-solid-angle
/// rings times uniform azimuth, so every sample carries the same weight and the
/// result is a plain hit ratio.
double quadratureFraction(double half_fov, double separation, double body_radius) {
  constexpr int kRings = 400;
  constexpr int kSpokes = 400;
  const double cos_fov = std::cos(half_fov);
  int inside = 0;
  for (int i = 0; i < kRings; ++i) {
    // Uniform in cosθ over [cos(half_fov), 1] -> equal solid angle per ring.
    const double cos_theta = 1.0 - (1.0 - cos_fov) * (i + 0.5) / kRings;
    const double sin_theta = std::sqrt(std::max(0.0, 1.0 - cos_theta * cos_theta));
    for (int j = 0; j < kSpokes; ++j) {
      const double phi = 2.0 * M_PI * (j + 0.5) / kSpokes;
      // Boresight along +z; the body centre lies in the xz-plane at `separation`.
      const Eigen::Vector3d p(sin_theta * std::cos(phi), sin_theta * std::sin(phi), cos_theta);
      const Eigen::Vector3d body(std::sin(separation), 0.0, std::cos(separation));
      if (std::acos(std::clamp(p.dot(body), -1.0, 1.0)) <= body_radius) {
        ++inside;
      }
    }
  }
  return static_cast<double>(inside) / (kRings * kSpokes);
}

}  // namespace

// --- Limb geometry -----------------------------------------------------------

TEST(Occlusion, EarthApparentRadiusMatchesTheAnalyticValueAtLeo) {
  const double r = kRe + 500e3;
  // Boresight straight down (nadir): the clearance is -asin(Re/r), i.e. the
  // boresight sits one full apparent radius inside the disk.
  const double expected = std::asin(kRe / r);
  const double clearance =
      sensors::limbClearance(Eigen::Vector3d(-1.0, 0.0, 0.0), Eigen::Vector3d(-r, 0.0, 0.0), kRe);
  EXPECT_NEAR(clearance, -expected, 1e-12);
  EXPECT_NEAR(expected, 67.6 * kDeg2Rad, 1.0 * kDeg2Rad) << "~68 deg at 500 km";
}

TEST(Occlusion, ClearanceIsMeasuredFromTheLimbNotTheCentre) {
  // A boresight 75° off nadir at 500 km clears the ~68° limb by ~7°, even though
  // it is nowhere near 90° from the Earth's centre. Measuring from the centre
  // would report 75° of clearance and let a tracker stare through the atmosphere.
  const double r = kRe + 500e3;
  const Eigen::Vector3d to_earth(-r, 0.0, 0.0);
  const double off_nadir = 75.0 * kDeg2Rad;
  const Eigen::Vector3d boresight(-std::cos(off_nadir), std::sin(off_nadir), 0.0);
  const double clearance = sensors::limbClearance(boresight, to_earth, kRe);
  EXPECT_NEAR(clearance, off_nadir - std::asin(kRe / r), 1e-12);
  EXPECT_GT(clearance, 0.0);
  EXPECT_LT(clearance, 10.0 * kDeg2Rad);
}

// --- Field-of-view coverage --------------------------------------------------

TEST(Occlusion, CoveredFractionHandlesTheContainmentCases) {
  const double fov = 10.0 * kDeg2Rad;
  // Disjoint.
  EXPECT_DOUBLE_EQ(sensors::fovCoveredFraction(fov, 40.0 * kDeg2Rad, 5.0 * kDeg2Rad), 0.0);
  // FOV entirely inside a much larger body: fully blocked.
  EXPECT_DOUBLE_EQ(sensors::fovCoveredFraction(fov, 5.0 * kDeg2Rad, 60.0 * kDeg2Rad), 1.0);
  // Small body entirely inside the FOV: the ratio of areas.
  EXPECT_NEAR(sensors::fovCoveredFraction(fov, 2.0 * kDeg2Rad, 5.0 * kDeg2Rad), 0.25, 1e-12);
  // Concentric and equal: exactly filled.
  EXPECT_NEAR(sensors::fovCoveredFraction(fov, 0.0, fov), 1.0, 1e-12);
  // Half-covered: a body edge exactly on the boresight with a huge body radius
  // cuts the FOV in half.
  EXPECT_NEAR(sensors::fovCoveredFraction(fov, 60.0 * kDeg2Rad, 60.0 * kDeg2Rad), 0.5, 0.02);
}

TEST(Occlusion, CoveredFractionAgreesWithSphericalQuadrature) {
  // The planar lens formula is an approximation on a sphere, and this is what
  // turns the header's error claim into a measured one. The tolerances are
  // per-case rather than global so the claim is pinned *by regime*: the error
  // grows with the field of view, and a single loose bound would hide that a
  // small-FOV case had quietly degraded.
  struct Case {
    double half_fov_deg;
    double separation_deg;
    double body_deg;
    double tolerance;
  };

  const Case cases[] = {
      {7.5, 60.0, 67.6, 0.008},   // ST-16 FOV grazing the Earth limb
      {7.5, 67.6, 67.6, 0.008},   // boresight exactly on the limb -> about half
      {7.5, 72.0, 67.6, 0.008},   // just clear
      {10.0, 8.0, 5.0, 0.008},    // small body partly in a small FOV
      {10.0, 12.0, 5.0, 0.008},   // small body on the FOV edge
      {15.0, 60.0, 67.6, 0.010},  // wider FOV, deeper overlap
      {30.0, 80.0, 67.6, 0.020},  // widest FOV the header's claim covers
  };
  for (const Case& c : cases) {
    const double half_fov = c.half_fov_deg * kDeg2Rad;
    const double sep = c.separation_deg * kDeg2Rad;
    const double body = c.body_deg * kDeg2Rad;
    EXPECT_NEAR(sensors::fovCoveredFraction(half_fov, sep, body),
                quadratureFraction(half_fov, sep, body), c.tolerance)
        << "fov=" << c.half_fov_deg << " sep=" << c.separation_deg << " body=" << c.body_deg;
  }
}

TEST(Occlusion, EarthFractionSweepsSmoothlyFromClearToFullyBlocked) {
  // The point of reporting a fraction rather than a bool: the transition is
  // continuous and monotone, so a consumer sees an outage approaching.
  const double half_fov = 7.5 * kDeg2Rad;
  const double r = kRe + 500e3;
  double previous = -1.0;
  for (int off_nadir_deg = 180; off_nadir_deg >= 0; off_nadir_deg -= 5) {
    const double a = off_nadir_deg * kDeg2Rad;
    sensors::SkyGeometry s = sky();
    // Boresight swung from zenith (180° off nadir) down to nadir.
    const Eigen::Vector3d boresight(-std::cos(a), std::sin(a), 0.0);
    const auto state = sensors::evaluateLineOfSight(boresight, half_fov, s, {});
    EXPECT_GE(state.earth_fraction, previous - 1e-12) << "at " << off_nadir_deg << " deg off nadir";
    previous = state.earth_fraction;
  }
  EXPECT_DOUBLE_EQ(previous, 1.0) << "nadir stare is fully blocked";
  (void)r;
}

// --- Atmosphere --------------------------------------------------------------

TEST(Occlusion, AtmosphereFractionLeadsTheSolidEarthFraction) {
  // The optical limb sits above the hard one, so as the boresight sweeps toward
  // the Earth the atmosphere is covered first — and at the grazing angle where
  // the solid disk is still clear, the airglow layer already is not.
  const double half_fov = 7.5 * kDeg2Rad;
  const double r = kRe + 500e3;
  const double solid_limb = std::asin(kRe / r);
  const double air_limb = std::asin((kRe + 100e3) / r);
  EXPECT_GT(air_limb, solid_limb);

  // Aim so the FOV straddles the atmospheric limb but not the solid one.
  const double off_nadir = 0.5 * (solid_limb + air_limb) + half_fov;
  const Eigen::Vector3d boresight(-std::cos(off_nadir), std::sin(off_nadir), 0.0);
  const auto state = sensors::evaluateLineOfSight(boresight, half_fov, sky(), {});
  EXPECT_GT(state.earth_atmosphere_fraction, state.earth_fraction);
  EXPECT_DOUBLE_EQ(state.earth_fraction, 0.0) << "solid Earth still clear of the FOV";
  EXPECT_GT(state.earth_atmosphere_fraction, 0.0) << "but the air is not";
}

TEST(Occlusion, AtmosphereHeightIsConfigurable) {
  // A horizon sensor in the CO2 band sees a limb tens of km higher than the
  // visible one, so the thickness is a scenario parameter, not a constant.
  const double half_fov = 5.0 * kDeg2Rad;
  const double r = kRe + 500e3;
  const double off_nadir = std::asin((kRe + 150e3) / r);  // between the two limbs below
  const Eigen::Vector3d boresight(-std::cos(off_nadir), std::sin(off_nadir), 0.0);

  sensors::SkyGeometry thin = sky();
  thin.atmosphere_height_m = 0.0;  // no atmosphere: solid disk only
  sensors::SkyGeometry thick = sky();
  thick.atmosphere_height_m = 300e3;

  const auto a = sensors::evaluateLineOfSight(boresight, half_fov, thin, {});
  const auto b = sensors::evaluateLineOfSight(boresight, half_fov, thick, {});
  EXPECT_DOUBLE_EQ(a.earth_atmosphere_fraction, a.earth_fraction)
      << "zero thickness collapses onto the solid disk";
  EXPECT_GT(b.earth_atmosphere_fraction, a.earth_atmosphere_fraction);
  // And the solid-Earth fraction is untouched by the atmosphere setting.
  EXPECT_DOUBLE_EQ(a.earth_fraction, b.earth_fraction);
}

TEST(Occlusion, KeepOutIsJudgedAgainstTheAtmosphericLimb) {
  // Conservative by construction: a boresight clear of the solid limb but inside
  // the airglow layer is a violation.
  const double r = kRe + 500e3;
  const double solid_limb = std::asin(kRe / r);
  const double air_limb = std::asin((kRe + 100e3) / r);
  const double off_nadir = 0.5 * (solid_limb + air_limb);
  const Eigen::Vector3d boresight(-std::cos(off_nadir), std::sin(off_nadir), 0.0);

  sensors::KeepOutSpec keep_out;
  keep_out.earth_rad = 1e-6;  // essentially "must clear the limb"
  EXPECT_EQ(sensors::evaluateLineOfSight(boresight, 0.0, sky(), keep_out).occluder,
            sensors::Occluder::kEarth);
}

// --- Keep-out verdicts -------------------------------------------------------

TEST(Occlusion, EarthBlocksANadirBoresight) {
  sensors::KeepOutSpec keep_out;
  keep_out.earth_rad = 10.0 * kDeg2Rad;
  EXPECT_EQ(
      sensors::evaluateLineOfSight(Eigen::Vector3d(-1.0, 0.0, 0.0), 0.0, sky(), keep_out).occluder,
      sensors::Occluder::kEarth);
  // Zenith is clear: ~90° from the Earth centre direction, well outside the limb.
  EXPECT_EQ(
      sensors::evaluateLineOfSight(Eigen::Vector3d(1.0, 0.0, 0.0), 0.0, sky(), keep_out).occluder,
      sensors::Occluder::kNone);
}

TEST(Occlusion, SunAndMoonKeepOutConesAreEnforced) {
  sensors::KeepOutSpec keep_out;
  keep_out.sun_rad = 45.0 * kDeg2Rad;
  keep_out.moon_rad = 25.0 * kDeg2Rad;

  // Straight at the Sun (+y) and straight at the Moon (+z).
  EXPECT_EQ(
      sensors::evaluateLineOfSight(Eigen::Vector3d(0.0, 1.0, 0.0), 0.0, sky(), keep_out).occluder,
      sensors::Occluder::kSun);
  EXPECT_EQ(
      sensors::evaluateLineOfSight(Eigen::Vector3d(0.0, 0.0, 1.0), 0.0, sky(), keep_out).occluder,
      sensors::Occluder::kMoon);

  // 50° off the Sun clears its 45° cone; 30° does not.
  const auto off = [](double deg) {
    return Eigen::Vector3d(0.0, std::cos(deg * kDeg2Rad), std::sin(deg * kDeg2Rad));
  };
  EXPECT_EQ(sensors::evaluateLineOfSight(off(50.0), 0.0, sky(), keep_out).occluder,
            sensors::Occluder::kNone);
  EXPECT_EQ(sensors::evaluateLineOfSight(off(30.0), 0.0, sky(), keep_out).occluder,
            sensors::Occluder::kSun);
}

TEST(Occlusion, ZeroKeepOutDisablesThatConstraint) {
  // A sensor pays only for what it configures: with no Earth keep-out, even a
  // nadir stare passes the verdict — though the fraction still reports the truth.
  const auto state =
      sensors::evaluateLineOfSight(Eigen::Vector3d(-1.0, 0.0, 0.0), 5.0 * kDeg2Rad, sky(), {});
  EXPECT_EQ(state.occluder, sensors::Occluder::kNone);
  EXPECT_DOUBLE_EQ(state.earth_fraction, 1.0);
}

TEST(Occlusion, EarthIsReportedInPreferenceToAnOverlappingSunCone) {
  // Both constraints violated at once: naming the Earth is the more useful
  // diagnosis, and makes the result deterministic rather than order-dependent.
  sensors::SkyGeometry s = sky();
  s.sun = -s.sat.normalized() * kAu;  // Sun directly behind the Earth from the sat
  sensors::KeepOutSpec keep_out;
  keep_out.earth_rad = 10.0 * kDeg2Rad;
  keep_out.sun_rad = 45.0 * kDeg2Rad;
  EXPECT_EQ(sensors::evaluateLineOfSight(-s.sat, 0.0, s, keep_out).occluder,
            sensors::Occluder::kEarth);
}

TEST(Occlusion, BlockedFractionTakesTheLargestContributor) {
  sensors::OcclusionState state;
  state.earth_fraction = 0.2;
  state.earth_atmosphere_fraction = 0.3;
  state.sun_fraction = 0.1;
  EXPECT_DOUBLE_EQ(state.blockedFraction(), 0.3);
}

// --- Robustness --------------------------------------------------------------

TEST(Occlusion, DegenerateGeometryFailsOpen) {
  // A zero-length boresight or a coincident body cannot be evaluated. Returning
  // "unconstrained" keeps a bad input from silently blinding a sensor, which
  // would look like a plausible outage instead of the bug it is.
  EXPECT_TRUE(std::isinf(
      sensors::limbClearance(Eigen::Vector3d::Zero(), Eigen::Vector3d(1.0, 0.0, 0.0), kRe)));
  EXPECT_TRUE(std::isinf(
      sensors::limbClearance(Eigen::Vector3d(1.0, 0.0, 0.0), Eigen::Vector3d::Zero(), kRe)));
  // Inside the body, everything is blocked.
  EXPECT_LT(
      sensors::limbClearance(Eigen::Vector3d(1.0, 0.0, 0.0), Eigen::Vector3d(1.0, 0.0, 0.0), kRe),
      0.0);
}

TEST(Occlusion, ZeroFieldOfViewReportsNoFractionRatherThanNaN) {
  // A spec with no FOV must not put a NaN in a telemetry channel by dividing by
  // a zero area. The keep-out verdict still works.
  const auto state = sensors::evaluateLineOfSight(Eigen::Vector3d(-1.0, 0.0, 0.0), 0.0, sky(), {});
  EXPECT_DOUBLE_EQ(state.earth_fraction, 0.0);
  EXPECT_DOUBLE_EQ(state.earth_atmosphere_fraction, 0.0);
  EXPECT_DOUBLE_EQ(state.blockedFraction(), 0.0);
}
