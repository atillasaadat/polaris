/// @file Unit tests for the shared line-of-sight occlusion model (§6.1).
///
/// The geometry is pinned against hand-computable cases: at 500 km the Earth's
/// apparent radius is a known ~70°, so a nadir boresight is deep inside the disk
/// and a zenith boresight is clear by a known margin. The rest of the tests cover
/// the properties that make the model shareable — each body checked from its
/// *limb* rather than its centre, a zero keep-out meaning "unconstrained", and
/// degenerate geometry failing open rather than blinding a sensor with a NaN.

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

}  // namespace

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

TEST(Occlusion, EarthBlocksANadirBoresight) {
  sensors::KeepOutSpec keep_out;
  keep_out.earth_rad = 10.0 * kDeg2Rad;
  EXPECT_EQ(sensors::checkLineOfSight(Eigen::Vector3d(-1.0, 0.0, 0.0), sky(), keep_out),
            sensors::Occluder::kEarth);
  // Zenith is clear: ~90° from the Earth centre direction, well outside the limb.
  EXPECT_EQ(sensors::checkLineOfSight(Eigen::Vector3d(1.0, 0.0, 0.0), sky(), keep_out),
            sensors::Occluder::kNone);
}

TEST(Occlusion, SunAndMoonKeepOutConesAreEnforced) {
  sensors::KeepOutSpec keep_out;
  keep_out.sun_rad = 45.0 * kDeg2Rad;
  keep_out.moon_rad = 25.0 * kDeg2Rad;

  // Straight at the Sun (+y) and straight at the Moon (+z).
  EXPECT_EQ(sensors::checkLineOfSight(Eigen::Vector3d(0.0, 1.0, 0.0), sky(), keep_out),
            sensors::Occluder::kSun);
  EXPECT_EQ(sensors::checkLineOfSight(Eigen::Vector3d(0.0, 0.0, 1.0), sky(), keep_out),
            sensors::Occluder::kMoon);

  // 50° off the Sun clears its 45° cone; 30° does not.
  const auto off = [](double deg) {
    return Eigen::Vector3d(0.0, std::cos(deg * kDeg2Rad), std::sin(deg * kDeg2Rad));
  };
  EXPECT_EQ(sensors::checkLineOfSight(off(50.0), sky(), keep_out), sensors::Occluder::kNone);
  EXPECT_EQ(sensors::checkLineOfSight(off(30.0), sky(), keep_out), sensors::Occluder::kSun);
}

TEST(Occlusion, ZeroKeepOutDisablesThatConstraint) {
  // A sensor pays only for what it configures: with no Earth keep-out, even a
  // nadir stare is unconstrained.
  sensors::KeepOutSpec none;
  EXPECT_EQ(sensors::checkLineOfSight(Eigen::Vector3d(-1.0, 0.0, 0.0), sky(), none),
            sensors::Occluder::kNone);
}

TEST(Occlusion, EarthIsReportedInPreferenceToAnOverlappingSunCone) {
  // Both constraints violated at once: naming the Earth is the more useful
  // diagnosis, and makes the result deterministic rather than order-dependent.
  sensors::SkyGeometry s = sky();
  s.sun = -s.sat.normalized() * kAu;  // Sun directly behind the Earth from the sat
  sensors::KeepOutSpec keep_out;
  keep_out.earth_rad = 10.0 * kDeg2Rad;
  keep_out.sun_rad = 45.0 * kDeg2Rad;
  EXPECT_EQ(sensors::checkLineOfSight(-s.sat, s, keep_out), sensors::Occluder::kEarth);
}

TEST(Occlusion, DegenerateGeometryFailsOpen) {
  // A zero-length boresight or a coincident body cannot be evaluated. Returning
  // "unconstrained" keeps a bad input from silently blinding a sensor, which
  // would look like a plausible outage instead of the bug it is.
  EXPECT_TRUE(std::isinf(
      sensors::limbClearance(Eigen::Vector3d::Zero(), Eigen::Vector3d(1.0, 0.0, 0.0), kRe)));
  EXPECT_TRUE(std::isinf(
      sensors::limbClearance(Eigen::Vector3d(1.0, 0.0, 0.0), Eigen::Vector3d::Zero(), kRe)));
  // Inside the body, everything is blocked.
  EXPECT_TRUE(std::isinf(
      sensors::limbClearance(Eigen::Vector3d(1.0, 0.0, 0.0), Eigen::Vector3d(1.0, 0.0, 0.0), kRe)));
  EXPECT_LT(
      sensors::limbClearance(Eigen::Vector3d(1.0, 0.0, 0.0), Eigen::Vector3d(1.0, 0.0, 0.0), kRe),
      0.0);
}
