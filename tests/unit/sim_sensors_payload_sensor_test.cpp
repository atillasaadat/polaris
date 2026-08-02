/// @file Unit tests for the generic payload-sensor geometry model (§6.3).
///
/// Five concerns. (1) The **frame convention**: sensor +Z is the boresight and
/// the mounting is the only thing that moves it — if that ever stops holding,
/// every payload on every vehicle silently points somewhere else, so it is
/// pinned first. (2) **Field-of-view shape**: conic, square and rectangular are
/// inferred from which half-angles the catalog entry sets, and the shapes must
/// actually differ where they should — a rectangular field accepts a corner
/// direction that the equal-area circle rejects. (3) The **equivalent cone**
/// handed to the shared occlusion evaluator preserves solid angle.
/// (4) **Occlusion and the pointing angles** come from the §6.1 model, not a
/// private copy: the same Earth that blocks a star tracker blocks a payload.
/// (5) **Fault hooks**: a dropout invalidates without erasing the geometry, and
/// an injected misalignment moves the boresight and nothing else.

#include <gtest/gtest.h>

#include <cmath>
#include <Eigen/Core>
#include <map>
#include <string>

#include "constants/constants.hpp"
#include "math/frames.hpp"
#include "math/quaternion.hpp"
#include "math/typed_vector.hpp"
#include "sensors/payload_sensor.hpp"

namespace {

namespace sensors = polaris::sim::sensors;
namespace pm = polaris::math;
namespace frames = polaris::math::frames;

constexpr double kDeg = 0.017453292519943295;
constexpr double kRe = polaris::constants::wgs84::kSemiMajorAxis;
constexpr double kAu = polaris::constants::bodies::kAstronomicalUnit;

/// The catalog template's params (config/hardware/payload_sensor/generic_imager
/// .yaml). A fixture mirroring a real catalog entry pins the *conversion*; the
/// values themselves are pinned against the YAML in the config-compiler tests.
std::map<std::string, double> imagerParams() {
  return {{"half_fov_x_deg", 5.0},    {"half_fov_y_deg", 4.0}, {"pixels_x", 2048.0},
          {"pixels_y", 1536.0},       {"update_rate_hz", 1.0}, {"sun_exclusion_deg", 30.0},
          {"moon_exclusion_deg", 5.0}};
}

/// 500 km on +x, Sun far out on +y, Moon far out on +z — three perpendicular
/// directions, so one constraint can be exercised without the others.
sensors::SkyGeometry sky() {
  sensors::SkyGeometry s;
  s.sat = Eigen::Vector3d(kRe + 500e3, 0.0, 0.0);
  s.sun = Eigen::Vector3d(0.0, kAu, 0.0);
  s.moon = Eigen::Vector3d(0.0, 0.0, 3.844e8);
  return s;
}

/// An attitude putting the boresight (body +Z) at zenith, ECI +x — clear of all
/// three bodies in the fixture, so a validity check is testing the fault hook
/// and not the Moon keep-out.
pm::Quaternion zenithPointing() {
  return pm::Quaternion::FromAxisAngle(Eigen::Vector3d::UnitY(), 90.0 * kDeg);
}

sensors::PayloadSensorInput inputAt(const pm::Quaternion& attitude) {
  sensors::PayloadSensorInput in;
  in.attitude = pm::Quat<frames::Body, frames::ECI>(attitude);
  in.sky = sky();
  return in;
}

// ── (1) The frame convention ────────────────────────────────────────────────

TEST(PayloadSensor, BoresightIsSensorPlusZUnderAnyMounting) {
  const auto spec = sensors::PayloadSensorSpec::fromParams(imagerParams());

  // Identity mounting: the boresight is body +Z.
  EXPECT_TRUE(sensors::PayloadSensor(spec, Eigen::Matrix3d::Identity())
                  .boresightBody()
                  .isApprox(Eigen::Vector3d::UnitZ()));

  // A mounting rotation carries sensor +Z into body axes, and it is the *third
  // column* that does it — the same relation the config compiler's quaternion
  // form produces (tools/configc/compiler.py::_mounting_dcm).
  const Eigen::Matrix3d mounting =
      pm::Quaternion::FromAxisAngle(Eigen::Vector3d::UnitX(), 30.0 * kDeg).toRotationMatrix();
  const sensors::PayloadSensor canted(spec, mounting);
  EXPECT_TRUE(canted.boresightBody().isApprox(mounting.col(2)));
  EXPECT_NEAR(canted.boresightBody().norm(), 1.0, 1e-12);
  // 30° about X tilts the boresight off body +Z by 30°, into the Y–Z plane.
  EXPECT_NEAR(std::acos(canted.boresightBody().z()), 30.0 * kDeg, 1e-9);
  EXPECT_NEAR(canted.boresightBody().x(), 0.0, 1e-12);
}

TEST(PayloadSensor, BoresightIsRotatedIntoEciByTheTruthAttitude) {
  const auto spec = sensors::PayloadSensorSpec::fromParams(imagerParams());
  const sensors::PayloadSensor payload(spec, Eigen::Matrix3d::Identity());

  // Attitude rotating ECI +x into body +z: the boresight (body +Z) must then
  // report as ECI +x. Getting this backwards is the classic frame inversion, and
  // it would put every payload's line of sight at its mirror image.
  const pm::Quaternion attitude =
      pm::Quaternion::FromAxisAngle(Eigen::Vector3d::UnitY(), 90.0 * kDeg);
  ASSERT_TRUE(attitude.rotate(Eigen::Vector3d::UnitX()).isApprox(Eigen::Vector3d::UnitZ(), 1e-9));

  const auto sample = payload.sample(polaris::time::Tai{}, inputAt(attitude));
  EXPECT_TRUE(sample.boresight_eci.eigen().isApprox(Eigen::Vector3d::UnitX(), 1e-9));
}

// ── (2) Field-of-view shapes ────────────────────────────────────────────────

TEST(PayloadSensor, ShapeIsInferredFromWhichHalfAnglesAreSet) {
  const auto rect = sensors::PayloadSensorSpec::fromParams(imagerParams());
  EXPECT_EQ(rect.shape, sensors::FovShape::kRectangular);
  EXPECT_NEAR(rect.half_fov_x_rad, 5.0 * kDeg, 1e-12);
  EXPECT_NEAR(rect.half_fov_y_rad, 4.0 * kDeg, 1e-12);

  const auto square =
      sensors::PayloadSensorSpec::fromParams({{"half_fov_x_deg", 3.0}, {"half_fov_y_deg", 3.0}});
  EXPECT_EQ(square.shape, sensors::FovShape::kSquare);

  // A single half-angle is a cone, and the Y half-angle mirrors it so no
  // consumer reading `half_fov_y_rad` sees a zero field.
  const auto conic = sensors::PayloadSensorSpec::fromParams({{"half_fov_deg", 2.5}});
  EXPECT_EQ(conic.shape, sensors::FovShape::kConic);
  EXPECT_NEAR(conic.half_fov_x_rad, 2.5 * kDeg, 1e-12);
  EXPECT_NEAR(conic.half_fov_y_rad, 2.5 * kDeg, 1e-12);

  // Per-axis keys win when both forms are present — the documented precedence,
  // because the axis detail is the more specific statement and the alternative
  // is silently ignoring it.
  const auto both = sensors::PayloadSensorSpec::fromParams(
      {{"half_fov_deg", 20.0}, {"half_fov_x_deg", 5.0}, {"half_fov_y_deg", 4.0}});
  EXPECT_EQ(both.shape, sensors::FovShape::kRectangular);
  EXPECT_NEAR(both.half_fov_x_rad, 5.0 * kDeg, 1e-12)
      << "half_fov_deg must be dropped, not blended";

  // One per-axis key without the other is a rectangle with a zero-width axis. It
  // is *not* silently promoted to a cone: half-writing the per-axis form is a
  // configuration error, and the vehicle builder rejects the zero half-angle it
  // produces here rather than inventing the missing number.
  const auto x_only = sensors::PayloadSensorSpec::fromParams({{"half_fov_x_deg", 5.0}});
  EXPECT_EQ(x_only.shape, sensors::FovShape::kRectangular);
  EXPECT_NEAR(x_only.half_fov_x_rad, 5.0 * kDeg, 1e-12);
  EXPECT_DOUBLE_EQ(x_only.half_fov_y_rad, 0.0);

  // A laser: one pixel, and that is a valid configuration rather than a
  // defaulted-away mistake.
  const auto laser =
      sensors::PayloadSensorSpec::fromParams({{"half_fov_deg", 0.05}, {"pixels_x", 1.0}});
  EXPECT_EQ(laser.pixels_x, 1);
  EXPECT_EQ(laser.pixels_y, 1) << "an unset pixel count means 1, not 0";
}

TEST(PayloadSensor, RectangularFieldAcceptsCornersACircleOfEqualAreaRejects) {
  const auto spec = sensors::PayloadSensorSpec::fromParams(imagerParams());
  const sensors::PayloadSensor payload(spec, Eigen::Matrix3d::Identity());

  const auto at = [](double x_deg, double y_deg) {
    return Eigen::Vector3d(std::tan(x_deg * kDeg), std::tan(y_deg * kDeg), 1.0);
  };

  EXPECT_TRUE(payload.inFieldOfView(at(0.0, 0.0)));
  EXPECT_TRUE(payload.inFieldOfView(at(4.9, 3.9))) << "just inside the corner";
  EXPECT_FALSE(payload.inFieldOfView(at(5.1, 0.0))) << "outside in X, inside in Y";
  EXPECT_FALSE(payload.inFieldOfView(at(0.0, 4.1))) << "inside in X, outside in Y";

  // The corner is what makes the shape matter: it is 6.25° off the boresight,
  // beyond the 5.04° equal-solid-angle cone, and still in the field.
  const Eigen::Vector3d corner = at(4.9, 3.9).normalized();
  EXPECT_GT(std::acos(corner.z()), spec.equivalentHalfFovRad());

  // Nothing behind the sensor is ever in view, whatever the shape.
  EXPECT_FALSE(payload.inFieldOfView(Eigen::Vector3d(0.0, 0.0, -1.0)));
  EXPECT_FALSE(payload.inFieldOfView(Eigen::Vector3d::Zero()));
}

TEST(PayloadSensor, ConicFieldIsASimpleConeTest) {
  const auto spec = sensors::PayloadSensorSpec::fromParams({{"half_fov_deg", 10.0}});
  const sensors::PayloadSensor payload(spec, Eigen::Matrix3d::Identity());
  EXPECT_TRUE(
      payload.inFieldOfView(Eigen::Vector3d(std::sin(9.9 * kDeg), 0.0, std::cos(9.9 * kDeg))));
  EXPECT_FALSE(
      payload.inFieldOfView(Eigen::Vector3d(std::sin(10.1 * kDeg), 0.0, std::cos(10.1 * kDeg))));
  // Circular: a diagonal direction at the same total angle is treated the same,
  // which is exactly what a rectangular field does not do.
  const double half = 9.9 * kDeg / std::sqrt(2.0);
  EXPECT_TRUE(
      payload.inFieldOfView(Eigen::Vector3d(std::sin(half), std::sin(half), std::cos(9.9 * kDeg))));
  EXPECT_EQ(spec.equivalentHalfFovRad(), spec.half_fov_x_rad);
}

// ── (3) The equivalent cone ─────────────────────────────────────────────────

TEST(PayloadSensor, EquivalentConePreservesSolidAngle) {
  const auto spec = sensors::PayloadSensorSpec::fromParams(imagerParams());
  const double equivalent = spec.equivalentHalfFovRad();

  // Ω_rect = 4 asin(sin hx · sin hy) must equal Ω_cone = 2π(1 − cos h_e).
  const double omega_rect =
      4.0 * std::asin(std::sin(spec.half_fov_x_rad) * std::sin(spec.half_fov_y_rad));
  EXPECT_NEAR(2.0 * M_PI * (1.0 - std::cos(equivalent)), omega_rect, 1e-12);
  // And it sits between the inscribed and circumscribed circles, as any
  // equal-area equivalent must.
  EXPECT_GT(equivalent, spec.half_fov_y_rad);
  EXPECT_LT(equivalent, std::hypot(spec.half_fov_x_rad, spec.half_fov_y_rad));
}

TEST(PayloadSensor, InstantaneousFieldOfViewIsTheFullFieldPerPixel) {
  const auto spec = sensors::PayloadSensorSpec::fromParams(imagerParams());
  EXPECT_NEAR(spec.ifovXRad(), 2.0 * 5.0 * kDeg / 2048.0, 1e-15);
  EXPECT_NEAR(spec.ifovYRad(), 2.0 * 4.0 * kDeg / 1536.0, 1e-15);
}

// ── (4) Shared occlusion + the pointing angles ──────────────────────────────

TEST(PayloadSensor, EarthBlocksANadirStaringPayloadOnlyIfItAsksToBeBlocked) {
  // The template has no Earth exclusion — an Earth-observing payload stares
  // down on purpose — so a nadir boresight stays valid while still reporting a
  // fully covered field. Validity and coverage are separate answers (§6.1).
  const auto spec = sensors::PayloadSensorSpec::fromParams(imagerParams());
  const sensors::PayloadSensor payload(spec, Eigen::Matrix3d::Identity());

  // Body +Z along ECI −x = nadir, given the fixture's +x spacecraft position.
  const pm::Quaternion nadir_pointing =
      pm::Quaternion::FromAxisAngle(Eigen::Vector3d::UnitY(), -90.0 * kDeg);
  const auto sample = payload.sample(polaris::time::Tai{}, inputAt(nadir_pointing));

  ASSERT_TRUE(sample.boresight_eci.eigen().isApprox(-Eigen::Vector3d::UnitX(), 1e-9));
  EXPECT_TRUE(sample.valid);
  EXPECT_EQ(sample.occlusion.occluder, sensors::Occluder::kNone);
  EXPECT_NEAR(sample.occlusion.earth_fraction, 1.0, 1e-9);
  EXPECT_NEAR(sample.occlusion.nadir_angle_rad, 0.0, 1e-9);
  EXPECT_NEAR(sample.occlusion.sun_angle_rad, M_PI_2, 1e-4);

  // The same payload with an Earth keep-out configured *is* blocked — one model,
  // and the difference is the configuration, not the geometry.
  auto params = imagerParams();
  params["earth_exclusion_deg"] = 20.0;
  const sensors::PayloadSensor astronomy(sensors::PayloadSensorSpec::fromParams(params),
                                         Eigen::Matrix3d::Identity());
  const auto blocked = astronomy.sample(polaris::time::Tai{}, inputAt(nadir_pointing));
  EXPECT_FALSE(blocked.valid);
  EXPECT_EQ(blocked.occlusion.occluder, sensors::Occluder::kEarth);
}

TEST(PayloadSensor, SunKeepOutInvalidatesAndTheSunAngleShowsItComing) {
  const auto spec = sensors::PayloadSensorSpec::fromParams(imagerParams());
  const sensors::PayloadSensor payload(spec, Eigen::Matrix3d::Identity());

  // Body +Z along ECI +y — straight at the Sun, inside the 30° exclusion.
  const pm::Quaternion sunward =
      pm::Quaternion::FromAxisAngle(Eigen::Vector3d::UnitX(), -90.0 * kDeg);
  const auto sample = payload.sample(polaris::time::Tai{}, inputAt(sunward));
  ASSERT_TRUE(sample.boresight_eci.eigen().isApprox(Eigen::Vector3d::UnitY(), 1e-9));
  EXPECT_FALSE(sample.valid);
  EXPECT_EQ(sample.occlusion.occluder, sensors::Occluder::kSun);
  // Not machine zero: the Sun is 1 AU out on +y while the spacecraft sits
  // 6900 km off the origin on +x, which is 9.5 arcsec of real parallax.
  EXPECT_NEAR(sample.occlusion.sun_angle_rad, 0.0, 1e-4);

  // 40° off the Sun: outside the cone, so valid — but the angle still reports
  // the approach, which is the whole reason it is carried alongside the verdict.
  const pm::Quaternion off =
      pm::Quaternion::FromAxisAngle(Eigen::Vector3d::UnitX(), -(90.0 - 40.0) * kDeg);
  const auto clear = payload.sample(polaris::time::Tai{}, inputAt(off));
  EXPECT_TRUE(clear.valid);
  EXPECT_NEAR(clear.occlusion.sun_angle_rad, 40.0 * kDeg, 1e-3);
}

// ── (5) Fault hooks ─────────────────────────────────────────────────────────

TEST(PayloadSensor, DropoutInvalidatesWithoutErasingTheGeometry) {
  const auto spec = sensors::PayloadSensorSpec::fromParams(imagerParams());
  sensors::PayloadSensor payload(spec, Eigen::Matrix3d::Identity());
  const auto input = inputAt(zenithPointing());

  const auto healthy = payload.sample(polaris::time::Tai{}, input);
  ASSERT_TRUE(healthy.valid);

  payload.setDropout(true);
  const auto dead = payload.sample(polaris::time::Tai{}, input);
  EXPECT_FALSE(dead.valid);
  // Still reports where it was looking: a consumer needs to distinguish "the
  // instrument failed" from "the instrument was blinded", and the occluder is
  // how it tells.
  EXPECT_EQ(dead.occlusion.occluder, sensors::Occluder::kNone);
  EXPECT_TRUE(dead.boresight_eci.eigen().isApprox(healthy.boresight_eci.eigen()));

  payload.clearFaults();
  EXPECT_TRUE(payload.sample(polaris::time::Tai{}, input).valid);
}

TEST(PayloadSensor, InjectedMisalignmentMovesTheBoresightAndNothingElse) {
  const auto spec = sensors::PayloadSensorSpec::fromParams(imagerParams());
  sensors::PayloadSensor payload(spec, Eigen::Matrix3d::Identity());
  const auto input = inputAt(zenithPointing());

  // 0.5° about body X: the boresight tips that far, and the sample stays valid —
  // this is the silent failure, so nothing in the geometry may flag it.
  const double tilt = 0.5 * kDeg;
  payload.injectBoresightMisalignment(pm::Vec3<frames::Body>(Eigen::Vector3d(tilt, 0.0, 0.0)));
  EXPECT_NEAR(std::acos(payload.boresightBody().dot(Eigen::Vector3d::UnitZ())), tilt, 1e-6);
  EXPECT_NEAR(payload.boresightBody().norm(), 1.0, 1e-12);
  EXPECT_TRUE(payload.sample(polaris::time::Tai{}, input).valid);

  // Replaces rather than accumulates, so a scenario script cannot drift by
  // re-asserting the same fault every step.
  payload.injectBoresightMisalignment(pm::Vec3<frames::Body>(Eigen::Vector3d(tilt, 0.0, 0.0)));
  EXPECT_NEAR(std::acos(payload.boresightBody().dot(Eigen::Vector3d::UnitZ())), tilt, 1e-6);

  payload.clearFaults();
  EXPECT_TRUE(payload.boresightBody().isApprox(Eigen::Vector3d::UnitZ()));
}

}  // namespace
