/// @file Unit tests for config-driven vehicle assembly (REQ-CFG-001/002;
/// design doc §19.3, §19.4).
///
/// The property under test is that **the config is the only source of hardware
/// parameters**. There is no in-code catalog to fall back on, so these tests
/// check the whole path a real run takes: a resolved unit's datasheet-native
/// params reach the model's `fromParams` untouched, swapping a `model_id`'s
/// params swaps the flown hardware, and a unit the config asked for is never
/// silently dropped. The seeding contract is checked too — adding hardware must
/// not perturb the random stream of hardware already there (§3.5).

#include <gtest/gtest.h>

#include <cmath>
#include <string>

#include "math/frames.hpp"
#include "math/typed_vector.hpp"
#include "scenario/vehicle.hpp"
#include "time/timescales.hpp"

namespace {

namespace scenario = polaris::sim::scenario;
namespace pm = polaris::math;
namespace pmf = polaris::math::frames;
namespace pt = polaris::time;

using Vec3B = pm::Vec3<pmf::Body>;
const pt::Tai kEpoch = pt::Tai::fromNanosecondsSinceEpoch(1767225637000000000LL);

/// A resolved IMU unit as the compiler would emit it (params inlined from the
/// library entry, in the datasheet's own units).
scenario::UnitConfig imuUnit(const std::string& name, double arw_deg_sqrt_hr) {
  scenario::UnitConfig u;
  u.name = name;
  u.model_id = "TEST-IMU";
  u.kind = "imu";
  u.params = {{"gyro_range_deg_s", 400.0},
              {"gyro_arw_deg_sqrt_hr", arw_deg_sqrt_hr},
              {"sample_rate_hz", 250.0}};
  return u;
}

scenario::UnitConfig wheelUnit(const std::string& name, double momentum_nms) {
  scenario::UnitConfig u;
  u.name = name;
  u.model_id = "TEST-RW";
  u.kind = "reaction_wheel";
  u.params = {
      {"max_torque_nm", 0.1}, {"max_momentum_nms", momentum_nms}, {"max_speed_rpm", 6000.0}};
  return u;
}

scenario::UnitConfig starTrackerUnit(const std::string& name) {
  scenario::UnitConfig u;
  u.name = name;
  u.model_id = "TEST-ST";
  u.kind = "star_tracker";
  u.params = {{"temporal_noise_xy_arcsec_3sigma", 11.0},
              {"temporal_noise_z_arcsec_3sigma", 70.0},
              {"fov_deg", 15.0},
              {"earth_exclusion_deg", 22.0}};
  return u;
}

scenario::UnitConfig mtqUnit(const std::string& name) {
  scenario::UnitConfig u;
  u.name = name;
  u.model_id = "TEST-MTQ";
  u.kind = "magnetorquer";
  u.params = {{"max_dipole_am2", 15.0}, {"residual_dipole_am2", 0.5}};
  return u;
}

}  // namespace

TEST(Vehicle, BuildsEveryModelledKindFromResolvedParams) {
  scenario::SpacecraftConfig sc;
  sc.sensors = {imuUnit("imu_a", 0.15), starTrackerUnit("st_a")};
  sc.actuators = {wheelUnit("rw_1", 0.4), mtqUnit("mtq_x")};

  scenario::Vehicle v;
  std::string error;
  ASSERT_TRUE(scenario::buildVehicle(sc, 1234, v, &error)) << error;
  ASSERT_EQ(v.imus.size(), 1u);
  ASSERT_EQ(v.star_trackers.size(), 1u);
  ASSERT_EQ(v.wheels.size(), 1u);
  ASSERT_EQ(v.magnetorquers.size(), 1u);
  EXPECT_EQ(v.modelledCount(), 4u);
  EXPECT_TRUE(v.unmodelled.empty());

  // The identity survives, so telemetry and errors can name the unit.
  EXPECT_EQ(v.wheels[0].name, "rw_1");
  EXPECT_EQ(v.wheels[0].model_id, "TEST-RW");
  // And the parameters actually reached the model: a wheel with this momentum and
  // speed has this rotor inertia, so its stored momentum after a commanded torque
  // is the config's physics, not a default.
  v.wheels[0].model.commandTorque(0.01);
  v.wheels[0].model.step(1.0);
  EXPECT_NEAR(v.wheels[0].model.momentum(), 0.01, 1e-12);
}

TEST(Vehicle, ParamsChangeTheFlownHardware) {
  // The point of the whole mechanism (REQ-CFG-002): the same code flies different
  // hardware because the config said so. Two wheels differing only in the
  // library's max_momentum_nms must spin up at different rates.
  scenario::SpacecraftConfig small;
  small.actuators = {wheelUnit("rw_1", 0.1)};
  scenario::SpacecraftConfig large;
  large.actuators = {wheelUnit("rw_1", 1.0)};

  scenario::Vehicle a;
  scenario::Vehicle b;
  ASSERT_TRUE(scenario::buildVehicle(small, 1, a, nullptr));
  ASSERT_TRUE(scenario::buildVehicle(large, 1, b, nullptr));
  a.wheels[0].model.commandTorque(0.01);
  b.wheels[0].model.commandTorque(0.01);
  a.wheels[0].model.step(1.0);
  b.wheels[0].model.step(1.0);
  EXPECT_GT(a.wheels[0].model.speed(), b.wheels[0].model.speed() * 5.0)
      << "a lighter rotor must accelerate faster under the same torque";
}

TEST(Vehicle, BuildsTheWheelAssemblyFromSpinAxes) {
  // A body-diagonal 4-wheel pyramid: each wheel carries its spin axis, and the
  // builder consolidates them into W (columns in wheels order), spanning 3 axes.
  const double s = 1.0 / std::sqrt(3.0);
  const Eigen::Vector3d axes[4] = {{s, s, s}, {-s, s, s}, {-s, -s, s}, {s, -s, s}};
  scenario::SpacecraftConfig sc;
  for (int i = 0; i < 4; ++i) {
    scenario::UnitConfig u = wheelUnit("rw_" + std::to_string(i + 1), 0.4);
    u.spin_axis = axes[i];
    sc.actuators.push_back(u);
  }

  scenario::Vehicle v;
  std::string error;
  ASSERT_TRUE(scenario::buildVehicle(sc, 1, v, &error)) << error;
  ASSERT_EQ(v.wheels.size(), 4u);
  ASSERT_EQ(v.rw_assembly.size(), 4);
  EXPECT_TRUE(v.rw_assembly.spansThreeAxes());
  // Column order matches wheel order, and the axes are unit.
  EXPECT_NEAR(v.rw_assembly.matrix().col(0).x(), s, 1e-12);
  EXPECT_NEAR(v.rw_assembly.matrix().col(0).norm(), 1.0, 1e-12);
}

TEST(Vehicle, WheelAssemblyFallsBackToTheMountingColumn) {
  // With no spin_axis, the wheel's axis is the third column of its mounting DCM
  // (default identity → +z). Four such wheels are collinear and cannot span.
  scenario::SpacecraftConfig sc;
  sc.actuators = {wheelUnit("rw_1", 0.4), wheelUnit("rw_2", 0.4), wheelUnit("rw_3", 0.4)};
  scenario::Vehicle v;
  ASSERT_TRUE(scenario::buildVehicle(sc, 1, v, nullptr));
  ASSERT_EQ(v.rw_assembly.size(), 3);
  EXPECT_FALSE(v.rw_assembly.spansThreeAxes());  // all +z
  EXPECT_NEAR(v.rw_assembly.matrix().col(0).z(), 1.0, 1e-12);
}

TEST(Vehicle, UnmodelledKindsAreReportedNotDropped) {
  scenario::UnitConfig thruster;
  thruster.name = "acs_1";
  thruster.model_id = "THR-GENERIC";
  thruster.kind = "thruster";  // no truth model yet (§7 propulsion)
  thruster.params = {{"thrust_n", 0.05}};

  scenario::SpacecraftConfig sc;
  sc.sensors = {imuUnit("imu_a", 0.15)};
  sc.actuators = {thruster};

  scenario::Vehicle v;
  ASSERT_TRUE(scenario::buildVehicle(sc, 1, v, nullptr));
  EXPECT_EQ(v.imus.size(), 1u);
  ASSERT_EQ(v.unmodelled.size(), 1u);
  EXPECT_EQ(v.unmodelled[0], "acs_1:thruster");
}

TEST(Vehicle, BuildsSunSensorsAndMagnetometersFromConfig) {
  // Coarse attitude (§8.1) needs sun sensor + magnetometer + IMU. Until this
  // push the first two had no path from the config into a model at all, so a
  // vehicle could name them and fly without them.
  scenario::UnitConfig ss;
  ss.name = "ss_zp";
  ss.model_id = "CSS-GENERIC";
  ss.kind = "sun_sensor";
  ss.params = {{"diode_count", 1.0}, {"half_fov_deg", 60.0}, {"full_scale_counts", 4095.0}};

  scenario::UnitConfig mag;
  mag.name = "mag_a";
  mag.model_id = "MAG-GENERIC";
  mag.kind = "magnetometer";
  mag.params = {{"range_ut", 100.0}, {"bias_ut", 1.0}, {"noise_ut_rms", 0.05}};

  scenario::SpacecraftConfig sc;
  sc.sensors = {imuUnit("imu_a", 0.15), ss, mag};

  scenario::Vehicle v;
  std::string error;
  ASSERT_TRUE(scenario::buildVehicle(sc, 1234, v, &error)) << error;
  ASSERT_EQ(v.sun_sensors.size(), 1u);
  ASSERT_EQ(v.magnetometers.size(), 1u);
  EXPECT_TRUE(v.unmodelled.empty());
  EXPECT_EQ(v.sun_sensors[0].name, "ss_zp");
  EXPECT_EQ(v.magnetometers[0].model_id, "MAG-GENERIC");
}

TEST(Vehicle, NoiseSettingsBuildIdealSensors) {
  // With sensor noise disabled the whole suite is ideal: a measurement equals the
  // truth it was given. This is the with/without-noise switch, wired end to end
  // through buildVehicle to every model.
  scenario::UnitConfig mag;
  mag.name = "mag_a";
  mag.model_id = "MAG-GENERIC";
  mag.kind = "magnetometer";
  // Deliberately large bias/noise so a non-ideal build would be obviously off.
  mag.params = {{"range_ut", 100.0}, {"bias_ut", 5.0}, {"noise_ut_rms", 1.0}};

  scenario::SpacecraftConfig sc;
  sc.sensors = {imuUnit("imu_a", 0.5), mag};

  scenario::Vehicle v;
  std::string error;
  ASSERT_TRUE(scenario::buildVehicle(sc, 1234, v, &error, scenario::NoiseSettings{false, false}))
      << error;

  // Magnetometer: the measured field is exactly the truth field, no bias/noise.
  const Vec3B b_truth(30.0e-6, -12.0e-6, 5.0e-6);
  const auto m = v.magnetometers[0].model.sample(kEpoch, b_truth);
  EXPECT_TRUE(m.field_tesla.eigen().isApprox(b_truth.eigen(), 1e-15));

  // IMU: the measured rate is exactly the truth rate, no turn-on bias or ARW.
  const Vec3B rate_truth(0.01, -0.02, 0.015);
  const Vec3B sf_truth(0.0, 0.0, 0.0);
  const auto s = v.imus[0].model.sample(kEpoch, 0.1, rate_truth, sf_truth);
  ASSERT_TRUE(s.valid);
  EXPECT_TRUE(s.angular_rate_rads.eigen().isApprox(rate_truth.eigen(), 1e-15));
}

TEST(Vehicle, PerUnitNoiseOverrideWinsOverTheGlobal) {
  // A unit's own noise_enabled beats the scenario switch in both directions.
  auto mag = [](const std::string& name) {
    scenario::UnitConfig u;
    u.name = name;
    u.model_id = "MAG-GENERIC";
    u.kind = "magnetometer";
    u.params = {{"range_ut", 100.0}, {"bias_ut", 5.0}, {"noise_ut_rms", 0.0}};  // bias only
    return u;
  };
  const Vec3B truth(30.0e-6, 0.0, 0.0);

  // Global noise ON, but one unit forces it OFF: that one is ideal, the other has
  // its 5 µT bias.
  {
    scenario::UnitConfig forced_off = mag("mag_off");
    forced_off.noise_enabled = false;
    scenario::SpacecraftConfig sc;
    sc.sensors = {forced_off, mag("mag_on")};
    scenario::Vehicle v;
    ASSERT_TRUE(scenario::buildVehicle(sc, 1, v, nullptr));  // default NoiseSettings = on
    EXPECT_TRUE(v.magnetometers[0]
                    .model.sample(kEpoch, truth)
                    .field_tesla.eigen()
                    .isApprox(truth.eigen(), 1e-15));
    EXPECT_FALSE(v.magnetometers[1]
                     .model.sample(kEpoch, truth)
                     .field_tesla.eigen()
                     .isApprox(truth.eigen(), 1e-9))
        << "the un-overridden unit still obeys the global switch";
  }

  // Global noise OFF, but one unit forces it ON: that one is noisy, the other ideal.
  {
    scenario::UnitConfig forced_on = mag("mag_on");
    forced_on.noise_enabled = true;
    scenario::SpacecraftConfig sc;
    sc.sensors = {forced_on, mag("mag_off")};
    scenario::Vehicle v;
    ASSERT_TRUE(scenario::buildVehicle(sc, 1, v, nullptr, scenario::NoiseSettings{false, false}));
    EXPECT_FALSE(v.magnetometers[0]
                     .model.sample(kEpoch, truth)
                     .field_tesla.eigen()
                     .isApprox(truth.eigen(), 1e-9));
    EXPECT_TRUE(v.magnetometers[1]
                    .model.sample(kEpoch, truth)
                    .field_tesla.eigen()
                    .isApprox(truth.eigen(), 1e-15));
  }
}

TEST(Vehicle, RejectsASunSensorWithNoFieldOfView) {
  // Without an acceptance cone the cosine cut-off never fires and a cell reports
  // sunlight while facing away from the Sun.
  scenario::UnitConfig ss;
  ss.name = "ss_zp";
  ss.model_id = "CSS-BROKEN";
  ss.kind = "sun_sensor";
  ss.params = {{"diode_count", 1.0}, {"full_scale_counts", 4095.0}};

  scenario::SpacecraftConfig sc;
  sc.sensors = {ss};
  scenario::Vehicle v;
  std::string error;
  EXPECT_FALSE(scenario::buildVehicle(sc, 1, v, &error));
  EXPECT_NE(error.find("half_fov_deg"), std::string::npos) << error;
}

TEST(Vehicle, TheTwoSunSensorOutputContractsAreValidatedSeparately) {
  // Regression: requiring both full_scale_counts and an accuracy figure would
  // reject every real part, since no unit quotes both. A digital part reports a
  // vector and has no full scale; an analogue one has no quoted accuracy because
  // the FSW is what turns its counts into an angle.
  scenario::UnitConfig digital;
  digital.name = "ss_zp";
  digital.model_id = "GS-NANOSENSE-FSS";
  digital.kind = "sun_sensor";
  digital.params = {{"diode_count", 4.0},
                    {"half_fov_deg", 60.0},
                    {"accuracy_inner_half_angle_deg", 45.0},
                    {"accuracy_inner_deg_3sigma", 0.5},
                    {"accuracy_outer_deg_3sigma", 2.0}};

  scenario::SpacecraftConfig sc;
  sc.sensors = {digital};
  scenario::Vehicle v;
  std::string error;
  EXPECT_TRUE(scenario::buildVehicle(sc, 1, v, &error)) << error;

  // But an analogue part still owes its full scale.
  scenario::UnitConfig analogue = digital;
  analogue.params = {{"diode_count", 1.0}, {"half_fov_deg", 60.0}};
  scenario::SpacecraftConfig sc2;
  sc2.sensors = {analogue};
  scenario::Vehicle v2;
  EXPECT_FALSE(scenario::buildVehicle(sc2, 1, v2, &error));
  EXPECT_NE(error.find("full_scale_counts"), std::string::npos) << error;
}

TEST(Vehicle, StarTrackerBoresightComesFromTheMounting) {
  // The mounting DCM is what aims the tracker, so it must reach the model — a
  // keep-out check against the wrong boresight would silently validate solutions
  // taken while staring at the Earth.
  scenario::UnitConfig st = starTrackerUnit("st_a");
  st.mounting_dcm << 0, 0, 1, 0, 1, 0, -1, 0, 0;  // sensor +z -> body +x

  scenario::SpacecraftConfig sc;
  sc.sensors = {st};
  scenario::Vehicle v;
  std::string error;
  ASSERT_TRUE(scenario::buildVehicle(sc, 1, v, &error)) << error;
  EXPECT_TRUE(v.star_trackers[0].model.boresightBody().isApprox(Eigen::Vector3d::UnitX()));
}

TEST(Vehicle, RejectsAStarTrackerWithNoEarthConstraint) {
  // With neither a FOV nor an exclusion angle the Earth keep-out collapses to
  // zero and the tracker solves happily while pointed at the ground.
  scenario::UnitConfig st = starTrackerUnit("st_a");
  st.params.erase("fov_deg");
  st.params.erase("earth_exclusion_deg");

  scenario::SpacecraftConfig sc;
  sc.sensors = {st};
  scenario::Vehicle v;
  std::string error;
  EXPECT_FALSE(scenario::buildVehicle(sc, 1, v, &error));
  EXPECT_NE(error.find("earth_exclusion_deg"), std::string::npos) << error;
}

TEST(Vehicle, AStarTrackerNeedsOnlyOneOfFovOrExclusionAngle) {
  // Vendors quote an exclusion angle (Sodern: 22°) rather than deriving one from
  // the field of view, so either key on its own is a complete configuration.
  scenario::UnitConfig exclusion_only = starTrackerUnit("st_a");
  exclusion_only.params.erase("fov_deg");
  scenario::SpacecraftConfig sc;
  sc.sensors = {exclusion_only};
  scenario::Vehicle v;
  std::string error;
  EXPECT_TRUE(scenario::buildVehicle(sc, 1, v, &error)) << error;
}

TEST(Vehicle, RejectsDuplicateUnitNames) {
  // Two units sharing a name share a noise stream, which would make nominally
  // independent sensors perfectly correlated.
  scenario::SpacecraftConfig sc;
  sc.sensors = {imuUnit("imu_a", 0.15), imuUnit("imu_a", 0.15)};

  scenario::Vehicle v;
  std::string error;
  EXPECT_FALSE(scenario::buildVehicle(sc, 1, v, &error));
  EXPECT_NE(error.find("duplicate"), std::string::npos) << error;
}

TEST(Vehicle, RejectsAUnitThatResolvedToNoParameters) {
  // An empty param map builds an ideal, unlimited device — a resolution failure
  // that must not pass as a working unit.
  scenario::UnitConfig bare;
  bare.name = "rw_1";
  bare.model_id = "RW-BROKEN";
  bare.kind = "reaction_wheel";

  scenario::SpacecraftConfig sc;
  sc.actuators = {bare};
  scenario::Vehicle v;
  std::string error;
  EXPECT_FALSE(scenario::buildVehicle(sc, 1, v, &error));
  EXPECT_NE(error.find("no parameters"), std::string::npos) << error;
}

TEST(Vehicle, RejectsAWheelWithNoRotorInertia) {
  scenario::UnitConfig u;
  u.name = "rw_1";
  u.model_id = "RW-BROKEN";
  u.kind = "reaction_wheel";
  u.params = {{"max_torque_nm", 0.1}};  // no momentum/speed and no inertia

  scenario::SpacecraftConfig sc;
  sc.actuators = {u};
  scenario::Vehicle v;
  std::string error;
  EXPECT_FALSE(scenario::buildVehicle(sc, 1, v, &error));
  EXPECT_NE(error.find("rotor inertia"), std::string::npos) << error;
}

TEST(Vehicle, AddingHardwareDoesNotPerturbExistingNoiseStreams) {
  // §3.5: streams are keyed by unit name, not list position, so installing a
  // second IMU ahead of the first must leave the first's samples bit-identical.
  // Were this to fail, every Monte Carlo baseline would silently invalidate on a
  // config edit that touched unrelated hardware.
  scenario::SpacecraftConfig one;
  one.sensors = {imuUnit("imu_a", 0.15)};
  scenario::SpacecraftConfig two;
  two.sensors = {imuUnit("imu_b", 0.15), imuUnit("imu_a", 0.15)};

  scenario::Vehicle v1;
  scenario::Vehicle v2;
  ASSERT_TRUE(scenario::buildVehicle(one, 0xC0FFEE, v1, nullptr));
  ASSERT_TRUE(scenario::buildVehicle(two, 0xC0FFEE, v2, nullptr));
  ASSERT_EQ(v2.imus[1].name, "imu_a");

  const Eigen::Vector3d rate(0.01, 0.0, 0.0);
  const Eigen::Vector3d sf(0.0, 0.0, -9.80665);
  for (int i = 0; i < 50; ++i) {
    EXPECT_EQ(
        v1.imus[0].model.sample(kEpoch, 0.01, Vec3B(rate), Vec3B(sf)).angular_rate_rads.eigen(),
        v2.imus[1].model.sample(kEpoch, 0.01, Vec3B(rate), Vec3B(sf)).angular_rate_rads.eigen())
        << "sample " << i;
  }
}

TEST(Vehicle, DifferentUnitsGetIndependentStreams) {
  scenario::SpacecraftConfig sc;
  sc.sensors = {imuUnit("imu_a", 0.15), imuUnit("imu_b", 0.15)};
  scenario::Vehicle v;
  ASSERT_TRUE(scenario::buildVehicle(sc, 42, v, nullptr));

  const Eigen::Vector3d zero = Eigen::Vector3d::Zero();
  const auto a = v.imus[0].model.sample(kEpoch, 0.01, Vec3B(zero), Vec3B(zero));
  const auto b = v.imus[1].model.sample(kEpoch, 0.01, Vec3B(zero), Vec3B(zero));
  EXPECT_NE(a.angular_rate_rads.eigen(), b.angular_rate_rads.eigen());
}
