/// @file Unit tests for the §2.4 closed loop (plant → sensors → FSW → actuators).
///
/// The contracts that make the execution model trustworthy, each pinned:
/// commands apply on the *next* macro-step (causality); the total angular
/// momentum of body + wheels is conserved when the only torques are internal
/// (the actuator-feedback sign convention, end to end through the plant); the
/// IMU buffer folds in every native-rate sample with no loss; discrete sensors
/// publish latest-valid on their own grid; the GNSS fault schedule applies at
/// sample time; and the whole loop is bit-reproducible run to run.

#include <gtest/gtest.h>

#include <cmath>
#include <Eigen/Core>
#include <string>
#include <vector>

#include "io/closed_loop.hpp"
#include "scenario/sim_config.hpp"
#include "scenario/sim_runner.hpp"
#include "scenario/vehicle.hpp"

namespace {

namespace io = polaris::sim::io;
namespace scenario = polaris::sim::scenario;
namespace pm = polaris::math;
namespace pt = polaris::time;

/// Free space at a LEO-like radius: no gravity, no environment models — the
/// analytic baseline where actuator feedback is the only physics.
scenario::SimConfig freeSpace(double duration_s, double fsw_rate_hz = 10.0) {
  scenario::SimConfig c;
  c.scenario_name = "closed-loop-test";
  c.spacecraft.name = "test-vehicle";
  c.spacecraft.mass_kg = 12.0;
  c.spacecraft.inertia_kgm2 = Eigen::Vector3d(0.12, 0.12, 0.10).asDiagonal();
  c.initial_state.epoch = pt::Tai::fromNanosecondsSinceEpoch(1767225637000000000LL);
  c.initial_state.position = pm::Vec3<pm::frames::ECI>(Eigen::Vector3d(6.878137e6, 0.0, 0.0));
  c.initial_state.velocity = pm::Vec3<pm::frames::ECI>(Eigen::Vector3d(0.0, 7612.0, 0.0));
  c.initial_state.attitude = pm::Quat<pm::frames::Body, pm::frames::ECI>::Identity();
  c.initial_state.body_rate = pm::Vec3<pm::frames::Body>(Eigen::Vector3d::Zero());
  c.environment.gravity_degree = -1;
  c.environment.magnetic_field = scenario::MagneticModel::kNone;
  c.environment.drag_enabled = false;
  c.environment.srp_enabled = false;
  // §5.3 disturbance torques off: this is the analytic no-environment baseline, so
  // actuator feedback must be the only torque on the body. Stating it explicitly
  // rather than relying on the free-drift/spherical-inertia geometry that makes
  // the gravity gradient happen to vanish here.
  c.environment.gravity_gradient_torque_enabled = false;
  c.environment.aero_torque_enabled = false;
  c.environment.srp_torque_enabled = false;
  c.environment.residual_dipole_torque_enabled = false;
  c.propagation.duration_s = duration_s;
  c.propagation.output_step_s = duration_s;
  c.propagation.fsw_rate_hz = fsw_rate_hz;
  return c;
}

scenario::UnitConfig wheelUnit(const std::string& name, const Eigen::Vector3d& axis) {
  scenario::UnitConfig u;
  u.name = name;
  u.model_id = "TEST-RW";
  u.kind = "reaction_wheel";
  u.params = {{"max_torque_nm", 0.1}, {"rotor_inertia_kg_m2", 1.0e-3}, {"max_speed_rpm", 60000.0}};
  u.spin_axis = axis;
  return u;
}

scenario::UnitConfig imuUnit(const std::string& name, double rate_hz) {
  scenario::UnitConfig u;
  u.name = name;
  u.model_id = "TEST-IMU";
  u.kind = "imu";
  u.params = {{"gyro_range_deg_s", 400.0}, {"sample_rate_hz", rate_hz}};
  return u;
}

scenario::UnitConfig gnssUnit(const std::string& name, double rate_hz) {
  scenario::UnitConfig u;
  u.name = name;
  u.model_id = "TEST-GNSS";
  u.kind = "gnss";
  u.params = {{"horizontal_position_rms_m", 1.2},
              {"velocity_accuracy_m_s_rms", 0.03},
              {"max_rate_hz", rate_hz}};
  return u;
}

scenario::DataPaths goldenPaths() {
  return scenario::DataPaths::under(POLARIS_GOLDEN_DIR);
}

}  // namespace

namespace {
/// Vehicle suite builder — keeps the tests readable.
scenario::SpacecraftConfig suite(std::vector<scenario::UnitConfig> sensors,
                                 std::vector<scenario::UnitConfig> actuators) {
  scenario::SpacecraftConfig sc;
  sc.sensors = std::move(sensors);
  sc.actuators = std::move(actuators);
  return sc;
}
}  // namespace

TEST(ClosedLoop, WheelTorqueConservesTotalAngularMomentum) {
  // The actuator-feedback sign convention, end to end: an internal wheel torque
  // spins the rotor one way and the body the other, and the TOTAL angular
  // momentum H = I_body·ω + a·h_wheel stays put (free space, no external
  // torque). A sign error anywhere in wheel → W → wrench → plant breaks this.
  scenario::SimConfig config = freeSpace(2.0);
  scenario::Vehicle vehicle;
  std::string error;
  ASSERT_TRUE(
      scenario::buildVehicle(suite({}, {wheelUnit("rw_1", {0.0, 0.0, 1.0})}), 1, vehicle, &error))
      << error;

  scenario::SimRunner runner;
  io::ClosedLoop loop(runner, vehicle);
  ASSERT_TRUE(runner.build(config, goldenPaths(), &error, loop.wrench())) << error;

  const double torque_cmd = 0.01;
  auto fsw = [&](const io::FswInputs&) {
    io::FswOutputs out;
    out.wheels.push_back({io::WheelCommand::Mode::kTorque, torque_cmd});
    return out;
  };

  std::vector<io::MacroSample> trace;
  ASSERT_TRUE(loop.run(fsw, &trace, &error)) << error;
  ASSERT_GE(trace.size(), 3u);

  const Eigen::Matrix3d inertia = config.spacecraft.inertia_kgm2;
  const Eigen::Vector3d axis(0.0, 0.0, 1.0);

  // Causality: the first macro interval ran under the initial zero command.
  EXPECT_NEAR(trace[1].state.body_rate.eigen().norm(), 0.0, 1e-12)
      << "commands must apply on the NEXT step, not the one they were issued in";
  // After that the wheel spins up and the body counter-rotates.
  EXPECT_LT(trace.back().state.body_rate.eigen().z(), -1e-3);
  EXPECT_GT(vehicle.wheels[0].model.speed(), 1.0);

  // Total angular momentum about the spin axis stays zero (started at rest).
  const Eigen::Vector3d h_body = inertia * trace.back().state.body_rate.eigen();
  const double h_total = h_body.z() + vehicle.wheels[0].model.momentum();
  EXPECT_NEAR(h_total, 0.0, 1e-9);
}

TEST(ClosedLoop, ImuBufferFoldsInEveryNativeSample) {
  // §2.4: a 10 Hz consumer receives everything a 250 Hz instrument measured.
  // With noise off and a constant tumble, each macro read must carry exactly
  // rate/fsw_rate samples and the delta-angle sum ω·Δt.
  scenario::SimConfig config = freeSpace(0.5);
  // Spherical inertia: torque-free ω×(Iω) vanishes, so the body rate really is
  // constant and the expected delta-angle below is exact. (With the asymmetric
  // default inertia the Euler coupling precesses ω and the sum would drift.)
  config.spacecraft.inertia_kgm2 = Eigen::Matrix3d::Identity() * 0.1;
  config.initial_state.body_rate = pm::Vec3<pm::frames::Body>(Eigen::Vector3d(0.02, -0.01, 0.03));

  scenario::Vehicle vehicle;
  std::string error;
  ASSERT_TRUE(scenario::buildVehicle(suite({imuUnit("imu_a", 250.0)}, {}), 1, vehicle, &error,
                                     scenario::NoiseSettings{false, false}))
      << error;

  scenario::SimRunner runner;
  io::ClosedLoop loop(runner, vehicle);
  ASSERT_TRUE(runner.build(config, goldenPaths(), &error, loop.wrench())) << error;

  std::vector<int> samples_per_read;
  std::vector<Eigen::Vector3d> delta_angles;
  auto fsw = [&](const io::FswInputs& in) {
    samples_per_read.push_back(in.imus[0].samples);
    delta_angles.push_back(in.imus[0].delta_angle_rad.eigen());
    return io::FswOutputs{};
  };

  ASSERT_TRUE(loop.run(fsw, nullptr, &error)) << error;
  ASSERT_EQ(samples_per_read.size(), 5u);  // 0.5 s at 10 Hz
  const Eigen::Vector3d expected = config.initial_state.body_rate.eigen() * 0.1;
  for (std::size_t i = 0; i < samples_per_read.size(); ++i) {
    EXPECT_EQ(samples_per_read[i], 25) << "macro " << i;  // 250 Hz / 10 Hz
    EXPECT_TRUE(delta_angles[i].isApprox(expected, 1e-9)) << "macro " << i;
  }
}

TEST(ClosedLoop, IsBitReproducibleAcrossRuns) {
  // The whole point of §3.5 plumbed through §2.4: identical {config, seed} ⇒
  // identical trajectories AND identical noisy measurements, run to run.
  auto once = [](std::vector<io::MacroSample>& trace, std::vector<Eigen::Vector3d>& gnss_pos) {
    scenario::SimConfig config = freeSpace(0.5);
    scenario::Vehicle vehicle;
    std::string error;
    ASSERT_TRUE(scenario::buildVehicle(
        suite({imuUnit("imu_a", 100.0), gnssUnit("gps_a", 10.0)}, {}), 42, vehicle, &error))
        << error;
    scenario::SimRunner runner;
    io::ClosedLoop loop(runner, vehicle);
    loop.setDataPaths(scenario::DataPaths::under(POLARIS_GOLDEN_DIR));
    ASSERT_TRUE(
        runner.build(config, scenario::DataPaths::under(POLARIS_GOLDEN_DIR), &error, loop.wrench()))
        << error;
    auto fsw = [&](const io::FswInputs& in) {
      if (in.gnss[0].ever_sampled) {
        gnss_pos.push_back(in.gnss[0].measurement.position_m.eigen());
      }
      return io::FswOutputs{};
    };
    ASSERT_TRUE(loop.run(fsw, &trace, &error)) << error;
  };

  std::vector<io::MacroSample> trace_a;
  std::vector<io::MacroSample> trace_b;
  std::vector<Eigen::Vector3d> gnss_a;
  std::vector<Eigen::Vector3d> gnss_b;
  once(trace_a, gnss_a);
  once(trace_b, gnss_b);

  ASSERT_EQ(trace_a.size(), trace_b.size());
  for (std::size_t i = 0; i < trace_a.size(); ++i) {
    EXPECT_EQ(trace_a[i].state.position.eigen(), trace_b[i].state.position.eigen()) << i;
    EXPECT_EQ(trace_a[i].state.body_rate.eigen(), trace_b[i].state.body_rate.eigen()) << i;
  }
  ASSERT_FALSE(gnss_a.empty());
  ASSERT_EQ(gnss_a.size(), gnss_b.size());
  for (std::size_t i = 0; i < gnss_a.size(); ++i) {
    EXPECT_EQ(gnss_a[i], gnss_b[i]) << i;  // bitwise: same noise draws
  }
}

TEST(ClosedLoop, GnssFaultScheduleAppliesAtSampleTime) {
  // A scheduled outage window must invalidate exactly the fixes inside it —
  // the deferred §9.2 binding, now live in the loop.
  scenario::SimConfig config = freeSpace(1.0);
  scenario::GnssFaultEvent outage;
  outage.unit = "gps_a";
  outage.type = scenario::GnssFaultEvent::Type::kOutage;
  outage.start_s = 0.35;
  outage.stop_s = 0.65;
  config.environment.gnss_fault_events.push_back(outage);

  scenario::Vehicle vehicle;
  std::string error;
  ASSERT_TRUE(scenario::buildVehicle(suite({gnssUnit("gps_a", 10.0)}, {}), 7, vehicle, &error))
      << error;

  scenario::SimRunner runner;
  io::ClosedLoop loop(runner, vehicle);
  loop.setDataPaths(goldenPaths());
  ASSERT_TRUE(runner.build(config, goldenPaths(), &error, loop.wrench())) << error;

  std::vector<bool> valid;
  auto fsw = [&](const io::FswInputs& in) {
    valid.push_back(in.gnss[0].measurement.valid);
    return io::FswOutputs{};
  };
  ASSERT_TRUE(loop.run(fsw, nullptr, &error)) << error;
  ASSERT_EQ(valid.size(), 10u);
  // Boundaries at 0.1..1.0 s; fixes at 0.1..1.0 s. The 0.4/0.5/0.6 s fixes fall
  // inside [0.35, 0.65); everything else is clean.
  const std::vector<bool> expected = {true,  true, true, false, false,
                                      false, true, true, true,  true};
  EXPECT_EQ(valid, expected);
}

TEST(ClosedLoop, DiscreteSensorsPublishOnTheirOwnGrid) {
  // A 5 Hz star tracker under a 10 Hz FSW: nothing to publish at the first
  // boundary (its first sample is due at 0.2 s), latest-valid thereafter.
  scenario::SimConfig config = freeSpace(0.4);
  scenario::UnitConfig st;
  st.name = "st_a";
  st.model_id = "TEST-ST";
  st.kind = "star_tracker";
  st.params = {{"temporal_noise_xy_arcsec_3sigma", 11.0},
               {"temporal_noise_z_arcsec_3sigma", 70.0},
               {"fov_deg", 15.0},
               {"earth_exclusion_deg", 22.0},
               {"update_rate_hz", 5.0}};

  scenario::Vehicle vehicle;
  std::string error;
  ASSERT_TRUE(scenario::buildVehicle(suite({st}, {}), 3, vehicle, &error)) << error;

  scenario::SimRunner runner;
  io::ClosedLoop loop(runner, vehicle);
  loop.setDataPaths(goldenPaths());
  ASSERT_TRUE(runner.build(config, goldenPaths(), &error, loop.wrench())) << error;

  std::vector<bool> ever;
  std::vector<pt::Tai> tags;
  auto fsw = [&](const io::FswInputs& in) {
    ever.push_back(in.star_trackers[0].ever_sampled);
    tags.push_back(in.star_trackers[0].measurement.time_tag);
    return io::FswOutputs{};
  };
  ASSERT_TRUE(loop.run(fsw, nullptr, &error)) << error;
  ASSERT_EQ(ever.size(), 4u);
  EXPECT_FALSE(ever[0]);  // boundary 0.1 s: first ST sample not due until 0.2 s
  EXPECT_TRUE(ever[1]);
  EXPECT_TRUE(ever[2]);
  // Latest-valid: at 0.3 s the newest sample is still the 0.2 s one…
  EXPECT_EQ(tags[2].nanosecondsSinceEpoch(), tags[1].nanosecondsSinceEpoch());
  // …and at 0.4 s the 0.4 s sample has landed.
  EXPECT_GT(tags[3].nanosecondsSinceEpoch(), tags[2].nanosecondsSinceEpoch());
}

TEST(ClosedLoop, PayloadGeometryIsSampledAndStaysSimSide) {
  // A payload sensor is sampled on its own grid like any other discrete unit,
  // but its products never enter FswInputs: they are truth-derived geometry, and
  // the §2.3 boundary is what this asserts. If a payload ever needs to reach the
  // flight side it goes through a flight-side component and a wire record, not
  // through here.
  scenario::SimConfig config = freeSpace(0.4);
  scenario::UnitConfig payload;
  payload.name = "imager_a";
  payload.model_id = "TEST-PAYLOAD";
  payload.kind = "payload_sensor";
  payload.params = {{"half_fov_x_deg", 5.0},
                    {"half_fov_y_deg", 4.0},
                    {"pixels_x", 2048.0},
                    {"pixels_y", 1536.0},
                    {"update_rate_hz", 5.0}};

  scenario::Vehicle vehicle;
  std::string error;
  ASSERT_TRUE(scenario::buildVehicle(suite({payload}, {}), 3, vehicle, &error)) << error;

  scenario::SimRunner runner;
  io::ClosedLoop loop(runner, vehicle);
  loop.setDataPaths(goldenPaths());
  ASSERT_TRUE(runner.build(config, goldenPaths(), &error, loop.wrench())) << error;

  // Capture what the FSW is actually handed: the boundary is enforced by the
  // type system (FswInputs has no payload member), but a future refactor could
  // add one, and this is the assertion that would fail if it did.
  int boundaries = 0;
  auto fsw = [&](const io::FswInputs& in) {
    ++boundaries;
    EXPECT_TRUE(in.imus.empty());
    EXPECT_TRUE(in.star_trackers.empty());
    EXPECT_TRUE(in.sun_sensors.empty());
    EXPECT_TRUE(in.magnetometers.empty());
    EXPECT_TRUE(in.gnss.empty()) << "the vehicle's only unit is a payload — nothing may reach "
                                    "the FSW through a sensor vector";
    return io::FswOutputs{};
  };
  ASSERT_TRUE(loop.run(fsw, nullptr, &error)) << error;
  EXPECT_EQ(boundaries, 4) << "0.4 s at 10 Hz";

  ASSERT_EQ(loop.payloadGeometry().size(), 1u);
  const auto& geometry = loop.payloadGeometry()[0];
  EXPECT_EQ(geometry.name, "imager_a");
  EXPECT_TRUE(geometry.ever_sampled) << "a 5 Hz payload must have sampled over 0.4 s";
  // The boresight is a real direction, and the shared occlusion model filled in
  // the pointing angles — the vehicle starts at identity attitude on the +x axis,
  // so body +Z is at right angles to nadir.
  EXPECT_NEAR(geometry.measurement.boresight_eci.eigen().norm(), 1.0, 1e-12);
  EXPECT_NEAR(geometry.measurement.occlusion.nadir_angle_rad, M_PI_2, 1e-9);
  EXPECT_LE(geometry.measurement.occlusion.sun_angle_rad, M_PI);
}
