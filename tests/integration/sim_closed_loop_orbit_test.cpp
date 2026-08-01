/// @file Full-stack truth-side validation through the §2.4 closed loop
/// (REQ-SIM-001/002/003, §2.4, §6).
///
/// The unit suites verify each model alone and the loop's mechanics; these tests
/// verify that the whole truth side tells ONE physical story across an orbit
/// with attitude: the loop adds no dynamics of its own, the sensing chain
/// carries complete attitude information, and every sensor's output agrees with
/// geometry recomputed independently from the truth state — ECEF norms, the
/// rotated magnetic field, the Earth-limb star-tracker keep-out, the eclipse
/// arc, and the ground track. Sensors run ideal (noise off) so the assertions
/// are equalities against physics, not statistical bounds.

#include <gtest/gtest.h>

#include <cmath>
#include <Eigen/Core>
#include <string>
#include <vector>

#include "constants/constants.hpp"
#include "frames/eci_ecef.hpp"
#include "frames/eop.hpp"
#include "io/closed_loop.hpp"
#include "scenario/sim_config.hpp"
#include "scenario/sim_runner.hpp"
#include "scenario/vehicle.hpp"
#include "sensors/gnss_jamming.hpp"
#include "time/leap_seconds.hpp"
#include "time/tdb.hpp"
#include "world/eclipse.hpp"
#include "world/eop_file.hpp"

namespace {

namespace io = polaris::sim::io;
namespace scenario = polaris::sim::scenario;
namespace sensors = polaris::sim::sensors;
namespace world = polaris::sim::world;
namespace pm = polaris::math;
namespace pt = polaris::time;
namespace pc = polaris::constants;
namespace pf = polaris::frames;
namespace ps = polaris::state;

constexpr double kDeg2Rad = 0.017453292519943295;

scenario::DataPaths goldenPaths() {
  return scenario::DataPaths::under(POLARIS_GOLDEN_DIR);
}

/// A 500 km orbit with the full environment on, at the 2026-01-01 epoch the
/// committed EOP/ephemeris/space-weather products cover.
scenario::SimConfig fullEnvironmentOrbit(double duration_s, double inc_deg, double fsw_rate_hz) {
  scenario::SimConfig c;
  c.scenario_name = "closed-loop-orbit";
  c.spacecraft.name = "test-vehicle";
  c.spacecraft.mass_kg = 12.0;
  c.spacecraft.inertia_kgm2 = Eigen::Vector3d(0.12, 0.12, 0.10).asDiagonal();
  c.spacecraft.drag_area_m2 = 0.06;
  c.spacecraft.srp_area_m2 = 0.06;

  const double radius = 6.878137e6;
  const double speed = std::sqrt(pc::wgs84::kGM / radius);
  const double inc = inc_deg * kDeg2Rad;
  c.initial_state.epoch = pt::Tai::fromNanosecondsSinceEpoch(1767225637000000000LL);
  c.initial_state.position = pm::Vec3<pm::frames::ECI>(Eigen::Vector3d(radius, 0.0, 0.0));
  c.initial_state.velocity =
      pm::Vec3<pm::frames::ECI>(Eigen::Vector3d(0.0, speed * std::cos(inc), speed * std::sin(inc)));
  c.initial_state.attitude = pm::Quat<pm::frames::Body, pm::frames::ECI>::Identity();
  c.initial_state.body_rate = pm::Vec3<pm::frames::Body>(Eigen::Vector3d::Zero());

  c.environment.gravity_degree = 8;
  c.environment.sun_third_body = true;
  c.environment.moon_third_body = true;
  c.environment.srp_enabled = true;
  c.environment.drag_enabled = true;
  c.environment.magnetic_field = scenario::MagneticModel::kIgrf;
  // §5.3 disturbance torques left ON (the schema default), stated explicitly
  // because this is the full-environment closed-loop case: the point is that
  // the controller holds attitude against the real environment, so the ~1e-8
  // N·m the four torques contribute is the thing being flown against.
  c.environment.gravity_gradient_torque_enabled = true;
  c.environment.aero_torque_enabled = true;
  c.environment.srp_torque_enabled = true;
  c.environment.residual_dipole_torque_enabled = true;

  c.propagation.duration_s = duration_s;
  c.propagation.output_step_s = duration_s;
  c.propagation.fsw_rate_hz = fsw_rate_hz;
  return c;
}

scenario::SpacecraftConfig suite(std::vector<scenario::UnitConfig> sensor_units) {
  scenario::SpacecraftConfig sc;
  sc.sensors = std::move(sensor_units);
  return sc;
}

scenario::UnitConfig unit(const std::string& name, const std::string& kind,
                          std::map<std::string, double> params) {
  scenario::UnitConfig u;
  u.name = name;
  u.model_id = "TEST-" + kind;
  u.kind = kind;
  u.params = std::move(params);
  return u;
}

}  // namespace

TEST(ClosedLoopOrbit, ZeroCommandLoopMatchesOpenLoopPropagation) {
  // The loop machinery must add no dynamics of its own: with no sensors and no
  // commands it makes the same propagate() calls runAt() makes. Not bitwise,
  // deliberately: the loop pins each state's epoch to the exact integer-ns
  // event grid (so sensor timing cannot drift over a long run) while the
  // open-loop path accumulates float-second epoch conversions — sub-ns epoch
  // differences that show up as last-ULP state differences. The tolerances
  // below are ~1e6 times tighter than any physical claim in the suite.
  scenario::SimConfig config = fullEnvironmentOrbit(30.0, 45.0, 10.0);

  scenario::Vehicle vehicle;  // deliberately empty: no sensors, no actuators
  std::string error;
  ASSERT_TRUE(scenario::buildVehicle(scenario::SpacecraftConfig{}, 1, vehicle, &error)) << error;

  scenario::SimRunner loop_runner;
  io::ClosedLoop loop(loop_runner, vehicle);
  ASSERT_TRUE(loop_runner.build(config, goldenPaths(), &error, loop.wrench())) << error;
  std::vector<io::MacroSample> trace;
  ASSERT_TRUE(loop.run({}, &trace, &error)) << error;
  ASSERT_EQ(trace.size(), 301u);  // t=0 plus 300 boundaries at 10 Hz

  scenario::SimRunner reference;
  ASSERT_TRUE(reference.build(config, goldenPaths(), &error)) << error;
  std::vector<double> times;
  for (int i = 1; i <= 300; ++i) {
    times.push_back(0.1 * static_cast<double>(i));
  }
  std::vector<scenario::TrajectorySample> ref;
  ASSERT_TRUE(reference.runAt(times, ref, &error)) << error;
  ASSERT_EQ(ref.size(), times.size());

  for (std::size_t i = 0; i < ref.size(); ++i) {
    EXPECT_LT((trace[i + 1].state.position.eigen() - ref[i].state.position.eigen()).norm(), 1e-5)
        << i;
    EXPECT_LT((trace[i + 1].state.velocity.eigen() - ref[i].state.velocity.eigen()).norm(), 1e-8)
        << i;
    EXPECT_LT((trace[i + 1].state.attitude.core().coeffs() - ref[i].state.attitude.core().coeffs())
                  .norm(),
              1e-10)
        << i;
  }
}

TEST(ClosedLoopOrbit, IdealImuDeadReckoningReproducesTruthAttitude) {
  // The sensing chain carries complete attitude information: composing the
  // ideal IMU's per-sample delta-angles over 200 s of full-environment
  // flight (gravity-gradient torque exciting the rates) must reproduce the
  // truth attitude. FSW rate = IMU rate so each read is one sample, composed in
  // order. This is a truth-side dead-reckoning — the germ of the Phase-4
  // propagation the estimator will do.
  scenario::SimConfig config = fullEnvironmentOrbit(200.0, 45.0, 25.0);
  // Mild asymmetry: enough for gravity-gradient torque to act, small enough
  // that the end-of-interval rate sampling error (ω̇·dt²/2 per step) stays
  // far below the assertion tolerance.
  config.spacecraft.inertia_kgm2 = Eigen::Vector3d(0.100, 0.100, 0.101).asDiagonal();
  config.initial_state.body_rate = pm::Vec3<pm::frames::Body>(Eigen::Vector3d(0.02, -0.015, 0.01));

  scenario::Vehicle vehicle;
  std::string error;
  ASSERT_TRUE(scenario::buildVehicle(
      suite({unit("imu_a", "imu", {{"gyro_range_deg_s", 400.0}, {"sample_rate_hz", 25.0}})}), 1,
      vehicle, &error, scenario::NoiseSettings{false, false}))
      << error;

  scenario::SimRunner runner;
  io::ClosedLoop loop(runner, vehicle);
  ASSERT_TRUE(runner.build(config, goldenPaths(), &error, loop.wrench())) << error;

  pm::Quaternion dead_reckoned = config.initial_state.attitude.core();
  auto fsw = [&](const io::FswInputs& in) {
    const Eigen::Vector3d delta = in.imus[0].delta_angle_rad.eigen();
    const double angle = delta.norm();
    if (angle > 0.0) {
      // q_body(t+dt) = δq ⊗ q_body(t): the body rotated by δθ in body axes.
      dead_reckoned = pm::Quaternion::FromAxisAngle(delta / angle, angle) * dead_reckoned;
    }
    return io::FswOutputs{};
  };

  std::vector<io::MacroSample> trace;
  ASSERT_TRUE(loop.run(fsw, &trace, &error)) << error;

  const pm::Quaternion truth = trace.back().state.attitude.core();
  const pm::Quaternion delta = dead_reckoned * truth.conjugate();
  const double error_rad = 2.0 * std::asin(std::min(1.0, delta.vec().norm()));
  EXPECT_LT(error_rad, 0.01 * kDeg2Rad)
      << "dead-reckoned attitude drifted " << error_rad / kDeg2Rad << " deg over 200 s";
  // And the body genuinely rotated — the check above is not comparing two
  // identity quaternions.
  const pm::Quaternion total = truth * config.initial_state.attitude.core().conjugate();
  // ~5.4 rad of accumulated body rotation wraps into [0, π] as a net rotation
  // angle; anything clearly nonzero proves the comparison above is not between
  // two identity quaternions.
  EXPECT_GT(2.0 * std::asin(std::min(1.0, total.vec().norm())), 0.5);
}

TEST(ClosedLoopOrbit, SensorsAgreeWithIndependentlyRecomputedGeometry) {
  // One full orbit, ideal sensors, every sample cross-checked against geometry
  // recomputed straight from the truth trace — including with formulas
  // independent of the model code (the star-tracker limb predicate below).
  const double radius = 6.878137e6;
  const double period = 2.0 * M_PI * std::sqrt(std::pow(radius, 3) / pc::wgs84::kGM);
  scenario::SimConfig config = fullEnvironmentOrbit(period, 45.0, 0.5);

  // Build the runner first: the Sun's direction at the epoch aims the sun
  // sensor's mounting below (with an inertial-hold attitude, a +z-mounted cell
  // would spend the whole January orbit >100° from a Sun at dec −23°).
  scenario::SimRunner runner;
  std::string error;

  scenario::Vehicle vehicle;
  io::ClosedLoop loop(runner, vehicle);
  loop.setDataPaths(goldenPaths());
  ASSERT_TRUE(runner.build(config, goldenPaths(), &error, loop.wrench())) << error;

  // Aim the sun sensor at the epoch Sun: boresight (mounting z-column) along
  // the sat→Sun direction, so the lit arc reads sun_present and the eclipse
  // arc reads absent — both sides of the predicate get exercised.
  {
    pm::Vec3<pm::frames::ECI> sun0;
    ASSERT_TRUE(runner.sunPositionFn()(pt::toTdb(pt::toTt(config.initial_state.epoch)), sun0));
    const Eigen::Vector3d dir = (sun0.eigen() - config.initial_state.position.eigen()).normalized();
    const Eigen::Matrix3d mount =
        Eigen::Quaterniond::FromTwoVectors(Eigen::Vector3d::UnitZ(), dir).toRotationMatrix();
    scenario::SpacecraftConfig sc = suite({
        unit("st_a", "star_tracker",
             {{"temporal_noise_xy_arcsec_3sigma", 11.0},
              {"temporal_noise_z_arcsec_3sigma", 70.0},
              {"fov_deg", 15.0},
              {"earth_exclusion_deg", 22.0},
              {"update_rate_hz", 0.5}}),
        unit("ss_a", "sun_sensor",
             {{"half_fov_deg", 60.0},
              {"accuracy_inner_half_angle_deg", 45.0},
              {"accuracy_inner_deg_3sigma", 0.5},
              {"accuracy_outer_deg_3sigma", 2.0},
              {"update_rate_hz", 0.5}}),
        unit("mag_a", "magnetometer", {{"range_ut", 100.0}}),
        unit("gps_a", "gnss",
             {{"horizontal_position_rms_m", 1.2},
              {"velocity_accuracy_m_s_rms", 0.03},
              {"max_rate_hz", 0.5}}),
    });
    sc.sensors[1].mounting_dcm = mount;
    ASSERT_TRUE(
        scenario::buildVehicle(sc, 1, vehicle, &error, scenario::NoiseSettings{false, false}))
        << error;
  }

  struct Snapshot {
    sensors::StarTrackerMeasurement st;
    sensors::SunSensorMeasurement ss;
    sensors::MagnetometerMeasurement mag;
    sensors::GnssMeasurement gnss;
  };

  std::vector<Snapshot> reads;
  auto fsw = [&](const io::FswInputs& in) {
    reads.push_back({in.star_trackers[0].measurement, in.sun_sensors[0].measurement,
                     in.magnetometers[0].measurement, in.gnss[0].measurement});
    return io::FswOutputs{};
  };

  std::vector<io::MacroSample> trace;
  ASSERT_TRUE(loop.run(fsw, &trace, &error)) << error;
  ASSERT_EQ(reads.size() + 1, trace.size());

  // Independent resources for the recomputation.
  const pt::LeapSecondTable leap = pt::LeapSecondTable::historical();
  pf::EopTable<512> eop;
  const pt::Tai start = config.initial_state.epoch;
  const pt::Tai end = start + pt::Duration::fromSecondsF(config.propagation.duration_s);
  ASSERT_TRUE(world::loadEopFile(goldenPaths().eop, leap, start, end, eop, nullptr));
  const world::MagneticFieldFn field = runner.magneticFieldFn();
  const world::BodyPositionFn sun = runner.sunPositionFn();
  ASSERT_TRUE(field);
  ASSERT_TRUE(sun);

  const double atmosphere_limb =
      pc::wgs84::kSemiMajorAxis + config.environment.occultation_atmosphere_m;
  int st_valid = 0;
  int st_blocked = 0;
  int ss_present = 0;
  int eclipsed = 0;
  double max_lat_deg = 0.0;

  for (std::size_t k = 0; k < reads.size(); ++k) {
    const ps::TruthState& s = trace[k + 1].state;  // same boundary as read k
    const Snapshot& m = reads[k];
    const Eigen::Vector3d r = s.position.eigen();
    const pt::Tai t = s.epoch;

    // --- GNSS: a rotation preserves the norm, and the ideal fix round-trips
    // back to the exact truth state through the inverse reduction.
    ASSERT_TRUE(m.gnss.valid) << k;
    EXPECT_NEAR(m.gnss.position_m.eigen().norm(), r.norm(), 1e-6) << k;
    pm::Vec3<pm::frames::ECI> r_back;
    pm::Vec3<pm::frames::ECI> v_back;
    ASSERT_TRUE(
        pf::eciStateFromEcef(t, eop, leap, m.gnss.position_m, m.gnss.velocity_m_s, r_back, v_back))
        << k;
    EXPECT_LT((r_back.eigen() - r).norm(), 1e-5) << k;
    EXPECT_LT((v_back.eigen() - s.velocity.eigen()).norm(), 1e-8) << k;

    // Ground track: geodetic latitude bounded by the inclination.
    double lat = 0.0;
    double lon = 0.0;
    sensors::ecefToGeodeticDeg(m.gnss.position_m, lat, lon);
    max_lat_deg = std::max(max_lat_deg, std::abs(lat));

    // --- Magnetometer: the ideal measurement is exactly the IGRF field at the
    // truth position, rotated into the body by the truth attitude.
    pm::Vec3<pm::frames::ECI> b_eci;
    ASSERT_TRUE(field(t, s.position, b_eci)) << k;
    const Eigen::Vector3d b_body = s.attitude.rotate(b_eci).eigen();
    EXPECT_TRUE(m.mag.field_tesla.eigen().isApprox(b_body, 1e-12)) << k;

    // --- Star tracker: validity must equal the Earth-limb geometry, computed
    // here with an independent formula (angle to the Earth's centre minus the
    // apparent limb radius vs. the exclusion angle — not the occlusion code).
    const Eigen::Vector3d boresight_eci =
        s.attitude.inverse().rotate(pm::Vec3<pm::frames::Body>(Eigen::Vector3d::UnitZ())).eigen();
    const Eigen::Vector3d to_earth = -r.normalized();
    const double centre_angle =
        std::acos(std::clamp(boresight_eci.normalized().dot(to_earth), -1.0, 1.0));
    const double limb_radius = std::asin(std::min(1.0, atmosphere_limb / r.norm()));
    const bool geometry_clear = (centre_angle - limb_radius) >= 22.0 * kDeg2Rad;
    EXPECT_EQ(m.st.valid, geometry_clear) << "sample " << k;
    if (m.st.valid) {
      ++st_valid;
      // Ideal tracker: the reported attitude IS the truth attitude.
      EXPECT_TRUE(m.st.attitude.core().coeffs().isApprox(s.attitude.core().coeffs(), 1e-12)) << k;
    } else {
      ++st_blocked;
    }

    // --- Sun sensor: presence must equal (lit AND within the acceptance cone),
    // recomputed from the eclipse model and the truth sun geometry; and the
    // ideal vector is the exact truth direction.
    pm::Vec3<pm::frames::ECI> sun_eci;
    ASSERT_TRUE(sun(pt::toTdb(pt::toTt(t)), sun_eci)) << k;
    const double shadow = world::shadowFactor(r, sun_eci.eigen());
    const Eigen::Vector3d sun_dir_body =
        s.attitude.rotate(pm::Vec3<pm::frames::ECI>((sun_eci.eigen() - r).normalized())).eigen();
    const Eigen::Vector3d ss_boresight = vehicle.sun_sensors[0].mounting_dcm.col(2);
    const double incidence = std::acos(std::clamp(sun_dir_body.dot(ss_boresight), -1.0, 1.0));
    const bool expect_present = shadow > 0.05 && incidence <= 60.0 * kDeg2Rad;
    EXPECT_EQ(m.ss.sun_present, expect_present) << "sample " << k;
    if (m.ss.sun_present) {
      ++ss_present;
      EXPECT_TRUE(m.ss.sun_dir_body.eigen().isApprox(sun_dir_body, 1e-9)) << k;
    }
    if (shadow < 0.05) {
      ++eclipsed;
    }
  }

  // The orbit genuinely exercised both sides of each predicate: the inertially
  // fixed boresight swept past the Earth and away from it, and the 45°-inclined
  // 500 km orbit crossed the January shadow cone.
  EXPECT_GT(st_valid, 0);
  EXPECT_GT(st_blocked, 0);
  EXPECT_GT(ss_present, 0);
  EXPECT_GT(eclipsed, 0);
  // Ground track never exceeds the inclination (plus a hair of geodetic
  // latitude inflation on an oblate Earth).
  EXPECT_LT(max_lat_deg, 45.0 + 0.5);
  EXPECT_GT(max_lat_deg, 40.0);  // and the orbit really is inclined
}
