/// @file Two-process SITL lockstep integration test (design doc §2.2, §2.4).
///
/// Spawns the real `flight_PolarisFsw` binary with `-s <port>`, runs the truth
/// sim's closed loop against it over the F´ TCP transport, and requires the
/// resulting truth trace to be **bitwise identical** to the same run with the
/// in-process zero-command callback. The Push-34 flight bridge answers zero
/// commands, so any difference is a transport defect: a dropped barrier, a
/// mis-framed message, or nondeterminism leaking in through the socket path.
///
/// Skips (never fails) when the flight binary is absent — CI's unit-test job
/// builds only the native-ut tree. Set `POLARIS_FSW_BIN` to override the
/// default `build-artifacts/Linux/flight_PolarisFsw/bin/flight_PolarisFsw`.
///
/// Verifies REQ-SIM-004 (measurements-only boundary, now across processes) and
/// the §2.4 bit-reproducibility claim through the real transport.

#include <gtest/gtest.h>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cmath>
#include <cstdlib>
#include <Eigen/Core>
#include <string>
#include <vector>

#include "io/closed_loop.hpp"
#include "io/sitl_server.hpp"
#include "scenario/sim_config.hpp"
#include "scenario/sim_runner.hpp"
#include "scenario/vehicle.hpp"

namespace {

namespace io = polaris::sim::io;
namespace scenario = polaris::sim::scenario;
namespace pm = polaris::math;
namespace pt = polaris::time;

scenario::DataPaths goldenPaths() {
  return scenario::DataPaths::under(POLARIS_GOLDEN_DIR);
}

std::string fswBinaryPath() {
  if (const char* env = std::getenv("POLARIS_FSW_BIN")) {
    return env;
  }
  return "build-artifacts/Linux/flight_PolarisFsw/bin/flight_PolarisFsw";
}

/// Two-body 500 km orbit, environment mostly off — the plant is not under test
/// here; the transport is.
scenario::SimConfig transportOrbit(double duration_s) {
  scenario::SimConfig c;
  c.scenario_name = "sitl-lockstep";
  c.spacecraft.name = "sitl-vehicle";
  c.spacecraft.mass_kg = 12.0;
  c.spacecraft.inertia_kgm2 = Eigen::Vector3d(0.12, 0.12, 0.10).asDiagonal();
  c.initial_state.epoch = pt::Tai::fromNanosecondsSinceEpoch(1767225637000000000LL);
  c.initial_state.position = pm::Vec3<pm::frames::ECI>(Eigen::Vector3d(6.878137e6, 0.0, 0.0));
  c.initial_state.velocity = pm::Vec3<pm::frames::ECI>(Eigen::Vector3d(0.0, 7612.0, 0.0));
  c.initial_state.attitude = pm::Quat<pm::frames::Body, pm::frames::ECI>::Identity();
  c.initial_state.body_rate = pm::Vec3<pm::frames::Body>(Eigen::Vector3d(0.01, 0.0, 0.02));
  c.environment.gravity_degree = -1;
  c.environment.magnetic_field = scenario::MagneticModel::kNone;
  c.environment.drag_enabled = false;
  c.environment.srp_enabled = false;
  c.propagation.duration_s = duration_s;
  c.propagation.output_step_s = duration_s;
  c.propagation.fsw_rate_hz = 10.0;
  return c;
}

scenario::UnitConfig unit(const std::string& name, const std::string& model,
                          const std::string& kind, std::map<std::string, double> params) {
  scenario::UnitConfig u;
  u.name = name;
  u.model_id = model;
  u.kind = kind;
  u.params = std::move(params);
  return u;
}

/// IMU + GNSS + one wheel: exercises the accumulation record, a discrete
/// record, and the reply's command records.
scenario::SpacecraftConfig sitlSuite() {
  scenario::SpacecraftConfig sc;
  sc.sensors.push_back(
      unit("imu-a", "TEST-IMU", "imu", {{"gyro_range_deg_s", 400.0}, {"sample_rate_hz", 50.0}}));
  sc.sensors.push_back(unit("gnss-a", "TEST-GNSS", "gnss",
                            {{"horizontal_position_rms_m", 1.2},
                             {"velocity_accuracy_m_s_rms", 0.03},
                             {"max_rate_hz", 1.0}}));
  scenario::UnitConfig rw =
      unit("rw-x", "TEST-RW", "reaction_wheel",
           {{"max_torque_nm", 0.1}, {"rotor_inertia_kg_m2", 1.0e-3}, {"max_speed_rpm", 60000.0}});
  rw.spin_axis = Eigen::Vector3d::UnitX();
  sc.actuators.push_back(rw);
  return sc;
}

/// Run the loop with @p fsw and return the macro trace.
std::vector<io::MacroSample> runLoop(const scenario::SimConfig& config,
                                     const io::FswCallback& fsw) {
  scenario::Vehicle vehicle;
  std::string error;
  EXPECT_TRUE(scenario::buildVehicle(sitlSuite(), 1, vehicle, &error)) << error;
  scenario::SimRunner runner;
  io::ClosedLoop loop(runner, vehicle);
  EXPECT_TRUE(runner.build(config, goldenPaths(), &error, loop.wrench())) << error;
  std::vector<io::MacroSample> trace;
  EXPECT_TRUE(loop.run(fsw, &trace, &error)) << error;
  return trace;
}

TEST(SitlLockstep, TwoProcessTraceIsBitIdenticalToInProcessZeroCommands) {
  const std::string bin = fswBinaryPath();
  if (::access(bin.c_str(), X_OK) != 0) {
    GTEST_SKIP() << "flight binary not built at " << bin
                 << " (run `uv run fprime-util build`, or set POLARIS_FSW_BIN)";
  }

  const scenario::SimConfig config = transportOrbit(10.0);  // 100 barriers at 10 Hz

  // Reference: in-process default callback (open loop, zero commands).
  const std::vector<io::MacroSample> ref = runLoop(config, io::FswCallback{});

  // SITL: same run, commands travel to the real F´ process and back.
  io::SitlServer::Counts counts;
  counts.imu = 1;
  counts.gnss = 1;
  counts.wheel = 1;
  io::SitlServer server(counts, 100'000'000LL);
  ASSERT_TRUE(server.start(0)) << server.lastError();

  const pid_t pid = ::fork();
  ASSERT_GE(pid, 0);
  if (pid == 0) {
    // Child: quiet stdout/stderr, exec the deployment against our port.
    ::freopen("/dev/null", "w", stdout);
    ::freopen("/dev/null", "w", stderr);
    const std::string port = std::to_string(server.port());
    ::execl(bin.c_str(), bin.c_str(), "-s", port.c_str(), static_cast<char*>(nullptr));
    _exit(127);  // exec failed
  }

  const std::vector<io::MacroSample> sitl = runLoop(config, server.callback());
  EXPECT_TRUE(server.healthy()) << server.lastError();
  EXPECT_EQ(server.stepsExchanged(), 100u);
  server.stop();  // sends SHUTDOWN

  // The deployment keeps running after SHUTDOWN (it only stops replying);
  // terminate it and reap.
  ::kill(pid, SIGTERM);
  int status = 0;
  if (::waitpid(pid, &status, WNOHANG) == 0) {
    ::usleep(500 * 1000);
    if (::waitpid(pid, &status, WNOHANG) == 0) {
      ::kill(pid, SIGKILL);
      ::waitpid(pid, &status, 0);
    }
  }

  // Bitwise: the transport must add nothing — same states, same epochs.
  ASSERT_EQ(sitl.size(), ref.size());
  for (std::size_t i = 0; i < ref.size(); ++i) {
    EXPECT_EQ(sitl[i].state.epoch.nanosecondsSinceEpoch(),
              ref[i].state.epoch.nanosecondsSinceEpoch());
    EXPECT_TRUE(sitl[i].state.position.eigen() == ref[i].state.position.eigen()) << "step " << i;
    EXPECT_TRUE(sitl[i].state.velocity.eigen() == ref[i].state.velocity.eigen()) << "step " << i;
    EXPECT_TRUE(sitl[i].state.attitude.core().coeffs() == ref[i].state.attitude.core().coeffs())
        << "step " << i;
    EXPECT_TRUE(sitl[i].state.body_rate.eigen() == ref[i].state.body_rate.eigen()) << "step " << i;
  }
}

}  // namespace
