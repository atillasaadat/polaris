/// @file Closed-loop attitude estimation in SITL (design doc §8.1, §19.3).
///
/// The end-to-end gate on the **tuning delivery path**: the config compiler
/// resolves this vehicle's estimator tuning, encodes it into a `Svc::PrmDb`
/// parameter file, the deployment loads that file at startup (`-P`), and the
/// `flight::AttitudeEstimator` therefore configures itself and acquires an
/// attitude from the sun/magnetometer pair the truth sim feeds it over the SITL
/// wire. Push 40 shipped the estimator; without this path it telemeters mode
/// INVALID and one `ConfigInvalid`, because there are no flight defaults
/// (§19.3) — so "the parameter file is right" and "the estimator works" are the
/// same assertion, and this test makes it once, through the real chain:
///
///   leo_smallsat.yaml -> configc -> PrmDb.dat -> prmDb -> attitudeEstimator
///
/// The assertions read the deployment's own event stream, which is what an
/// operator would see: the parameter file loaded with every record, no
/// `ConfigInvalid` from either validity gate, `AttitudeAcquired`, and — since
/// Push 44 — `FineModeEngaged` with no demotion over the run, which is the
/// fine↔coarse arbitration of REQ-ADET-004 exercised on the vehicle.
///
/// Note the covariance comparison that would round this out (fine trace below
/// the coarse floor) is asserted in the component harness instead: the only
/// channel back from the deployment here is its text event stream, and adding a
/// periodic covariance EVR to serve a test would be telemetry noise on a real
/// vehicle.
///
/// Skips (never fails) when the flight binary or the Python toolchain is absent
/// — CI's unit-test job builds only the native-ut tree. `POLARIS_FSW_BIN` and
/// `POLARIS_PYTHON` override the defaults.
///
/// Verifies REQ-CFG-001 (every consumer reads compiler-derived artifacts) and
/// REQ-ADET-002/REQ-ADET-004 (coarse attitude determination and its mode
/// reporting) on the vehicle rather than in the library.

#include <gtest/gtest.h>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <Eigen/Geometry>
#include <fstream>
#include <sstream>
#include <string>

#include "ephemeris/analytic_sun.hpp"
#include "io/closed_loop.hpp"
#include "io/sitl_server.hpp"
#include "scenario/sim_config.hpp"
#include "scenario/sim_runner.hpp"
#include "scenario/vehicle.hpp"
#include "time/tdb.hpp"
#include "time/timescales.hpp"

namespace {

namespace io = polaris::sim::io;
namespace scenario = polaris::sim::scenario;
namespace pm = polaris::math;
namespace pt = polaris::time;

/// Scenario epoch: 2026-01-01, inside the committed ephemeris/EOP coverage and
/// inside the IGRF-14 snapshot's validity, so the estimator runs on precise
/// references rather than exercising the coarse fallbacks (which have their own
/// tests).
constexpr std::int64_t kEpochTaiNs = 1767225637000000000LL;
/// Decimal year matching kEpochTaiNs, passed to the deployment as -Y so the
/// onboard IGRF snapshot is the one that brackets the scenario rather than the
/// one the workstation clock happens to select.
constexpr const char* kEpochDecimalYear = "2026.0";
/// 10 s at the 10 Hz GNC rate: 100 barriers. The estimator acquires on its
/// first valid TRIAD, so this is generous.
constexpr double kDurationS = 10.0;

std::string envOr(const char* name, const char* fallback) {
  const char* value = std::getenv(name);
  return (value != nullptr) ? std::string(value) : std::string(fallback);
}

std::string fswBinaryPath() {
  return envOr("POLARIS_FSW_BIN", "build-artifacts/Linux/flight_PolarisFsw/bin/flight_PolarisFsw");
}

/// Interpreter that can `import configc`: the uv-managed project venv by
/// default (`uv sync`), overridable for other layouts.
std::string pythonPath() {
  return envOr("POLARIS_PYTHON", ".venv/bin/python3");
}

std::string dictionaryPath() {
  return envOr("POLARIS_FSW_DICT",
               "build-artifacts/Linux/flight_PolarisFsw/dict/PolarisFswTopologyDictionary.json");
}

/// Attitude placing the sun sensor boresight (unit +Z, identity mounting) on the
/// Sun at the scenario epoch, so the sun pair is available from the first cycle.
/// Derived from the analytic ephemeris rather than hard-coded: the point of the
/// test is the tuning path, and a stale hand-computed quaternion would turn an
/// epoch change into a mystery failure.
pm::Quat<pm::frames::Body, pm::frames::ECI> sunPointingAttitude() {
  const pt::Tai epoch = pt::Tai::fromNanosecondsSinceEpoch(kEpochTaiNs);
  const Eigen::Vector3d sun =
      polaris::ephemeris::sunPositionEci(pt::toTdb(pt::toTt(epoch))).eigen().normalized();
  // Body <- ECI rotation whose third row is the sun direction, i.e. R * sun = +Z.
  const Eigen::Vector3d x = sun.unitOrthogonal();
  const Eigen::Matrix3d dcm =
      (Eigen::Matrix3d() << x.transpose(), sun.cross(x).transpose(), sun.transpose()).finished();
  return pm::Quat<pm::frames::Body, pm::frames::ECI>(pm::Quaternion::FromRotationMatrix(dcm));
}

/// 500 km orbit with the geomagnetic field on — unlike the transport tests, the
/// plant here has to produce a *meaningful* field and sun geometry, because the
/// estimator's TRIAD solve is what is under test.
scenario::SimConfig estimationOrbit() {
  scenario::SimConfig c;
  c.scenario_name = "sitl-attitude-tuning";
  c.spacecraft.name = "leo-smallsat-ref";
  c.spacecraft.mass_kg = 12.0;
  c.spacecraft.inertia_kgm2 = Eigen::Vector3d(0.12, 0.12, 0.10).asDiagonal();
  c.initial_state.epoch = pt::Tai::fromNanosecondsSinceEpoch(kEpochTaiNs);
  c.initial_state.position = pm::Vec3<pm::frames::ECI>(Eigen::Vector3d(6.878137e6, 0.0, 0.0));
  c.initial_state.velocity = pm::Vec3<pm::frames::ECI>(Eigen::Vector3d(0.0, 7612.0, 0.0));
  c.initial_state.attitude = sunPointingAttitude();
  // Near-inertially-fixed: the sun stays in the sensor's 60 deg field for the
  // whole run, so an acquisition failure means the estimator, not the geometry.
  c.initial_state.body_rate = pm::Vec3<pm::frames::Body>(Eigen::Vector3d(0.0, 0.0, 1.0e-3));
  c.environment.gravity_degree = -1;  // two-body: the plant is not under test
  c.environment.magnetic_field = scenario::MagneticModel::kIgrf;
  c.environment.drag_enabled = false;
  c.environment.srp_enabled = false;
  // §5.3 disturbance torques off: the near-inertial hold this test depends on is a
  // property of the initial rate, and a disturbance torque would slowly turn the
  // sun out of the sensor's field for reasons that have nothing to do with the
  // estimator under test.
  c.environment.gravity_gradient_torque_enabled = false;
  c.environment.aero_torque_enabled = false;
  c.environment.srp_torque_enabled = false;
  c.environment.residual_dipole_torque_enabled = false;
  c.propagation.duration_s = kDurationS;
  c.propagation.output_step_s = kDurationS;
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

/// The coarse-attitude suite of config/spacecraft/leo_smallsat.yaml, with the
/// same hardware-library parameters the tuning in that file was derived from:
/// STIM300 gyro, GomSpace NanoSense FSS, generic magnetometer, NovAtel GNSS.
/// The numbers must stay in step with config/hardware/ — tuning derived from
/// one unit and validated against another proves nothing.
scenario::SpacecraftConfig estimationSuite() {
  scenario::SpacecraftConfig sc;
  sc.sensors.push_back(unit("imu_a", "STIM300", "imu",
                            {{"gyro_range_deg_s", 480.0},
                             {"gyro_arw_deg_sqrt_hr", 0.15},
                             {"gyro_bias_instability_deg_hr", 0.3},
                             {"gyro_bias_correlation_s", 100.0},
                             {"sample_rate_hz", 100.0}}));
  sc.sensors.push_back(unit("ss_zp", "GS-NANOSENSE-FSS", "sun_sensor",
                            {{"half_fov_deg", 60.0},
                             {"accuracy_inner_half_angle_deg", 45.0},
                             {"accuracy_inner_deg_3sigma", 0.5},
                             {"accuracy_outer_deg_3sigma", 2.0},
                             {"albedo_error_deg", 12.0},
                             {"update_rate_hz", 100.0},
                             {"sun_present_threshold", 0.05}}));
  sc.sensors.push_back(unit("mag_a", "MAG-GENERIC", "magnetometer",
                            {{"range_ut", 100.0},
                             {"bias_ut", 1.0},
                             {"noise_ut_rms", 0.05},
                             {"resolution_nt", 10.0},
                             {"scale_factor_pct", 1.0},
                             {"misalignment_mrad", 5.0}}));
  sc.sensors.push_back(unit("gps_a", "NOVATEL-OEM7600", "gnss",
                            {{"horizontal_position_rms_m", 1.2},
                             {"velocity_accuracy_m_s_rms", 0.03},
                             {"max_rate_hz", 10.0}}));
  return sc;
}

io::SitlServer::Counts estimationCounts() {
  io::SitlServer::Counts counts;
  counts.imu = 1;
  counts.sun_sensor = 1;
  counts.magnetometer = 1;
  counts.gnss = 1;
  return counts;
}

/// Run the config compiler over the reference vehicle, writing PrmDb.dat (and
/// the JSON artifacts) into @p out_dir. Returns the exit status; diagnostics go
/// to @p err_path. The caller has already established the interpreter exists, so
/// a nonzero status here is a real failure (a config or emitter regression), not
/// a missing toolchain — it must not be swallowed as a skip.
int compileConfig(const std::string& out_dir, const std::string& err_path) {
  std::ostringstream cmd;
  // configc creates out_dir itself, but the shell opens the stderr redirect
  // first, so the directory has to exist before the command runs.
  cmd << "mkdir -p '" << out_dir << "' && PYTHONPATH=tools '" << pythonPath() << "' -m configc"
      << " --config config/spacecraft/leo_smallsat.yaml"
      << " --hardware config/hardware"
      << " --dictionary '" << dictionaryPath() << "' --out '" << out_dir << "'"
      << " >/dev/null 2>'" << err_path << "'";
  return std::system(cmd.str().c_str());
}

/// Fork + exec the deployment against the SITL port, with the compiled
/// parameter file, logging its event stream to @p log_path.
pid_t spawnFsw(const std::string& bin, std::uint16_t port, const std::string& prm_path,
               const std::string& log_path) {
  const pid_t pid = ::fork();
  if (pid == 0) {
    ::freopen(log_path.c_str(), "w", stdout);
    ::freopen("/dev/null", "w", stderr);
    const std::string port_str = std::to_string(port);
    ::execl(bin.c_str(), bin.c_str(), "-s", port_str.c_str(), "-P", prm_path.c_str(), "-Y",
            kEpochDecimalYear, static_cast<char*>(nullptr));
    _exit(127);  // exec failed
  }
  return pid;
}

void reapFsw(pid_t pid) {
  ::kill(pid, SIGTERM);
  int status = 0;
  if (::waitpid(pid, &status, WNOHANG) == 0) {
    ::usleep(500 * 1000);
    if (::waitpid(pid, &status, WNOHANG) == 0) {
      ::kill(pid, SIGKILL);
      ::waitpid(pid, &status, 0);
    }
  }
}

std::string readFile(const std::string& path) {
  std::ifstream in(path);
  std::ostringstream text;
  text << in.rdbuf();
  return text.str();
}

TEST(SitlAttitudeTuning, CompiledParametersLetTheEstimatorAcquireAttitude) {
  const std::string bin = fswBinaryPath();
  if (::access(bin.c_str(), X_OK) != 0) {
    GTEST_SKIP() << "flight binary not built at " << bin
                 << " (run `uv run fprime-util build`, or set POLARIS_FSW_BIN)";
  }
  if (::access(pythonPath().c_str(), X_OK) != 0) {
    GTEST_SKIP() << "no Python interpreter at " << pythonPath()
                 << " (run `uv sync`, or set POLARIS_PYTHON)";
  }

  // Per-process working directory: concurrent ctest jobs must not share the
  // parameter file or the captured log.
  const std::string work_dir = "build-artifacts/test-attitude-tuning-" + std::to_string(::getpid());
  const std::string err_path = work_dir + "/configc.err";
  const std::string prm_path = work_dir + "/PrmDb.dat";
  const std::string log_path = work_dir + "/fsw.log";
  ASSERT_EQ(compileConfig(work_dir, err_path), 0) << "config compiler failed:\n"
                                                  << readFile(err_path);
  ASSERT_EQ(::access(prm_path.c_str(), R_OK), 0) << "configc produced no " << prm_path;

  io::SitlServer server(estimationCounts(), 100'000'000LL);
  ASSERT_TRUE(server.start(0)) << server.lastError();

  const pid_t pid = spawnFsw(bin, server.port(), prm_path, log_path);
  ASSERT_GE(pid, 0);

  scenario::Vehicle vehicle;
  std::string error;
  ASSERT_TRUE(scenario::buildVehicle(estimationSuite(), 1, vehicle, &error)) << error;
  scenario::SimRunner runner;
  io::ClosedLoop loop(runner, vehicle);
  ASSERT_TRUE(runner.build(estimationOrbit(), scenario::DataPaths::under(POLARIS_GOLDEN_DIR),
                           &error, loop.wrench()))
      << error;
  std::vector<io::MacroSample> trace;
  ASSERT_TRUE(loop.run(server.callback(), &trace, &error)) << error;
  EXPECT_TRUE(server.healthy()) << server.lastError();
  EXPECT_EQ(server.stepsExchanged(), 100u);
  server.stop();  // sends SHUTDOWN
  reapFsw(pid);

  const std::string log = readFile(log_path);
  ASSERT_FALSE(log.empty()) << "no event stream captured at " << log_path;

  // 1. Every record the compiler emitted reached the database. A partial load
  //    is as bad as none: the estimator refuses on the first missing value.
  EXPECT_NE(log.find("PrmFileLoadComplete"), std::string::npos)
      << "prmDb never loaded the compiled parameter file:\n"
      << log;
  EXPECT_NE(log.find("Records: 19"), std::string::npos)
      << "prmDb loaded a record count other than the 19 declared parameters:\n"
      << log;

  // 2. The estimator accepted the whole tuning set — both gates. ConfigInvalid
  //    is the edge-gated refusal Push 40 emits when a coarse parameter is
  //    missing or out of range; FineConfigInvalid is its non-fatal fine-mode
  //    counterpart, and a vehicle stuck on the coarse floor because its MEKF
  //    tuning never arrived is exactly the silent degradation to catch here.
  EXPECT_EQ(log.find("configuration invalid"), std::string::npos)
      << "estimator refused the compiled tuning:\n"
      << log;
  EXPECT_EQ(log.find("running coarse-only"), std::string::npos)
      << "estimator refused the compiled fine-mode tuning:\n"
      << log;

  // 3. Configured *and* working: a TRIAD was accepted and the attitude left the
  //    INVALID mode (REQ-ADET-004).
  EXPECT_NE(log.find("Attitude acquired"), std::string::npos)
      << "estimator never acquired an attitude over " << kDurationS << " s:\n"
      << log;
  EXPECT_EQ(log.find("Attitude lost"), std::string::npos)
      << "estimator lost the attitude it had acquired:\n"
      << log;

  // 4. And the fine mode engaged off a Davenport seed and stayed engaged: the
  //    arbitration clause of REQ-ADET-004, asserted on the vehicle rather than
  //    in the component harness. A demotion inside a 10 s run with good geometry
  //    and both sensors healthy would mean the flight tuning is wrong.
  EXPECT_NE(log.find("Fine mode engaged"), std::string::npos)
      << "estimator never promoted to fine mode over " << kDurationS << " s:\n"
      << log;
  EXPECT_EQ(log.find("Fine mode demoted"), std::string::npos)
      << "estimator could not hold the fine solution it acquired:\n"
      << log;
}

}  // namespace
