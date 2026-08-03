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

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <Eigen/Geometry>
#include <fstream>
#include <limits>
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
/// @p magCalSamples > 0 additionally commands MAG_CAL_START for that many
/// samples at startup (`-M`), which is how the calibration test gets a command
/// into a deployment with no ground link attached.
pid_t spawnFsw(const std::string& bin, std::uint16_t port, const std::string& prm_path,
               const std::string& log_path, unsigned magCalSamples = 0) {
  const pid_t pid = ::fork();
  if (pid == 0) {
    ::freopen(log_path.c_str(), "w", stdout);
    ::freopen("/dev/null", "w", stderr);
    const std::string port_str = std::to_string(port);
    const std::string cal_str = std::to_string(magCalSamples);
    ::execl(bin.c_str(), bin.c_str(), "-s", port_str.c_str(), "-P", prm_path.c_str(), "-Y",
            kEpochDecimalYear, "-M", cal_str.c_str(), static_cast<char*>(nullptr));
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

/// Occurrences of @p needle in @p text. Event streams are the only channel back
/// from the deployment, so "fired once" and "fired every cycle" are told apart
/// by counting rather than by finding.
std::size_t countOf(const std::string& text, const std::string& needle) {
  std::size_t n = 0;
  for (std::size_t at = text.find(needle); at != std::string::npos;
       at = text.find(needle, at + 1)) {
    ++n;
  }
  return n;
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
  EXPECT_NE(log.find("Records: 37"), std::string::npos)
      << "prmDb loaded a record count other than the 37 declared parameters:\n"
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
  // The third gate (Push 47): a vehicle whose sun measurements silently carry
  // their full Earth albedo because the correction's tuning never arrived is the
  // same silent degradation, one budget term down.
  EXPECT_EQ(log.find("running uncorrected"), std::string::npos)
      << "estimator refused the compiled albedo-correction tuning:\n"
      << log;

  // 2b. The sun *reference* term (Push 48). This deployment loads the DE440
  //     Chebyshev tables at setup and their coverage spans the scenario epoch,
  //     so the ephemeris query answers at grade PRECISE and the estimator
  //     composes the sun systematic with `SigmaSunEphemPreciseRad` — the
  //     arcsecond-class term — rather than the 7 mrad analytic fallback. What
  //     pins that here is the *absence* of a degrade on the EPHEMERIS domain:
  //     the grade is what selects the term, so a run that quietly fell back to
  //     the analytic ephemeris would say so here and nowhere else.
  //
  //     The match is on the format prefix, so it is **domain-agnostic**: an EOP
  //     degrade would trip it too. That is deliberate and it errs the safe way —
  //     both domains are served by the same uploaded tables and both are
  //     expected PRECISE here, so a degrade on either is worth failing on, and
  //     the worst case is an over-strict test rather than one that passes while
  //     the ephemeris quietly fell back. Narrowing to the domain would mean
  //     matching F´'s rendering of an enum argument in the text log, which is a
  //     formatting detail this test should not be pinned to. The domain
  //     *argument* is asserted where it can be read structurally: the
  //     AttitudeEstimator component test `SunSigmaFollowsTheEphemerisGrade`.
  //
  //     Asserting on the reported covariance instead would be the more direct
  //     test and is deliberately not done: the two terms differ by 4% in the
  //     composed systematic on the acquisition cycle (which is uncorrected —
  //     there is no attitude yet to place the Earth with), and a threshold
  //     inside that gap is a flaky test rather than a strong one. The
  //     tables-versus-fallback covariance difference *is* asserted, on a
  //     controlled pair of components, by the AttitudeEstimator component test
  //     `SunSigmaFollowsTheEphemerisGrade`.
  //
  //     A scenario that let the tables *expire* mid-run would add the transition
  //     — ReferenceDegraded, the sigma widening, ReferenceRecovered on re-upload
  //     — and needs an ephemeris file whose coverage ends inside the run; that
  //     is a fixture this test does not have and Push 48 did not build.
  EXPECT_EQ(log.find("Attitude reference degraded"), std::string::npos)
      << "the onboard ephemeris did not serve PRECISE for the whole run, so the sun "
         "reference fell back to the analytic ephemeris:\n"
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

// ----------------------------------------------------------------------
// Commanded magnetometer calibration (design doc §8.1)
// ----------------------------------------------------------------------

/// Long enough for a 4500-sample window at 10 Hz plus the cycles that prove the
/// estimator survived applying the result. The orbital arc matters as much as
/// the duration: the ellipsoid fit needs the IGRF *magnitude* to vary across the
/// window, and a vehicle parked at one point sees it constant, which leaves the
/// quadric's diagonal terms degenerate with its constant term and the fit
/// refused on CONDITION. 500 s covers ~31 deg of arc, enough to separate them —
/// a first attempt at 300 s (~19 deg) with the tuning test's axisymmetric
/// inertia was refused on COVERAGE at 0.25.
constexpr double kCalDurationS = 500.0;

/// Samples commanded, i.e. 450 s of the 500 s run. The remainder is the vehicle
/// flying with the correction applied.
constexpr unsigned kCalSamples = 4500;

/// Tumble driving the collection window, ~3.7 deg/s: a plausible post-separation
/// rate, and about all three axes. A torque-free body spins about a nearly fixed
/// inertial axis, so the body-frame field direction sweeps a *band* rather than
/// the sphere; the orbital motion turning the field in inertial space over 500 s
/// is what widens that band past the coverage gate. Both are needed.
const Eigen::Vector3d kCalBodyRate(0.050, 0.040, 0.030);

/// Same plant as the tuning test, tumbling, and long enough to fly the window.
scenario::SimConfig calibrationOrbit() {
  scenario::SimConfig c = estimationOrbit();
  c.scenario_name = "sitl-mag-calibration";
  // Fully asymmetric inertia, unlike the tuning test's axisymmetric pair: with
  // two equal moments the free motion is pure coning about the symmetry axis and
  // the swept band is narrow. Distinct moments make the polhode precess, which
  // widens it. Still a 6U-class 12 kg vehicle.
  c.spacecraft.inertia_kgm2 = Eigen::Vector3d(0.12, 0.09, 0.06).asDiagonal();
  c.initial_state.body_rate = pm::Vec3<pm::frames::Body>(kCalBodyRate);
  c.propagation.duration_s = kCalDurationS;
  c.propagation.output_step_s = kCalDurationS;
  return c;
}

/// First floating-point number following @p key in @p text, or NaN.
double valueAfter(const std::string& text, const std::string& key) {
  const std::size_t at = text.find(key);
  if (at == std::string::npos) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  return std::strtod(text.c_str() + at + key.size(), nullptr);
}

TEST(SitlMagCalibration, CommandedCalibrationCollectsFitsAndApplies) {
  const std::string bin = fswBinaryPath();
  if (::access(bin.c_str(), X_OK) != 0) {
    GTEST_SKIP() << "flight binary not built at " << bin
                 << " (run `uv run fprime-util build`, or set POLARIS_FSW_BIN)";
  }
  if (::access(pythonPath().c_str(), X_OK) != 0) {
    GTEST_SKIP() << "no Python interpreter at " << pythonPath()
                 << " (run `uv sync`, or set POLARIS_PYTHON)";
  }

  const std::string work_dir = "build-artifacts/test-mag-calibration-" + std::to_string(::getpid());
  const std::string err_path = work_dir + "/configc.err";
  const std::string prm_path = work_dir + "/PrmDb.dat";
  const std::string log_path = work_dir + "/fsw.log";
  ASSERT_EQ(compileConfig(work_dir, err_path), 0) << "config compiler failed:\n"
                                                  << readFile(err_path);

  io::SitlServer server(estimationCounts(), 100'000'000LL);
  ASSERT_TRUE(server.start(0)) << server.lastError();

  // -M commands MAG_CAL_START at setup. On a flight vehicle this command comes
  // from the ground; the deployment here has no uplink, and the point of the
  // test is the collect-fit-apply chain, not the radio.
  const pid_t pid = spawnFsw(bin, server.port(), prm_path, log_path, kCalSamples);
  ASSERT_GE(pid, 0);

  scenario::Vehicle vehicle;
  std::string error;
  ASSERT_TRUE(scenario::buildVehicle(estimationSuite(), 1, vehicle, &error)) << error;
  scenario::SimRunner runner;
  io::ClosedLoop loop(runner, vehicle);
  ASSERT_TRUE(runner.build(calibrationOrbit(), scenario::DataPaths::under(POLARIS_GOLDEN_DIR),
                           &error, loop.wrench()))
      << error;
  std::vector<io::MacroSample> trace;
  ASSERT_TRUE(loop.run(server.callback(), &trace, &error)) << error;
  EXPECT_TRUE(server.healthy()) << server.lastError();
  server.stop();
  reapFsw(pid);

  const std::string log = readFile(log_path);
  ASSERT_FALSE(log.empty()) << "no event stream captured at " << log_path;

  // 1. The command was accepted and a window opened for what was asked.
  EXPECT_NE(log.find("Magnetometer calibration started"), std::string::npos)
      << "MAG_CAL_START never reached the estimator:\n"
      << log;

  // 2. The fit was accepted. A refusal here names its own gate, so the message
  //    is the diagnosis — print the log rather than guessing at one.
  EXPECT_EQ(log.find("Magnetometer calibration refused"), std::string::npos)
      << "the flight tuning refused a fit over a full tumbling window:\n"
      << log;
  ASSERT_NE(log.find("Magnetometer calibration applied"), std::string::npos)
      << "no calibration was applied over " << kCalDurationS << " s:\n"
      << log;

  // 3. And it is a *good* fit. The uncalibrated magnetic systematic on this
  //    suite is 1.9 deg (34 mrad); 5 mrad is an order of magnitude better and
  //    inside the 0.2-0.5 deg class §8.1 commits the calibration at. The floor
  //    is the magnetometer's own noise, 0.05 uT on a ~30 uT field = 1.7 mrad.
  const double residual = valueAfter(log, "residual=");
  const double coverage = valueAfter(log, "coverage=");
  EXPECT_TRUE(std::isfinite(residual)) << "no residual in the completion event:\n" << log;
  EXPECT_LT(residual, 5.0e-3) << "calibration residual " << residual << " rad is too large:\n"
                              << log;
  EXPECT_GE(coverage, 0.35) << "the tumble did not span enough field directions:\n" << log;

  // 4. The estimator is still healthy with the correction applied: it acquired,
  //    it promoted, and nothing about the calibration knocked either over. Note
  //    the seed-observability caveat in §8.1 does not bite here — the *sigma*
  //    parameters still describe an uncalibrated magnetometer, which is exactly
  //    the state a vehicle is in between a good fit and the ground uplinking
  //    re-derived values, and the vehicle has to fly through it.
  EXPECT_EQ(log.find("configuration invalid"), std::string::npos) << log;
  EXPECT_NE(log.find("Attitude acquired"), std::string::npos)
      << "estimator never acquired an attitude:\n"
      << log;
  EXPECT_NE(log.find("Fine mode engaged"), std::string::npos)
      << "estimator never promoted to fine mode:\n"
      << log;
}

// ----------------------------------------------------------------------
// Multi-IMU voting and its FDIR response (design doc §8.2, §9.2)
// ----------------------------------------------------------------------

/// Redundant gyro suite: two identical STIM300s, the reference vehicle's, in the
/// order `config/spacecraft/leo_smallsat.yaml` declares them — so the port index
/// the FDIR events name is the same index there.
scenario::SpacecraftConfig votingSuite() {
  scenario::SpacecraftConfig sc = estimationSuite();
  // By value: push_back reallocates, and a reference into the vector would dangle.
  scenario::UnitConfig second = sc.sensors.front();
  second.name = "imu_b";
  sc.sensors.push_back(second);
  return sc;
}

io::SitlServer::Counts votingCounts() {
  io::SitlServer::Counts counts = estimationCounts();
  counts.imu = 2;
  return counts;
}

/// Long enough to fly healthy, break a unit, and recover it: the re-admission
/// policy is 10 consecutive plausible cycles (1 s at 10 Hz) on the flight
/// tuning, so a few seconds either side of each transition is plenty.
constexpr double kVotingDurationS = 30.0;
constexpr unsigned kFaultStep = 100;    ///< 10 s in, well past fine-mode promotion
constexpr unsigned kRecoverStep = 200;  ///< 20 s in, 10 s of running degraded

/// Gyro bias jump injected into one unit [rad/s]: 0.05 = 2.9 deg/s, about six
/// times the pairwise disagreement gate (0.0087) and a tenth of the plausibility
/// limit (0.5236), so the fault is invisible to every per-unit gate and visible
/// only to the pair.
constexpr double kImuFaultRadps = 0.05;

/// `flight.attitudeEstimator.ImuAmbiguityEscalateCycles` in the reference
/// vehicle config, mirrored so the expected escalation count is derived rather
/// than pasted. A change to the flight value fails this test loudly, which is
/// the intent — the cadence is the operator-facing contract.
constexpr std::size_t kFlightEscalateCycles = 600;

scenario::SimConfig votingOrbit() {
  scenario::SimConfig c = estimationOrbit();
  c.scenario_name = "sitl-imu-voting";
  c.propagation.duration_s = kVotingDurationS;
  c.propagation.output_step_s = kVotingDurationS;
  return c;
}

/// The ambiguity case runs longer than the identification one, and has to:
/// asserting a *cadence* needs at least two escalations, so the run must span
/// two 600-cycle horizons at the flight tuning. 130 s = 1300 cycles gives
/// exactly two, with the second well clear of the boundary.
constexpr double kAmbiguityDurationS = 130.0;

scenario::SimConfig ambiguityOrbit() {
  scenario::SimConfig c = votingOrbit();
  c.scenario_name = "sitl-imu-ambiguity";
  c.propagation.duration_s = kAmbiguityDurationS;
  c.propagation.output_step_s = kAmbiguityDurationS;
  return c;
}

/// The §8.2/§9.2 fault path **the reference vehicle actually flies**: with two
/// IMUs there is no median to hide behind, so a disagreement has to be detected
/// on the pair and then *attributed* by a third information source — the MEKF's
/// propagated body rate. One unit is faulted mid-run, the estimator must
/// identify and isolate it, keep publishing an attitude across the transition,
/// and re-admit the unit when it recovers.
///
/// **The fault is deliberately a plausible one.** 0.05 rad/s (2.9 deg/s) is about
/// six times the pairwise disagreement gate and a tenth of the 30 deg/s
/// plausibility limit, so no per-unit gate can see it and the only thing that
/// can is the comparison — which is the case a two-unit suite exists to handle
/// and the one a mean would silently split the difference on. A hard-railed unit
/// is the *easier* fault (the rate-limit gate catches it before the pair is ever
/// consulted) and is covered in the component harness and `imu_voting_test.cpp`;
/// this test spends the process on the harder one.
///
/// Verifies REQ-ADET-008 and REQ-ADET-009 on the vehicle.
TEST(SitlImuVoting, DisagreeingImuIsIdentifiedAndTheAttitudeSurvives) {
  const std::string bin = fswBinaryPath();
  if (::access(bin.c_str(), X_OK) != 0) {
    GTEST_SKIP() << "flight binary not built at " << bin
                 << " (run `uv run fprime-util build`, or set POLARIS_FSW_BIN)";
  }
  if (::access(pythonPath().c_str(), X_OK) != 0) {
    GTEST_SKIP() << "no Python interpreter at " << pythonPath()
                 << " (run `uv sync`, or set POLARIS_PYTHON)";
  }

  const std::string work_dir = "build-artifacts/test-imu-voting-" + std::to_string(::getpid());
  const std::string err_path = work_dir + "/configc.err";
  const std::string prm_path = work_dir + "/PrmDb.dat";
  const std::string log_path = work_dir + "/fsw.log";
  ASSERT_EQ(compileConfig(work_dir, err_path), 0) << "config compiler failed:\n"
                                                  << readFile(err_path);

  io::SitlServer server(votingCounts(), 100'000'000LL);
  ASSERT_TRUE(server.start(0)) << server.lastError();

  const pid_t pid = spawnFsw(bin, server.port(), prm_path, log_path);
  ASSERT_GE(pid, 0);

  scenario::Vehicle vehicle;
  std::string error;
  ASSERT_TRUE(scenario::buildVehicle(votingSuite(), 1, vehicle, &error)) << error;
  ASSERT_EQ(vehicle.imus.size(), 2u);
  scenario::SimRunner runner;
  io::ClosedLoop loop(runner, vehicle);
  ASSERT_TRUE(runner.build(votingOrbit(), scenario::DataPaths::under(POLARIS_GOLDEN_DIR), &error,
                           loop.wrench()))
      << error;

  // Fault injection on the macro-step seam: the callback runs once per exchanged
  // step, so incrementing here counts steps and the injected bias takes effect
  // from the next sample onwards. 0.05 rad/s is individually plausible and only
  // the pair can see it — see the note above.
  unsigned step = 0;
  const io::FswCallback inner = server.callback();
  const io::FswCallback faulted = [&](const io::FswInputs& in) {
    if (step == kFaultStep) {
      vehicle.imus[1].model.injectGyroBiasJump(
          pm::Vec3<pm::frames::Body>(kImuFaultRadps, 0.0, 0.0));
    }
    if (step == kRecoverStep) {
      vehicle.imus[1].model.clearFaults();
    }
    ++step;
    return inner(in);
  };

  std::vector<io::MacroSample> trace;
  ASSERT_TRUE(loop.run(faulted, &trace, &error)) << error;
  EXPECT_TRUE(server.healthy()) << server.lastError();
  server.stop();
  reapFsw(pid);

  const std::string log = readFile(log_path);
  ASSERT_FALSE(log.empty()) << "no event stream captured at " << log_path;

  // 1. Healthy first: the run has to have been working before the fault, or the
  //    rest of the assertions are about a vehicle that never flew.
  ASSERT_NE(log.find("Attitude acquired"), std::string::npos)
      << "estimator never acquired an attitude before the fault:\n"
      << log;
  ASSERT_NE(log.find("Fine mode engaged"), std::string::npos)
      << "estimator never promoted to fine mode before the fault:\n"
      << log;

  // 2. The fault was detected AND attributed. Attribution is the whole point on a
  //    two-unit suite: the pair can only say "one of us is lying", so the reason
  //    has to read OUTVOTED — the MEKF's propagated rate is what named the unit.
  //    A RATE_LIMIT here would mean the injected fault was too crude to exercise
  //    this path, and an ImuVoteAmbiguous would mean the tie-break was not
  //    available when it should have been.
  EXPECT_NE(log.find("IMU 1 excluded from the rate vote"), std::string::npos)
      << "the disagreeing IMU was never excluded:\n"
      << log;
  EXPECT_NE(log.find("OUTVOTED"), std::string::npos)
      << "the exclusion did not come from the filter's identification — a two-unit "
         "disagreement was detected but not attributed:\n"
      << log;
  EXPECT_EQ(log.find("Two IMUs disagree"), std::string::npos)
      << "the disagreement went unattributed while a fine solution was available:\n"
      << log;
  // Exactly one exclusion over the whole 10 s the fault is held. More than one
  // is the C1 flap: an outvoted unit re-admitting itself on plausibility alone,
  // losing the same comparison, and re-excluding — forever, at the re-admission
  // period.
  EXPECT_EQ(countOf(log, "excluded from the rate vote"), 1u)
      << "the sustained fault was reported more than once: the unit flapped\n"
      << log;

  // 3. It cost nothing that matters. This is the single-fault-survival claim: the
  //    surviving unit carried the rate through, so the attitude solution never
  //    went invalid and the filter never lost confidence in its own measurements.
  EXPECT_EQ(log.find("Attitude lost"), std::string::npos)
      << "a single faulted IMU cost the attitude solution:\n"
      << log;
  EXPECT_EQ(log.find("Fine mode demoted"), std::string::npos)
      << "a single faulted IMU demoted the fine solution — note the circularity "
         "this would expose: the filter that identified the fault is the one that "
         "then has to survive it:\n"
      << log;
  // And the healthy unit was not blamed for it.
  EXPECT_EQ(log.find("IMU 0 excluded"), std::string::npos) << log;

  // 4. Recovery is automatic, and it happens (§9.2): the unit behaved for
  //    ImuReadmitCycles consecutive cycles and came back. Spending a unit of
  //    redundancy permanently on a transient is the more expensive error, which
  //    is why the policy is hysteresis rather than a command.
  EXPECT_NE(log.find("IMU 1 re-admitted to the rate vote"), std::string::npos)
      << "the recovered IMU was never re-admitted:\n"
      << log;
}

/// The other half of the two-unit story, witnessed on the vehicle rather than
/// only in unit tests: the **deliberate non-monotonicity**. Faulting a unit from
/// the first step means the disagreement is present before the MEKF has a rate
/// to be believed against, so there is nothing to attribute it with — and the
/// vote publishes **no rate at all**, where a *single* unit in exactly the same
/// state would have been passed straight through.
///
/// That is chosen, not fallen into: one unit carries no evidence of a fault,
/// while two disagreeing units carry evidence and no attribution, and a
/// possibly-wrong rate propagates into the attitude where the coast does not.
/// What this run pins is that the cost is bounded to the *rate* — the vector
/// pairs still acquire an attitude — and that nothing is latched, because
/// latching the wrong unit is worse than carrying the disagreement.
TEST(SitlImuVoting, UnattributableDisagreementRefusesTheRateAndKeepsTheAttitude) {
  const std::string bin = fswBinaryPath();
  if (::access(bin.c_str(), X_OK) != 0) {
    GTEST_SKIP() << "flight binary not built at " << bin
                 << " (run `uv run fprime-util build`, or set POLARIS_FSW_BIN)";
  }
  if (::access(pythonPath().c_str(), X_OK) != 0) {
    GTEST_SKIP() << "no Python interpreter at " << pythonPath()
                 << " (run `uv sync`, or set POLARIS_PYTHON)";
  }

  const std::string work_dir = "build-artifacts/test-imu-ambiguous-" + std::to_string(::getpid());
  const std::string err_path = work_dir + "/configc.err";
  const std::string prm_path = work_dir + "/PrmDb.dat";
  const std::string log_path = work_dir + "/fsw.log";
  ASSERT_EQ(compileConfig(work_dir, err_path), 0) << "config compiler failed:\n"
                                                  << readFile(err_path);

  io::SitlServer server(votingCounts(), 100'000'000LL);
  ASSERT_TRUE(server.start(0)) << server.lastError();

  const pid_t pid = spawnFsw(bin, server.port(), prm_path, log_path);
  ASSERT_GE(pid, 0);

  scenario::Vehicle vehicle;
  std::string error;
  ASSERT_TRUE(scenario::buildVehicle(votingSuite(), 1, vehicle, &error)) << error;
  scenario::SimRunner runner;
  io::ClosedLoop loop(runner, vehicle);
  ASSERT_TRUE(runner.build(ambiguityOrbit(), scenario::DataPaths::under(POLARIS_GOLDEN_DIR), &error,
                           loop.wrench()))
      << error;

  // Faulted from the first step, so the disagreement is in force before the
  // filter has ever published a rate — and it stays that way, because the vote
  // withholds the rate that the filter would need in order to acquire one. That
  // self-sustaining state is the point: it is what a two-IMU vehicle sits in
  // when it meets this fault outside fine mode.
  const io::FswCallback inner = server.callback();
  bool injected = false;
  const io::FswCallback faulted = [&](const io::FswInputs& in) {
    if (!injected) {
      vehicle.imus[1].model.injectGyroBiasJump(
          pm::Vec3<pm::frames::Body>(kImuFaultRadps, 0.0, 0.0));
      injected = true;
    }
    return inner(in);
  };

  std::vector<io::MacroSample> trace;
  ASSERT_TRUE(loop.run(faulted, &trace, &error)) << error;
  EXPECT_TRUE(server.healthy()) << server.lastError();
  server.stop();
  reapFsw(pid);

  const std::string log = readFile(log_path);
  ASSERT_FALSE(log.empty()) << "no event stream captured at " << log_path;

  // 1. The refusal fired, and it is the *unattributed* one — the whole
  //    distinction from the identification test above.
  EXPECT_NE(log.find("Two IMUs disagree"), std::string::npos)
      << "the unattributable disagreement was never reported:\n"
      << log;

  // 2. Edge-gated: the condition persists for the whole run, so a per-cycle
  //    event would be 300 of them. One is the contract.
  const std::size_t reports = countOf(log, "Two IMUs disagree");
  EXPECT_EQ(reports, 1u) << "ImuVoteAmbiguous is not edge-gated: " << reports << " reports in "
                         << kAmbiguityDurationS << " s\n"
                         << log;

  // 2b. ...which is exactly why the escalation exists. Edge-gating alone would
  //     leave a *permanent* fault reported once and then silent for the rest of
  //     the flight. The escalation fires at the configured horizon and repeats
  //     at that period — bounded, not per cycle.
  const std::size_t escalations = countOf(log, "unattributable for");
  const std::size_t expected =
      static_cast<std::size_t>(kAmbiguityDurationS * 10.0) / kFlightEscalateCycles;
  static_assert(static_cast<std::size_t>(kAmbiguityDurationS * 10.0) / kFlightEscalateCycles >= 2,
                "the run must span two horizons or it pins the first fire, not the cadence");
  EXPECT_GE(escalations, 1u) << "a permanently rate-less vehicle never escalated:\n" << log;
  EXPECT_EQ(escalations, expected)
      << "escalation cadence is not the configured horizon: " << escalations << " in "
      << kAmbiguityDurationS << " s, expected " << expected << "\n"
      << log;

  // 3. Nothing was latched. Detection is not attribution, and excluding the
  //    wrong unit would spend the vehicle's remaining redundancy on a guess.
  EXPECT_EQ(log.find("excluded from the rate vote"), std::string::npos)
      << "an unattributable disagreement latched an exclusion anyway:\n"
      << log;

  // 4. The cost is bounded to the body rate. The sun and magnetic pairs are
  //    untouched by a gyro fault, so TRIAD still acquires and the vehicle keeps
  //    an attitude — which is exactly why withholding the rate is the safe
  //    response rather than a self-inflicted outage.
  EXPECT_NE(log.find("Attitude acquired"), std::string::npos)
      << "withholding the rate cost the attitude as well:\n"
      << log;
  EXPECT_EQ(log.find("Attitude lost"), std::string::npos)
      << "the estimator could not hold the attitude through the rate refusal:\n"
      << log;
}

}  // namespace
