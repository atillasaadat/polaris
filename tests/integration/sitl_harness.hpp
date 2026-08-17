#ifndef POLARIS_TESTS_INTEGRATION_SITL_HARNESS_HPP
#define POLARIS_TESTS_INTEGRATION_SITL_HARNESS_HPP

/// @file
/// @brief Shared fixture for the SITL integration cases (design doc §19.3, §23.1).
///
/// Every SITL case runs the same chain — `leo_smallsat.yaml -> configc ->
/// PrmDb.dat -> the forked deployment <-> the truth sim over the SITL wire` — and
/// reads the deployment's own event stream back. That plumbing lives here so the
/// case files carry only what distinguishes them: the plant, the sensor suite,
/// the injected fault and the assertions on the mode machine's response.
///
/// It is a header of `inline` functions rather than a library because the whole
/// of it is fixture construction: the suites below must stay in step with
/// `config/spacecraft/leo_smallsat.yaml` **by index**, since the per-unit
/// parameters the compiler emits (sun and star-tracker boresights) are flat
/// arrays in port-array order, and a suite that declares its units in a different
/// order would be corrected with another unit's geometry.
///
/// Cases skip (never fail) when the flight binary or the Python toolchain is
/// absent — CI's unit-test job builds only the native-ut tree. `POLARIS_FSW_BIN`,
/// `POLARIS_PYTHON` and `POLARIS_FSW_DICT` override the defaults.

#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <Eigen/Geometry>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <utility>

#include "ephemeris/analytic_sun.hpp"
#include "io/sitl_server.hpp"
#include "math/frames.hpp"
#include "math/typed_vector.hpp"
#include "scenario/sim_config.hpp"
#include "scenario/vehicle.hpp"
#include "time/tdb.hpp"
#include "time/timescales.hpp"

namespace polaris::test::sitl {

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

inline std::string envOr(const char* name, const char* fallback) {
  const char* value = std::getenv(name);
  return (value != nullptr) ? std::string(value) : std::string(fallback);
}

inline std::string fswBinaryPath() {
  return envOr("POLARIS_FSW_BIN", "build-artifacts/Linux/flight_PolarisFsw/bin/flight_PolarisFsw");
}

/// Interpreter that can `import configc`: the uv-managed project venv by
/// default (`uv sync`), overridable for other layouts.
inline std::string pythonPath() {
  return envOr("POLARIS_PYTHON", ".venv/bin/python3");
}

inline std::string dictionaryPath() {
  return envOr("POLARIS_FSW_DICT",
               "build-artifacts/Linux/flight_PolarisFsw/dict/PolarisFswTopologyDictionary.json");
}

/// Run the config compiler over the reference vehicle, writing PrmDb.dat (and
/// the JSON artifacts) into @p out_dir. Returns the exit status; diagnostics go
/// to @p err_path. The caller has already established the interpreter exists, so
/// a nonzero status here is a real failure (a config or emitter regression), not
/// a missing toolchain — it must not be swallowed as a skip.
inline int compileConfig(const std::string& out_dir, const std::string& err_path) {
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
/// into a deployment with no ground link attached. @p ctrlMode (`-c`) and
/// @p ctrlTargetQ (`-q`) are the §8.5 equivalent for the attitude controller:
/// 1 = DETUMBLE, 2 = POINT, 0 = leave it in IDLE.
/// @p feedforward, when non-negative, forces the §8.5 disturbance-feedforward
/// tiers on (1) or off (0) with `-F model,observer`, so a row can fly the same
/// vehicle both ways and measure the difference. Negative leaves the committed
/// ParameterDb values in force, which is what every other row uses.
/// @p odResetCycle > 0 commands the orbit filter's OD_RESET on that GNC cycle
/// (`-R`), the §8.3 reset-and-reseed row's way of getting a mid-run command in.
/// @p burnSpec, when non-null, is "cycle,durationS,throttle" for `-b`: arm a
/// §17 BURN_START on that GNC cycle, the burn rows' way of firing a thruster.
inline pid_t spawnFsw(const std::string& bin, std::uint16_t port, const std::string& prm_path,
                      const std::string& log_path, unsigned magCalSamples = 0,
                      unsigned stAlignPairs = 0, unsigned stAlignUnit = 1, unsigned ctrlMode = 0,
                      const double* ctrlTargetQ = nullptr, int feedforward = -1,
                      unsigned odResetCycle = 0, const char* burnSpec = nullptr,
                      int odAccelInput = -1, int wheelBias = -1) {
  const pid_t pid = ::fork();
  if (pid == 0) {
    // A child whose log cannot be opened must not fly and report nothing: the
    // parent reads the log as the run's event stream.
    if (::freopen(log_path.c_str(), "w", stdout) == nullptr ||
        ::freopen("/dev/null", "w", stderr) == nullptr) {
      _exit(126);
    }
    const std::string port_str = std::to_string(port);
    const std::string cal_str = std::to_string(magCalSamples);
    const std::string align_str = std::to_string(stAlignUnit) + "," + std::to_string(stAlignPairs);
    const std::string ctrl_str = std::to_string(ctrlMode);
    std::ostringstream target;
    if (ctrlTargetQ != nullptr) {
      target << ctrlTargetQ[0] << "," << ctrlTargetQ[1] << "," << ctrlTargetQ[2] << ","
             << ctrlTargetQ[3];
    } else {
      target << "0,0,0,0";
    }
    const std::string target_str = target.str();
    // Both tiers move together: the rows that use this are asking "with or
    // without feedforward", not "which tier".
    const std::string ff_str =
        feedforward < 0 ? std::string("")
                        : std::to_string(feedforward) + "," + std::to_string(feedforward);
    const std::string reset_str = std::to_string(odResetCycle);
    const std::string bias_str = std::to_string(wheelBias);
    // Optional overrides are passed only when asked for, so a row that does not
    // set one flies the ParameterDb value rather than a default of ours.
    const char* argv_[32] = {
        bin.c_str(),       "-s", port_str.c_str(),   "-P", prm_path.c_str(),  "-Y",
        kEpochDecimalYear, "-M", cal_str.c_str(),    "-A", align_str.c_str(), "-c",
        ctrl_str.c_str(),  "-q", target_str.c_str(), "-R", reset_str.c_str()};
    int argc_ = 17;
    if (feedforward >= 0) {
      argv_[argc_++] = "-F";
      argv_[argc_++] = ff_str.c_str();
    }
    // `-b cycle,durationS,throttle` arms a §17 burn (see spawnFsw's doc).
    if (burnSpec != nullptr) {
      argv_[argc_++] = "-b";
      argv_[argc_++] = burnSpec;
    }
    const std::string accel_str = std::to_string(odAccelInput);
    if (odAccelInput >= 0) {
      argv_[argc_++] = "-N";
      argv_[argc_++] = accel_str.c_str();
    }
    if (wheelBias >= 0) {
      argv_[argc_++] = "-W";
      argv_[argc_++] = bias_str.c_str();
    }
    argv_[argc_] = nullptr;
    // execv takes char* const*; the strings are not modified.
    ::execv(bin.c_str(), const_cast<char* const*>(argv_));
    _exit(127);  // exec failed
  }
  return pid;
}

inline void reapFsw(pid_t pid) {
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
inline std::size_t countOf(const std::string& text, const std::string& needle) {
  std::size_t n = 0;
  for (std::size_t at = text.find(needle); at != std::string::npos;
       at = text.find(needle, at + 1)) {
    ++n;
  }
  return n;
}

inline std::string readFile(const std::string& path) {
  std::ifstream in(path);
  std::ostringstream text;
  text << in.rdbuf();
  return text.str();
}

/// Offset of @p needle in @p text, or `std::string::npos`. Named so an ordering
/// assertion ("the demotion came before the re-promotion") reads as one.
inline std::size_t indexOf(const std::string& text, const std::string& needle) {
  return text.find(needle);
}

// ----------------------------------------------------------------------
// Geometry
// ----------------------------------------------------------------------

/// Attitude placing the sun sensor boresight (unit +Z, identity mounting) on the
/// Sun at the scenario epoch, so the sun pair is available from the first cycle.
/// Derived from the analytic ephemeris rather than hard-coded: the point of most
/// of these cases is the mode machine, and a stale hand-computed quaternion would
/// turn an epoch change into a mystery failure.
inline pm::Quat<pm::frames::Body, pm::frames::ECI> sunPointingAttitude() {
  const pt::Tai epoch = pt::Tai::fromNanosecondsSinceEpoch(kEpochTaiNs);
  const Eigen::Vector3d sun =
      polaris::ephemeris::sunPositionEci(pt::toTdb(pt::toTt(epoch))).eigen().normalized();
  // Body <- ECI rotation whose third row is the sun direction, i.e. R * sun = +Z.
  const Eigen::Vector3d x = sun.unitOrthogonal();
  const Eigen::Matrix3d dcm =
      (Eigen::Matrix3d() << x.transpose(), sun.cross(x).transpose(), sun.transpose()).finished();
  return pm::Quat<pm::frames::Body, pm::frames::ECI>(pm::Quaternion::FromRotationMatrix(dcm));
}

// ----------------------------------------------------------------------
// Plants
// ----------------------------------------------------------------------

/// 500 km orbit with the geomagnetic field on — unlike the transport tests, the
/// plant here has to produce a *meaningful* field and sun geometry, because the
/// estimator's TRIAD solve is what is under test.
///
/// @p durationS the arc to fly; the caller sizes it from where its transition is
/// expected, since SITL wall time is roughly a tenth of sim time.
inline scenario::SimConfig estimationOrbit(double durationS = 10.0, int gravityDegree = 8) {
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
  // 8x8 EGM2008 by default — the same truncation the flight orbit filter
  // carries, so the filter's model error against this plant is zero and a NIS
  // rejection in a SITL row means the row, not the fixture. This read `-1`
  // ("two-body", said the comment) until Push 65's orbit estimator was the first
  // consumer to notice that -1 is *free drift*: every SITL row before it flew a
  // straight line at 7.6 km/s. Push 67 finished the move: the two control rows
  // tuned on the straight line (sun acquisition, feedforward A/B) were
  // re-baselined on the orbit — longer windows, no physics — and the
  // magnetometer-calibration residual that read 7.45 mrad on the orbit against
  // 2.43 straight turned out to be a fit defect the straight line had hidden
  // (a free constant term, `lib/gnc/mag_calibration`), now 2.23 mrad. Nothing
  // asks for free drift any more, and nothing should: the orbit filter's model
  // is gravity, so a straight line has no position. The plant is still not
  // under test; the field costs a 45-coefficient recursion per RK stage.
  c.environment.gravity_degree = gravityDegree;
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
  c.propagation.duration_s = durationS;
  c.propagation.output_step_s = durationS;
  c.propagation.fsw_rate_hz = 10.0;
  return c;
}

// ----------------------------------------------------------------------
// Suites
// ----------------------------------------------------------------------

inline scenario::UnitConfig unit(const std::string& name, const std::string& model,
                                 const std::string& kind, std::map<std::string, double> params) {
  scenario::UnitConfig u;
  u.name = name;
  u.model_id = model;
  u.kind = kind;
  u.params = std::move(params);
  return u;
}

/// Datasheet parameters of the parts the reference vehicle flies. They must stay
/// in step with `config/hardware/` — tuning derived from one unit and validated
/// against another proves nothing.
inline const std::map<std::string, double>& stim300Params() {
  static const std::map<std::string, double> p{{"gyro_range_deg_s", 480.0},
                                               {"gyro_arw_deg_sqrt_hr", 0.15},
                                               {"gyro_bias_instability_deg_hr", 0.3},
                                               {"gyro_bias_correlation_s", 100.0},
                                               {"sample_rate_hz", 100.0}};
  return p;
}

inline const std::map<std::string, double>& nanosenseFssParams() {
  static const std::map<std::string, double> p{{"half_fov_deg", 60.0},
                                               {"accuracy_inner_half_angle_deg", 45.0},
                                               {"accuracy_inner_deg_3sigma", 0.5},
                                               {"accuracy_outer_deg_3sigma", 2.0},
                                               {"albedo_error_deg", 12.0},
                                               {"update_rate_hz", 100.0},
                                               {"sun_present_threshold", 0.05}};
  return p;
}

inline const std::map<std::string, double>& genericMagParams() {
  static const std::map<std::string, double> p{
      {"range_ut", 100.0},     {"bias_ut", 1.0},          {"noise_ut_rms", 0.05},
      {"resolution_nt", 10.0}, {"scale_factor_pct", 1.0}, {"misalignment_mrad", 5.0}};
  return p;
}

inline const std::map<std::string, double>& aurigaParams() {
  static const std::map<std::string, double> p{{"lf_spatial_xy_arcsec_3sigma", 9.0},
                                               {"lf_spatial_z_arcsec_3sigma", 51.0},
                                               {"hf_spatial_xy_arcsec_3sigma", 6.6},
                                               {"hf_spatial_z_arcsec_3sigma", 38.0},
                                               {"lf_spatial_correlation_s", 300.0},
                                               {"hf_spatial_correlation_s", 10.0},
                                               {"temporal_noise_xy_arcsec_3sigma", 11.0},
                                               {"temporal_noise_z_arcsec_3sigma", 70.0},
                                               {"bias_deg", 0.017},
                                               {"thermo_elastic_arcsec_per_c", 1.5},
                                               {"acquisition_rate_deg_s", 2.0},
                                               {"tracking_rate_deg_s", 3.0},
                                               {"acquisition_accel_deg_s2", 1.0},
                                               {"tracking_accel_deg_s2", 2.5},
                                               {"lost_in_space_s", 3.8},
                                               {"update_rate_hz", 10.0},
                                               {"fov_deg", 20.0},
                                               {"sun_exclusion_deg", 35.0},
                                               {"earth_exclusion_deg", 22.0},
                                               {"moon_exclusion_deg", 0.0}};
  return p;
}

/// The coarse-attitude suite of config/spacecraft/leo_smallsat.yaml: STIM300
/// gyro, GomSpace NanoSense FSS, generic magnetometer, NovAtel GNSS — one of
/// each, at index 0, which is where the vehicle declares them.
inline scenario::SpacecraftConfig estimationSuite() {
  scenario::SpacecraftConfig sc;
  sc.sensors.push_back(unit("imu_a", "STIM300", "imu", stim300Params()));
  sc.sensors.push_back(unit("ss_zp", "GS-NANOSENSE-FSS", "sun_sensor", nanosenseFssParams()));
  sc.sensors.push_back(unit("mag_a", "MAG-GENERIC", "magnetometer", genericMagParams()));
  sc.sensors.push_back(unit("gps_a", "NOVATEL-OEM7600", "gnss",
                            {{"horizontal_position_rms_m", 1.2},
                             {"velocity_accuracy_m_s_rms", 0.03},
                             {"max_rate_hz", 10.0}}));
  return sc;
}

inline io::SitlServer::Counts estimationCounts() {
  io::SitlServer::Counts counts;
  counts.imu = 1;
  counts.sun_sensor = 1;
  counts.magnetometer = 1;
  counts.gnss = 1;
  return counts;
}

/// Rotation carrying the sensor's own +Z onto @p boresight, about the axis
/// perpendicular to both — the mounting DCM the scenario config takes, and the
/// same rotation the vehicle YAML carries as a quaternion.
inline Eigen::Matrix3d mountingOnto(const Eigen::Vector3d& boresight) {
  return Eigen::Quaterniond::FromTwoVectors(Eigen::Vector3d::UnitZ(), boresight.normalized())
      .toRotationMatrix();
}

/// The reference vehicle's two AURIGA star trackers on top of a suite, with the
/// mountings `config/spacecraft/leo_smallsat.yaml` flies: boresights at
/// (∓1, 0, −1)/√2, i.e. 90 deg apart and both 135 deg from the payload/array face
/// (§8.2). st_a is the king (port index 0), whose mounting defines the frame.
inline scenario::SpacecraftConfig withTrackers(scenario::SpacecraftConfig sc) {
  const double s = 1.0 / std::sqrt(2.0);
  scenario::UnitConfig king = unit("st_a", "AURIGA", "star_tracker", aurigaParams());
  king.mounting_dcm = mountingOnto(Eigen::Vector3d(-s, 0.0, -s));
  scenario::UnitConfig second = unit("st_b", "AURIGA", "star_tracker", aurigaParams());
  second.mounting_dcm = mountingOnto(Eigen::Vector3d(s, 0.0, -s));
  sc.sensors.push_back(king);
  sc.sensors.push_back(second);
  return sc;
}

inline scenario::SpacecraftConfig trackerSuite() {
  return withTrackers(estimationSuite());
}

inline io::SitlServer::Counts trackerCounts() {
  io::SitlServer::Counts counts = estimationCounts();
  counts.star_tracker = 2;
  return counts;
}

/// Redundant gyro suite: two identical STIM300s, the reference vehicle's, in the
/// order `config/spacecraft/leo_smallsat.yaml` declares them — so the port index
/// the FDIR events name is the same index there.
inline scenario::SpacecraftConfig votingSuite() {
  scenario::SpacecraftConfig sc = estimationSuite();
  // By value: push_back reallocates, and a reference into the vector would dangle.
  scenario::UnitConfig second = sc.sensors.front();
  second.name = "imu_b";
  sc.sensors.push_back(second);
  return sc;
}

inline io::SitlServer::Counts votingCounts() {
  io::SitlServer::Counts counts = estimationCounts();
  counts.imu = 2;
  return counts;
}

// ----------------------------------------------------------------------
// The fault-injection matrix fixture (§9, §23.1.1)
// ----------------------------------------------------------------------

/// Port index of each unit in the @ref faultMatrixSuite, which is the index every
/// FDIR event names. They are the reference vehicle's own indices — the suite is
/// declared in `config/spacecraft/leo_smallsat.yaml` order, because the per-unit
/// parameters (sun and star-tracker boresights) are flat arrays in that order.
enum : int {
  kImuA = 0,
  kImuB = 1,
  kSsZp = 0,  ///< +Z, the solar-array normal
  kSsZm = 1,
  kSsXp = 2,  ///< +X — the unit **selected** in the matrix geometry
  kSsXm = 3,
  kSsYp = 4,  ///< +Y — the **runner-up**, at the same incidence
  kSsYm = 5,
  kMagA = 0,
  kMagB = 1,
  kStA = 0,  ///< the king: its mounting defines the body frame
  kStB = 1,
};

/// The reference vehicle's whole attitude suite: two STIM300s, six FSS units on
/// the six faces, two magnetometers, one GNSS receiver and the two AURIGA
/// trackers — every unit `config/spacecraft/leo_smallsat.yaml` flies, at the
/// index it flies at.
///
/// One suite for the entire fault matrix, so a case differs from its neighbours
/// only in the fault it injects. A case that needs a unit *absent* takes it away
/// with a dropout rather than with a second suite: "no star tracker" and "both
/// star trackers failed" are the same state to the estimator, and building the
/// second one out of the same fixture is what keeps the matrix comparable.
inline scenario::SpacecraftConfig faultMatrixSuite() {
  scenario::SpacecraftConfig sc;
  sc.sensors.push_back(unit("imu_a", "STIM300", "imu", stim300Params()));
  sc.sensors.push_back(unit("imu_b", "STIM300", "imu", stim300Params()));

  // Six faces, in port order. A vector-output part is insensitive to roll about
  // its boresight (it reports a direction, not a frame), so the boresight is the
  // whole of the mounting that matters here — which is why these are derived from
  // the face normals rather than transcribed from the vehicle's quaternions.
  const char* const names[6] = {"ss_zp", "ss_zm", "ss_xp", "ss_xm", "ss_yp", "ss_ym"};
  const Eigen::Vector3d faces[6] = {Eigen::Vector3d::UnitZ(), -Eigen::Vector3d::UnitZ(),
                                    Eigen::Vector3d::UnitX(), -Eigen::Vector3d::UnitX(),
                                    Eigen::Vector3d::UnitY(), -Eigen::Vector3d::UnitY()};
  for (int i = 0; i < 6; ++i) {
    scenario::UnitConfig u = unit(names[i], "GS-NANOSENSE-FSS", "sun_sensor", nanosenseFssParams());
    u.mounting_dcm = mountingOnto(faces[i]);
    sc.sensors.push_back(u);
  }

  // Identically mounted: redundant units, not a skewed array (§8.2).
  sc.sensors.push_back(unit("mag_a", "MAG-GENERIC", "magnetometer", genericMagParams()));
  sc.sensors.push_back(unit("mag_b", "MAG-GENERIC", "magnetometer", genericMagParams()));
  sc.sensors.push_back(unit("gps_a", "NOVATEL-OEM7600", "gnss",
                            {{"horizontal_position_rms_m", 1.2},
                             {"velocity_accuracy_m_s_rms", 0.03},
                             {"max_rate_hz", 10.0}}));
  return withTrackers(std::move(sc));
}

inline io::SitlServer::Counts faultMatrixCounts() {
  io::SitlServer::Counts counts;
  counts.imu = 2;
  counts.sun_sensor = 6;
  counts.magnetometer = 2;
  counts.gnss = 1;
  counts.star_tracker = 2;
  return counts;
}

/// Body ← ECI attitude in which **every** source in the suite is simultaneously
/// available, which is what makes one geometry serve the whole matrix: without
/// it, "both trackers failed, fall back to sun+mag" cannot be told from "there
/// was never a sun pair to fall back to".
///
/// Two constraints, and they nearly conflict:
///
///  * **Both tracker boresights clear of the Earth.** They sit 135° from body +Z,
///    the Earth's angular radius from 500 km is 68° and the AURIGA's exclusion is
///    22°, so a boresight needs more than 90° from nadir. Putting nadir on body
///    **+Z** gives both of them 135°, i.e. 45° of margin — the Earth-pointing
///    attitude the mounting was designed against (§8.2).
///  * **Two sun sensors in view at once**, so the §8.2 cross-unit check has a
///    runner-up to resolve against. With nadir on +Z the Sun sits near the body
///    X–Y plane (it is ~100° from nadir at this epoch), so the azimuth is the one
///    free parameter left: a **roll about +Z** placing the Sun at 45° of azimuth
///    puts it equidistant from the +X and +Y faces, inside both 60° fields.
///
/// The roll is solved from the ephemeris rather than written down, so an epoch
/// change moves the attitude with it instead of silently voiding the geometry;
/// `assertMatrixGeometry` then re-checks the angles the argument above assumed.
inline pm::Quat<pm::frames::Body, pm::frames::ECI> faultMatrixAttitude() {
  const pt::Tai epoch = pt::Tai::fromNanosecondsSinceEpoch(kEpochTaiNs);
  const Eigen::Vector3d sun_eci =
      polaris::ephemeris::sunPositionEci(pt::toTdb(pt::toTt(epoch))).eigen().normalized();
  // The vehicle sits at ECI +X, so nadir is -X. Body +Z on nadir, body +X on
  // +Z_ECI and body +Y on +Y_ECI, which is right-handed (Z_ECI x Y_ECI = -X_ECI).
  Eigen::Matrix3d base;
  base.row(0) = Eigen::Vector3d::UnitZ();
  base.row(1) = Eigen::Vector3d::UnitY();
  base.row(2) = -Eigen::Vector3d::UnitX();

  const Eigen::Vector3d sun_body = base * sun_eci;
  const double roll = 0.25 * M_PI - std::atan2(sun_body.y(), sun_body.x());
  const Eigen::Matrix3d dcm = Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitZ()) * base;
  return pm::Quat<pm::frames::Body, pm::frames::ECI>(pm::Quaternion::FromRotationMatrix(dcm));
}

/// Angles the matrix geometry is built on, in radians, for the fixture's own
/// self-check: the Sun's incidence on the +X and +Y sun sensors, and each tracker
/// boresight's separation from nadir.
struct MatrixGeometry {
  double sun_incidence_xp_rad = 0.0;
  double sun_incidence_yp_rad = 0.0;
  double tracker_a_from_nadir_rad = 0.0;
  double tracker_b_from_nadir_rad = 0.0;
};

inline MatrixGeometry matrixGeometry() {
  const pt::Tai epoch = pt::Tai::fromNanosecondsSinceEpoch(kEpochTaiNs);
  const Eigen::Vector3d sun_eci =
      polaris::ephemeris::sunPositionEci(pt::toTdb(pt::toTt(epoch))).eigen().normalized();
  const Eigen::Matrix3d dcm = faultMatrixAttitude().core().toRotationMatrix();
  const Eigen::Vector3d sun_body = dcm * sun_eci;
  const Eigen::Vector3d nadir_body = dcm * -Eigen::Vector3d::UnitX();
  const double s = 1.0 / std::sqrt(2.0);
  const Eigen::Vector3d bore_a(-s, 0.0, -s);
  const Eigen::Vector3d bore_b(s, 0.0, -s);
  const auto angle = [](const Eigen::Vector3d& a, const Eigen::Vector3d& b) {
    return std::atan2(a.cross(b).norm(), a.dot(b));
  };
  MatrixGeometry g;
  g.sun_incidence_xp_rad = angle(sun_body, Eigen::Vector3d::UnitX());
  g.sun_incidence_yp_rad = angle(sun_body, Eigen::Vector3d::UnitY());
  g.tracker_a_from_nadir_rad = angle(bore_a, nadir_body);
  g.tracker_b_from_nadir_rad = angle(bore_b, nadir_body);
  return g;
}

/// The plant for every fault-matrix case: the estimation orbit flown in the
/// all-sources-available attitude, inertially fixed so the trackers stay inside
/// their 2 °/s acquisition envelope from the first cycle and the geometry above
/// holds for the whole arc (nadir moves ~0.06 °/s).
/// Port index of each actuator, which is the index the §8.5 commands and the §9
/// interlock events name — `config/spacecraft/leo_smallsat.yaml` build order.
constexpr unsigned kMtqX = 0;
constexpr unsigned kMtqY = 1;
constexpr unsigned kMtqZ = 2;

/// The reference vehicle's suite for the §8.5 control rows, **read from the
/// compiled config** at @p simSetupPath rather than transcribed.
///
/// Every other suite in this header restates the catalog by hand, and that is
/// exactly how the rod settle time drifted 5x from the committed YAML while
/// REQ-ACTL-004 was being verified against the transcription. The control rows
/// already run `configc`, which resolves `config/hardware/**` into
/// `sim_setup.json`, so the honest suite is the one that file describes — the
/// reference vehicle, its mountings, its mounting *positions* (which the §7
/// near-field model needs, since the coupling goes as 1/r^3) and its catalog
/// values, by construction and not by copy.
///
/// **Ceiling, stated:** the rods' `dipole_axis` does not reach the truth side at
/// all — `sim::actuators::Magnetorquer` takes a body-frame dipole vector, so the
/// axis is purely a *flight* fact (which rod the controller resolves which
/// component onto, via `MtqAxesBody`). A wrong `MtqAxesBody` is therefore
/// invisible in SITL; the config compiler's cross-check against `dipole_axis` is
/// what catches it, and that check is where the coverage lives.
inline bool controlSuite(const std::string& simSetupPath, scenario::SpacecraftConfig& out,
                         std::string* error) {
  scenario::SimConfig compiled;
  if (!scenario::loadSimConfig(simSetupPath, pt::LeapSecondTable::historical(), compiled, error)) {
    return false;
  }
  out = compiled.spacecraft;
  return true;
}

inline io::SitlServer::Counts controlCounts() {
  io::SitlServer::Counts counts = faultMatrixCounts();
  counts.wheel = 4;
  counts.mtq = 3;
  return counts;
}

inline scenario::SimConfig faultMatrixOrbit(double durationS, const char* name) {
  scenario::SimConfig c = estimationOrbit(durationS);
  c.scenario_name = name;
  c.initial_state.attitude = faultMatrixAttitude();
  c.initial_state.body_rate = pm::Vec3<pm::frames::Body>(Eigen::Vector3d::Zero());
  return c;
}

}  // namespace polaris::test::sitl

#endif  // POLARIS_TESTS_INTEGRATION_SITL_HARNESS_HPP
