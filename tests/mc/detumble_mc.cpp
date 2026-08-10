/// @file
/// @brief The orbital-timescale B-dot detumble Monte Carlo campaign (design doc
/// §13, §23.2; REQ-ACTL-001's recorded "owed" item).
///
/// REQ-ACTL-001 is verified today on B-dot's **fast phase** only — below
/// 3.8 deg/s within 200 s of engaging. What it does not yet carry is a bound on
/// the time to the `DetumbleExitRadps` completion predicate, because that is a
/// different physical process on a different timescale: a B-dot law damps only
/// the body-rate components *perpendicular* to the field, and the component
/// along it produces no `dB/dt` in body axes and is invisible to the law. What
/// removes it is the field direction turning over the orbit — a ~1e-3 rad/s
/// process against a ~5e-2 rad/s spin — so the residual unwinds over **orbits**
/// and its duration depends on where in the field geometry the vehicle started
/// and which way it was spinning. That is a distribution, not a number, and this
/// driver is what measures it.
///
/// **Why this is a driver and not a test.** It is a measurement instrument, like
/// `tests/benchmark/`: it makes no pass/fail claim, it costs orbits of sim time
/// per run, and putting it behind a CI gate would turn runner speed into flakes.
/// It is therefore deliberately **not registered with ctest** (see
/// `tests/mc/CMakeLists.txt`). The verdict is produced downstream by
/// `analysis/detumble`, which reads the JSONL this writes.
///
/// **Why the real flight binary and not an in-process B-dot.** B-dot's input is
/// not the magnetometer; it is the *voted, plausibility-gated, quiet-window*
/// field that `AttitudeEstimator` publishes, and whose availability the §7
/// MTQ/MAG duty-cycle interlock controls. Reconstructing that chain sim-side
/// would be a transcription of the flight path — exactly the defect class the
/// review-lessons catalog records twice (a harness that restates flight rules
/// drifts toward passing, and a harness that supplies data a real gate would
/// suppress is testing the harness). So each run forks the same deployment the
/// SITL rows fly, over the same lockstep wire. The cost is the SITL exchange at
/// every macro step; measured, that is roughly 28x real time, so one 8-orbit run
/// is a bit over an hour of wall clock. Runs are sequential within one process;
/// the campaign is parallelised by **sharding across processes**, which is what
/// the README's `xargs -P` recipe does.
///
/// **The vehicle is the one that ships.** Everything — orbit, environment,
/// hardware, gains, and the `DetumbleExitRadps` / `DetumbleConfirmCycles` values
/// the completion metric is computed from — is read from the config compiler's
/// output for `config/spacecraft/leo_smallsat.yaml`. Nothing here transcribes a
/// catalog value.
///
/// Usage (see `analysis/detumble/README.md` for the full recipe):
/// @code
///   ./polaris_detumble_mc --runs 2 --duration-s 600    # smoke: proves the harness
///   ./polaris_detumble_mc --first-run 0 --runs 3 --duration-s 45416
///       --out build-artifacts/detumble-mc/shard-0.jsonl
/// @endcode

#include <fcntl.h>
#include <signal.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <Eigen/Geometry>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <nlohmann/json.hpp>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "io/closed_loop.hpp"
#include "math/frames.hpp"
#include "math/quaternion.hpp"
#include "math/typed_vector.hpp"
#include "scenario/sim_config.hpp"
#include "scenario/sim_runner.hpp"
#include "scenario/vehicle.hpp"
#include "sitl_harness.hpp"
#include "world/magnetic_field.hpp"

namespace {

namespace io = polaris::sim::io;
namespace scenario = polaris::sim::scenario;
namespace pm = polaris::math;
namespace pt = polaris::time;
using polaris::test::sitl::compileConfig;
using polaris::test::sitl::controlCounts;
using polaris::test::sitl::fswBinaryPath;
using polaris::test::sitl::kEpochDecimalYear;
using polaris::test::sitl::pythonPath;
using polaris::test::sitl::readFile;

// ----------------------------------------------------------------------
// Campaign options
// ----------------------------------------------------------------------

struct Options {
  int runs = 4;
  int first_run = 0;
  std::uint64_t seed = 20260101;  ///< master seed; the vehicle config's own
  double duration_s = 45416.0;    ///< 8 orbits at 5677 s
  double profile_step_s = 60.0;
  /// Control mode latched in the deployment: 1 = DETUMBLE (the campaign), 0 =
  /// IDLE. The IDLE setting flies the identical dispersed scenario with the
  /// control law switched off, which is the ablation that separates what B-dot
  /// did from what the environment did to the same vehicle — the review-lessons
  /// rule that an attribution needs an ablation and not a narrative.
  unsigned ctrl_mode = 1;
  std::string out = "build-artifacts/detumble-mc/runs.jsonl";
  std::string work = "build-artifacts/detumble-mc";
};

/// Tip-off rate magnitude band [deg/s]. REQ-ACTL-001 is written against a 5 deg/s
/// separation tip-off, which is the demanding end; a dispenser delivers less than
/// its specified maximum most of the time. Uniform over [2, 5] rather than
/// clustered at 5 so the campaign can *test* whether magnitude drives the tail at
/// all — the physics argument says it should not (the fast phase removes the
/// perpendicular rate in minutes regardless), and a correlation reported against
/// an actual spread is worth more than an assumption.
constexpr double kTipoffMinDegS = 2.0;
constexpr double kTipoffMaxDegS = 5.0;

// ----------------------------------------------------------------------
// One run's dispersion draw
// ----------------------------------------------------------------------

/// What is dispersed, and why each one is here.
///
///  * **Rate magnitude and direction.** The direction relative to the field is
///    the whole point: the invisible component is the projection of the tumble
///    onto **B**, so a spin already aligned with the local field line starts with
///    the entire rate in the tail and a spin perpendicular to it starts with
///    none. Direction is uniform on the sphere (normalised Gaussian, which has no
///    pole bias); magnitude is uniform over the tip-off band.
///  * **Initial attitude.** B-dot does not read attitude, but the rod triad is
///    body-fixed and the clamp is *per rod*, so which body axes the demanded
///    dipole falls on — and therefore when the law saturates — depends on the
///    vehicle's orientation in the field. Uniform on SO(3) (Shoemake).
///  * **RAAN and argument of latitude.** Where the orbit plane sits relative to
///    the geomagnetic dipole, and where in that plane the run starts. This is the
///    dominant geometric driver of how fast the field direction turns in body
///    axes, which is the only mechanism that removes the tail.
///  * **Epoch offset.** The dipole is tilted ~11 deg from the spin axis, so Earth
///    rotation moves it relative to a fixed orbit plane on a sidereal-day cycle;
///    the epoch offset covers that phase (and moves the Sun, which sets the
///    eclipse pattern the sensors see). Uniform over one sidereal day.
struct Draw {
  double rate_deg_s = 0.0;
  Eigen::Vector3d rate_axis{Eigen::Vector3d::UnitZ()};
  Eigen::Quaterniond attitude{Eigen::Quaterniond::Identity()};
  double draan_rad = 0.0;
  double du_rad = 0.0;
  double depoch_s = 0.0;
  std::uint64_t seed = 0;
};

/// Per-run substream: a run's draw and its sensor noise must not shift when the
/// campaign's size changes, so the stream is derived from `{master seed, run
/// index}` rather than taken from a single walked generator (the review-lessons
/// rule on paired comparisons).
Draw drawFor(std::uint64_t master_seed, int run_index) {
  std::seed_seq seq{static_cast<std::uint32_t>(master_seed & 0xffffffffu),
                    static_cast<std::uint32_t>(master_seed >> 32),
                    static_cast<std::uint32_t>(run_index)};
  std::mt19937_64 rng(seq);
  std::uniform_real_distribution<double> unit(0.0, 1.0);
  std::normal_distribution<double> gauss(0.0, 1.0);

  Draw d;
  d.seed = rng();
  d.rate_deg_s = kTipoffMinDegS + (kTipoffMaxDegS - kTipoffMinDegS) * unit(rng);
  do {
    d.rate_axis = Eigen::Vector3d(gauss(rng), gauss(rng), gauss(rng));
  } while (d.rate_axis.norm() < 1.0e-9);
  d.rate_axis.normalize();

  // Shoemake's uniform quaternion.
  const double u1 = unit(rng);
  const double u2 = unit(rng);
  const double u3 = unit(rng);
  const double s1 = std::sqrt(1.0 - u1);
  const double s2 = std::sqrt(u1);
  d.attitude = Eigen::Quaterniond(s2 * std::cos(2.0 * M_PI * u3), s1 * std::sin(2.0 * M_PI * u2),
                                  s1 * std::cos(2.0 * M_PI * u2), s2 * std::sin(2.0 * M_PI * u3));
  d.attitude.normalize();

  d.draan_rad = 2.0 * M_PI * unit(rng);
  d.du_rad = 2.0 * M_PI * unit(rng);
  d.depoch_s = 86164.0905 * unit(rng);  // one sidereal day
  return d;
}

/// Apply @p d to the compiled reference scenario.
///
/// The orbit dispersion is done as **rotations of the compiled Cartesian state**,
/// not by re-deriving elements: rotating `r` and `v` about ECI +Z by ΔΩ moves the
/// node exactly for any orbit, and rotating both about the orbit normal by Δu
/// moves the argument of latitude exactly for a *circular* one. The reference
/// orbit is circular (`ecc: 0.0`), which `applyDraw` checks rather than assumes —
/// design doc §19.3 keeps Keplerian-to-Cartesian conversion in the config
/// compiler, and this is the way to disperse the orbit without a second
/// implementation of it on the C++ side.
bool applyDraw(const Draw& d, double duration_s, int run_index, scenario::SimConfig& c,
               std::string* error) {
  const Eigen::Vector3d r = c.initial_state.position.eigen();
  const Eigen::Vector3d v = c.initial_state.velocity.eigen();
  const Eigen::Vector3d h = r.cross(v);
  if (h.norm() <= 0.0 || std::abs(r.dot(v)) > 1.0e-6 * r.norm() * v.norm()) {
    *error = "the compiled reference orbit is not circular; the Δu rotation is only exact for e=0";
    return false;
  }
  const Eigen::Matrix3d node =
      Eigen::AngleAxisd(d.draan_rad, Eigen::Vector3d::UnitZ()).toRotationMatrix();
  const Eigen::Matrix3d along = Eigen::AngleAxisd(d.du_rad, h.normalized()).toRotationMatrix();
  const Eigen::Matrix3d rot = node * along;
  c.initial_state.position = pm::Vec3<pm::frames::ECI>(rot * r);
  c.initial_state.velocity = pm::Vec3<pm::frames::ECI>(rot * v);

  c.initial_state.epoch =
      pt::Tai::fromNanosecondsSinceEpoch(c.initial_state.epoch.nanosecondsSinceEpoch() +
                                         static_cast<std::int64_t>(d.depoch_s * 1.0e9));
  c.initial_state.attitude = pm::Quat<pm::frames::Body, pm::frames::ECI>(
      polaris::math::Quaternion(d.attitude.w(), d.attitude.x(), d.attitude.y(), d.attitude.z())
          .canonical());
  c.initial_state.body_rate = pm::Vec3<pm::frames::Body>(d.rate_deg_s * M_PI / 180.0 * d.rate_axis);

  c.propagation.duration_s = duration_s;
  c.seed = d.seed;
  c.scenario_name = "detumble-mc-" + std::to_string(run_index);

  // **Confounds removed, deliberately.** The committed scenario schedules a GNSS
  // outage and a spoof (they exercise §9.2 in the fault matrix) and enables the
  // geographic jamming map. Either one starves B-dot of its only input — no fix,
  // no onboard field, no voted field — for as long as it lasts, which would put a
  // fault-response duration into a distribution that is supposed to characterise
  // convergence geometry. Detumble *through* a GNSS outage is a worthwhile row
  // and it belongs in the FDIR suite, not here.
  c.environment.gnss_fault_events.clear();
  c.environment.gnss_jamming_enabled = false;
  return true;
}

// ----------------------------------------------------------------------
// One run's product
// ----------------------------------------------------------------------

struct Record {
  int run_index = 0;
  Draw draw;
  bool healthy = false;
  double t_engage_s = -1.0;  ///< first macro step the rods carried a dipole
  double t_exit_s = -1.0;    ///< completion confirmed, seconds since epoch
  double t_first_below_s = -1.0;
  double rate_initial_deg_s = 0.0;
  double rate_at_fast_phase_deg_s = 0.0;  ///< 200 s after engagement (REQ-ACTL-001)
  double rate_final_deg_s = 0.0;
  double rate_min_deg_s = 0.0;
  double peak_rate_after_fast_phase_deg_s = 0.0;
  /// Angle between the spin axis and the local geomagnetic field [deg], at
  /// engagement and again at the end of the fast phase. **The physical predictor
  /// of the tail**: B-dot cannot see the rate component along **B**, so what
  /// survives the fast phase is the projection of the spin onto the field line,
  /// and how long it then takes is set by how much of it there is. Recording the
  /// angle at both instants is what lets the analysis show that it is the
  /// *post-fast-phase* geometry that predicts the duration and the initial draw
  /// that does not.
  double initial_spin_field_angle_deg = -1.0;
  double fast_phase_spin_field_angle_deg = -1.0;
  /// Largest wheel torque the deployment commanded [N·m]. DETUMBLE is a purely
  /// magnetic mode, so this must be zero for the whole run; a nonzero value
  /// means body rate is being traded with rotor momentum and |omega| stops being
  /// a statement about how detumbled the vehicle is.
  double peak_wheel_torque_nm = 0.0;
  /// Rotational kinetic energy 0.5*w'Jw [J] at engagement, at its minimum, and at
  /// the end. B-dot's guarantee is dissipativity in *energy*, not monotonicity in
  /// |omega|, and with a near-isotropic inertia the two can only differ by
  /// sqrt(Jmax/Jmin). Recording energy is therefore what turns "the rate went
  /// back up" into either a redistribution or a defect.
  double energy_initial_j = 0.0;
  double energy_min_j = 0.0;
  double energy_final_j = 0.0;
  double wall_s = 0.0;
  std::vector<double> profile_t_s;
  std::vector<double> profile_rate_deg_s;
  std::string note;
};

// ----------------------------------------------------------------------
// Spawning the deployment from a multithreaded driver
// ----------------------------------------------------------------------

/// Fork+exec the flight binary. Unlike the SITL harness's version every string is
/// built **before** the fork and the child touches only async-signal-safe calls
/// (`open`, `dup2`, `execv`). The driver is single-threaded so this is belt and
/// braces today, but it is the property that made dropping the worker pool a
/// safe change rather than a hopeful one.
pid_t spawnFsw(const std::string& bin, std::uint16_t port, const std::string& prm_path,
               const std::string& log_path, unsigned ctrl_mode) {
  const std::string port_str = std::to_string(port);
  const std::string ctrl_str = std::to_string(ctrl_mode);
  std::vector<std::string> args{bin,  "-s",    port_str, "-P", prm_path, "-Y", kEpochDecimalYear,
                                "-c", ctrl_str};
  std::vector<char*> argv;
  argv.reserve(args.size() + 1);
  for (std::string& a : args) {
    argv.push_back(a.data());
  }
  argv.push_back(nullptr);
  const std::string log = log_path;

  const pid_t pid = ::fork();
  if (pid == 0) {
    const int fd = ::open(log.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) {
      _exit(126);
    }
    ::dup2(fd, STDOUT_FILENO);
    const int null_fd = ::open("/dev/null", O_WRONLY);
    if (null_fd >= 0) {
      ::dup2(null_fd, STDERR_FILENO);
    }
    ::execv(argv[0], argv.data());
    _exit(127);
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

// ----------------------------------------------------------------------
// Flying one run
// ----------------------------------------------------------------------

/// Completion predicate, evaluated on **truth**: the rate norm below
/// @p exit_radps for @p confirm_cycles consecutive macro steps. The thresholds
/// are the deployment's own `DetumbleExitRadps` / `DetumbleConfirmCycles`, read
/// from the compiled parameter artifact.
///
/// Truth-side rather than through the flight predicate because `DetumbleComplete`
/// is *telemetry*, and the SITL wire carries only the deployment's event stream.
/// It is also the measure a handover time should be set from: the question is
/// when the vehicle is actually slow enough for the next mode, not when the
/// estimator believes it is. The two differ by the estimator's rate error, which
/// on the coarse pair is small against a 0.5 deg/s threshold.
struct Completion {
  double t_exit_s = -1.0;
  double t_first_below_s = -1.0;
};

Completion completionOf(const std::vector<io::MacroSample>& trace, double exit_radps,
                        std::uint32_t confirm_cycles) {
  Completion c;
  std::uint32_t streak = 0;
  for (const io::MacroSample& s : trace) {
    if (s.state.body_rate.eigen().norm() < exit_radps) {
      if (streak == 0) {
        c.t_first_below_s = s.t_s;
      }
      ++streak;
      if (streak >= confirm_cycles && c.t_exit_s < 0.0) {
        c.t_exit_s = s.t_s;
        return c;
      }
    } else {
      streak = 0;
      c.t_first_below_s = -1.0;
    }
  }
  return c;
}

/// Trace index of the engagement instant, clamped into range.
std::size_t engage_index_or_zero(double origin_s, double dt_s, std::size_t size) {
  const auto index = static_cast<std::size_t>(origin_s / dt_s);
  return index < size ? index : 0;
}

Record flyOne(int run_index, const Options& opt, const scenario::SimConfig& reference,
              const std::string& prm_path, double exit_radps, std::uint32_t confirm_cycles) {
  const auto started = std::chrono::steady_clock::now();
  Record rec;
  rec.run_index = run_index;
  rec.draw = drawFor(opt.seed, run_index);

  scenario::SimConfig config = reference;
  std::string error;
  if (!applyDraw(rec.draw, opt.duration_s, run_index, config, &error)) {
    rec.note = error;
    return rec;
  }

  const std::string log_path = opt.work + "/fsw-" + std::to_string(run_index) + ".log";
  const auto macro_ns = static_cast<std::int64_t>(1.0e9 / config.propagation.fsw_rate_hz);
  io::SitlServer server(controlCounts(), macro_ns);
  if (!server.start(0)) {
    rec.note = server.lastError();
    return rec;
  }
  const pid_t pid = spawnFsw(fswBinaryPath(), server.port(), prm_path, log_path, opt.ctrl_mode);
  if (pid < 0) {
    rec.note = "fork failed";
    return rec;
  }

  scenario::Vehicle vehicle;
  if (!scenario::buildVehicle(config.spacecraft, config.seed, vehicle, &error)) {
    rec.note = error;
    reapFsw(pid);
    return rec;
  }
  scenario::SimRunner runner;
  io::ClosedLoop loop(runner, vehicle);
  if (!runner.build(config, scenario::DataPaths::under(POLARIS_GOLDEN_DIR), &error,
                    loop.wrench())) {
    rec.note = error;
    reapFsw(pid);
    return rec;
  }

  // Engagement is read off the wire, not assumed: the first macro step on which
  // the deployment scheduled an on-window is the first step B-dot commanded
  // anything, which is the instant REQ-ACTL-001's window opens. It is not step
  // zero — the reference vehicle's receiver quotes a 34 s cold start, and until
  // it has a fix there is no onboard field and so no voted field to difference.
  std::int64_t engage_step = -1;
  std::int64_t step = 0;
  double peak_wheel_torque = 0.0;
  const io::FswCallback inner = server.callback();
  const io::FswCallback watched = [&](const io::FswInputs& in) {
    const io::FswOutputs out = inner(in);
    if (engage_step < 0 && out.mtq_on_window_s > 0.0) {
      engage_step = step;
    }
    for (const io::WheelCommand& w : out.wheels) {
      peak_wheel_torque = std::max(peak_wheel_torque, std::abs(w.value));
    }
    ++step;
    return out;
  };

  std::vector<io::MacroSample> trace;
  const bool ran = loop.run(watched, &trace, &error);
  rec.healthy = ran && server.healthy();
  server.stop();
  reapFsw(pid);
  if (!ran) {
    rec.note = error;
    return rec;
  }
  if (trace.empty()) {
    rec.note = "empty trace";
    return rec;
  }

  const double dt = 1.0 / config.propagation.fsw_rate_hz;
  const auto deg = [](double radps) { return radps * 180.0 / M_PI; };
  rec.t_engage_s = engage_step < 0 ? -1.0 : static_cast<double>(engage_step) * dt;
  rec.rate_initial_deg_s = deg(trace.front().state.body_rate.eigen().norm());
  rec.rate_final_deg_s = deg(trace.back().state.body_rate.eigen().norm());

  const Completion c = completionOf(trace, exit_radps, confirm_cycles);
  // Reported relative to engagement, the same origin REQ-ACTL-001's fast-phase
  // bound uses, so a handover time set from this distribution is a statement
  // about the control law rather than about the GNSS receiver's datasheet.
  const double origin = rec.t_engage_s < 0.0 ? 0.0 : rec.t_engage_s;
  rec.t_exit_s = c.t_exit_s < 0.0 ? -1.0 : c.t_exit_s - origin;
  rec.t_first_below_s = c.t_first_below_s < 0.0 ? -1.0 : c.t_first_below_s - origin;

  const auto fast_phase_index = static_cast<std::size_t>((origin + 200.0) / dt);
  double min_rate = rec.rate_initial_deg_s;
  double peak_after = 0.0;
  for (std::size_t i = 0; i < trace.size(); ++i) {
    const double r = deg(trace[i].state.body_rate.eigen().norm());
    min_rate = std::min(min_rate, r);
    if (i >= fast_phase_index) {
      peak_after = std::max(peak_after, r);
    }
  }
  rec.rate_min_deg_s = min_rate;
  rec.peak_rate_after_fast_phase_deg_s = peak_after;
  rec.peak_wheel_torque_nm = peak_wheel_torque;

  const Eigen::Matrix3d inertia = config.spacecraft.inertia_kgm2;
  const auto energyOf = [&inertia](const io::MacroSample& s) {
    const Eigen::Vector3d w = s.state.body_rate.eigen();
    return 0.5 * w.dot(inertia * w);
  };
  rec.energy_initial_j = energyOf(trace[engage_index_or_zero(origin, dt, trace.size())]);
  rec.energy_final_j = energyOf(trace.back());
  rec.energy_min_j = rec.energy_initial_j;
  for (const io::MacroSample& s : trace) {
    rec.energy_min_j = std::min(rec.energy_min_j, energyOf(s));
  }
  rec.rate_at_fast_phase_deg_s = fast_phase_index < trace.size()
                                     ? deg(trace[fast_phase_index].state.body_rate.eigen().norm())
                                     : rec.rate_final_deg_s;

  // The spin/field geometry, from the runner's own wired IGRF resolver — the
  // same field the magnetorquers torque against, not a second evaluation of it.
  const polaris::sim::world::MagneticFieldFn field = runner.magneticFieldFn();
  const auto spinFieldAngle = [&field](const io::MacroSample& s) {
    pm::Vec3<pm::frames::ECI> b_eci;
    if (!field || !field(s.state.epoch, s.state.position, b_eci)) {
      return -1.0;
    }
    // Body rate into ECI: the stored quaternion is Body <- ECI, so its rotation
    // matrix transposed carries a body vector out to ECI.
    const Eigen::Vector3d spin_eci =
        s.state.attitude.core().toRotationMatrix().transpose() * s.state.body_rate.eigen();
    if (spin_eci.norm() <= 0.0 || b_eci.eigen().norm() <= 0.0) {
      return -1.0;
    }
    // atan2 form, never acos of a dot product (house rule, lib/README.md).
    return std::atan2(spin_eci.cross(b_eci.eigen()).norm(), spin_eci.dot(b_eci.eigen())) * 180.0 /
           M_PI;
  };
  const auto engage_index = static_cast<std::size_t>(origin / dt);
  if (engage_index < trace.size()) {
    rec.initial_spin_field_angle_deg = spinFieldAngle(trace[engage_index]);
  }
  if (fast_phase_index < trace.size()) {
    rec.fast_phase_spin_field_angle_deg = spinFieldAngle(trace[fast_phase_index]);
  }

  const auto stride = static_cast<std::size_t>(std::max(1.0, opt.profile_step_s / dt));
  for (std::size_t i = 0; i < trace.size(); i += stride) {
    rec.profile_t_s.push_back(trace[i].t_s - origin);
    rec.profile_rate_deg_s.push_back(deg(trace[i].state.body_rate.eigen().norm()));
  }

  rec.wall_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
  // The deployment's event stream is kept only when it is evidence: a run in
  // which B-dot never commanded a rod is a run whose event log is the only thing
  // that says why, and 93 healthy logs are 93 files nobody reads.
  if (rec.healthy && rec.t_engage_s >= 0.0) {
    std::remove(log_path.c_str());
  } else {
    rec.note = rec.note.empty() ? ("see " + log_path) : rec.note;
  }
  return rec;
}

nlohmann::json toJson(const Record& r, const scenario::SimConfig& reference, double exit_radps,
                      std::uint32_t confirm_cycles) {
  nlohmann::json j;
  // The completion thresholds travel *with* the record. Downstream then draws
  // and reports the threshold the runs were actually judged against instead of
  // a transcribed one, so a figure cannot quietly outlive a change to the
  // committed value.
  j["exit_threshold_deg_s"] = exit_radps * 180.0 / M_PI;
  j["confirm_cycles"] = confirm_cycles;
  j["run_index"] = r.run_index;
  j["seed"] = r.draw.seed;
  j["config_hash"] = reference.config_hash;
  j["scenario"] = reference.scenario_name;
  j["dispersion"] = {
      {"rate_deg_s", r.draw.rate_deg_s},
      {"rate_axis_body", {r.draw.rate_axis.x(), r.draw.rate_axis.y(), r.draw.rate_axis.z()}},
      {"attitude_wxyz",
       {r.draw.attitude.w(), r.draw.attitude.x(), r.draw.attitude.y(), r.draw.attitude.z()}},
      {"delta_raan_deg", r.draw.draan_rad * 180.0 / M_PI},
      {"delta_arglat_deg", r.draw.du_rad * 180.0 / M_PI},
      {"delta_epoch_s", r.draw.depoch_s}};
  j["healthy"] = r.healthy;
  j["note"] = r.note;
  j["t_engage_s"] = r.t_engage_s;
  j["t_exit_s"] = r.t_exit_s;
  j["t_first_below_s"] = r.t_first_below_s;
  j["rate_initial_deg_s"] = r.rate_initial_deg_s;
  j["rate_at_fast_phase_deg_s"] = r.rate_at_fast_phase_deg_s;
  j["rate_final_deg_s"] = r.rate_final_deg_s;
  j["rate_min_deg_s"] = r.rate_min_deg_s;
  j["peak_rate_after_fast_phase_deg_s"] = r.peak_rate_after_fast_phase_deg_s;
  j["initial_spin_field_angle_deg"] = r.initial_spin_field_angle_deg;
  j["fast_phase_spin_field_angle_deg"] = r.fast_phase_spin_field_angle_deg;
  j["peak_wheel_torque_nm"] = r.peak_wheel_torque_nm;
  j["energy_initial_j"] = r.energy_initial_j;
  j["energy_min_j"] = r.energy_min_j;
  j["energy_final_j"] = r.energy_final_j;
  j["wall_s"] = r.wall_s;
  j["profile_t_s"] = r.profile_t_s;
  j["profile_rate_deg_s"] = r.profile_rate_deg_s;
  return j;
}

// ----------------------------------------------------------------------
// Driver
// ----------------------------------------------------------------------

bool parseArgs(int argc, char** argv, Options& opt) {
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    const auto next = [&]() { return (i + 1 < argc) ? std::string(argv[++i]) : std::string(); };
    if (a == "--runs") {
      opt.runs = std::stoi(next());
    } else if (a == "--first-run") {
      opt.first_run = std::stoi(next());
    } else if (a == "--seed") {
      opt.seed = std::stoull(next());
    } else if (a == "--duration-s") {
      opt.duration_s = std::stod(next());
    } else if (a == "--profile-step-s") {
      opt.profile_step_s = std::stod(next());
    } else if (a == "--ctrl-mode") {
      opt.ctrl_mode = static_cast<unsigned>(std::stoul(next()));
    } else if (a == "--out") {
      opt.out = next();
    } else if (a == "--work") {
      opt.work = next();
    } else {
      std::cerr << "usage: polaris_detumble_mc [--runs N] [--first-run I] [--seed S]\n"
                << "       [--duration-s D] [--profile-step-s P] [--ctrl-mode 0|1]\n"
                << "       [--out FILE.jsonl] [--work DIR]\n"
                << "Runs are sequential; parallelise with shards (see the README):\n"
                << "  --first-run I --runs N --out shard-I.jsonl --work DIR/I\n";
      return false;
    }
  }
  return opt.runs > 0 && opt.duration_s > 0.0;
}

}  // namespace

int main(int argc, char** argv) {
  Options opt;
  if (!parseArgs(argc, argv, opt)) {
    return 2;
  }
  if (::access(fswBinaryPath().c_str(), X_OK) != 0) {
    std::cerr << "flight binary not built at " << fswBinaryPath()
              << " — run `uv run fprime-util build` first\n";
    return 1;
  }
  if (::access(pythonPath().c_str(), X_OK) != 0) {
    std::cerr << "python interpreter not found at " << pythonPath() << " — run `uv sync`\n";
    return 1;
  }

  const std::string cfg_dir = opt.work + "/config";
  if (compileConfig(cfg_dir, cfg_dir + "/configc.err") != 0) {
    std::cerr << "config compiler failed:\n" << readFile(cfg_dir + "/configc.err") << "\n";
    return 1;
  }

  scenario::SimConfig reference;
  std::string error;
  if (!scenario::loadSimConfig(cfg_dir + "/sim_setup.json", pt::LeapSecondTable::historical(),
                               reference, &error)) {
    std::cerr << "could not read the compiled scenario: " << error << "\n";
    return 1;
  }

  // The completion thresholds are the deployment's, read from the same artifact
  // that programs its ParameterDb. Transcribing them here would be the defect
  // the review-lessons catalog names: a harness that restates a flight value
  // drifts toward passing while the committed value moves underneath it.
  double exit_radps = 0.0;
  std::uint32_t confirm_cycles = 0;
  try {
    std::ifstream params(cfg_dir + "/fprime_params.json");
    const nlohmann::json fsw = nlohmann::json::parse(params).at("fsw_parameters");
    exit_radps = fsw.at("flight.attitudeController.DetumbleExitRadps").get<double>();
    confirm_cycles = fsw.at("flight.attitudeController.DetumbleConfirmCycles").get<std::uint32_t>();
  } catch (const std::exception& e) {
    std::cerr << "could not read the detumble thresholds from fprime_params.json: " << e.what()
              << "\n";
    return 1;
  }
  if (!(exit_radps > 0.0) || confirm_cycles == 0) {
    std::cerr << "the compiled detumble thresholds are not usable\n";
    return 1;
  }

  std::ofstream out(opt.out);
  if (!out) {
    std::cerr << "cannot write " << opt.out << "\n";
    return 1;
  }
  std::cerr << "detumble MC: " << opt.runs << " runs x " << opt.duration_s << " s ("
            << (opt.duration_s / 5677.0) << " orbits), seed " << opt.seed << ", mode "
            << opt.ctrl_mode << "\n  exit " << (exit_radps * 180.0 / M_PI) << " deg/s held "
            << confirm_cycles << " cycles; config " << reference.config_hash.substr(0, 16) << "\n";

  // **Sequential, and parallelised by sharding the process rather than
  // threading it.** An in-process worker pool has to `fork` the deployment from
  // one of several threads, and a child that allocates between `fork` and `exec`
  // can deadlock on a lock another thread held — which is not theoretical here:
  // the first threaded version produced runs in which the deployment came up but
  // never commanded a rod, i.e. silently wrong data rather than a crash. Shards
  // are also independently restartable, which a half-finished pool is not. See
  // analysis/detumble/README.md for the `xargs -P` recipe.
  const auto campaign_started = std::chrono::steady_clock::now();
  const std::string prm_path = cfg_dir + "/PrmDb.dat";
  for (int index = opt.first_run; index < opt.first_run + opt.runs; ++index) {
    const Record rec = flyOne(index, opt, reference, prm_path, exit_radps, confirm_cycles);
    out << toJson(rec, reference, exit_radps, confirm_cycles).dump() << "\n";
    out.flush();
    std::cerr << "  [" << (index - opt.first_run + 1) << "/" << opt.runs << "] run " << index
              << ": " << (rec.healthy ? "ok" : ("FAILED " + rec.note)) << ", t_exit "
              << (rec.t_exit_s < 0.0 ? std::string("not reached")
                                     : std::to_string(rec.t_exit_s) + " s")
              << ", " << std::fixed << std::setprecision(1) << rec.wall_s << " s wall\n";
  }

  const double wall =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - campaign_started).count();
  std::cerr << "campaign finished in " << wall << " s -> " << opt.out << "\n";
  return 0;
}
