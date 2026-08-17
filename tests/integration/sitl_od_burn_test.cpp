/// @file The orbit filter under thrust and long coast (design doc §8.3, §17;
/// REQ-ODP-007, REQ-ODP-008, REQ-MAN-001; Push 70).
///
/// The rows the Ceresoli et al. (2025) review turned into work: a finite burn
/// while the filter tracks, the same burn inside a GNSS outage with the filter
/// told and blind, a twenty-minute coast on the degraded horizon that keeps
/// the magnetic reference, and a receiver whose delivery latency jitters. Each
/// row reads the deployment's own `OrbitStatus` events (position, sigma,
/// quality once a minute) against the truth trace at the same epoch, so what
/// is asserted is the filter's *error*, not only its verdicts.
///
///   row                                        | asserted
///   -------------------------------------------|----------------------------------
///   60 s burn, GNSS nominal                    | tracked: no drop, refusals rare, error < 30 m
///   60 s burn inside a 900 s outage, told      | coasted on the thrust: error at outage end < 100
///   m same, filter blind (-N 0)                  | error at outage end > 1 km — the paper's 5 vs 9
///   km 1200 s outage, no burn                     | DEGRADED at 300 s, magnetic reference kept,
///   error < 100 m, no re-seed receiver latency 50 ms +/- 5 ms jitter     | nothing: refusals at
///   the false-alarm rate
///
/// Skips (never fails) when the flight binary or the Python toolchain is absent.

#include <gtest/gtest.h>

#include <cmath>
#include <cstdlib>
#include <string>
#include <vector>

#include "io/closed_loop.hpp"
#include "scenario/sim_runner.hpp"
#include "sitl_harness.hpp"

namespace {

using namespace polaris::test::sitl;  // NOLINT(build/namespaces) — the shared SITL fixture

constexpr double kFineHorizonS = 300.0;     // MaxCoastS on the reference vehicle
constexpr unsigned kBurnStartCycle = 3000;  // 300 s in
constexpr double kBurnDurationS = 60.0;
constexpr const char* kBurnSpec = "3000,60,1.0";  // -b cycle,durationS,throttle

struct RunResult {
  std::string log;
  bool sim_healthy = false;
  std::vector<io::MacroSample> trace;
};

/// One `OrbitStatus` event, parsed from the log.
struct Status {
  std::string quality;
  double age_s = 0.0;
  double sigma_m = 0.0;
  Eigen::Vector3d pos_m = Eigen::Vector3d::Zero();
  double t_s = 0.0;  ///< seconds since the scenario epoch, from the EVR timestamp
};

/// The fault-matrix suite plus the reference vehicle's thruster on body +X.
scenario::SpacecraftConfig burnSuite(double receiverLatencyS = 0.0, double jitterS = 0.0) {
  scenario::SpacecraftConfig sc = faultMatrixSuite();
  for (auto& u : sc.sensors) {
    if (u.kind == "gnss") {
      u.params["fix_latency_s"] = receiverLatencyS;
      u.params["fix_latency_jitter_s"] = jitterS;
    }
  }
  scenario::UnitConfig thr = unit("thr_x", "MONOPROP-05N", "thruster",
                                  {{"thrust_n", 0.5},
                                   {"isp_s", 220.0},
                                   {"min_impulse_bit_ns", 0.005},
                                   {"rise_time_s", 0.05},
                                   {"fall_time_s", 0.03},
                                   {"thrust_scale_error", 0.02},
                                   {"thrust_noise_frac", 0.01},
                                   {"misalignment_rad", 0.005}});
  thr.thrust_axis = Eigen::Vector3d::UnitX();
  sc.actuators.push_back(thr);
  return sc;
}

io::SitlServer::Counts burnCounts() {
  io::SitlServer::Counts counts = faultMatrixCounts();
  counts.thruster = 1;
  return counts;
}

scenario::SimConfig withOutage(scenario::SimConfig orbit, double startS, double stopS) {
  scenario::GnssFaultEvent ev;
  ev.unit = "gps_a";
  ev.type = scenario::GnssFaultEvent::Type::kOutage;
  ev.start_s = startS;
  ev.stop_s = stopS;
  orbit.environment.gnss_fault_events.push_back(ev);
  return orbit;
}

RunResult flyBurn(const std::string& tag, const scenario::SimConfig& orbit,
                  const scenario::SpacecraftConfig& suite, const char* burnSpec,
                  int accelInput = -1) {
  RunResult result;
  const std::string work_dir =
      "build-artifacts/test-burn-" + tag + "-" + std::to_string(::getpid());
  const std::string err_path = work_dir + "/configc.err";
  const std::string prm_path = work_dir + "/PrmDb.dat";
  const std::string log_path = work_dir + "/fsw.log";
  if (compileConfig(work_dir, err_path) != 0) {
    ADD_FAILURE() << "config compiler failed:\n" << readFile(err_path);
    return result;
  }
  io::SitlServer server(burnCounts(), 100'000'000LL);
  if (!server.start(0)) {
    ADD_FAILURE() << server.lastError();
    return result;
  }
  const pid_t pid = spawnFsw(fswBinaryPath(), server.port(), prm_path, log_path,
                             /*magCalSamples=*/0, /*stAlignPairs=*/100, /*stAlignUnit=*/1,
                             /*ctrlMode=*/0, nullptr, -1, /*odResetCycle=*/0, burnSpec, accelInput);
  if (pid < 0) {
    ADD_FAILURE() << "fork failed";
    return result;
  }
  scenario::Vehicle vehicle;
  std::string error;
  if (!scenario::buildVehicle(suite, 1, vehicle, &error)) {
    ADD_FAILURE() << error;
    reapFsw(pid);
    return result;
  }
  scenario::SimRunner runner;
  io::ClosedLoop loop(runner, vehicle);
  if (!runner.build(orbit, scenario::DataPaths::under(POLARIS_GOLDEN_DIR), &error, loop.wrench())) {
    ADD_FAILURE() << error;
    reapFsw(pid);
    return result;
  }
  if (!loop.run(server.callback(), &result.trace, &error)) {
    ADD_FAILURE() << error;
  }
  result.sim_healthy = server.healthy();
  server.stop();
  reapFsw(pid);
  result.log = readFile(log_path);
  if (std::getenv("POLARIS_KEEP_SITL_LOGS") == nullptr) {
    const int rm_rc = std::system(("rm -rf '" + work_dir + "'").c_str());
    static_cast<void>(rm_rc);
  }
  return result;
}

#define POLARIS_REQUIRE_SITL_TOOLCHAIN()                                           \
  do {                                                                             \
    if (::access(fswBinaryPath().c_str(), X_OK) != 0) {                            \
      GTEST_SKIP() << "flight binary not built at " << fswBinaryPath()             \
                   << " (run `uv run fprime-util build`, or set POLARIS_FSW_BIN)"; \
    }                                                                              \
    if (::access(pythonPath().c_str(), X_OK) != 0) {                               \
      GTEST_SKIP() << "no Python interpreter at " << pythonPath()                  \
                   << " (run `uv sync`, or set POLARIS_PYTHON)";                   \
    }                                                                              \
  } while (false)

/// Every `OrbitStatus` event in the log. The EVR line carries the sim time as
/// "(3:<seconds>,<micros>)" — the timestamp SitlBridge pushes — which is how a
/// status is placed on the truth trace.
std::vector<Status> statuses(const std::string& log) {
  std::vector<Status> out;
  std::size_t at = 0;
  while ((at = log.find("Orbit status: quality ", at)) != std::string::npos) {
    const std::size_t line_start = log.rfind('\n', at);
    const std::string line =
        log.substr(line_start == std::string::npos ? 0 : line_start + 1, log.find('\n', at) - at);
    Status s;
    // "... (3:1767225937,100000) ..." → seconds since epoch
    const std::size_t tb = line.find("(3:");
    if (tb != std::string::npos) {
      long long sec = 0;
      long micro = 0;
      if (std::sscanf(line.c_str() + tb, "(3:%lld,%ld)", &sec, &micro) == 2) {
        s.t_s = static_cast<double>(sec - kEpochTaiNs / 1000000000LL) + micro * 1.0e-6;
      }
    }
    char q[16] = {};
    if (std::sscanf(log.c_str() + at,
                    "Orbit status: quality %15s (%*d) age %lf s sigma %lf m pos [%lf %lf %lf] m", q,
                    &s.age_s, &s.sigma_m, &s.pos_m.x(), &s.pos_m.y(), &s.pos_m.z()) == 6) {
      s.quality = q;
      out.push_back(s);
    }
    at += 10;
  }
  return out;
}

/// Truth position at @p t_s from the trace (nearest sample; the trace is on the
/// 0.1 s macro grid and the status is emitted on a cycle boundary).
bool truthAt(const RunResult& run, double t_s, Eigen::Vector3d& out) {
  if (run.trace.empty()) {
    return false;
  }
  std::size_t best = 0;
  double best_dt = INFINITY;
  for (std::size_t i = 0; i < run.trace.size(); ++i) {
    const double dt = std::abs(run.trace[i].t_s - t_s);
    if (dt < best_dt) {
      best_dt = dt;
      best = i;
    }
  }
  if (best_dt > 0.2) {
    return false;
  }
  out = run.trace[best].state.position.eigen();
  return true;
}

/// Position error of the last status at or before @p t_s, or NaN.
double errorAtOrBefore(const RunResult& run, double t_s) {
  double err = NAN;
  for (const Status& s : statuses(run.log)) {
    Eigen::Vector3d truth;
    if (s.t_s <= t_s && s.quality != "NONE" && truthAt(run, s.t_s, truth)) {
      err = (s.pos_m - truth).norm();
    }
  }
  return err;
}

double worstErrorFineOrDegraded(const RunResult& run) {
  double worst = 0.0;
  for (const Status& s : statuses(run.log)) {
    Eigen::Vector3d truth;
    if (s.quality != "NONE" && truthAt(run, s.t_s, truth)) {
      worst = std::max(worst, (s.pos_m - truth).norm());
    }
  }
  return worst;
}

unsigned lastRefusalCount(const std::string& log) {
  const std::size_t at = log.rfind(" refusals so far)");
  if (at == std::string::npos) {
    return 0;
  }
  const std::size_t open = log.rfind('(', at);
  return static_cast<unsigned>(std::stoul(log.substr(open + 1, at - open - 1)));
}

void expectBurnHappened(const RunResult& run) {
  EXPECT_NE(run.log.find("Burn started"), std::string::npos) << run.log;
  EXPECT_NE(run.log.find("Burn completed"), std::string::npos) << run.log;
  // The truth plant fired: mass went down.
  ASSERT_FALSE(run.trace.empty());
  EXPECT_LT(run.trace.back().mass_kg, run.trace.front().mass_kg)
      << "the truth thruster never depleted mass";
}

// ----------------------------------------------------------------------
// A burn while tracking (§17 → §8.3)
// ----------------------------------------------------------------------

/// **A finite burn is a fix the filter expects, not a storm it survives.** The
/// executor's commanded acceleration reaches the filter one cycle ahead of the
/// thrust reaching the plant, so the propagate step already carries it: the
/// fixes through the burn are accepted, nothing is dropped, and the tracked
/// error stays at the receiver's noise.
TEST(SitlOdBurn, BurnWhileTrackingIsAbsorbedNotRefused) {
  RecordProperty("verifies", "REQ-ODP-008,REQ-MAN-001");
  POLARIS_REQUIRE_SITL_TOOLCHAIN();
  const RunResult run =
      flyBurn("tracked", faultMatrixOrbit(480.0, "sitl-od-burn-tracked"), burnSuite(), kBurnSpec);
  ASSERT_TRUE(run.sim_healthy);
  ASSERT_FALSE(run.log.empty());
  expectBurnHappened(run);
  EXPECT_NE(run.log.find("NonGravAccelApplied"), std::string::npos)
      << "the filter never saw the executor's acceleration:\n"
      << run.log;
  EXPECT_EQ(countOf(run.log, "Orbit solution seeded"), 1u) << run.log;
  EXPECT_EQ(run.log.find("Orbit solution dropped"), std::string::npos) << run.log;
  EXPECT_EQ(run.log.find("Orbit solution degraded"), std::string::npos) << run.log;
  // Refusals through the burn stay at the gate's false-alarm rate: under 1 % of
  // the run's ~4800 fixes.
  EXPECT_LT(lastRefusalCount(run.log), 48u) << run.log;
  const double worst = worstErrorFineOrDegraded(run);
  RecordProperty("tracked_worst_error_m", std::to_string(worst));
  EXPECT_LT(worst, 30.0) << "the tracked solution left the receiver's noise through the burn";
}

// ----------------------------------------------------------------------
// The same burn inside an outage: told, and blind
// ----------------------------------------------------------------------

/// **Told, the filter coasts through a burn in an outage on the commanded
/// thrust; blind, it does not.** The paper's headline (5 km told vs 9 km blind
/// on a J2 model, 0.03 m/s² for up to 480 s): here a 60 s, 0.042 m/s² burn
/// (2.5 m/s) 260 s into a 900 s outage — the fed filter's error at the end of
/// the outage is metres-class on the 8x8 model, the blind filter's is the
/// kilometre the missing Δv integrates to.
TEST(SitlOdBurn, BurnInsideAnOutageIsCoastedOnTheCommandedThrust) {
  RecordProperty("verifies", "REQ-ODP-008,REQ-ODP-007");
  POLARIS_REQUIRE_SITL_TOOLCHAIN();
  constexpr double kOutageStartS = 40.0;
  constexpr double kOutageStopS = 940.0;  // 900 s: inside the 1800 s degraded horizon
  const scenario::SimConfig orbit =
      withOutage(faultMatrixOrbit(1000.0, "sitl-od-burn-outage"), kOutageStartS, kOutageStopS);
  const RunResult told = flyBurn("outage-told", orbit, burnSuite(), kBurnSpec, /*accel=*/1);
  const RunResult blind = flyBurn("outage-blind", orbit, burnSuite(), kBurnSpec, /*accel=*/0);
  ASSERT_TRUE(told.sim_healthy);
  ASSERT_TRUE(blind.sim_healthy);
  expectBurnHappened(told);
  expectBurnHappened(blind);
  // Both coasted the whole outage on the degraded horizon: degraded at 300 s
  // past the last fix, never dropped, one seed.
  for (const RunResult* r : {&told, &blind}) {
    EXPECT_NE(r->log.find("Orbit solution degraded"), std::string::npos) << r->log;
    EXPECT_EQ(r->log.find("Orbit solution dropped"), std::string::npos) << r->log;
    EXPECT_EQ(countOf(r->log, "Orbit solution seeded"), 1u) << r->log;
  }
  const double err_told = errorAtOrBefore(told, kOutageStopS);
  const double err_blind = errorAtOrBefore(blind, kOutageStopS);
  RecordProperty("outage_end_error_told_m", std::to_string(err_told));
  RecordProperty("outage_end_error_blind_m", std::to_string(err_blind));
  ASSERT_TRUE(std::isfinite(err_told) && std::isfinite(err_blind)) << told.log;
  EXPECT_LT(err_told, 100.0) << "the filter was told the thrust and still lost the coast";
  EXPECT_GT(err_blind, 1000.0) << "the blind filter did not lose the burn's Δv — the burn did "
                                  "not reach the plant, or the -N 0 hook did not bite";
  EXPECT_LT(err_told, 0.05 * err_blind);
  // The blind filter's own covariance does not cover its error (it was never
  // told), so on the fixes' return the gate refuses first; the told filter
  // re-acquires by update. Both end the run with a solution.
  EXPECT_NE(told.log.find("NonGravAccelApplied"), std::string::npos) << told.log;
  EXPECT_EQ(blind.log.find("NonGravAccelApplied"), std::string::npos) << blind.log;
}

// ----------------------------------------------------------------------
// The degraded horizon (§8.3): twenty minutes without a fix
// ----------------------------------------------------------------------

/// **Past the fine horizon the position is degraded, not gone.** Until Push 70
/// the solution was dropped at 300 s and the attitude estimator lost its
/// magnetic pair for no physical reason — a kilometre of position error moves
/// the IGRF reference by ~1e-3 deg. Now: DEGRADED at 300 s, the reference kept
/// (no `PositionUnavailable`), the 8x8 model coasts twenty minutes to metres,
/// and the returning fixes are absorbed on the grown covariance — one seed,
/// no drop.
TEST(SitlOdBurn, TwentyMinuteCoastIsDegradedNotDroppedAndKeepsTheReference) {
  RecordProperty("verifies", "REQ-ODP-007");
  POLARIS_REQUIRE_SITL_TOOLCHAIN();
  constexpr double kOutageStartS = 40.0;
  constexpr double kOutageStopS = 1240.0;  // 1200 s
  const RunResult run =
      flyBurn("coast-20min",
              withOutage(faultMatrixOrbit(1300.0, "sitl-od-coast"), kOutageStartS, kOutageStopS),
              burnSuite(), /*burnSpec=*/nullptr);
  ASSERT_TRUE(run.sim_healthy);
  ASSERT_FALSE(run.log.empty());
  const std::size_t degraded = indexOf(run.log, "Orbit solution degraded");
  ASSERT_NE(degraded, std::string::npos) << run.log;
  EXPECT_EQ(run.log.find("Orbit solution dropped"), std::string::npos)
      << "a 1200 s outage dropped a solution with an 1800 s degraded horizon:\n"
      << run.log;
  EXPECT_EQ(run.log.find("No valid orbit solution"), std::string::npos)
      << "the attitude estimator lost its magnetic reference on a DEGRADED solution:\n"
      << run.log;
  EXPECT_EQ(countOf(run.log, "Orbit solution seeded"), 1u)
      << "the returning fixes re-seeded instead of updating:\n"
      << run.log;
  EXPECT_EQ(run.log.find("Attitude lost"), std::string::npos) << run.log;
  const double err_end = errorAtOrBefore(run, kOutageStopS);
  RecordProperty("coast_end_error_m", std::to_string(err_end));
  ASSERT_TRUE(std::isfinite(err_end)) << run.log;
  EXPECT_LT(err_end, 100.0) << "an 8x8 model coasts twenty minutes to tens of metres, not "
                            << err_end;
  // The degraded edge sits at the fine horizon after the last fix, not before.
  const std::vector<Status> all = statuses(run.log);
  double first_degraded_age = NAN;
  for (const Status& s : all) {
    if (s.quality == "DEGRADED") {
      first_degraded_age = s.age_s;
      break;
    }
  }
  ASSERT_TRUE(std::isfinite(first_degraded_age)) << run.log;
  EXPECT_GT(first_degraded_age, kFineHorizonS);
}

// ----------------------------------------------------------------------
// Receiver delivery latency, with jitter
// ----------------------------------------------------------------------

/// **Latency and its jitter cost nothing, because the fix carries its own
/// epoch.** The paper measured 15 ms +/- 7.5 ms delivery delays on a CubeSat
/// bus and ~150 m of error when uncompensated by a mean-delay advance. The
/// filter here advances each fix by its own tag (r + v tau + 1/2 a tau^2), so a
/// 50 ms latency with 5 ms jitter is invisible: refusals at the false-alarm
/// rate, error at the receiver's noise.
TEST(SitlOdBurn, ReceiverLatencyJitterIsInvisibleToTheFilter) {
  RecordProperty("verifies", "REQ-ODP-001");
  POLARIS_REQUIRE_SITL_TOOLCHAIN();
  const RunResult run = flyBurn("jitter", faultMatrixOrbit(300.0, "sitl-od-jitter"),
                                burnSuite(/*latency=*/0.05, /*jitter=*/0.005), nullptr);
  ASSERT_TRUE(run.sim_healthy);
  ASSERT_FALSE(run.log.empty());
  EXPECT_EQ(countOf(run.log, "Orbit solution seeded"), 1u) << run.log;
  EXPECT_EQ(run.log.find("Orbit solution dropped"), std::string::npos) << run.log;
  EXPECT_EQ(run.log.find("NON_MONOTONIC_EPOCH"), std::string::npos)
      << "a jittered delivery was refused as a clock fault:\n"
      << run.log;
  EXPECT_LT(lastRefusalCount(run.log), 30u) << run.log;  // < 1 % of ~3000 fixes
  const double worst = worstErrorFineOrDegraded(run);
  RecordProperty("jitter_worst_error_m", std::to_string(worst));
  EXPECT_LT(worst, 30.0);
}

}  // namespace
