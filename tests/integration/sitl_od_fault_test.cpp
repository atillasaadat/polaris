/// @file The SITL GNSS / orbit-filter fault rows (design doc §8.3, §9.2, §23.1.1).
///
/// The `OrbitEstimator` seam flown against the truth sim's GNSS fault schedule,
/// one case per row. As in the attitude fault matrix, what is asserted is the
/// **path** — seeded, refused by which name, dropped, re-seeded, and what the
/// attitude estimator's magnetic reference paid — not accuracy, which the
/// `analysis/od` campaign measures against the library.
///
///   fault                                   | asserted path
///   ----------------------------------------|----------------------------------
///   outage inside the fine horizon          | coasts, no drop, no reference lost
///   outage past the fine horizon            | DEGRADED, reference kept, re-acquired by update
///   5 km spoof step inside the horizon      | every spoofed fix refused by the NIS gate
///   5 km spoof step past the fine horizon   | refused throughout on the degraded horizon;
///                                           |   truth accepted on the grown covariance
///   receiver clock 5 s behind               | refused as a stale epoch, by name
///   OD_RESET mid-run                        | dropped by command, re-seeded next fix
///   primary receiver lost, second present   | the second carries it, nothing dropped
///   one orbit period, no fault              | one seed, no drop, refusals stay rare
///
/// Faults go on the scenario's own `gnss_fault_events` schedule, not on a
/// per-step hook: the closed loop reconciles every receiver to the schedule on
/// each sample, so a flag set on the model directly is overwritten before the
/// next fix (`applyGnssFaults` is idempotent by design, and this file learned it).
///
/// The slow ramp — the spoof that walks a filter off its trajectory without a
/// single alarm — is deliberately **not** here: it produces no event at all on
/// the topology, and its measure is the position error the `spoof_ramp`
/// scenario of the MC campaign records against truth (§23.2).
///
/// Skips (never fails) when the flight binary or the Python toolchain is absent.
///
/// Verifies REQ-ODP-007.

#include <gtest/gtest.h>

#include <string>

#include "io/closed_loop.hpp"
#include "scenario/sim_runner.hpp"
#include "sitl_harness.hpp"

namespace {

using namespace polaris::test::sitl;  // NOLINT(build/namespaces) — the shared SITL fixture

/// Orbit-filter coast horizon [s] the vehicle config flies (`MaxCoastS`). Rows
/// sit on each side of it. Pinned rather than read from the YAML so a change to
/// the horizon fails this file with a number.
constexpr double kOrbitCoastHorizonS = 300.0;

/// Faults open here: after the first fixes have seeded the filter (the fixture
/// receiver has no cold start, so the seed lands on the first cycle) and after
/// the attitude side is on its top rung.
constexpr double kFaultStartS = 40.0;

/// Refusal names as the `FixRefused` event prints them (`OdRefusal`).
constexpr const char* kRefusedByNis = "refused: MEASUREMENT_REJECTED";
constexpr const char* kRefusedByEpoch = "refused: NON_MONOTONIC_EPOCH";

struct RunResult {
  std::string log;
  bool sim_healthy = false;
};

scenario::SimConfig withFault(scenario::SimConfig orbit, scenario::GnssFaultEvent::Type type,
                              double startS, double stopS, const char* unit = "gps_a",
                              double spoofM = 0.0, double clockJumpS = 0.0) {
  scenario::GnssFaultEvent ev;
  ev.unit = unit;
  ev.type = type;
  ev.start_s = startS;
  ev.stop_s = stopS;
  ev.spoof_offset_ecef_m = Eigen::Vector3d(spoofM, 0.0, 0.0);
  ev.clock_jump_s = clockJumpS;
  orbit.environment.gnss_fault_events.push_back(ev);
  return orbit;
}

scenario::SimConfig withGnssOutage(scenario::SimConfig orbit, double startS, double stopS,
                                   const char* unit = "gps_a") {
  return withFault(std::move(orbit), scenario::GnssFaultEvent::Type::kOutage, startS, stopS, unit);
}

/// The fault-matrix suite plus a second receiver at index 1, for the failover
/// row. The reference vehicle flies one; the topology wires two so a vehicle
/// that carries two needs no FSW change.
scenario::SpacecraftConfig twoReceiverSuite() {
  scenario::SpacecraftConfig sc = faultMatrixSuite();
  sc.sensors.push_back(unit("gps_b", "NOVATEL-OEM7600", "gnss",
                            {{"horizontal_position_rms_m", 1.2},
                             {"velocity_accuracy_m_s_rms", 0.03},
                             {"max_rate_hz", 10.0}}));
  return sc;
}

/// Compile the vehicle config, fork the deployment, fly @p orbit against it and
/// return the captured event stream. @p resetCycle > 0 commands OD_RESET on that
/// GNC cycle (`-R`). The tracker alignment window is commanded as in the
/// attitude matrix so the attitude side is on the same rung in every row.
RunResult flyOd(const std::string& tag, const scenario::SimConfig& orbit,
                const scenario::SpacecraftConfig& suite, const io::SitlServer::Counts& counts,
                unsigned resetCycle = 0) {
  RunResult result;
  const std::string work_dir = "build-artifacts/test-od-" + tag + "-" + std::to_string(::getpid());
  const std::string err_path = work_dir + "/configc.err";
  const std::string prm_path = work_dir + "/PrmDb.dat";
  const std::string log_path = work_dir + "/fsw.log";
  if (compileConfig(work_dir, err_path) != 0) {
    ADD_FAILURE() << "config compiler failed:\n" << readFile(err_path);
    return result;
  }
  io::SitlServer server(counts, 100'000'000LL);
  if (!server.start(0)) {
    ADD_FAILURE() << server.lastError();
    return result;
  }
  const pid_t pid =
      spawnFsw(fswBinaryPath(), server.port(), prm_path, log_path, /*magCalSamples=*/0,
               /*stAlignPairs=*/100, /*stAlignUnit=*/1, /*ctrlMode=*/0, nullptr, -1, resetCycle);
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
  std::vector<io::MacroSample> trace;
  if (!loop.run(server.callback(), &trace, &error)) {
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

RunResult flyOd(const std::string& tag, const scenario::SimConfig& orbit, unsigned resetCycle = 0) {
  return flyOd(tag, orbit, faultMatrixSuite(), faultMatrixCounts(), resetCycle);
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

/// The refusal counter the last `FixRefused` event printed ("(N refusals so far)"),
/// or 0 when none fired. The event is edge-gated on the reason, so this is a
/// lower bound on the refusals — enough to tell "rare" from "every cycle".
unsigned lastRefusalCount(const std::string& log) {
  const std::size_t at = log.rfind(" refusals so far)");
  if (at == std::string::npos) {
    return 0;
  }
  const std::size_t open = log.rfind('(', at);
  return static_cast<unsigned>(std::stoul(log.substr(open + 1, at - open - 1)));
}

/// The common health of every row: the run happened, the filter seeded, and the
/// attitude was never lost — the orbit filter can only ever cost the magnetic
/// pair, never the attitude solution.
void expectHealthyRun(const RunResult& run) {
  ASSERT_TRUE(run.sim_healthy);
  ASSERT_FALSE(run.log.empty());
  ASSERT_NE(run.log.find("Orbit solution seeded"), std::string::npos) << run.log;
  EXPECT_EQ(run.log.find("Attitude lost"), std::string::npos) << run.log;
  EXPECT_EQ(run.log.find("Orbit filter fault"), std::string::npos) << run.log;
}

// ----------------------------------------------------------------------
// Outages (§9.2)
// ----------------------------------------------------------------------

/// **A short outage costs nothing.** Sixty seconds without a fix is a fifth of
/// the coast horizon: the solution stays valid on propagation, the attitude
/// estimator keeps its magnetic reference, and no FDIR edge fires anywhere. Until
/// Push 65 the attitude estimator read the receiver directly and the *first*
/// missed fix cost the magnetic pair.
TEST(SitlOdFault, GnssOutageInsideTheHorizonCostsNoReference) {
  RecordProperty("verifies", "REQ-ODP-007");
  POLARIS_REQUIRE_SITL_TOOLCHAIN();
  const RunResult run =
      flyOd("gnss-short",
            withGnssOutage(faultMatrixOrbit(120.0, "sitl-od-gnss-short"), kFaultStartS, 100.0));
  expectHealthyRun(run);
  EXPECT_EQ(countOf(run.log, "Orbit solution seeded"), 1u) << run.log;
  EXPECT_EQ(run.log.find("Orbit solution dropped"), std::string::npos)
      << "a 60 s outage dropped a solution with a 300 s coast horizon:\n"
      << run.log;
  EXPECT_EQ(run.log.find("No valid orbit solution"), std::string::npos)
      << "the attitude estimator lost its magnetic reference across a gap the filter should "
         "have coasted:\n"
      << run.log;
}

/// **A long outage is degraded, not dropped.** Until Push 70 the solution was
/// dropped at the 300 s horizon and the attitude estimator lost its magnetic
/// pair for no physical reason. Now past the fine horizon the solution is
/// DEGRADED — coasted with the covariance the process noise grows — the
/// attitude estimator keeps its reference (its own tolerance is 20 km of
/// sigma, and an 8x8 model coasts metres), and the returning fixes are
/// absorbed by update: one seed, no drop. The drop lives 1800 s out
/// (`TwentyMinuteCoast…` in sitl_od_burn_test.cpp measures the coast itself).
TEST(SitlOdFault, GnssOutagePastTheFineHorizonIsDegradedNotDropped) {
  RecordProperty("verifies", "REQ-ODP-007");
  POLARIS_REQUIRE_SITL_TOOLCHAIN();
  constexpr double kStopS = kFaultStartS + kOrbitCoastHorizonS + 30.0;  // 370 s
  const RunResult run =
      flyOd("gnss-long",
            withGnssOutage(faultMatrixOrbit(430.0, "sitl-od-gnss-long"), kFaultStartS, kStopS));
  expectHealthyRun(run);
  const std::size_t seeded = indexOf(run.log, "Orbit solution seeded");
  const std::size_t degraded = indexOf(run.log, "Orbit solution degraded");
  ASSERT_NE(degraded, std::string::npos) << "the solution never crossed the fine horizon:\n"
                                         << run.log;
  EXPECT_LT(seeded, degraded) << run.log;
  EXPECT_EQ(run.log.find("Orbit solution dropped"), std::string::npos)
      << "a 330 s outage dropped a solution with an 1800 s degraded horizon:\n"
      << run.log;
  EXPECT_EQ(run.log.find("No valid orbit solution"), std::string::npos)
      << "the attitude estimator lost its magnetic reference on a DEGRADED solution:\n"
      << run.log;
  EXPECT_EQ(countOf(run.log, "Orbit solution seeded"), 1u)
      << "the returning fixes re-seeded instead of updating the degraded solution:\n"
      << run.log;
}

// ----------------------------------------------------------------------
// Spoofing (§9.2): the fix stays valid and lies
// ----------------------------------------------------------------------

/// **A spoof step inside the horizon is refused, not followed.** 5 km is ~3000σ
/// of the receiver's position noise; the NIS gate refuses every spoofed fix by
/// name and the solution coasts through on propagation, exactly as it would an
/// outage — so nothing is dropped and the magnetic reference never goes.
TEST(SitlOdFault, SpoofStepInsideTheHorizonIsRefusedNotFollowed) {
  RecordProperty("verifies", "REQ-ODP-007");
  POLARIS_REQUIRE_SITL_TOOLCHAIN();
  const RunResult run =
      flyOd("spoof-short", withFault(faultMatrixOrbit(120.0, "sitl-od-spoof-short"),
                                     scenario::GnssFaultEvent::Type::kSpoof, kFaultStartS, 100.0,
                                     "gps_a", /*spoofM=*/5000.0));
  expectHealthyRun(run);
  EXPECT_EQ(countOf(run.log, "Orbit solution seeded"), 1u) << run.log;
  EXPECT_NE(run.log.find(kRefusedByNis), std::string::npos)
      << "a 5 km step was folded into the solution:\n"
      << run.log;
  EXPECT_EQ(run.log.find("Orbit solution dropped"), std::string::npos) << run.log;
  EXPECT_EQ(run.log.find("No valid orbit solution"), std::string::npos) << run.log;
}

/// **A spoof that outlasts the fine horizon no longer wins the seed.** Until
/// Push 70 the filter refused the lie for 300 s, dropped the solution, and
/// re-seeded from the next fix — the spoof — after which truth was the
/// outlier. On the degraded horizon the filter keeps refusing the spoof for as
/// long as it lasts (a 5 km step is ~3000σ of a covariance that grows by
/// centimetres a minute), and when truth returns it is accepted on that grown
/// covariance: one seed, no drop, no re-seed onto the lie. The residual moves
/// out to a spoof longer than the degraded horizon (1800 s), where a filter
/// with no solution still has no basis to refuse one — the raw-measurement
/// path §8.3 records as owed is what closes that.
TEST(SitlOdFault, SpoofStepPastTheFineHorizonIsRefusedThroughout) {
  RecordProperty("verifies", "REQ-ODP-007");
  POLARIS_REQUIRE_SITL_TOOLCHAIN();
  constexpr double kStopS = kFaultStartS + kOrbitCoastHorizonS + 30.0;  // 370 s
  const RunResult run =
      flyOd("spoof-long", withFault(faultMatrixOrbit(430.0, "sitl-od-spoof-long"),
                                    scenario::GnssFaultEvent::Type::kSpoof, kFaultStartS, kStopS,
                                    "gps_a", /*spoofM=*/5000.0));
  expectHealthyRun(run);
  EXPECT_NE(run.log.find("Orbit solution degraded"), std::string::npos) << run.log;
  EXPECT_EQ(run.log.find("Orbit solution dropped"), std::string::npos)
      << "the spoof was followed instead of refused:\n"
      << run.log;
  EXPECT_EQ(countOf(run.log, "Orbit solution seeded"), 1u)
      << "the filter re-seeded — onto the spoof or onto truth, neither is right:\n"
      << run.log;
  EXPECT_NE(run.log.find(kRefusedByNis), std::string::npos) << run.log;
  EXPECT_EQ(run.log.find("No valid orbit solution"), std::string::npos) << run.log;
}

// ----------------------------------------------------------------------
// Receiver clock (§9.2)
// ----------------------------------------------------------------------

/// **A clock 5 s behind makes every fix stale, and it is refused by that name.**
/// The latency correction is bounded at `MaxFixLatencyS`; a fix older than that
/// is `NON_MONOTONIC_EPOCH`, not `MEASUREMENT_REJECTED` — an FDIR rule keyed on
/// the receiver's time going wrong must see the receiver's time going wrong.
/// Inside the horizon the solution coasts and nothing is dropped.
TEST(SitlOdFault, ClockJumpIsRefusedAsAStaleEpoch) {
  RecordProperty("verifies", "REQ-ODP-007");
  POLARIS_REQUIRE_SITL_TOOLCHAIN();
  const RunResult run =
      flyOd("clock", withFault(faultMatrixOrbit(120.0, "sitl-od-clock"),
                               scenario::GnssFaultEvent::Type::kClockJump, kFaultStartS, 100.0,
                               "gps_a", 0.0, /*clockJumpS=*/-5.0));
  expectHealthyRun(run);
  EXPECT_NE(run.log.find(kRefusedByEpoch), std::string::npos)
      << "a fix 5 s stale was not refused as a stale epoch:\n"
      << run.log;
  EXPECT_EQ(run.log.find("Orbit solution dropped"), std::string::npos) << run.log;
  EXPECT_EQ(countOf(run.log, "Orbit solution seeded"), 1u) << run.log;
}

// ----------------------------------------------------------------------
// Command and redundancy
// ----------------------------------------------------------------------

/// **OD_RESET drops the solution by command and the next fix re-seeds it.**
/// The reset is the ground's tool for a filter it no longer trusts; what it
/// must not do is leave the vehicle without a solution for longer than one fix.
TEST(SitlOdFault, ResetCommandDropsAndReseedsOnTheNextFix) {
  RecordProperty("verifies", "REQ-ODP-007");
  POLARIS_REQUIRE_SITL_TOOLCHAIN();
  const RunResult run = flyOd("reset", faultMatrixOrbit(120.0, "sitl-od-reset"),
                              /*resetCycle=*/600);  // 60 s in
  expectHealthyRun(run);
  const std::size_t reset = indexOf(run.log, "OrbitReset");
  ASSERT_NE(reset, std::string::npos) << "OD_RESET never ran:\n" << run.log;
  EXPECT_EQ(countOf(run.log, "Orbit solution seeded"), 2u)
      << "the filter did not re-seed after the commanded reset:\n"
      << run.log;
  EXPECT_LT(reset, run.log.rfind("Orbit solution seeded")) << run.log;
  EXPECT_EQ(run.log.find("Orbit solution dropped"), std::string::npos)
      << "a commanded reset was reported as a coast expiry:\n"
      << run.log;
}

/// **A second receiver carries a primary outage.** The estimator folds in the
/// freshest valid fix across its receivers, so losing gps_a for longer than the
/// horizon costs nothing when gps_b is present: one seed, no drop, no reference
/// lost. The reference vehicle flies one receiver; this is the row that says the
/// topology is ready for two.
TEST(SitlOdFault, SecondReceiverCarriesAPrimaryOutage) {
  RecordProperty("verifies", "REQ-ODP-007");
  POLARIS_REQUIRE_SITL_TOOLCHAIN();
  io::SitlServer::Counts counts = faultMatrixCounts();
  counts.gnss = 2;
  constexpr double kStopS = kFaultStartS + kOrbitCoastHorizonS + 30.0;  // 370 s
  const RunResult run = flyOd(
      "failover",
      withGnssOutage(faultMatrixOrbit(430.0, "sitl-od-failover"), kFaultStartS, kStopS, "gps_a"),
      twoReceiverSuite(), counts);
  expectHealthyRun(run);
  EXPECT_EQ(countOf(run.log, "Orbit solution seeded"), 1u) << run.log;
  EXPECT_EQ(run.log.find("Orbit solution dropped"), std::string::npos)
      << "the second receiver did not carry the solution through the primary's outage:\n"
      << run.log;
  EXPECT_EQ(run.log.find("No valid orbit solution"), std::string::npos) << run.log;
}

// ----------------------------------------------------------------------
// Horizon: one orbit period, no fault
// ----------------------------------------------------------------------

/// **One orbit, one solution.** ~95 min on the 8×8 truth plant with the onboard
/// model at the same degree: the filter seeds once, never drops, never faults,
/// and the NIS gate's refusals stay at the false-alarm rate a 0.999 gate implies
/// (well under 1 % of ~57 000 fixes) rather than tracking a divergence. This is
/// the horizon the component rows above do not reach. Measured: 75 refusals
/// over the orbit, 0.13 % — the gate's own false-alarm rate, nothing more.
TEST(SitlOdFault, OneOrbitPeriodHoldsOneSolution) {
  RecordProperty("verifies", "REQ-ODP-007");
  POLARIS_REQUIRE_SITL_TOOLCHAIN();
  constexpr double kOrbitPeriodS = 5700.0;
  const RunResult run = flyOd("orbit", faultMatrixOrbit(kOrbitPeriodS, "sitl-od-orbit"));
  expectHealthyRun(run);
  EXPECT_EQ(countOf(run.log, "Orbit solution seeded"), 1u) << run.log;
  EXPECT_EQ(run.log.find("Orbit solution dropped"), std::string::npos) << run.log;
  EXPECT_EQ(run.log.find("No valid orbit solution"), std::string::npos) << run.log;
  EXPECT_EQ(run.log.find(kRefusedByEpoch), std::string::npos) << run.log;
  EXPECT_LT(lastRefusalCount(run.log), 570u)
      << "the NIS gate refused more than 1 % of an orbit's fixes:\n"
      << run.log;
}

}  // namespace
