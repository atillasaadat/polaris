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

#include <cmath>
#include <cstdlib>
#include <limits>
#include <string>

#include "io/closed_loop.hpp"
#include "scenario/sim_runner.hpp"
#include "sitl_harness.hpp"

namespace {

using namespace polaris::test::sitl;  // NOLINT(build/namespaces) — the shared SITL fixture

/// 10 s at the 10 Hz GNC rate: 100 barriers. The estimator acquires on its
/// first valid TRIAD, so this is generous.
constexpr double kDurationS = 10.0;

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
  EXPECT_NE(log.find("Records: 100"), std::string::npos)
      << "prmDb loaded a record count other than the 100 declared parameters "
         "(55 estimator + 45 controller, Push 56):\n"
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
  // The fourth gate (Push 52): a vehicle that never reaches the star-tracker rung
  // of the §8.2 ladder because its fusion tuning never arrived flies at
  // REQ-ADET-006 accuracy while its telemetry says "fine mode" — the same silent
  // degradation, two orders of magnitude down.
  EXPECT_EQ(log.find("no tracker fused"), std::string::npos)
      << "estimator refused the compiled star-tracker tuning:\n"
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

// ── §8.2 star-tracker fusion on the vehicle (REQ-ADET-007, REQ-ADET-012/013) ──
//
// What these two add over the component harness is the *plant*: the trackers'
// solutions come out of the real `sim::sensors::StarTracker` with its own error
// stack and its own availability state machine, through the SITL wire, into the
// real mode ladder. The harness proves the component's logic; this proves the
// vehicle reaches the top rung on hardware that behaves like hardware.

namespace {

/// Pairs the on-orbit alignment window collects: 100, the reference vehicle's
/// `StAlignMinSamples`, i.e. 10 s at the 10 Hz rate with both trackers solving.
/// The AURIGA needs `lost_in_space_s` = 3.8 s to acquire from cold, so the run
/// has to be long enough for acquisition *plus* the window.
constexpr unsigned kAlignPairs = 100;
constexpr double kTrackerDurationS = 30.0;

/// The estimation orbit flown **nadir-pointing**, which is the attitude the
/// tracker mounting was designed against, and run long enough for both units to
/// acquire and an alignment window to close.
///
/// The sun-pointing attitude the other cases in this file use is the wrong one
/// here, and instructively so: it puts +Z on the Sun, which leaves the two
/// tracker boresights — 135° from +Z — pointing wherever the Earth happens to be,
/// and one of them lands inside the AURIGA's 22° Earth exclusion. That is not a
/// defect, it is the keep-out geometry doing its job: the mounting is chosen so
/// that in the **Earth-pointing** attitude both boresights sit 135° from nadir,
/// which is 45° clear of the limb (§8.2, and the derivation in the vehicle YAML).
///
/// With +Z on nadir the sun sensor faces the Earth and there is no sun pair at
/// all, so the coarse chain never acquires. That makes this the stronger test
/// rather than a weaker one: the fine mode has to be seeded by a tracker on its
/// own, with no Davenport solve and no coarse floor underneath — the eclipse and
/// cold-start path, exercised on the vehicle.
scenario::SimConfig trackerOrbit() {
  scenario::SimConfig c = estimationOrbit();
  c.scenario_name = "sitl-star-tracker-fusion";
  // Body <- ECI with body +Z on nadir. The vehicle sits at ECI +X, so nadir is
  // -X; body +X is placed on +Z_ECI and body +Y on +Y_ECI, which is right-handed
  // (Z_ECI x Y_ECI = -X_ECI). Both tracker boresights are then 135 deg from nadir.
  Eigen::Matrix3d dcm;
  dcm.row(0) = Eigen::Vector3d::UnitZ();
  dcm.row(1) = Eigen::Vector3d::UnitY();
  dcm.row(2) = -Eigen::Vector3d::UnitX();
  c.initial_state.attitude =
      pm::Quat<pm::frames::Body, pm::frames::ECI>(pm::Quaternion::FromRotationMatrix(dcm));
  // Inertially fixed. Over 30 s the nadir direction moves ~1.9 deg, far inside the
  // 45 deg of Earth-exclusion margin, and a zero rate keeps the AURIGA inside its
  // 2 deg/s acquisition envelope from the first cycle.
  c.initial_state.body_rate = pm::Vec3<pm::frames::Body>(Eigen::Vector3d::Zero());
  c.propagation.duration_s = kTrackerDurationS;
  c.propagation.output_step_s = kTrackerDurationS;
  return c;
}

}  // namespace

TEST(SitlStarTracker, CommandedAlignmentAndDualTrackerFineMode) {
  const std::string bin = fswBinaryPath();
  if (::access(bin.c_str(), X_OK) != 0) {
    GTEST_SKIP() << "flight binary not built at " << bin
                 << " (run `uv run fprime-util build`, or set POLARIS_FSW_BIN)";
  }
  if (::access(pythonPath().c_str(), X_OK) != 0) {
    GTEST_SKIP() << "no Python interpreter at " << pythonPath()
                 << " (run `uv sync`, or set POLARIS_PYTHON)";
  }

  const std::string work_dir = "build-artifacts/test-st-fusion-" + std::to_string(::getpid());
  const std::string err_path = work_dir + "/configc.err";
  const std::string prm_path = work_dir + "/PrmDb.dat";
  const std::string log_path = work_dir + "/fsw.log";
  ASSERT_EQ(compileConfig(work_dir, err_path), 0) << "config compiler failed:\n"
                                                  << readFile(err_path);

  io::SitlServer server(trackerCounts(), 100'000'000LL);
  ASSERT_TRUE(server.start(0)) << server.lastError();

  // -A commands ST_ALIGN_CAL_START on unit 1 (st_b, against the king st_a). On a
  // flight vehicle this comes from the ground; the deployment here has no uplink,
  // and what is under test is the collect-fit-apply chain, not the radio.
  const pid_t pid = spawnFsw(bin, server.port(), prm_path, log_path, /*magCalSamples=*/0,
                             kAlignPairs, /*stAlignUnit=*/1);
  ASSERT_GE(pid, 0);

  scenario::Vehicle vehicle;
  std::string error;
  ASSERT_TRUE(scenario::buildVehicle(trackerSuite(), 1, vehicle, &error)) << error;
  scenario::SimRunner runner;
  io::ClosedLoop loop(runner, vehicle);
  ASSERT_TRUE(runner.build(trackerOrbit(), scenario::DataPaths::under(POLARIS_GOLDEN_DIR), &error,
                           loop.wrench()))
      << error;
  std::vector<io::MacroSample> trace;
  ASSERT_TRUE(loop.run(server.callback(), &trace, &error)) << error;
  EXPECT_TRUE(server.healthy()) << server.lastError();
  server.stop();
  reapFsw(pid);

  const std::string log = readFile(log_path);
  ASSERT_FALSE(log.empty()) << "no event stream captured at " << log_path;

  // 1. The tracker tuning was accepted. Its own gate, so a failure here says
  //    "the ladder is capped" rather than "the estimator is down".
  EXPECT_EQ(log.find("no tracker fused"), std::string::npos)
      << "estimator refused the compiled star-tracker tuning:\n"
      << log;

  // 2. The vehicle reached the **top rung**, on solutions from the real tracker
  //    model rather than a harness stub.
  EXPECT_NE(log.find("Fine-mode source changed"), std::string::npos)
      << "the ladder never moved off SUN_MAG — check the tracker keep-out geometry:\n"
      << log;
  EXPECT_NE(log.find("-> STAR_TRACKER"), std::string::npos) << "no cycle fused a star tracker:\n"
                                                            << log;

  // 3. The commanded alignment ran end to end: window opened, pairs collected,
  //    fit accepted and applied.
  EXPECT_NE(log.find("Inter-tracker alignment collection started on unit 1"), std::string::npos)
      << "the startup ST_ALIGN_CAL_START never opened a window:\n"
      << log;
  EXPECT_NE(log.find("Inter-tracker alignment fitted on unit 1"), std::string::npos)
      << "the alignment window never produced an accepted fit:\n"
      << log;
  EXPECT_EQ(log.find("alignment on unit 1 rejected"), std::string::npos)
      << "the alignment fit was refused:\n"
      << log;

  // 4. Nothing was lost getting there. A rung change is not a demotion: the
  //    filter keeps its state and the published solution stays valid, so neither
  //    a demotion nor an AttitudeLost may appear.
  // Acquisition here is the *tracker's*: with +Z on nadir there is no sun pair, so
  // the coarse chain never solves and the fine mode is seeded from a tracker
  // solution alone — the path that makes the top rung reachable in eclipse.
  EXPECT_NE(log.find("Attitude acquired"), std::string::npos) << log;
  EXPECT_NE(log.find("Fine mode engaged"), std::string::npos) << log;
  EXPECT_EQ(log.find("Attitude lost"), std::string::npos)
      << "the vehicle lost its attitude during tracker fusion:\n"
      << log;
  EXPECT_EQ(log.find("Fine mode demoted"), std::string::npos)
      << "fine mode was demoted over a healthy run — a real tuning defect:\n"
      << log;

  // 5. And the sources the ladder demoted did not start alerting. The sun and
  //    magnetic pairs are healthy here, so their monitors must stay quiet; an
  //    alert would mean the residual is being computed against the wrong frame,
  //    which is the mistake this whole path is easiest to make.
  EXPECT_EQ(log.find("Residual monitor"), std::string::npos)
      << "a residual monitor alerted on healthy sources — suspect the frame the "
         "residual is computed in:\n"
      << log;
}
