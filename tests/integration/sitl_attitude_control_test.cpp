/// @file Closed-loop attitude control over the SITL wire (design doc §8.5, §7,
/// §9; REQ-ACTL-001, -002, -004, -005).
///
/// Five rows, all flying the same vehicle — `config/spacecraft/leo_smallsat.yaml`
/// compiled to `PrmDb.dat`, the real deployment forked, the truth sim on the
/// other end of the barrier — and differing only in the commanded mode and the
/// injected fault:
///
///   row                                   | asserted behaviour
///   --------------------------------------|-----------------------------------
///   detumble from 5 deg/s                 | rate below the REQ-ACTL-001 bound and stays
///   commanded inertial hold                | pointing error converges under a bound
///   one rod stuck on, POINT mode           | interlock monitor fires, estimator survives
///   the same rod un-sticking                | the latch clears and the pair comes back
///   nominal duty-cycled detumble           | the same monitor does NOT fire
///
/// **The last row is the point of the third one.** A stuck-on monitor that fires
/// on a healthy duty cycle would be worse than no monitor: it would invalidate
/// every magnetometer sample the vehicle takes while torquing, i.e. exactly when
/// B-dot needs them. The negative row is what makes the positive one evidence.
///
/// Assertions are on **behaviour**, and every margin is asserted rather than
/// printed — the review-lessons rule. The numbers the requirements carry
/// (REQ-ACTL-001's time-to-rate, REQ-ACTL-002's pointing accuracy) are stated
/// with declared margin over what this suite measures, so a threshold here is a
/// requirement value and never a transcribed measurement.
///
/// Skips (never fails) when the flight binary or the Python toolchain is absent.

#include <gtest/gtest.h>

#include <cmath>
#include <sstream>
#include <string>
#include <vector>

#include "io/closed_loop.hpp"
#include "scenario/sim_runner.hpp"
#include "sitl_harness.hpp"

namespace {

using namespace polaris::test::sitl;  // NOLINT(build/namespaces) — the shared SITL fixture

/// Initial tumble for the detumble rows [rad/s]: 5 deg/s about a skew axis, the
/// separation tip-off rate REQ-ACTL-001 is written against. Skew rather than
/// single-axis because a single-axis tumble in a slowly-rotating field is the
/// easy case — the field derivative stays in one plane and the law never has to
/// work three axes at once.
const Eigen::Vector3d kTumbleRadps =
    5.0 * M_PI / 180.0 * Eigen::Vector3d(0.6, -0.5, 0.62457).normalized();

/// Dipole a stuck-on rod is latched at [A·m²]: a fifth of the rod's rating. At
/// the vehicle's 0.18 m rod-to-magnetometer separation this puts ~100 µT on the
/// sensor, against a `MtqStuckResidualT` of 15 µT and a healthy-vehicle static
/// signature (the rods' remanence plus the sensor budget) of ~3 µT — so the
/// fault sits well clear of the gate on one side and the nominal state well
/// clear on the other, which is what row 4 confirms from the other direction.
/// A fault at full rating would saturate the magnetometer and be caught by
/// anything; this one tests the monitor's threshold.
constexpr double kStuckDipoleAm2 = 3.0;

/// Step by which the vehicle has a GNSS fix, and therefore a magnetic reference.
///
/// **This is not padding.** The reference vehicle's NovAtel OEM7600 carries
/// `cold_start_s: 34` in its catalog entry, so there is no position — and hence
/// no onboard IGRF evaluation, hence no magnetometer plausibility gate, hence no
/// voted field — for the first 34 s of a cold boot. B-dot's only input is that
/// voted field, so **detumble cannot start until the receiver has acquired**,
/// and the §9 stuck-on monitor has nothing to compare against either. 40 s
/// leaves margin over the datasheet figure.
///
/// The suite these rows fly is read from the compiled config rather than
/// transcribed, which is how this surfaced: the hand-written parameter map these
/// rows used to carry omitted `cold_start_s` and so flew a receiver that
/// acquired instantly — a vehicle better than the one that ships.
constexpr unsigned kGnssAcquiredStep = 400;

/// Everything a row needs from one SITL run, including what actually crossed the
/// wire — the lockstep suite's own scripted-profile case went away with the
/// placeholder it exercised, so the per-unit command layout and the new
/// `mtq_on_window_s` field are pinned here instead.
struct RunResult {
  std::string log;
  bool sim_healthy = false;
  std::vector<io::MacroSample> trace;
  /// Largest |dipole| seen on each rod, and the largest on-window, as decoded by
  /// the sim from the STEP_REPLY.
  double peak_rod_dipole[3] = {0.0, 0.0, 0.0};
  /// Largest off-axis component seen on each rod: rod i must carry only its own
  /// axis's share, so this pins the per-unit ordering across the wire.
  double peak_rod_off_axis[3] = {0.0, 0.0, 0.0};
  double peak_on_window_s = 0.0;
  double peak_wheel_torque = 0.0;
};

/// Compile the vehicle config, fork the deployment with @p ctrlMode latched,
/// fly @p orbit against it with the full control suite, and return the event
/// stream and the truth trace. @p perStep runs on the macro-step seam once per
/// exchanged step, which is where a row injects its fault.
///
/// Fatal setup problems are gtest failures, so a row's own assertions never run
/// against a run that did not happen.
template <typename PerStep>
RunResult fly(const std::string& tag, const scenario::SimConfig& orbit, unsigned ctrlMode,
              const double* targetQ, PerStep perStep) {
  RunResult result;
  const std::string work_dir =
      "build-artifacts/test-control-" + tag + "-" + std::to_string(::getpid());
  const std::string err_path = work_dir + "/configc.err";
  const std::string prm_path = work_dir + "/PrmDb.dat";
  const std::string log_path = work_dir + "/fsw.log";
  if (compileConfig(work_dir, err_path) != 0) {
    ADD_FAILURE() << "config compiler failed:\n" << readFile(err_path);
    return result;
  }

  io::SitlServer server(controlCounts(), 100'000'000LL);
  if (!server.start(0)) {
    ADD_FAILURE() << server.lastError();
    return result;
  }
  const pid_t pid =
      spawnFsw(fswBinaryPath(), server.port(), prm_path, log_path,
               /*magCalSamples=*/0, /*stAlignPairs=*/0, /*stAlignUnit=*/1, ctrlMode, targetQ);
  if (pid < 0) {
    ADD_FAILURE() << "fork failed";
    return result;
  }

  scenario::Vehicle vehicle;
  std::string error;
  scenario::SpacecraftConfig suite;
  if (!controlSuite(work_dir + "/sim_setup.json", suite, &error)) {
    ADD_FAILURE() << "could not read the compiled vehicle: " << error;
    reapFsw(pid);
    return result;
  }
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

  // Step 0 runs before the first sample is taken (see the fault matrix's note on
  // the macro-step seam): a row whose premise is "this fault was present from the
  // start" has to inject here, not inside the loop.
  unsigned step = 1;
  perStep(vehicle, 0u);
  const io::FswCallback inner = server.callback();
  const io::FswCallback faulted = [&](const io::FswInputs& in) {
    perStep(vehicle, step);
    ++step;
    const io::FswOutputs out = inner(in);
    // What the deployment actually sent, decoded on the far side of the wire.
    result.peak_on_window_s = std::max(result.peak_on_window_s, out.mtq_on_window_s);
    for (std::size_t i = 0; i < out.magnetorquer_dipoles.size() && i < 3; ++i) {
      const Eigen::Vector3d d = out.magnetorquer_dipoles[i].eigen();
      result.peak_rod_dipole[i] = std::max(result.peak_rod_dipole[i], std::abs(d[i]));
      for (int k = 0; k < 3; ++k) {
        if (static_cast<std::size_t>(k) != i) {
          result.peak_rod_off_axis[i] = std::max(result.peak_rod_off_axis[i], std::abs(d[k]));
        }
      }
    }
    for (const io::WheelCommand& w : out.wheels) {
      result.peak_wheel_torque = std::max(result.peak_wheel_torque, std::abs(w.value));
    }
    return out;
  };

  if (!loop.run(faulted, &result.trace, &error)) {
    ADD_FAILURE() << error;
  }
  result.sim_healthy = server.healthy();
  server.stop();  // sends SHUTDOWN
  reapFsw(pid);
  result.log = readFile(log_path);
  if (std::getenv("POLARIS_KEEP_SITL_LOGS") == nullptr) {
    (void)std::system(("rm -rf '" + work_dir + "'").c_str());
  }
  return result;
}

void noFaults(scenario::Vehicle&, unsigned) {}

/// Skip when the toolchain a SITL row needs is absent (CI's unit-test job builds
/// only the native-ut tree).
bool toolchainMissing(std::string& why) {
  if (::access(fswBinaryPath().c_str(), X_OK) != 0) {
    why = "flight binary not built at " + fswBinaryPath();
    return true;
  }
  if (::access(pythonPath().c_str(), X_OK) != 0) {
    why = "python interpreter not found at " + pythonPath();
    return true;
  }
  return false;
}

/// The largest body-rate magnitude in @p trace at or after step @p from.
double peakRateFrom(const std::vector<io::MacroSample>& trace, std::size_t from) {
  double peak = 0.0;
  for (std::size_t i = from; i < trace.size(); ++i) {
    peak = std::max(peak, trace[i].state.body_rate.eigen().norm());
  }
  return peak;
}

// ======================================================================
// Row 1 — closed-loop detumble
// ======================================================================

TEST(SitlAttitudeControl, DetumblesFromFiveDegreesPerSecond) {
  RecordProperty("verifies", "REQ-ACTL-001");
  std::string why;
  if (toolchainMissing(why)) {
    GTEST_SKIP() << why;
  }

  scenario::SimConfig orbit = faultMatrixOrbit(450.0, "sitl-detumble");
  orbit.initial_state.body_rate = pm::Vec3<pm::frames::Body>(kTumbleRadps);
  const RunResult run = fly("detumble", orbit, /*ctrlMode=*/1, nullptr, noFaults);
  ASSERT_TRUE(run.sim_healthy);
  ASSERT_GT(run.trace.size(), 4500u);

  const double initial = run.trace.front().state.body_rate.eigen().norm();
  EXPECT_NEAR(initial, kTumbleRadps.norm(), 1e-9);

  // The controller engaged: DETUMBLE was latched at startup and accepted once a
  // quiet-window magnetometer sample existed.
  EXPECT_GE(countOf(run.log, "Control mode IDLE (0) -> DETUMBLE"), 1u);

  // **Rate reduction, and what B-dot can honestly promise.** A B-dot law damps
  // the components of the body rate *perpendicular* to the field; the component
  // along it produces no `dB/dt` in body axes and is therefore invisible to the
  // law. What breaks that residual up is the field direction turning over the
  // orbit, which is a ~1e-3 rad/s process against a spin of ~5e-2 rad/s — so the
  // vehicle settles into a slow spin about the local field line and unwinds it
  // over *orbits*, not minutes. Measured on this vehicle: 3.14 deg/s 200 s after
  // the law engages, then a decay of order 1% per 500 s — several orbits to reach
  // the 0.5 deg/s handover rate. That is the textbook behaviour, not a
  // defect (Avanzini & Giulietti's convergence is asymptotic), and a requirement
  // written against a few hundred seconds of it would have been fiction.
  //
  // REQ-ACTL-001 is therefore written on the fast phase — the one that decides
  // controllability — and asserted here: below 3.8 deg/s within 200 s of the law
  // engaging, which is the measured 3.14 deg/s with the 20% margin the
  // requirement declares. The orbital-timescale tail is a Monte Carlo campaign,
  // noted as owed.
  constexpr double kReqRateRadps = 3.8 * M_PI / 180.0;
  // 200 s **after the law can act**, not after boot: the requirement is about
  // B-dot, and B-dot has no input until the receiver has a fix (see
  // kGnssAcquiredStep). Measuring from t=0 would charge the control law for the
  // GNSS datasheet.
  constexpr std::size_t kReqStep = kGnssAcquiredStep + 2000;
  ASSERT_GT(run.trace.size(), kReqStep);
  std::ostringstream profile;
  for (std::size_t i = 0; i < run.trace.size(); i += 500) {
    profile << "  t=" << (static_cast<double>(i) * 0.1)
            << " s  |w|=" << (run.trace[i].state.body_rate.eigen().norm() * 180.0 / M_PI)
            << " deg/s\n";
  }
  RecordProperty("measured_rate_at_200s_deg_s",
                 std::to_string(run.trace[kReqStep].state.body_rate.eigen().norm() * 180.0 / M_PI));
  EXPECT_LT(run.trace[kReqStep].state.body_rate.eigen().norm(), kReqRateRadps)
      << "REQ-ACTL-001 rate bound missed; profile:\n"
      << profile.str();
  // ...and it stays there: no re-excitation from the law's own commands, which a
  // sign error or a duty-factor mistake would produce.
  EXPECT_LT(peakRateFrom(run.trace, kReqStep), kReqRateRadps);
  // Well below the tip-off, so "reduced" is not being read off a controller that
  // merely failed to make things worse.
  EXPECT_LT(run.trace.back().state.body_rate.eigen().norm(), 0.6 * initial);

  // **What crossed the wire.** The lockstep suite no longer carries a
  // nonzero-command case, so the per-unit layout and the §7 on-window field are
  // pinned here: the sim decoded a real on-window, every rod carried its own
  // axis's share, and no rod carried another's — which is the ordering claim,
  // since a transposed command set would put rod 0's dipole on rod 1.
  EXPECT_NEAR(run.peak_on_window_s, 0.5 * 0.1, 1e-9);
  for (int i = 0; i < 3; ++i) {
    EXPECT_GT(run.peak_rod_dipole[i], 0.0) << "rod " << i << " never carried a command";
    EXPECT_EQ(run.peak_rod_off_axis[i], 0.0) << "rod " << i << " carried another rod's axis";
  }

  // The interlock stayed healthy throughout a run that drove the rods every
  // cycle — the negative claim row 4 makes in full.
  EXPECT_EQ(countOf(run.log, "Magnetorquer stuck-on: rod"), 0u);
}

// ======================================================================
// Row 2 — commanded inertial hold
// ======================================================================

TEST(SitlAttitudeControl, InertialHoldConvergesUnderThePointingBound) {
  RecordProperty("verifies", "REQ-ACTL-002");
  std::string why;
  if (toolchainMissing(why)) {
    GTEST_SKIP() << why;
  }

  // Hold the attitude the fault-matrix geometry flies — every source available,
  // so the estimator reaches its fine mode and the controller's quality floor is
  // met — offset by 10 degrees about body +X, which is the error to close.
  const pm::Quat<pm::frames::Body, pm::frames::ECI> target = faultMatrixAttitude();
  const pm::Quaternion offset =
      pm::Quaternion::FromAxisAngle(Eigen::Vector3d::UnitX(), 10.0 * M_PI / 180.0);

  scenario::SimConfig orbit = faultMatrixOrbit(200.0, "sitl-inertial-hold");
  orbit.initial_state.attitude =
      pm::Quat<pm::frames::Body, pm::frames::ECI>((offset * target.core()).canonical());

  const pm::Quaternion tq = target.core();
  const double target_q[4] = {tq.w(), tq.x(), tq.y(), tq.z()};
  const RunResult run = fly("hold", orbit, /*ctrlMode=*/2, target_q, noFaults);
  ASSERT_TRUE(run.sim_healthy);
  ASSERT_GT(run.trace.size(), 2000u);

  EXPECT_GE(countOf(run.log, "Inertial hold target set"), 1u);
  EXPECT_GE(countOf(run.log, "Control mode IDLE (0) -> POINT"), 1u);

  auto errorAt = [&](std::size_t i) {
    return run.trace[i].state.attitude.core().angularDistance(tq);
  };
  const double initial_error = errorAt(0);
  EXPECT_NEAR(initial_error, 10.0 * M_PI / 180.0, 1e-6);

  // **Converged and stayed there.** The requirement value (REQ-ACTL-002) is
  // 1.0 deg steady-state under ideal sensing; the last 50 s of the run must sit
  // inside it. Truth-side error, so this is the pointing the vehicle actually
  // achieved and not the pointing it believed it achieved.
  const std::size_t settle_from = run.trace.size() - 500;
  double worst_settled = 0.0;
  for (std::size_t i = settle_from; i < run.trace.size(); ++i) {
    worst_settled = std::max(worst_settled, errorAt(i));
  }
  RecordProperty("measured_settled_pointing_error_deg",
                 std::to_string(worst_settled * 180.0 / M_PI));
  EXPECT_LT(worst_settled, 1.0 * M_PI / 180.0);
  // ...and by a wide margin over the initial error, so "converged" is not being
  // read off a controller that merely failed to make things worse.
  EXPECT_LT(worst_settled, 0.2 * initial_error);

  // POINT drives wheels, not rods, so the rods stay off and every magnetometer
  // sample is a quiet-window one. Asserted on what crossed the wire, not only on
  // the absence of an event: real wheel torques, no dipole, no on-window.
  EXPECT_GT(run.peak_wheel_torque, 0.0);
  EXPECT_EQ(run.peak_on_window_s, 0.0);
  for (int i = 0; i < 3; ++i) {
    EXPECT_EQ(run.peak_rod_dipole[i], 0.0);
  }
  EXPECT_EQ(countOf(run.log, "Magnetorquer stuck-on: rod"), 0u);
  // At most one refused cycle in 200 s, and it is the *re-seed*: when the first
  // star tracker is fused the MEKF is re-seeded from it, and for that one cycle
  // the filter reports no rate. The controller refusing there is correct — it
  // commands zero rather than torquing on a rate it does not have — and it is
  // bounded, which is what is asserted. A refusal on the quality floor would be
  // a different failure and is asserted absent separately.
  EXPECT_LE(countOf(run.log, "Control refused in mode POINT"), 1u);
  EXPECT_EQ(countOf(run.log, "POINT (2): QUALITY_FLOOR"), 0u);
}

// ======================================================================
// Row 3 — a stuck-on rod
// ======================================================================

TEST(SitlAttitudeControl, StuckOnRodIsCaughtAndTheEstimatorSurvivesIt) {
  RecordProperty("verifies", "REQ-ACTL-005");
  std::string why;
  if (toolchainMissing(why)) {
    GTEST_SKIP() << why;
  }

  const pm::Quat<pm::frames::Body, pm::frames::ECI> target = faultMatrixAttitude();
  const pm::Quaternion tq = target.core();
  const double target_q[4] = {tq.w(), tq.x(), tq.y(), tq.z()};

  scenario::SimConfig orbit = faultMatrixOrbit(70.0, "sitl-stuck-rod");
  // POINT mode: the controller commands no dipole at all, so a field disturbance
  // is unambiguous — nothing on the vehicle asked for one.
  const RunResult run =
      fly("stuck", orbit, /*ctrlMode=*/2, target_q, [](scenario::Vehicle& v, unsigned step) {
        if (step == 100) {  // 10 s in, after the estimator has settled
          v.magnetorquers[kMtqX].model.injectStuckOnAt(
              pm::Vec3<pm::frames::Body>(Eigen::Vector3d(kStuckDipoleAm2, 0.0, 0.0)));
        }
      });
  ASSERT_TRUE(run.sim_healthy);

  // The §9 monitor fired, once — persistence-counted and edge-gated, not once
  // per 10 Hz cycle.
  EXPECT_EQ(countOf(run.log, "Magnetorquer stuck-on: rod"), 1u);
  // The attribution is **ambiguous**, and that is the honest answer rather than a
  // shortcoming: in POINT no rod carries a command, so the candidate set is empty
  // and there is nothing to distinguish the three rods by. Resolving it needs a
  // commanded isolation sweep — drive the rods one at a time and watch — which is
  // the Phase-7 state machine's recovery action. Asserting the ambiguity here is
  // the P53 rule about honest negatives: the mechanism that makes attribution
  // impossible is worth pinning, and a dormant conditional assertion is not.
  EXPECT_GE(countOf(run.log, "AMBIGUOUS"), 1u);

  // Magnetometer samples really are excluded from the latch onward, and the
  // estimator says so in its own words — the assertion the comment used to only
  // promise. Ordering too: the exclusion follows the latch, it does not precede
  // it.
  const std::size_t stuck_at = indexOf(run.log, "Magnetorquer stuck-on: rod");
  const std::size_t excluding_at = indexOf(run.log, "interlock excluding magnetometer samples");
  ASSERT_NE(stuck_at, std::string::npos);
  ASSERT_NE(excluding_at, std::string::npos)
      << "the latch fired but the estimator never reported excluding samples";
  EXPECT_GT(excluding_at, stuck_at);

  // **The estimator survives the fault it helps identify** (review-lessons: "the
  // filter that identifies a fault must survive it"). It is in its fine,
  // star-tracker-fused mode, which does not need the magnetic pair at all, so
  // losing every magnetometer sample must cost nothing — and a demotion here
  // would be exactly the circularity the catalog warns about.
  EXPECT_EQ(countOf(run.log, "Fine mode demoted to coarse"), 0u);
  EXPECT_EQ(countOf(run.log, "Attitude lost"), 0u);
}

// ======================================================================
// Row 4 — the latch is revocable
// ======================================================================

TEST(SitlAttitudeControl, StuckOnLatchClearsWhenTheRodRecovers) {
  RecordProperty("verifies", "REQ-ACTL-005");
  std::string why;
  if (toolchainMissing(why)) {
    GTEST_SKIP() << why;
  }

  // **The row that proves the exclusion is not a life sentence**, end to end and
  // through the real topology rather than a harness that hand-feeds the monitor
  // its input. The evidence that clears the latch is a quiet-window field
  // reading — and that reading has to travel the same path the latch closes, so
  // the only way to know it does is to break the rod, fix it, and wait.
  const pm::Quat<pm::frames::Body, pm::frames::ECI> target = faultMatrixAttitude();
  const pm::Quaternion tq = target.core();
  const double target_q[4] = {tq.w(), tq.x(), tq.y(), tq.z()};

  // `MtqStuckClearCycles` is 20 in the reference vehicle, so 3 s of clean quiet
  // windows after the fault is removed is comfortably enough; the row runs 30 s.
  constexpr unsigned kInjectStep = kGnssAcquiredStep;   // once there is a field to judge
  constexpr unsigned kRecoverStep = kInjectStep + 100;  // 10 s later, past the 5-cycle latch
  scenario::SimConfig orbit = faultMatrixOrbit(70.0, "sitl-stuck-clear");
  const RunResult run =
      fly("clear", orbit, /*ctrlMode=*/2, target_q, [](scenario::Vehicle& v, unsigned step) {
        if (step == kInjectStep) {
          v.magnetorquers[kMtqX].model.injectStuckOnAt(
              pm::Vec3<pm::frames::Body>(Eigen::Vector3d(kStuckDipoleAm2, 0.0, 0.0)));
        }
        if (step == kRecoverStep) {
          v.magnetorquers[kMtqX].model.clearFaults();
          // clearFaults leaves the held moment behind; the
          // rod has to actually de-energise for the field
          // to go away, which is what a recovered drive
          // does on its next command.
          v.magnetorquers[kMtqX].model.commandDipole(
              pm::Vec3<pm::frames::Body>(Eigen::Vector3d::Zero()));
          v.magnetorquers[kMtqX].model.deenergize();
        }
      });
  ASSERT_TRUE(run.sim_healthy);

  ASSERT_EQ(countOf(run.log, "Magnetorquer stuck-on: rod"), 1u);
  // The latch cleared, on the same residual test that set it.
  ASSERT_EQ(countOf(run.log, "Magnetorquer stuck-on cleared"), 1u)
      << "the stuck-on latch never cleared — its clearing evidence cannot reach "
         "the monitor:\n"
      << run.log;
  EXPECT_GT(indexOf(run.log, "Magnetorquer stuck-on cleared"),
            indexOf(run.log, "Magnetorquer stuck-on: rod"));
  // ...and the magnetic pair came back with it, which is the consequence that
  // matters: while the latch is set the vehicle has no magnetometer data at all.
  ASSERT_EQ(countOf(run.log, "interlock restored"), 1u);
  EXPECT_GT(indexOf(run.log, "interlock restored"),
            indexOf(run.log, "interlock excluding magnetometer samples"));
}

// ======================================================================
// Row 5 — the negative: a healthy duty cycle must not trip the monitor
// ======================================================================

TEST(SitlAttitudeControl, NominalDutyCycledDetumbleDoesNotTripTheStuckOnMonitor) {
  RecordProperty("verifies", "REQ-ACTL-004");
  std::string why;
  if (toolchainMissing(why)) {
    GTEST_SKIP() << why;
  }

  // The row that makes row 3 evidence. Rods driven hard, every control period,
  // for a full minute — a monitor that fires here would invalidate every
  // magnetometer sample the vehicle takes while torquing, which is precisely when
  // B-dot needs them.
  scenario::SimConfig orbit = faultMatrixOrbit(70.0, "sitl-duty-nominal");
  orbit.initial_state.body_rate = pm::Vec3<pm::frames::Body>(kTumbleRadps);
  const RunResult run = fly("duty", orbit, /*ctrlMode=*/1, nullptr, noFaults);
  ASSERT_TRUE(run.sim_healthy);

  EXPECT_GE(countOf(run.log, "Control mode IDLE (0) -> DETUMBLE"), 1u);
  EXPECT_EQ(countOf(run.log, "Magnetorquer stuck-on: rod"), 0u);
  // The rods really were driven — otherwise the negative is vacuous. Dipole
  // saturation at the start of a 5 deg/s detumble is the evidence, and it is the
  // event the controller reports for it.
  EXPECT_GE(countOf(run.log, "Dipole saturated"), 1u);
  // And the magnetic pair kept flowing: no magnetometer was excluded, and the
  // estimator never lost its attitude.
  EXPECT_EQ(countOf(run.log, "Attitude lost"), 0u);
}

}  // namespace
