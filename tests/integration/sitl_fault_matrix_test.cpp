/// @file The SITL fault-injection matrix (design doc §9, §23.1.1).
///
/// One case per row of the matrix below, each flying the **same** vehicle, the
/// same orbit and the same attitude, and differing only in the fault injected
/// into the truth sim. What is asserted is not accuracy — the accuracy campaigns
/// do that — but **which path the estimator's mode machine takes**: the source
/// the fine solution is being built from, which unit was blamed, whether the
/// blame was earned, and what it cost.
///
///   fault                                     | asserted path
///   ------------------------------------------|--------------------------------
///   trackers arrive after a dark start        | NONE -> STAR_TRACKER, no re-seed
///   king star tracker lost                    | fine stays STAR_TRACKER on st_b
///   both star trackers lost, then returned    | down to SUN_MAG and back up, no loss
///   trackers arrive after a SUN_MAG promotion | filter re-seeded from the tracker
///   trackers 17 deg wrong, SS+MAG mode        | refused and reported, nothing latched
///   one magnetometer biased, tracker mode     | mag_b OUTVOTED, re-admitted, no demotion
///   one magnetometer biased, SS+MAG mode      | ambiguous, nothing latched
///   selected sun sensor biased, tracker mode  | cross-check overrides onto ss_yp
///   sun lost from every field of view         | fine coasts on the magnetic pair
///   gyro pair ambiguous, no tracker           | no rate, TRIAD holds, never attributed
///   magnetometer fault before any reference   | no blame until a tracker is fused
///
/// **The geometry is what makes this one file rather than nine.** The attitude
/// in `sitl_harness.hpp` is solved so that both star trackers, two sun sensors
/// and the magnetometers are all available at once; a case that needs a source
/// absent takes it away with a dropout. "No star tracker" and "both star
/// trackers failed" are the same state to the estimator, and building the second
/// from the first is what makes the rows comparable.
///
/// Rows 4 and 5 exist because this matrix **found a defect and drove its fix**: a
/// tracker arriving after the fine mode had promoted on the vector pairs used to
/// be latched out rather than fused, which made the top rung unreachable from a
/// sunlit boot and unrecoverable after any outage. Row 4 is the fix, row 5 is the
/// failure mode the fix creates and closes.
/// `TrackerArrivingAfterSunMagPromotionIsAdoptedNotBlamed` carries the whole story
/// with the observed event stream. Two rows still start **dark** (sun sensors
/// dropped out): the dark-start row, which is about that path, and the cascade,
/// whose premise is a fault present before the vehicle has *any* attitude
/// solution. Everything else now boots sunlit, as a real vehicle mostly does.
///
/// Skips (never fails) when the flight binary or the Python toolchain is absent.
///
/// Verifies REQ-FDIR-005 through REQ-FDIR-013, and is one of the artifacts
/// REQ-FDIR-004 (fault-injection verification) requires. The GNSS/orbit-filter
/// rows (REQ-ODP-007) live in `sitl_od_fault_test.cpp`.

#include <gtest/gtest.h>

#include <cmath>
#include <string>

#include "io/closed_loop.hpp"
#include "scenario/sim_runner.hpp"
#include "sitl_harness.hpp"

namespace {

using namespace polaris::test::sitl;  // NOLINT(build/namespaces) — the shared SITL fixture

/// Pairs the on-orbit alignment window collects, the reference vehicle's
/// `StAlignMinSamples`: 100 simultaneous solutions, i.e. 10 s at the 10 Hz rate.
/// **Every tracker row needs it.** A non-king tracker is not fused at all until it
/// has been calibrated against the king (§8.2, the honest launch state), so
/// without this window st_b is a spectator and "the king fails and st_b carries
/// it" would be testing an empty suite.
constexpr unsigned kAlignPairs = 100;

/// Sun sensors restored here — the "eclipse exit" of the dark-start protocol.
/// 16 s: past the AURIGA's 3.8 s cold acquisition and past the 10 s alignment
/// window that follows it, so the vehicle is on its top rung with both trackers
/// fused before the Sun returns.
constexpr unsigned kSunReturnSteps = 160;

/// Faults land here, 4 s after the Sun is back, so the pre-fault state is
/// established and unambiguous.
constexpr unsigned kFaultSteps = 200;

/// Sun-sensor calibration shift injected [rad]: 0.30 = 17.2°, which is above the
/// `MonitorSunCrossUnitRad` threshold of 0.15 rad on the *disagreement* and above
/// it again on the faulted unit's own residual against the tracker-fused
/// solution, while the healthy runner-up stays well inside it. Both halves of the
/// "decisive margin" rule the override needs (§8.2) are therefore satisfied, and
/// deliberately so — a fault sized between the two would resolve as ambiguous,
/// which is a different row.
constexpr double kSunFaultRad = 0.30;

/// Magnetometer bias jump injected [T]: 10 µT. The pairwise disagreement gate is
/// `MagDisagreementT` = 5 µT and the plausibility band is 0.5–1.6 × the modelled
/// ~30 µT magnitude, so this fault is **twice** the disagreement gate and still
/// comfortably inside the band — invisible to every per-unit gate, visible only
/// to the pair. A railed unit is the easier fault (the magnitude gate catches it
/// before the pair is consulted) and is covered in `mag_voting_test.cpp`.
constexpr double kMagFaultTesla = 10.0e-6;

/// `flight.attitudeEstimator.MonitorAlertCycles` in the reference vehicle config,
/// mirrored so the expected refusal cadence is derived rather than pasted. A
/// change to the flight value fails the bad-tracker row loudly, which is the
/// intent — the cadence is the operator-facing contract.
constexpr std::size_t kFlightMonitorAlertCycles = 50;

/// Star-tracker attitude fault injected [rad]: 0.30 = 17.2°, far outside the ~5°
/// (3σ of the coarse solution's systematic floor) the sun/magnetic data can
/// support on this suite — so the arbitration guard must refuse to adopt it. A
/// fault *inside* that gate is indistinguishable from truth given only the vector
/// pairs, and adopting it is then the right call anyway: the tracker is the better
/// instrument and the cost is bounded by what the vector solution cost already.
constexpr double kStFaultRad = 0.30;

/// Gyro bias jump injected [rad/s], as in the Push 51 cases: 0.05 = 2.9 °/s, six
/// times the pairwise gate and a tenth of the plausibility limit, so no per-unit
/// gate can see it and only the pair can.
constexpr double kImuFaultRadps = 0.05;

/// Everything a case needs from one SITL run: the deployment's event stream and
/// the fact that the run itself was healthy.
struct RunResult {
  std::string log;
  bool sim_healthy = false;
};

/// Compile the vehicle config, fork the deployment, fly @p orbit against it with
/// the full fault-matrix suite, and return the captured event stream. @p perStep
/// runs on the macro-step seam once per exchanged step with the step index, which
/// is where a case injects and clears its faults; the vehicle it mutates is the
/// one being sampled, so a fault takes effect from the next sample onwards.
///
/// @p alignPairs > 0 commands ST_ALIGN_CAL_START on st_b at startup. On a flight
/// vehicle that command comes from the ground; the deployment here has no uplink,
/// and what the tracker rows need is the *result* of the calibration, not the
/// radio.
///
/// Fatal setup problems are reported as gtest failures, so a case's own
/// assertions never run against a run that did not happen.
template <typename PerStep>
RunResult flyWithFaults(const std::string& tag, const scenario::SimConfig& orbit,
                        unsigned alignPairs, PerStep perStep) {
  RunResult result;
  // Per-process working directory: concurrent ctest jobs must not share the
  // parameter file or the captured log.
  const std::string work_dir =
      "build-artifacts/test-fault-" + tag + "-" + std::to_string(::getpid());
  const std::string err_path = work_dir + "/configc.err";
  const std::string prm_path = work_dir + "/PrmDb.dat";
  const std::string log_path = work_dir + "/fsw.log";
  if (compileConfig(work_dir, err_path) != 0) {
    ADD_FAILURE() << "config compiler failed:\n" << readFile(err_path);
    return result;
  }

  io::SitlServer server(faultMatrixCounts(), 100'000'000LL);
  if (!server.start(0)) {
    ADD_FAILURE() << server.lastError();
    return result;
  }
  const pid_t pid = spawnFsw(fswBinaryPath(), server.port(), prm_path, log_path,
                             /*magCalSamples=*/0, alignPairs, /*stAlignUnit=*/1);
  if (pid < 0) {
    ADD_FAILURE() << "fork failed";
    return result;
  }

  scenario::Vehicle vehicle;
  std::string error;
  if (!scenario::buildVehicle(faultMatrixSuite(), 1, vehicle, &error)) {
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

  // **Step 0 runs before the first sample is taken.** The closed loop samples the
  // sensors and *then* calls back, so a fault injected on the macro-step seam
  // takes effect from the next sample onwards — which means a fault "at step 0"
  // injected inside the loop would still let one clean cycle through, and one
  // clean cycle is all the estimator needs to promote off a Davenport seed. A row
  // whose premise is "this source was never available" has to establish that
  // before the run starts.
  unsigned step = 1;
  perStep(vehicle, 0u);
  const io::FswCallback inner = server.callback();
  const io::FswCallback faulted = [&](const io::FswInputs& in) {
    perStep(vehicle, step);
    ++step;
    return inner(in);
  };

  std::vector<io::MacroSample> trace;
  if (!loop.run(faulted, &trace, &error)) {
    ADD_FAILURE() << error;
  }
  result.sim_healthy = server.healthy();
  server.stop();  // sends SHUTDOWN
  reapFsw(pid);
  result.log = readFile(log_path);
  // The whole log is in memory now and every failure message prints it, so the
  // directory is dead weight — and ~180 of them had accumulated under
  // build-artifacts before anyone noticed. `POLARIS_KEEP_SITL_LOGS=1` keeps them,
  // which is how you read the event stream of a run that *passed*.
  if (std::getenv("POLARIS_KEEP_SITL_LOGS") == nullptr) {
    // Best-effort cleanup; glibc marks system() warn_unused_result and a (void)
    // cast does not silence it, so the result is named and ignored.
    const int rm_rc = std::system(("rm -rf '" + work_dir + "'").c_str());
    static_cast<void>(rm_rc);
  }
  return result;
}

/// Drop every sun sensor (the dark start) and bring them back at
/// @ref kSunReturnSteps. Shared by every tracker-mode row so the protocol is one
/// statement rather than five copies that can drift apart.
void darkStart(scenario::Vehicle& v, unsigned step) {
  if (step == 0) {
    for (auto& ss : v.sun_sensors) {
      ss.model.setDropout(true);
    }
  }
  if (step == kSunReturnSteps) {
    for (auto& ss : v.sun_sensors) {
      ss.model.clearFaults();
    }
  }
}

void dropTrackers(scenario::Vehicle& v) {
  v.star_trackers[kStA].model.setDropout(true);
  v.star_trackers[kStB].model.setDropout(true);
}

/// True when the flight binary and the Python toolchain are both present. Every
/// case opens with this and skips rather than failing when they are not — CI's
/// unit-test job builds only the native-ut tree.
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

// ----------------------------------------------------------------------
// The fixture's own geometry, checked before anything is asserted on it
// ----------------------------------------------------------------------

/// The matrix attitude is solved from the ephemeris, so the angles the whole file
/// depends on are a *result* rather than a constant. This is the case that says
/// so: if an epoch or an orbit change ever voids the geometry, this fails with
/// the numbers instead of every other case failing with a mode transition that
/// never happened.
TEST(SitlFaultMatrix, GeometryMakesEverySourceAvailableAtOnce) {
  const MatrixGeometry g = matrixGeometry();
  const double deg = 180.0 / M_PI;

  // Both sun sensors inside the FSS's 60° acceptance cone, and by a margin: the
  // selected unit and its runner-up must both stay in view across the whole arc.
  EXPECT_LT(g.sun_incidence_xp_rad * deg, 55.0)
      << "the +X sun sensor is not in view: " << g.sun_incidence_xp_rad * deg << " deg";
  EXPECT_LT(g.sun_incidence_yp_rad * deg, 55.0)
      << "the +Y sun sensor is not in view: " << g.sun_incidence_yp_rad * deg << " deg";
  // ...and at the *same* incidence, which is what makes the selection the flight
  // rule's tie-break (equal reported σ, lowest index wins) rather than an accident.
  EXPECT_NEAR(g.sun_incidence_xp_rad, g.sun_incidence_yp_rad, 1.0e-9);

  // Both tracker boresights clear of the Earth: 68° of limb plus a 22° exclusion
  // is 90°, and the mounting is chosen to leave margin on top of that.
  EXPECT_GT(g.tracker_a_from_nadir_rad * deg, 100.0)
      << "star tracker A is inside the Earth exclusion: " << g.tracker_a_from_nadir_rad * deg
      << " deg from nadir";
  EXPECT_GT(g.tracker_b_from_nadir_rad * deg, 100.0)
      << "star tracker B is inside the Earth exclusion: " << g.tracker_b_from_nadir_rad * deg
      << " deg from nadir";
}

// ----------------------------------------------------------------------
// Star-tracker faults
// ----------------------------------------------------------------------

/// **A dark start climbs straight to the top rung.** The Sun is out of every
/// field of view, so the coarse chain cannot solve and there is no Davenport seed
/// to be had; the fine mode is seeded by a tracker on its own, which is the path
/// that makes the top rung reachable in eclipse and at cold start. The Sun then
/// returns and must change nothing: with a tracker accepted the vector pairs are
/// monitors, not measurements, so no rung change and no re-seed may follow.
///
/// This is the baseline every tracker row below is differenced against — if it
/// fails, the faulted rows are asserting on a vehicle that was never where they
/// assume it was.
///
/// Verifies REQ-FDIR-005 and REQ-FDIR-007.
TEST(SitlFaultMatrix, DarkStartSeedsFromATrackerAndSurvivesSunrise) {
  RecordProperty("verifies", "REQ-FDIR-005;REQ-FDIR-007");
  POLARIS_REQUIRE_SITL_TOOLCHAIN();

  const RunResult run = flyWithFaults("st-dark", faultMatrixOrbit(25.0, "sitl-fault-st-dark"),
                                      kAlignPairs, darkStart);
  ASSERT_TRUE(run.sim_healthy);
  ASSERT_FALSE(run.log.empty());

  // 1. Seeded from a tracker, with no coarse floor underneath it.
  ASSERT_NE(run.log.find("Fine mode engaged"), std::string::npos) << run.log;
  EXPECT_NE(run.log.find("NONE (0) -> STAR_TRACKER"), std::string::npos)
      << "the ladder did not go straight to the trackers from a dark start:\n"
      << run.log;

  // 2. The second unit earned its way into the fusion: a non-king tracker is not
  //    fused at all until it is calibrated against the king (§8.2).
  EXPECT_NE(run.log.find("Inter-tracker alignment fitted on unit 1"), std::string::npos)
      << "the alignment window never closed, so st_b is a spectator in every row below:\n"
      << run.log;

  // 3. Sunrise changes nothing. One source change over the whole run, no re-seed,
  //    no demotion — the vector pairs became monitors, which is the ladder's own
  //    claim about what happens when a better source is present.
  EXPECT_EQ(countOf(run.log, "Fine-mode source changed"), 1u)
      << "the returning Sun moved the ladder off the trackers:\n"
      << run.log;
  EXPECT_EQ(countOf(run.log, "Fine mode engaged"), 1u)
      << "the filter was re-seeded after sunrise, so its state was not kept:\n"
      << run.log;
  EXPECT_EQ(run.log.find("Fine mode demoted"), std::string::npos) << run.log;
  EXPECT_EQ(run.log.find("Attitude lost"), std::string::npos) << run.log;

  // 4. And the demoted sources stayed quiet. A monitor alerting on a healthy sun
  //    or magnetic pair means the residual is being computed in the wrong frame,
  //    which is the mistake this path is easiest to make.
  EXPECT_EQ(run.log.find("Residual monitor"), std::string::npos)
      << "a residual monitor alerted on healthy sources — suspect the frame:\n"
      << run.log;
}

/// **The king fails and the frame survives it.** st_a is the king — its mounting
/// *defines* the body frame — so losing it is the one tracker fault that could
/// plausibly cost the top rung rather than an increment of accuracy. It must not:
/// st_b has been calibrated into the king's frame and carries the solution alone.
/// A suite whose finest rung depended on one unit would have no redundancy where
/// it matters most, which is the P52 lesson in the direction the design says it
/// was fixed.
///
/// Verifies REQ-FDIR-006.
TEST(SitlFaultMatrix, KingTrackerLostKeepsTheFineSolutionOnTheSecondUnit) {
  RecordProperty("verifies", "REQ-FDIR-006");
  POLARIS_REQUIRE_SITL_TOOLCHAIN();

  const RunResult run = flyWithFaults("st-king", faultMatrixOrbit(30.0, "sitl-fault-st-king"),
                                      kAlignPairs, [](scenario::Vehicle& v, unsigned step) {
                                        if (step == kFaultSteps) {
                                          v.star_trackers[kStA].model.setDropout(true);
                                        }
                                      });
  ASSERT_TRUE(run.sim_healthy);
  ASSERT_FALSE(run.log.empty());

  // 1. The vehicle was on the top rung, with both units fused, before the fault.
  ASSERT_NE(run.log.find("-> STAR_TRACKER"), std::string::npos) << run.log;
  ASSERT_NE(run.log.find("Inter-tracker alignment fitted on unit 1"), std::string::npos)
      << "st_b was never calibrated, so it could not have carried anything:\n"
      << run.log;

  // 2. And it stayed there. Two source changes, both before the fault — the
  //    sunlit promotion onto the vector pairs and the adoption of the trackers
  //    that follows — and **no third**: losing the king must not move the ladder.
  EXPECT_EQ(countOf(run.log, "Fine-mode source changed"), 2u)
      << "losing the king moved the ladder; the second tracker should have carried it:\n"
      << run.log;
  EXPECT_EQ(run.log.find("STAR_TRACKER (2) -> SUN_MAG"), std::string::npos)
      << "the ladder fell to the vector pairs with a healthy second tracker available:\n"
      << run.log;

  // 3. It cost nothing. A dropout is an absence, not a lie, so nothing may be
  //    latched out and nothing may be demoted for it.
  EXPECT_EQ(run.log.find("Star tracker 0 excluded"), std::string::npos)
      << "a dropped-out tracker was blamed for a fault it did not commit:\n"
      << run.log;
  EXPECT_EQ(run.log.find("Fine mode demoted"), std::string::npos) << run.log;
  EXPECT_EQ(run.log.find("Attitude lost"), std::string::npos) << run.log;
}

/// **Both trackers fail, then come back: down one rung and back up.** The
/// design's claim is that a rung change is not a demotion — the filter keeps its
/// state and its covariance, and the published solution stays valid throughout —
/// so what is asserted is `FineSourceChanged` in both directions with no
/// `FineModeDemoted` and no `AttitudeLost` anywhere. The vector pairs are
/// underneath the trackers precisely so that losing the trackers costs accuracy
/// and nothing else.
///
/// **The climb back is the half that did not work before Push 53** — it is the
/// same rejected transition as a sunlit boot, since the filter has been pulled
/// toward the vector solution during the outage and the returning trackers then
/// disagree with an overconfident covariance. It is asserted here as well as in
/// `TrackerArrivingAfterSunMagPromotionIsAdoptedNotBlamed` because they are
/// different entries: that one is a cold arrival, this one a *recovery*, and a
/// vehicle that could reach its top rung only once per boot would still be
/// broken.
///
/// Verifies REQ-FDIR-005, REQ-FDIR-006 and REQ-FDIR-013.
TEST(SitlFaultMatrix, BothTrackersLostFallsToSunMagAndClimbsBackOnReturn) {
  RecordProperty("verifies", "REQ-FDIR-005;REQ-FDIR-006;REQ-FDIR-013");
  POLARIS_REQUIRE_SITL_TOOLCHAIN();

  constexpr unsigned kReturnSteps = 300;  // 10 s of running on the vector pairs
  const RunResult run = flyWithFaults("st-both", faultMatrixOrbit(45.0, "sitl-fault-st-both"),
                                      kAlignPairs, [](scenario::Vehicle& v, unsigned step) {
                                        if (step == kFaultSteps) {
                                          dropTrackers(v);
                                        }
                                        if (step == kReturnSteps) {
                                          v.star_trackers[kStA].model.clearFaults();
                                          v.star_trackers[kStB].model.clearFaults();
                                        }
                                      });
  ASSERT_TRUE(run.sim_healthy);
  ASSERT_FALSE(run.log.empty());

  // 1. Down and back up, in that order. The counts are the assertion that each is
  //    an edge rather than a per-cycle report.
  const std::size_t down = indexOf(run.log, "STAR_TRACKER (2) -> SUN_MAG");
  const std::size_t up_again = run.log.rfind("SUN_MAG (1) -> STAR_TRACKER");
  ASSERT_NE(down, std::string::npos) << "the tracker outage never moved the ladder:\n" << run.log;
  ASSERT_NE(up_again, std::string::npos)
      << "the ladder never climbed back after the trackers returned — the recovery is "
         "the same rejected transition a sunlit boot meets:\n"
      << run.log;
  EXPECT_LT(down, up_again) << "the transitions are in the wrong order:\n" << run.log;

  // 2. A rung change is not a demotion, and it is not a loss — in either
  //    direction.
  EXPECT_EQ(run.log.find("Fine mode demoted"), std::string::npos)
      << "a tracker outage or its recovery demoted the fine mode:\n"
      << run.log;
  EXPECT_EQ(run.log.find("Attitude lost"), std::string::npos)
      << "a tracker outage cost the attitude solution:\n"
      << run.log;
  EXPECT_EQ(countOf(run.log, "Fine mode engaged"), 1u)
      << "the filter was dropped and re-promoted — a demotion, which is what this row "
         "exists to rule out. Note a re-seed is NOT a demotion and does not fire this "
         "event: the arbitration replaces the filter's state deliberately, and "
         "FineReseededFromStarTracker is what says so (REQ-FDIR-013):\n"
      << run.log;

  // 3. Nothing was blamed at any point. Both units were absent, not wrong, and
  //    the returning pair disagreeing with a stale covariance is not their fault
  //    either.
  EXPECT_EQ(run.log.find("excluded from the fusion"), std::string::npos) << run.log;
}

/// **The rung the ladder could not climb, and now can — the defect this matrix
/// found, and the fix it drove.**
///
/// The vehicle boots sunlit, so the fine mode promotes off a Davenport seed
/// within one cycle, well before the AURIGA's 3.8 s cold acquisition. When the
/// trackers then arrive they have to be *adopted* by a filter already running on
/// the vector pairs — and before Push 53 they never were: every tracker update
/// failed the MEKF's χ²₃ attitude gate, and after `MekfNisStreak` = 20 consecutive
/// rejections the **tracker** was latched out while the mode was demoted for
/// `NIS_STREAK`. The arcsecond-class instrument was blamed for disagreeing with a
/// degrees-class solution.
///
/// The mechanism is structural rather than a tuning accident. A SS+MAG solution's
/// error is dominated by the magnetometer's ~34 mrad **systematic**, which the
/// filter's white-`R` model averages down, so within seconds the reported
/// covariance is a few mrad while the true error is tens. `S = HPHᵀ + R` is then a
/// few mrad² (a tracker's own R is ~0.05 mrad) against a residual that is the
/// whole systematic — NIS two orders of magnitude over the gate, permanently.
/// Re-admission could not rescue it either: `readmitStarTrackers` judged the
/// excluded unit against that same solution at 3σ_z ≈ 0.7 mrad.
///
/// What it cost, both observed on this fixture: with the **king only** fused (the
/// launch state, before `ST_ALIGN_CAL` — a non-king unit is not fused until it is
/// calibrated against the king, §8.2) the vehicle stayed on SUN_MAG for the rest
/// of the flight, its best instrument permanently offline and silent after one
/// EVR. With **both** calibrated it healed, but only by accident of ordering, and
/// still spent one spurious `StUnitExcluded` on the frame-defining unit and one
/// `FineModeDemoted` getting there.
///
///     t=0.1  FineModeEngaged    seed cov trace=0.004138 rad^2  (Davenport)
///     t=0.2  FineSourceChanged  NONE -> SUN_MAG
///     t=5.6  StUnitExcluded     Star tracker 0, 20 consecutive NIS rejections
///     t=5.6  FineModeDemoted    reason=NIS_STREAK
///     t=5.8  FineModeEngaged    seed cov trace=0.002159 rad^2  (Davenport)
///     t=5.9  FineSourceChanged  NONE -> SUN_MAG    (and there it stays)
///
/// **The fix** (`AttitudeEstimator::arbitrateRejectedTrackers`): a gate rejection
/// while the solution is not tracker-sourced is not evidence about the tracker, so
/// the verdict is taken against the **coarse** solution instead — the one attitude
/// covariance on the vehicle that converges to its systematic floor rather than to
/// zero, and therefore does not lie. Inside 3σ of it the filter is the outlier and
/// is re-seeded from the tracker (`FineReseededFromStarTracker`); outside it the
/// comparison is like for like and the unit is excluded. That second branch is the
/// subject of `BadTrackerInSunMagModeIsRefusedAndReportedNotAdopted` below, which
/// is the failure mode this fix creates and closes.
///
/// Verifies REQ-FDIR-013.
TEST(SitlFaultMatrix, TrackerArrivingAfterSunMagPromotionIsAdoptedNotBlamed) {
  RecordProperty("verifies", "REQ-FDIR-013");
  POLARIS_REQUIRE_SITL_TOOLCHAIN();

  const RunResult run =
      flyWithFaults("st-late", faultMatrixOrbit(30.0, "sitl-fault-st-late"), kAlignPairs,
                    [](scenario::Vehicle&, unsigned) {});  // no fault: sunlit nominal
  ASSERT_TRUE(run.sim_healthy);
  ASSERT_FALSE(run.log.empty());

  // 1. The premise: the vehicle really did promote on the vector pairs first,
  //    which is what makes the transition below the hard one.
  ASSERT_NE(run.log.find("NONE (0) -> SUN_MAG"), std::string::npos)
      << "the run did not start on the vector pairs, so it tested nothing:\n"
      << run.log;

  // 2. The arbitration fired, exactly once, and said so. The event is the whole
  //    operator-facing point: the same telemetry without it reads as "the tracker
  //    was bad", which is precisely backwards.
  EXPECT_EQ(countOf(run.log, "Fine solution re-seeded from star tracker"), 1u)
      << "the filter never adopted the arriving tracker, or adopted it repeatedly:\n"
      << run.log;

  // 3. And the ladder climbed.
  EXPECT_NE(run.log.find("SUN_MAG (1) -> STAR_TRACKER"), std::string::npos)
      << "the ladder never reached the trackers from a sunlit start:\n"
      << run.log;

  // 4. Nothing was blamed and nothing was lost on the way. These are the two
  //    assertions the defect used to fail.
  EXPECT_EQ(run.log.find("excluded from the fusion"), std::string::npos)
      << "a healthy tracker was latched out for disagreeing with the vector pairs:\n"
      << run.log;
  EXPECT_EQ(run.log.find("Fine mode demoted"), std::string::npos)
      << "the arrival of a better source demoted the fine mode:\n"
      << run.log;
  EXPECT_EQ(run.log.find("Attitude lost"), std::string::npos) << run.log;
}

/// **The failure mode the fix creates, closed in the same change.** Adopting a
/// tracker the filter is rejecting is only right when the tracker is the better
/// source. A tracker that is *actually* wrong must not be able to hijack the
/// solution by being rejected persistently enough — which, without a guard, is
/// exactly what the fix would let it do, and the vehicle would then fly a wrong
/// attitude at arcsecond-class reported covariance. That is a worse state than the
/// defect it replaces.
///
/// The guard is a Mahalanobis comparison against the coarse solution's own
/// covariance at χ²₃(0.999). Here both trackers carry a 0.30 rad (17°) attitude
/// fault, far outside anything that covariance supports, so the arbitration must
/// refuse to adopt them — and must *report* the refusal, because a unit that is
/// unusable for the rest of the flight and silent about it is the other half of
/// the same defect.
///
/// **It must not latch an exclusion**, which is the subtler half. Convicting on
/// disagreement with the *coarse* solution while `readmitStarTrackers` paroles on
/// agreement with the *fine* one is a criterion mismatch, and an exclusion whose
/// parole test differs from its conviction test is permanent by construction —
/// the catalog's first FDIR rule. So the unit stays a candidate and the refusal
/// repeats at a bounded cadence instead.
///
/// Anti-thrash falls out of the two branches rather than from a rate limiter: a
/// re-seed succeeds by construction (the filter is initialised *to* the tracker,
/// so the next cycle accepts it and the rung moves), and a refusal changes no
/// state at all. Neither can oscillate; the assertions below pin the ladder
/// standing still.
///
/// Verifies REQ-FDIR-013.
TEST(SitlFaultMatrix, BadTrackerInSunMagModeIsRefusedAndReportedNotAdopted) {
  RecordProperty("verifies", "REQ-FDIR-013");
  POLARIS_REQUIRE_SITL_TOOLCHAIN();

  // No alignment window: st_b is uncalibrated and therefore not a fusion
  // candidate at all (§8.2), which leaves the king as the only unit under test
  // and the case unambiguous.
  const RunResult run = flyWithFaults("st-bad", faultMatrixOrbit(30.0, "sitl-fault-st-bad"),
                                      /*alignPairs=*/0, [](scenario::Vehicle& v, unsigned step) {
                                        if (step == 0) {
                                          v.star_trackers[kStA].model.injectAttitudeBias(
                                              pm::Vec3<pm::frames::Body>(0.0, 0.0, kStFaultRad));
                                          v.star_trackers[kStB].model.injectAttitudeBias(
                                              pm::Vec3<pm::frames::Body>(0.0, 0.0, kStFaultRad));
                                        }
                                      });
  ASSERT_TRUE(run.sim_healthy);
  ASSERT_FALSE(run.log.empty());

  ASSERT_NE(run.log.find("NONE (0) -> SUN_MAG"), std::string::npos) << run.log;

  // 1. **Not adopted.** The vehicle must not have re-seeded onto a tracker that
  //    disagrees with everything the vector data supports.
  EXPECT_EQ(run.log.find("Fine solution re-seeded from star tracker"), std::string::npos)
      << "a 17 deg tracker fault hijacked the solution — the coarse-covariance guard "
         "is not holding:\n"
      << run.log;
  EXPECT_EQ(run.log.find("-> STAR_TRACKER"), std::string::npos)
      << "the ladder climbed onto a faulted tracker:\n"
      << run.log;

  // 2. **Reported, and not latched.** Silence would be one failure — a grossly
  //    wrong unit rejected forever with nothing on the event stream — and an
  //    exclusion would be the other: it would be convicted on agreement with the
  //    coarse solution and paroled on agreement with the fine one, which is a
  //    criterion mismatch and therefore permanent. The unit stays a candidate.
  EXPECT_NE(run.log.find("Star tracker 0 not adopted"), std::string::npos)
      << "the faulted tracker was neither adopted nor reported — it would sit rejected "
         "and silent for the rest of the flight:\n"
      << run.log;
  EXPECT_EQ(run.log.find("excluded from the fusion"), std::string::npos)
      << "the refusal latched an exclusion whose re-admission criterion it can never "
         "meet:\n"
      << run.log;

  // 3. **Bounded cadence, not per cycle and not once.** The condition holds for
  //    the whole run, so the refusal repeats every MonitorAlertCycles — which is
  //    what makes it visible on a permanent fault without being 10 Hz of noise.
  //    The bound is what the cadence allows over the arc, computed rather than
  //    pasted so a tuning change fails loudly.
  constexpr double kRunS = 30.0;
  constexpr unsigned kArmSteps = 20 + 40;  // MekfNisStreak, after ~4 s of acquisition
  const std::size_t refusals = countOf(run.log, "not adopted");
  const std::size_t allowed =
      static_cast<std::size_t>(kRunS * 10.0 - kArmSteps) / kFlightMonitorAlertCycles + 1;
  EXPECT_GE(refusals, 1u) << "a permanently unusable tracker never reported:\n" << run.log;
  EXPECT_LE(refusals, allowed) << "the refusal is not on its bounded cadence: " << refusals
                               << " reports, at most " << allowed << " expected\n"
                               << run.log;

  // 4. **No thrash.** The arbitration never adopted, so the ladder never moved and
  //    the solution stayed where it belongs: on the vector pairs, published and
  //    undemoted. One source change over the run — the original promotion.
  EXPECT_EQ(countOf(run.log, "Fine solution re-seeded from star tracker"), 0u)
      << "the arbitration adopted a tracker it had refused:\n"
      << run.log;
  EXPECT_EQ(countOf(run.log, "Fine-mode source changed"), 1u)
      << "the ladder moved on a faulted tracker:\n"
      << run.log;
  EXPECT_EQ(run.log.find("Fine mode demoted"), std::string::npos)
      << "a faulted tracker demoted a fine mode that was running perfectly well on "
         "the vector pairs:\n"
      << run.log;
  EXPECT_EQ(run.log.find("Attitude lost"), std::string::npos) << run.log;
}

// ----------------------------------------------------------------------
// Magnetometer faults
// ----------------------------------------------------------------------

/// **A magnetometer is identified while the reference is independent.** In
/// tracker-fused fine mode the published attitude owes nothing to the
/// magnetometers, so the modelled IGRF field rotated through it is a genuinely
/// independent statement of what the pair should read — and the vote may use it
/// to attribute a disagreement. The faulted unit is excluded, the healthy one is
/// not, and the solution does not notice.
///
/// The last clause is the one that matters: **the filter that identifies the
/// fault has to survive it** (P51/P52). A demotion here would mean the reference
/// was circular after all.
///
/// Verifies REQ-FDIR-008.
TEST(SitlFaultMatrix, BiasedMagnetometerIsIdentifiedInTrackerModeAndReadmitted) {
  RecordProperty("verifies", "REQ-FDIR-008");
  POLARIS_REQUIRE_SITL_TOOLCHAIN();

  constexpr unsigned kClearSteps = 320;  // 12 s excluded, then 8 s of recovery
  const RunResult run = flyWithFaults("mag-st", faultMatrixOrbit(40.0, "sitl-fault-mag-st"),
                                      kAlignPairs, [](scenario::Vehicle& v, unsigned step) {
                                        if (step == kFaultSteps) {
                                          v.magnetometers[kMagB].model.injectBiasJump(
                                              pm::Vec3<pm::frames::Body>(kMagFaultTesla, 0.0, 0.0));
                                        }
                                        if (step == kClearSteps) {
                                          v.magnetometers[kMagB].model.clearFaults();
                                        }
                                      });
  ASSERT_TRUE(run.sim_healthy);
  ASSERT_FALSE(run.log.empty());

  ASSERT_NE(run.log.find("-> STAR_TRACKER"), std::string::npos)
      << "the reference was never independent, so this case tested nothing:\n"
      << run.log;

  // 1. Detected AND attributed. OUTVOTED is what says the identification ran; a
  //    FIELD_MAGNITUDE exclusion would mean the injected fault was crude enough
  //    for the per-unit gate, which is a different (and easier) row.
  EXPECT_NE(run.log.find("Magnetometer 1 excluded from the voted field"), std::string::npos)
      << "the biased magnetometer was never excluded:\n"
      << run.log;
  EXPECT_NE(run.log.find("OUTVOTED"), std::string::npos)
      << "the exclusion did not come from the identification — the disagreement was "
         "detected but not attributed:\n"
      << run.log;
  EXPECT_EQ(run.log.find("Two magnetometers disagree"), std::string::npos)
      << "the disagreement went unattributed while a tracker-fused reference was available:\n"
      << run.log;

  // 2. The healthy unit was not blamed. This is the fault-tolerance *inversion*
  //    the circularity gate exists to prevent, and it is worth its own assertion.
  EXPECT_EQ(run.log.find("Magnetometer 0 excluded"), std::string::npos)
      << "the HEALTHY magnetometer was latched out — the identification reference is "
         "not independent of the units it judges:\n"
      << run.log;

  // 3. Exactly one exclusion over the whole time the fault is held: an excluded
  //    unit that re-admits on a criterion it did not fail flaps forever, one event
  //    pair per lap (the P51 lesson).
  EXPECT_EQ(countOf(run.log, "excluded from the voted field"), 1u)
      << "the sustained fault was reported more than once: the unit flapped\n"
      << run.log;

  // 4. It cost nothing that matters.
  EXPECT_EQ(run.log.find("Fine mode demoted"), std::string::npos)
      << "identifying a magnetometer fault demoted the filter that identified it:\n"
      << run.log;
  EXPECT_EQ(run.log.find("Attitude lost"), std::string::npos) << run.log;

  // 5. Recovery is automatic and it happens: the unit agreed for
  //    MagReadmitCycles consecutive cycles and came back.
  EXPECT_NE(run.log.find("Magnetometer 1 re-admitted"), std::string::npos)
      << "the recovered magnetometer was never re-admitted:\n"
      << run.log;
}

/// **The same fault with no independent reference: refused, not guessed.** With
/// no tracker fused the fine solution is built from the sun and magnetic pairs,
/// so the rotated field reference owes part of itself to the very units it would
/// be judging. The vote must refuse it and report an honest ambiguity — because
/// the failure mode of using it is not "no identification" but the *wrong*
/// identification: a drifted unit drags the solution, which drags the reference
/// toward the drifted unit, which latches out the healthy one.
///
/// What it costs is one of two vector pairs for as long as the disagreement
/// lasts. Deliberately cheaper than the IMU equivalent, which costs the body
/// rate.
///
/// Verifies REQ-FDIR-008.
TEST(SitlFaultMatrix, BiasedMagnetometerInSunMagModeIsAmbiguousAndLatchesNothing) {
  RecordProperty("verifies", "REQ-FDIR-008");
  POLARIS_REQUIRE_SITL_TOOLCHAIN();

  const RunResult run =
      flyWithFaults("mag-ambiguous", faultMatrixOrbit(25.0, "sitl-fault-mag-ambiguous"),
                    /*alignPairs=*/0, [](scenario::Vehicle& v, unsigned step) {
                      // Both trackers out from the first step: the vehicle flies its SS+MAG
                      // rung for the whole run, which is the mode this row is about.
                      if (step == 0) {
                        dropTrackers(v);
                      }
                      if (step == 100) {
                        v.magnetometers[kMagB].model.injectBiasJump(
                            pm::Vec3<pm::frames::Body>(kMagFaultTesla, 0.0, 0.0));
                      }
                    });
  ASSERT_TRUE(run.sim_healthy);
  ASSERT_FALSE(run.log.empty());

  // 0. The premise: no tracker was ever fused, so the reference is never
  //    independent.
  ASSERT_EQ(run.log.find("-> STAR_TRACKER"), std::string::npos)
      << "a tracker was fused, so this run was not in SS+MAG mode:\n"
      << run.log;
  ASSERT_NE(run.log.find("Fine mode engaged"), std::string::npos)
      << "the vehicle never reached fine mode on the vector pairs:\n"
      << run.log;

  // 1. The refusal fired, and it is the *unattributed* one.
  EXPECT_NE(run.log.find("Two magnetometers disagree"), std::string::npos)
      << "the disagreement was never reported:\n"
      << run.log;
  // Edge-gated: the condition persists for the rest of the run, so a per-cycle
  // event would be well over a hundred of them.
  EXPECT_EQ(countOf(run.log, "Two magnetometers disagree"), 1u)
      << "MagVoteAmbiguous is not edge-gated:\n"
      << run.log;

  // 2. **Nothing was latched.** Detection is not attribution, and excluding on a
  //    reference the checked unit helped build would spend the vehicle's
  //    redundancy on a coin flip — with a bias toward blaming the healthy unit.
  EXPECT_EQ(run.log.find("excluded from the voted field"), std::string::npos)
      << "an unattributable disagreement latched an exclusion anyway:\n"
      << run.log;

  // 3. And the cost is bounded to the magnetic pair. The sun pair is untouched,
  //    the MEKF keeps updating from it, and the solution stays published.
  EXPECT_EQ(run.log.find("Attitude lost"), std::string::npos)
      << "losing the magnetic pair cost the attitude:\n"
      << run.log;
}

// ----------------------------------------------------------------------
// Sun-sensor faults
// ----------------------------------------------------------------------

/// **A confidently wrong sun sensor is detected by its neighbour and resolved by
/// the tracker.** The selected unit's calibration shifts by 17°; it keeps
/// reporting valid, sun-present, and its datasheet σ, so it stays *selected* —
/// the selector orders on reported σ and a unit that is confidently wrong wins.
/// Nothing but the runner-up can see the fault, and nothing but a
/// sun-independent attitude can say which of the two is lying.
///
/// The two halves are asserted separately because they are separate mechanisms:
/// the monitor alert (detection, available in every mode, needs no attitude) and
/// the override (resolution, gated on a tracker-fused solution).
///
/// Verifies REQ-FDIR-009.
TEST(SitlFaultMatrix, BiasedSunSensorIsCrossCheckedAndOverriddenInTrackerMode) {
  RecordProperty("verifies", "REQ-FDIR-009");
  POLARIS_REQUIRE_SITL_TOOLCHAIN();

  // The monitor needs MonitorAlertCycles = 50 consecutive cycles (5 s) over
  // threshold before it alerts, and only then may the override fire. 40 s leaves
  // 20 s after the fault — four times the persistence.
  const RunResult run = flyWithFaults("sun-cross", faultMatrixOrbit(40.0, "sitl-fault-sun-cross"),
                                      kAlignPairs, [](scenario::Vehicle& v, unsigned step) {
                                        if (step == kFaultSteps) {
                                          v.sun_sensors[kSsXp].model.injectDirectionBias(
                                              pm::Vec3<pm::frames::Body>(0.0, 0.0, kSunFaultRad));
                                        }
                                      });
  ASSERT_TRUE(run.sim_healthy);
  ASSERT_FALSE(run.log.empty());

  ASSERT_NE(run.log.find("-> STAR_TRACKER"), std::string::npos)
      << "no tracker was fused, so no sun-independent reference existed to resolve with:\n"
      << run.log;

  // 1. Detection: the cross-unit monitor alerted. It compares one sensor with
  //    another and needs no attitude at all, which is what makes it the one
  //    cross-check available in Safe mode.
  EXPECT_NE(run.log.find("SUN_CROSS_UNIT"), std::string::npos)
      << "two sun sensors 17 deg apart did not raise the cross-unit monitor:\n"
      << run.log;

  // 2. Resolution: the selection moved to the runner-up, and it named the right
  //    two units. The healthy unit is ss_yp at index 4.
  EXPECT_NE(run.log.find("Sun sensor 2 disagrees with the fine solution; using unit 4"),
            std::string::npos)
      << "the cross-check never overrode the faulted unit, or it chose the wrong one:\n"
      << run.log;

  // 3. It cost nothing. The trackers carry the solution throughout; the sun pair
  //    is a monitor in this mode, not a measurement.
  EXPECT_EQ(run.log.find("Fine mode demoted"), std::string::npos) << run.log;
  EXPECT_EQ(run.log.find("Attitude lost"), std::string::npos) << run.log;
}

/// **The Sun leaves every field of view and the vehicle does not lose its
/// attitude.** Injected as a dropout on all six units, which is the fault
/// equivalent of an eclipse — and the assertion is the behaviour rather than the
/// hoped-for one: with the trackers also gone, the MEKF keeps being updated by
/// the *magnetic* pair alone, so it neither coasts out nor demotes, and the sun
/// sensors' silence raises nothing.
///
/// **What this row does not reach, and why that is not a gap here.** The coast
/// horizons are `MekfMaxCoastSec` = 300 s and `MaxCoastSec` = 2400 s, sized for
/// the 35.6-minute eclipse of this orbit. A SITL run long enough to expire either
/// would cost more than every other row put together, so coast expiry is verified
/// where it is cheap — in the component harness, on a driven clock — and what is
/// verified here is the part only the vehicle can show: that a total loss of the
/// Sun costs neither a mode nor an event.
///
/// Verifies REQ-FDIR-010.
TEST(SitlFaultMatrix, SunLostFromEveryFieldOfViewCoastsOnTheMagneticPair) {
  RecordProperty("verifies", "REQ-FDIR-010");
  POLARIS_REQUIRE_SITL_TOOLCHAIN();

  const RunResult run = flyWithFaults("sun-lost", faultMatrixOrbit(25.0, "sitl-fault-sun-lost"),
                                      /*alignPairs=*/0, [](scenario::Vehicle& v, unsigned step) {
                                        if (step == 0) {
                                          dropTrackers(v);
                                        }
                                        if (step == 100) {
                                          for (auto& ss : v.sun_sensors) {
                                            ss.model.setDropout(true);
                                          }
                                        }
                                        if (step == 200) {  // the Sun comes back
                                          for (auto& ss : v.sun_sensors) {
                                            ss.model.clearFaults();
                                          }
                                        }
                                      });
  ASSERT_TRUE(run.sim_healthy);
  ASSERT_FALSE(run.log.empty());

  ASSERT_NE(run.log.find("Fine mode engaged"), std::string::npos)
      << "the vehicle never reached fine mode before the sun outage:\n"
      << run.log;

  // 1. No loss and no demotion: the magnetic pair alone holds the filter up, and
  //    both coast horizons are far longer than this outage.
  EXPECT_EQ(run.log.find("Attitude lost"), std::string::npos)
      << "a 10 s sun outage cost the attitude, well inside the 300 s fine coast horizon:\n"
      << run.log;
  EXPECT_EQ(run.log.find("Fine mode demoted"), std::string::npos)
      << "a sun outage demoted the fine mode; the magnetic pair was still updating it:\n"
      << run.log;

  // 2. A sun sensor reporting no sun is a **geometry fact, not a fault**, so
  //    nothing may be latched, blamed or alerted for it — which is exactly what an
  //    eclipse would otherwise trip on every orbit.
  EXPECT_EQ(run.log.find("Sun sensor"), std::string::npos)
      << "a sun dropout raised a sun-sensor FDIR event; eclipse would do the same:\n"
      << run.log;
  EXPECT_EQ(run.log.find("Residual monitor"), std::string::npos)
      << "a monitor alerted on a source that was simply absent:\n"
      << run.log;
}

// ----------------------------------------------------------------------
// Cross-domain cascades
// ----------------------------------------------------------------------

/// **An unattributable gyro pair with no tracker above it, and the honest answer
/// that a fine solution does not rescue it.** The IMU pair disagrees from before
/// the first sample, so there is no body rate; the trackers are out for the whole
/// run, so the top rung is unavailable and the fine mode can only be seeded from
/// the vector pairs.
///
/// The ladder lands where the design says: the sun and magnetic pairs are
/// independent of a gyro fault, so TRIAD acquires and the filter promotes to
/// SUN_MAG **with no gyro propagation under it**. The cost is bounded to the
/// rate, which is exactly why withholding it is the safe response rather than a
/// self-inflicted outage.
///
/// **What does not happen, and it is worth knowing it does not.** The
/// disagreement is never attributed, not even after the filter has a solution.
/// The vote's only tie-break is `Mekf::bodyRate`, which is the *propagated gyro*
/// — and `rateValid` is false precisely when no usable gyro has been seen. So the
/// one source of evidence the vote could use is the one the fault removed, and
/// the vehicle stays rate-less until the ground acts on
/// `ImuVoteAmbiguousPersistent`. That is self-consistent (an attitude filter with
/// no gyro carries no independent statement about gyros) and it is the
/// conservative outcome — latching the wrong unit is worse than carrying the
/// disagreement — but it means **no amount of attitude accuracy resolves a gyro
/// pair**. Differentiating successive tracker attitudes would, and the design
/// does not do that today.
///
/// Verifies REQ-FDIR-011.
TEST(SitlFaultMatrix, AmbiguousGyroPairWithNoTrackerHoldsAttitudeAndIsNeverAttributed) {
  RecordProperty("verifies", "REQ-FDIR-011");
  POLARIS_REQUIRE_SITL_TOOLCHAIN();

  const RunResult run = flyWithFaults("imu-st", faultMatrixOrbit(25.0, "sitl-fault-imu-st"),
                                      /*alignPairs=*/0, [](scenario::Vehicle& v, unsigned step) {
                                        if (step == 0) {
                                          v.imus[kImuB].model.injectGyroBiasJump(
                                              pm::Vec3<pm::frames::Body>(kImuFaultRadps, 0.0, 0.0));
                                          dropTrackers(v);
                                        }
                                      });
  ASSERT_TRUE(run.sim_healthy);
  ASSERT_FALSE(run.log.empty());

  // 1. The disagreement was reported, edge-gated.
  const std::size_t ambiguous = indexOf(run.log, "Two IMUs disagree");
  ASSERT_NE(ambiguous, std::string::npos) << "the gyro disagreement was never reported:\n"
                                          << run.log;
  EXPECT_EQ(countOf(run.log, "Two IMUs disagree"), 1u) << "ImuVoteAmbiguous is not edge-gated:\n"
                                                       << run.log;

  // 2. The cost is bounded to the body rate: the sun and magnetic pairs are
  //    untouched by a gyro fault, so TRIAD acquires with no gyro propagation
  //    under it and the vehicle keeps an attitude.
  ASSERT_NE(run.log.find("Attitude acquired"), std::string::npos)
      << "withholding the rate cost the attitude, which no vector pair depends on:\n"
      << run.log;
  EXPECT_EQ(run.log.find("Attitude lost"), std::string::npos) << run.log;

  // 3. The fine mode engaged anyway — on the vector pairs, with no rate under it.
  const std::size_t engaged = indexOf(run.log, "Fine mode engaged");
  ASSERT_NE(engaged, std::string::npos)
      << "the vehicle could not promote without a gyro, which the vector pairs do not need:\n"
      << run.log;
  EXPECT_NE(run.log.find("NONE (0) -> SUN_MAG"), std::string::npos)
      << "the ladder did not land on the vector pairs:\n"
      << run.log;

  // 4. **And nothing was latched, for the whole run, even with a fine solution
  //    in hand.** See the header: the reference that would attribute this is the
  //    propagated gyro, which the fault removed. Asserted rather than left
  //    implicit, because "the exclusion never came" and "the test forgot to look"
  //    are otherwise the same green.
  EXPECT_EQ(run.log.find("excluded from the rate vote"), std::string::npos)
      << "an unattributable disagreement latched an exclusion anyway — on what evidence?\n"
      << run.log;
}

/// **A magnetometer fault that predates any independent reference.** The vehicle
/// starts dark with the fault already present: there is no Sun, so no coarse
/// solution and no fine mode, and the magnetometer vote therefore has no attitude
/// to rotate a reference with. It must report an honest ambiguity and blame
/// nobody. The trackers then acquire, seed the filter, and the reference becomes
/// independent — and only *then* is the fault attributed, to the unit that
/// actually has it.
///
/// This is the cascade the whole circularity argument is about: one physical
/// fault, two different and both correct responses depending on what the vehicle
/// can honestly say about it, and the wrong unit never blamed in between.
///
/// Verifies REQ-FDIR-012.
TEST(SitlFaultMatrix, MagnetometerFaultIsBlamedOnlyOnceAnIndependentReferenceExists) {
  RecordProperty("verifies", "REQ-FDIR-012");
  POLARIS_REQUIRE_SITL_TOOLCHAIN();

  const RunResult run = flyWithFaults("cascade", faultMatrixOrbit(30.0, "sitl-fault-cascade"),
                                      kAlignPairs, [](scenario::Vehicle& v, unsigned step) {
                                        darkStart(v, step);
                                        if (step == 0) {
                                          v.magnetometers[kMagB].model.injectBiasJump(
                                              pm::Vec3<pm::frames::Body>(kMagFaultTesla, 0.0, 0.0));
                                        }
                                      });
  ASSERT_TRUE(run.sim_healthy);
  ASSERT_FALSE(run.log.empty());

  // 1. First response, before any tracker has solved: ambiguous, nothing latched.
  const std::size_t ambiguous = indexOf(run.log, "Two magnetometers disagree");
  ASSERT_NE(ambiguous, std::string::npos)
      << "the disagreement was never reported while there was no reference:\n"
      << run.log;

  // 2. Second response, once a tracker-fused solution exists: the fault is
  //    attributed, and to the faulted unit.
  const std::size_t rung = indexOf(run.log, "-> STAR_TRACKER");
  const std::size_t excluded = indexOf(run.log, "Magnetometer 1 excluded from the voted field");
  ASSERT_NE(rung, std::string::npos) << "the trackers never seeded the filter:\n" << run.log;
  ASSERT_NE(excluded, std::string::npos)
      << "the fault was never attributed even once a tracker-fused reference was available:\n"
      << run.log;

  // 3. **The ordering is the whole case.** Nothing may be blamed before the
  //    reference is independent, and the ambiguity must have been reported first.
  EXPECT_LT(ambiguous, excluded)
      << "a magnetometer was blamed before the disagreement was even reported:\n"
      << run.log;
  EXPECT_LT(rung, excluded)
      << "a magnetometer was blamed while there was no attitude independent of it — this "
         "is the circularity the gate exists to prevent:\n"
      << run.log;

  // 4. And the healthy unit was never blamed, in either phase.
  EXPECT_EQ(run.log.find("Magnetometer 0 excluded"), std::string::npos)
      << "the HEALTHY magnetometer was latched out:\n"
      << run.log;
  EXPECT_EQ(run.log.find("Attitude lost"), std::string::npos) << run.log;
}

}  // namespace
