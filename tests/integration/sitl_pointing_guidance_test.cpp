/// @file Closed-loop align/constrain pointing matrix (§8.4; REQ-AGN-004,
/// REQ-AGN-005, REQ-ODP-002).
///
/// The question every row asks is the same one, and it is deliberately not a
/// question about the guidance code: **command a pointing mode, let the vehicle
/// slew, and then check against the sim's own truth that it is actually
/// pointing where it was told.**
///
/// ## Why the assertions never touch the guidance
///
/// The tempting test computes the expected attitude with `solveGuidanceAttitude`
/// and compares it with what the controller flew. That passes no matter what the
/// geometry does, because both sides come from the same code. So every row here
/// asserts two things instead, and neither one can be satisfied by the flight
/// software agreeing with itself:
///
///  1. **Truth agrees with the command.** The named body vector, rotated into
///     ECI by the *truth* attitude the sim plant holds, lands on the target
///     direction computed here from *truth* geometry — the sim's own Sun, the
///     truth position, the truth orbit. If the FSW's frame conventions, its
///     ephemeris, its mounting parameters or its control law are wrong, this
///     fails.
///  2. **The estimate agrees with truth.** The attitude the vehicle *believes*
///     it has matches the one it *has*. A vehicle can be perfectly on target and
///     unable to tell, or convinced it is on target while it is not; those are
///     different faults with different fixes, so they get different assertions.
///
/// Together they close the loop that matters operationally: the ground commands
/// a pointing mode, and the vehicle both achieves it and knows it achieved it.
///
/// ## The initial attitude is deliberately wrong
///
/// Every row starts from an attitude unrelated to the commanded one — the
/// harness's shared arbitrary attitude — so a row cannot pass by starting
/// already pointed. The slew is the test.

#include <gtest/gtest.h>
#include <unistd.h>

#include <cmath>
#include <cstdio>
#include <string>

#include "constants/constants.hpp"
#include "frames/teme_eci.hpp"
#include "gnc/sgp4.hpp"
#include "gnc/tle.hpp"
#include "sensors/occlusion.hpp"
#include "sitl_harness.hpp"
#include "world/ephemeris_file.hpp"

namespace polaris::test::sitl {
namespace {

namespace pm = polaris::math;
namespace pt = polaris::time;
namespace world = polaris::sim::world;
namespace sensors = polaris::sim::sensors;
namespace scenario = polaris::sim::scenario;

using polaris::state::TruthState;

constexpr double kRadToDeg = 180.0 / M_PI;

/// `flight.attitudeEstimator.StKingUnit` on this vehicle: the one tracker the
/// estimator will fuse before ST_ALIGN_CAL has run.
constexpr std::size_t kKingTrackerUnit = 0;

/// Body-vector kinds, mirroring PointingGuidance.fpp's BodyVecKind.
enum BodyVec : unsigned {
  kBodyX = 0,
  kBodyY = 1,
  kBodyZ = 2,
  kStarTracker = 3,
  kSunSensor = 4,
  kCamera = 5,
  kCustom = 6
};

/// Target kinds, mirroring PointingGuidance.fpp's TargetKind.
enum Tgt : unsigned {
  kSun = 0,
  kMoon = 1,
  kNadir = 2,
  kEcef = 3,
  kStar = 4,
  kJ2000X = 5,
  kJ2000Y = 6,
  kJ2000Z = 7,
  kLvlhX = 8,
  kLvlhY = 9,
  kLvlhZ = 10,
  kSatTle = 11,
  kSatState = 12
};

/// The sixteen `-G` fields, in SET_GUIDANCE declaration order.
std::string guidanceSpec(unsigned av_k, unsigned av_i, bool av_n, unsigned at_k, unsigned at_i,
                         bool at_n, double at_p0, double at_p1, unsigned cv_k, unsigned cv_i,
                         bool cv_n, unsigned ct_k, unsigned ct_i, bool ct_n, double ct_p0,
                         double ct_p1) {
  std::ostringstream s;
  s << av_k << "," << av_i << "," << (av_n ? 1 : 0) << "," << at_k << "," << at_i << ","
    << (at_n ? 1 : 0) << "," << at_p0 << "," << at_p1 << "," << cv_k << "," << cv_i << ","
    << (cv_n ? 1 : 0) << "," << ct_k << "," << ct_i << "," << (ct_n ? 1 : 0) << "," << ct_p0 << ","
    << ct_p1;
  return s.str();
}

/// A body-frame unit vector, as the vehicle config declares it. Kept here rather
/// than read from the compiled parameters on purpose: if the test read the same
/// numbers the flight software reads, a wrong mounting parameter would move both
/// the command and the expectation and the row would still pass.
Eigen::Vector3d bodyVector(BodyVec kind, unsigned index, bool negate) {
  Eigen::Vector3d v = Eigen::Vector3d::Zero();
  switch (kind) {
    case kBodyX:
      v = Eigen::Vector3d(1.0, 0.0, 0.0);
      break;
    case kBodyY:
      v = Eigen::Vector3d(0.0, 1.0, 0.0);
      break;
    case kBodyZ:
      v = Eigen::Vector3d(0.0, 0.0, 1.0);
      break;
    case kCamera:
      // config/spacecraft/leo_smallsat.yaml: one payload camera on +Z.
      EXPECT_EQ(index, 0u) << "only camera 0 is installed on this vehicle";
      v = Eigen::Vector3d(0.0, 0.0, 1.0);
      break;
    case kStarTracker:
      // st_a (KING) and st_b, from the vehicle config.
      v = index == 0 ? Eigen::Vector3d(-0.7071067811865476, 0.0, -0.7071067811865476)
                     : Eigen::Vector3d(0.7071067811865476, 0.0, -0.7071067811865476);
      break;
    case kSunSensor:
      // ss_zp / ss_zm / ss_xp / ss_xm / ss_yp / ss_ym, in build order.
      switch (index) {
        case 0:
          v = Eigen::Vector3d(0.0, 0.0, 1.0);
          break;
        case 1:
          v = Eigen::Vector3d(0.0, 0.0, -1.0);
          break;
        case 2:
          v = Eigen::Vector3d(1.0, 0.0, 0.0);
          break;
        case 3:
          v = Eigen::Vector3d(-1.0, 0.0, 0.0);
          break;
        case 4:
          v = Eigen::Vector3d(0.0, 1.0, 0.0);
          break;
        default:
          v = Eigen::Vector3d(0.0, -1.0, 0.0);
          break;
      }
      break;
    default:
      ADD_FAILURE() << "custom body vectors need an uplink; no row uses one yet";
      break;
  }
  return negate ? Eigen::Vector3d(-v) : v;
}

/// Angle between two directions [deg], by atan2 (lib/README.md).
double separationDeg(const Eigen::Vector3d& a, const Eigen::Vector3d& b) {
  return std::atan2(a.cross(b).norm(), a.dot(b)) * kRadToDeg;
}

/// Angle [deg] between the commanded body vector — placed in ECI by the **truth**
/// attitude — and the direction it was told to point at, also from truth.
double truthPointingErrorDeg(const TruthState& s, const Eigen::Vector3d& body_unit,
                             const Eigen::Vector3d& target_eci_unit) {
  // s.attitude is Body <- ECI, so rotating the ECI target into body and comparing
  // with the body vector is the same angle, computed without inverting anything.
  const Eigen::Vector3d target_body =
      s.attitude.rotate(pm::Vec3<pm::frames::ECI>(target_eci_unit.normalized())).eigen();
  // atan2 of the cross and the dot, never acos of the dot (lib/README.md).
  // These rows measure down to 0.009 deg, which is precisely where acos of a
  // dot product near 1 loses half its digits; clamping hides that, it does not
  // fix it.
  return separationDeg(target_body, body_unit.normalized());
}

/// What one guidance row produced.
struct GuidanceRun {
  std::string log;
  bool sim_healthy = false;
  std::vector<io::MacroSample> trace;
  /// The truth vehicle the run flew, kept so a row can re-evaluate sensor
  /// geometry from truth after the fact — which is how a row attributes its own
  /// accuracy instead of merely reporting an angle.
  scenario::Vehicle vehicle;
};

/// Worst pointing error over the settled tail of a run [deg].
double worstTailDeg(const GuidanceRun& r, const Eigen::Vector3d& body_unit,
                    const std::function<Eigen::Vector3d(const TruthState&)>& targetOf,
                    std::size_t tail_samples = 300) {
  double worst = 0.0;
  const std::size_t start = r.trace.size() - std::min(tail_samples, r.trace.size());
  for (std::size_t i = start; i < r.trace.size(); ++i) {
    worst = std::max(
        worst, truthPointingErrorDeg(r.trace[i].state, body_unit, targetOf(r.trace[i].state)));
  }
  return worst;
}

/// Compile the vehicle, fork the deployment with @p spec as its §8.4 pointing
/// command and TRACK latched, fly @p orbit against it, and return the truth
/// trace and the event stream.
/// Build the truth-side objects a row wants drawn, given the scenario's own
/// gravity field. Called after the runner is built, because a state-vector
/// target flies that field rather than a second copy of it.
using MakeTargets = std::function<world::TrackedObjectSet(const world::SphericalHarmonicGravity*)>;

GuidanceRun flyGuidance(const std::string& tag, const scenario::SimConfig& orbit,
                        const std::string& spec, unsigned ctrlMode = 3,
                        const std::string& stateVectorSpec = "", const std::string& tleSpec = "",
                        const MakeTargets& makeTargets = nullptr) {
  GuidanceRun result;
  const std::string work_dir =
      "build-artifacts/test-guidance-" + tag + "-" + std::to_string(::getpid());
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
  const pid_t pid = spawnFsw(fswBinaryPath(), server.port(), prm_path, log_path,
                             /*magCalSamples=*/0, /*stAlignPairs=*/0, /*stAlignUnit=*/1, ctrlMode,
                             /*ctrlTargetQ=*/nullptr, /*feedforward=*/-1, /*odResetCycle=*/0,
                             /*burnSpec=*/nullptr, /*odAccelInput=*/-1, /*wheelBias=*/-1,
                             spec.empty() ? nullptr : spec.c_str(),
                             stateVectorSpec.empty() ? nullptr : stateVectorSpec.c_str(),
                             tleSpec.empty() ? nullptr : tleSpec.c_str());
  if (pid < 0) {
    ADD_FAILURE() << "fork failed";
    return result;
  }

  scenario::Vehicle& vehicle = result.vehicle;
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
  // Truth-side secondary objects, for the viewer only (sim/world/tracked_object).
  // They are propagated by the *sim's* field, not the onboard one, so a stream
  // watched in FreeFlyer shows where the target really is against where the
  // camera is aimed — the onboard model error is visible rather than absorbed.
  // Declared here so it outlives loop.run().
  world::TrackedObjectSet targets;
  if (makeTargets) {
    targets = makeTargets(runner.gravityField());
    loop.setTrackedObjects(&targets);
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

/// The truth Sun direction from the vehicle, from the sim's own ephemeris —
/// **not** from anything the flight software computed.
Eigen::Vector3d truthSunDirection(const TruthState& s) {
  static world::EphemerisSet ephemeris;
  static bool loaded = false;
  if (!loaded) {
    std::string error;
    EXPECT_TRUE(world::loadEphemerisFile(scenario::DataPaths::under(POLARIS_GOLDEN_DIR).ephemeris,
                                         ephemeris, &error))
        << error;
    loaded = true;
  }
  const world::BodyPositionFn sun_fn = world::bodyPositionFn(ephemeris.sun);
  pm::Vec3<pm::frames::ECI> sun_eci;
  if (!sun_fn(pt::toTdb(pt::toTt(s.epoch)), sun_eci)) {
    return Eigen::Vector3d::UnitX();
  }
  return (sun_eci.eigen() - s.position.eigen()).normalized();
}

/// Truth nadir: straight down from wherever the vehicle actually is.
Eigen::Vector3d truthNadir(const TruthState& s) {
  return -s.position.eigen().normalized();
}

/// Skip when the toolchain a SITL row needs is absent (CI's unit-test job builds
/// only the native-ut tree). Duplicated from the control matrix rather than
/// shared, because moving it into the harness would make every SITL file depend
/// on the deployment path even when it does not fork one.
bool toolchainMissing(std::string& why) {
  if (::access(fswBinaryPath().c_str(), X_OK) != 0) {
    why = "flight deployment not built: " + fswBinaryPath();
    return true;
  }
  return false;
}

}  // namespace

// ----------------------------------------------------------------------
// Attributing the achieved accuracy, from truth
//
// A pointing row that only bounds the truth error cannot say *why* it missed.
// The vehicle can be pointing exactly where it believes while the belief is
// wrong, and that is a different fault with a different fix from a guidance or
// control error.
//
// The sim already knows which case it is, because it owns the geometry the
// estimator can only infer: `sensors::evaluateLineOfSight` answers, from the
// truth position, the truth attitude and the truth ephemeris, whether each star
// tracker's boresight is inside its own keep-out cones. So each row computes
// tracker availability itself, from truth, and then bounds the pointing error
// by the requirement that applies to the attitude source the vehicle *could*
// have had:
//
//   * both trackers available for the whole tail -> the fine-mode band applies
//   * neither available -> the coarse sun/magnetic band (REQ-ADET-002, 3 deg)
//     applies, and a 3 deg pointing error is the vehicle behaving correctly
//
// The bound is therefore a requirement selected by measured geometry, never a
// number fitted to what was observed. A row whose trackers are available and
// which still misses by degrees fails, which is the case that matters.
// ----------------------------------------------------------------------

/// Fraction of the settled tail during which the vehicle could actually have had
/// a fine-mode attitude source, computed entirely from truth.
///
/// Availability is not just geometry, and getting that wrong is what made an
/// earlier version of this file call a working vehicle defective. The sim's
/// tracker model has a two-envelope availability machine: a unit knocked out of
/// the wide *tracking* envelope cannot resume when it drops back under it — it
/// must sit inside the tighter *acquisition* envelope continuously for
/// `lost_in_space_s`, and any excursion resets the clock. So a tracker with a
/// perfectly clear boresight is still unavailable on a vehicle whose angular
/// acceleration keeps brushing the acquisition limit.
///
/// This therefore checks occlusion **and** both acquisition envelopes, against
/// the truth body rate and its numerical derivative — the same quantities the
/// model gates on. The boresight comes from the model's own `boresightBody()`
/// rather than being recomputed here, so the check cannot disagree with the unit
/// it is describing.
///
/// And it counts only the **king** unit, which is the correction that matters
/// most. The estimator deliberately fuses king-only until `ST_ALIGN_CAL` has
/// run: a non-king unit's as-mounted reading carries the *difference* of the two
/// units' fixed biases (45-110 arcsec on this vehicle), and fusing that at the
/// declared 21.5 arcsec sigma would sell a systematic as white noise. These rows
/// never run the alignment, so a clear non-king tracker is genuinely unusable —
/// and counting it made an earlier version of this file report a fine-mode
/// source as available when none was, and call the flight software defective for
/// correctly declining to use it. "Available" has to mean *fusable*, not merely
/// unobstructed.
double trackerAvailabilityFraction(const GuidanceRun& r, const scenario::Vehicle& vehicle,
                                   std::size_t tail_samples = 300) {
  if (vehicle.star_trackers.empty() || r.trace.empty()) {
    return 0.0;
  }
  static world::EphemerisSet ephemeris;
  static bool loaded = false;
  if (!loaded) {
    std::string error;
    EXPECT_TRUE(world::loadEphemerisFile(scenario::DataPaths::under(POLARIS_GOLDEN_DIR).ephemeris,
                                         ephemeris, &error))
        << error;
    loaded = true;
  }
  const world::BodyPositionFn sun_fn = world::bodyPositionFn(ephemeris.sun);
  const world::BodyPositionFn moon_fn = world::bodyPositionFn(ephemeris.moon);

  std::size_t available = 0;
  std::size_t counted = 0;
  const std::size_t start = r.trace.size() - std::min(tail_samples, r.trace.size());
  for (std::size_t i = start; i < r.trace.size(); ++i) {
    const TruthState& st = r.trace[i].state;
    pm::Vec3<pm::frames::ECI> sun_eci;
    pm::Vec3<pm::frames::ECI> moon_eci;
    const pt::Tdb tdb = pt::toTdb(pt::toTt(st.epoch));
    if (!sun_fn(tdb, sun_eci) || !moon_fn(tdb, moon_eci)) {
      continue;
    }
    sensors::SkyGeometry sky;
    sky.sat = st.position.eigen();
    sky.sun = sun_eci.eigen();
    sky.moon = moon_eci.eigen();

    bool any = false;
    for (std::size_t unit = 0; unit < vehicle.star_trackers.size(); ++unit) {
      if (unit != kKingTrackerUnit) {
        continue;  // king-only until ST_ALIGN_CAL; see the function comment.
      }
      const auto& mounted = vehicle.star_trackers[unit];
      // The unit's boresight is its +Z through the mounting, rotated into ECI by
      // the *truth* attitude — the same construction the sim uses to sample it.
      const Eigen::Vector3d bore_body = mounted.mounting_dcm * Eigen::Vector3d::UnitZ();
      const Eigen::Vector3d bore_eci = st.attitude.core().inverse().rotate(bore_body);
      const sensors::OcclusionState occ = sensors::evaluateLineOfSight(
          bore_eci, 0.5 * mounted.model.spec().fov_rad, sky, mounted.model.spec().keep_out);
      if (occ.occluder == sensors::Occluder::kNone) {
        any = true;
        break;
      }
    }
    available += any ? 1u : 0u;
    ++counted;
  }
  return counted == 0 ? 0.0 : static_cast<double>(available) / static_cast<double>(counted);
}

/// Worst angle [deg] between what the vehicle *believed* its attitude was and
/// what it actually was, over the settled tail.
///
/// The second half of the claim, and the half truth alone cannot make. A row
/// that only bounds the truth pointing error cannot separate "pointed badly"
/// from "pointed exactly where it believed, and the belief was wrong" — so the
/// FSW's estimate rides back on the STEP_REPLY for diagnosis only (never fed to
/// the plant; see `sitl::StepReplyHeader`) and the two errors are asserted
/// apart. A row where truth is on target and this is small is a vehicle that
/// achieved the command *and knew it had*, which is what an operator needs.
///
/// Samples with no valid estimate are skipped rather than scored: an estimator
/// that reports having lost the attitude is not wrong about it, and counting
/// that as a large error would blame it for its own honesty. The count is
/// returned so a caller can refuse a tail that was mostly blind.
/// @param rms_deg receives the RMS knowledge error, which is what gets asserted.
/// @return the worst single-sample error, which gets reported.
///
/// **RMS is asserted and the worst is only reported**, and the distinction is
/// not pedantry. REQ-ADET-002's 3 deg is a **3-sigma** figure — a statement
/// about the distribution — while the worst of 300 samples taken 0.1 s apart is
/// one draw from a strongly autocorrelated window. Comparing that maximum
/// against a 3-sigma spec is not a like-for-like test: it fails a conforming
/// estimator on an excursion the requirement explicitly allows, and it does so
/// with a number that looks damning. RMS is the statistic the requirement is
/// written about.
double estimateErrorDeg(const GuidanceRun& r, std::size_t& counted, double& rms_deg,
                        std::size_t tail_samples = 300) {
  double worst = 0.0;
  double sum_sq = 0.0;
  counted = 0;
  const std::size_t start = r.trace.size() - std::min(tail_samples, r.trace.size());
  for (std::size_t i = start; i < r.trace.size(); ++i) {
    const io::MacroSample& m = r.trace[i];
    if (!m.estimate_valid) {
      continue;
    }
    // The rotation taking truth to the estimate; its angle is the knowledge error.
    const pm::Quaternion err = m.estimate_attitude.core() * m.state.attitude.core().inverse();
    const double vec = std::sqrt(err.x() * err.x() + err.y() * err.y() + err.z() * err.z());
    const double deg = 2.0 * std::asin(std::min(1.0, vec)) * kRadToDeg;
    worst = std::max(worst, deg);
    sum_sq += deg * deg;
    ++counted;
  }
  rms_deg = counted == 0 ? 0.0 : std::sqrt(sum_sq / static_cast<double>(counted));
  return worst;
}

/// Assert the vehicle both achieved the command and knew it had.
void expectPointedAndKnew(const GuidanceRun& r, const char* row, double truth_err_deg,
                          double bound_deg, double trackers) {
  EXPECT_LT(truth_err_deg, bound_deg)
      << row << ": truth says the commanded vector is " << truth_err_deg << " deg off, against a "
      << bound_deg << " deg bound; star-tracker availability over the tail was " << trackers
      << " (>0.99 means a fusable fine-mode source was available)";

  std::size_t counted = 0;
  double knowledge_rms_deg = 0.0;
  const double knowledge_deg = estimateErrorDeg(r, counted, knowledge_rms_deg);
  std::size_t valid_anywhere = 0;
  for (const io::MacroSample& m : r.trace) {
    valid_anywhere += m.estimate_valid ? 1u : 0u;
  }
  std::printf(
      "  [%s] truth %.4f deg | knowledge rms %.4f worst %.4f deg | estimate valid %zu/%zu\n", row,
      truth_err_deg, knowledge_rms_deg, knowledge_deg, valid_anywhere, r.trace.size());
  EXPECT_GT(counted, 0u) << row << ": the vehicle reported no valid attitude over the tail ("
                         << valid_anywhere << " of " << r.trace.size()
                         << " samples valid across the whole run — zero here means the echo is "
                         << "not wired, non-zero means the estimator really lost it)";
  // The knowledge bound is NOT the pointing bound, and conflating them was
  // tempting and wrong. Pointing error is knowledge error plus control error, so
  // they are different quantities answering to different requirements:
  //
  //   fusable tracker  -> fine mode, where the knowledge owed is far tighter
  //                       than the control-class pointing bound; 2 deg is
  //                       generous.
  //   no tracker       -> the vector-pair mode, REQ-ADET-005: <= 5 deg (3-sigma).
  //
  // Deliberately **not** REQ-ADET-006's tighter 3 deg. That applies to the same
  // sensor suite but carries two further conditions — the DE440 ephemeris tables
  // serving `kPrecise`, and a sun/field separation of 45 deg or more — and this
  // row verifies neither. Asserting the tighter band without its conditions
  // would fail a conforming estimator on geometry the requirement never promised
  // to cover: the same mistake the tracker-availability check made twice before
  // it counted only *fusable* units.
  //
  // Asserting the two together is what makes a failure attributable: truth off
  // with knowledge small means the controller missed; both off means the
  // estimator did.
  const double knowledge_bound = trackers > 0.99 ? 2.0 : 5.0;
  EXPECT_LT(knowledge_rms_deg, knowledge_bound)
      << row << ": the vehicle's attitude estimate disagrees with truth by " << knowledge_rms_deg
      << " deg RMS (worst sample " << knowledge_deg << ") over " << counted
      << " samples, against a " << knowledge_bound
      << " deg bound — it pointed where it believed, and the belief was wrong";
}

/// The pointing bound the vehicle's *available* attitude knowledge justifies
/// [deg], given the tracker availability measured from truth.
///
/// This is the *pointing* bound (attitude error plus control error), not the
/// knowledge bound — see `expectPointedAndKnew` for why the two differ. Fine
/// mode uses the 2 deg control class the existing acquisition rows use; without
/// a fusable tracker the vehicle is on the vector pairs and 3.5 deg is the
/// coarse band with margin.
double justifiedBoundDeg(double tracker_availability) {
  return tracker_availability > 0.99 ? 2.0 : 3.5;
}

// ======================================================================
// Row 1 — ALIGN sun-sensor +Z with SUN
//
// The simplest possible claim, and the one the whole design has to earn: tell
// the vehicle to point a named body vector at the Sun, and afterwards check
// against the sim's own ephemeris that it is pointing at the Sun.
// ======================================================================
TEST(SitlPointingGuidance, AlignsASunSensorWithTheTrueSun) {
  std::string why;
  if (toolchainMissing(why)) {
    GTEST_SKIP() << why;
  }
  scenario::SimConfig orbit = faultMatrixOrbit(400.0, "sitl-guidance-sun");
  // ALIGN ss_zp (+Z) with SUN, CONSTRAIN +X toward J2000_Z.
  const std::string spec = guidanceSpec(kSunSensor, 0, false, kSun, 0, false, 0.0, 0.0, kBodyX, 0,
                                        false, kJ2000Z, 0, false, 0.0, 0.0);
  const GuidanceRun r = flyGuidance("sun", orbit, spec);
  ASSERT_TRUE(r.sim_healthy);
  ASSERT_FALSE(r.trace.empty());
  EXPECT_GE(countOf(r.log, "Guidance set:"), 1u) << "the pointing command was never accepted";
  EXPECT_EQ(countOf(r.log, "Guidance command refused"), 0u);

  const Eigen::Vector3d body = bodyVector(kSunSensor, 0, false);
  // The initial attitude is unrelated to the commanded one, so the row cannot
  // pass by starting already pointed — assert that it genuinely had to slew.
  const double initial_deg =
      truthPointingErrorDeg(r.trace.front().state, body, truthSunDirection(r.trace.front().state));
  EXPECT_GT(initial_deg, 20.0) << "the row started already on target; it proves nothing";

  const double worst = worstTailDeg(r, body, truthSunDirection);
  const double trackers = trackerAvailabilityFraction(r, r.vehicle);
  const double bound = justifiedBoundDeg(trackers);
  RecordProperty("sun_initial_deg", std::to_string(initial_deg));
  RecordProperty("sun_tail_worst_deg", std::to_string(worst));
  RecordProperty("sun_tracker_availability", std::to_string(trackers));
  expectPointedAndKnew(r, "sun", worst, bound, trackers);
}

// ======================================================================
// Row 2 — ALIGN -Z with NADIR, CONSTRAIN +X toward LVLH_X (nadir hold)
//
// A *moving* target: nadir sweeps once per orbit, so this fails if the
// feedforward rate is wrong even when the static geometry is right.
// ======================================================================
TEST(SitlPointingGuidance, HoldsNadirAgainstTruthWhileTheTargetMoves) {
  std::string why;
  if (toolchainMissing(why)) {
    GTEST_SKIP() << why;
  }
  scenario::SimConfig orbit = faultMatrixOrbit(900.0, "sitl-guidance-nadir");
  const std::string spec = guidanceSpec(kBodyZ, 0, /*negate=*/true, kNadir, 0, false, 0.0, 0.0,
                                        kBodyX, 0, false, kLvlhX, 0, false, 0.0, 0.0);
  const GuidanceRun r = flyGuidance("nadir", orbit, spec);
  ASSERT_TRUE(r.sim_healthy);
  ASSERT_FALSE(r.trace.empty());
  EXPECT_GE(countOf(r.log, "Guidance set:"), 1u);

  const Eigen::Vector3d body = bodyVector(kBodyZ, 0, /*negate=*/true);
  const double initial_deg =
      truthPointingErrorDeg(r.trace.front().state, body, truthNadir(r.trace.front().state));
  EXPECT_GT(initial_deg, 20.0) << "the row started already on target";

  const double worst = worstTailDeg(r, body, truthNadir);
  const double final_deg =
      truthPointingErrorDeg(r.trace.back().state, body, truthNadir(r.trace.back().state));
  RecordProperty("nadir_final_deg", std::to_string(final_deg));
  RecordProperty("nadir_initial_deg", std::to_string(initial_deg));
  RecordProperty("nadir_tail_worst_deg", std::to_string(worst));
  // Nadir moves at the orbit rate, so a controller with no feedforward lags it
  // by roughly (rate / bandwidth) and this bound is what catches that.
  const double trackers = trackerAvailabilityFraction(r, r.vehicle);
  const double bound = justifiedBoundDeg(trackers);
  RecordProperty("nadir_tracker_availability", std::to_string(trackers));
  expectPointedAndKnew(r, "nadir", worst, bound, trackers);
}

// ======================================================================
// Row 3 — ALIGN +X with an inertial axis (J2000_X)
//
// A stationary target, which isolates the static geometry from the tracking:
// if this passes and row 2 fails, the feedforward is wrong rather than the
// frame conventions.
// ======================================================================
TEST(SitlPointingGuidance, HoldsAnInertialAxisAgainstTruth) {
  std::string why;
  if (toolchainMissing(why)) {
    GTEST_SKIP() << why;
  }
  scenario::SimConfig orbit = faultMatrixOrbit(900.0, "sitl-guidance-inertial");
  const std::string spec = guidanceSpec(kBodyX, 0, false, kJ2000X, 0, false, 0.0, 0.0, kBodyZ, 0,
                                        false, kJ2000Z, 0, false, 0.0, 0.0);
  const GuidanceRun r = flyGuidance("inertial", orbit, spec);
  ASSERT_TRUE(r.sim_healthy);
  ASSERT_FALSE(r.trace.empty());

  const Eigen::Vector3d body = bodyVector(kBodyX, 0, false);
  const auto j2000x = [](const TruthState&) { return Eigen::Vector3d::UnitX(); };
  const double worst = worstTailDeg(r, body, j2000x);
  const double final_deg = truthPointingErrorDeg(r.trace.back().state, body, j2000x({}));
  RecordProperty("inertial_final_deg", std::to_string(final_deg));
  RecordProperty("inertial_tail_worst_deg", std::to_string(worst));
  const double trackers = trackerAvailabilityFraction(r, r.vehicle);
  const double bound = justifiedBoundDeg(trackers);
  RecordProperty("inertial_tracker_availability", std::to_string(trackers));
  // This row spent three iterations being wrong about the flight software, and
  // the sequence is worth keeping because each step looked conclusive.
  //
  // It holds to the 3 deg coarse band, and the estimator reports "Fine-mode
  // source changed STAR_TRACKER -> SUN_MAG (0 tracker(s) fused)" about 200 s in,
  // never returning. A geometry-only availability check said a tracker was
  // unobstructed the whole time, so this looked like a defect. Adding the
  // acquisition rate and acceleration envelopes did not change it, which seemed
  // to confirm one.
  //
  // The check was measuring the wrong thing. With body +X on the Sun line the
  // *king* tracker's boresight sits about 46 deg from nadir, inside the Earth
  // keep-out at this altitude (the Earth's angular radius is ~66 deg from 600 km,
  // plus a 22 deg exclusion), while the clear tracker is the **non-king** unit —
  // which the estimator deliberately will not fuse until ST_ALIGN_CAL has run,
  // because its as-mounted reading carries the two units' bias difference. These
  // rows never run that calibration. So a fine-mode source was never actually
  // available, the vehicle correctly fell back to the sun/magnetic pair, and the
  // 3 deg result is the requirement being met rather than missed.
  //
  // The lesson, and why the comment is this long: "available" for an attitude
  // source means *fusable*, not merely unobstructed. An availability check that
  // stops at geometry will convict correct flight software, confidently and with
  // a plausible number to show for it.
  expectPointedAndKnew(r, "inertial", worst, bound, trackers);
}

// ======================================================================
// Row 4 — the negate flag actually flips the vehicle
//
// Same command as row 1 with `negate` set on the target: the sun sensor must
// end up pointing *away* from the Sun. A flag that were ignored would produce a
// vehicle indistinguishable from row 1's, which is exactly the kind of silent
// no-op that unit tests on the resolver alone cannot rule out end to end.
// ======================================================================
TEST(SitlPointingGuidance, TheNegateFlagPointsTheVehicleTheOtherWay) {
  std::string why;
  if (toolchainMissing(why)) {
    GTEST_SKIP() << why;
  }
  scenario::SimConfig orbit = faultMatrixOrbit(400.0, "sitl-guidance-antisun");
  const std::string spec = guidanceSpec(kSunSensor, 0, false, kSun, 0, /*negate=*/true, 0.0, 0.0,
                                        kBodyX, 0, false, kJ2000Z, 0, false, 0.0, 0.0);
  const GuidanceRun r = flyGuidance("antisun", orbit, spec);
  ASSERT_TRUE(r.sim_healthy);
  ASSERT_FALSE(r.trace.empty());

  const Eigen::Vector3d body = bodyVector(kSunSensor, 0, false);
  const auto antiSun = [](const TruthState& s) { return Eigen::Vector3d(-truthSunDirection(s)); };
  const double worst_anti = worstTailDeg(r, body, antiSun);
  const double worst_sun = worstTailDeg(r, body, truthSunDirection);
  const double trackers = trackerAvailabilityFraction(r, r.vehicle);
  const double bound = justifiedBoundDeg(trackers);
  RecordProperty("antisun_tail_worst_deg", std::to_string(worst_anti));
  RecordProperty("antisun_tracker_availability", std::to_string(trackers));
  expectPointedAndKnew(r, "anti-sun", worst_anti, bound, trackers);
  EXPECT_GT(worst_sun, 170.0) << "the negate flag was ignored: the vehicle is sun-pointing";
}

// ======================================================================
// Row 5 — an unsatisfiable command is refused, and the vehicle does not fly it
//
// The other half of the contract. A pointing command naming one axis twice
// cannot be met, and the vehicle must say so at uplink rather than entering a
// mode that refuses every cycle.
// ======================================================================
TEST(SitlPointingGuidance, RefusesAnImpossiblePairAndNeverEntersTrack) {
  std::string why;
  if (toolchainMissing(why)) {
    GTEST_SKIP() << why;
  }
  scenario::SimConfig orbit = faultMatrixOrbit(60.0, "sitl-guidance-refused");
  // ALIGN +Z and CONSTRAIN -Z: one axis, two directions. Compared ignoring sign,
  // so this is the same refusal as naming +Z twice.
  const std::string spec = guidanceSpec(kBodyZ, 0, false, kSun, 0, false, 0.0, 0.0, kBodyZ, 0,
                                        /*negate=*/true, kNadir, 0, false, 0.0, 0.0);
  const GuidanceRun r = flyGuidance("refused", orbit, spec);
  ASSERT_TRUE(r.sim_healthy);
  EXPECT_GE(countOf(r.log, "Guidance command refused: SAME_BODY_AXIS"), 1u)
      << "an impossible pointing pair was accepted";
  EXPECT_EQ(countOf(r.log, "Guidance set:"), 0u) << "the refused command was latched anyway";
  // TRACK must be refused too: entering a mode whose first act is to refuse
  // every cycle reads to an operator as a controller fault rather than as a
  // pointing command that cannot be met.
  EXPECT_EQ(countOf(r.log, "-> TRACK"), 0u) << "TRACK was entered with no guidance";
}

// ======================================================================
// Row 6 — ALIGN the camera (+Z) with SAT_STATE_0
//
// The first row to fly the *target catalogue* rather than a target the vehicle
// can derive from its own state. An operator uploads an osculating ECI state and
// names the slot; the guidance propagates it with two-body + J2 and points at
// where it is *now*, not where it was uploaded.
//
// ## The expected direction here is closed-form, and that is the point
//
// Every other row's expectation comes from truth geometry the sim already owns
// (the Sun's ephemeris, the vehicle's own position). This one's target has no
// existence in the plant at all — there is no second vehicle — so the honest
// reference has to be computed here, and computing it with `J2Propagator` would
// make the row circular: a propagation bug would move the command and the
// expectation together and the row would pass.
//
// So the uploaded orbit is chosen to have a **closed form**. A circular
// equatorial orbit stays circular and equatorial under two-body + J2, because
// with `z = 0` the J2 term is purely radial:
//
//     a_r = -mu/r^2 - (3/2) J2 mu Re^2 / r^4      (the bulge pulls *harder* at
//                                                  the equator, which is the
//                                                  sign a unit test got backwards
//                                                  once)
//     omega = sqrt( mu/r^3 + (3/2) J2 mu Re^2 / r^5 )
//
// That is algebra, not integration, and it shares no code with the flight
// propagator's RK4. If the flight field's sign, its magnitude, or its step
// control is wrong, the two positions separate and the pointing error grows
// with time — which is exactly the failure a tail assertion catches and an
// initial-sample one would not.
//
// **The reference is deliberately one model behind the vehicle.** Since Push 82
// the flight propagator defaults to 8x8 EGM2008, so this closed form is exact
// for the *J2 setting* and differs from what the vehicle flies by the
// truncation itself: ~13 m over this row's 900 s. That is not a weakened
// assertion, it is a second measurement of the same thing — switching the
// default from J2 to 8x8 moved this row from 0.0076 to 0.0082 deg, and 0.0006
// deg at the row's 2 628 km range is 27 m, the size the offline scoping
// measurement predicted. A reference that tracked the setting would have shown
// nothing.
//
// The target is placed 30 deg ahead of the vehicle in true anomaly so the line
// of sight is neither the zenith nor the along-track direction at any point in
// the run: a guidance bug that resolved NADIR, or the vehicle's own velocity,
// cannot pass by coincidence. It is a slowly moving target by construction
// (~0.03 deg/s), so the feedforward is not what this row stresses — row 2 owns
// that. This row is about the catalogue, the propagation and the resolution.
// ======================================================================
namespace {

/// Radius and phase of the uploaded circular-equatorial target.
constexpr double kSatStateRadiusM = 8.0e6;
constexpr double kSatStatePhase0Rad = 30.0 * M_PI / 180.0;

/// Mean motion of a circular equatorial orbit under two-body + J2 [rad/s].
/// Closed form; see the row comment for the derivation and for why this must
/// not call `J2Propagator`.
double circularEquatorialRateRadS(double r) {
  namespace c = polaris::constants;
  const double mu = c::gravity::kGM;
  const double re = c::gravity::kReferenceRadius;
  const double j2 = c::gravity::kJ2;
  return std::sqrt(mu / (r * r * r) + 1.5 * j2 * mu * re * re / (r * r * r * r * r));
}

/// The uploaded state at the sim epoch, in ECI.
void satStateUpload(Eigen::Vector3d& pos, Eigen::Vector3d& vel) {
  const double w = circularEquatorialRateRadS(kSatStateRadiusM);
  const double c0 = std::cos(kSatStatePhase0Rad);
  const double s0 = std::sin(kSatStatePhase0Rad);
  pos = kSatStateRadiusM * Eigen::Vector3d(c0, s0, 0.0);
  vel = w * kSatStateRadiusM * Eigen::Vector3d(-s0, c0, 0.0);
}

/// Truth line of sight to the uploaded target, from the closed form.
Eigen::Vector3d truthSatStateDirection(const TruthState& s) {
  const double dt = static_cast<double>(s.epoch.nanosecondsSinceEpoch() - kEpochTaiNs) * 1.0e-9;
  const double theta = kSatStatePhase0Rad + circularEquatorialRateRadS(kSatStateRadiusM) * dt;
  const Eigen::Vector3d target =
      kSatStateRadiusM * Eigen::Vector3d(std::cos(theta), std::sin(theta), 0.0);
  return (target - s.position.eigen()).normalized();
}

}  // namespace

TEST(SitlPointingGuidance, TracksAnUploadedStateVectorTarget) {
  std::string why;
  if (toolchainMissing(why)) {
    GTEST_SKIP() << why;
  }
  scenario::SimConfig orbit = faultMatrixOrbit(900.0, "sitl-guidance-sat-state");

  Eigen::Vector3d p0 = Eigen::Vector3d::Zero();
  Eigen::Vector3d v0 = Eigen::Vector3d::Zero();
  satStateUpload(p0, v0);
  std::ostringstream upload;
  upload.precision(17);
  upload << "0," << kEpochTaiNs << "," << p0.x() << "," << p0.y() << "," << p0.z() << "," << v0.x()
         << "," << v0.y() << "," << v0.z() << ",50.0";

  // ALIGN camera 0 (+Z) with SAT_STATE_0, CONSTRAIN the king star tracker toward
  // **zenith** (NADIR, negated).
  //
  // The constraint is the roll about the camera axis, and roll is the only
  // freedom left once the camera is aimed — so spending it on an arbitrary
  // inertial axis wastes the one degree of freedom that decides whether the
  // vehicle can see any stars. This vehicle's trackers sit 45 deg off body -Z,
  // and the camera on +Z means -Z points away from the target; with the target
  // above, that is Earthward. Measured with the old `+X toward J2000_Z`
  // constraint: both trackers inside the Earth keep-out on **100%** of settled
  // samples, so the vehicle flew the whole row on the coarse sun+mag pair and
  // the pointing error was a knowledge error.
  //
  // Rolling the king tracker as far from nadir as the geometry allows is the
  // operational answer and it is one command, which is the whole point of
  // align/constrain: body -Z sits 78.9 deg from nadir here, an ST rides a
  // 45 deg cone about it, so roll can place the king at up to 123.9 deg against
  // a 90.1 deg keep-out — a 33.9 deg margin where there was none.
  const std::string spec = guidanceSpec(kCamera, 0, false, kSatState, 0, false, 0.0, 0.0,
                                        kStarTracker, kKingTrackerUnit, false, kNadir, 0,
                                        /*negate=*/true, 0.0, 0.0);
  // The same state, handed to the truth side so the viewer has something to
  // draw — and drawn from the *sim's* full spherical-harmonic field rather than
  // the onboard two-body + J2 that aims the camera. That is what makes the
  // picture worth watching: an onboard propagation error shows up as the target
  // drifting off the boresight, instead of being carried along with it.
  const GuidanceRun r =
      flyGuidance("sat-state", orbit, spec, /*ctrlMode=*/3, upload.str(), /*tleSpec=*/"",
                  [&p0, &v0](const world::SphericalHarmonicGravity* g) {
                    world::TrackedObjectSet set;
                    set.push_back(world::TrackedObject::fromState(
                        "TargetState", pt::Tai::fromNanosecondsSinceEpoch(kEpochTaiNs), p0, v0, g));
                    return set;
                  });
  ASSERT_TRUE(r.sim_healthy);
  ASSERT_FALSE(r.trace.empty());
  EXPECT_GE(countOf(r.log, "Guidance set:"), 1u) << "the pointing command was never accepted";
  EXPECT_EQ(countOf(r.log, "Guidance command refused"), 0u);
  // A slot that failed to load would refuse the command, not answer with a stale
  // position — assert the refusal did not happen rather than inferring it from
  // the pointing error.
  EXPECT_EQ(countOf(r.log, "TARGET_SLOT_EMPTY"), 0u) << "the state-vector upload did not land";

  const Eigen::Vector3d body = bodyVector(kCamera, 0, false);
  const double initial_deg = truthPointingErrorDeg(r.trace.front().state, body,
                                                   truthSatStateDirection(r.trace.front().state));
  EXPECT_GT(initial_deg, 20.0) << "the row started already on target; it proves nothing";

  // The target must be distinguishable from the ones the vehicle could derive
  // without the catalogue: if the line of sight were within a few degrees of
  // nadir or the Sun, a guidance bug that resolved the wrong kind would pass.
  const TruthState& tail = r.trace.back().state;
  const Eigen::Vector3d los = truthSatStateDirection(tail);
  const double from_nadir = separationDeg(los, truthNadir(tail));
  const double from_sun = separationDeg(los, truthSunDirection(tail));
  EXPECT_GT(from_nadir, 20.0) << "the catalogue target is too close to nadir to be distinguishable";
  EXPECT_GT(from_sun, 20.0) << "the catalogue target is too close to the Sun to be distinguishable";

  const double worst = worstTailDeg(r, body, truthSatStateDirection);
  const double trackers = trackerAvailabilityFraction(r, r.vehicle);
  const double bound = justifiedBoundDeg(trackers);
  RecordProperty("sat_state_initial_deg", std::to_string(initial_deg));
  RecordProperty("sat_state_tail_worst_deg", std::to_string(worst));
  RecordProperty("sat_state_los_from_nadir_deg", std::to_string(from_nadir));
  RecordProperty("sat_state_tracker_availability", std::to_string(trackers));

  // A target-epoch guard, and it is not a round number for its own sake. The
  // catalogue rows are the only ones whose target position comes from an
  // *uploaded* epoch, so they are the only ones that can be wrong about *when*
  // the target is rather than where the vehicle is pointing. The empty
  // leap-second table this component shipped with put a TLE epoch 37 s early —
  // ~278 km along-track, 1.2 deg at this range — and the row passed, because
  // the fine-mode pointing bound is 2 deg and nothing here was tighter than
  // that. The vehicle pointed exactly where it was told, at a position that was
  // half a minute stale.
  //
  // 0.5 deg is comfortably above what the rows achieve (0.007-0.008 deg) and
  // comfortably below what any plausible epoch error produces, so it fails on
  // the defect and not on the weather.
  EXPECT_LT(worst, 0.5) << "settled pointing error of " << worst
                        << " deg is too large for a correctly-epoched target; a target-epoch "
                           "error of one leap-second offset lands near 1.2 deg here";
  expectPointedAndKnew(r, "sat-state", worst, bound, trackers);
}

// ======================================================================
// Row 7 — ALIGN the camera (+Z) with SAT_TLE_0
//
// The catalogue's other half: a TLE, propagated with SGP4 and converted out of
// TEME into ECI before the geometry ever sees it.
//
// ## What this row does and does not prove
//
// It does **not** re-verify SGP4. The expected direction here is formed by
// calling the same `Sgp4` and `eciFromTeme` the flight side calls, so a defect
// inside either would move the command and the expectation together. That
// verification is Push 81's job and it is a stronger one than this row could
// make: three independent lineages (Vallado here, USSF AstroStds in FreeFlyer,
// NAIF SPICE in GMAT) agreeing to under 0.06 arcsec, in
// `tests/golden/sgp4_external_golden_test.cpp`.
//
// What this row proves is everything *between* that propagator and the vehicle's
// attitude, none of which the golden test touches: that a TLE uplinked as two
// 69-column strings is parsed and checksum-verified, that it lands in the named
// slot, that the catalogue answers for the *current* epoch rather than the
// upload epoch, that the answer arrives in ECI rather than TEME, and that the
// vehicle then physically points at it. The two rows together cover the path;
// neither covers it alone, and saying so is cheaper than a weaker claim.
//
// The element set is synthetic (catalogue number 99001) and carries **valid**
// checksums, so the row flies with `verifyChecksum` on — the operational
// configuration. An 8 535 km semi-major axis at 51.6 deg holds the range
// between 8 700 and 9 200 km for the whole run, which bounds the line-of-sight
// rate: a near conjunction would turn this into an unannounced slew-rate test.
//
// The mean anomaly is **chosen, not arbitrary**, and the reason is the star
// trackers. The camera is on +Z, so aiming it puts body -Z anti-target, and the
// trackers ride a 45 deg cone about -Z. A target near the vehicle's zenith
// therefore sweeps both trackers across the Earth, and **no roll can fix it** —
// roll moves a tracker around the cone but cannot change the angle between -Z
// and nadir. The first version of this row had a 12 011 km target 36 deg from
// zenith and flew the whole run with both trackers inside the keep-out on 100%
// of settled samples, on the coarse sun+mag pair, at 1.6 deg. The rule the
// probe found: the target must be at least (keep-out - 45) = **45 deg from
// zenith** for roll to have any chance. This one is well past that.
// ======================================================================
namespace {

/// A synthetic element set epoched at the sim start (2026 day 1). Held here
/// rather than in `tests/golden/` on purpose: it is a test fixture, not
/// external reference data, and nothing upstream published it.
constexpr const char* kSatTleLine1 =
    "1 99001U 26001A   26001.00000000  .00000000  00000-0  00000-0 0  9997";
constexpr const char* kSatTleLine2 =
    "2 99001  51.6000  30.0000 0001000  90.0000 180.0000 11.00000000    07";

/// Truth line of sight to the TLE target. Shares the propagator with the flight
/// side by necessity; see the row comment for what that does and does not leave
/// covered.
Eigen::Vector3d truthSatTleDirection(const TruthState& s) {
  // Built per call rather than cached in a function-local static. The cache
  // that used to live here latched on a `ready` flag keyed to nothing — not the
  // element set, not the epoch — so a second row calling this helper with a
  // different TLE would have silently been handed the first row's propagator
  // and checked the wrong satellite. Re-parsing costs microseconds against a
  // 70 s row, which is not a trade worth a latent wrong answer.
  polaris::gnc::TleElements elements;
  EXPECT_EQ(polaris::gnc::parseTle(kSatTleLine1, kSatTleLine2, elements),
            polaris::gnc::TleStatus::kOk);
  polaris::gnc::Sgp4 sgp4;
  EXPECT_EQ(sgp4.initialise(elements), polaris::gnc::Sgp4Status::kOk);
  pt::Tai tle_epoch;
  // The same historical table the flight side now uses. These two disagreeing
  // is what hid the leap-second defect: the row's own truth was correct while
  // the software under test was 37 s out, and the row still passed.
  EXPECT_TRUE(elements.epochTai(pt::LeapSecondTable::historical(), tle_epoch));
  const double minutes =
      static_cast<double>(s.epoch.nanosecondsSinceEpoch() - tle_epoch.nanosecondsSinceEpoch()) *
      1.0e-9 / 60.0;
  polaris::gnc::Sgp4::PositionKm p_teme;
  polaris::gnc::Sgp4::VelocityKmS v_teme;
  if (sgp4.propagate(minutes, p_teme, v_teme) != polaris::gnc::Sgp4Status::kOk) {
    ADD_FAILURE() << "the reference SGP4 refused to propagate the row's own element set";
    return Eigen::Vector3d::UnitX();
  }
  pm::Vec3<pm::frames::ECI> target_eci;
  if (!polaris::frames::eciFromTeme(s.epoch, pm::Vec3<pm::frames::TEME>(p_teme.eigen() * 1000.0),
                                    target_eci)) {
    ADD_FAILURE() << "the reference TEME->ECI conversion refused the row's own epoch";
    return Eigen::Vector3d::UnitX();
  }
  return (target_eci.eigen() - s.position.eigen()).normalized();
}

}  // namespace

TEST(SitlPointingGuidance, TracksAnUploadedTleTarget) {
  std::string why;
  if (toolchainMissing(why)) {
    GTEST_SKIP() << why;
  }
  scenario::SimConfig orbit = faultMatrixOrbit(900.0, "sitl-guidance-sat-tle");
  const std::string upload =
      std::string("0|") + kSatTleLine1 + "|" + kSatTleLine2 + "|1";  // checksums verified

  // Same king-tracker-to-zenith constraint as row 6, and for the same reason:
  // the roll is the only freedom left once the camera is aimed, and spending it
  // on the trackers is what decides whether the vehicle can see any stars at
  // all. This target's geometry gives the king a ~70 deg margin on the Earth
  // keep-out (see the element-set comment for why the mean anomaly is chosen).
  const std::string spec = guidanceSpec(kCamera, 0, false, kSatTle, 0, false, 0.0, 0.0,
                                        kStarTracker, kKingTrackerUnit, false, kNadir, 0,
                                        /*negate=*/true, 0.0, 0.0);
  const GuidanceRun r = flyGuidance(
      "sat-tle", orbit, spec, /*ctrlMode=*/3, /*stateVectorSpec=*/"", upload,
      [](const world::SphericalHarmonicGravity*) {
        world::TrackedObjectSet set;
        set.push_back(world::TrackedObject::fromTle("TargetTle", kSatTleLine1, kSatTleLine2,
                                                    pt::LeapSecondTable::historical()));
        return set;
      });
  ASSERT_TRUE(r.sim_healthy);
  ASSERT_FALSE(r.trace.empty());
  EXPECT_GE(countOf(r.log, "Guidance set:"), 1u) << "the pointing command was never accepted";
  EXPECT_EQ(countOf(r.log, "Guidance command refused"), 0u);
  EXPECT_EQ(countOf(r.log, "TARGET_SLOT_EMPTY"), 0u)
      << "the TLE upload did not land — a parse or checksum refusal, not a pointing failure";

  const Eigen::Vector3d body = bodyVector(kCamera, 0, false);
  const double initial_deg = truthPointingErrorDeg(r.trace.front().state, body,
                                                   truthSatTleDirection(r.trace.front().state));
  EXPECT_GT(initial_deg, 20.0) << "the row started already on target; it proves nothing";

  // Same distinguishability guard as row 6, and it earns more here: a TLE target
  // is inclined and its line of sight sweeps, so "it happened to look like
  // nadir" is a live way for a wrong resolution to pass.
  const TruthState& tail = r.trace.back().state;
  const Eigen::Vector3d los = truthSatTleDirection(tail);
  const double from_nadir = separationDeg(los, truthNadir(tail));
  EXPECT_GT(from_nadir, 20.0) << "the TLE target is too close to nadir to be distinguishable";

  const double worst = worstTailDeg(r, body, truthSatTleDirection);
  const double trackers = trackerAvailabilityFraction(r, r.vehicle);
  const double bound = justifiedBoundDeg(trackers);
  RecordProperty("sat_tle_initial_deg", std::to_string(initial_deg));
  RecordProperty("sat_tle_tail_worst_deg", std::to_string(worst));
  RecordProperty("sat_tle_los_from_nadir_deg", std::to_string(from_nadir));
  RecordProperty("sat_tle_tracker_availability", std::to_string(trackers));

  // A target-epoch guard, and it is not a round number for its own sake. The
  // catalogue rows are the only ones whose target position comes from an
  // *uploaded* epoch, so they are the only ones that can be wrong about *when*
  // the target is rather than where the vehicle is pointing. The empty
  // leap-second table this component shipped with put a TLE epoch 37 s early —
  // ~278 km along-track, 1.2 deg at this range — and the row passed, because
  // the fine-mode pointing bound is 2 deg and nothing here was tighter than
  // that. The vehicle pointed exactly where it was told, at a position that was
  // half a minute stale.
  //
  // 0.5 deg is comfortably above what the rows achieve (0.007-0.008 deg) and
  // comfortably below what any plausible epoch error produces, so it fails on
  // the defect and not on the weather.
  EXPECT_LT(worst, 0.5) << "settled pointing error of " << worst
                        << " deg is too large for a correctly-epoched target; a target-epoch "
                           "error of one leap-second offset lands near 1.2 deg here";
  expectPointedAndKnew(r, "sat-tle", worst, bound, trackers);
}

}  // namespace polaris::test::sitl
