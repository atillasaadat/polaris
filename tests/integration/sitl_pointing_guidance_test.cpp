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
#include <string>

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

/// Angle [deg] between the commanded body vector — placed in ECI by the **truth**
/// attitude — and the direction it was told to point at, also from truth.
double truthPointingErrorDeg(const TruthState& s, const Eigen::Vector3d& body_unit,
                             const Eigen::Vector3d& target_eci_unit) {
  // s.attitude is Body <- ECI, so rotating the ECI target into body and comparing
  // with the body vector is the same angle, computed without inverting anything.
  const Eigen::Vector3d target_body =
      s.attitude.rotate(pm::Vec3<pm::frames::ECI>(target_eci_unit.normalized())).eigen();
  const double c = std::clamp(target_body.dot(body_unit.normalized()), -1.0, 1.0);
  return std::acos(c) * kRadToDeg;
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
GuidanceRun flyGuidance(const std::string& tag, const scenario::SimConfig& orbit,
                        const std::string& spec, unsigned ctrlMode = 3) {
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
                             spec.empty() ? nullptr : spec.c_str());
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

/// The pointing bound the vehicle's *available* attitude knowledge justifies
/// [deg], given the tracker availability measured from truth.
///
/// Fine mode is bounded well above REQ-ADET-003's 0.05 deg because a pointing
/// error is the attitude error plus the control error, and this is the latter's
/// class bound — the same 2 deg the existing acquisition rows use. Coarse is
/// REQ-ADET-002's 3 deg plus margin.
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
  RecordProperty("sun_initial_deg", std::to_string(initial_deg));
  RecordProperty("sun_tail_worst_deg", std::to_string(worst));
  EXPECT_LT(worst, 2.0) << "commanded the sun sensor at the Sun and truth says it is " << worst
                        << " deg off";
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
  EXPECT_LT(worst, bound) << "nadir hold is " << worst << " deg off in truth, against a " << bound
                          << " deg bound; star-tracker availability over the tail was " << trackers;
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
  EXPECT_LT(worst, bound) << "inertial hold is " << worst << " deg off in truth, against a "
                          << bound << " deg bound; star-tracker availability over the tail was "
                          << trackers
                          << " (>0.99 means a fine-mode source was geometrically available)";
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
  EXPECT_LT(worst_anti, bound) << "anti-sun pointing is " << worst_anti
                               << " deg off in truth, against a " << bound
                               << " deg bound; star-tracker availability was " << trackers;
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

}  // namespace polaris::test::sitl
