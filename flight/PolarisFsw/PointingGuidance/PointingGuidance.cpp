// ======================================================================
// \title  PointingGuidance.cpp
// \brief  Align/constrain pointing guidance (design doc §8.4).
// ======================================================================

#include "flight/PolarisFsw/PointingGuidance/PointingGuidance.hpp"

#include <cmath>
#include <cstring>

#include "frames/eci_ecef.hpp"
#include "frames/eop.hpp"

namespace flight {
namespace {

namespace pg = polaris::gnc;
namespace pm = polaris::math;
namespace pmf = polaris::math::frames;
namespace pt = polaris::time;

constexpr double kDegToRad = 3.14159265358979323846 / 180.0;

pt::Tai taiFromNs(I64 ns) {
  return pt::Tai::fromNanosecondsSinceEpoch(ns);
}

pg::BodyVectorKind toLibBodyKind(PointingGuidance_BodyVecKind k) {
  switch (k.e) {
    case PointingGuidance_BodyVecKind::BODY_X:
      return pg::BodyVectorKind::kBodyX;
    case PointingGuidance_BodyVecKind::BODY_Y:
      return pg::BodyVectorKind::kBodyY;
    case PointingGuidance_BodyVecKind::BODY_Z:
      return pg::BodyVectorKind::kBodyZ;
    case PointingGuidance_BodyVecKind::STAR_TRACKER:
      return pg::BodyVectorKind::kStarTracker;
    case PointingGuidance_BodyVecKind::SUN_SENSOR:
      return pg::BodyVectorKind::kSunSensor;
    case PointingGuidance_BodyVecKind::CAMERA:
      return pg::BodyVectorKind::kCamera;
    default:
      return pg::BodyVectorKind::kCustom;
  }
}

pg::PointingTargetKind toLibTargetKind(PointingGuidance_TargetKind k) {
  switch (k.e) {
    case PointingGuidance_TargetKind::SUN:
      return pg::PointingTargetKind::kSun;
    case PointingGuidance_TargetKind::MOON:
      return pg::PointingTargetKind::kMoon;
    case PointingGuidance_TargetKind::NADIR:
      return pg::PointingTargetKind::kNadir;
    case PointingGuidance_TargetKind::ECEF_TARGET:
      return pg::PointingTargetKind::kEcefPoint;
    case PointingGuidance_TargetKind::STAR_J2000:
      return pg::PointingTargetKind::kStarJ2000;
    case PointingGuidance_TargetKind::J2000_X:
      return pg::PointingTargetKind::kJ2000X;
    case PointingGuidance_TargetKind::J2000_Y:
      return pg::PointingTargetKind::kJ2000Y;
    case PointingGuidance_TargetKind::J2000_Z:
      return pg::PointingTargetKind::kJ2000Z;
    case PointingGuidance_TargetKind::LVLH_X:
      return pg::PointingTargetKind::kLvlhX;
    case PointingGuidance_TargetKind::LVLH_Y:
      return pg::PointingTargetKind::kLvlhY;
    case PointingGuidance_TargetKind::LVLH_Z:
      return pg::PointingTargetKind::kLvlhZ;
    case PointingGuidance_TargetKind::SAT_TLE:
      return pg::PointingTargetKind::kSatTle;
    default:
      return pg::PointingTargetKind::kSatState;
  }
}

}  // namespace

PointingGuidance::PointingGuidance(const char* compName)
    : PointingGuidanceComponentBase(compName) {}

void PointingGuidance ::commandGuidanceAtStartup(U32 alignVecKind, U32 alignVecIndex,
                                                 bool alignVecNegate, U32 alignTgtKind,
                                                 U32 alignTgtIndex, bool alignTgtNegate,
                                                 F64 alignTgtParam0, F64 alignTgtParam1,
                                                 U32 conVecKind, U32 conVecIndex, bool conVecNegate,
                                                 U32 conTgtKind, U32 conTgtIndex, bool conTgtNegate,
                                                 F64 conTgtParam0, F64 conTgtParam1) {
  // Serialised through the real command port rather than by assigning command_
  // directly: the point of the hook is that a SITL row exercises the *uplink*
  // path, validation included, so a row that names an impossible pair is
  // refused here exactly as it would be from the ground.
  Fw::CmdArgBuffer args;
  const auto ok = [](Fw::SerializeStatus s) { return s == Fw::FW_SERIALIZE_OK; };
  if (ok(args.serializeFrom(static_cast<U8>(alignVecKind))) &&
      ok(args.serializeFrom(static_cast<U8>(alignVecIndex))) &&
      ok(args.serializeFrom(alignVecNegate)) &&
      ok(args.serializeFrom(static_cast<U8>(alignTgtKind))) &&
      ok(args.serializeFrom(static_cast<U8>(alignTgtIndex))) &&
      ok(args.serializeFrom(alignTgtNegate)) && ok(args.serializeFrom(alignTgtParam0)) &&
      ok(args.serializeFrom(alignTgtParam1)) &&
      ok(args.serializeFrom(static_cast<U8>(conVecKind))) &&
      ok(args.serializeFrom(static_cast<U8>(conVecIndex))) &&
      ok(args.serializeFrom(conVecNegate)) && ok(args.serializeFrom(static_cast<U8>(conTgtKind))) &&
      ok(args.serializeFrom(static_cast<U8>(conTgtIndex))) &&
      ok(args.serializeFrom(conTgtNegate)) && ok(args.serializeFrom(conTgtParam0)) &&
      ok(args.serializeFrom(conTgtParam1))) {
    this->get_cmdIn_InputPort(0)->invoke(this->getIdBase() + OPCODE_SET_GUIDANCE, 0, args);
  }
}

void PointingGuidance ::commandStateVectorAtStartup(U32 slot, I64 epochTaiNs, const F64 posM[3],
                                                    const F64 velMps[3], F64 sigmaM) {
  Fw::CmdArgBuffer args;
  const auto ok = [](Fw::SerializeStatus s) { return s == Fw::FW_SERIALIZE_OK; };
  if (ok(args.serializeFrom(static_cast<U8>(slot))) && ok(args.serializeFrom(epochTaiNs)) &&
      ok(args.serializeFrom(posM[0])) && ok(args.serializeFrom(posM[1])) &&
      ok(args.serializeFrom(posM[2])) && ok(args.serializeFrom(velMps[0])) &&
      ok(args.serializeFrom(velMps[1])) && ok(args.serializeFrom(velMps[2])) &&
      ok(args.serializeFrom(sigmaM))) {
    this->get_cmdIn_InputPort(0)->invoke(this->getIdBase() + OPCODE_LOAD_STATE_VECTOR, 0, args);
  }
}

namespace {

/// Join two command-string halves into a fixed 69-column buffer.
///
/// Returns false unless the pieces reassemble to exactly `kTleLineColumns`,
/// which is the check that turns an uplink truncation into a named refusal
/// instead of a parse error several fields downstream. No allocation: the
/// flight path must not call operator new after init.
bool joinTleLine(const Fw::CmdStringArg& a, const Fw::CmdStringArg& b,
                 char (&out)[PointingGuidance::kTleLineColumns + 1]) {
  const FwSizeType na = a.length();
  const FwSizeType nb = b.length();
  if (na + nb != PointingGuidance::kTleLineColumns) {
    return false;
  }
  std::memcpy(out, a.toChar(), na);
  std::memcpy(out + na, b.toChar(), nb);
  out[PointingGuidance::kTleLineColumns] = '\0';
  return true;
}

}  // namespace

void PointingGuidance ::commandTleAtStartup(U32 slot, const char* line1, const char* line2,
                                            bool verifyChecksum) {
  if (line1 == nullptr || line2 == nullptr) {
    return;
  }
  if (std::strlen(line1) != kTleLineColumns || std::strlen(line2) != kTleLineColumns) {
    return;  // the handler would refuse it; do not spend an opcode saying so
  }
  // The same split the ground tool performs, for the same reason. Fixed buffers
  // rather than substr for the same reason as the handler: no allocation.
  constexpr std::size_t kTail = kTleLineColumns - kTleSplitColumn;
  char h1a[kTleSplitColumn + 1] = {};
  char h1b[kTail + 1] = {};
  char h2a[kTleSplitColumn + 1] = {};
  char h2b[kTail + 1] = {};
  std::memcpy(h1a, line1, kTleSplitColumn);
  std::memcpy(h1b, line1 + kTleSplitColumn, kTail);
  std::memcpy(h2a, line2, kTleSplitColumn);
  std::memcpy(h2b, line2 + kTleSplitColumn, kTail);
  const Fw::CmdStringArg l1a(h1a);
  const Fw::CmdStringArg l1b(h1b);
  const Fw::CmdStringArg l2a(h2a);
  const Fw::CmdStringArg l2b(h2b);
  Fw::CmdArgBuffer args;
  const auto ok = [](Fw::SerializeStatus s) { return s == Fw::FW_SERIALIZE_OK; };
  if (ok(args.serializeFrom(static_cast<U8>(slot))) && ok(args.serializeFrom(l1a)) &&
      ok(args.serializeFrom(l1b)) && ok(args.serializeFrom(l2a)) && ok(args.serializeFrom(l2b)) &&
      ok(args.serializeFrom(verifyChecksum))) {
    this->get_cmdIn_InputPort(0)->invoke(this->getIdBase() + OPCODE_LOAD_TLE, 0, args);
  }
}

// ---------------------------------------------------------------------------
// Parameters
// ---------------------------------------------------------------------------

void PointingGuidance::reloadMountingParameters() {
  Fw::ParamValid valid = Fw::ParamValid::INVALID;

  const Vec3F64PerUnit st = this->paramGet_StBoresightsBody(valid);
  const bool st_ok = valid == Fw::ParamValid::VALID;
  const Vec3F64PerUnit ss = this->paramGet_SunSensorBoresightsBody(valid);
  const bool ss_ok = valid == Fw::ParamValid::VALID;
  const Vec3F64PerUnit cam = this->paramGet_CameraBoresightsBody(valid);
  const bool cam_ok = valid == Fw::ParamValid::VALID;

  const F64 age = this->paramGet_MaxOrbitStateAgeSec(valid);
  const bool age_ok = valid == Fw::ParamValid::VALID && std::isfinite(age) && age > 0.0;

  // The target field. Pushed into the catalogue rather than consulted per query,
  // because changing it resets each slot's integration cursor and that must
  // happen once on a parameter change, not on every cycle.
  const U8 tgt_degree = this->paramGet_TargetGeopotentialDegree(valid);
  const bool degree_ok = valid == Fw::ParamValid::VALID;
  const U8 tgt_order = this->paramGet_TargetGeopotentialOrder(valid);
  const bool order_ok = valid == Fw::ParamValid::VALID;
  if (degree_ok && order_ok) {
    pg::TargetForceModel model;
    model.degree = static_cast<int>(tgt_degree);
    model.order = static_cast<int>(tgt_order);
    catalog_.setForceModel(model);  // clamps to what the compiled table supports
  }

  // A slot whose vector is null is *not* installed. Zeroed entries are how the
  // estimator's mounting parameters already spell "no unit here", and mirroring
  // that convention means naming an absent unit is refused rather than resolving
  // to a null vector — see BodyVectorTable::resolve.
  auto install = [&](pg::BodyVectorKind kind, const Vec3F64PerUnit& src, bool ok) {
    if (!ok) {
      return;
    }
    // Vec3F64PerUnit is a flat [GncMaxUnits * 3] array — unit i occupies
    // [3i, 3i+2] — which is the same packing the estimator reads it with.
    for (int i = 0; i < polaris::gnc::kMaxSensorUnits; ++i) {
      const U32 base = static_cast<U32>(3 * i);
      if (base + 2U >= static_cast<U32>(Vec3F64PerUnit::SIZE)) {
        break;
      }
      const pm::Vec3<pmf::Body> body(src[base], src[base + 1U], src[base + 2U]);
      if (body.isFinite() && body.norm() > 0.0) {
        (void)body_vectors_.setSensor(kind, i, body);
      }
    }
  };
  install(pg::BodyVectorKind::kStarTracker, st, st_ok);
  install(pg::BodyVectorKind::kSunSensor, ss, ss_ok);
  install(pg::BodyVectorKind::kCamera, cam, cam_ok);

  max_orbit_age_s_ = age_ok ? age : 0.0;
  // Every group, not just the scalar. With `configured_ = age_ok` alone, a run
  // where MaxOrbitStateAgeSec loaded but the boresight table did not would stop
  // retrying with the table still empty, and every command naming a
  // STAR_TRACKER / SUN_SENSOR / CAMERA would be refused BODY_VECTOR_UNKNOWN —
  // "the unit is not installed" — which is a configuration failure wearing a
  // geometry failure's name, the exact thing the retry above exists to prevent.
  const bool all_ok = age_ok && st_ok && ss_ok && cam_ok && degree_ok && order_ok;
  if (!all_ok && !config_warned_) {
    // Once, not per cycle: the retry runs every cycle until it succeeds, and an
    // event per cycle would bury the transition.
    config_warned_ = true;
    this->log_WARNING_LO_GuidanceCommandRefused(GuidanceRefusal::BAD_PARAMETERS);
  }
  if (all_ok) {
    config_warned_ = false;
  }
  configured_ = all_ok;
}

void PointingGuidance::parameterUpdated(FwPrmIdType) {
  reloadMountingParameters();
}

// ---------------------------------------------------------------------------
// Inputs
// ---------------------------------------------------------------------------

void PointingGuidance::orbitStateIn_handler(FwIndexType, const OrbitEstimate& estimate) {
  orbit_ = estimate;
  orbit_fresh_ = estimate.get_valid();
}

// ---------------------------------------------------------------------------
// The cycle
// ---------------------------------------------------------------------------

bool PointingGuidance::buildContext(pt::Tai now, pg::GuidanceContext& ctx,
                                    pg::GuidanceStatus& status) {
  ctx.t = now;
  ctx.catalog = &catalog_;
  ctx.ground_points = &ground_points_;

  // ---- Orbit state, with an age bound. Beyond it every orbit-relative target
  // is refused rather than pointed from a coasted state whose error nobody
  // bounded — the same policy the OD's own degraded horizon applies.
  if (orbit_fresh_ && configured_ && orbit_.get_ageSec() <= max_orbit_age_s_) {
    const Vec3F64& r = orbit_.get_posEciM();
    const Vec3F64& v = orbit_.get_velEciMps();
    ctx.observer_position_m = pm::Vec3<pmf::ECI>(r[0], r[1], r[2]);
    ctx.observer_velocity_m_s = pm::Vec3<pmf::ECI>(v[0], v[1], v[2]);
    ctx.observer_valid = ctx.observer_position_m.isFinite() && ctx.observer_velocity_m_s.isFinite();
  } else {
    ctx.observer_valid = false;
  }

  // ---- Sun and Moon. Absence is not fatal: only a command that names them
  // fails, and it fails with NO_EPHEMERIS rather than as a generic refusal.
  PosEciMeters sun;
  PosEciMeters moon;
  bool sun_ok = false;
  bool moon_ok = false;
  if (this->isConnected_getBodyPosition_OutputPort(0)) {
    sun_ok = this->getBodyPosition_out(0, OnboardBody::SUN, now.nanosecondsSinceEpoch(), sun);
    moon_ok = this->getBodyPosition_out(0, OnboardBody::MOON, now.nanosecondsSinceEpoch(), moon);
  }
  // A COARSE grade is accepted rather than refused: the analytic Sun is good to
  // ~0.01 deg, which is far inside any pointing this guidance supports, and
  // refusing it would make sun-pointing depend on a table upload — exactly the
  // coupling the coarse fallback exists to break.
  if (sun_ok) {
    ctx.sun_position_m = pm::Vec3<pmf::ECI>(sun.get_x(), sun.get_y(), sun.get_z());
    ctx.sun_valid = ctx.sun_position_m.isFinite();
  }
  if (moon_ok) {
    ctx.moon_position_m = pm::Vec3<pmf::ECI>(moon.get_x(), moon.get_y(), moon.get_z());
    ctx.moon_valid = ctx.moon_position_m.isFinite();
  }
  // Third-body velocities are not served by OnboardTables, so the Sun/Moon
  // line-of-sight rate carries only the parallax term and says so through
  // ResolvedDirection::rate_known. See GuidanceContext.

  // ---- Earth orientation, needed only by ECEF_TARGET.
  // Earth orientation. ECEF_TARGET needs the rotation at `now`; the
  // state-vector propagator needs the EOP record itself, because it integrates
  // across a span and reduces at each of its own sub-steps. Both come from one
  // lookup, and both are absent together — which is the honest coupling, since
  // they fail for the same reason.
  EopSample eop;
  if (this->isConnected_getEopAt_OutputPort(0) &&
      this->getEopAt_out(0, now.nanosecondsSinceEpoch(), eop)) {
    eop_.ut1_minus_tai = eop.get_ut1MinusTai();
    eop_.xp_arcsec = eop.get_xpArcsec();
    eop_.yp_arcsec = eop.get_ypArcsec();
    eop_valid_ = true;
    pm::Quat<pmf::ECI, pmf::ECEF> q_eci_ecef;
    if (polaris::frames::eciFromEcef(now, eop_, q_eci_ecef)) {
      ctx.eci_from_ecef = q_eci_ecef.core().toRotationMatrix().transpose();
      ctx.earth_orientation_valid = true;
    }
  } else {
    eop_valid_ = false;
  }
  ctx.eop = eop_valid_ ? &eop_ : nullptr;

  status = pg::GuidanceStatus::kOk;
  return true;
}

void PointingGuidance::publishNoTarget(pt::Tai now, pg::GuidanceStatus status) {
  AttitudeTarget out;
  out.set_epochTaiNs(now.nanosecondsSinceEpoch());
  out.set_valid(false);
  if (this->isConnected_guidanceOut_OutputPort(0)) {
    this->guidanceOut_out(0, out);
  }

  const bool was_ok = refusal_streak_ == 0;
  ++refusal_streak_;
  last_status_ = status;
  // One event on the transition, then throttled: a geometry that has become
  // unsatisfiable stays unsatisfiable for many cycles, and one event per cycle
  // would bury the transition that matters.
  if (was_ok) {
    this->log_WARNING_HI_GuidanceLost(toRefusal(status));
  }
  this->tlmWrite_GuidanceValid(false);
  this->tlmWrite_LastRefusal(toRefusal(status));
  this->tlmWrite_RefusalStreak(refusal_streak_);
  this->tlmWrite_CommandedRateRadS(0.0);
}

void PointingGuidance::run_handler(FwIndexType, U32) {
  // Read the mounting parameters until they take. `parameterUpdated` fires on a
  // ground PRM_SET, not on the initial load from PrmDb, so a component that only
  // listened for it would fly with an empty table and refuse every command with
  // BODY_VECTOR_UNKNOWN — a configuration failure wearing a geometry failure's
  // name. Retried rather than read once because the load completes after
  // topology setup.
  if (!configured_) {
    reloadMountingParameters();
  }

  Fw::Time fw_now = this->getTime();
  const pt::Tai now =
      pt::Tai::fromNanosecondsSinceEpoch(static_cast<I64>(fw_now.getSeconds()) * 1000000000LL +
                                         static_cast<I64>(fw_now.getUSeconds()) * 1000LL);

  this->tlmWrite_TleSlotsUsed(static_cast<U8>(catalog_.occupiedCount(pg::TargetKind::kTle)));
  this->tlmWrite_StateSlotsUsed(
      static_cast<U8>(catalog_.occupiedCount(pg::TargetKind::kStateVector)));
  this->tlmWrite_GroundPointsUsed(static_cast<U8>(ground_points_.count()));
  this->tlmWrite_CustomVecsUsed(static_cast<U8>(body_vectors_.customCount()));

  // A startup command retried until the parameters it names have loaded. Once
  // it validates it becomes the active command; a *genuinely* invalid one keeps
  // failing and stays visible as an unfulfilled latch rather than being
  // silently dropped at setup.
  if (pending_) {
    if (pg::validateGuidanceCommand(pending_command_, body_vectors_, &catalog_, &ground_points_) ==
        pg::GuidanceStatus::kOk) {
      command_ = pending_command_;
      commanded_ = true;
      pending_ = false;
      this->log_ACTIVITY_HI_GuidanceCommanded(
          static_cast<BodyVecKind::T>(command_.align_vector.kind), command_.align_vector.index,
          static_cast<TargetKind::T>(command_.align_target.kind), command_.align_target.index);
    }
  }

  if (!commanded_) {
    // Uncommanded is the flight *default* — a pointing command arrives by
    // uplink — so this publishes the invalid target and says so in telemetry
    // without raising an operator alert. It used to route through
    // publishNoTarget, which fires WARNING_HI on the transition: a
    // high-severity event announcing that nothing is wrong, on every boot,
    // which is how a channel earns the filter that later hides the real one.
    AttitudeTarget out;
    out.set_epochTaiNs(now.nanosecondsSinceEpoch());
    out.set_valid(false);
    if (this->isConnected_guidanceOut_OutputPort(0)) {
      this->guidanceOut_out(0, out);
    }
    this->tlmWrite_GuidanceValid(false);
    this->tlmWrite_LastRefusal(GuidanceRefusal::NOT_COMMANDED);
    return;
  }

  pg::GuidanceContext ctx;
  pg::GuidanceStatus status = pg::GuidanceStatus::kOk;
  (void)buildContext(now, ctx, status);

  pm::Quat<pmf::Body, pmf::ECI> q;
  pm::Vec3<pmf::Body> rate;
  status = pg::solveGuidanceAttitude(command_, body_vectors_, ctx, q, rate);
  if (status != pg::GuidanceStatus::kOk) {
    publishNoTarget(now, status);
    return;
  }

  AttitudeTarget out;
  out.set_epochTaiNs(now.nanosecondsSinceEpoch());
  QuatF64 qq;
  qq[0] = q.core().w();
  qq[1] = q.core().x();
  qq[2] = q.core().y();
  qq[3] = q.core().z();
  out.set_qBodyFromEci(qq);
  Vec3F64 rr;
  rr[0] = rate.eigen().x();
  rr[1] = rate.eigen().y();
  rr[2] = rate.eigen().z();
  out.set_rateBodyRadS(rr);
  out.set_valid(true);
  if (this->isConnected_guidanceOut_OutputPort(0)) {
    this->guidanceOut_out(0, out);
  }

  if (refusal_streak_ > 0) {
    this->log_ACTIVITY_HI_GuidanceRecovered(refusal_streak_);
  }
  refusal_streak_ = 0;
  last_status_ = pg::GuidanceStatus::kOk;

  // Range and uncertainty of the aligned target, so an operator can see that a
  // stale TLE is worse than a fresh ground solution instead of both looking
  // identical from the outside.
  double range_m = 0.0;
  double sigma_m = 0.0;
  if (command_.align_target.usesIndex() &&
      (command_.align_target.kind == pg::PointingTargetKind::kSatTle ||
       command_.align_target.kind == pg::PointingTargetKind::kSatState)) {
    pg::TargetState st;
    const pg::TargetKind k = command_.align_target.kind == pg::PointingTargetKind::kSatTle
                                 ? pg::TargetKind::kTle
                                 : pg::TargetKind::kStateVector;
    // ctx.eop, the same Earth orientation the solve above used. Passing nullptr
    // here would report a range computed at a *different* fidelity from the
    // attitude actually commanded — and, since the propagator re-seeds its
    // cursor when the effective model changes, would additionally re-walk the
    // whole span twice per cycle: 2 x the measured 141 ms cold catch-up inside
    // a 100 ms frame.
    if (catalog_.positionAt(k, static_cast<int>(command_.align_target.index), now, st, ctx.eop) ==
        pg::TargetStatus::kOk) {
      range_m = (st.position_m.eigen() - ctx.observer_position_m.eigen()).norm();
      sigma_m = st.sigma_m;
    }
  }

  this->tlmWrite_GuidanceValid(true);
  this->tlmWrite_RefusalStreak(0);
  this->tlmWrite_TargetRangeM(range_m);
  this->tlmWrite_TargetSigmaM(sigma_m);
  this->tlmWrite_CommandedRateRadS(rate.norm());
}

// ---------------------------------------------------------------------------
// Commands
// ---------------------------------------------------------------------------

PointingGuidance::TargetRefusal PointingGuidance::toRefusal(pg::TargetStatus status) {
  switch (status) {
    case pg::TargetStatus::kBadSlot:
      return TargetRefusal::BAD_SLOT;
    case pg::TargetStatus::kBadElements:
      return TargetRefusal::BAD_ELEMENTS;
    case pg::TargetStatus::kBadEpoch:
      return TargetRefusal::BAD_EPOCH;
    case pg::TargetStatus::kBadOrbit:
      return TargetRefusal::BAD_ORBIT;
    case pg::TargetStatus::kOk:
    case pg::TargetStatus::kEmpty:
    case pg::TargetStatus::kPropagationFailed:
      break;
  }
  return TargetRefusal::OTHER;
}

PointingGuidance::GuidanceRefusal PointingGuidance::toRefusal(pg::GuidanceStatus status) {
  switch (status) {
    case pg::GuidanceStatus::kSameBodyAxis:
      return GuidanceRefusal::SAME_BODY_AXIS;
    case pg::GuidanceStatus::kBodyVectorUnknown:
      return GuidanceRefusal::BODY_VECTOR_UNKNOWN;
    case pg::GuidanceStatus::kBodyAxesParallel:
      return GuidanceRefusal::BODY_AXES_PARALLEL;
    case pg::GuidanceStatus::kTargetUnavailable:
      return GuidanceRefusal::TARGET_UNAVAILABLE;
    case pg::GuidanceStatus::kTargetSlotEmpty:
      return GuidanceRefusal::TARGET_SLOT_EMPTY;
    case pg::GuidanceStatus::kBadParameters:
      return GuidanceRefusal::BAD_PARAMETERS;
    case pg::GuidanceStatus::kNoEphemeris:
      return GuidanceRefusal::NO_EPHEMERIS;
    case pg::GuidanceStatus::kNoEarthOrientation:
      return GuidanceRefusal::NO_EARTH_ORIENTATION;
    case pg::GuidanceStatus::kNoOrbitState:
      return GuidanceRefusal::NO_ORBIT_STATE;
    case pg::GuidanceStatus::kDirectionsCollinear:
      return GuidanceRefusal::DIRECTIONS_COLLINEAR;
    case pg::GuidanceStatus::kBadInput:
      return GuidanceRefusal::BAD_INPUT;
    case pg::GuidanceStatus::kOk:
    default:
      return GuidanceRefusal::NOT_COMMANDED;
  }
}

void PointingGuidance::SET_GUIDANCE_cmdHandler(
    FwOpcodeType opCode, U32 cmdSeq, BodyVecKind alignVecKind, U8 alignVecIndex,
    bool alignVecNegate, TargetKind alignTgtKind, U8 alignTgtIndex, bool alignTgtNegate,
    F64 alignTgtParam0, F64 alignTgtParam1, BodyVecKind conVecKind, U8 conVecIndex,
    bool conVecNegate, TargetKind conTgtKind, U8 conTgtIndex, bool conTgtNegate, F64 conTgtParam0,
    F64 conTgtParam1) {
  pg::GuidanceCommand cmd;
  cmd.align_vector.kind = toLibBodyKind(alignVecKind);
  cmd.align_vector.index = alignVecIndex;
  cmd.align_vector.negate = alignVecNegate;
  cmd.align_target.kind = toLibTargetKind(alignTgtKind);
  cmd.align_target.index = alignTgtIndex;
  cmd.align_target.negate = alignTgtNegate;
  cmd.align_target.params[0] = alignTgtParam0;
  cmd.align_target.params[1] = alignTgtParam1;

  cmd.constrain_vector.kind = toLibBodyKind(conVecKind);
  cmd.constrain_vector.index = conVecIndex;
  cmd.constrain_vector.negate = conVecNegate;
  cmd.constrain_target.kind = toLibTargetKind(conTgtKind);
  cmd.constrain_target.index = conTgtIndex;
  cmd.constrain_target.negate = conTgtNegate;
  cmd.constrain_target.params[0] = conTgtParam0;
  cmd.constrain_target.params[1] = conTgtParam1;

  // Validated as a *pair*: the two body vectors must not name one axis and must
  // not resolve to one direction. Accepting the halves separately would leave a
  // window in which a half-updated, unvalidated command is flying.
  const pg::GuidanceStatus status =
      pg::validateGuidanceCommand(cmd, body_vectors_, &catalog_, &ground_points_);
  if (status != pg::GuidanceStatus::kOk) {
    // BODY_VECTOR_UNKNOWN before the mounting parameters have loaded is a
    // *timing* problem, not a bad command, and only at startup: latch it and
    // retry each cycle rather than rejecting a command that would be accepted a
    // moment later. Every other refusal is a real one and is reported now.
    if (status == pg::GuidanceStatus::kBodyVectorUnknown && !configured_) {
      pending_command_ = cmd;
      pending_ = true;
      this->cmdResponse_out(opCode, cmdSeq, Fw::CmdResponse::OK);
      return;
    }
    this->log_WARNING_LO_GuidanceCommandRefused(toRefusal(status));
    this->cmdResponse_out(opCode, cmdSeq, Fw::CmdResponse::VALIDATION_ERROR);
    return;
  }

  command_ = cmd;
  commanded_ = true;
  // Retire the startup latch. Without this, a latched startup command that
  // could not validate yet (parameters not loaded) stays pending, and the cycle
  // after the parameters arrive it silently overwrites whatever the ground
  // commanded in the meantime — the operator sees OK, then the vehicle slews
  // back to the old target with only a duplicate-looking EVR to show for it.
  pending_ = false;
  this->log_ACTIVITY_HI_GuidanceCommanded(alignVecKind, alignVecIndex, alignTgtKind, alignTgtIndex);
  this->cmdResponse_out(opCode, cmdSeq, Fw::CmdResponse::OK);
}

void PointingGuidance::CLEAR_GUIDANCE_cmdHandler(FwOpcodeType opCode, U32 cmdSeq) {
  commanded_ = false;
  pending_ = false;
  this->cmdResponse_out(opCode, cmdSeq, Fw::CmdResponse::OK);
}

void PointingGuidance::LOAD_TLE_cmdHandler(FwOpcodeType opCode, U32 cmdSeq, U8 slot,
                                           const Fw::CmdStringArg& line1a,
                                           const Fw::CmdStringArg& line1b,
                                           const Fw::CmdStringArg& line2a,
                                           const Fw::CmdStringArg& line2b, bool verifyChecksum) {
  // Reassemble the halves the uplink had to split (see LOAD_TLE in the .fpp for
  // why it is split at all), then refuse anything that is not exactly 69
  // columns. That length check is the whole point: a truncated line still
  // parses far enough to fail somewhere specific and misleading, and the first
  // time this happened the reported cause was a checksum error four fields
  // downstream of the actual loss.
  // Fixed buffers, not std::string: 69 columns is past any small-string
  // optimisation, so concatenating would call operator new on the command
  // thread in steady state — the allocation-in-flight rule, not a style note.
  // The catalogue takes string_view, so nothing downstream wanted an owning
  // string in the first place.
  char l1[kTleLineColumns + 1] = {};
  char l2[kTleLineColumns + 1] = {};
  const bool joined = joinTleLine(line1a, line1b, l1) && joinTleLine(line2a, line2b, l2);
  if (!joined) {
    this->log_WARNING_LO_TargetLoadRefused(true, slot, TargetRefusal::LINE_LENGTH);
    this->cmdResponse_out(opCode, cmdSeq, Fw::CmdResponse::VALIDATION_ERROR);
    return;
  }
  const pg::TleChecksumPolicy policy =
      verifyChecksum ? pg::TleChecksumPolicy::kVerify : pg::TleChecksumPolicy::kIgnore;
  const pg::TargetStatus s = catalog_.loadTle(static_cast<int>(slot), std::string_view(l1),
                                              std::string_view(l2), leap_, policy);
  if (s != pg::TargetStatus::kOk) {
    this->log_WARNING_LO_TargetLoadRefused(true, slot, toRefusal(s));
    this->cmdResponse_out(opCode, cmdSeq, Fw::CmdResponse::VALIDATION_ERROR);
    return;
  }
  this->log_ACTIVITY_HI_TargetLoaded(true, slot);
  this->cmdResponse_out(opCode, cmdSeq, Fw::CmdResponse::OK);
}

void PointingGuidance::LOAD_STATE_VECTOR_cmdHandler(FwOpcodeType opCode, U32 cmdSeq, U8 slot,
                                                    I64 epochTaiNs, F64 posXM, F64 posYM, F64 posZM,
                                                    F64 velXMps, F64 velYMps, F64 velZMps,
                                                    F64 sigmaM) {
  pg::StateVectorSlot sv;
  sv.epoch = taiFromNs(epochTaiNs);
  sv.position_m = pm::Vec3<pmf::ECI>(posXM, posYM, posZM);
  sv.velocity_m_s = pm::Vec3<pmf::ECI>(velXMps, velYMps, velZMps);
  sv.sigma_at_epoch_m = sigmaM;
  const pg::TargetStatus sv_status = catalog_.loadStateVector(static_cast<int>(slot), sv);
  if (sv_status != pg::TargetStatus::kOk) {
    this->log_WARNING_LO_TargetLoadRefused(false, slot, toRefusal(sv_status));
    this->cmdResponse_out(opCode, cmdSeq, Fw::CmdResponse::VALIDATION_ERROR);
    return;
  }
  this->log_ACTIVITY_HI_TargetLoaded(false, slot);
  this->cmdResponse_out(opCode, cmdSeq, Fw::CmdResponse::OK);
}

void PointingGuidance::CLEAR_TARGET_SLOT_cmdHandler(FwOpcodeType opCode, U32 cmdSeq, bool isTle,
                                                    U8 slot) {
  const pg::TargetKind k = isTle ? pg::TargetKind::kTle : pg::TargetKind::kStateVector;
  if (catalog_.clear(k, static_cast<int>(slot)) != pg::TargetStatus::kOk) {
    this->cmdResponse_out(opCode, cmdSeq, Fw::CmdResponse::VALIDATION_ERROR);
    return;
  }
  this->cmdResponse_out(opCode, cmdSeq, Fw::CmdResponse::OK);
}

void PointingGuidance::SET_GROUND_POINT_cmdHandler(FwOpcodeType opCode, U32 cmdSeq, U8 slot,
                                                   F64 latitudeDeg, F64 longitudeDeg, F64 heightM) {
  // Degrees on the wire, radians inside: a station is published in degrees and
  // an operator types what the mission document says.
  if (!ground_points_.set(static_cast<int>(slot), latitudeDeg * kDegToRad, longitudeDeg * kDegToRad,
                          heightM)) {
    this->log_WARNING_LO_StoreRefused(true, slot);
    this->cmdResponse_out(opCode, cmdSeq, Fw::CmdResponse::VALIDATION_ERROR);
    return;
  }
  this->log_ACTIVITY_LO_StoreUpdated(true, slot, true);
  this->cmdResponse_out(opCode, cmdSeq, Fw::CmdResponse::OK);
}

void PointingGuidance::CLEAR_GROUND_POINT_cmdHandler(FwOpcodeType opCode, U32 cmdSeq, U8 slot) {
  if (!ground_points_.clear(static_cast<int>(slot))) {
    this->log_WARNING_LO_StoreRefused(true, slot);
    this->cmdResponse_out(opCode, cmdSeq, Fw::CmdResponse::VALIDATION_ERROR);
    return;
  }
  this->log_ACTIVITY_LO_StoreUpdated(true, slot, false);
  this->cmdResponse_out(opCode, cmdSeq, Fw::CmdResponse::OK);
}

void PointingGuidance::SET_CUSTOM_BODY_VEC_cmdHandler(FwOpcodeType opCode, U32 cmdSeq, U8 slot,
                                                      F64 x, F64 y, F64 z) {
  if (!body_vectors_.setCustom(static_cast<int>(slot), pm::Vec3<pmf::Body>(x, y, z))) {
    this->log_WARNING_LO_StoreRefused(false, slot);
    this->cmdResponse_out(opCode, cmdSeq, Fw::CmdResponse::VALIDATION_ERROR);
    return;
  }
  this->log_ACTIVITY_LO_StoreUpdated(false, slot, true);
  this->cmdResponse_out(opCode, cmdSeq, Fw::CmdResponse::OK);
}

void PointingGuidance::CLEAR_CUSTOM_BODY_VEC_cmdHandler(FwOpcodeType opCode, U32 cmdSeq, U8 slot) {
  if (!body_vectors_.clearCustom(static_cast<int>(slot))) {
    this->log_WARNING_LO_StoreRefused(false, slot);
    this->cmdResponse_out(opCode, cmdSeq, Fw::CmdResponse::VALIDATION_ERROR);
    return;
  }
  this->log_ACTIVITY_LO_StoreUpdated(false, slot, false);
  this->cmdResponse_out(opCode, cmdSeq, Fw::CmdResponse::OK);
}

}  // namespace flight
