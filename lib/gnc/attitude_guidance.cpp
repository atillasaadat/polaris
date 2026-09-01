#include "gnc/attitude_guidance.hpp"

#include <cmath>

#include "constants/constants.hpp"
#include "math/frame_geometry.hpp"

namespace polaris::gnc {
namespace {

namespace c = polaris::constants;

/// Below this the two commanded directions are treated as collinear.
///
/// sin(0.5 deg) — set where the *result* stops being usable rather than where
/// the arithmetic stops working. At half a degree of separation the roll
/// solution swings by tens of degrees for arc-minute changes in the inputs, so
/// an answer here would be far less certain than it looks.
constexpr double kCollinearSin = 8.7e-3;

/// Body axes closer than this cannot define a frame at all.
constexpr double kAxisParallelSin = 1.0e-6;

/// Geodetic (lat, lon, height) on the WGS-84 ellipsoid to ECEF [m].
///
/// The standard closed form. Written here rather than pulled from a library
/// because it is eight lines and this is its only caller; if a second appears it
/// should move to `lib/frames`.
Eigen::Vector3d ecefFromGeodetic(double lat_rad, double lon_rad, double height_m) {
  const double sin_lat = std::sin(lat_rad);
  const double cos_lat = std::cos(lat_rad);
  const double n =
      c::wgs84::kSemiMajorAxis / std::sqrt(1.0 - c::wgs84::kEccentricitySq * sin_lat * sin_lat);
  return Eigen::Vector3d((n + height_m) * cos_lat * std::cos(lon_rad),
                         (n + height_m) * cos_lat * std::sin(lon_rad),
                         (n * (1.0 - c::wgs84::kEccentricitySq) + height_m) * sin_lat);
}

/// Direction and rotation rate of the line of sight from observer to target.
///
/// `omega = (r x v) / |r|^2` on the *relative* state — the rotation rate of the
/// unit line-of-sight vector. Range rate contributes nothing, which is exactly
/// right: closing on a target does not rotate the direction to it.
bool lineOfSight(const Eigen::Vector3d& obs_r, const Eigen::Vector3d& obs_v,
                 const Eigen::Vector3d& tgt_r, const Eigen::Vector3d& tgt_v,
                 ResolvedDirection& out) {
  const Eigen::Vector3d r = tgt_r - obs_r;
  const double range2 = r.squaredNorm();
  if (!(range2 > 0.0)) {
    return false;
  }
  const double range = std::sqrt(range2);
  out.unit = math::Vec3<math::frames::ECI>(r / range);
  out.rate_rad_s = math::Vec3<math::frames::ECI>((r.cross(tgt_v - obs_v)) / range2);
  return out.unit.isFinite() && out.rate_rad_s.isFinite();
}

/// The inertial angular velocity of the orbit frame, `(r x v)/|r|^2`. Every
/// orbit-tied direction — nadir and all three LVLH axes — turns at exactly this.
Eigen::Vector3d orbitRate(const Eigen::Vector3d& r, const Eigen::Vector3d& v) {
  return r.cross(v) / r.squaredNorm();
}

void applySign(bool negate, ResolvedDirection& d) {
  if (negate) {
    d.unit = math::Vec3<math::frames::ECI>(-d.unit.eigen());
    // The rate is the rotation rate of the *axis*, which negating does not
    // change: -u rotates with the same angular velocity as u.
  }
}

}  // namespace

const char* toString(GuidanceStatus status) {
  switch (status) {
    case GuidanceStatus::kOk:
      return "OK";
    case GuidanceStatus::kSameBodyAxis:
      return "SAME_BODY_AXIS";
    case GuidanceStatus::kBodyVectorUnknown:
      return "BODY_VECTOR_UNKNOWN";
    case GuidanceStatus::kBodyAxesParallel:
      return "BODY_AXES_PARALLEL";
    case GuidanceStatus::kTargetUnavailable:
      return "TARGET_UNAVAILABLE";
    case GuidanceStatus::kTargetSlotEmpty:
      return "TARGET_SLOT_EMPTY";
    case GuidanceStatus::kBadParameters:
      return "BAD_PARAMETERS";
    case GuidanceStatus::kNoEphemeris:
      return "NO_EPHEMERIS";
    case GuidanceStatus::kNoEarthOrientation:
      return "NO_EARTH_ORIENTATION";
    case GuidanceStatus::kNoOrbitState:
      return "NO_ORBIT_STATE";
    case GuidanceStatus::kDirectionsCollinear:
      return "DIRECTIONS_COLLINEAR";
    case GuidanceStatus::kBadInput:
      return "BAD_INPUT";
  }
  return "UNKNOWN";
}

GuidanceStatus resolveTarget(const PointingTargetRef& ref, const GuidanceContext& ctx,
                             ResolvedDirection& out) {
  ResolvedDirection d;
  d.rate_rad_s = math::Vec3<math::frames::ECI>::Zero();

  // Directions that need this vehicle's own state, gathered once so the
  // requirement is stated in one place rather than repeated per branch.
  const bool needs_state =
      ref.kind == PointingTargetKind::kSun || ref.kind == PointingTargetKind::kMoon ||
      ref.kind == PointingTargetKind::kNadir || ref.kind == PointingTargetKind::kEcefPoint ||
      ref.kind == PointingTargetKind::kLvlhX || ref.kind == PointingTargetKind::kLvlhY ||
      ref.kind == PointingTargetKind::kLvlhZ || ref.kind == PointingTargetKind::kSatTle ||
      ref.kind == PointingTargetKind::kSatState;
  if (needs_state && !ctx.observer_valid) {
    return GuidanceStatus::kNoOrbitState;
  }
  const Eigen::Vector3d obs_r = ctx.observer_position_m.eigen();
  const Eigen::Vector3d obs_v = ctx.observer_velocity_m_s.eigen();

  switch (ref.kind) {
    case PointingTargetKind::kSun:
    case PointingTargetKind::kMoon: {
      const bool is_sun = ref.kind == PointingTargetKind::kSun;
      if (is_sun ? !ctx.sun_valid : !ctx.moon_valid) {
        return GuidanceStatus::kNoEphemeris;
      }
      const Eigen::Vector3d body_r = (is_sun ? ctx.sun_position_m : ctx.moon_position_m).eigen();
      const bool have_v = is_sun ? ctx.sun_velocity_valid : ctx.moon_velocity_valid;
      const Eigen::Vector3d body_v =
          have_v ? (is_sun ? ctx.sun_velocity_m_s : ctx.moon_velocity_m_s).eigen()
                 : Eigen::Vector3d::Zero();
      if (!lineOfSight(obs_r, obs_v, body_r, body_v, d)) {
        return GuidanceStatus::kTargetUnavailable;
      }
      // Without the body's own velocity only the parallax term is modelled, and
      // that is reported rather than hidden — see GuidanceContext.
      d.rate_known = have_v;
      break;
    }

    case PointingTargetKind::kNadir: {
      const double rn = obs_r.norm();
      if (!(rn > 0.0)) {
        return GuidanceStatus::kNoOrbitState;
      }
      d.unit = math::Vec3<math::frames::ECI>(-obs_r / rn);
      d.rate_rad_s = math::Vec3<math::frames::ECI>(orbitRate(obs_r, obs_v));
      d.rate_known = true;
      break;
    }

    case PointingTargetKind::kEcefPoint: {
      if (!ctx.earth_orientation_valid) {
        // The one target kind that needs EOP. Everything else in this file keeps
        // working through an outage, so this is refused specifically rather than
        // taking the whole guidance mode down with it.
        return GuidanceStatus::kNoEarthOrientation;
      }
      if (ctx.ground_points == nullptr) {
        return GuidanceStatus::kTargetUnavailable;
      }
      GroundPointTable::Point gp;
      if (!ctx.ground_points->get(static_cast<int>(ref.index), gp)) {
        return GuidanceStatus::kTargetSlotEmpty;
      }
      const Eigen::Vector3d p_ecef =
          ecefFromGeodetic(gp.latitude_rad, gp.longitude_rad, gp.height_m);
      const Eigen::Vector3d p_eci = ctx.eci_from_ecef * p_ecef;
      // The point is fixed in ECEF, so in ECI it moves with the Earth's rotation
      // about the pole. Omitting this term would leave a station-track lagging by
      // the Earth rate, which at the surface is 465 m/s.
      const Eigen::Vector3d earth_omega(0.0, 0.0, c::wgs84::kEarthRate);
      const Eigen::Vector3d p_eci_dot = earth_omega.cross(p_eci);
      if (!lineOfSight(obs_r, obs_v, p_eci, p_eci_dot, d)) {
        return GuidanceStatus::kTargetUnavailable;
      }
      d.rate_known = true;
      break;
    }

    case PointingTargetKind::kStarJ2000: {
      const double ra = ref.params[0];
      const double dec = ref.params[1];
      if (!std::isfinite(ra) || !std::isfinite(dec) || std::fabs(dec) > M_PI / 2.0 + 1e-12) {
        return GuidanceStatus::kBadParameters;
      }
      d.unit = math::Vec3<math::frames::ECI>(std::cos(dec) * std::cos(ra),
                                             std::cos(dec) * std::sin(ra), std::sin(dec));
      // A star is inertially fixed at this precision: proper motion is
      // milliarcseconds per year, far below anything a control loop resolves.
      d.rate_known = true;
      break;
    }

    case PointingTargetKind::kJ2000X:
      d.unit = math::Vec3<math::frames::ECI>(1.0, 0.0, 0.0);
      d.rate_known = true;
      break;
    case PointingTargetKind::kJ2000Y:
      d.unit = math::Vec3<math::frames::ECI>(0.0, 1.0, 0.0);
      d.rate_known = true;
      break;
    case PointingTargetKind::kJ2000Z:
      d.unit = math::Vec3<math::frames::ECI>(0.0, 0.0, 1.0);
      d.rate_known = true;
      break;

    case PointingTargetKind::kLvlhX:
    case PointingTargetKind::kLvlhY:
    case PointingTargetKind::kLvlhZ: {
      // Built from the repository's one LVLH definition (z = -r̂ nadir,
      // y = -ĥ, x = y×z ≈ +velocity) rather than re-derived here, so a pointing
      // command and a covariance display cannot disagree about which way LVLH x
      // points.
      math::Quat<math::frames::LVLH, math::frames::ECI> q_lvlh;
      if (!math::lvlhFromEci(ctx.observer_position_m, ctx.observer_velocity_m_s, q_lvlh)) {
        return GuidanceStatus::kNoOrbitState;
      }
      const int axis = static_cast<int>(ref.kind) - static_cast<int>(PointingTargetKind::kLvlhX);
      math::Vec3<math::frames::LVLH> unit_lvlh(axis == 0 ? 1.0 : 0.0, axis == 1 ? 1.0 : 0.0,
                                               axis == 2 ? 1.0 : 0.0);
      d.unit = q_lvlh.inverse().rotate(unit_lvlh);
      d.rate_rad_s = math::Vec3<math::frames::ECI>(orbitRate(obs_r, obs_v));
      d.rate_known = true;
      break;
    }

    case PointingTargetKind::kSatTle:
    case PointingTargetKind::kSatState: {
      if (ctx.catalog == nullptr) {
        return GuidanceStatus::kTargetUnavailable;
      }
      const TargetKind kind =
          ref.kind == PointingTargetKind::kSatTle ? TargetKind::kTle : TargetKind::kStateVector;
      TargetState st;
      const TargetStatus s = ctx.catalog->positionAt(kind, static_cast<int>(ref.index), ctx.t, st);
      if (s == TargetStatus::kEmpty) {
        return GuidanceStatus::kTargetSlotEmpty;
      }
      if (s != TargetStatus::kOk) {
        return GuidanceStatus::kTargetUnavailable;
      }
      if (!lineOfSight(obs_r, obs_v, st.position_m.eigen(), st.velocity_m_s.eigen(), d)) {
        return GuidanceStatus::kTargetUnavailable;
      }
      d.rate_known = true;
      break;
    }

    default:
      return GuidanceStatus::kBadParameters;
  }

  if (!d.unit.isFinite() || !(d.unit.norm() > 0.0)) {
    return GuidanceStatus::kTargetUnavailable;
  }
  applySign(ref.negate, d);
  out = d;
  return GuidanceStatus::kOk;
}

GuidanceStatus validateGuidanceCommand(const GuidanceCommand& cmd, const BodyVectorTable& table,
                                       const TargetCatalog* catalog,
                                       const GroundPointTable* ground_points) {
  // 1. The two body vectors must not be the same axis. Checked on the axis
  //    rather than the full signed direction: aligning +X while constraining -X
  //    is exactly as unsatisfiable as naming +X twice, and a sign-aware
  //    comparison would let it through.
  if (cmd.align_vector.sameAxisAs(cmd.constrain_vector)) {
    return GuidanceStatus::kSameBodyAxis;
  }

  // 2. Both must resolve — an uninstalled star tracker or an unwritten custom
  //    slot is an operator error to report now, not a degenerate axis to
  //    discover mid-slew.
  math::Vec3<math::frames::Body> a_body;
  math::Vec3<math::frames::Body> c_body;
  if (!table.resolve(cmd.align_vector, a_body) || !table.resolve(cmd.constrain_vector, c_body)) {
    return GuidanceStatus::kBodyVectorUnknown;
  }

  // 3. Two distinct names can still be one direction — two cameras boresighted
  //    together, or a custom vector written parallel to a structural axis. Only
  //    the resolved geometry can see that.
  if (a_body.eigen().cross(c_body.eigen()).norm() < kAxisParallelSin) {
    return GuidanceStatus::kBodyAxesParallel;
  }

  // 4. Parameterised targets must carry sane numbers, and named slots must be
  //    occupied. Both are properties of the command, so both belong here rather
  //    than in the per-cycle solve.
  const PointingTargetRef* refs[2] = {&cmd.align_target, &cmd.constrain_target};
  for (const PointingTargetRef* ref : refs) {
    if (ref->usesParams()) {
      for (double p : ref->params) {
        if (!std::isfinite(p)) {
          return GuidanceStatus::kBadParameters;
        }
      }
      // Declination outside +/-90 deg is not a direction, it is a typo.
      if (std::fabs(ref->params[1]) > M_PI / 2.0 + 1e-12) {
        return GuidanceStatus::kBadParameters;
      }
    }
    if (ref->usesIndex()) {
      const int i = static_cast<int>(ref->index);
      if (i < 0 || i >= ref->indexBound()) {
        return GuidanceStatus::kBadParameters;
      }
      if (ref->kind == PointingTargetKind::kEcefPoint) {
        // An empty ground-station slot is an operator error worth reporting at
        // command time; discovering it mid-pass would waste the pass.
        if (ground_points == nullptr || !ground_points->isSet(i)) {
          return GuidanceStatus::kTargetSlotEmpty;
        }
      } else {
        const TargetKind k =
            ref->kind == PointingTargetKind::kSatTle ? TargetKind::kTle : TargetKind::kStateVector;
        if (catalog == nullptr || !catalog->isOccupied(k, i)) {
          return GuidanceStatus::kTargetSlotEmpty;
        }
      }
    }
  }

  // Deliberately NOT checked here: whether the two inertial directions are
  // collinear. That is a property of the sky at a moment, not of the command —
  // see the header. It is re-checked every cycle in solveGuidanceAttitude.
  return GuidanceStatus::kOk;
}

GuidanceStatus alignConstrainQuaternion(const math::Vec3<math::frames::Body>& align_body,
                                        const math::Vec3<math::frames::ECI>& align_eci,
                                        const math::Vec3<math::frames::Body>& constrain_body,
                                        const math::Vec3<math::frames::ECI>& constrain_eci,
                                        math::Quat<math::frames::Body, math::frames::ECI>& out) {
  if (!align_body.isFinite() || !align_eci.isFinite() || !constrain_body.isFinite() ||
      !constrain_eci.isFinite()) {
    return GuidanceStatus::kBadInput;
  }
  const Eigen::Vector3d b1 = align_body.eigen();
  const Eigen::Vector3d b2 = constrain_body.eigen();
  const Eigen::Vector3d i1 = align_eci.eigen();
  const Eigen::Vector3d i2 = constrain_eci.eigen();
  if (!(b1.norm() > 0.0) || !(b2.norm() > 0.0) || !(i1.norm() > 0.0) || !(i2.norm() > 0.0)) {
    return GuidanceStatus::kBadInput;
  }
  const Eigen::Vector3d b1u = b1.normalized();
  const Eigen::Vector3d b2u = b2.normalized();
  if (b1u.cross(b2u).norm() < kAxisParallelSin) {
    return GuidanceStatus::kBodyAxesParallel;
  }
  const Eigen::Vector3d i1u = i1.normalized();
  const Eigen::Vector3d i2u = i2.normalized();

  // The component of the constraint perpendicular to the aligned direction is
  // the only part that carries roll information. When it vanishes the constraint
  // is silent and every roll satisfies it equally — refused rather than resolved
  // by an arbitrary pick, because an arbitrary roll is stable and plausible and
  // therefore invisible.
  const Eigen::Vector3d perp = i2u - i1u * i1u.dot(i2u);
  if (perp.norm() < kCollinearSin) {
    return GuidanceStatus::kDirectionsCollinear;
  }
  const Eigen::Vector3d e2 = perp.normalized();
  const Eigen::Vector3d e3 = i1u.cross(e2);

  // Two orthonormal triads built by the *same* Gram-Schmidt, so the residual
  // non-orthogonality of the inputs is absorbed identically on both sides and
  // the composed rotation is orthonormal by construction rather than by
  // renormalising a nearly-orthonormal product afterwards.
  const Eigen::Vector3d c2 = (b2u - b1u * b1u.dot(b2u)).normalized();
  const Eigen::Vector3d c3 = b1u.cross(c2);

  Eigen::Matrix3d body_axes;
  body_axes.col(0) = b1u;
  body_axes.col(1) = c2;
  body_axes.col(2) = c3;
  Eigen::Matrix3d eci_axes;
  eci_axes.col(0) = i1u;
  eci_axes.col(1) = e2;
  eci_axes.col(2) = e3;

  const Eigen::Matrix3d dcm = body_axes * eci_axes.transpose();
  if (!dcm.allFinite()) {
    return GuidanceStatus::kBadInput;
  }
  out = math::Quat<math::frames::Body, math::frames::ECI>(math::Quaternion::FromRotationMatrix(dcm))
            .canonical();
  return GuidanceStatus::kOk;
}

GuidanceStatus solveGuidanceAttitude(
    const GuidanceCommand& cmd, const BodyVectorTable& table, const GuidanceContext& ctx,
    math::Quat<math::frames::Body, math::frames::ECI>& q_body_from_eci,
    math::Vec3<math::frames::Body>& rate_body_rad_s) {
  math::Vec3<math::frames::Body> a_body;
  math::Vec3<math::frames::Body> c_body;
  if (!table.resolve(cmd.align_vector, a_body) || !table.resolve(cmd.constrain_vector, c_body)) {
    return GuidanceStatus::kBodyVectorUnknown;
  }

  ResolvedDirection a_dir;
  GuidanceStatus s = resolveTarget(cmd.align_target, ctx, a_dir);
  if (s != GuidanceStatus::kOk) {
    return s;
  }
  ResolvedDirection c_dir;
  s = resolveTarget(cmd.constrain_target, ctx, c_dir);
  if (s != GuidanceStatus::kOk) {
    return s;
  }

  math::Quat<math::frames::Body, math::frames::ECI> q;
  s = alignConstrainQuaternion(a_body, a_dir.unit, c_body, c_dir.unit, q);
  if (s != GuidanceStatus::kOk) {
    return s;
  }

  // ---- The feedforward rate, differentiated analytically.
  //
  // The commanded frame is an orthonormal triad built from the two directions,
  // so its inertial angular velocity is omega = 1/2 sum_i (e_i x e_i_dot). Each
  // direction's own rate is known from its resolution, and the Gram-Schmidt is
  // differentiated in closed form. Doing it this way rather than by differencing
  // successive commanded quaternions matters: differencing amplifies a target's
  // position noise by 1/dt, and a target propagated from a day-old TLE has
  // plenty of it.
  const Eigen::Vector3d u1 = a_dir.unit.eigen();
  const Eigen::Vector3d u2 = c_dir.unit.eigen();
  const Eigen::Vector3d u1d = a_dir.rate_rad_s.eigen().cross(u1);
  const Eigen::Vector3d u2d = c_dir.rate_rad_s.eigen().cross(u2);

  const Eigen::Vector3d w = u2 - u1 * u1.dot(u2);
  const double wn = w.norm();
  if (!(wn > 0.0)) {
    return GuidanceStatus::kDirectionsCollinear;
  }
  const Eigen::Vector3d e1 = u1;
  const Eigen::Vector3d e2 = w / wn;
  const Eigen::Vector3d e3 = e1.cross(e2);

  const Eigen::Vector3d wd = u2d - u1d * u1.dot(u2) - u1 * (u1d.dot(u2) + u1.dot(u2d));
  const Eigen::Vector3d e1d = u1d;
  const Eigen::Vector3d e2d = (wd - e2 * e2.dot(wd)) / wn;
  const Eigen::Vector3d e3d = e1d.cross(e2) + e1.cross(e2d);

  const Eigen::Vector3d omega_eci = 0.5 * (e1.cross(e1d) + e2.cross(e2d) + e3.cross(e3d));
  if (!omega_eci.allFinite()) {
    return GuidanceStatus::kBadInput;
  }

  q_body_from_eci = q;
  rate_body_rad_s = q.rotate(math::Vec3<math::frames::ECI>(omega_eci));
  return GuidanceStatus::kOk;
}

}  // namespace polaris::gnc
