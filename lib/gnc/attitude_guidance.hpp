#ifndef POLARIS_GNC_ATTITUDE_GUIDANCE_HPP
#define POLARIS_GNC_ATTITUDE_GUIDANCE_HPP

/// @file
/// @brief Align/constrain attitude guidance (design doc §8.4; REQ-AGN-004,
/// REQ-AGN-005).
///
/// One command shape for every pointing mode the vehicle has:
///
///     ALIGN     <body vector>  with   <inertial target>   — exact, 2 DOF
///     CONSTRAIN <body vector>  toward <inertial target>   — best effort, 1 DOF
///
/// Nadir hold is `ALIGN -Z with NADIR, CONSTRAIN +X toward LVLH_X`. Sun-safe is
/// `ALIGN array-normal with SUN, CONSTRAIN radiator anti-SUN`. Ground-station
/// track is `ALIGN antenna with ECEF_TARGET`. Satellite track is `ALIGN camera
/// with SAT_TLE_2`. None of those is a mode in the code — they are four commands
/// built from `pointing_refs.hpp`'s two noun lists, and they all go through the
/// validation and the solve in this file.
///
/// That is the whole design argument. A vehicle that implements each mode
/// separately accumulates one geometry bug per mode and one validation gap per
/// mode; this way there is one of each to get right, and a new mode is an
/// operator command rather than a software change.
///
/// ## What is validated, and when
///
/// Two checks with genuinely different timing, and conflating them is the trap:
///
///  - **Static** (@ref validateGuidanceCommand) — properties of the *command*:
///    the two body vectors are not the same axis, both resolve, slots named are
///    occupied, parameters are in range. These are true or false the moment the
///    command arrives, so they are checked once, at command time, and the
///    command is rejected with a reason the operator can act on.
///  - **Geometric** (@ref solveGuidanceAttitude) — properties of the *sky*: the
///    two inertial directions must not be collinear right now. This is
///    time-varying — a constraint that is fine at command time can degenerate an
///    orbit later as the Sun, the target and the vehicle line up — so it cannot
///    be settled at command time and must be re-checked every cycle.
///
/// A command that passes the static check can still fail geometrically later,
/// and that is not a defect: it is the mode telling the truth about a geometry
/// that has become unsatisfiable. It is reported, never resolved by silently
/// picking a roll.
///
/// Flight-safe (§3.6): no heap, no exceptions, fixed-size Eigen, no recursion.

#include "gnc/pointing_refs.hpp"
#include "gnc/target_catalog.hpp"
#include "math/frames.hpp"
#include "math/quaternion.hpp"
#include "math/typed_vector.hpp"
#include "time/timescales.hpp"

namespace polaris::gnc {

/// Why a guidance command was rejected or a cycle produced no attitude.
enum class GuidanceStatus : std::uint8_t {
  kOk = 0,
  kSameBodyAxis,         ///< align and constrain name the same body axis
  kBodyVectorUnknown,    ///< a named body vector is not installed or not stored
  kBodyAxesParallel,     ///< the two body vectors resolve to (nearly) one axis
  kTargetUnavailable,    ///< a named target has no position right now
  kTargetSlotEmpty,      ///< a catalogue slot named by the command is empty
  kBadParameters,        ///< a parameterised target's numbers are out of range
  kNoEphemeris,          ///< Sun/Moon needed but the ephemeris did not answer
  kNoEarthOrientation,   ///< an ECEF target was named with no usable EOP
  kNoOrbitState,         ///< nadir/LVLH/line-of-sight needs a vehicle state
  kDirectionsCollinear,  ///< align and constrain targets are collinear *now*
  kBadInput,             ///< a non-finite input reached the solver
};

const char* toString(GuidanceStatus status);

/// A complete pointing command: two body vectors and the two things they aim at.
struct GuidanceCommand {
  BodyVectorRef align_vector;
  PointingTargetRef align_target;
  BodyVectorRef constrain_vector;
  PointingTargetRef constrain_target;
};

/// Everything the resolvers need that is not in the command.
///
/// Passed as plain data rather than as callbacks: the flight standard forbids
/// `std::function`, and a struct of already-fetched values also makes every
/// resolution testable without an ephemeris, an EOP table or a topology.
struct GuidanceContext {
  time::Tai t;

  /// This vehicle's state. Required by nadir, LVLH and every line-of-sight
  /// target; `kNoOrbitState` is returned when one of those is named without it.
  math::Vec3<math::frames::ECI> observer_position_m;
  math::Vec3<math::frames::ECI> observer_velocity_m_s;
  bool observer_valid = false;

  /// Geocentric Sun and Moon positions [m], from `OnboardTables`.
  math::Vec3<math::frames::ECI> sun_position_m;
  bool sun_valid = false;
  math::Vec3<math::frames::ECI> moon_position_m;
  bool moon_valid = false;

  /// Third-body velocities [m/s], when the caller can supply them.
  ///
  /// `OnboardTables` serves positions only, so these are usually absent — and
  /// the flags exist so that absence is *stated* rather than encoded as a zero
  /// vector. Without them a Sun or Moon line-of-sight rate carries only the
  /// parallax term from this vehicle's own motion, and `ResolvedDirection` says
  /// so through `rate_known`. The omission is not negligible for fine pointing:
  /// the Sun's apparent motion is ~2e-7 rad/s, which is 0.04 deg over a
  /// three-minute observation, comparable to a tight pointing budget. A caller
  /// that needs it can finite-difference the ephemeris and fill these in.
  math::Vec3<math::frames::ECI> sun_velocity_m_s;
  bool sun_velocity_valid = false;
  math::Vec3<math::frames::ECI> moon_velocity_m_s;
  bool moon_velocity_valid = false;

  /// ECI <- ECEF rotation for `kEcefPoint`. Unlike everything else here this
  /// needs Earth-orientation data, which is why an ECEF target is the one kind
  /// that can be refused during a long EOP outage while the rest keep working.
  Eigen::Matrix3d eci_from_ecef = Eigen::Matrix3d::Identity();
  bool earth_orientation_valid = false;

  /// Propagated target slots. May be null when no command names one.
  const TargetCatalog* catalog = nullptr;

  /// Stored Earth-fixed points. May be null when no command names one.
  const GroundPointTable* ground_points = nullptr;
};

/// The unit direction a target reference names, and how fast it is turning.
struct ResolvedDirection {
  math::Vec3<math::frames::ECI> unit;
  /// Inertial angular rate of @ref unit [rad/s]. Zero for a fixed inertial
  /// direction; non-zero for anything tied to the orbit or to a moving body.
  math::Vec3<math::frames::ECI> rate_rad_s;
  /// True when @ref rate_rad_s was actually computed rather than assumed zero.
  /// A caller feeding a rate feedforward needs to know the difference between
  /// "this direction is stationary" and "nobody worked out how fast it moves".
  bool rate_known = false;
};

/// Resolve one target reference to a direction at `ctx.t`.
GuidanceStatus resolveTarget(const PointingTargetRef& ref, const GuidanceContext& ctx,
                             ResolvedDirection& out);

/// Static validation of a command, at the moment it arrives.
///
/// Everything checkable without knowing where anything is: see the header on why
/// collinearity of the *targets* is deliberately not among it.
GuidanceStatus validateGuidanceCommand(const GuidanceCommand& cmd, const BodyVectorTable& table,
                                       const TargetCatalog* catalog,
                                       const GroundPointTable* ground_points);

/// The attitude and body rate that satisfy @p cmd at `ctx.t`.
///
/// @param q_body_from_eci  the commanded attitude, canonical (q0 >= 0)
/// @param rate_body_rad_s  feedforward body rate; zero when no target moves
GuidanceStatus solveGuidanceAttitude(
    const GuidanceCommand& cmd, const BodyVectorTable& table, const GuidanceContext& ctx,
    math::Quat<math::frames::Body, math::frames::ECI>& q_body_from_eci,
    math::Vec3<math::frames::Body>& rate_body_rad_s);

/// The pure geometry, exposed for testing and for callers that have already
/// resolved their directions: the attitude putting @p align_body on @p align_eci
/// exactly and @p constrain_body as close as possible to @p constrain_eci.
GuidanceStatus alignConstrainQuaternion(const math::Vec3<math::frames::Body>& align_body,
                                        const math::Vec3<math::frames::ECI>& align_eci,
                                        const math::Vec3<math::frames::Body>& constrain_body,
                                        const math::Vec3<math::frames::ECI>& constrain_eci,
                                        math::Quat<math::frames::Body, math::frames::ECI>& out);

}  // namespace polaris::gnc

#endif  // POLARIS_GNC_ATTITUDE_GUIDANCE_HPP
