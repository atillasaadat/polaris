#ifndef POLARIS_GNC_TARGET_POINTING_HPP
#define POLARIS_GNC_TARGET_POINTING_HPP

/// @file
/// @brief Boresight-to-target attitude guidance (design doc §8.4; REQ-AGN-004).
///
/// Turns "point the camera at that object" into the quaternion and body rate the
/// §8.5 controller already knows how to track. This is the piece that was
/// missing: `AttitudeController` has only ever accepted `CTRL_SET_TARGET_Q`, a
/// **fixed** inertial quaternion, which cannot express a target that moves.
///
/// ## Pointing is two degrees of freedom, not three
///
/// Aligning a boresight with a direction leaves the **roll about that boresight
/// free**, and an attitude command must fix all three. Every pointing mode
/// therefore needs a second, weaker constraint, and the choice is a mission
/// decision rather than a mathematical one — it sets where the solar arrays and
/// the radiator end up while the instrument is on target.
///
/// So the caller supplies both: a body axis to align with the target
/// (`boresight_body`) and a second body axis to bring **as close as possible**
/// to a reference direction (`secondary_body` toward `secondary_ref_eci`). The
/// first is satisfied exactly; the second is satisfied in the plane
/// perpendicular to it. Naming them explicitly is deliberate — a hidden default
/// would make the array and radiator geometry an accident of the implementation.
///
/// **The degenerate case is refused, not fudged.** When the reference direction
/// is parallel to the target direction, the secondary constraint says nothing
/// about roll and any answer is as good as any other. Silently picking one
/// produces an attitude that is stable, plausible, and unrelated to what the
/// operator asked for — so `TargetPointingStatus::kSecondaryDegenerate` is
/// returned instead and the caller decides.
///
/// ## Rate feedforward
///
/// A tracking command is not a sequence of hold commands. The line of sight to a
/// close, fast target sweeps quickly — a co-altitude object passing at a few km
/// has a line-of-sight rate of degrees per second — and a controller given only
/// a position error lags it by roughly (rate / bandwidth). @ref
/// targetPointingRate supplies the feedforward body rate that removes that lag,
/// computed from the **relative** state, which is what makes this tracking
/// rather than repeated repointing.
///
/// The rate is differentiated analytically from the relative position and
/// velocity rather than by differencing successive quaternions: differencing
/// amplifies the target's own position noise by 1/dt, and a target propagated
/// from a day-old TLE has plenty of it.
///
/// Flight-safe (§3.6): no heap, no exceptions, fixed-size Eigen, no recursion.

#include "math/frames.hpp"
#include "math/quaternion.hpp"
#include "math/typed_vector.hpp"

namespace polaris::gnc {

/// Why a pointing solution could not be formed.
enum class TargetPointingStatus : unsigned char {
  kOk = 0,
  kBadInput,             ///< a non-finite input
  kCoincident,           ///< the target is at the observer: no line of sight exists
  kBadAxes,              ///< boresight and secondary body axes are null or parallel
  kSecondaryDegenerate,  ///< the reference direction is parallel to the line of sight
};

const char* toString(TargetPointingStatus status);

/// The attitude that puts @p boresight_body on the target, with @p
/// secondary_body brought as close as possible to @p secondary_ref_eci.
///
/// @param observer_eci_m   this vehicle's ECI position [m]
/// @param target_eci_m     the target's ECI position [m]
/// @param boresight_body   instrument boresight in body axes (need not be unit)
/// @param secondary_body   the body axis whose placement resolves roll
/// @param secondary_ref_eci  the inertial direction to bring it toward
/// @param out              Body <- ECI, canonical (q0 >= 0)
TargetPointingStatus targetPointingQuaternion(
    const math::Vec3<math::frames::ECI>& observer_eci_m,
    const math::Vec3<math::frames::ECI>& target_eci_m,
    const math::Vec3<math::frames::Body>& boresight_body,
    const math::Vec3<math::frames::Body>& secondary_body,
    const math::Vec3<math::frames::ECI>& secondary_ref_eci,
    math::Quat<math::frames::Body, math::frames::ECI>& out);

/// The inertial angular rate of the line of sight [rad/s], in ECI.
///
/// The rotation rate of the unit line-of-sight vector: `(r x v) / |r|^2` on the
/// **relative** state. This is the feedforward that turns repeated repointing
/// into tracking; without it a controller lags a moving target by roughly the
/// line-of-sight rate divided by its bandwidth.
///
/// Only the component perpendicular to the line of sight is physical — motion
/// *along* the line of sight changes range, not direction — and this returns
/// exactly that, which is why the roll rate about the boresight is not included
/// and must come from the secondary constraint if it is needed.
TargetPointingStatus targetPointingRate(const math::Vec3<math::frames::ECI>& observer_eci_m,
                                        const math::Vec3<math::frames::ECI>& observer_vel_m_s,
                                        const math::Vec3<math::frames::ECI>& target_eci_m,
                                        const math::Vec3<math::frames::ECI>& target_vel_m_s,
                                        math::Vec3<math::frames::ECI>& rate_eci_rad_s);

}  // namespace polaris::gnc

#endif  // POLARIS_GNC_TARGET_POINTING_HPP
