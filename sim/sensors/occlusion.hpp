#ifndef POLARIS_SIM_SENSORS_OCCLUSION_HPP
#define POLARIS_SIM_SENSORS_OCCLUSION_HPP

/// @file
/// @brief Shared line-of-sight occlusion / bright-body keep-out model for
/// optical and celestial sensors (design doc §6.1).
///
/// Every optical sensor — star tracker, sun sensor, and later any camera or
/// horizon sensor — is blinded by the same three things: the Earth filling part
/// of its field of view, the Sun in or near its boresight, and the Moon doing
/// the same more weakly. Modelling that once and sharing it is what keeps a star
/// tracker and a sun sensor from disagreeing about whether the Earth is in the
/// way, which is exactly the kind of inconsistency that produces an estimator
/// that works in simulation and not in flight.
///
/// The geometry is deliberately the same as the §5.2 conical eclipse model: each
/// body is a sphere with an apparent angular radius seen from the spacecraft,
///
///   ρ = asin(R_body / d)          d = distance to the body's centre,
///
/// and the constraint is violated when the angle between the boresight and the
/// body's centre falls inside `ρ + keep_out`. Using the *limb* rather than the
/// centre matters at LEO, where the Earth's apparent radius is ~70° — treating
/// it as a point would declare a nadir-pointing tracker perfectly happy.
///
/// The Earth is a sphere of the WGS84 equatorial radius (as in §5.2); oblateness
/// and atmospheric refraction perturb only the last fraction of a degree at the
/// limb, well inside the margin any real keep-out angle carries. Stray light
/// scattered off the sunlit limb, baffle rejection curves, and albedo-driven
/// degradation are **not** modelled — the model answers "is the line of sight
/// blocked or inside a keep-out cone", not "how much stray light is there".
///
/// References:
///  - Markley & Crassidis, *Fundamentals of Spacecraft Attitude Determination
///    and Control*, 2014, §4.2 (star tracker field-of-view constraints).
///    [markley2014]
///  - Vallado, *Fundamentals of Astrodynamics and Applications*, 4th ed., §5.3
///    (apparent-disk geometry). [vallado2013]

#include <Eigen/Core>

namespace polaris::sim::sensors {

/// Which body, if any, spoils a line of sight. Reported rather than reduced to a
/// bool so telemetry and the FDIR suite can distinguish "Earth in the FOV" from
/// "Sun in the keep-out cone" — they have different operational responses.
enum class Occluder {
  kNone,   ///< line of sight is clear
  kEarth,  ///< Earth disk (limb) intrudes
  kSun,    ///< Sun inside its keep-out cone
  kMoon,   ///< Moon inside its keep-out cone
};

/// Per-sensor keep-out angles [rad], measured from each body's **limb**, not its
/// centre. Zero disables that constraint; a sensor pays only for what it
/// configures.
struct KeepOutSpec {
  double earth_rad = 0.0;  ///< required clearance from the Earth limb
  double sun_rad = 0.0;    ///< required clearance from the solar limb
  double moon_rad = 0.0;   ///< required clearance from the lunar limb
};

/// Geometry a line-of-sight check needs, all in ECI [m].
struct SkyGeometry {
  Eigen::Vector3d sat = Eigen::Vector3d::Zero();   ///< geocentric spacecraft position
  Eigen::Vector3d sun = Eigen::Vector3d::Zero();   ///< geocentric Sun position
  Eigen::Vector3d moon = Eigen::Vector3d::Zero();  ///< geocentric Moon position
};

/// Angular clearance [rad] between @p boresight_eci and a body of radius
/// @p body_radius_m whose centre is at @p to_body (vector from the spacecraft).
///
/// Positive is clear sky between the boresight and the body's limb; negative
/// means the boresight is inside the disk. Returns +∞ for a degenerate input
/// (zero-length boresight or a body at the spacecraft), i.e. "unconstrained" —
/// a geometry that cannot be evaluated must not silently blind a sensor.
double limbClearance(const Eigen::Vector3d& boresight_eci, const Eigen::Vector3d& to_body,
                     double body_radius_m);

/// Check one line of sight against every configured keep-out.
///
/// Earth is checked first: at LEO it is the dominant constraint, and reporting
/// it in preference to a Sun cone that happens to overlap gives the more useful
/// diagnosis.
///
/// @param boresight_eci Sensor boresight direction in ECI (need not be unit).
/// @param sky           Spacecraft, Sun, and Moon positions (ECI).
/// @param keep_out      Per-body clearance requirements.
Occluder checkLineOfSight(const Eigen::Vector3d& boresight_eci, const SkyGeometry& sky,
                          const KeepOutSpec& keep_out);

}  // namespace polaris::sim::sensors

#endif  // POLARIS_SIM_SENSORS_OCCLUSION_HPP
