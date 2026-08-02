#ifndef POLARIS_SIM_SENSORS_OCCLUSION_HPP
#define POLARIS_SIM_SENSORS_OCCLUSION_HPP

/// @file
/// @brief Shared line-of-sight occlusion / bright-body keep-out model for
/// optical and celestial sensors (design doc §6.1).
///
/// Every optical sensor — star tracker, sun sensor, and later any camera or
/// horizon sensor — is spoiled by the same three things: the Earth filling part
/// of its field of view, the Sun in or near its boresight, and the Moon doing
/// the same more weakly. Modelling that once and sharing it is what keeps a star
/// tracker and a sun sensor from disagreeing about whether the Earth is in the
/// way, which is exactly the kind of inconsistency that produces an estimator
/// that works in simulation and not in flight.
///
/// The model answers three different questions, because sensors need all of them:
///
///  - **Is a keep-out violated?** A hard constraint — the unit's baffle rejection
///    and the vendor's exclusion angles. Drives a validity flag.
///  - **What fraction of the field of view does each body cover?** A continuous
///    quantity, needed for graded degradation (a tracker losing part of its star
///    field, a sun sensor's albedo contribution) rather than a cliff at the
///    keep-out edge. Also what makes an outage's *onset* smooth, which matters
///    for anything that differentiates a measurement stream.
///  - **Where is the boresight pointing, relative to the Sun and to nadir?** Two
///    plain angles, reported for every line of sight. They are neither of the
///    above: outside the keep-out and outside the field of view both other
///    answers saturate, while an observation plan, a thermal check, or a payload
///    duty-cycle rule needs the actual separation. Computed here rather than by
///    each sensor for the same reason the rest is — one geometry, one answer
///    (§6.1), so a star tracker and a payload imager cannot disagree about where
///    the Sun is.
///
/// **The atmosphere is part of the Earth here.** A line of sight grazing the
/// limb passes through airglow, scattered light, and refraction long before it
/// touches the solid surface, so the optically obstructing body is a sphere of
/// `R_earth + atmosphere_height_m` (100 km by default — the Kármán line, which is
/// where the airglow layer and the bulk of limb radiance sit). Both fractions are
/// reported: the **solid** Earth disk and the **Earth + atmosphere** disk. Which
/// one a sensor cares about depends on the sensor — a horizon sensor tracks the
/// atmospheric limb deliberately, while a star tracker is blinded by it — so the
/// model reports both rather than choosing. Keep-out clearances are measured from
/// the **atmospheric** limb, the conservative choice.
///
/// Geometry: each body is a sphere with an apparent angular radius seen from the
/// spacecraft, ρ = asin(R_body / d), the same construction as the §5.2 conical
/// eclipse model. Using the *limb* rather than the centre matters at LEO, where
/// the Earth's apparent radius is ~70° — treating it as a point would declare a
/// nadir-pointing tracker perfectly happy.
///
/// **Approximation in the fraction.** The overlap of the FOV cone with a body's
/// cone is computed with the planar two-circle lens formula applied to the
/// angular radii, the same approximation §5.2 uses for the overlapping Sun and
/// Earth disks. It is exact in the small-angle limit and degrades as the field of
/// view grows. Measured against a brute-force spherical quadrature
/// (`tests/unit/sim_sensors_occlusion_test.cpp`, which pins these per regime):
/// better than **0.008** in fraction for half-fields up to 10°, **0.01** at 15°,
/// and **0.02** at 30° — even against the Earth's ~68° disk, the demanding case.
/// (The residual is real curvature, not noise: with the boresight exactly on the
/// Earth's limb the planar formula gives 0.5 by symmetry, where the sphere gives
/// 0.494.)
/// That covers every star tracker and most sun sensors. A sensor with a much
/// wider field (a fisheye coarse sun sensor, a horizon sensor spanning the whole
/// limb) needs the exact spherical-cap intersection instead.
/// Stray-light falloff, baffle rejection curves, and refraction are **not**
/// modelled: this answers "how much of the view is blocked", not "how much stray
/// light reaches the detector".
///
/// References:
///  - Markley & Crassidis, *Fundamentals of Spacecraft Attitude Determination
///    and Control*, 2014, §4.2 (star tracker field-of-view constraints).
///    [markley2014]
///  - Vallado, *Fundamentals of Astrodynamics and Applications*, 4th ed., §5.3
///    (apparent-disk geometry). [vallado2013]
///
/// Implements the shared line-of-sight occlusion model of REQ-SIM-003.

#include <Eigen/Core>

#include "math/fov_overlap.hpp"

namespace polaris::sim::sensors {

/// Which body, if any, violates a keep-out. Reported rather than reduced to a
/// bool so telemetry and the FDIR suite can distinguish "Earth in the FOV" from
/// "Sun in the keep-out cone" — they have different operational responses.
enum class Occluder {
  kNone,   ///< every configured keep-out is satisfied
  kEarth,  ///< Earth (including its atmosphere) intrudes
  kSun,    ///< Sun inside its keep-out cone
  kMoon,   ///< Moon inside its keep-out cone
};

/// Per-sensor keep-out angles [rad], measured from each body's **limb**, not its
/// centre. Zero disables that constraint; a sensor pays only for what it
/// configures.
struct KeepOutSpec {
  double earth_rad = 0.0;  ///< required clearance from the atmospheric limb
  double sun_rad = 0.0;    ///< required clearance from the solar limb
  double moon_rad = 0.0;   ///< required clearance from the lunar limb
};

/// Default optical atmosphere thickness [m] — the Kármán line. Overridable per
/// scenario (`environment.occultation_atmosphere_km`, design doc §19.1): 100 km
/// is right for visible-band blinding, but a horizon sensor working in the 15 µm
/// CO2 band sees a limb tens of kilometres higher.
inline constexpr double kDefaultAtmosphereHeight = 100.0e3;

/// Geometry a line-of-sight check needs. Positions in ECI [m].
struct SkyGeometry {
  Eigen::Vector3d sat = Eigen::Vector3d::Zero();   ///< geocentric spacecraft position
  Eigen::Vector3d sun = Eigen::Vector3d::Zero();   ///< geocentric Sun position
  Eigen::Vector3d moon = Eigen::Vector3d::Zero();  ///< geocentric Moon position
  /// Optically obstructing atmosphere thickness above the Earth's surface [m].
  double atmosphere_height_m = kDefaultAtmosphereHeight;
};

/// What a line of sight sees. Fractions are of the sensor's **field of view**,
/// in [0, 1] — 1.0 means the body fills it completely.
struct OcclusionState {
  Occluder occluder = Occluder::kNone;
  /// Fraction of the FOV covered by the **solid** Earth disk.
  double earth_fraction = 0.0;
  /// Fraction covered by the Earth **plus its atmosphere** — always ≥
  /// `earth_fraction`. The difference is the airglow annulus, which is what
  /// spoils a star field before the hard limb ever appears.
  double earth_atmosphere_fraction = 0.0;
  double sun_fraction = 0.0;
  double moon_fraction = 0.0;

  /// Angle [rad] from the boresight to the **Sun centre**. Not a keep-out
  /// verdict and not a fraction: the continuous pointing quantity an operator
  /// plans against ("how close does this observation come to the Sun?"), which
  /// neither of the other two answers — a boresight 3° outside a 35° exclusion
  /// cone and one 90° away both report `kNone` and a zero fraction.
  double sun_angle_rad = M_PI;
  /// Angle [rad] from the boresight to **nadir** (the Earth centre, not the
  /// limb). The companion pointing quantity: 0 is straight down, π is zenith.
  double nadir_angle_rad = M_PI;
  // Both default to π — the "pointed as far away as possible" value — and stay
  // there unless the geometry they need was actually supplied: the nadir angle
  // needs a spacecraft position, the Sun angle needs a Sun position. That is
  // `limbClearance`'s +∞ convention (a geometry that cannot be evaluated must
  // not silently look like a pointing violation), and the Sun check is on
  // `sky.sun` specifically — with it left at the origin the vector to it is a
  // perfectly finite `-sat`, which would report a Sun angle equal to the nadir
  // angle: a fabricated pointing quantity that reads exactly like a real one.

  /// Fraction of the FOV covered by any body — the union, approximated as the
  /// largest single contributor,
  /// \f$\max(\text{earth\_atmosphere\_fraction},\ \text{sun\_fraction},\ \text{moon\_fraction})\f$.
  /// Bodies overlapping each other in one FOV is a geometry no real sensor is
  /// operated in (it means staring at an eclipse), so the approximation costs
  /// nothing real and keeps the value monotone.
  double blockedFraction() const;
};

/// Angular clearance [rad] between @p boresight_eci and a body of radius
/// @p body_radius_m whose centre is at @p to_body (vector from the spacecraft).
///
/// With separation \f$\psi\f$ (boresight to body centre) and distance
/// \f$d = \|\text{to\_body}\|\f$, the clearance to the body's limb is
/// \f[
///   c = \psi - \arcsin\!\big(R / d\big),
/// \f]
/// \f$R\f$ = @p body_radius_m. Positive is clear sky between the boresight and the
/// limb; negative means the boresight is inside the disk. Returns \f$-\infty\f$
/// when the spacecraft is inside the body (\f$d \le R\f$) and \f$+\infty\f$ for a
/// degenerate input (zero-length boresight or a body at the spacecraft), i.e.
/// "unconstrained" — a geometry that cannot be evaluated must not silently blind
/// a sensor.
double limbClearance(const Eigen::Vector3d& boresight_eci, const Eigen::Vector3d& to_body,
                     double body_radius_m);

/// Fraction of a circular field of view covered by a body's disk — the shared
/// implementation in `lib/math/fov_overlap.hpp`, re-exported here so the sim
/// spelling is unchanged.
///
/// It moved out of this file when the flight albedo correction (§8.1) needed the
/// same weighting: a correction computed from a *different* overlap formula than
/// the one the sensor error was generated with removes an error the sensor never
/// had. Same rule as the rest of this header — one geometry, one answer — now
/// applied across the sim/flight seam rather than only between optical sensors.
using polaris::math::fovCoveredFraction;

/// Evaluate one line of sight: keep-out violations, per-body FOV coverage, and
/// the boresight's Sun and nadir angles.
///
/// Earth is checked first for the keep-out verdict: at LEO it is the constraint
/// that usually decides, and reporting it in preference to a Sun cone that
/// happens to overlap gives the more useful diagnosis and a deterministic answer.
///
/// @param boresight_eci Sensor boresight direction in ECI (need not be unit).
/// @param half_fov_rad  Sensor half field of view [rad]; 0 leaves fractions at 0.
/// @param sky           Spacecraft, Sun, and Moon positions, plus the atmosphere.
/// @param keep_out      Per-body clearance requirements.
OcclusionState evaluateLineOfSight(const Eigen::Vector3d& boresight_eci, double half_fov_rad,
                                   const SkyGeometry& sky, const KeepOutSpec& keep_out);

}  // namespace polaris::sim::sensors

#endif  // POLARIS_SIM_SENSORS_OCCLUSION_HPP
