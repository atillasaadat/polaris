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
/// The model answers two different questions, because sensors need both:
///
///  - **Is a keep-out violated?** A hard constraint — the unit's baffle rejection
///    and the vendor's exclusion angles. Drives a validity flag.
///  - **What fraction of the field of view does each body cover?** A continuous
///    quantity, needed for graded degradation (a tracker losing part of its star
///    field, a sun sensor's albedo contribution) rather than a cliff at the
///    keep-out edge. Also what makes an outage's *onset* smooth, which matters
///    for anything that differentiates a measurement stream.
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

#include <Eigen/Core>

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

  /// Fraction of the FOV covered by any body — the union, approximated as the
  /// largest single contributor. Bodies overlapping each other in one FOV is a
  /// geometry no real sensor is operated in (it means staring at an eclipse),
  /// so the approximation costs nothing real and keeps the value monotone.
  double blockedFraction() const;
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

/// Fraction of a circular field of view of half-angle @p half_fov_rad covered by
/// a disk of angular radius @p body_radius_rad whose centre is @p separation_rad
/// from the boresight.
///
/// Returns 0 for a non-positive FOV: a sensor with no field of view has no
/// fraction to report, and dividing by its area would be a NaN in a telemetry
/// channel.
double fovCoveredFraction(double half_fov_rad, double separation_rad, double body_radius_rad);

/// Evaluate one line of sight: keep-out violations and per-body FOV coverage.
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
