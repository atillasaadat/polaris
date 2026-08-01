#ifndef POLARIS_GNC_TRIAD_HPP
#define POLARIS_GNC_TRIAD_HPP

/// @file
/// @brief TRIAD deterministic single-frame attitude initializer (design doc
/// §8.1; REQ-ADET-003).
///
/// TRIAD (Black 1964 [black1964]; Markley & Crassidis §5.2 [markley2014])
/// recovers a full attitude from **two** non-parallel vectors observed in the
/// body frame together with their inertial (ECI) references. It is the
/// cold-start/acquisition path for the coarse estimator (§8.1) and the seed for
/// the MEKF: no prior state, no iteration, no filter convergence assumed.
///
/// Onboard the two pairs are the **sun direction** (sun sensor vs. sun
/// ephemeris) as primary and the **magnetic field** (magnetometer vs. onboard
/// IGRF-14) as secondary. The primary is fitted *exactly* — TRIAD is
/// deliberately asymmetric — so the more accurate observation belongs there,
/// which is why the sun vector leads.
///
/// **Frames and units.** Body observations are `Vec3<Body>`, references are
/// `Vec3<ECI>`; neither needs to be unit length (both are normalised inside).
/// The solution is `Quat<Body, ECI>` (Body ← ECI, JPL scalar-first, canonical
/// `q0 ≥ 0`). Measurement uncertainties are per-axis transverse angular
/// standard deviations in **radians**; the covariance is in **rad²** on the
/// body-frame attitude error `δθ`, the same error state as
/// `state::ErrorState::kAttitude`, so it can seed a filter directly.
///
/// **Assumptions.**
///  - The QUEST measurement model: each observation error is small, transverse
///    to its own vector, and isotropic in that transverse plane with variance
///    `σᵢ²` per axis; the two observations are independent.
///  - The inertial references are treated as exact. Ephemeris and IGRF errors
///    are therefore folded into the observation `σᵢ` by the caller (root-sum-
///    square of sensor and reference uncertainty) — the analytic Sun fallback
///    (§11.3) is ~0.4°, which is not negligible against a coarse sun sensor.
///
/// **Degenerate geometry.** As the two vectors approach parallel the rotation
/// about the primary becomes unobservable and the covariance diverges as
/// `1/sin²θ`. That is a normal flight condition (the sun and field lines do
/// line up), so it is reported as an invalid solution, never asserted.
///
/// Flight path: no heap, no exceptions, fixed-size Eigen, return codes checked.
///
/// References:
///  - Black, "A passive system for determining the attitude of a satellite",
///    AIAA J. 2(7):1350-1351, 1964 (the original TRIAD). [black1964]
///  - Markley & Crassidis, *Fundamentals of Spacecraft Attitude Determination
///    and Control*, 2014, §5.2 (TRIAD and its error covariance). [markley2014]
///  - Shuster & Oh, "Three-Axis Attitude Determination from Vector
///    Observations", J. Guidance & Control 4(1):70-77, 1981 (the QUEST
///    measurement model and the TRIAD covariance). [shuster1981]

#include <Eigen/Core>

#include "math/frames.hpp"
#include "math/quaternion.hpp"
#include "math/typed_vector.hpp"

namespace polaris::gnc {

/// One vector observation: what the sensor sees in Body, what the model says it
/// should be in ECI, and how well the pair is known.
struct VectorObservation {
  /// Observed direction in the body frame [dimensionless direction; any length].
  math::Vec3<math::frames::Body> body{};
  /// Modelled direction in ECI [dimensionless direction; any length].
  math::Vec3<math::frames::ECI> reference{};
  /// Transverse 1σ angular uncertainty of the pair [rad] (sensor ⊕ reference).
  double sigma_rad{0.0};
};

/// Inputs to one TRIAD solve. The primary observation is fitted exactly, so it
/// carries the better-known direction (onboard: the sun vector).
struct TriadInput {
  VectorObservation primary{};    ///< exactly fitted observation (sun)
  VectorObservation secondary{};  ///< observation resolving the roll about the primary (mag)
  /// Geometry gate: reject the solve when `|sin θ|` between the two directions
  /// falls below this [dimensionless]. Applied in **both** frames. No default —
  /// the gate is mission configuration (§19.3), and zero fails validation.
  double min_sin_angle{0.0};
};

/// Result of a TRIAD solve. Read @ref valid before anything else.
struct TriadSolution {
  /// Attitude Body ← ECI (JPL scalar-first, canonical `q0 ≥ 0`).
  math::Quat<math::frames::Body, math::frames::ECI> attitude{};
  /// Attitude-error covariance `E[δθ δθᵀ]` [rad²], body-frame axes.
  Eigen::Matrix3d covariance{Eigen::Matrix3d::Zero()};
  /// Angle between the two body observations [rad], in `(0, π)`; diagnostic.
  double separation_rad{0.0};
  /// True when @ref attitude and @ref covariance are usable.
  bool valid{false};
};

/// Solve TRIAD for @p in, writing @p out.
///
/// Builds the orthonormal triad `{b̂₁, (b̂₁×b̂₂)/|·|, b̂₁×(b̂₁×b̂₂)/|·|}` in each
/// frame and forms the attitude as the product of the two triad matrices
/// (Markley & Crassidis Eq. 5.4 [markley2014]), then the Shuster covariance
/// (see the .cpp for the derivation).
///
/// @param in  observation pair and geometry gate
/// @param out solution; left with `valid == false` on any rejection
/// @return `true` iff a valid solution was produced. Rejections: a zero-length
///         or non-finite input vector, a non-positive `sigma_rad`, a
///         non-positive `min_sin_angle`, near-parallel geometry in either
///         frame, or a non-finite result.
bool triad(const TriadInput& in, TriadSolution& out);

/// Shuster's TRIAD attitude-error covariance for one geometry and one pair of
/// uncertainties, without solving for the attitude.
///
/// Exposed separately because the covariance is **linear in the two variances**:
/// `P(σ₁², σ₂²) = σ₁²·A(b̂₁,b̂₂) + σ₂²·B(b̂₁,b̂₂)`. A caller that has split its
/// error budget into independent contributions can therefore evaluate each on
/// the same geometry and add them — which is exactly how
/// @ref CoarseAttitudeEstimator separates the white part of the budget (which
/// averages down under repeated fixes) from the systematic part (which does
/// not, and so becomes a covariance floor).
///
/// @param primary_body   primary observation in Body [any length]
/// @param secondary_body secondary observation in Body [any length]
/// @param sigma_primary_rad   transverse 1σ of the primary [rad], `>= 0`
/// @param sigma_secondary_rad transverse 1σ of the secondary [rad], `>= 0`
/// @param min_sin_angle  geometry gate on `|sin θ|` [dimensionless], `> 0`
/// @param out            covariance `E[δθ δθᵀ]` [rad²], body-frame axes
/// @return `true` iff the inputs and geometry are usable; @p out is untouched
///         otherwise.
bool triadCovariance(const math::Vec3<math::frames::Body>& primary_body,
                     const math::Vec3<math::frames::Body>& secondary_body, double sigma_primary_rad,
                     double sigma_secondary_rad, double min_sin_angle, Eigen::Matrix3d& out);

}  // namespace polaris::gnc

#endif  // POLARIS_GNC_TRIAD_HPP
