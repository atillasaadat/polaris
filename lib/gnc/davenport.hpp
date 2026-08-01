#ifndef POLARIS_GNC_DAVENPORT_HPP
#define POLARIS_GNC_DAVENPORT_HPP

/// @file
/// @brief Davenport q-method — optimal attitude from N weighted vector pairs
/// (design doc §8.1; REQ-ADET-003).
///
/// Where @ref triad takes exactly two observations and fits the primary
/// *exactly*, the q-method (Davenport 1968 [davenport1968]; Markley & Crassidis
/// §5.3 [markley2014]) solves **Wahba's problem** for any number of pairs,
/// weighting each by its own uncertainty:
/// \f[
///   \min_A \tfrac{1}{2}\sum_i w_i \left\| \hat{\mathbf b}_i - A\,\hat{\mathbf r}_i \right\|^2 ,
///   \qquad w_i = 1/\sigma_i^2 .
/// \f]
/// Maximising the equivalent gain \f$\sum_i w_i \hat{\mathbf b}_i^\top A \hat{\mathbf r}_i\f$
/// over unit quaternions turns into the eigenvalue problem \f$K\,\bar q = \lambda_{\max}\bar q\f$
/// with the symmetric 4×4
/// \f[
///   B = \sum_i w_i \hat{\mathbf b}_i \hat{\mathbf r}_i^\top, \quad
///   S = B + B^\top, \quad \sigma = \operatorname{tr} B, \quad
///   \mathbf z = \sum_i w_i\,\hat{\mathbf b}_i \times \hat{\mathbf r}_i,
/// \f]
/// \f[
///   K = \begin{bmatrix} \sigma & \mathbf z^\top \\ \mathbf z & S - \sigma I \end{bmatrix}
/// \f]
/// written here **scalar-first**, matching the JPL quaternion layout (Markley's
/// K is the same matrix with the scalar row/column last; the JPL/Shuster
/// attitude matrix and quaternion product are the ones this project already
/// uses, §3.3, so no convention translation is needed beyond that permutation).
///
/// **Why the q-method and not QUEST.** QUEST reaches the same optimum faster by
/// Newton-iterating the characteristic polynomial from the starting guess
/// \f$\lambda \approx \sum w_i\f$. That iteration has a data-dependent
/// termination, no error return when it fails to converge, and a documented
/// failure mode when the two largest eigenvalues of K approach each other. Here
/// the whole problem is a **symmetric 4×4**, so Eigen's
/// `SelfAdjointEigenSolver<Matrix4d>` handles it on fixed-size storage with an
/// internally bounded iteration budget and an explicit `info()` status when that
/// budget is exhausted — a bounded worst case with a checkable failure, which is
/// worth far more on a flight path than the microseconds QUEST saves on a 4×4.
///
/// **Where it is used.** Cold start and re-initialisation of the MEKF (§8.1)
/// **only** — never in the steady-state loop, which is the MEKF's job. The
/// q-method has no memory: it throws away the prior and the gyro every time it
/// runs, which is exactly what you want when acquiring and exactly what you do
/// not want once converged.
///
/// **Covariance.** Under the QUEST measurement model
/// \f$E[\delta\hat{\mathbf b}_i \delta\hat{\mathbf b}_i^\top] = \sigma_i^2 (I - \hat{\mathbf
/// b}_i\hat{\mathbf b}_i^\top)\f$ with independent observations, the maximum-likelihood weighting
/// \f$w_i = 1/\sigma_i^2\f$ makes the q-method optimum efficient, and its
/// attitude-error covariance is the inverse Fisher information
/// \f[
///   P_{\theta\theta} = \Big[ \sum_i \sigma_i^{-2}\,(I - \hat{\mathbf b}_i \hat{\mathbf b}_i^\top)
///   \Big]^{-1}
/// \f]
/// (Shuster & Oh 1981 [shuster1981]; Markley & Crassidis §5.3). This is the
/// same result Shuster derived for QUEST — it belongs to the Wahba optimum, not
/// to the algorithm used to find it, so it applies verbatim here. For two
/// observations it agrees with @ref triadCovariance up to TRIAD's deliberate
/// sub-optimality (TRIAD fits the primary exactly instead of weighting both).
///
/// **Frames, units, conventions.** Body observations are `Vec3<Body>`,
/// references `Vec3<ECI>`; neither needs unit length. The solution is
/// `Quat<Body, ECI>` (Body ← ECI, JPL scalar-first, canonical `q0 ≥ 0`).
/// Uncertainties are per-axis transverse angular 1σ in **radians**; the
/// covariance is **rad²** on the body-frame attitude error `δθ` defined by
/// `A_est = (I − [δθ×])·A_true` (the same convention @ref triad quotes), which
/// is the error state of `state::ErrorState::kAttitude`, so it seeds the MEKF
/// directly — @ref Mekf uses the opposite sign for `δθ`, which a covariance
/// does not notice. The
/// inertial references are treated as exact — reference error is the caller's
/// to fold into `sigma_rad` (root-sum-square), as it is for TRIAD.
///
/// **Degeneracy is refused, never asserted.** Fewer than two usable pairs, a
/// zero-length or non-finite vector, a non-positive σ, a near-collinear
/// observation set (the information matrix loses rank and the rotation about the
/// common direction is unobservable), or a non-finite result all return `false`.
/// Near-collinear sun/field geometry is a normal flight condition.
///
/// Flight path: no heap, no exceptions, fixed-size Eigen, bounded loops, return
/// codes checked.
///
/// References:
///  - Davenport, "A Vector Approach to the Algebra of Rotations with
///    Applications", NASA TN D-4696, 1968 (the q-method). [davenport1968]
///  - Markley & Crassidis, *Fundamentals of Spacecraft Attitude Determination
///    and Control*, 2014, §5.3 (Wahba's problem, Davenport's q-method, QUEST,
///    and the attitude-profile covariance). [markley2014]
///  - Shuster & Oh, "Three-Axis Attitude Determination from Vector
///    Observations", J. Guidance & Control 4(1):70-77, 1981 (the QUEST
///    measurement model and covariance). [shuster1981]

#include <Eigen/Core>

#include "gnc/triad.hpp"
#include "math/frames.hpp"
#include "math/quaternion.hpp"
#include "math/typed_vector.hpp"

namespace polaris::gnc {

/// Inputs to one q-method solve: up to @ref kMaxObservations vector pairs and
/// the observability gate they must clear together.
///
/// The observation type is @ref VectorObservation, shared with @ref triad — the
/// per-pair `sigma_rad` **is** the weighting: \f$w_i = 1/\sigma_i^2\f$, the
/// maximum-likelihood choice, which is what makes the reported covariance the
/// inverse Fisher information rather than an arbitrary weighted fit.
struct DavenportInput {
  /// Fixed capacity — the array is a member, so a solve allocates nothing.
  /// Sized to `flight::GncMaxUnits` (8), the port-array width the §8.2 fusion
  /// layer will present.
  static constexpr int kMaxObservations = 8;

  /// The observation set; only the first @ref count entries are read.
  VectorObservation observations[kMaxObservations]{};
  /// Number of populated entries, in `[2, kMaxObservations]`.
  int count{0};

  /// Observability gate [dimensionless], in `(0, 1)`. The solve is refused
  /// unless `λ_min / λ_max` of the Fisher information matrix
  /// `M = Σ σᵢ⁻²(I − b̂ᵢb̂ᵢᵀ)` reaches this value, in **both** the body and the
  /// reference set.
  ///
  /// The ratio is scale-free, so it gates geometry rather than noise level. Two
  /// equally-weighted orthogonal observations give exactly `0.5`; a pair
  /// separated by θ gives roughly `sin²θ / 2` (≈ 0.015 at 10°). No default —
  /// this is mission configuration (§19.3), and zero fails validation.
  double min_observability{0.0};
};

/// Result of a q-method solve. Read @ref valid before anything else.
struct DavenportSolution {
  /// Attitude Body ← ECI (JPL scalar-first, canonical `q0 ≥ 0`).
  math::Quat<math::frames::Body, math::frames::ECI> attitude{};
  /// Attitude-error covariance `E[δθ δθᵀ]` [rad²], body-frame axes.
  Eigen::Matrix3d covariance{Eigen::Matrix3d::Zero()};
  /// Wahba loss at the optimum, `Σ wᵢ − λ_max` [dimensionless]. Zero for a
  /// noise-free consistent set; a large value means the observations disagree
  /// with each other by more than their weights allow — a fault signature, not
  /// a geometry problem. Diagnostic only; nothing here gates on it.
  double loss{0.0};
  /// Ratio `λ_min/λ_max` of the body information matrix [dimensionless]; the
  /// quantity @ref DavenportInput::min_observability gates. Diagnostic.
  double observability{0.0};
  /// True when @ref attitude and @ref covariance are usable.
  bool valid{false};
};

/// Solve Wahba's problem for @p in by Davenport's q-method, writing @p out.
///
/// Builds `K` from the weighted attitude profile matrix, takes the eigenvector
/// of its largest eigenvalue as the optimal quaternion, and forms the inverse
/// Fisher-information covariance on the same observation set.
///
/// @param in  observation set and observability gate
/// @param out solution; left with `valid == false` on any rejection
/// @return `true` iff a valid solution was produced. Rejections: `count`
///         outside `[2, kMaxObservations]`, a non-positive `min_observability`,
///         a zero-length/non-finite vector, a non-positive `sigma_rad`, an
///         observation set below the observability gate in either frame, an
///         eigen-decomposition that did not converge, or a non-finite result.
///         Nothing here asserts on measurement content.
bool davenport(const DavenportInput& in, DavenportSolution& out);

}  // namespace polaris::gnc

#endif  // POLARIS_GNC_DAVENPORT_HPP
