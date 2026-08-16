#ifndef POLARIS_GNC_ST_ALIGNMENT_HPP
#define POLARIS_GNC_ST_ALIGNMENT_HPP

/// @file
/// @brief Commanded inter-star-tracker alignment calibration (design doc §8.2;
/// REQ-ADET-013). Estimates the constant rotation between a second star tracker
/// and the **king** tracker, from simultaneous attitude solutions.
///
/// **The king tracker defines the body frame.** One tracker is designated the
/// attitude reference for the vehicle: its mounting *is* the body frame by
/// definition, so it has no alignment to estimate and none is ever estimated for
/// it. That is a modelling choice with a real consequence — it removes an
/// unobservable degree of freedom rather than hiding one. Two trackers on a
/// structure give you the *relative* rotation between them and nothing about
/// either one's absolute mounting: an unmodelled common rotation of the whole
/// assembly is indistinguishable from a rotation of the body frame, so estimating
/// two absolute alignments from tracker data alone is estimating six parameters
/// from three observable ones. Naming a king collapses that to the three that are
/// observable, and everything downstream — guidance, control, payload pointing —
/// is stated in the frame the king realises.
///
/// **What is estimated.** With `q_k` and `q_2` the king's and the second unit's
/// simultaneous Body ← ECI solutions (each already through its own nominal
/// mounting, so a perfectly aligned pair reads `q_2 = q_k`), the residual
/// misalignment is the constant rotation
/// \f[ q_{\mathrm{rel}} = q_2 \otimes q_k^{-1}, \f]
/// a Body ← Body rotation. The calibration estimates it as the **average** of the
/// per-sample \f$q_{\mathrm{rel},i}\f$ and publishes its inverse as the
/// correction, so a corrected second-tracker reading
/// \f$q_{\mathrm{rel}}^{-1} \otimes q_2\f$ is stated in the king's frame and the
/// two units can be fused as measurements of the same thing.
///
/// **The estimator.** Quaternion averaging in the maximum-likelihood sense
/// [markley2007]: the average is the eigenvector of
/// \f[ M = \sum_i q_{\mathrm{rel},i}\, q_{\mathrm{rel},i}^{\mathsf T} \f]
/// belonging to its largest eigenvalue. This is the right average and not merely a
/// convenient one — arithmetic-mean-then-renormalise is only its first-order
/// approximation and biases with dispersion, and the outer product is invariant to
/// the \f$\pm q\f$ sign ambiguity, so no canonicalisation pass is needed and no
/// sample can cancel another by having been reported with the opposite sign.
///
/// It is also **O(1) in the window length**: `M` is a 4×4 accumulator, so a
/// collection window streams through it and nothing is stored. The same property
/// the magnetometer calibration's normal equations have, for the same reason — a
/// flight component must not hold a window's worth of samples.
///
/// Solved with Eigen's fixed-size `SelfAdjointEigenSolver<Matrix4d>`, the same
/// choice and for the same reason as `gnc::davenport`: on a 4×4 the microseconds a
/// hand-rolled iteration saves are worth far less than a bounded budget with an
/// explicit `info()` status.
///
/// **Quality comes out of the same eigenvalues, exactly.** With \f$N\f$ samples
/// and \f$\theta_i\f$ the angle between sample \f$i\f$ and the fitted average,
/// \f$q_i \cdot \bar q = \cos(\theta_i/2)\f$, so
/// \f$\lambda_{\max} = \sum_i \cos^2(\theta_i/2)\f$ and
/// \f[ \theta_{\mathrm{rms}} = 2\sqrt{1 - \lambda_{\max}/N} \f]
/// is the root-mean-square residual **without revisiting a sample** — it is
/// \f$2\sqrt{\langle \sin^2(\theta/2)\rangle}\f$ written exactly, not a small-angle
/// approximation of the residual (the small-angle step is only in reading
/// \f$2\sin(\theta/2) \approx \theta\f$, which at the arcsecond dispersions this
/// fit sees is exact to parts in \f$10^{11}\f$).
///
/// One numerical consequence is worth stating because it looks like a defect and
/// is not: the square root **amplifies** round-off near zero. On noise-free data
/// \f$\lambda_{\max}/N\f$ differs from 1 by ~\f$10^{-16}\f$, so the reported
/// residual floors at ~\f$2\times 10^{-8}\f$ rad (4 milliarcseconds) rather than
/// at zero. That is six orders below any residual gate a real tracker pair
/// justifies, so it costs nothing — but a caller must not read "residual == 0" as
/// the perfect-fit condition, and a test must not assert it.
///
/// **Every unobservable or untrustworthy case is a refusal**, never an assert and
/// never a silently applied correction:
///
///  - **too few samples** (@ref StAlignmentConfig::min_samples). Three parameters
///    are determined by one pair, so unlike the magnetometer fit there is no
///    algebraic minimum above one — but a single pair reports zero residual
///    whatever the truth, so a residual gate only means something on a
///    well-overdetermined window, and the sample floor is what makes it so.
///  - **inconsistent pairs** (@ref StAlignmentConfig::min_eigen_gap). The gate is
///    the normalised gap \f$(\lambda_{\max} - \lambda_2)/N\f$ between the top two
///    eigenvalues: near 1 when every sample agrees on one rotation, collapsing
///    towards 0 when they do not. This is the analogue of `davenport`'s
///    observability ratio, and it is worth being precise about what it catches,
///    because it is **not** a geometry gate. Attitude pairs have no degenerate
///    geometry — one pair already determines all three parameters, at any
///    attitude — so there is no equivalent of TRIAD's near-parallel refusal here.
///    What the gap detects is a *fault*: one tracker delivering solutions that do
///    not sit at a fixed rotation from the other's (a mis-identified star field, a
///    unit reporting stale or another unit's solution, a mounting that is moving).
///    Refusing there is refusing to average a rotation that does not exist.
///  - **dispersion** (@ref StAlignmentConfig::max_residual_rad): a fit whose
///    residual is larger than the trackers' own combined noise is measuring
///    something other than a fixed misalignment, and applying it would inject that
///    something into the fused solution.
///  - **numerical failure**: a non-converged eigensolve, a non-finite or
///    unnormalisable average.
///
/// There is deliberately **no "must beat the uncalibrated fit" gate** of the kind
/// the magnetometer calibration carries. The two are not analogous: an ellipsoid
/// fit can converge on a worse sensor model, whereas an alignment estimate is a
/// mean of a quantity that is either constant (and then the mean is right) or not
/// constant (and then the eigen-gap gate catches it). Adding a comparison against
/// "no correction" would only ever fire when the true misalignment is smaller than
/// the noise — a case where applying the estimate is harmless.
///
/// **Frames, units, conventions.** JPL scalar-first quaternions, canonical
/// `q0 ≥ 0` (§3.3); angles in radians; SI throughout. Inputs are `Quat<Body, ECI>`
/// and the correction is `Quat<Body, Body>`, which is the type-level statement
/// that this is a frame *refinement* and not an attitude.
///
/// **Simultaneity is the caller's.** The two solutions must be from the same
/// cycle: a vehicle rotating at even 0.1 °/s smears 360 arcsec of spurious
/// misalignment into a 1 s time skew, which is three times the effect being
/// measured. The component only offers a sample when both units delivered a fresh,
/// valid solution in the same estimation cycle.
///
/// **Flight path.** Fixed-size storage, no heap, no exceptions, no recursion,
/// bounded loops, every return code checked, finiteness checks on the output. No
/// F´ types and no I/O — the `AttitudeEstimator` component wraps this into the
/// commanded `ST_ALIGN_CAL_START` / `ABORT` / `CLEAR` flow, the same shape as the
/// Push 46 magnetometer calibration.
///
/// References:
///  - Markley, Cheng, Crassidis & Oshman, "Averaging Quaternions", *J. Guidance,
///    Control, and Dynamics* 30(4):1193-1197, 2007 (the maximum-eigenvalue
///    average and why the naive mean is only its first-order form).
///    [markley2007]
///  - Markley & Crassidis, *Fundamentals of Spacecraft Attitude Determination and
///    Control*, Springer 2014, §5.3 (the same 4×4 eigenproblem underlying
///    Davenport's q-method) and §6.2.4 (fusing multiple attitude sensors).
///    [markley2014]
///  - Design doc §8.1, §8.2 (king-tracker architecture and the commanded
///    inter-tracker calibration, user decision 2026-08-02).

#include <cstdint>
#include <Eigen/Core>

#include "math/frames.hpp"
#include "math/quaternion.hpp"

namespace polaris::gnc {

/// Tuning for @ref StAlignmentAccumulator. No defaults (§19.3): an unconfigured
/// accumulator refuses every sample and every fit.
struct StAlignmentConfig {
  /// Fewest accepted sample pairs a fit is attempted on. Must be at least 2 — one
  /// pair fits exactly and reports a zero residual whatever the truth, so the
  /// quality gates below would be meaningless.
  std::uint32_t min_samples = 0;

  /// Largest accepted RMS residual [rad]. Size it from the two trackers' combined
  /// per-sample noise with margin: above it the pairs are not describing a fixed
  /// rotation. Must be positive.
  double max_residual_rad = 0.0;

  /// Smallest accepted normalised eigen-gap `(λ_max − λ₂)/N` [dimensionless], in
  /// `(0, 1)`. Near 1 means every sample agrees on one rotation. See the file
  /// header for what this catches and — importantly — what it is not.
  double min_eigen_gap = 0.0;

  /// Range gate, applied at construction.
  bool isValid() const;
};

/// Why a fit was refused. Reported so the ground knows whether to collect longer,
/// investigate a unit, or stop trying — the same reasoning the magnetometer
/// calibration's rejection reasons follow.
enum class StAlignmentRejection : std::uint8_t {
  kNone = 0,        ///< the fit succeeded
  kConfig = 1,      ///< the accumulator is unconfigured
  kSamples = 2,     ///< fewer accepted pairs than StAlignmentConfig::min_samples
  kDegenerate = 3,  ///< eigen-gap below the gate: the pairs do not share one rotation
  kDispersion = 4,  ///< RMS residual above StAlignmentConfig::max_residual_rad
  kNumerical = 5    ///< eigensolve did not converge, or a non-finite result
};

/// A fitted (or default, un-fitted) inter-tracker alignment.
struct StAlignmentResult {
  /// The correction to apply to the second unit's reported attitude:
  /// `q_king_frame = correction ⊗ q_reported`. It is `q_rel⁻¹` for the `q_rel` of
  /// the file header. Identity and @ref valid false when nothing is fitted, which
  /// is why @ref applyStAlignment can be called unconditionally.
  math::Quat<math::frames::Body, math::frames::Body> correction{
      math::Quat<math::frames::Body, math::frames::Body>::Identity()};

  /// RMS angle between the samples and the fitted average [rad] — the number the
  /// ground grades the calibration on.
  double residual_angle_rad = 0.0;

  /// The eigenvalue-gap quality metric `(λ_max − λ₂)/N` [dimensionless].
  double eigen_gap = 0.0;

  /// Total misalignment angle the correction removes [rad], `2·atan2(‖v‖, |q₀|)`
  /// of the correction — the size of what was found, which is what tells an
  /// operator whether the unit is mounted where the drawing says.
  double misalignment_angle_rad = 0.0;

  /// Accepted pairs the fit ran on.
  std::uint32_t samples = 0;

  /// A correction is fitted and may be applied.
  bool valid = false;
};

/// Streaming accumulator for the commanded inter-tracker alignment calibration.
/// Fixed storage — one 4×4 symmetric matrix and a count — so a collection window
/// of any length allocates nothing.
class StAlignmentAccumulator {
 public:
  StAlignmentAccumulator() = default;

  /// Build with @p config. An invalid config leaves the accumulator inert: every
  /// sample is refused and @ref fit returns @ref StAlignmentRejection::kConfig.
  explicit StAlignmentAccumulator(const StAlignmentConfig& config);

  bool isConfigured() const { return configured_; }

  const StAlignmentConfig& config() const { return config_; }

  /// Drop every accumulated sample, keeping the configuration. Called when a
  /// window opens, so a new window never inherits the previous one's data.
  void reset();

  /// Accept one **simultaneous** pair of solutions.
  ///
  /// @param king   the king tracker's attitude, Body ← ECI. This unit's mounting
  ///               defines the body frame, so its reading is taken as the frame
  ///               itself.
  /// @param second the second unit's attitude, Body ← ECI, as reported through its
  ///               nominal mounting and **without** any previously applied
  ///               correction — a fit fed its own correction would refit the
  ///               identity, exactly as the magnetometer fit must see raw samples.
  /// @return false (and nothing accumulated) for an unconfigured accumulator or a
  ///         non-finite or unnormalisable input. Never an assert: these are wire
  ///         values.
  bool addSample(const math::Quat<math::frames::Body, math::frames::ECI>& king,
                 const math::Quat<math::frames::Body, math::frames::ECI>& second);

  /// Accepted pairs so far.
  std::uint32_t sampleCount() const { return samples_; }

  /// Solve for the alignment.
  ///
  /// @param out receives the correction and its quality on success, and is left
  ///        default-constructed (so `out.valid == false`) on every refusal — a
  ///        refused fit must never leave a half-written correction the caller
  ///        could apply.
  /// @return @ref StAlignmentRejection::kNone on success, otherwise the gate that
  ///         closed.
  StAlignmentRejection fit(StAlignmentResult& out) const;

 private:
  StAlignmentConfig config_{};
  bool configured_ = false;
  Eigen::Matrix4d moment_{Eigen::Matrix4d::Zero()};  ///< Σ q_rel q_relᵀ
  std::uint32_t samples_ = 0;
};

/// Apply @p alignment to a second-tracker reading, returning it in the king's
/// frame. Passes @p measured through unchanged when nothing is fitted, which is
/// what lets the component call it unconditionally at **one** application point —
/// so the MEKF, the residual monitors and any future consumer cannot disagree
/// about which frame a tracker's solution was in.
math::Quat<math::frames::Body, math::frames::ECI> applyStAlignment(
    const StAlignmentResult& alignment,
    const math::Quat<math::frames::Body, math::frames::ECI>& measured);

}  // namespace polaris::gnc

#endif  // POLARIS_GNC_ST_ALIGNMENT_HPP
