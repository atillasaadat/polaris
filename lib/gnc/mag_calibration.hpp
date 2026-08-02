#ifndef POLARIS_GNC_MAG_CALIBRATION_HPP
#define POLARIS_GNC_MAG_CALIBRATION_HPP

/// @file
/// @brief Attitude-independent magnetometer hard/soft-iron calibration
/// (design doc §8.1, "Calibration is commanded, executed, and assessed on
/// orbit"; supports REQ-ADET-005/006).
///
/// **Why this exists.** On the reference vehicle the magnetic pair carries a
/// 1.9° 1σ *systematic* term — an uncalibrated hard iron of ~1 µT against a
/// ~30 µT LEO field, plus a few milliradians of soft iron and misalignment.
/// Systematics do not average down (§8.1), so that term is the floor of both
/// the coarse chain and the MEKF, and no amount of filtering removes it. What
/// removes it is measuring it. Recovering the 1.9° term to the 0.2–0.5° class
/// is the evidence path that tightens REQ-ADET-005 to 5° and REQ-ADET-006 to
/// 3°.
///
/// **Why it can run from Safe mode.** The fit is **attitude-free**. The sensor
/// model is
/// \f[
///   \mathbf m_k = S\,\mathbf B_k + \mathbf b + \boldsymbol\nu_k ,
/// \f]
/// with \f$S\f$ the soft-iron/scale/misalignment matrix (\f$\approx I\f$),
/// \f$\mathbf b\f$ the hard-iron offset and \f$\mathbf B_k\f$ the true field in
/// body axes. The attitude is unknown, but its *magnitude* is not:
/// \f$|\mathbf B_k| = F_k\f$, the onboard IGRF-14 field magnitude at the
/// sample's position and time (§11.3). Eliminating the attitude leaves the
/// scalar constraint
/// \f[
///   (\mathbf m_k - \mathbf b)^\top A\,(\mathbf m_k - \mathbf b) = F_k^2 ,
///   \qquad A \equiv S^{-\top} S^{-1},
/// \f]
/// i.e. the measurements lie on an **ellipsoid** whose centre is the hard iron
/// and whose shape is the soft iron. No attitude solution enters, so this works
/// on an uncalibrated, tumbling vehicle with no star tracker — which is exactly
/// why the design sequences it first among the calibration items.
///
/// **The fit is one linear least-squares solve.** Expanding the quadric and
/// collecting the ten unknowns
/// \f$\theta = [A_{11},A_{22},A_{33},A_{12},A_{13},A_{23},\;
/// \mathbf v,\;c]\f$ with \f$\mathbf v = A\mathbf b\f$ and
/// \f$c = \mathbf b^\top A \mathbf b\f$ gives a model that is **linear in
/// \f$\theta\f$**:
/// \f[
///   \underbrace{[\,x_1^2,\,x_2^2,\,x_3^2,\,2x_1x_2,\,2x_1x_3,\,2x_2x_3,\,
///   -2x_1,\,-2x_2,\,-2x_3,\,1\,]}_{h(\mathbf x_k)^\top}\;\theta \;=\; f_k^2 ,
/// \f]
/// with \f$\mathbf x = \mathbf m / F_\text{nom}\f$ and \f$f = F/F_\text{nom}\f$
/// (see @ref MagCalibrationConfig::nominal_field_t — a preconditioner, not a
/// model parameter). This is Alonso & Shuster's *centered* attitude-independent
/// measurement in its complete, nine-plus-one-parameter form
/// [alonso2002complete]; the ellipsoid reading of the same equation is
/// [vasconcelos2011]. Note the right-hand side is **inhomogeneous** — the IGRF
/// magnitude supplies absolute scale — so unlike a general quadric fit there is
/// no scale ambiguity to normalise away.
///
/// The estimator is therefore \f$\hat\theta = N^{-1} g\f$ with
/// \f$N = \sum_k h_k h_k^\top\f$ (10×10) and \f$g = \sum_k h_k f_k^2\f$. Both
/// accumulate **sample by sample**: memory is O(1) in the number of samples, so
/// the component streams a collection window through @ref
/// MagCalibrationAccumulator::addSample and never stores it.
///
/// **Why not TWOSTEP.** Unweighted least squares on this equation is biased at
/// second order in \f$\nu/F\f$, because the noise enters the quadratic terms;
/// TWOSTEP [alonso2002twostep] and the Crassidis EKF/UKF variants
/// [crassidis2005] exist to remove exactly that bias. On this vehicle
/// \f$\nu/F \approx 0.05/30 = 1.7\times10^{-3}\f$, so the bias is parts in
/// \f$10^6\f$ of the field — three orders below the 0.2–0.5° (≈ 3–9 mrad)
/// target. Paying for ML optimality would buy nothing measurable and would cost
/// an iteration-to-convergence loop on a flight path. The batch-linear solve is
/// a fixed, bounded amount of arithmetic with an explicit failure status, which
/// is the right trade here.
///
/// **The unobservable rotation.** Magnitudes constrain only
/// \f$A = S^{-\top}S^{-1}\f$, so \f$S^{-1}\f$ is recovered up to an arbitrary
/// left rotation: \f$RS^{-1}\f$ fits the data identically for any \f$R\f$. That
/// rotation is *physically* unobservable without an attitude reference — it is
/// a boresight/mounting error, not an iron error. The standard resolution is to
/// take the **symmetric positive-definite square root** \f$M = A^{1/2}\f$
/// (unique, [higham2008] Ch. 6), which is the correct choice for a body-frame
/// iron correction: a real soft-iron/permeability tensor is symmetric, so the
/// symmetric factor *is* the physical one, and any residual mounting rotation
/// belongs to the alignment calibration (§8.1, sequenced after tracker fusion)
/// rather than here.
///
/// **Sample pairing and the GNSS staleness skew.** The component pairs each
/// reading with the IGRF magnitude at the position of that cycle's GNSS fix,
/// which the §9.1 staleness gate admits up to `MaxMeasAgeSec` (1 s on the
/// reference vehicle) old. At LEO speeds a second of skew displaces the
/// evaluation point ~7.6 km, worth ~3e-4 of relative field magnitude. Because
/// the skew is uncorrelated with the vehicle's attitude it enters the fit as a
/// near-isotropic scale error on `A`, so it inflates the reported residual
/// marginally and does **not** rotate the corrected vector — an isotropic scale
/// commutes with direction. Tightening the pairing buys nothing measurable
/// against the 0.2-0.5 deg target.
///
/// **Frames, units, conventions.** Raw measurements and the recovered hard iron
/// are `Vec3<Body>` in **teslas**; the IGRF magnitude is a scalar in teslas;
/// @ref MagCalibrationResult::soft_iron_inverse is dimensionless. The
/// correction to apply downstream is
/// `B_corrected = soft_iron_inverse * (m_raw − hard_iron_offset)` — see
/// @ref applyMagCalibration. All SI.
///
/// **Refusal, never assertion.** Every failure mode — too few samples, samples
/// spanning too narrow a cone of directions, an ill-conditioned normal matrix, a
/// fitted quadric that is not positive definite, a fit whose residual does not
/// beat the uncalibrated one — returns `false`. A calibration that makes the
/// magnetometer *worse* is the one outcome worth guarding hardest against, so it
/// has its own gate.
///
/// **Flight path.** Fixed-size Eigen, no heap, no exceptions, no recursion, no
/// unbounded loops, return codes checked, finiteness checks on every output. No
/// F´ types and no I/O — the `AttitudeEstimator` component wraps this.
///
/// References:
///  - Alonso & Shuster, "Complete Linear Attitude-Independent Magnetometer
///    Calibration", J. Astronautical Sciences 50(4):477-490, 2002 (the
///    ten-parameter centered model this fit solves). [alonso2002complete]
///  - Alonso & Shuster, "TWOSTEP: A Fast Robust Algorithm for
///    Attitude-Independent Magnetometer-Bias Determination", J. Astronautical
///    Sciences 50(4):433-451, 2002 (the ML treatment and the noise-induced bias
///    it corrects). [alonso2002twostep]
///  - Vasconcelos et al., "Geometric Approach to Strapdown Magnetometer
///    Calibration in Sensor Frame", IEEE Trans. Aerospace and Electronic
///    Systems 47(2):1293-1306, 2011 (the ellipsoid formulation, the symmetric
///    factorisation, and the observability discussion). [vasconcelos2011]
///  - Crassidis, Lai & Harman, "Real-Time Attitude-Independent Three-Axis
///    Magnetometer Calibration", J. Guidance, Control, and Dynamics
///    28(1):115-120, 2005 (recursive alternatives). [crassidis2005]
///  - Higham, *Functions of Matrices: Theory and Computation*, 2008, Ch. 6 (the
///    unique symmetric positive-definite square root). [higham2008]

#include <cstdint>
#include <Eigen/Core>

#include "math/frames.hpp"
#include "math/typed_vector.hpp"

namespace polaris::gnc {

/// Tuning for @ref MagCalibrationAccumulator. Every value is mission
/// configuration (§19.3); there are no flight defaults, and a
/// default-constructed config deliberately fails @ref isValid.
struct MagCalibrationConfig {
  /// Field magnitude used to non-dimensionalise the fit [T], e.g. 30 µT in LEO.
  ///
  /// **Numerics only.** The normal equations are exactly diagonally
  /// preconditioned by this value, so the solution is invariant to it up to
  /// round-off; without it the ten unknowns would span `|m|²~10⁻⁹` against
  /// `1`, which is 18 orders of dynamic range in a 10×10 normal matrix. Any
  /// value within a factor of a few of the true field works.
  double nominal_field_t{0.0};
  /// Smallest accepted `|m_raw|` and IGRF magnitude [T]. Rejects a dead or
  /// unpowered sensor and a nonsensical reference.
  double min_field_t{0.0};
  /// Largest accepted `|m_raw|` and IGRF magnitude [T]. Rejects a saturated
  /// reading or a magnetorquer-contaminated sample.
  double max_field_t{0.0};
  /// Fewest samples that may be solved, floored at **twice** the ten unknowns by
  /// @ref isValid. The *algebraic* minimum is 10, which fits exactly and reports
  /// zero residual regardless of truth — so every quality gate below is
  /// meaningless there, and the improvement gate would wave the fit through on
  /// its own zero residual. Refusing that configuration structurally is cheaper
  /// than catching it downstream. The reference vehicle uses 100 — a
  /// ten-times-overdetermined fit, ~10 s of a 10 Hz collection window.
  std::int32_t min_samples{0};
  /// Smallest accepted @ref MagCalibrationResult::coverage [dimensionless], in
  /// `(0, 1]`. See that field for the geometry this number measures. The
  /// reference vehicle uses 0.35, roughly a 42° half-angle cone of sampled
  /// field directions.
  double min_coverage{0.0};
  /// Largest accepted condition number `λ_max/λ_min` of the (preconditioned)
  /// normal matrix [dimensionless], `> 1`. This is the rigorous observability
  /// gate — it sees every way the ten parameters can fail to separate, where
  /// @ref min_coverage only sees the direction spread. The reference vehicle
  /// uses 1e6.
  double max_condition{0.0};
  /// The calibrated residual must beat the uncalibrated residual by at least
  /// this factor [dimensionless], `>= 1`. A "calibration" that does not improve
  /// the scalar check is refused rather than applied, as is one where the
  /// uncalibrated residual was already zero — there is nothing to beat. The
  /// reference vehicle uses 2.
  double min_residual_improvement{0.0};

  /// True when every field is finite and in range (and `min_field < max_field`,
  /// `min_samples >= 10`). Checked once at construction.
  bool isValid() const;
};

/// Product of one calibration fit. Read @ref valid before anything else.
struct MagCalibrationResult {
  /// Hard-iron offset `b` [T, Body] — subtract this from the raw reading.
  math::Vec3<math::frames::Body> hard_iron_offset{};
  /// The soft-iron correction to **apply**, \f$M = A^{1/2} \simeq S^{-1}\f$
  /// [dimensionless], symmetric positive definite:
  /// `B_corrected = M · (m_raw − hard_iron_offset)`. Recovered up to the
  /// unobservable left rotation discussed in the file header.
  Eigen::Matrix3d soft_iron_inverse{Eigen::Matrix3d::Identity()};
  /// RMS of the scalar check `| M(m−b) | − F` after calibration [T].
  double residual_rms_t{0.0};
  /// RMS of the same check on the *raw* data, `|m| − F` [T]. The before/after
  /// pair is what @ref MagCalibrationConfig::min_residual_improvement gates,
  /// and what the ground sees in telemetry.
  double uncalibrated_residual_rms_t{0.0};
  /// @ref residual_rms_t as an angle [rad]: `residual_rms_t / mean_field_t`, the
  /// direction error a field-magnitude error of that size corresponds to.
  ///
  /// **Indicative, not a bound.** The scalar check is blind to the component of
  /// the error transverse to the field, which is precisely the component that
  /// rotates the vector. It is the right *number to telemeter* because it is
  /// the only accuracy figure available without an attitude reference, and it
  /// tracks the true direction error closely once the systematic part has been
  /// removed — but the honest verification of the direction error is the
  /// Monte Carlo campaign, not this field.
  double residual_angle_rad{0.0};
  /// Mean IGRF magnitude over the accepted samples [T].
  double mean_field_t{0.0};
  /// Orientation coverage of the accepted samples, `3·λ_min(D)` where
  /// `D = (1/N) Σ m̂ₖ m̂ₖᵀ` is the second-moment matrix of the sampled field
  /// **directions** [dimensionless], in `[0, 1]`.
  ///
  /// Geometric meaning: `D` has unit trace, so `λ_min` is the fraction of the
  /// directional variance in the worst-covered axis and `1/3` is the isotropic
  /// value — hence the factor 3, which puts an isotropic (or hemispherical)
  /// sample set at **1.0** and a set confined to a plane or a line at **0**.
  /// For directions spread uniformly over a cone of half-angle α the metric is
  /// `(3/2)·⟨sin²θ⟩`: 0.09 at α = 20°, 0.32 at 40°, 0.40 at 45°, 1.0 at 90°.
  /// A calibration fitted inside a narrow cone extrapolates the ellipsoid over
  /// directions it never saw, which is worse than no calibration — so this is a
  /// gate, not a diagnostic.
  double coverage{0.0};
  /// Condition number `λ_max/λ_min` of the preconditioned normal matrix
  /// [dimensionless]; the quantity @ref MagCalibrationConfig::max_condition
  /// gates. Diagnostic.
  double condition{0.0};
  /// Samples accepted into the fit.
  std::int32_t sample_count{0};
  /// True when every field above is usable.
  bool valid{false};
};

/// Why @ref MagCalibrationAccumulator::solve refused, for the ground.
///
/// A refusal is only actionable if it names which gate closed: `Coverage` means
/// "tumble further", `Samples` means "collect longer", and `Condition` /
/// `NoImprovement` / `Numerical` mean the window is not going to produce a
/// calibration however long it runs. The bare `solve(out)` overload drops this;
/// the flight component keeps it and puts it in the refusal EVR.
enum class MagCalibrationRefusal {
  None,           ///< the solve succeeded
  NotConfigured,  ///< the accumulator's config failed MagCalibrationConfig::isValid
  Samples,        ///< fewer than `min_samples` accepted
  Coverage,       ///< sampled directions span less than `min_coverage`
  Condition,      ///< normal-matrix condition above `max_condition`
  NoImprovement,  ///< the fit failed `min_residual_improvement` against the raw data
  Numerical       ///< an eigen-decomposition failed, or the fitted quadric/result was unusable
};

/// Streaming accumulator for one calibration window (design doc §8.1).
///
/// Hold one instance, @ref reset it when `MAG_CAL_START` opens a window, feed
/// every accepted magnetometer sample to @ref addSample as it arrives, and call
/// @ref solve when the window closes. Storage is fixed and independent of the
/// number of samples: the object carries the 10×10 normal matrix, a 10-vector,
/// a 3×3 direction moment and four scalars, and allocates nothing ever.
class MagCalibrationAccumulator {
 public:
  /// Number of unknowns in the quadric fit — six for the symmetric `A`, three
  /// for `A·b`, one for `bᵀAb`. Also the algebraic minimum sample count.
  static constexpr int kParameters = 10;

  /// Construct with @p config. If the config is invalid the accumulator is
  /// inert: @ref addSample and @ref solve return false forever. Check
  /// @ref isConfigured.
  explicit MagCalibrationAccumulator(const MagCalibrationConfig& config);

  /// True when the configuration passed @ref MagCalibrationConfig::isValid.
  bool isConfigured() const { return configured_; }

  /// Discard every accumulated sample. Configuration is retained.
  void reset();

  /// Accumulate one raw magnetometer sample against its IGRF field magnitude.
  ///
  /// Bounded work: one 10-element row, one rank-1 update of the normal matrix,
  /// no branches on sample history.
  ///
  /// @param m_raw           uncorrected magnetometer reading [T, Body]
  /// @param igrf_magnitude_t modelled field magnitude `|B_igrf(r_k, t_k)|` [T]
  /// @return `true` iff the sample was accepted. Rejected — and *not*
  ///         accumulated — when the accumulator is unconfigured, either input
  ///         is non-finite, or either magnitude falls outside
  ///         `[min_field_t, max_field_t]`. A rejected sample leaves the
  ///         accumulator bit-unchanged.
  bool addSample(const math::Vec3<math::frames::Body>& m_raw, double igrf_magnitude_t);

  /// Samples accepted since the last @ref reset.
  std::int32_t sampleCount() const { return count_; }

  /// Orientation coverage of the samples so far — see
  /// @ref MagCalibrationResult::coverage for the definition and the geometry.
  /// Zero with no samples.
  ///
  /// Available **during** collection, not only after a solve, because it is the
  /// number that says whether the window can still succeed: a ground operator
  /// watching coverage climb knows when the vehicle has tumbled far enough,
  /// where a refused solve at the end of the window only says it had not. Costs
  /// one 3×3 symmetric eigen-decomposition per call.
  double coverage() const;

  /// Solve for the calibration, writing @p out.
  ///
  /// Const: solving does not consume the accumulator, so a component may solve
  /// at intervals through a long window and keep collecting.
  ///
  /// @param out result; left default-constructed (`valid == false`) on any
  ///            refusal, so nothing half-written is ever published
  /// @return `true` iff a usable calibration was produced. Refusals: an
  ///         unconfigured accumulator, fewer than `min_samples`, coverage below
  ///         `min_coverage`, normal-matrix condition above `max_condition`, an
  ///         eigen-decomposition that did not converge, a fitted quadric `A`
  ///         that is not positive definite (noise or coverage, never a field),
  ///         a calibrated residual that fails `min_residual_improvement`
  ///         against the uncalibrated one, or a non-finite result.
  bool solve(MagCalibrationResult& out) const;

  /// As @ref solve, additionally naming the gate that closed in @p why (set to
  /// @ref MagCalibrationRefusal::None on success). A refusal an operator can act
  /// on has to say *which* refusal it was — see @ref MagCalibrationRefusal.
  bool solve(MagCalibrationResult& out, MagCalibrationRefusal& why) const;

 private:
  MagCalibrationConfig cfg_{};
  /// `Σ hₖhₖᵀ`, non-dimensionalised by `nominal_field_t` [dimensionless].
  Eigen::Matrix<double, kParameters, kParameters> normal_{
      Eigen::Matrix<double, kParameters, kParameters>::Zero()};
  /// `Σ hₖfₖ²` [dimensionless].
  Eigen::Matrix<double, kParameters, 1> rhs_{Eigen::Matrix<double, kParameters, 1>::Zero()};
  /// `Σ m̂ₖm̂ₖᵀ` over unit measurement directions — the coverage moment.
  Eigen::Matrix3d directions_{Eigen::Matrix3d::Zero()};
  double sum_f2_squared_{0.0};  ///< `Σ fₖ⁴`, closing the residual sum of squares
  double sum_field_t_{0.0};     ///< `Σ Fₖ` [T], for the mean field
  std::int32_t count_{0};
  bool configured_{false};
};

/// Apply a calibration to one raw reading:
/// `soft_iron_inverse · (m_raw − hard_iron_offset)` [T, Body].
///
/// Returns @p m_raw unchanged when @p cal is not valid, so a magnetometer
/// processing path can call this unconditionally and an un-calibrated vehicle
/// simply passes through.
math::Vec3<math::frames::Body> applyMagCalibration(const MagCalibrationResult& cal,
                                                   const math::Vec3<math::frames::Body>& m_raw);

}  // namespace polaris::gnc

#endif  // POLARIS_GNC_MAG_CALIBRATION_HPP
