#ifndef POLARIS_GNC_DIPOLE_ESTIMATION_HPP
#define POLARIS_GNC_DIPOLE_ESTIMATION_HPP

/// @file
/// @brief Tier-3 residual-dipole estimation: fitting \f$\mathbf m_{res}\f$ from
/// the tier-2 observer's unmodelled-torque estimate (design doc §8.5 tier 3;
/// supports REQ-ACTL-011).
///
/// **What tier 3 is for.** Tier 1 (@ref residualDipoleTorque) feeds forward
/// \f$\mathbf m_{res}\times\mathbf B\f$ using `residual_dipole_am2` from config —
/// a **magnetic-cleanliness allocation**, i.e. the largest moment the build is
/// allowed to have, not a measurement of the moment it has. Tier 3 replaces that
/// allocation with a fit against flight data. It is the attitude-side half of
/// §8.5 tier 3; the **drag scale factor** is the orbit-side half and shipped with
/// Push 76 (@ref polaris::gnc::OrbitOd), where it reached the same verdict this
/// one did — its signal buried under the noise the filter already budgets, so it
/// ships tested and disabled with the sigma that says why. An SRP scale factor
/// stays owed and needs an onboard SRP term before it can have a coefficient.
///
/// **The estimation problem is linear.** The tier-2 observer
/// (@ref DisturbanceObserver) publishes \f$\hat{\boldsymbol\tau}\f$, the
/// unmodelled external torque. Its dipole part is \f$\boldsymbol\tau =
/// \mathbf m\times\mathbf B = -[\mathbf B\times]\,\mathbf m\f$, which is linear
/// in the unknown \f$\mathbf m\f$, so each accepted cycle is one three-row linear
/// measurement \f$\boldsymbol\tau = H\mathbf m\f$ with \f$H = -[\mathbf
/// B\times]\f$. This is the standard on-orbit dipole identification of Inamori,
/// Sako & Nakasuka [inamori2011], solved recursively rather than in batch.
///
/// **\f$H\f$ is rank 2, always.** \f$[\mathbf B\times]\f$ annihilates
/// \f$\mathbf B\f$, so the component of \f$\mathbf m\f$ **along the field** is
/// unobservable at every instant — the same structural fact that makes the
/// magnetorquers instantaneously rank 4 of 6 (§8.5, REQ-ACTL-007). The three
/// components separate only as the field direction *turns* in body axes, which
/// for a LEO vehicle happens over the orbit. Accumulating
/// \f[
///   N = \sum_k w_k H_k^\top H_k, \qquad
///   \mathbf g = \sum_k w_k H_k^\top \boldsymbol\tau_k, \qquad
///   \hat{\mathbf m} = N^{-1}\mathbf g
/// \f]
/// is therefore an *information-form* recursive least squares [ljung1999 §11.2]
/// in which \f$N\f$ is **singular** until the geometry has moved, and the
/// estimator's first duty is to say so rather than to invert it. Storage is
/// O(1) in the number of samples (a 3×3 and a 3-vector), like
/// @ref MagCalibrationAccumulator.
///
/// **Two gates, measuring two different things** (both reported, and
/// @ref DipoleRefusal names which one closed):
///  * @ref DipoleEstimatorConfig::min_observability on
///    \f$\lambda_{\min}(N)/\lambda_{\max}(N)\f$ — pure *geometry*: has the field
///    turned? Exactly the information-matrix ratio the flight Davenport seed
///    gates on (`SeedMinObservability`) and `analysis/control/observability.py`
///    sweeps, so tier 3 and the rest of the vehicle measure observability with
///    one function. A rank-2 \f$N\f$ scores 0, and publishing from one would be
///    fabricating the third component out of the pseudo-inverse's null space.
///  * @ref DipoleEstimatorConfig::min_information on \f$\lambda_{\min}(N)\f$
///    itself — *evidence*: a perfect ratio from three samples is still a fit on
///    three samples. Because the fit is preconditioned by
///    @ref DipoleEstimatorConfig::nominal_field_t, \f$\lambda_{\min}\f$ is a
///    dimensionless count of effective samples along the worst-determined axis,
///    and the gate is equivalently a bound on the published sigma:
///    \f$\sigma_{worst} \le \sigma_\tau/(B_{nom}\sqrt{\lambda_{\min}})\f$.
///
/// **The observer's estimate is not pure dipole, and no covariance says so.**
/// \f$\hat{\boldsymbol\tau}\f$ carries *every* unmodelled torque — aero CP–CM
/// offset, SRP, wheel bearing friction, an unlatched deployment — and this fit
/// will assign to \f$\mathbf m\f$ whatever part of them correlates with
/// \f$\mathbf B\f$ over the memory window. @ref DipoleResult::sigma_am2 is the
/// propagation of the observer's *random* error only; the non-dipole content is
/// a **bias**, and a bias is invisible to a least-squares covariance by
/// construction. Two consequences, both deliberate:
///  * The estimator does **not** try to gate accumulation on "regimes where the
///    dipole dominates". There is no onboard test for that — knowing the aero
///    and SRP torques well enough to declare them subdominant would mean already
///    having the tier-3 orbit-side fit this push does not have, and a gate built
///    on the tier-1 model would only be checking the model against itself.
///  * What protects the vehicle instead is the bound below, which is a check
///    against an *independently established* fact (the cleanliness budget)
///    rather than against the estimator's own opinion of itself.
///
/// **It must not be able to make pointing worse** — the estimate feeds tier-1
/// feedforward, so a wrong fit is an injected disturbance. `residual_dipole_am2`
/// is a magnetic-cleanliness **bound** established on the ground; a fit outside
/// it indicts the fit, not the vehicle. @ref DipoleEstimatorConfig::max_dipole_am2
/// carries that bound (with margin) and a solution outside it is **refused**, not
/// clamped: clamping would publish a direction the data never supported at a
/// magnitude the policy chose. The refusal is deliberately **not latched** — it is
/// re-evaluated from scratch every cycle, so the criterion that ends it is
/// bit-identical to the criterion that caused it (`‖m̂‖ ≤ max_dipole_am2`), and a
/// window that drifts back inside the bound publishes again with no operator
/// action. An estimator has no business writing life sentences.
///
/// **Sample cadence is part of the estimator, not the caller's business.** The
/// observer's output is a first-order low pass of time constant \f$\tau_{lp}\f$
/// (200 s on the reference vehicle), so successive 10 Hz reads of it are the same
/// filtered value seen 2000 times. Accumulating them would inflate \f$N\f$ by
/// three orders of magnitude and publish a sigma three orders too small —
/// counting correlated reads as independent evidence is the most direct way an
/// estimator can lie about itself. @ref DipoleEstimatorConfig::min_sample_interval_s
/// therefore rejects samples arriving faster than the observer decorrelates
/// (\f$\ge 2\tau_{lp}\f$), and the reference vehicle feeds this once per 400 s.
/// Note that the achievable precision is then **independent of
/// \f$\tau_{lp}\f$**: the filter's noise falls as \f$1/\sqrt{\tau_{lp}}\f$ while
/// the admissible cadence grows as \f$\tau_{lp}\f$, and the two cancel exactly.
/// Retuning the observer to help tier 3 would buy nothing; what precision does
/// depend on is below.
///
/// **Forgetting.** The physical moment moves with the power state (a payload
/// duty cycle re-routes current, and current loops are the moment), so the
/// accumulator forgets exponentially in *time* rather than in samples:
/// \f$w = e^{-\Delta t/T_f}\f$ [ljung1999 §11.2]. Time-based rather than
/// sample-based forgetting means a dropout ages the information correctly
/// instead of freezing it, and a clock jump forward simply discards a window it
/// has no reason to trust. Two bounds fence \f$T_f\f$ in, and it is worth being
/// exact about which one binds: the field must turn appreciably *within* the
/// memory window (it turns twice per orbit, so this bounds \f$T_f\f$ below at the
/// orbit scale — but at a 400 s cadence one sample interval already turns the
/// field ~50°, so on this vehicle the binding lower bound is instead simply
/// having enough samples in the window), and above, the drift it must track. The
/// reference vehicle uses 3 orbits (17031 s).
///
/// **Precision saturates at the forgetting horizon, not at the pass length** —
/// the consequence most easily got wrong by analogy with a growing-memory least
/// squares. \f$N\f$ reaches a steady state at \f$\approx T_f/\Delta t\f$
/// effective samples, so a longer pass buys **nothing** and the sigma floor is a
/// function of \f$T_f\f$ alone:
/// \f$\sigma_{floor}\approx\sigma_\tau/(B_{nom}\sqrt{0.3\,T_f/\Delta t})\f$.
/// Measured (`tests/unit/dipole_estimation_test.cpp`): 3.1e-2 A·m² at
/// \f$T_f\f$ = 3 orbits, unchanged over a pass four times longer, and 9.7e-3
/// A·m² at 30 orbits. The ground's only lever on precision is therefore the
/// forgetting time, traded directly against drift tracking — which is why a
/// *commanded calibration pass*, during which the power state is held, should run
/// a longer \f$T_f\f$ than the drift-tracking default.
///
/// **What that means for this vehicle, measured.** The reference *allocation* is
/// `[0.002, -0.001, 0.0015]` A·m², an ~8e-8 N·m signature against a 3.2e-6 N·m
/// observer floor: an SNR of 0.025 per sample, and a published sigma (1.4e-2 to
/// 2.1e-2 A·m² after a day) an order of magnitude larger than the dipole itself.
/// **Tier 3 cannot refine the feedforward on a magnetically clean vehicle, and
/// the sigma says so rather than the estimator pretending otherwise.** What it
/// does resolve is a dipole of the size the §9 anomaly monitor exists for
/// (`DisturbanceBudgetNm` = 2e-5 N·m ≈ 0.7 A·m² at LEO field strength): 10 % on
/// all three axes in **5200 s — 1.4 h, under one orbit** — at
/// \f$T_f\f$ = 30 orbits. That is the calibration-pass length to give the ground.
/// Tier 3's honest role here is therefore **diagnosis**: when the momentum
/// anomaly latches, this names the offending moment in body axes with a sigma.
/// Feedforward refinement is what it does on a vehicle whose dipole is large
/// enough to be worth feeding forward. The estimator is the same either way; the
/// gates and the sigma are what keep the distinction from being a matter of
/// opinion.
///
/// With no observer noise at all the remaining cost is pure geometry — the field
/// has to turn before the third component exists — and that floor is **2800 s,
/// half an orbit**, whatever the dipole is.
///
/// **Frames, units, conventions.** Torque `Vec3<Body>` in N·m, field
/// `Vec3<Body>` in tesla, dipole and its sigma `Vec3<Body>` in A·m²; time tags
/// TAI nanoseconds (§3.2). SI throughout.
///
/// **Flight path.** Fixed-size Eigen (3×3 and 3-vectors), no heap, no
/// exceptions, no recursion, no unbounded loops — the symmetric eigenproblem uses
/// Eigen's closed-form `computeDirect` for 3×3, so the work per call is fixed —
/// return codes checked, finiteness guards on every published output. No F´ types
/// and no I/O.
///
/// References:
///  - Inamori, Sako & Nakasuka, "Magnetic dipole moment estimation and
///    compensation for an accurate attitude control in nano-satellite missions",
///    *Acta Astronautica* 68(9-10):2038-2046, 2011 — on-orbit identification of
///    the residual moment from the `m×B` signature. [inamori2011]
///  - Ljung, *System Identification: Theory for the User*, 2nd ed., 1999, §11.2 —
///    recursive least squares in information form with exponential forgetting,
///    and the excitation condition its convergence needs. [ljung1999]
///  - Markley & Crassidis, *Fundamentals of Spacecraft Attitude Determination
///    and Control*, 2014, §3.2.4 (magnetic disturbance torque) and §7.5
///    (magnetic control). [markley2014]
///  - Wertz (ed.), *Spacecraft Attitude Determination and Control*, 1978, §17.2
///    (environmental torques, residual dipole among them). [wertz1978]
///  - Design doc §8.5 (the three feedforward tiers), §9 (the momentum-anomaly
///    monitor this diagnoses).

#include <cstdint>
#include <Eigen/Core>

#include "math/frames.hpp"
#include "math/typed_vector.hpp"

namespace polaris::gnc {

/// Why a @ref DipoleEstimator::update published no estimate.
///
/// A refusal is only actionable if it names its gate: `kNoObservability` means
/// "the field has not turned — keep collecting, the geometry is coming",
/// `kInsufficientInformation` means "keep collecting, the noise is not down
/// yet", and `kOutOfBounds` means "stop: this fit disagrees with the vehicle's
/// cleanliness budget and one of the two is wrong".
enum class DipoleRefusal : std::uint8_t {
  kNone = 0,          ///< an estimate was published
  kUnconfigured,      ///< config failed @ref DipoleEstimatorConfig::isValid
  kBadInput,          ///< non-finite torque, field or a non-finite accumulator update
  kFieldOutOfRange,   ///< `|B|` outside the configured band; sample not accumulated
  kSampleTooSoon,     ///< arrived inside `min_sample_interval_s`; sample not accumulated
  kNonMonotonicTime,  ///< the time tag did not advance; sample not accumulated
  kNoObservability,   ///< `λ_min/λ_max` below `min_observability` — the field has not turned
  kInsufficientInformation,  ///< `λ_min` below `min_information` — not enough evidence yet
  kNumerical,                ///< the eigenproblem or the solve failed, or the result was non-finite
  kOutOfBounds,              ///< the fit exceeded `max_dipole_am2`
};

/// Tuning for @ref DipoleEstimator. Mission configuration (§19.3): no in-code
/// defaults, and a default-constructed config deliberately fails @ref isValid.
struct DipoleEstimatorConfig {
  /// Field magnitude the fit is non-dimensionalised by [T], e.g. 3e-5 in LEO.
  ///
  /// **Numerics only.** The normal equations are exactly diagonally
  /// preconditioned by it, so \f$\hat{\mathbf m}\f$ is invariant to the choice up
  /// to round-off; without it \f$N\f$ would carry entries near `1e-9` and
  /// \f$\lambda_{\min}\f$ would be a threshold nobody could set by inspection.
  /// With it, \f$\lambda_{\min}\f$ reads as an effective sample count. Any value
  /// within a factor of a few of the true field works.
  double nominal_field_t = 0.0;

  /// Smallest accepted `|B|` [T]. Rejects a dead sensor and a field estimate
  /// that has collapsed.
  double min_field_t = 0.0;

  /// Largest accepted `|B|` [T]. Rejects saturation and — the case that matters
  /// here — a magnetorquer near-field, which is hundreds of microtesla and would
  /// enter the fit as a field the vehicle's own dipole never saw (§7, and the
  /// review lesson that the vehicle's own actuators are a permanent signal in its
  /// own sensors). The IGRF band `[2.2e-5, 5.2e-5]` T that `mag_voting` uses is
  /// the natural setting.
  double max_field_t = 0.0;

  /// Exponential forgetting time constant \f$T_f\f$ [s]: a sample's weight
  /// decays as \f$e^{-\Delta t/T_f}\f$. Long enough that the field turns
  /// appreciably inside the window (so bound below by the orbit period), short
  /// enough to track power-state drift of the physical moment.
  double forgetting_time_s = 0.0;

  /// Shortest interval [s] between accumulated samples. Set at or above twice
  /// the tier-2 observer's low-pass time constant, so consecutive accumulations
  /// carry independent noise — see the file header on why a faster feed makes the
  /// estimator lie about its own sigma.
  double min_sample_interval_s = 0.0;

  /// Smallest accepted \f$\lambda_{\min}(N)/\lambda_{\max}(N)\f$
  /// [dimensionless, `(0, 1]`]. The geometry gate: zero for a field that has not
  /// turned, and 1 for information spread isotropically. The same ratio the
  /// flight Davenport seed gate and `analysis/control/observability.py` use.
  double min_observability = 0.0;

  /// Smallest accepted \f$\lambda_{\min}(N)\f$ [dimensionless, `> 0`]. The
  /// evidence gate, readable as an effective sample count on the worst-determined
  /// axis, and equivalently a bound on the published sigma:
  /// \f$\sigma_{worst}\le\sigma_\tau/(B_{nom}\sqrt{\lambda_{\min}})\f$.
  double min_information = 0.0;

  /// 1σ of the tier-2 observer's torque estimate [N·m] — its *filtered* noise
  /// floor, not the raw difference quotient's. Sets @ref DipoleResult::sigma_am2
  /// and nothing else; it does not gate.
  double torque_sigma_nm = 0.0;

  /// Largest \f$\|\hat{\mathbf m}\|\f$ [A·m²] that may be published. The
  /// vehicle's magnetic-cleanliness allocation with margin — the fit is refused
  /// outside it rather than clamped into it. A fit that keeps hitting this bound
  /// is evidence about the *fit* (an unmodelled torque correlating with `B`) or
  /// about the *build* (the allocation was violated), and either way it is the
  /// ground's call, not a saturation the flight software makes quietly.
  double max_dipole_am2 = 0.0;

  /// Every field finite and positive, `min_field_t < max_field_t`, and
  /// `min_observability` in `(0, 1]`.
  bool isValid() const;
};

/// One @ref DipoleEstimator::update's product.
struct DipoleResult {
  /// Fitted body-fixed residual moment \f$\hat{\mathbf m}\f$ [A·m², Body].
  math::Vec3<math::frames::Body> dipole_am2{Eigen::Vector3d::Zero()};

  /// Per-axis 1σ of @ref dipole_am2 [A·m², Body], \f$\sigma_\tau/B_{nom}\cdot
  /// \sqrt{\mathrm{diag}(N^{-1})}\f$.
  ///
  /// **The observer's random error only.** Under exponential forgetting this is
  /// conservative by roughly \f$\sqrt2\f$ (the standard RLS convention drops the
  /// \f$\sum w^2/\sum w\f$ factor, which is \f$\approx 1/2\f$), which is the safe
  /// direction. What it does **not** contain is the non-dipole torque the fit
  /// absorbs — that is a bias, and no least-squares covariance sees a bias. Read
  /// the file header before treating this as an accuracy statement.
  math::Vec3<math::frames::Body> sigma_am2{Eigen::Vector3d::Zero()};

  /// \f$\lambda_{\min}(N)/\lambda_{\max}(N)\f$ [dimensionless]; the geometry
  /// gate's quantity. Reported on every cycle, including refusals, because
  /// watching it climb is how the ground knows a calibration pass is working.
  double observability = 0.0;

  /// \f$\lambda_{\min}(N)\f$ [dimensionless]; the evidence gate's quantity.
  double information = 0.0;

  /// \f$\sum_k w_k\f$, the forgetting-weighted sample count [dimensionless].
  /// Diagnostic: it saturates at \f$\approx T_f/\Delta t\f$, which is what says
  /// the memory window is full.
  double effective_samples = 0.0;

  /// @ref dipole_am2 and @ref sigma_am2 are usable.
  bool valid = false;

  /// Why not, when @ref valid is false.
  DipoleRefusal refusal = DipoleRefusal::kUnconfigured;
};

/// Recursive least-squares residual-dipole estimator (design doc §8.5 tier 3,
/// attitude side).
///
/// Hold one instance, feed it the tier-2 observer's estimate paired with the
/// onboard field, and read @ref estimate when @ref hasEstimate. It is a
/// *fitter*, not a controller: it publishes or refuses, and never modifies
/// anything.
class DipoleEstimator {
 public:
  DipoleEstimator() = default;

  /// Build with @p config. An invalid config leaves the estimator **inert** —
  /// every @ref update returns @ref DipoleRefusal::kUnconfigured.
  explicit DipoleEstimator(const DipoleEstimatorConfig& config);

  bool isConfigured() const { return configured_; }

  /// Fold in one (torque, field) pair and re-solve.
  ///
  /// @param observer_torque_nm the tier-2 observer's unmodelled-torque estimate
  ///        \f$\hat{\boldsymbol\tau}\f$ [N·m, Body]. Pass the estimate the
  ///        observer published, *not* its raw residual: the sigma model assumes
  ///        the filtered value, and `min_sample_interval_s` assumes its
  ///        correlation time.
  /// @param field_tesla the onboard field in body axes [T]. Must be the *same*
  ///        field the tier-1 feedforward uses, since the two are the same
  ///        physical quantity in the same equation.
  /// @param time_tag_tai_ns TAI ns of this sample.
  /// @param out receives the estimate and the diagnostics. @ref
  ///        DipoleResult::observability, @ref DipoleResult::information and
  ///        @ref DipoleResult::effective_samples are filled on gate refusals too.
  /// @return true when @p out carries a usable estimate. On every refusal the
  ///         running estimate is **held**, never zeroed — a refused cycle is not
  ///         evidence the dipole went away.
  bool update(const math::Vec3<math::frames::Body>& observer_torque_nm,
              const math::Vec3<math::frames::Body>& field_tesla, std::int64_t time_tag_tai_ns,
              DipoleResult& out);

  /// The last published fit [A·m², Body]. Zero until @ref hasEstimate.
  const math::Vec3<math::frames::Body>& estimate() const { return estimate_; }

  /// Per-axis 1σ of @ref estimate [A·m², Body]. Zero until @ref hasEstimate.
  const math::Vec3<math::frames::Body>& sigma() const { return sigma_; }

  /// At least one fit has passed every gate. Until then @ref estimate is a zero
  /// meaning *unknown*, and a caller must keep using the configured allocation
  /// rather than feeding forward a zero dipole.
  bool hasEstimate() const { return have_estimate_; }

  /// Drop the accumulator, the published estimate and the sample anchor (mode
  /// entry, a commanded restart of a calibration pass, or a tuning change that
  /// invalidates the accumulated information). Configuration is retained.
  void reset();

 private:
  /// Solve the accumulated system into @p out, running every gate. Does not
  /// modify the accumulator.
  bool solveInto(DipoleResult& out) const;

  DipoleEstimatorConfig config_{};
  bool configured_ = false;
  bool have_previous_ = false;
  bool have_estimate_ = false;
  /// \f$\sum w_k H_k^\top H_k\f$, preconditioned by `nominal_field_t`
  /// [dimensionless].
  Eigen::Matrix3d information_{Eigen::Matrix3d::Zero()};
  /// \f$\sum w_k H_k^\top\boldsymbol\tau_k / B_{nom}^2\f$ [A·m²].
  Eigen::Vector3d rhs_{Eigen::Vector3d::Zero()};
  double effective_samples_ = 0.0;
  std::int64_t previous_time_ns_ = 0;
  math::Vec3<math::frames::Body> estimate_{Eigen::Vector3d::Zero()};
  math::Vec3<math::frames::Body> sigma_{Eigen::Vector3d::Zero()};
};

}  // namespace polaris::gnc

#endif  // POLARIS_GNC_DIPOLE_ESTIMATION_HPP
