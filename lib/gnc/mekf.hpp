#ifndef POLARIS_GNC_MEKF_HPP
#define POLARIS_GNC_MEKF_HPP

/// @file
/// @brief Fine-mode attitude MEKF — 6-state multiplicative EKF, attitude error
/// + gyro bias (design doc §8.1; REQ-ADET-001, REQ-ADET-004).
///
/// The **fine mode**: where @ref CoarseAttitudeEstimator buys always-available
/// at the price of optimality, this is the optimal fusion the coarse mode
/// deliberately is not. Gyros carry the attitude between measurements, vector
/// measurements correct it, and the gyro **bias** — the term that makes an
/// unaided gyro useless within minutes — is estimated alongside it.
///
/// **Multiplicative, and why.** A 3-DOF rotation cannot carry a full-rank 4×4
/// covariance: the unit-norm constraint makes it singular. So the reference
/// attitude stays a unit quaternion (JPL scalar-first, `q0 ≥ 0`, never leaves
/// SO(3), no singularity) and the covariance lives on a **minimal 3-parameter
/// local error** `δθ` that is composed onto the quaternion multiplicatively and
/// **reset to zero after every update**. That is the entire content of
/// "multiplicative" EKF (Lefferts, Markley & Shuster 1982 [lefferts1982]) — the
/// error parameterisation is a linearisation detail local to one step, not a
/// second attitude representation with its own singularity.
///
/// **Error state** (6): `x = [δθ; δb]`, body-frame attitude error [rad] and
/// gyro-bias error [rad/s], laid out in the same order as
/// `state::ErrorState::kAttitude` / `kGyroBias`, so the 15×15 canonical
/// covariance takes this filter's blocks verbatim.
///
/// **Convention.** `δθ` is defined by `q_true = δq ⊗ q̂` with
/// `δq ≈ [1, δθ/2]`, equivalently `A(q_true) = (I − [δθ×]) A(q̂)`, and
/// `δb = b_true − b̂`. Under the gyro model `ω_meas = ω_true + b + n_v`,
/// `ḃ = n_u`, the error dynamics are
/// \f[
///   \dot{\delta\boldsymbol\theta} = -[\hat{\boldsymbol\omega}\times]\,\delta\boldsymbol\theta
///     - \delta\mathbf b - \mathbf n_v, \qquad \dot{\delta\mathbf b} = \mathbf n_u ,
/// \f]
/// with \f$\hat{\boldsymbol\omega} = \boldsymbol\omega_\mathrm{meas} - \hat{\mathbf b}\f$
/// (Markley & Crassidis §6.2.4 [markley2014]; Trawny & Roumeliotis §3
/// [trawny2005]). Note `δθ` here has the opposite sign to the convention
/// @ref triad states for its covariance — which is why a TRIAD or
/// @ref davenport solution seeds this filter directly: a covariance is
/// unchanged by `δθ → −δθ`.
///
/// **Discrete propagation.** The quaternion is advanced by the same closed-form
/// constant-rate kinematics the coarse estimator and the truth plant use. The
/// state transition is the exact
/// \f$\Phi = \begin{bmatrix}\Phi_{11} & \Phi_{12}\\ 0 & I\end{bmatrix}\f$ with
/// \f$\Phi_{11} = \exp(-[\hat{\boldsymbol\omega}\times]\Delta t)\f$ (the
/// propagation increment's own attitude matrix) and
/// \f$\Phi_{12} = -\int_0^{\Delta t}\exp(-[\hat{\boldsymbol\omega}\times]s)\,ds\f$
/// in closed form — the cross term is what lets a bias error show up as an
/// attitude error, i.e. what makes the bias observable at all. On a step where
/// the gyro was **not** used, \f$\Phi_{12}\f$ is **zero** rather than
/// \f$-\Delta t\,I\f$: the bias estimate never entered that propagation, so it
/// caused none of the attitude error accumulated over it, and carrying the
/// coupling anyway would invent a correlation the filter would later "correct"
/// by dragging the bias.
///
/// The discrete process noise keeps its **cross-coupling blocks** for the same
/// reason (Farrenkopf 1978 [farrenkopf1978]; Markley & Crassidis §6.2.4, the
/// Eq. 6.93 form):
/// \f[
///   Q_d = \begin{bmatrix}
///     (\sigma_v^2 \Delta t + \tfrac{1}{3}\sigma_u^2 \Delta t^3) I
///       & -\tfrac{1}{2}\sigma_u^2 \Delta t^2 I
///     \\ -\tfrac{1}{2}\sigma_u^2 \Delta t^2 I & \sigma_u^2 \Delta t\, I
///   \end{bmatrix} .
/// \f]
/// A diagonal shortcut would drop the `Δt²` term, which is exactly the
/// correlation that tells the filter an attitude error and a bias error of the
/// right sign are the *same* error seen twice — without it the bias converges
/// slowly or not at all. This form assumes `|ω|Δt ≪ 1` (rotation of the noise
/// axes over one step is neglected); at the 10 Hz GNC rate (§2.4) and any rate
/// a controlled vehicle sees, that is exact to well below the noise itself.
///
/// **Update.** Vector measurements are processed **one at a time** — sun, then
/// magnetic field, then any further pair — so N-sensor fusion (§8.2) is just
/// more calls, with no interface change and no growing matrix. Each is a unit
/// direction with sensitivity `H = [[b̂_pred ×] 0₃]` where
/// `b̂_pred = A(q̂) r̂`, a **Joseph-form** covariance update (which stays
/// symmetric positive-definite under round-off and a suboptimal gain, unlike
/// `(I−KH)P`), then the multiplicative reset: `q̂⁺ = δq̂ ⊗ q̂`,
/// `b̂⁺ = b̂ + δb̂`, error state back to zero.
///
/// **Attitude measurements** — a star tracker's complete Body ← ECI solution —
/// go through @ref Mekf::updateAttitude instead, with `H = [I₃ 0₃]`, a full 3×3
/// `R` (a tracker's error is strongly anisotropic about its boresight, and
/// carrying that is what makes two non-parallel trackers worth more than two
/// parallel ones), and its own χ²₃ gate. Same Joseph form, same multiplicative
/// reset, same rejection accounting; only the sensitivity, the innovation
/// reduction and the degrees of freedom differ.
///
/// **Measurement noise is the caller's, and white.** `sigma_rad` is passed per
/// update, giving `R = σ²I`. Two things follow that a caller must know.
/// *First*, the isotropic `σ²I` rather than the rank-2 `σ²(I − b̂b̂ᵀ)` of the
/// QUEST model: the transverse form makes `S = HPHᵀ + R` singular along `b̂`
/// (H's null space is the same direction), and the along-vector component of
/// the innovation carries no attitude information either way, so the isotropic
/// form is the standard and harmless regularisation (Markley & Crassidis
/// §6.2.3). *Second*, and this one bites: **the filter treats R as white.**
/// Fusing a systematic error — analytic-ephemeris bias, IGRF model error,
/// sensor misalignment (§8.1, and the split @ref CoarseAttitudeConfig makes
/// explicit) — as if it were white noise makes the filter **overconfident**: it
/// averages down an offset that does not average down, and the covariance it
/// reports converges below the true error. The caller inflates `sigma_rad` to
/// cover its systematic budget (root-sum-square is the usual choice); this
/// filter has no way to tell the two apart and does not pretend to.
///
/// **Consistency (REQ-ADET-004).** Every update reports its innovation, the
/// innovation covariance `S`, and the **NIS** `yᵀS⁻¹y`. A measurement whose NIS
/// exceeds `MekfConfig::nis_gate` is **rejected** — not applied, counted in
/// @ref Mekf::rejectedCount — which is the divergence guard: an outlier that
/// would drag the reference attitude off is refused, and a *persistent* stream
/// of rejections is the signal FDIR acts on. @ref Mekf::nees computes the
/// 6-state NEES against a known truth for analysis and Monte-Carlo consistency
/// testing (Bar-Shalom §5.4 [barshalom2001]); it has no onboard use, since
/// onboard there is no truth.
///
/// **Refusal, not assertion.** Unconfigured, uninitialised, `dt ≤ 0` (backwards
/// *or* stuck clock — re-running an update at the same epoch would fold the same
/// measurement in twice), non-finite or zero-length inputs, a non-positive σ, a
/// failed matrix inverse: all return `false` and leave the solution untouched.
/// A non-finite *internal* result is unrecoverable, so it drops the filter to
/// cold start (leaving @ref Mekf::rejectedCount standing — a filter that just
/// diverged is when FDIR most needs to see what it had been rejecting; only a
/// commanded @ref Mekf::reset clears it). Nothing asserts on measurement
/// content. Past `max_coast_s` without an accepted update the attitude reads
/// **invalid** — but unlike the coarse mode the solution is *not* dropped,
/// because a Kalman gain against a grown covariance already takes the returning
/// measurement almost whole, so there is nothing to gain by throwing away a
/// converged bias estimate. That retention is **internal**: past the horizon
/// @ref writeToEstimatedState flags `gyro_bias` invalid along with the
/// attitude, so a future arbitration layer cannot reach the retained bias
/// through `EstimatedState` and must go through this object.
///
/// **Cold start.** @ref Mekf::initialize is seeded from @ref davenport (or
/// TRIAD — same interface, `Quat<Body,ECI>` plus its `δθ` covariance). Per the
/// project decision, the deterministic single-frame solvers are
/// **cold-start/re-init only** and never run in the steady-state loop.
///
/// **Frames, units, conventions.** Attitude `Quat<Body, ECI>`, rates and biases
/// `Vec3<Body>` [rad/s], times TAI (§3.2), covariance rad² / rad²s⁻² blocks,
/// all SI. Flight path: fixed-size Eigen, no heap, no exceptions, no recursion,
/// bounded loops, return codes checked, finiteness-guarded output. No F´ types
/// and no I/O.
///
/// References:
///  - Lefferts, Markley & Shuster, "Kalman Filtering for Spacecraft Attitude
///    Estimation", J. Guidance, Control & Dynamics 5(5):417-429, 1982 (the
///    MEKF). [lefferts1982]
///  - Markley & Crassidis, *Fundamentals of Spacecraft Attitude Determination
///    and Control*, 2014, §6.2.4 (MEKF: error dynamics, discrete Φ and Qd,
///    vector-measurement sensitivity, multiplicative reset). [markley2014]
///  - Farrenkopf, "Analytic Steady-State Accuracy Solutions for Two Common
///    Spacecraft Attitude Estimators", J. Guidance & Control 1(4):282-284, 1978
///    (the ARW/RRW gyro noise model). [farrenkopf1978]
///  - Bar-Shalom, Li & Kirubarajan, *Estimation with Applications to Tracking
///    and Navigation*, 2001, §5.4 (NEES/NIS consistency tests). [barshalom2001]
///  - Trawny & Roumeliotis, "Indirect Kalman Filter for 3D Attitude
///    Estimation", UMN MARS Lab TR 2005-002 (JPL-convention error state and
///    reset). [trawny2005]
///  - Carpenter & D'Souza (eds.), *Navigation Filter Best Practices*,
///    NASA/TP-2018-219822, 2018 — Ch. 7 (definiteness check), §9.1 (the
///    accept/inhibit/force editing flag), §9.2 (covariance re-initialisation
///    without a state change), §9.3 (uplinkable tuning); and Dennehy &
///    Carpenter, NESC Technical Bulletin 20-03, 2020, items (d), (f), (g).
///    [carpenter2018, dennehy2020]; and, from the same TP, §3.1 (measurement
///    latency), §3.2 / Algorithm 3.1 (order-invariant same-epoch update), §5.2.4
///    (first-order Gauss-Markov bias), Ch. 8 Eq. 8.76 (Reynolds covariance
///    reset). Reynolds, "Asymptotically Optimal Attitude Filtering with
///    Guaranteed Convergence", JGCD 31(1), 2008. [reynolds2008]

#include <cstdint>
#include <Eigen/Core>

#include "gnc/measurement_policy.hpp"
#include "math/frames.hpp"
#include "math/quaternion.hpp"
#include "math/typed_vector.hpp"
#include "state/estimated_state.hpp"
#include "time/timescales.hpp"

namespace polaris::gnc {

/// Tuning for @ref Mekf. Every value is mission configuration (§19.3) — there
/// are no flight defaults worth trusting, so @ref isValid gates the constructor
/// and an invalid config leaves the filter permanently inert.
struct MekfConfig {
  /// Gyro **angle random walk** σ_v [rad·s^(-1/2)]: the attitude-error standard
  /// deviation accumulated per √second of propagation. Same parametrisation as
  /// `CoarseAttitudeConfig::gyro_arw`. Must be positive.
  double arw_rad_per_sqrt_s{0.0};
  /// Gyro **rate random walk** σ_u [rad·s^(-3/2)]: the bias standard deviation
  /// accumulated per √second. Datasheets more often quote a bias instability in
  /// °/h with a correlation time; convert to the equivalent random walk before
  /// setting this. May be zero (a bias modelled as exactly constant), in which
  /// case the bias covariance never grows and the filter will eventually stop
  /// learning it — rarely what real hardware wants.
  double rrw_rad_per_s_per_sqrt_s{0.0};
  /// NIS rejection threshold [dimensionless] for one 3-vector update. Compare
  /// against the chi-square quantile for **2** degrees of freedom, not 3: the
  /// innovation between two unit vectors is transverse by construction, so the
  /// third component carries no variance even though `R = σ²I` budgets for it
  /// (χ²₂ at 99.9% ≈ 13.8). Must be positive.
  double nis_gate{0.0};
  /// NIS rejection threshold for one **attitude** update (@ref
  /// Mekf::updateAttitude). A separate value from @ref nis_gate, and separate
  /// because the degrees of freedom differ: a star tracker's innovation is a full
  /// 3-DOF rotation with no degenerate direction, so this is χ²**₃** (99.9% ≈
  /// 16.27) where the vector gate is χ²₂. Reusing the vector gate would reject at
  /// the wrong tail probability in the tighter direction — a 2-DOF threshold
  /// applied to a 3-DOF statistic gates at ~99.9% → ~99.0%, i.e. ten times the
  /// false-rejection rate, on the one measurement source the fine mode is built
  /// around. Must be positive.
  double attitude_nis_gate{0.0};
  /// Longest interval without an accepted update before the attitude is
  /// declared invalid [s]. The filter state is kept — see the file header.
  double max_coast_s{0.0};
  /// Largest accepted propagation step [s]. A longer gap is treated as a
  /// dropout — covariance inflated, attitude held — rather than extrapolated on
  /// a stale rate.
  double max_dt_s{0.0};

  /// Gyro-bias correlation time τ [s] (NASA/TP-2018-219822 §5.2.4, Push 72).
  /// **Zero keeps the random-walk bias model** (`ḃ = n_u`), the flown default.
  /// Positive makes the bias a **first-order Gauss-Markov** process,
  /// `ḃ = −b/τ + n_u`: `Φ₂₂ = e^{−Δt/τ} I`, `Q₂₂ = σ_u² τ/2 (1 − e^{−2Δt/τ}) I`
  /// (which is `σ_u² Δt` for Δt ≪ τ, so the early-time evolution mimics the
  /// random walk exactly as the TP prescribes), and the bias *estimate* decays
  /// toward zero between measurements at the same rate. The TP recommends the
  /// FOGM "for applications in which there are measurements continually
  /// available to persistently excite it", because its variance is bounded —
  /// `σ_u² τ/2` — where the random walk's grows without limit through a coast;
  /// its §5.2.7 caveat is the other edge: a data outage long against τ decays
  /// a bias the vehicle still has. The reference vehicle's IMU model carries a
  /// **constant** turn-on bias under a τ = 100 s in-run drift, so the flown
  /// value stays 0 (see `config/spacecraft/leo_smallsat.yaml`); the option is
  /// here for a vehicle whose bias is the in-run drift alone. Must be finite
  /// and ≥ 0.
  double bias_tau_s{0.0};

  /// Apply the **Reynolds covariance reset** on every multiplicative reset
  /// (TP Eq. 8.76 [carpenter2018]; Reynolds 2008 [reynolds2008]):
  /// `P ← (I − [δθ̂×]/2) P (I − [δθ̂×]/2)ᵀ` on the attitude rows and columns.
  /// The reset changes the frame the attitude covariance is expressed in, and
  /// to first order that is a rotation by δθ̂/2; most applications omit it, but
  /// Reynolds found it speeds convergence and adds robustness on large updates
  /// — exactly the re-acquisition-after-coast case here — and that omitting it
  /// can lead to divergence. On by default; a plain flag because it is an
  /// algorithm choice, not a tuning.
  bool reynolds_reset{true};

  /// True when every field is finite and in range. Checked once at construction.
  bool isValid() const;
};

/// Diagnostics from one measurement update, vector or attitude. Populated
/// whether or not the measurement was accepted, so the NIS of a *rejected*
/// measurement is available to telemetry and FDIR.
struct MekfUpdate {
  /// Innovation, body frame: `y = b̂_meas − A(q̂)r̂` [dimensionless direction]
  /// from @ref Mekf::update, or `z = δθ` [rad] from @ref Mekf::updateAttitude.
  Eigen::Vector3d innovation{Eigen::Vector3d::Zero()};
  /// Innovation covariance `S = HPHᵀ + R`, body frame; units follow
  /// @ref innovation.
  Eigen::Matrix3d innovation_cov{Eigen::Matrix3d::Identity()};
  /// Normalised innovation squared `yᵀS⁻¹y` [dimensionless]; ~χ²₂ for a vector
  /// update and ~χ²₃ for an attitude update when the filter is consistent.
  double nis{0.0};
  /// The measurement passed the NIS gate and was applied.
  bool accepted{false};
  /// Applied **past** the gate under `force` (implies @ref accepted). Reported
  /// apart so a forced update is never read as a consistent one.
  bool forced{false};
};

/// 6-state multiplicative EKF for attitude and gyro bias (design doc §8.1).
///
/// One instance per vehicle. Seed it once with @ref initialize, then per cycle
/// call @ref propagate with the gyro and @ref update once per available vector
/// pair. Plain value with fixed-size storage; allocates nothing.
class Mekf {
 public:
  /// Error-state dimension.
  static constexpr int kDim = 6;
  /// Row/column of the attitude error `δθ` within @ref Covariance.
  static constexpr int kAttitude = 0;
  /// Row/column of the gyro-bias error `δb` within @ref Covariance.
  static constexpr int kGyroBias = 3;
  /// Error-state covariance, blocked `[δθ; δb]` — the same order and meaning as
  /// the corresponding blocks of `state::Covariance`.
  using Covariance = Eigen::Matrix<double, kDim, kDim>;

  /// Construct with @p config. If the config is invalid the filter is inert:
  /// every entry point returns false forever. Check @ref isConfigured.
  explicit Mekf(const MekfConfig& config);

  /// True when the configuration passed @ref MekfConfig::isValid.
  bool isConfigured() const { return configured_; }

  /// True once @ref initialize has succeeded and no unrecoverable fault has
  /// dropped the filter.
  bool isInitialised() const { return initialised_; }

  /// Drop the solution (cold start) on command. Configuration is retained and
  /// the rejected count **is** cleared — this is the operator saying "start
  /// over". An internal fault drops the solution without clearing that count.
  void reset();

  /// Seed the filter from a deterministic single-frame solution (@ref davenport
  /// or @ref triad) — the cold-start and re-init path.
  ///
  /// @param epoch         TAI time tag of @p attitude (§3.2)
  /// @param attitude      Body ← ECI, need not be exactly unit norm
  /// @param attitude_cov  `E[δθ δθᵀ]` [rad²], body axes — take it straight from
  ///                      the initializer's covariance
  /// @param gyro_bias     initial bias estimate [rad/s], body axes; zero is a
  ///                      fine choice if nothing better is known
  /// @param bias_cov      `E[δb δbᵀ]` [rad²/s²]; size it to the hardware's
  ///                      turn-on bias repeatability, not to zero
  /// @return `true` iff the filter is now initialised. Refuses an unconfigured
  ///         filter, non-finite or unnormalisable inputs, and a covariance that
  ///         is not finite **or not positive-definite** — an indefinite seed
  ///         would make `S` indefinite and the NIS gate meaningless. The
  ///         previous state is left untouched on refusal.
  bool initialize(const time::Tai& epoch,
                  const math::Quat<math::frames::Body, math::frames::ECI>& attitude,
                  const Eigen::Matrix3d& attitude_cov,
                  const math::Vec3<math::frames::Body>& gyro_bias, const Eigen::Matrix3d& bias_cov);

  /// Propagate the reference attitude and the covariance to @p epoch on the
  /// bias-corrected gyro.
  ///
  /// @param epoch      TAI time tag of this measurement; must be **strictly**
  ///                   after the last one
  /// @param gyro       measured body rate [rad/s], **not** bias-corrected
  /// @param gyro_valid the gyro reading is fresh and usable (§9.1). When false —
  ///                   or when the step exceeds `max_dt_s`, or the reading is
  ///                   non-finite — the attitude is **held**, the `Φ₁₂`
  ///                   attitude/bias coupling is **zero** (the bias took no part
  ///                   in a propagation that did not happen), and only the
  ///                   process noise is added, which is a lower bound on the
  ///                   true growth; `max_coast_s` is the real guard there.
  /// @return `true` on a completed step. `false` for an unconfigured or
  ///         uninitialised filter or a non-increasing epoch (state untouched),
  ///         or for a non-finite internal result (filter reset).
  bool propagate(const time::Tai& epoch, const math::Vec3<math::frames::Body>& gyro,
                 bool gyro_valid);

  /// Fold in one vector measurement: a direction observed in body axes against
  /// its inertial reference. Call once per available pair per cycle.
  ///
  /// @param body_meas measured direction in Body [any length; normalised inside]
  /// @param reference modelled direction in ECI [any length; normalised inside]
  /// @param sigma_rad transverse 1σ of the pair [rad], `> 0` — sensor **and**
  ///                  reference error, inflated for the systematic part of the
  ///                  budget (see the file header)
  /// @param out       innovation, `S`, NIS and the accept decision — filled on
  ///                  a gate rejection too
  /// @return `true` iff the measurement was applied, i.e. `out.accepted`. A
  ///         gate rejection is a normal outcome, not an error: it returns
  ///         `false` with the diagnostics populated and the count incremented.
  ///         Malformed inputs return `false` with @p out default-constructed.
  /// @param force    apply past the NIS gate (NASA/TP-2018-219822 §9.1's
  ///                  "force" editing flag; NESC TB 20-03 item d). Overrides the
  ///                  *gate* only — a negative or non-finite NIS is the covariance
  ///                  gone bad and still refuses. Counted in @ref forcedCount.
  bool update(const math::Vec3<math::frames::Body>& body_meas,
              const math::Vec3<math::frames::ECI>& reference, double sigma_rad, MekfUpdate& out,
              bool force = false);

  /// Fold in one **attitude** measurement: a complete Body ← ECI solution, which
  /// is what a star tracker reports (design doc §8.2; Markley & Crassidis §6.2.4
  /// and §7.1 [markley2014]; Lefferts, Markley & Shuster §III [lefferts1982]).
  ///
  /// **Why this is not three vector updates.** A tracker's output is already the
  /// attitude, so the natural measurement model is `z = δθ` with `H = [I₃ 0₃]` —
  /// no direction is degenerate and the update is the full 3 DOF. Decomposing it
  /// into observed star directions would need the star catalogue the tracker does
  /// not publish; feeding the *quaternion* in as a pair of synthetic vectors
  /// would double-count the same information and lose the anisotropy below.
  ///
  /// **The innovation is exact, not small-angle.** `z` is the rotation vector of
  /// `q_meas ⊗ q̂⁻¹` taken the short way round (`2·atan2(‖v‖, |q₀|)·v̂`, §3.3),
  /// the same reduction @ref nees uses, so it is the *same* `δθ` the filter's
  /// error state is defined on (`q_true = δq ⊗ q̂`) rather than its linearisation.
  /// That matters at acquisition, where a tracker returning after a coast can be
  /// tens of degrees from the reference and `2·vec(δq)` would understate it.
  ///
  /// **R is a full 3×3, and that is the point of a second tracker.** A star
  /// tracker's error is strongly **anisotropic** — about-boresight is ~6× the
  /// cross-boresight terms for the reference vehicle's AURIGA — so the caller
  /// passes `R = A_body←st · diag(σ⊥², σ⊥², σ∥²) · A_body←stᵀ`, the unit's own
  /// covariance rotated into body axes. Two trackers with non-parallel boresights
  /// then each carry the other's weak direction, which is the whole reason the
  /// vehicle flies two of them (§8.1); an isotropic `σ²I` would throw that away
  /// and report a covariance the geometry does not support.
  ///
  /// As with @ref update, `R` is treated as **white**: the tracker's fixed bias
  /// and its low-frequency spatial term do not average down, so the caller
  /// inflates R for them or the filter converges below its true error.
  ///
  /// @param measured  attitude Body ← ECI as reported, already corrected for
  ///                  mounting and (for a non-king unit) inter-tracker alignment
  ///                  by the caller — this filter has no idea which tracker it is
  ///                  reading
  /// @param noise_cov `E[δθ δθᵀ]` [rad²] in **body** axes; must be finite,
  ///                  symmetric and positive-definite
  /// @param out       innovation, `S`, NIS and the accept decision — filled on a
  ///                  gate rejection too, exactly as @ref update does
  /// @param force    apply past the attitude NIS gate (TP §9.1 "force"); the
  ///                  gate only — a negative or non-finite NIS still refuses
  /// @return `true` iff the measurement was applied. A @ref
  ///         MekfConfig::attitude_nis_gate rejection returns `false` with the
  ///         diagnostics populated and @ref rejectedCount incremented; malformed
  ///         inputs return `false` with @p out default-constructed, so the caller
  ///         can tell a divergence guard from a refusal on the count alone.
  /// @param latency_s how far behind the filter epoch the measurement was
  ///                  taken [s] (TP §3.1, Push 72). The measured attitude is
  ///                  advanced to the filter epoch on the filter's own
  ///                  bias-corrected rate, `q_now = exp([ω̂×] τ) ⊗ q_meas` (the
  ///                  same increment @ref propagate applies), and `R` is
  ///                  inflated by the rate error integrated over it,
  ///                  `σ_v² τ I + τ² P_bb`. Zero — the default — applies the
  ///                  measurement at the filter epoch as before. A star tracker's
  ///                  frame is exposed and processed before it is reported, ~one
  ///                  update period; at the 0.5 °/s slew limit 100 ms is
  ///                  0.9 mrad, above the tracker's own σ. Must be finite and
  ///                  ≥ 0; refused otherwise. Requires a usable rate (a
  ///                  propagate with a gyro this epoch); without one a non-zero
  ///                  latency is refused rather than applied on nothing.
  bool updateAttitude(const math::Quat<math::frames::Body, math::frames::ECI>& measured,
                      const Eigen::Matrix3d& noise_cov, MekfUpdate& out, bool force = false,
                      double latency_s = 0.0);

  /// Open a **same-epoch measurement batch** (NASA/TP-2018-219822 §3.2,
  /// Algorithm 3.1 — "Measurement Update Invariant to Order of Processing",
  /// Push 72). Until @ref endBatch, every accepted @ref update /
  /// @ref updateAttitude accumulates its correction in the error state instead
  /// of resetting the reference: each measurement's partials are evaluated at
  /// the **same** reference, its innovation is `y_j − H_j x̂_{j−1}` against the
  /// deviation accumulated so far, and the covariance is updated per
  /// measurement as usual. The result no longer depends on whether the sun or
  /// the magnetic field is processed first — with a reset between them, a
  /// powerful first measurement moves the linearisation point the second one
  /// is evaluated at, and the TP records that as a known source of divergence
  /// when a large prior error meets a precise measurement. Idempotent.
  void beginBatch();

  /// Close the batch: one multiplicative reset with the accumulated correction
  /// (exact axis-angle for the attitude, additive for the bias), the Reynolds
  /// covariance reset if configured, and the error state back to zero. A
  /// no-op when no batch is open. @ref propagate closes an open batch itself
  /// first — the TP: "it is imperative to perform a reset before beginning the
  /// time propagation".
  /// @return false only if the reset produced a non-finite result, in which
  ///         case the solution is dropped.
  bool endBatch();

  /// True while a batch is open (between @ref beginBatch and @ref endBatch);
  /// @ref attitude and @ref gyroBias then report the batch's reference, not the
  /// corrections accumulated so far.
  bool batchOpen() const { return batch_open_; }

  /// Swap the tuning under a running solution (NESC TB 20-03 item g;
  /// NASA/TP-2018-219822 §9.3): attitude, bias, covariance, epoch, age and
  /// counters are kept, only the noise/gate/horizon values change. Refused
  /// (returns false, **nothing** touched, old configuration kept) when @p
  /// config fails @ref MekfConfig::isValid — a bad upload must not make a
  /// running filter inert.
  bool retune(const MekfConfig& config);

  /// Re-initialise the covariance **without altering the state** (NESC TB 20-03
  /// item f; TP §9.2): `P = diag(σ_att² I, σ_bias² I)`, no cross terms. The
  /// remedy for a filter that has become over-confident and is editing good
  /// measurements while its attitude and bias are still sound; milder than
  /// @ref reset, and it keeps the converged bias. False when there is no
  /// solution or a σ is not positive and finite.
  bool reinitializeCovariance(double sigma_att_rad, double sigma_bias_rad_s);

  /// True when there is no solution, or when `P` is positive semi-definite by
  /// an LDLᵀ factorisation (TP Ch. 7's definiteness check, done explicitly on a
  /// full covariance). False is a filter-health signal.
  bool covarianceHealthy() const;

  /// Reference attitude Body ← ECI (JPL scalar-first, canonical `q0 ≥ 0`).
  math::Quat<math::frames::Body, math::frames::ECI> attitude() const {
    return math::Quat<math::frames::Body, math::frames::ECI>(attitude_);
  }

  /// Estimated gyro bias [rad/s], body axes.
  math::Vec3<math::frames::Body> gyroBias() const { return math::Vec3<math::frames::Body>(bias_); }

  /// Bias-corrected body rate from the last propagation [rad/s]; zero and
  /// @ref rateValid false when no usable gyro has been seen.
  math::Vec3<math::frames::Body> bodyRate() const { return math::Vec3<math::frames::Body>(rate_); }

  /// The last propagation had a usable gyro, so @ref bodyRate is meaningful.
  bool rateValid() const { return rate_valid_; }

  /// 6×6 error-state covariance `[δθ; δb]`, symmetric.
  const Covariance& covariance() const { return p_; }

  /// Time since the last **accepted** measurement update [s]; grows through
  /// eclipse and outage.
  double ageSeconds() const { return age_s_; }

  /// The attitude is initialised and inside the coast horizon.
  bool attitudeValid() const { return initialised_ && age_s_ <= cfg_.max_coast_s; }

  /// Measurements rejected by the NIS gate since construction or @ref reset.
  /// A rising count is the FDIR signal, not a single rejection.
  std::uint32_t rejectedCount() const { return rejected_; }

  /// Updates applied **past** their gate under `force` since construction or
  /// @ref reset (TP §9.1). Kept apart from @ref rejectedCount — a forced update
  /// is neither a rejection nor a consistent acceptance.
  std::uint32_t forcedCount() const { return forced_; }

  /// Analysis-only 6-state NEES `eᵀP⁻¹e` against a known truth, with
  /// `e = [δθ_true; δb_true]` (Bar-Shalom §5.4 [barshalom2001]). Averaged over
  /// Monte-Carlo runs it should sit inside the χ²₆ bounds; systematically above
  /// means overconfident, below means conservative. There is no truth onboard,
  /// so this exists for tests, replay, and covariance validation.
  ///
  /// @param attitude_true Body ← ECI truth
  /// @param bias_true     true gyro bias [rad/s], body axes
  /// @param out           NEES [dimensionless]
  /// @return `false` (leaving @p out untouched) when uninitialised, on
  ///         non-finite inputs, or if `P` cannot be inverted.
  bool nees(const math::Quat<math::frames::Body, math::frames::ECI>& attitude_true,
            const math::Vec3<math::frames::Body>& bias_true, double& out) const;

 private:
  /// Drop the solution while leaving the rejection count standing. The internal
  /// fault paths use this rather than @ref reset, because a filter that has just
  /// been dropped for divergence is exactly when FDIR needs to see how many
  /// measurements it had been rejecting on the way there.
  void dropSolution();
  /// Fold one accepted correction in: accumulate while a batch is open, else
  /// apply the multiplicative reset now. False (solution dropped) on a
  /// non-finite result.
  bool applyCorrection(const Eigen::Matrix<double, kDim, 1>& dx);
  /// The multiplicative reset with @p dx: exact axis-angle on the reference,
  /// additive on the bias, the Reynolds covariance reset if configured.
  bool applyReset(const Eigen::Matrix<double, kDim, 1>& dx);

  MekfConfig cfg_{};
  math::Quaternion attitude_{};                    ///< reference attitude, Body ← ECI
  Eigen::Vector3d bias_{Eigen::Vector3d::Zero()};  ///< gyro bias estimate [rad/s]
  Eigen::Vector3d rate_{Eigen::Vector3d::Zero()};  ///< last bias-corrected rate [rad/s]
  Covariance p_{Covariance::Zero()};               ///< error-state covariance
  time::Tai last_epoch_{};                         ///< epoch of `attitude_`
  double age_s_{0.0};                              ///< since the last accepted update [s]
  std::uint32_t rejected_{0};                      ///< NIS-gate rejections
  std::uint32_t forced_{0};                        ///< updates applied past the gate
  Eigen::Matrix<double, kDim, 1> dx_acc_{
      Eigen::Matrix<double, kDim, 1>::Zero()};  ///< batch error state
  bool batch_open_{false};
  bool configured_{false};
  bool initialised_{false};
  bool rate_valid_{false};
};

/// Copy a fine solution into the canonical onboard state (§8.0). Writes the
/// attitude, body rate, gyro bias, their validity flags, the **attitude and
/// gyro-bias blocks of the 15×15 error-state covariance including their cross
/// terms** (the correlation is what a consumer needs to reason about the two
/// together), and the mode — `Fine`, or `Invalid` when the attitude is past the
/// coast horizon or the filter is uninitialised. Position/velocity/accel-bias
/// fields are untouched: the orbit filter (§8.3) owns those.
///
/// As in the coarse path, the covariance blocks are written **only** when the
/// attitude is valid, so an invalid solution cannot stamp a meaningless
/// covariance over the last good one.
///
/// @param filter converged (or coasting) filter
/// @param epoch  TAI time tag to stamp on the state (§3.2)
/// @param state  canonical state to update in place
void writeToEstimatedState(const Mekf& filter, const time::Tai& epoch,
                           state::EstimatedState& state);

}  // namespace polaris::gnc

#endif  // POLARIS_GNC_MEKF_HPP
