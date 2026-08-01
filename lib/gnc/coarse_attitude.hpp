#ifndef POLARIS_GNC_COARSE_ATTITUDE_HPP
#define POLARIS_GNC_COARSE_ATTITUDE_HPP

/// @file
/// @brief Coarse attitude estimator — sun sensor + magnetometer + gyro
/// (design doc §8.1, §10; REQ-ADET-002, REQ-ADET-003).
///
/// The estimator behind the **Safe-mode floor**. When star trackers are
/// unavailable — occluded, faulted, slew-limited, or simply not yet acquired —
/// attitude comes from two coarse vector pairs (sun sensor vs. sun ephemeris,
/// magnetometer vs. onboard IGRF-14) with the gyro carrying the solution
/// *between* and *through* those updates. Because the analytic sun ephemeris
/// fallback (§11.3) is pure arithmetic, this estimator is **star-tracker- and
/// table-independent**: it can always be run.
///
/// **Structure.** Gyro propagation of the quaternion at the estimation rate, a
/// @ref triad solve whenever both vector pairs are fresh, valid, and
/// well-conditioned, and a **fixed-gain complementary blend** of the two along
/// their eigenaxis. A fixed gain, not a Kalman update: this is the coarse mode
/// whose job is a few-degree, always-available solution: the MEKF (§8.1 fine
/// mode) is where the optimal fusion belongs, and a filter that can diverge is
/// the wrong thing under the safe-mode floor.
///
/// **The covariance carries a systematic floor.** Repeated blending shrinks the
/// error covariance towards `k/(2−k)` of one TRIAD fix, which would be a lie:
/// most of a coarse budget is *systematic* — analytic-ephemeris error, IGRF
/// model error, sensor alignment — and the same offset every cycle does not
/// average down no matter how many fixes you take. The configured uncertainty
/// is therefore split per source into a **white** part (which the blend is
/// allowed to reduce) and a **systematic** part (which is re-evaluated on the
/// current geometry and added back after every blend). The published covariance
/// is the sum, so it converges to the systematic floor rather than to zero —
/// which is what makes it safe to seed the MEKF with.
///
/// **Eclipse and dropout.** With no sun vector the estimator gyro-propagates
/// and its covariance grows without bound; @ref CoarseAttitudeOutput::age_s
/// reports how long since the last vector fix. Past
/// `CoarseAttitudeConfig::max_coast_s` the attitude is declared **invalid**
/// rather than quietly drifting — consumers gate on validity (§9.1). A TRIAD
/// after the gap re-acquires immediately (the blend gain is bypassed on
/// re-acquisition, since the propagated attitude carries no information then).
///
/// **Frames, units, conventions.** Attitude is `Quat<Body, ECI>` (JPL
/// scalar-first, canonical `q0 ≥ 0`); body rate is `Vec3<Body>` in rad/s; times
/// are TAI (§3.2); the covariance is the body-frame attitude error `δθ` in
/// rad², matching `state::ErrorState::kAttitude`. All SI.
///
/// **Flight path.** Fixed-size Eigen, no heap, no exceptions, no recursion, no
/// unbounded loops, every return code checked, finiteness checks on the output.
/// No F´ types and no I/O — the F´ `AttitudeEstimator` component wraps this.
///
/// References:
///  - Markley & Crassidis, *Fundamentals of Spacecraft Attitude Determination
///    and Control*, 2014, §3.1 (quaternion kinematics), §5.2 (TRIAD), §7.1
///    (gyro error model / attitude propagation). [markley2014]
///  - Wertz (ed.), *Spacecraft Attitude Determination and Control*, 1978,
///    §12.2 (coarse sun/magnetic-field attitude determination). [wertz1978]
///  - Shuster & Oh, J. Guidance & Control 4(1):70-77, 1981 (TRIAD
///    covariance). [shuster1981]

#include <Eigen/Core>

#include "gnc/triad.hpp"
#include "math/frames.hpp"
#include "math/quaternion.hpp"
#include "math/typed_vector.hpp"
#include "state/estimated_state.hpp"
#include "time/timescales.hpp"

namespace polaris::gnc {

/// Tuning for @ref CoarseAttitudeEstimator. Every value is mission
/// configuration (§19.3) — there are no flight defaults worth trusting, so
/// @ref CoarseAttitudeConfig::isValid gates the constructor.
struct CoarseAttitudeConfig {
  /// **White** (cycle-to-cycle independent) part of the sun-pair 1σ transverse
  /// uncertainty [rad]: sensor noise, quantisation. This is the part repeated
  /// fixes average down.
  double sigma_sun_white_rad{0.0};
  /// **Systematic** part of the sun-pair 1σ transverse uncertainty [rad]:
  /// ephemeris error (the analytic fallback is ~0.4°, §11.3), sensor
  /// alignment/calibration. Constant across cycles, so it never averages down
  /// and becomes a covariance floor. May be zero.
  double sigma_sun_sys_rad{0.0};
  /// White part of the magnetic-pair 1σ transverse uncertainty [rad].
  double sigma_mag_white_rad{0.0};
  /// Systematic part of the magnetic-pair 1σ transverse uncertainty [rad]:
  /// IGRF model error, hard/soft-iron residual, alignment. May be zero.
  double sigma_mag_sys_rad{0.0};
  /// Gyro angle random walk [rad/s^(1/2)], i.e. the square root of the
  /// attitude-error variance accumulated per second of propagation.
  double gyro_arw{0.0};
  /// Minimum `|sin θ|` between the sun and field directions for a TRIAD solve
  /// [dimensionless]; below this the roll about the sun is unobservable. No
  /// default — mission configuration (§19.3); zero fails validation.
  double min_sin_angle{0.0};
  /// Complementary blend gain applied to the TRIAD-vs-propagated error, in
  /// `(0, 1]` [dimensionless]. 1 snaps to TRIAD; smaller values trade
  /// measurement noise for gyro smoothing. No default — mission configuration.
  double triad_gain{0.0};
  /// Longest gyro-only coast before the attitude is declared invalid [s].
  double max_coast_s{0.0};
  /// Largest accepted propagation step [s]. A longer gap is treated as a
  /// dropout (no propagation over it) rather than extrapolated.
  double max_dt_s{0.0};

  /// True when every field is finite and in range. Checked once at construction.
  bool isValid() const;
};

/// One estimation-cycle input. Vectors are frame-tagged and need not be unit
/// length; validity flags are the caller's (§9.1) — an invalid measurement is
/// excluded here, never used.
struct CoarseAttitudeInput {
  /// TAI time tag of this cycle (§3.2).
  time::Tai epoch{};

  /// Measured body rate [rad/s], **not** bias-corrected.
  math::Vec3<math::frames::Body> gyro{};
  /// Gyro bias to subtract [rad/s]; zero when no bias estimate exists.
  math::Vec3<math::frames::Body> gyro_bias{};
  /// Gyro measurement is valid and fresh.
  bool gyro_valid{false};

  /// Measured sun direction in Body [dimensionless direction].
  math::Vec3<math::frames::Body> sun_body{};
  /// Modelled sun direction in ECI [dimensionless direction].
  math::Vec3<math::frames::ECI> sun_ref{};
  /// Sun pair is valid this cycle (sensor in FOV, not eclipsed, reference known).
  bool sun_valid{false};

  /// Measured magnetic field in Body [T] (direction is what is used).
  math::Vec3<math::frames::Body> mag_body{};
  /// Modelled magnetic field in ECI [T] (direction is what is used).
  math::Vec3<math::frames::ECI> mag_ref{};
  /// Magnetic pair is valid this cycle.
  bool mag_valid{false};
};

/// Estimator product for one cycle. Gate on @ref attitude_valid /
/// @ref rate_valid before use.
struct CoarseAttitudeOutput {
  /// Attitude Body ← ECI (JPL scalar-first, canonical `q0 ≥ 0`).
  math::Quat<math::frames::Body, math::frames::ECI> attitude{};
  /// Bias-corrected body rate [rad/s].
  math::Vec3<math::frames::Body> body_rate{};
  /// Attitude-error covariance `E[δθ δθᵀ]` [rad²], body-frame axes.
  Eigen::Matrix3d covariance{Eigen::Matrix3d::Zero()};
  /// Time since the last accepted TRIAD update [s]; grows through eclipse.
  double age_s{0.0};
  /// A TRIAD solution was accepted and blended in this cycle.
  bool triad_applied{false};
  /// @ref attitude and @ref covariance are usable.
  bool attitude_valid{false};
  /// @ref body_rate is usable.
  bool rate_valid{false};
};

/// Sun + magnetometer + gyro coarse attitude estimator (design doc §8.1).
///
/// Hold one instance per vehicle and call @ref update once per estimation
/// cycle with monotonically increasing TAI epochs. The object is a plain value
/// with fixed-size storage; it allocates nothing after construction.
class CoarseAttitudeEstimator {
 public:
  /// Construct with @p config. If the config is invalid the estimator is inert:
  /// @ref update returns false forever. Check @ref isConfigured.
  explicit CoarseAttitudeEstimator(const CoarseAttitudeConfig& config);

  /// True when the configuration passed @ref CoarseAttitudeConfig::isValid.
  bool isConfigured() const { return configured_; }

  /// True once a TRIAD has initialised the solution at least once.
  bool isInitialised() const { return initialised_; }

  /// Drop the solution (cold start). Configuration is retained.
  void reset();

  /// Run one estimation cycle: propagate on the gyro, update on TRIAD when the
  /// vector pairs allow, and write @p out.
  ///
  /// Epochs must be strictly increasing: a repeated or backwards time tag is a
  /// clock fault, and re-applying a TRIAD at the same epoch would shrink the
  /// covariance on information the estimator has already used.
  ///
  /// @param in  measurements and their validity for this cycle
  /// @param out product; `attitude_valid == false` while uninitialised or after
  ///            a coast longer than `max_coast_s`. The cycle still runs in that
  ///            case, and an independently valid body rate is still published —
  ///            Safe-mode rate damping needs it before attitude acquisition.
  /// @return `true` iff the cycle produced a valid attitude. A **refused**
  ///         cycle — unconfigured estimator, non-increasing epoch, or a
  ///         non-finite internal result — returns `false` with @p out left
  ///         default-constructed (finite, all flags clear), so nothing
  ///         half-written or stale is ever published.
  bool update(const CoarseAttitudeInput& in, CoarseAttitudeOutput& out);

 private:
  /// Rotate the stored attitude and covariance forward by @p dt_s at @p rate
  /// [rad/s, Body] using the closed-form quaternion kinematics. Returns false
  /// if the propagated quaternion could not be normalised, in which case the
  /// stored state is left untouched and the cycle must be failed.
  bool propagate(const Eigen::Vector3d& rate, double dt_s);

  CoarseAttitudeConfig cfg_{};
  math::Quaternion attitude_{};  ///< Body ← ECI
  /// Reducible part of the attitude-error covariance [rad², Body]: TRIAD white
  /// noise plus accumulated gyro random walk. Shrinks under repeated fixes.
  Eigen::Matrix3d cov_random_{Eigen::Matrix3d::Zero()};
  /// Systematic floor [rad², Body], re-evaluated on the geometry of each fix.
  /// Never reduced by blending; published covariance is the sum of the two.
  Eigen::Matrix3d cov_systematic_{Eigen::Matrix3d::Zero()};
  time::Tai last_epoch_{};  ///< epoch of `attitude_`
  double age_s_{0.0};       ///< since last TRIAD [s]
  bool configured_{false};
  bool initialised_{false};
  bool have_epoch_{false};
};

/// Copy a coarse solution into the canonical onboard state (§8.0). Writes the
/// attitude, body rate, their validity flags, and the mode — `Coarse`, or
/// `Invalid` when the attitude is not usable. Position/velocity/bias fields are
/// untouched: the orbit filter (§8.3) owns those.
///
/// The **attitude block** of the 15×15 error-state covariance is written (and
/// `valid.covariance` set) only when the attitude is valid; an invalid solution
/// leaves the previous covariance in place rather than stamping a meaningless
/// one over it. Per `StateValidity::covariance`, the flag claims only the
/// blocks belonging to fields flagged valid — coarse mode populates the
/// attitude block and nothing else.
///
/// @param out    coarse estimator product
/// @param epoch  TAI time tag to stamp on the state (§3.2)
/// @param state  canonical state to update in place
void writeToEstimatedState(const CoarseAttitudeOutput& out, const time::Tai& epoch,
                           state::EstimatedState& state);

}  // namespace polaris::gnc

#endif  // POLARIS_GNC_COARSE_ATTITUDE_HPP
