#ifndef POLARIS_SIM_SENSORS_STAR_TRACKER_HPP
#define POLARIS_SIM_SENSORS_STAR_TRACKER_HPP

/// @file
/// @brief Star tracker truth model (design doc §6.2).
///
/// Unlike the other sensors here, a star tracker's output is not a measured
/// vector but a **full attitude solution** — the unit does its own centroiding,
/// star identification, and QUEST-class solve internally, and reports a
/// quaternion with a validity flag. So the error stack is not §6.1's
/// `M·truth + bias + noise`: it is a sum of physically distinct attitude-error
/// mechanisms, plus an availability model that is where most of the operational
/// behaviour lives.
///
/// **Why the error terms are kept separate.** Vendors quote four different
/// numbers because they behave differently in a control loop, and collapsing
/// them into one σ throws away exactly the information an ADCS analyst needs:
///
///  - **Bias** — a fixed per-unit offset (alignment and calibration residual).
///    It does not average down, and a pointing budget must carry it in full. On
///    the Auriga it is 0.017° worst case, ~60× the temporal noise: for a
///    long-exposure imaging mission it, not the noise, sets the error budget.
///  - **Low-frequency spatial (FOV) error** — optical distortion residual that
///    depends on *where in the field* the identified stars fall. It changes as
///    the star field drifts across the FOV, so it is slowly varying: correlated
///    over minutes, invisible to a filter tuned for white noise, and it walks a
///    pointing solution around in a way a spectrum-blind estimator will absorb
///    into its bias state.
///  - **High-frequency spatial (pixel) error** — sub-pixel centroiding error,
///    varying as star images move across pixel boundaries. Faster than the FOV
///    term, still not white.
///  - **Temporal noise** — the genuinely white, per-sample term. The only one
///    that averages down as √N, and the only one most estimators model.
///
/// Both spatial terms are modelled as Gauss-Markov processes. Their correlation
/// times are **modelling choices, not datasheet values** (no vendor quotes them),
/// exposed as parameters so a mission can tune them against flight data — the
/// same treatment `ImuSpec` gives its bias correlation time.
///
/// **Every axis pair is anisotropic, and that is the point.** Accuracy about the
/// boresight is roughly 6× worse than across it (Auriga: 51″ vs 9″ low-frequency,
/// 70″ vs 11″ temporal), because a rotation about the boresight moves each
/// identified star only by its small radial distance from the optical axis, while
/// a cross-axis rotation sweeps the whole field. An estimator handed an isotropic
/// tracker will be overconfident about roll.
///
/// **Availability is a state machine, not a threshold.** Acquisition and tracking
/// are separate regimes with separate limits, and the difference is operationally
/// large: the Auriga tracks through 3 °/s but only acquires below 2 °/s (0.3 °/s
/// in its baseline configuration), and tolerates 2.5 °/s² while tracking against
/// 1 °/s² while acquiring. So a vehicle that slews out of the tracking envelope
/// does not simply resume when it slows down — it must first drop below the
/// tighter *acquisition* envelope and then stay there for the lost-in-space time
/// (3.8 s typical) before a solution reappears. A model with one rate threshold
/// and no re-acquisition delay hides both effects, and hides them in the
/// direction that makes a slew look safer than it is.
///
/// Geometric validity comes from the shared §6.1 occlusion model: Sun and Earth
/// exclusion angles are quoted by vendors as absolute boresight-to-limb angles
/// (Auriga: 35° Sun, 22° Earth), so that is how they are configured here. The
/// Earth constraint is the larger of the quoted exclusion angle and the half
/// field of view — a limb inside the FOV floods the detector regardless of what
/// the baffle is rated for.
///
/// References:
///  - Markley & Crassidis, *Fundamentals of Spacecraft Attitude Determination
///    and Control*, 2014, §4.2 (star tracker models, cross/about-boresight
///    accuracy, low/high-frequency error decomposition). [markley2014]

#include <cstdint>
#include <Eigen/Core>
#include <map>
#include <string>

#include "math/frames.hpp"
#include "math/quaternion.hpp"
#include "math/typed_vector.hpp"
#include "random/rng.hpp"
#include "sensors/occlusion.hpp"
#include "time/timescales.hpp"

namespace polaris::sim::sensors {

/// A 1σ error pair for one mechanism: across the boresight and about it.
struct StarTrackerAxisSigma {
  double cross = 0.0;      ///< 1σ about the two axes perpendicular to the boresight [rad]
  double boresight = 0.0;  ///< 1σ about the boresight [rad] — typically ~6× worse
};

/// What the unit is doing. Availability is a state machine because acquisition
/// and tracking have different envelopes (see the file header).
enum class StarTrackerMode {
  kLost,       ///< no solution; not yet inside the acquisition envelope
  kAcquiring,  ///< inside the acquisition envelope, counting down lost-in-space
  kTracking,   ///< producing a valid attitude solution
};

/// One star tracker's performance and constraints, in SI.
struct StarTrackerSpec {
  /// White, per-sample noise. Averages down as √N.
  StarTrackerAxisSigma temporal;
  /// Low-frequency spatial (field-of-view / optical distortion) error.
  StarTrackerAxisSigma low_freq_spatial;
  /// High-frequency spatial (pixel / centroiding) error.
  StarTrackerAxisSigma high_freq_spatial;
  /// Correlation time of the low-frequency spatial error [s]. **Modelling
  /// choice**, not a datasheet value: it is the timescale on which the star field
  /// drifts across the FOV. 0 makes the term white.
  double low_freq_correlation_s = 0.0;
  /// Correlation time of the high-frequency spatial error [s]. Also a modelling
  /// choice; shorter than the low-frequency one by construction.
  double high_freq_correlation_s = 0.0;
  /// Bound on the fixed per-unit attitude bias [rad] (vendors quote a worst
  /// case). The realised bias is drawn once at construction within this bound.
  double bias_bound = 0.0;
  /// Thermo-elastic drift [rad per K of departure from the calibration
  /// temperature] — the mount and optical bench distorting with temperature.
  double thermo_elastic_per_k = 0.0;

  /// Maximum body rate at which the unit can still **acquire** [rad/s].
  double acquisition_rate_limit = 0.0;
  /// Maximum body rate at which it can stay **tracking** [rad/s] — higher.
  double tracking_rate_limit = 0.0;
  /// Maximum angular acceleration during acquisition [rad/s²].
  double acquisition_accel_limit = 0.0;
  /// Maximum angular acceleration while tracking [rad/s²] — higher.
  double tracking_accel_limit = 0.0;
  /// Time to first fix from lost-in-space [s]: how long the unit must sit inside
  /// the acquisition envelope before a solution appears.
  double lost_in_space_s = 0.0;

  /// Native attitude-solution rate [Hz]; informational — the §2.4 interface
  /// buffer decides when the FSW actually gets a new solution.
  double update_rate_hz = 0.0;
  /// Full circular field of view [rad].
  double fov_rad = 0.0;
  /// Bright-body exclusion, as absolute boresight-to-limb angles (§6.1).
  KeepOutSpec keep_out;
  /// Boresight direction in the sensor's own axes. +z by convention; the
  /// mounting rotation carries it into body axes.
  Eigen::Vector3d boresight_sensor = Eigen::Vector3d::UnitZ();

  /// Build a spec from datasheet-native hardware-library params (the keys used
  /// by `config/hardware/star_tracker/*.yaml`), converting each to SI —
  /// including the 3σ→1σ division vendors' spatial and noise figures carry.
  /// Missing keys default to 0 (that term disabled). See star_tracker.cpp for
  /// the key list.
  static StarTrackerSpec fromParams(const std::map<std::string, double>& params);
};

/// Everything one solve needs from the truth state.
struct StarTrackerInput {
  /// True Body←ECI attitude.
  math::Quat<math::frames::Body, math::frames::ECI> attitude{};
  /// True body rate [rad/s] — drives the rate envelopes.
  math::Vec3<math::frames::Body> body_rate{};
  /// True angular acceleration [rad/s²] — drives the acceleration envelopes,
  /// which bite during slew *transients* where the rate alone looks acceptable.
  math::Vec3<math::frames::Body> angular_accel{};
  /// Spacecraft / Sun / Moon geometry for the occlusion check.
  SkyGeometry sky{};
  /// Departure from the calibration temperature [K], for thermo-elastic drift.
  double temperature_delta_k = 0.0;
};

/// One attitude solution.
struct StarTrackerMeasurement {
  /// Measured attitude, Body←ECI. Meaningless when `valid` is false; the model
  /// still fills it with the truth-derived value rather than garbage.
  math::Quat<math::frames::Body, math::frames::ECI> attitude{};
  bool valid{true};
  StarTrackerMode mode{StarTrackerMode::kTracking};
  /// What the field of view was looking at: keep-out verdict plus the fraction
  /// of the FOV each body covers. Carried on every sample, valid or not — the
  /// fractions are how a consumer sees an outage *coming* rather than only that
  /// it arrived.
  OcclusionState occlusion{};
  /// True when the body rate exceeded the envelope for the current mode.
  bool rate_limited{false};
  /// True when the angular acceleration exceeded the envelope for the mode.
  bool accel_limited{false};
  /// Seconds spent inside the acquisition envelope so far. Counts up to
  /// `lost_in_space_s`, then the solution appears. Zero while tracking.
  double acquisition_elapsed_s{0.0};
  time::Tai time_tag{};
};

/// A star tracker. Construct with its spec, its unit→body mounting, and a
/// per-source stream id under the run's master seed. The fixed per-unit bias and
/// thermo-elastic axis are realised at construction, so the same
/// {spec, seed, stream_id} always builds the same physical unit.
class StarTracker {
 public:
  StarTracker(const StarTrackerSpec& spec, const Eigen::Matrix3d& mounting_dcm,
              std::uint64_t master_seed, std::uint64_t stream_id);

  /// Solve over an interval of @p dt seconds ending at truth time @p epoch.
  ///
  /// @p dt drives both the Gauss-Markov spatial errors and the lost-in-space
  /// countdown. A non-positive @p dt returns an invalid sample and draws no
  /// randomness, so a degenerate step cannot desynchronise the noise stream.
  StarTrackerMeasurement sample(const time::Tai& epoch, double dt, const StarTrackerInput& input);

  /// The boresight in body axes, as mounted.
  const Eigen::Vector3d& boresightBody() const { return boresight_body_; }

  /// Current mode, for tests and telemetry.
  StarTrackerMode mode() const { return mode_; }

  /// The realised per-unit fixed bias [rad], body axes. Exposed because a
  /// pointing budget needs the number this particular unit actually got.
  const Eigen::Vector3d& unitBias() const { return unit_bias_; }

  // --- Fault injection (§9) --------------------------------------------------

  /// Persistent attitude-error jump [rad], as a small-angle rotation vector in
  /// body axes, until cleared (replaces, not accumulates). A wrong answer, not a
  /// lost one: the solution stays valid, which is what the health monitors have
  /// to catch on their own.
  void injectAttitudeBias(const math::Vec3<math::frames::Body>& delta_rad) {
    fault_bias_ = delta_rad.eigen();
  }

  /// Force loss of star identification until cleared. Recovery goes through the
  /// full lost-in-space delay, like any other loss of track.
  void setDropout(bool dropped) { fault_dropout_ = dropped; }

  void clearFaults() {
    fault_bias_.setZero();
    fault_dropout_ = false;
  }

 private:
  /// Advance one Gauss-Markov error state by @p dt and return the new value.
  Eigen::Vector3d stepSpatial(const StarTrackerAxisSigma& sigma, double correlation_s, double dt,
                              Eigen::Vector3d& state, const Eigen::Vector3d& draw) const;

  /// Draw an anisotropic error vector in body axes from three unit normals.
  Eigen::Vector3d anisotropic(const StarTrackerAxisSigma& sigma, double g_cross_1, double g_cross_2,
                              double g_bore) const;

  StarTrackerSpec spec_;
  Eigen::Vector3d boresight_body_;
  Eigen::Vector3d cross_axis_1_;  ///< orthonormal basis completing the boresight
  Eigen::Vector3d cross_axis_2_;
  random::SplitMix64 rng_;

  // Fixed per-unit realisations, drawn once at construction.
  Eigen::Vector3d unit_bias_ = Eigen::Vector3d::Zero();
  Eigen::Vector3d thermo_axis_ = Eigen::Vector3d::Zero();

  // Evolving spatial-error states.
  Eigen::Vector3d low_freq_state_ = Eigen::Vector3d::Zero();
  Eigen::Vector3d high_freq_state_ = Eigen::Vector3d::Zero();

  StarTrackerMode mode_ = StarTrackerMode::kLost;
  double acquisition_elapsed_s_ = 0.0;

  Eigen::Vector3d fault_bias_ = Eigen::Vector3d::Zero();
  bool fault_dropout_ = false;
};

}  // namespace polaris::sim::sensors

#endif  // POLARIS_SIM_SENSORS_STAR_TRACKER_HPP
