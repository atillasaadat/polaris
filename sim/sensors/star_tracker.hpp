#ifndef POLARIS_SIM_SENSORS_STAR_TRACKER_HPP
#define POLARIS_SIM_SENSORS_STAR_TRACKER_HPP

/// @file
/// @brief Star tracker truth model (design doc §6.2).
///
/// Unlike the other sensors here, a star tracker's output is not a measured
/// vector but a **full attitude solution** — the unit does its own centroiding,
/// star identification, and QUEST-class solve internally, and reports a
/// quaternion with a validity flag. So the error stack is not §6.1's
/// `M·truth + bias + noise`: it is a small-angle perturbation applied to the
/// truth attitude, plus a validity model, and that validity model is where most
/// of the operational behaviour lives.
///
/// **The noise is anisotropic, and that is the point.** Accuracy about the
/// boresight is roughly an order of magnitude worse than across it (the ST-16's
/// 30″ vs 5″), because a rotation about the boresight moves each identified star
/// only by its small radial distance from the optical axis, while a cross-axis
/// rotation sweeps the whole field. An estimator that models the tracker as
/// isotropic will be overconfident about roll, so the truth model must reproduce
/// the asymmetry.
///
/// **Validity** (§6.1, shared occlusion model): the solution drops out when the
/// Earth intrudes on the field of view, when the Sun or Moon enters its keep-out
/// cone, or when the vehicle slews faster than the unit can integrate (star
/// images smear across the detector and identification fails). Each is reported
/// distinctly via @ref StarTrackerMeasurement::occluder / ::rate_limited so the
/// FDIR suite can tell an expected geometric outage from a control problem.
///
/// Rate gating is **not** here: `update_rate_hz` is carried for the §2.4
/// interface buffer, which decides when the FSW actually gets a new solution.
/// Calling `sample` faster than the native rate is the caller's error to make.
///
/// References:
///  - Markley & Crassidis, *Fundamentals of Spacecraft Attitude Determination
///    and Control*, 2014, §4.2 (star tracker models, cross/about-boresight
///    accuracy). [markley2014]

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

/// One star tracker's performance and constraints, in SI.
struct StarTrackerSpec {
  /// 1σ accuracy about the two axes perpendicular to the boresight [rad].
  double cross_axis_sigma = 0.0;
  /// 1σ accuracy about the boresight [rad] — typically ~6-10× worse.
  double boresight_sigma = 0.0;
  /// Native attitude-solution rate [Hz]; informational (see the file header).
  double update_rate_hz = 0.0;
  /// Full circular field of view [rad]. The Earth keep-out is measured from the
  /// FOV edge, so an Earth limb inside the FOV invalidates the solution.
  double fov_rad = 0.0;
  /// Maximum body rate the unit can still solve at [rad/s]; 0 disables the check.
  double max_slew_rate = 0.0;
  /// Bright-body clearance requirements (§6.1). The Earth entry is *additional*
  /// margin beyond the half-FOV.
  KeepOutSpec keep_out;
  /// Boresight direction in the sensor's own axes. +z by convention; the
  /// mounting rotation carries it into body axes.
  Eigen::Vector3d boresight_sensor = Eigen::Vector3d::UnitZ();

  /// Build a spec from datasheet-native hardware-library params (the keys used
  /// by `config/hardware/star_tracker/*.yaml`), converting each to SI. Missing
  /// keys default to 0 (that term disabled). See star_tracker.cpp for the list.
  static StarTrackerSpec fromParams(const std::map<std::string, double>& params);
};

/// One attitude solution.
struct StarTrackerMeasurement {
  /// Measured attitude, Body←ECI. Meaningless when `valid` is false; the model
  /// still fills it with the last-known truth-derived value rather than garbage.
  math::Quat<math::frames::Body, math::frames::ECI> attitude{};
  bool valid{true};
  /// Why the solution was lost, when it was lost geometrically.
  Occluder occluder{Occluder::kNone};
  /// True when the body rate exceeded `max_slew_rate` (image smear).
  bool rate_limited{false};
  time::Tai time_tag{};
};

/// A star tracker. Construct with its spec, its unit→body mounting, and a
/// per-source stream id under the run's master seed.
class StarTracker {
 public:
  StarTracker(const StarTrackerSpec& spec, const Eigen::Matrix3d& mounting_dcm,
              std::uint64_t master_seed, std::uint64_t stream_id)
      : spec_(spec),
        boresight_body_(mounting_dcm * spec.boresight_sensor),
        rng_(random::streamRng(master_seed, stream_id)) {}

  /// Solve at truth time @p epoch.
  ///
  /// @param truth_attitude True Body←ECI attitude.
  /// @param body_rate      True body rate [rad/s], for the slew-rate check.
  /// @param sky            Spacecraft/Sun/Moon ECI positions, for keep-out.
  StarTrackerMeasurement sample(const time::Tai& epoch,
                                const math::Quat<math::frames::Body, math::frames::ECI>& truth,
                                const math::Vec3<math::frames::Body>& body_rate,
                                const SkyGeometry& sky);

  /// The boresight in body axes, as mounted.
  const Eigen::Vector3d& boresightBody() const { return boresight_body_; }

  // --- Fault injection (§9) --------------------------------------------------

  /// Persistent attitude-error jump [rad], as a small-angle rotation vector in
  /// body axes, until cleared (replaces, not accumulates). Applied with the
  /// noise, so it perturbs the reported attitude exactly like a real bias would.
  void injectAttitudeBias(const math::Vec3<math::frames::Body>& delta_rad) {
    fault_bias_ = delta_rad.eigen();
  }

  /// Force every subsequent solution invalid (loss of star identification).
  void setDropout(bool dropped) { fault_dropout_ = dropped; }

  void clearFaults() {
    fault_bias_.setZero();
    fault_dropout_ = false;
  }

 private:
  StarTrackerSpec spec_;
  Eigen::Vector3d boresight_body_;
  random::SplitMix64 rng_;
  Eigen::Vector3d fault_bias_ = Eigen::Vector3d::Zero();
  bool fault_dropout_ = false;
};

}  // namespace polaris::sim::sensors

#endif  // POLARIS_SIM_SENSORS_STAR_TRACKER_HPP
