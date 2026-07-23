#include "sensors/star_tracker.hpp"

#include <algorithm>
#include <cmath>

namespace polaris::sim::sensors {
namespace {

constexpr double kDeg2Rad = 0.017453292519943295;
/// Arcseconds to radians: 1″ = (π/180)/3600.
constexpr double kArcsec2Rad = kDeg2Rad / 3600.0;

double get(const std::map<std::string, double>& p, const std::string& key) {
  const auto it = p.find(key);
  return it == p.end() ? 0.0 : it->second;
}

/// Vendors quote spatial and noise figures at 3σ; the model works in 1σ.
double arcsec3SigmaToRad(double v) {
  return v * kArcsec2Rad / 3.0;
}

/// Read one anisotropic 3σ arcsecond pair.
StarTrackerAxisSigma readAxisPair(const std::map<std::string, double>& p, const std::string& xy_key,
                                  const std::string& z_key) {
  StarTrackerAxisSigma s;
  s.cross = arcsec3SigmaToRad(get(p, xy_key));
  s.boresight = arcsec3SigmaToRad(get(p, z_key));
  return s;
}

/// Any unit vector perpendicular to @p v. Picks the seed axis least aligned with
/// @p v so the cross product is never near-degenerate.
Eigen::Vector3d anyPerpendicular(const Eigen::Vector3d& v) {
  Eigen::Index smallest = 0;
  v.cwiseAbs().minCoeff(&smallest);
  Eigen::Vector3d seed = Eigen::Vector3d::Zero();
  seed[smallest] = 1.0;
  return v.cross(seed).normalized();
}

/// An isotropically distributed unit vector from three normal draws.
Eigen::Vector3d randomDirection(random::SplitMix64& rng) {
  const Eigen::Vector3d v(rng.gaussian(), rng.gaussian(), rng.gaussian());
  const double norm = v.norm();
  return (norm > 0.0) ? Eigen::Vector3d(v / norm) : Eigen::Vector3d::UnitZ();
}

}  // namespace

StarTrackerSpec StarTrackerSpec::fromParams(const std::map<std::string, double>& p) {
  StarTrackerSpec s;

  // The four error mechanisms, each quoted XY (cross-boresight) / Z (about the
  // boresight) at 3σ in arcseconds — the vendor convention.
  s.temporal = readAxisPair(p, "temporal_noise_xy_arcsec_3sigma", "temporal_noise_z_arcsec_3sigma");
  s.low_freq_spatial = readAxisPair(p, "lf_spatial_xy_arcsec_3sigma", "lf_spatial_z_arcsec_3sigma");
  s.high_freq_spatial =
      readAxisPair(p, "hf_spatial_xy_arcsec_3sigma", "hf_spatial_z_arcsec_3sigma");
  s.low_freq_correlation_s = get(p, "lf_spatial_correlation_s");
  s.high_freq_correlation_s = get(p, "hf_spatial_correlation_s");

  // Bias is quoted as a worst case in degrees, thermo-elastic as arcsec per °C
  // (a K and a °C are the same increment, so no conversion beyond the angle).
  s.bias_bound = get(p, "bias_deg") * kDeg2Rad;
  s.thermo_elastic_per_k = get(p, "thermo_elastic_arcsec_per_c") * kArcsec2Rad;

  s.acquisition_rate_limit = get(p, "acquisition_rate_deg_s") * kDeg2Rad;
  s.tracking_rate_limit = get(p, "tracking_rate_deg_s") * kDeg2Rad;
  s.acquisition_accel_limit = get(p, "acquisition_accel_deg_s2") * kDeg2Rad;
  s.tracking_accel_limit = get(p, "tracking_accel_deg_s2") * kDeg2Rad;
  s.lost_in_space_s = get(p, "lost_in_space_s");

  s.update_rate_hz = get(p, "update_rate_hz");
  s.fov_rad = get(p, "fov_deg") * kDeg2Rad;

  // Sun and Earth exclusion are quoted as absolute boresight-to-limb angles. The
  // Earth constraint additionally cannot be looser than the field of view: a limb
  // inside the FOV floods the detector whatever the baffle is rated for.
  s.keep_out.sun_rad = get(p, "sun_exclusion_deg") * kDeg2Rad;
  s.keep_out.moon_rad = get(p, "moon_exclusion_deg") * kDeg2Rad;
  s.keep_out.earth_rad = std::max(get(p, "earth_exclusion_deg") * kDeg2Rad, 0.5 * s.fov_rad);
  return s;
}

StarTracker::StarTracker(const StarTrackerSpec& spec, const Eigen::Matrix3d& mounting_dcm,
                         std::uint64_t master_seed, std::uint64_t stream_id, bool noise_enabled)
    : spec_(spec),
      boresight_body_(mounting_dcm * spec.boresight_sensor),
      rng_(random::streamRng(master_seed, stream_id)),
      noise_enabled_(noise_enabled) {
  const Eigen::Vector3d bore = boresight_body_.normalized();
  cross_axis_1_ = anyPerpendicular(bore);
  cross_axis_2_ = bore.cross(cross_axis_1_);

  // Realise this specific unit. The bias is a bound, not a σ, so the direction is
  // isotropic and the magnitude uniform within the bound — one unit off the line
  // is not systematically at its worst case, but no unit is outside it.
  thermo_axis_ = randomDirection(rng_);
  const Eigen::Vector3d bias_direction = randomDirection(rng_);
  unit_bias_ = bias_direction * (spec_.bias_bound * rng_.uniform());
}

Eigen::Vector3d StarTracker::anisotropic(const StarTrackerAxisSigma& sigma, double g_cross_1,
                                         double g_cross_2, double g_bore) const {
  return sigma.cross * (g_cross_1 * cross_axis_1_ + g_cross_2 * cross_axis_2_) +
         sigma.boresight * g_bore * boresight_body_.normalized();
}

Eigen::Vector3d StarTracker::stepSpatial(const StarTrackerAxisSigma& sigma, double correlation_s,
                                         double dt, Eigen::Vector3d& state,
                                         const Eigen::Vector3d& draw) const {
  // First-order Gauss-Markov, initialised implicitly at zero and driven so the
  // stationary standard deviation is the quoted σ. With no correlation time the
  // process degenerates to white noise, which is the honest reading of "no
  // timescale was specified".
  if (!(correlation_s > 0.0)) {
    state = anisotropic(sigma, draw.x(), draw.y(), draw.z());
    return state;
  }
  const double phi = std::exp(-dt / correlation_s);
  const double q = std::sqrt(std::max(0.0, 1.0 - phi * phi));
  state = phi * state + q * anisotropic(sigma, draw.x(), draw.y(), draw.z());
  return state;
}

StarTrackerMeasurement StarTracker::sample(const time::Tai& epoch, double dt,
                                           const StarTrackerInput& input) {
  StarTrackerMeasurement m;
  m.time_tag = epoch;
  m.attitude = input.attitude;

  // A degenerate step draws nothing: advancing the stream on a zero-length
  // interval would make the noise depend on how the caller chose to step.
  if (!(dt > 0.0)) {
    m.valid = false;
    m.mode = mode_;
    m.acquisition_elapsed_s = acquisition_elapsed_s_;
    return m;
  }

  // Nine draws every call, in a fixed order, whether or not a solution results:
  // an outage must not shift the stream position, or a scenario that changed
  // only the geometry would silently change the noise on every later sample.
  const Eigen::Vector3d low_draw(rng_.gaussian(), rng_.gaussian(), rng_.gaussian());
  const Eigen::Vector3d high_draw(rng_.gaussian(), rng_.gaussian(), rng_.gaussian());
  const Eigen::Vector3d temporal_draw(rng_.gaussian(), rng_.gaussian(), rng_.gaussian());

  const Eigen::Vector3d boresight_eci =
      input.attitude.inverse().rotate(math::Vec3<math::frames::Body>(boresight_body_)).eigen();
  m.occlusion = evaluateLineOfSight(boresight_eci, 0.5 * spec_.fov_rad, input.sky, spec_.keep_out);

  // --- Availability state machine -------------------------------------------
  //
  // Acquisition and tracking are separate envelopes. A vehicle that slews out of
  // the (wider) tracking envelope cannot simply resume when it drops back under
  // it: it must fall inside the (tighter) acquisition envelope and stay there for
  // the lost-in-space time. That asymmetry is the operationally important part.
  const double rate = input.body_rate.eigen().norm();
  const double accel = input.angular_accel.eigen().norm();
  const bool geometry_ok = m.occlusion.occluder == Occluder::kNone && !fault_dropout_;

  const auto within = [](double value, double limit) { return !(limit > 0.0) || value <= limit; };
  const bool track_rate_ok = within(rate, spec_.tracking_rate_limit);
  const bool track_accel_ok = within(accel, spec_.tracking_accel_limit);
  const bool acquire_rate_ok = within(rate, spec_.acquisition_rate_limit);
  const bool acquire_accel_ok = within(accel, spec_.acquisition_accel_limit);

  if (!geometry_ok) {
    mode_ = StarTrackerMode::kLost;
    acquisition_elapsed_s_ = 0.0;
  } else if (mode_ == StarTrackerMode::kTracking) {
    if (track_rate_ok && track_accel_ok) {
      // Still tracking.
    } else {
      mode_ = StarTrackerMode::kLost;
      acquisition_elapsed_s_ = 0.0;
      m.rate_limited = !track_rate_ok;
      m.accel_limited = !track_accel_ok;
    }
  } else if (acquire_rate_ok && acquire_accel_ok) {
    acquisition_elapsed_s_ += dt;
    mode_ = (acquisition_elapsed_s_ >= spec_.lost_in_space_s) ? StarTrackerMode::kTracking
                                                              : StarTrackerMode::kAcquiring;
    if (mode_ == StarTrackerMode::kTracking) {
      acquisition_elapsed_s_ = 0.0;
    }
  } else {
    // Inside the tracking envelope but outside the acquisition one is exactly the
    // regime where a single-threshold model would wrongly report a solution.
    mode_ = StarTrackerMode::kLost;
    acquisition_elapsed_s_ = 0.0;
    m.rate_limited = !acquire_rate_ok;
    m.accel_limited = !acquire_accel_ok;
  }

  m.mode = mode_;
  m.valid = mode_ == StarTrackerMode::kTracking;
  m.acquisition_elapsed_s = acquisition_elapsed_s_;

  // --- Error composition -----------------------------------------------------
  //
  // The states advance whether or not a solution is reported: the optics keep
  // drifting while the unit is lost, so re-acquisition must not hand back a
  // conveniently reset error.
  const Eigen::Vector3d low_freq = stepSpatial(spec_.low_freq_spatial, spec_.low_freq_correlation_s,
                                               dt, low_freq_state_, low_draw);
  const Eigen::Vector3d high_freq = stepSpatial(
      spec_.high_freq_spatial, spec_.high_freq_correlation_s, dt, high_freq_state_, high_draw);
  const Eigen::Vector3d temporal =
      anisotropic(spec_.temporal, temporal_draw.x(), temporal_draw.y(), temporal_draw.z());
  const Eigen::Vector3d thermo =
      thermo_axis_ * (spec_.thermo_elastic_per_k * input.temperature_delta_k);

  // An ideal tracker reports the truth attitude: the states above still advance
  // (so enabling noise mid-run is not a discontinuity) but no error is applied.
  const Eigen::Vector3d error_body =
      noise_enabled_
          ? Eigen::Vector3d(unit_bias_ + thermo + low_freq + high_freq + temporal + fault_bias_)
          : Eigen::Vector3d(Eigen::Vector3d::Zero());

  // Small-angle rotation vector -> quaternion, applied on the body side:
  // q_meas = δq(error) ⊗ q_truth, so the error is expressed in body axes.
  const double angle = error_body.norm();
  math::Quaternion delta = math::Quaternion::Identity();
  if (angle > 0.0) {
    delta = math::Quaternion::FromAxisAngle(error_body / angle, angle);
  }
  m.attitude = math::Quat<math::frames::Body, math::frames::ECI>(delta * input.attitude.core());
  return m;
}

}  // namespace polaris::sim::sensors
