#include "sensors/sun_sensor.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include "constants/constants.hpp"

namespace polaris::sim::sensors {
namespace {

constexpr double kDeg2Rad = 0.017453292519943295;

double get(const std::map<std::string, double>& p, const std::string& key) {
  const auto it = p.find(key);
  return it == p.end() ? 0.0 : it->second;
}

/// Small-angle random tilt of @p v by 1σ @p sigma radians, in two perpendicular
/// directions. Used for the fixed per-diode normal misalignment.
Eigen::Vector3d tilt(const Eigen::Vector3d& v, double sigma, random::SplitMix64& rng) {
  // Two draws always, so whether a unit configures misalignment does not move
  // the stream for anything drawn after it.
  const double a = rng.gaussian();
  const double b = rng.gaussian();
  if (!(sigma > 0.0)) {
    return v;
  }
  Eigen::Index smallest = 0;
  v.cwiseAbs().minCoeff(&smallest);
  Eigen::Vector3d seed = Eigen::Vector3d::Zero();
  seed[smallest] = 1.0;
  const Eigen::Vector3d e1 = v.cross(seed).normalized();
  const Eigen::Vector3d e2 = v.cross(e1).normalized();
  return (v + sigma * (a * e1 + b * e2)).normalized();
}

}  // namespace

SunSensorSpec SunSensorSpec::fromParams(const std::map<std::string, double>& p) {
  SunSensorSpec s;
  // A part that quotes a vector accuracy is a digital part: the accuracy figure
  // only means anything about an output the unit itself computes.
  s.output =
      (get(p, "accuracy_inner_deg_3sigma") > 0.0 || get(p, "accuracy_outer_deg_3sigma") > 0.0)
          ? SunSensorOutput::kSunVector
          : SunSensorOutput::kDiodeCounts;
  s.diode_count = std::max(1, static_cast<int>(get(p, "diode_count")));
  s.diode_cant_rad = get(p, "diode_cant_deg") * kDeg2Rad;
  s.half_fov_rad = get(p, "half_fov_deg") * kDeg2Rad;

  s.full_scale_counts = get(p, "full_scale_counts");
  // A part that quotes no separate ceiling saturates at full scale — the honest
  // default, since a diode cannot report more than its own full-scale reading
  // without headroom the datasheet would have mentioned.
  const double saturation = get(p, "saturation_counts");
  s.saturation_counts = (saturation > 0.0) ? saturation : s.full_scale_counts;
  s.dark_counts = get(p, "dark_counts");
  s.noise_counts = get(p, "noise_counts_rms");
  s.resolution_counts = get(p, "resolution_counts");
  s.scale_factor = get(p, "scale_factor_pct") / 100.0;
  s.alignment_sigma = get(p, "alignment_mrad") * 1.0e-3;
  s.albedo_coefficient = get(p, "albedo_coefficient");

  // Vector-output accuracy. Vendors quote 3σ half-cone angles; the model works
  // in 1σ, so the division happens here rather than in every consumer.
  s.accuracy_inner_half_angle_rad = get(p, "accuracy_inner_half_angle_deg") * kDeg2Rad;
  s.accuracy_inner_sigma = get(p, "accuracy_inner_deg_3sigma") * kDeg2Rad / 3.0;
  s.accuracy_outer_sigma = get(p, "accuracy_outer_deg_3sigma") * kDeg2Rad / 3.0;
  // A part that quotes only one accuracy figure has it apply across its whole
  // field: the outer value falls back to the inner rather than to zero, because
  // zero would silently make the sensor perfect at wide incidence.
  if (!(s.accuracy_outer_sigma > 0.0)) {
    s.accuracy_outer_sigma = s.accuracy_inner_sigma;
  }
  s.albedo_error_rad = get(p, "albedo_error_deg") * kDeg2Rad;
  s.sample_period_s = get(p, "sample_period_ms") * 1.0e-3;

  s.update_rate_hz = get(p, "update_rate_hz");
  const double threshold = get(p, "sun_present_threshold");
  s.sun_present_threshold = (threshold > 0.0) ? threshold : 0.05;
  return s;
}

SunSensor::SunSensor(const SunSensorSpec& spec, const Eigen::Matrix3d& mounting_dcm,
                     std::uint64_t master_seed, std::uint64_t stream_id, bool noise_enabled)
    : spec_(spec), rng_(random::streamRng(master_seed, stream_id)), noise_enabled_(noise_enabled) {
  boresight_body_ = (mounting_dcm * Eigen::Vector3d::UnitZ()).normalized();
  const int n = std::max(1, spec_.diode_count);
  normals_body_.reserve(static_cast<std::size_t>(n));
  diode_scale_.reserve(static_cast<std::size_t>(n));
  diode_failed_.assign(static_cast<std::size_t>(n), false);

  const double cant = spec_.diode_cant_rad;
  for (int i = 0; i < n; ++i) {
    // Diodes sit at equal azimuths around the sensor boresight (+z), canted by
    // the configured angle. One diode degenerates to the boresight itself, which
    // is the coarse single-cell case.
    Eigen::Vector3d normal_sensor = Eigen::Vector3d::UnitZ();
    if (n > 1 && cant != 0.0) {
      const double azimuth = 2.0 * M_PI * static_cast<double>(i) / static_cast<double>(n);
      normal_sensor = Eigen::Vector3d(std::sin(cant) * std::cos(azimuth),
                                      std::sin(cant) * std::sin(azimuth), std::cos(cant));
    }
    // Realise this unit: a fixed normal misalignment and scale error per diode,
    // drawn once, so each seeded sensor is a distinct but in-spec device. An ideal
    // sensor keeps the nominal normal and unit scale (no per-diode error at all).
    const Eigen::Vector3d body_normal = (mounting_dcm * normal_sensor).normalized();
    normals_body_.push_back(noise_enabled ? tilt(body_normal, spec_.alignment_sigma, rng_)
                                          : body_normal);
    diode_scale_.push_back(noise_enabled ? 1.0 + spec_.scale_factor * rng_.gaussian() : 1.0);
  }
}

void SunSensor::failDiode(int index, bool failed) {
  if (index >= 0 && static_cast<std::size_t>(index) < diode_failed_.size()) {
    diode_failed_[static_cast<std::size_t>(index)] = failed;
  }
}

void SunSensor::clearFaults() {
  std::fill(diode_failed_.begin(), diode_failed_.end(), false);
  fault_dropout_ = false;
}

SunSensorMeasurement SunSensor::sample(const time::Tai& epoch, const SunSensorInput& input) {
  // Sample-period gating. A bus-attached part integrates over its own period;
  // polling it faster returns the register contents again, not a fresh
  // independent reading. Modelling that matters because an estimator handed
  // repeated values as if they were independent averages down noise that never
  // averaged, and grows confident on information it does not have.
  if (has_sampled_ && spec_.sample_period_s > 0.0) {
    const double elapsed = static_cast<double>(epoch.nanosecondsSinceEpoch() -
                                               last_sample_time_.nanosecondsSinceEpoch()) *
                           1.0e-9;
    if (elapsed >= 0.0 && elapsed < spec_.sample_period_s) {
      SunSensorMeasurement repeated = last_measurement_;
      repeated.time_tag = epoch;
      repeated.fresh = false;
      return repeated;
    }
  }

  SunSensorMeasurement m;
  m.time_tag = epoch;
  m.shadow_factor = input.shadow_factor;
  const std::size_t n = normals_body_.size();
  m.counts.assign(n, 0.0);
  m.albedo_counts.assign(n, 0.0);

  const Eigen::Vector3d sun_dir = input.sun_dir_body.eigen();
  const double sun_norm = sun_dir.norm();
  const bool sun_usable = sun_norm > 0.0 && input.shadow_factor > spec_.sun_present_threshold;
  const Eigen::Vector3d sun_hat =
      (sun_norm > 0.0) ? Eigen::Vector3d(sun_dir / sun_norm) : Eigen::Vector3d::Zero();

  // How sunlit the ground below is: the albedo a diode sees comes from the
  // sunlit part of the Earth in its field of view, and there is none on the
  // night side. First-order — a cosine phase factor, no terrain, no clouds.
  double dayside = 0.0;
  const double sat_norm = input.sky.sat.norm();
  const double sun_pos_norm = input.sky.sun.norm();
  if (sat_norm > 0.0 && sun_pos_norm > 0.0) {
    dayside = std::max(0.0, input.sky.sat.dot(input.sky.sun) / (sat_norm * sun_pos_norm));
  }

  // Earth geometry for the albedo view factor: its apparent angular radius, and
  // which way it is. Both zero out cleanly if the caller supplied no geometry.
  const Eigen::Vector3d nadir = input.nadir_dir_body.eigen();
  const double nadir_norm = nadir.norm();
  const Eigen::Vector3d nadir_hat =
      (nadir_norm > 0.0) ? Eigen::Vector3d(nadir / nadir_norm) : Eigen::Vector3d::Zero();
  double earth_angular_radius = 0.0;
  if (nadir_norm > 0.0 && sat_norm > constants::wgs84::kSemiMajorAxis) {
    earth_angular_radius = std::asin(constants::wgs84::kSemiMajorAxis / sat_norm);
  }

  const double cos_fov = std::cos(spec_.half_fov_rad);
  bool any_illuminated = false;

  // Truth incidence angle from the boresight — what selects the accuracy regime
  // for a vector-output part, and a useful diagnostic for an analogue one.
  m.incidence_angle_rad =
      sun_norm > 0.0 ? std::acos(std::clamp(boresight_body_.dot(sun_hat), -1.0, 1.0)) : M_PI;

  if (spec_.output == SunSensorOutput::kSunVector) {
    // The unit's own processing produces the vector; the datasheet specifies its
    // accuracy, so the model perturbs the truth direction by that much rather
    // than inventing the photocurrents and the proprietary calibration that
    // would have to undo them.
    const bool in_fov =
        sun_usable && (spec_.half_fov_rad <= 0.0 || m.incidence_angle_rad <= spec_.half_fov_rad);

    // Accuracy degrades with incidence angle: the quoted inner figure holds out
    // to the inner half-angle, the (worse) outer figure to the edge of the field.
    double sigma = (m.incidence_angle_rad <= spec_.accuracy_inner_half_angle_rad)
                       ? spec_.accuracy_inner_sigma
                       : spec_.accuracy_outer_sigma;

    // Albedo adds on top, scaled by how much sunlit Earth is in the field. This
    // is the term that dominates in LEO — vendors quote clean-sky accuracy and
    // then warn that uncorrected albedo costs an order of magnitude more.
    double albedo_sigma = 0.0;
    if (spec_.albedo_error_rad > 0.0 && dayside > 0.0 && spec_.half_fov_rad > 0.0 &&
        earth_angular_radius > 0.0) {
      const double separation = std::acos(std::clamp(boresight_body_.dot(nadir_hat), -1.0, 1.0));
      const double fraction =
          fovCoveredFraction(spec_.half_fov_rad, separation, earth_angular_radius);
      albedo_sigma = spec_.albedo_error_rad * fraction * dayside;
    }
    // Independent mechanisms add in quadrature. An ideal sensor reports the exact
    // truth direction, so the whole error collapses to zero.
    sigma = noise_enabled_ ? std::sqrt(sigma * sigma + albedo_sigma * albedo_sigma) : 0.0;
    m.accuracy_sigma_rad = sigma;

    // Two draws always, so the stream position does not depend on the geometry.
    const double g1 = rng_.gaussian();
    const double g2 = rng_.gaussian();
    if (in_fov && !fault_dropout_) {
      // Perturb the direction in the two axes perpendicular to it: a unit vector
      // error is a small rotation, not an additive vector, so this keeps the
      // result on the sphere instead of quietly changing its length.
      Eigen::Index smallest = 0;
      sun_hat.cwiseAbs().minCoeff(&smallest);
      Eigen::Vector3d seed = Eigen::Vector3d::Zero();
      seed[smallest] = 1.0;
      const Eigen::Vector3d e1 = sun_hat.cross(seed).normalized();
      const Eigen::Vector3d e2 = sun_hat.cross(e1).normalized();
      m.sun_dir_body =
          math::Vec3<math::frames::Body>((sun_hat + sigma * (g1 * e1 + g2 * e2)).normalized());
      any_illuminated = true;
    }

    // No photocurrents to report. Left **empty** rather than zero-filled: a
    // vector of zeros is indistinguishable from a set of dark cells, and a
    // consumer reading counts[i] on a digital part deserves to find nothing
    // there rather than a plausible-looking reading that means nothing.
    m.counts.clear();
    m.albedo_counts.clear();

    m.sun_present = sun_usable && in_fov;
    m.valid = m.sun_present && !fault_dropout_;
    last_measurement_ = m;
    last_sample_time_ = epoch;
    has_sampled_ = true;
    return m;
  }

  for (std::size_t i = 0; i < n; ++i) {
    const Eigen::Vector3d& normal = normals_body_[i];

    // Direct sunlight: the cosine law, cut off outside the acceptance cone.
    double direct = 0.0;
    const double cos_incidence = sun_usable ? normal.dot(sun_hat) : 0.0;
    const bool illuminated = sun_usable && cos_incidence > 0.0 &&
                             (spec_.half_fov_rad <= 0.0 || cos_incidence >= cos_fov);
    if (illuminated) {
      direct = spec_.full_scale_counts * diode_scale_[i] * cos_incidence * input.shadow_factor;
      any_illuminated = true;
    }

    // Earthshine: how much of this diode's view the Earth fills, times how
    // sunlit that ground is. The occlusion model (§6.1) supplies the fraction,
    // so a diode and a star tracker cannot disagree about where the Earth is.
    double albedo = 0.0;
    if (noise_enabled_ && spec_.albedo_coefficient > 0.0 && dayside > 0.0 &&
        spec_.half_fov_rad > 0.0 && earth_angular_radius > 0.0) {
      // Both vectors are in body axes, and the fraction depends only on the angle
      // between them, so no frame change is needed. The shared §6.1 helper does
      // the overlap, which is what keeps a diode and a star tracker from
      // disagreeing about how much Earth is in view.
      const double separation = std::acos(std::clamp(normal.dot(nadir_hat), -1.0, 1.0));
      const double fraction =
          fovCoveredFraction(spec_.half_fov_rad, separation, earth_angular_radius);
      albedo = spec_.albedo_coefficient * spec_.full_scale_counts * fraction * dayside;
    }
    m.albedo_counts[i] = albedo;

    // An ideal cell reports the clean cosine signal: no dark current, no noise,
    // no quantization. The rng draw still happens so the stream stays aligned.
    double count = direct + albedo + (noise_enabled_ ? spec_.dark_counts : 0.0);
    const double noise_draw = rng_.gaussian();
    if (noise_enabled_ && spec_.noise_counts > 0.0) {
      count += spec_.noise_counts * noise_draw;
    }
    if (diode_failed_[i]) {
      count = 0.0;  // a dead cell reads dark, not noisy
      m.albedo_counts[i] = 0.0;
    }
    if (noise_enabled_ && spec_.resolution_counts > 0.0) {
      count = std::round(count / spec_.resolution_counts) * spec_.resolution_counts;
    }
    // A photodiode reading cannot go negative, and cannot exceed the ADC range.
    m.counts[i] =
        std::clamp(count, 0.0,
                   spec_.saturation_counts > 0.0 ? spec_.saturation_counts
                                                 : std::numeric_limits<double>::infinity());
  }

  m.sun_present = sun_usable && any_illuminated;
  m.valid = m.sun_present && !fault_dropout_;
  last_measurement_ = m;
  last_sample_time_ = epoch;
  has_sampled_ = true;
  return m;
}

}  // namespace polaris::sim::sensors
