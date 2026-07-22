#include "sensors/imu.hpp"

#include <cmath>

namespace polaris::sim::sensors {

namespace {

constexpr double kDeg2Rad = 0.017453292519943295;
constexpr double kG = 9.80665;  // standard gravity [m/s²]

// Datasheet-native -> SI conversions.
constexpr double degPerHrToRadPerS(double v) {
  return v * kDeg2Rad / 3600.0;
}

constexpr double degPerSqrtHrToRadPerSqrtS(double v) {
  return v * kDeg2Rad / 60.0;
}  // √3600 = 60

constexpr double mgToMps2(double v) {
  return v * 1.0e-3 * kG;
}

constexpr double ugToMps2(double v) {
  return v * 1.0e-6 * kG;
}

double get(const std::map<std::string, double>& p, const std::string& key) {
  const auto it = p.find(key);
  return it == p.end() ? 0.0 : it->second;
}

/// Three independent Gaussian draws scaled by @p sigma. Always consumes exactly
/// three draws so a triad's stream position does not depend on @p sigma.
Eigen::Vector3d drawGaussian(random::SplitMix64& rng, double sigma) {
  return {sigma * rng.gaussian(), sigma * rng.gaussian(), sigma * rng.gaussian()};
}

/// The per-unit scale-factor + misalignment matrix, M = I + diag(scale) +
/// skew(misalignment). Both errors are drawn once from their 1σ datasheet bounds,
/// so the modelled unit is a specific, fixed, in-spec device.
Eigen::Matrix3d drawErrorMatrix(random::SplitMix64& rng, double scale_sigma,
                                double misalign_sigma) {
  const Eigen::Vector3d s = drawGaussian(rng, scale_sigma);     // scale-factor errors
  const Eigen::Vector3d a = drawGaussian(rng, misalign_sigma);  // small misalignment angles
  Eigen::Matrix3d m = Eigen::Matrix3d::Identity();
  m(0, 0) += s[0];
  m(1, 1) += s[1];
  m(2, 2) += s[2];
  // + skew(a): the small-angle axis misalignment / non-orthogonality.
  m(0, 1) += -a[2];
  m(0, 2) += a[1];
  m(1, 0) += a[2];
  m(1, 2) += -a[0];
  m(2, 0) += -a[1];
  m(2, 1) += a[0];
  return m;
}

}  // namespace

ImuSpec ImuSpec::fromParams(const std::map<std::string, double>& p) {
  ImuSpec spec;

  spec.gyro.range = get(p, "gyro_range_deg_s") * kDeg2Rad;
  spec.gyro.random_walk = degPerSqrtHrToRadPerSqrtS(get(p, "gyro_arw_deg_sqrt_hr"));
  spec.gyro.bias_instability = degPerHrToRadPerS(get(p, "gyro_bias_instability_deg_hr"));
  spec.gyro.bias_correlation_s = get(p, "gyro_bias_correlation_s");
  spec.gyro.bias_repeatability = degPerHrToRadPerS(get(p, "gyro_bias_repeatability_deg_hr"));
  spec.gyro.scale_factor = get(p, "gyro_scale_factor_ppm") * 1.0e-6;
  spec.gyro.misalignment = get(p, "gyro_misalignment_mrad") * 1.0e-3;
  spec.gyro.resolution = degPerHrToRadPerS(get(p, "gyro_resolution_deg_hr"));

  spec.accel.range = get(p, "accel_range_g") * kG;
  spec.accel.random_walk = get(p, "accel_vrw_m_s_sqrt_hr") / 60.0;  // (m/s)/√h -> (m/s)/√s
  spec.accel.bias_instability = mgToMps2(get(p, "accel_bias_instability_mg"));
  spec.accel.bias_correlation_s = get(p, "accel_bias_correlation_s");
  spec.accel.bias_repeatability = mgToMps2(get(p, "accel_bias_repeatability_mg"));
  spec.accel.scale_factor = get(p, "accel_scale_factor_ppm") * 1.0e-6;
  spec.accel.misalignment = get(p, "accel_misalignment_mrad") * 1.0e-3;
  spec.accel.resolution = ugToMps2(get(p, "accel_resolution_ug"));

  spec.gyro_g_sensitivity = degPerHrToRadPerS(get(p, "gyro_g_sensitivity_deg_hr_g")) / kG;
  spec.sample_rate_hz = get(p, "sample_rate_hz");
  return spec;
}

Imu::Imu(const ImuSpec& spec, std::uint64_t master_seed, std::uint64_t stream_id)
    : spec_(spec), rng_(random::streamRng(master_seed, stream_id)) {
  // Realise this unit's fixed miscalibration and turn-on biases once, in a fixed
  // draw order, so {spec, seed, stream_id} always builds the same device.
  gyro_err_.scale_misalignment =
      drawErrorMatrix(rng_, spec_.gyro.scale_factor, spec_.gyro.misalignment);
  accel_err_.scale_misalignment =
      drawErrorMatrix(rng_, spec_.accel.scale_factor, spec_.accel.misalignment);
  gyro_err_.resolution = spec_.gyro.resolution;
  gyro_err_.range = spec_.gyro.range;
  accel_err_.resolution = spec_.accel.resolution;
  accel_err_.range = spec_.accel.range;

  gyro_turn_on_ = drawGaussian(rng_, spec_.gyro.bias_repeatability);
  accel_turn_on_ = drawGaussian(rng_, spec_.accel.bias_repeatability);

  // Start the Gauss-Markov drift at its stationary distribution (σ = bias
  // instability), so it does not have to "warm up" from zero. With no correlation
  // time there is no drift process, so it starts (and stays) at zero.
  gyro_drift_ =
      drawGaussian(rng_, spec_.gyro.bias_correlation_s > 0.0 ? spec_.gyro.bias_instability : 0.0);
  accel_drift_ =
      drawGaussian(rng_, spec_.accel.bias_correlation_s > 0.0 ? spec_.accel.bias_instability : 0.0);
}

Eigen::Vector3d Imu::measureTriad(const TriadSpec& spec, VectorErrorModel& err,
                                  const Eigen::Vector3d& turn_on_bias, Eigen::Vector3d& drift_state,
                                  const Eigen::Vector3d& extra_bias, const Eigen::Vector3d& truth,
                                  double dt) {
  // First-order Gauss-Markov in-run bias drift: b_{k+1} = φ b_k + w, with
  // φ = exp(-dt/τ) and w ~ N(0, σ²(1-φ²)) so the stationary std stays σ. τ ≤ 0
  // disables the drift (φ = 1, w = 0) — the bias is then just turn-on + faults.
  double phi = 1.0;
  double q_sigma = 0.0;
  if (spec.bias_correlation_s > 0.0) {
    phi = std::exp(-dt / spec.bias_correlation_s);
    q_sigma = spec.bias_instability * std::sqrt(std::max(0.0, 1.0 - phi * phi));
  }
  // Always draw three (q_sigma = 0 when the drift is disabled) so the stream
  // position does not depend on whether this triad models drift.
  const Eigen::Vector3d w = drawGaussian(rng_, q_sigma);
  drift_state = phi * drift_state + w;

  err.bias = turn_on_bias + drift_state + extra_bias;
  // White noise from the random walk: per-sample σ = random_walk / √dt.
  const double noise_sigma = (dt > 0.0) ? spec.random_walk / std::sqrt(dt) : 0.0;
  err.noise_std = Eigen::Vector3d::Constant(noise_sigma);

  return err.apply(truth, rng_);
}

ImuSample Imu::sample(const time::Tai& epoch, double dt,
                      const math::Vec3<math::frames::Body>& true_rate_body,
                      const math::Vec3<math::frames::Body>& true_specific_force_body) {
  ImuSample s;
  s.time_tag = epoch;

  // A non-positive (or NaN) interval is not a measurement: return an invalid
  // sample without drawing, so a stray dt<=0 tick cannot half-advance the noise
  // stream and desync every later sample (the reproducibility invariant). This
  // mirrors RigidBody6Dof::propagate's dt<=0 no-op.
  if (!(dt > 0.0)) {
    s.valid = false;
    return s;
  }

  const Eigen::Vector3d true_sf = true_specific_force_body.eigen();
  // Gyro g-sensitivity: a rate bias proportional to the specific force.
  const Eigen::Vector3d gyro_g = spec_.gyro_g_sensitivity * true_sf;

  const Eigen::Vector3d rate = measureTriad(spec_.gyro, gyro_err_, gyro_turn_on_, gyro_drift_,
                                            gyro_g + gyro_fault_bias_, true_rate_body.eigen(), dt);
  const Eigen::Vector3d sf = measureTriad(spec_.accel, accel_err_, accel_turn_on_, accel_drift_,
                                          accel_fault_bias_, true_sf, dt);

  s.angular_rate_rads = math::Vec3<math::frames::Body>(rate);
  s.specific_force_mps2 = math::Vec3<math::frames::Body>(sf);
  s.delta_angle_rad = math::Vec3<math::frames::Body>(rate * dt);
  s.delta_velocity_mps = math::Vec3<math::frames::Body>(sf * dt);
  s.valid = !fault_dropout_;
  return s;
}

namespace catalog {

ImuSpec stim377h() {
  // Safran/Sensonor STIM377H product brief, Ed. 2022-04. Datasheet-native units;
  // bias_correlation_s is a modelling choice (not a datasheet field) — a
  // representative in-run correlation time for a tactical MEMS gyro/accel.
  return ImuSpec::fromParams({
      {"gyro_range_deg_s", 480.0},               // output cap (input range ±400 °/s)
      {"gyro_arw_deg_sqrt_hr", 0.15},            // angular random walk
      {"gyro_bias_instability_deg_hr", 0.3},     // in-run bias instability
      {"gyro_bias_correlation_s", 100.0},        // modelling choice
      {"gyro_bias_repeatability_deg_hr", 10.0},  // bias error over temp gradients, rms
      {"gyro_scale_factor_ppm", 500.0},          // scale-factor accuracy
      {"gyro_misalignment_mrad", 1.0},           // electronic axis alignment
      {"gyro_resolution_deg_hr", 0.22},          // resolution
      {"gyro_g_sensitivity_deg_hr_g", 7.0},      // linear-accel bias effect (no g-comp)
      {"accel_range_g", 10.0},                   // standard input range
      {"accel_vrw_m_s_sqrt_hr", 0.07},           // velocity random walk
      {"accel_bias_instability_mg", 0.04},       // bias instability
      {"accel_bias_correlation_s", 100.0},       // modelling choice
      {"accel_bias_repeatability_mg", 2.0},      // bias error over temp gradients, rms
      {"accel_scale_factor_ppm", 200.0},         // scale-factor accuracy
      {"accel_misalignment_mrad", 1.0},          // electronic axis alignment
      {"accel_resolution_ug", 1.9},              // resolution
      {"sample_rate_hz", 2000.0},                // max sample rate
  });
}

ImuSpec genericTactical() {
  // Representative tactical-grade MEMS IMU — round numbers to copy and refine.
  return ImuSpec::fromParams({
      {"gyro_range_deg_s", 500.0},
      {"gyro_arw_deg_sqrt_hr", 0.2},
      {"gyro_bias_instability_deg_hr", 1.0},
      {"gyro_bias_correlation_s", 100.0},
      {"gyro_bias_repeatability_deg_hr", 20.0},
      {"gyro_scale_factor_ppm", 1000.0},
      {"gyro_misalignment_mrad", 2.0},
      {"gyro_resolution_deg_hr", 1.0},
      {"gyro_g_sensitivity_deg_hr_g", 10.0},
      {"accel_range_g", 10.0},
      {"accel_vrw_m_s_sqrt_hr", 0.1},
      {"accel_bias_instability_mg", 0.1},
      {"accel_bias_correlation_s", 100.0},
      {"accel_bias_repeatability_mg", 5.0},
      {"accel_scale_factor_ppm", 500.0},
      {"accel_misalignment_mrad", 2.0},
      {"accel_resolution_ug", 10.0},
      {"sample_rate_hz", 250.0},
  });
}

}  // namespace catalog

}  // namespace polaris::sim::sensors
