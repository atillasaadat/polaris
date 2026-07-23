#include "sensors/magnetometer.hpp"

namespace polaris::sim::sensors {
namespace {

double get(const std::map<std::string, double>& p, const std::string& key) {
  const auto it = p.find(key);
  return it == p.end() ? 0.0 : it->second;
}

}  // namespace

VectorErrorModel magnetometerErrorFromParams(const std::map<std::string, double>& params,
                                             std::uint64_t master_seed, std::uint64_t stream_id) {
  VectorErrorModel error;
  constexpr double kMicroTesla = 1.0e-6;
  constexpr double kNanoTesla = 1.0e-9;

  error.range = get(params, "range_ut") * kMicroTesla;
  error.noise_std = Eigen::Vector3d::Constant(get(params, "noise_ut_rms") * kMicroTesla);
  error.resolution = get(params, "resolution_nt") * kNanoTesla;

  const double bias = get(params, "bias_ut") * kMicroTesla;
  const double scale = get(params, "scale_factor_pct") / 100.0;
  const double misalignment = get(params, "misalignment_mrad") * 1.0e-3;

  // The unit's fixed miscalibration is drawn from a stream **derived from, but
  // distinct from**, the one the sensor samples with: the calibration is realised
  // once at build time and the noise stream must start at its own beginning, so
  // that adding or removing a calibration term cannot shift the run's noise.
  random::SplitMix64 build_rng = random::streamRng(master_seed, stream_id ^ 0x9E37U);
  for (int i = 0; i < 3; ++i) {
    error.bias[i] = bias * build_rng.gaussian();
  }
  // Diagonal = per-axis scale error; off-diagonal = soft-iron cross-coupling and
  // axis non-orthogonality, which for a magnetometer are the same matrix.
  for (int row = 0; row < 3; ++row) {
    for (int col = 0; col < 3; ++col) {
      error.scale_misalignment(row, col) =
          (row == col) ? 1.0 + scale * build_rng.gaussian() : misalignment * build_rng.gaussian();
    }
  }
  return error;
}

MagnetometerMeasurement Magnetometer::sample(const time::Tai& epoch,
                                             const math::Vec3<math::frames::Body>& b_truth_body) {
  MagnetometerMeasurement m;
  m.time_tag = epoch;

  // The injected bias jump is a real hard-iron-like field offset, so it enters at
  // the same point as the nominal hard-iron bias — upstream of the ADC's
  // quantization and range saturation — not tacked on afterwards. Adding it here
  // means a large fault saturates to the sensor's rated envelope exactly as a
  // genuine one would, so an FDIR range monitor cannot tell the stimulus from the
  // real thing. Copying the model leaves the noise draw (and thus reproducibility)
  // untouched, since the bias does not feed the RNG.
  VectorErrorModel effective = error_;
  effective.bias += fault_bias_;
  m.field_tesla = math::Vec3<math::frames::Body>(effective.apply(b_truth_body.eigen(), rng_));

  // A dropout still advances the noise stream above (the sensor is sampling; its
  // output is simply flagged unusable), so enabling a dropout does not change the
  // noise any later valid sample sees — keeping the run reproducible.
  m.valid = !fault_dropout_;
  return m;
}

}  // namespace polaris::sim::sensors
