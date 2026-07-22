#include "sensors/magnetometer.hpp"

namespace polaris::sim::sensors {

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
