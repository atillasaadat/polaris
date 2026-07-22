/// @file Unit tests for the magnetometer truth model and the §6.1 error stack.
///
/// Two layers. The error-stack tests pin each imperfection in isolation — an
/// identity model must be a pass-through, and bias / scale-misalignment / noise /
/// quantization / saturation each move the output the way and only the way they
/// should. The magnetometer tests pin the sensor contract that FDIR and the
/// estimators rely on: bit-reproducibility from {config, seed}, independent noise
/// per source, and fault-injection hooks that are indistinguishable from the real
/// faults they stand in for while keeping the run reproducible.

#include <gtest/gtest.h>

#include <cmath>
#include <Eigen/Core>

#include "math/frames.hpp"
#include "math/typed_vector.hpp"
#include "random/rng.hpp"
#include "sensors/magnetometer.hpp"
#include "sensors/sensor_error.hpp"
#include "time/timescales.hpp"

namespace sensors = polaris::sim::sensors;
namespace pm = polaris::math;
namespace pmf = polaris::math::frames;
namespace pt = polaris::time;

namespace {
using Vec3B = pm::Vec3<pmf::Body>;
const pt::Tai kEpoch = pt::Tai::fromNanosecondsSinceEpoch(1767225637000000000LL);
const Eigen::Vector3d kTruth(30000e-9, -5000e-9, 42000e-9);  // a LEO-ish field [T]
}  // namespace

// --- Error stack -------------------------------------------------------------

TEST(VectorErrorModel, IdentityIsAPassThrough) {
  sensors::VectorErrorModel model;  // all defaults
  polaris::random::SplitMix64 rng(1);
  const Eigen::Vector3d out = model.apply(kTruth, rng);
  EXPECT_TRUE(out.isApprox(kTruth));
}

TEST(VectorErrorModel, BiasAndMatrixApplyBeforeNoise) {
  sensors::VectorErrorModel model;
  model.scale_misalignment = Eigen::Vector3d(1.01, 0.99, 1.0).asDiagonal();
  model.scale_misalignment(0, 1) = 0.02;  // cross-axis coupling
  model.bias = Eigen::Vector3d(100e-9, -50e-9, 25e-9);
  polaris::random::SplitMix64 rng(1);
  const Eigen::Vector3d expected = model.scale_misalignment * kTruth + model.bias;
  EXPECT_TRUE(model.apply(kTruth, rng).isApprox(expected));
}

TEST(VectorErrorModel, NoiseHasTheConfiguredPerAxisSigma) {
  sensors::VectorErrorModel model;
  model.noise_std = Eigen::Vector3d(200e-9, 400e-9, 800e-9);
  polaris::random::SplitMix64 rng(7);
  constexpr int n = 100000;
  Eigen::Vector3d sum = Eigen::Vector3d::Zero();
  Eigen::Vector3d sum_sq = Eigen::Vector3d::Zero();
  for (int i = 0; i < n; ++i) {
    const Eigen::Vector3d e = model.apply(kTruth, rng) - kTruth;  // isolate the error
    sum += e;
    sum_sq += e.cwiseProduct(e);
  }
  const Eigen::Vector3d mean = sum / n;
  const Eigen::Vector3d std = (sum_sq / n - mean.cwiseProduct(mean)).cwiseSqrt();
  for (int i = 0; i < 3; ++i) {
    EXPECT_NEAR(mean[i], 0.0, model.noise_std[i] * 0.05) << "axis " << i;
    EXPECT_NEAR(std[i], model.noise_std[i], model.noise_std[i] * 0.05) << "axis " << i;
  }
}

TEST(VectorErrorModel, QuantizationSnapsToTheLsb) {
  sensors::VectorErrorModel model;
  model.resolution = 1000e-9;  // 1000 nT LSB
  polaris::random::SplitMix64 rng(1);
  const Eigen::Vector3d out = model.apply(kTruth, rng);
  for (int i = 0; i < 3; ++i) {
    EXPECT_NEAR(std::remainder(out[i], model.resolution), 0.0, 1e-18) << "axis " << i;
    EXPECT_LE(std::abs(out[i] - kTruth[i]), model.resolution) << "axis " << i;
  }
}

TEST(VectorErrorModel, SaturationClampsToRange) {
  sensors::VectorErrorModel model;
  model.range = 20000e-9;  // ±20000 nT; truth exceeds this on two axes
  polaris::random::SplitMix64 rng(1);
  const Eigen::Vector3d out = model.apply(kTruth, rng);
  EXPECT_DOUBLE_EQ(out[0], model.range);  // +30000 -> +20000
  EXPECT_DOUBLE_EQ(out[1], -5000e-9);     // within range, untouched
  EXPECT_DOUBLE_EQ(out[2], model.range);  // +42000 -> +20000
}

// --- Magnetometer ------------------------------------------------------------

sensors::VectorErrorModel noisyModel() {
  sensors::VectorErrorModel model;
  model.bias = Eigen::Vector3d(150e-9, -100e-9, 80e-9);
  model.noise_std = Eigen::Vector3d(300e-9, 300e-9, 300e-9);
  return model;
}

TEST(Magnetometer, IsBitReproducibleFromSeed) {
  sensors::Magnetometer a(noisyModel(), 0xABCD, 1);
  sensors::Magnetometer b(noisyModel(), 0xABCD, 1);
  for (int i = 0; i < 100; ++i) {
    const auto ma = a.sample(kEpoch, Vec3B(kTruth));
    const auto mb = b.sample(kEpoch, Vec3B(kTruth));
    EXPECT_EQ(ma.field_tesla.eigen(), mb.field_tesla.eigen()) << "sample " << i;
  }
}

TEST(Magnetometer, DistinctStreamsGiveDistinctNoise) {
  sensors::Magnetometer a(noisyModel(), 0xABCD, 1);
  sensors::Magnetometer b(noisyModel(), 0xABCD, 2);  // different source id
  const auto ma = a.sample(kEpoch, Vec3B(kTruth));
  const auto mb = b.sample(kEpoch, Vec3B(kTruth));
  EXPECT_FALSE(ma.field_tesla.eigen().isApprox(mb.field_tesla.eigen()));
}

TEST(Magnetometer, BiasJumpFaultShiftsTheMeasurement) {
  const Eigen::Vector3d jump(2000e-9, 0.0, -1500e-9);
  sensors::Magnetometer nominal(noisyModel(), 0xABCD, 1);
  sensors::Magnetometer faulted(noisyModel(), 0xABCD, 1);
  faulted.injectBiasJump(Vec3B(jump));

  // Same seed + same draw order -> the only difference is the injected jump.
  const auto mn = nominal.sample(kEpoch, Vec3B(kTruth));
  const auto mf = faulted.sample(kEpoch, Vec3B(kTruth));
  EXPECT_TRUE((mf.field_tesla.eigen() - mn.field_tesla.eigen()).isApprox(jump));
  EXPECT_TRUE(mf.valid);  // a bias jump is a silent fault: still "valid"

  faulted.clearFaults();
  const auto mc = faulted.sample(kEpoch, Vec3B(kTruth));
  // After clearing, it tracks the nominal sensor's continuing sequence exactly.
  const auto mn2 = nominal.sample(kEpoch, Vec3B(kTruth));
  EXPECT_EQ(mc.field_tesla.eigen(), mn2.field_tesla.eigen());
}

TEST(Magnetometer, BiasJumpIsSubjectToRangeSaturation) {
  // A hard-iron-like fault enters upstream of the ADC, so a jump that would push
  // the reading past the sensor's rated range saturates there — it cannot report
  // a physically impossible field. An FDIR envelope monitor therefore sees the
  // same clamped value it would from any other over-range cause.
  sensors::VectorErrorModel model;
  model.range = 60000e-9;  // ±60000 nT rated envelope
  sensors::Magnetometer mag(model, 0xABCD, 1);
  mag.injectBiasJump(Vec3B(Eigen::Vector3d(1.0, 0.0, 0.0)));  // 1 T: absurdly large
  const auto m = mag.sample(kEpoch, Vec3B(kTruth));
  EXPECT_DOUBLE_EQ(m.field_tesla.eigen()[0], model.range) << "fault must clamp to range";
}

TEST(Magnetometer, DropoutFlagsInvalidButKeepsTheStreamAligned) {
  sensors::Magnetometer dropped(noisyModel(), 0xABCD, 1);
  sensors::Magnetometer reference(noisyModel(), 0xABCD, 1);

  dropped.setDropout(true);
  const auto md = dropped.sample(kEpoch, Vec3B(kTruth));
  const auto mr = reference.sample(kEpoch, Vec3B(kTruth));
  EXPECT_FALSE(md.valid);
  EXPECT_TRUE(mr.valid);
  // The dropped sample still consumed its noise draw, so the two sensors remain
  // phase-aligned: after clearing the dropout the next samples match.
  dropped.setDropout(false);
  const auto md2 = dropped.sample(kEpoch, Vec3B(kTruth));
  const auto mr2 = reference.sample(kEpoch, Vec3B(kTruth));
  EXPECT_TRUE(md2.valid);
  EXPECT_EQ(md2.field_tesla.eigen(), mr2.field_tesla.eigen());
}
