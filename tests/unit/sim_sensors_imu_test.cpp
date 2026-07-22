/// @file Unit tests for the IMU truth model and its datasheet-driven spec.
///
/// Three concerns. (1) The datasheet→SI conversion: the STIM377H catalog spec must
/// carry the product-brief numbers in SI, so a unit slip there is caught before it
/// silently mis-scales every run. (2) The error model itself: a perfect spec is a
/// pass-through; the random-walk noise scales as 1/√dt; the in-run bias is a
/// stationary, temporally-correlated Gauss-Markov process (not white); g-sensitivity
/// couples gyro bias to specific force. (3) The sensor contract: reproducible from
/// {spec, seed}, independent per stream, delta-angle = rate·dt, and fault hooks.

#include <gtest/gtest.h>

#include <cmath>
#include <Eigen/Core>
#include <vector>

#include "math/frames.hpp"
#include "math/typed_vector.hpp"
#include "sensors/imu.hpp"
#include "time/timescales.hpp"

namespace sensors = polaris::sim::sensors;
namespace pm = polaris::math;
namespace pmf = polaris::math::frames;
namespace pt = polaris::time;

namespace {
using Vec3B = pm::Vec3<pmf::Body>;
constexpr double kDeg2Rad = 0.017453292519943295;
constexpr double kG = 9.80665;
const pt::Tai kEpoch = pt::Tai::fromNanosecondsSinceEpoch(1767225637000000000LL);

/// An IMU spec with every error term off — a perfect instrument.
sensors::ImuSpec perfectSpec() {
  return sensors::ImuSpec{};
}
}  // namespace

// --- Datasheet -> SI ---------------------------------------------------------

TEST(ImuSpec, Stim377hMatchesTheDatasheetInSi) {
  const sensors::ImuSpec s = sensors::catalog::stim377h();
  // Gyro
  EXPECT_NEAR(s.gyro.random_walk, 0.15 * kDeg2Rad / 60.0, 1e-15);           // ARW [rad/√s]
  EXPECT_NEAR(s.gyro.bias_instability, 0.3 * kDeg2Rad / 3600.0, 1e-18);     // [rad/s]
  EXPECT_NEAR(s.gyro.bias_repeatability, 10.0 * kDeg2Rad / 3600.0, 1e-16);  // [rad/s]
  EXPECT_NEAR(s.gyro.scale_factor, 500e-6, 1e-12);
  EXPECT_NEAR(s.gyro.misalignment, 1e-3, 1e-15);
  EXPECT_NEAR(s.gyro.resolution, 0.22 * kDeg2Rad / 3600.0, 1e-18);
  EXPECT_NEAR(s.gyro.range, 480.0 * kDeg2Rad, 1e-12);
  EXPECT_NEAR(s.gyro_g_sensitivity, (7.0 * kDeg2Rad / 3600.0) / kG, 1e-18);
  // Accel
  EXPECT_NEAR(s.accel.random_walk, 0.07 / 60.0, 1e-15);        // VRW [(m/s)/√s]
  EXPECT_NEAR(s.accel.bias_instability, 0.04e-3 * kG, 1e-12);  // [m/s²]
  EXPECT_NEAR(s.accel.bias_repeatability, 2.0e-3 * kG, 1e-12);
  EXPECT_NEAR(s.accel.scale_factor, 200e-6, 1e-12);
  EXPECT_NEAR(s.accel.resolution, 1.9e-6 * kG, 1e-15);
  EXPECT_NEAR(s.accel.range, 10.0 * kG, 1e-12);
}

// --- Error model -------------------------------------------------------------

TEST(Imu, PerfectSpecIsAPassThrough) {
  sensors::Imu imu(perfectSpec(), 1, 1);
  const Eigen::Vector3d rate(0.01, -0.02, 0.03);
  const Eigen::Vector3d sf(0.1, 0.2, -0.3);
  const double dt = 0.1;
  const auto m = imu.sample(kEpoch, dt, Vec3B(rate), Vec3B(sf));
  EXPECT_TRUE(m.angular_rate_rads.eigen().isApprox(rate));
  EXPECT_TRUE(m.specific_force_mps2.eigen().isApprox(sf));
  EXPECT_TRUE(m.delta_angle_rad.eigen().isApprox(rate * dt));
  EXPECT_TRUE(m.delta_velocity_mps.eigen().isApprox(sf * dt));
}

TEST(Imu, IsBitReproducibleFromSeed) {
  sensors::Imu a(sensors::catalog::stim377h(), 0xABCD, 1);
  sensors::Imu b(sensors::catalog::stim377h(), 0xABCD, 1);
  const Eigen::Vector3d rate(0.01, 0.0, 0.0);
  const Eigen::Vector3d sf(0.0, 0.0, -kG);
  for (int i = 0; i < 100; ++i) {
    EXPECT_EQ(a.sample(kEpoch, 0.01, Vec3B(rate), Vec3B(sf)).angular_rate_rads.eigen(),
              b.sample(kEpoch, 0.01, Vec3B(rate), Vec3B(sf)).angular_rate_rads.eigen())
        << "sample " << i;
  }
}

TEST(Imu, RandomWalkNoiseScalesAsInverseSqrtDt) {
  // Only gyro ARW active: the per-sample rate-noise σ must be ARW/√dt, so halving
  // dt raises the σ by √2. This is the property that makes delta-angle noise
  // integrate correctly regardless of sample rate.
  sensors::ImuSpec spec;
  spec.gyro.random_walk = 1.0e-4;  // rad/√s
  const Eigen::Vector3d rate = Eigen::Vector3d::Zero();
  const Eigen::Vector3d sf = Eigen::Vector3d::Zero();

  auto rateNoiseStd = [&](double dt) {
    sensors::Imu imu(spec, 7, 1);
    constexpr int n = 200000;
    double sum = 0.0;
    double sum_sq = 0.0;
    for (int i = 0; i < n; ++i) {
      const double x = imu.sample(kEpoch, dt, Vec3B(rate), Vec3B(sf)).angular_rate_rads.eigen()[0];
      sum += x;
      sum_sq += x * x;
    }
    const double mean = sum / n;
    return std::sqrt(sum_sq / n - mean * mean);
  };

  const double s1 = rateNoiseStd(0.01);
  const double s2 = rateNoiseStd(0.0025);  // dt/4 -> σ should double
  EXPECT_NEAR(s1, 1.0e-4 / std::sqrt(0.01), 1.0e-4 / std::sqrt(0.01) * 0.05);
  EXPECT_NEAR(s2 / s1, 2.0, 0.05);
}

TEST(Imu, InRunBiasIsAStationaryCorrelatedProcess) {
  // Only the gyro Gauss-Markov drift active: over a long run its std must sit near
  // the datasheet bias instability (stationary), and consecutive samples must be
  // strongly correlated — a white bias would fail the lag-1 autocorrelation.
  sensors::ImuSpec spec;
  spec.gyro.bias_instability = 1.0e-6;  // rad/s
  spec.gyro.bias_correlation_s = 100.0;
  const double dt = 1.0;
  sensors::Imu imu(spec, 99, 1);

  std::vector<double> drift;
  drift.reserve(100000);
  for (int i = 0; i < 100000; ++i) {
    drift.push_back(
        imu.sample(kEpoch, dt, Vec3B(Eigen::Vector3d::Zero()), Vec3B(Eigen::Vector3d::Zero()))
            .angular_rate_rads.eigen()[0]);
  }
  double sum = 0.0;
  for (double d : drift)
    sum += d;
  const double mean = sum / drift.size();
  double var = 0.0;
  double cov1 = 0.0;
  for (std::size_t i = 0; i < drift.size(); ++i) {
    var += (drift[i] - mean) * (drift[i] - mean);
    if (i > 0)
      cov1 += (drift[i] - mean) * (drift[i - 1] - mean);
  }
  var /= drift.size();
  cov1 /= (drift.size() - 1);
  EXPECT_NEAR(std::sqrt(var), spec.gyro.bias_instability, spec.gyro.bias_instability * 0.1);
  // Lag-1 autocorrelation ≈ exp(-dt/τ) = exp(-0.01) ≈ 0.99 — strongly correlated.
  EXPECT_GT(cov1 / var, 0.9);
}

TEST(Imu, GyroGsensitivityCouplesToSpecificForce) {
  // Same seed, noise/drift off: the only difference between a zero-g sample and a
  // sample under specific force is the g-sensitivity bias = k_g · sf.
  sensors::ImuSpec spec;
  spec.gyro_g_sensitivity = sensors::catalog::stim377h().gyro_g_sensitivity;
  sensors::Imu a(spec, 5, 1);
  sensors::Imu b(spec, 5, 1);
  const Eigen::Vector3d rate(0.01, 0.0, 0.0);
  const Eigen::Vector3d sf(2.0 * kG, 0.0, 0.0);

  const auto m0 = a.sample(kEpoch, 0.1, Vec3B(rate), Vec3B(Eigen::Vector3d::Zero()));
  const auto mg = b.sample(kEpoch, 0.1, Vec3B(rate), Vec3B(sf));
  const Eigen::Vector3d expected = spec.gyro_g_sensitivity * sf;
  EXPECT_TRUE((mg.angular_rate_rads.eigen() - m0.angular_rate_rads.eigen()).isApprox(expected));
}

// --- Faults ------------------------------------------------------------------

TEST(Imu, BiasJumpAndDropoutFaults) {
  sensors::Imu nominal(sensors::catalog::stim377h(), 0xABCD, 1);
  sensors::Imu faulted(sensors::catalog::stim377h(), 0xABCD, 1);
  const Eigen::Vector3d jump(1e-3, 0.0, -2e-3);
  faulted.injectGyroBiasJump(Vec3B(jump));
  const Eigen::Vector3d rate(0.01, 0.0, 0.0);
  const Eigen::Vector3d sf(0.0, 0.0, -kG);

  const auto mn = nominal.sample(kEpoch, 0.01, Vec3B(rate), Vec3B(sf));
  const auto mf = faulted.sample(kEpoch, 0.01, Vec3B(rate), Vec3B(sf));
  // The fault enters with the bias, upstream of the ADC, so both samples are
  // quantized to the same LSB grid: the difference recovers the jump to within one
  // resolution step, confirming the fault is subject to quantization like a real one.
  const Eigen::Vector3d diff = mf.angular_rate_rads.eigen() - mn.angular_rate_rads.eigen();
  const double lsb = sensors::catalog::stim377h().gyro.resolution;
  for (int i = 0; i < 3; ++i) {
    EXPECT_LE(std::abs(diff[i] - jump[i]), lsb) << "axis " << i;
  }

  faulted.setDropout(true);
  EXPECT_FALSE(faulted.sample(kEpoch, 0.01, Vec3B(rate), Vec3B(sf)).valid);
}

TEST(Imu, NonPositiveDtIsAnInvalidNoOpThatDoesNotDesyncTheStream) {
  // A dt<=0 tick must draw nothing, so a run that hits one stays bit-identical to
  // one that never did (reproducibility). Sensor `a` takes a spurious dt=0 sample
  // before its real samples; `b` never does — their real samples must still match.
  sensors::Imu a(sensors::catalog::stim377h(), 0xABCD, 1);
  sensors::Imu b(sensors::catalog::stim377h(), 0xABCD, 1);
  const Eigen::Vector3d rate(0.02, -0.01, 0.005);
  const Eigen::Vector3d sf(0.0, 0.0, -kG);

  const auto invalid = a.sample(kEpoch, 0.0, Vec3B(rate), Vec3B(sf));
  EXPECT_FALSE(invalid.valid);
  for (int i = 0; i < 20; ++i) {
    EXPECT_EQ(a.sample(kEpoch, 0.01, Vec3B(rate), Vec3B(sf)).angular_rate_rads.eigen(),
              b.sample(kEpoch, 0.01, Vec3B(rate), Vec3B(sf)).angular_rate_rads.eigen())
        << "sample " << i;
  }
}
