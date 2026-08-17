/// @file Null-space wheel-speed bias servo (design doc §8.5; REQ-ACTL-012).
///
/// The property the whole module rests on is that the servo's torque has
/// **zero body torque** for any pattern, any momentum, any cap — that is what
/// makes a wheel-speed bias free — and the tests below hunt for the ways it
/// could fail: a pattern with a body-momentum part, a componentwise clip, an
/// array with no null space, a mis-set pattern.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <Eigen/Core>

#include "gnc/rw_bias.hpp"
#include "random/rng.hpp"

namespace {

namespace gnc = polaris::gnc;

/// The reference pyramid, negated spin axes as the allocation carries them.
gnc::RwBiasConfig pyramid(double bias = 3.0e-3, double gain = 0.02, double cap = 1.0e-4) {
  gnc::RwBiasConfig c;
  c.wheel_count = 4;
  const double s = 1.0 / std::sqrt(3.0);
  c.axes.col(0) = -Eigen::Vector3d(s, s, s);
  c.axes.col(1) = -Eigen::Vector3d(-s, s, s);
  c.axes.col(2) = -Eigen::Vector3d(-s, -s, s);
  c.axes.col(3) = -Eigen::Vector3d(s, -s, s);
  const double pattern[4] = {bias, -bias, bias, -bias};
  for (int i = 0; i < 4; ++i) {
    c.bias_nms[i] = pattern[i];
  }
  c.gain_per_s = gain;
  c.max_torque_nm = cap;
  return c;
}

Eigen::Vector3d bodyTorque(const gnc::RwBiasConfig& c, const gnc::RwBiasResult& r) {
  Eigen::Vector3d t = Eigen::Vector3d::Zero();
  for (int i = 0; i < c.wheel_count; ++i) {
    t += c.axes.col(i) * r.torque_nm[i];
  }
  return t;
}

TEST(RwBias, ThePyramidHasAOneDimensionalNullSpaceAndThePatternIsInIt) {
  gnc::RwBiasServo servo(pyramid());
  ASSERT_TRUE(servo.isConfigured());
  EXPECT_EQ(servo.nullDimension(), 1);
  // [+,-,+,-] is the null vector: the whole configured bias survives projection.
  for (int i = 0; i < 4; ++i) {
    EXPECT_NEAR(servo.effectiveBiasNms(i), pyramid().bias_nms[i], 1e-12);
  }
}

TEST(RwBias, ServoTorqueHasZeroBodyTorqueForRandomMomentaAndPatterns) {
  RecordProperty("verifies", "REQ-ACTL-012");
  polaris::random::SplitMix64 rng(polaris::random::streamSeed(0xB1A5u, 1));
  for (int trial = 0; trial < 500; ++trial) {
    gnc::RwBiasConfig c = pyramid();
    // A deliberately mis-set pattern with a body-momentum part.
    for (int i = 0; i < 4; ++i) {
      c.bias_nms[i] = 5.0e-3 * rng.gaussian();
    }
    c.max_torque_nm = (trial % 2 == 0) ? 1.0e-4 : 1.0e-7;  // half the trials clip hard
    gnc::RwBiasServo servo(c);
    double h[4];
    for (int i = 0; i < 4; ++i) {
      h[i] = 0.02 * rng.gaussian();
    }
    gnc::RwBiasResult r;
    ASSERT_TRUE(servo.update(h, r));
    EXPECT_LT(bodyTorque(c, r).norm(), 1e-15) << "trial " << trial;
    for (int i = 0; i < 4; ++i) {
      EXPECT_LE(std::abs(r.torque_nm[i]), c.max_torque_nm * (1.0 + 1e-12));
    }
  }
}

TEST(RwBias, ServoDrivesTheNullSpaceMomentumToThePatternAndThenRests) {
  const gnc::RwBiasConfig c = pyramid();
  gnc::RwBiasServo servo(c);
  const double inertia = 4.7746e-5;
  double speed[4] = {0.0, 0.0, 0.0, 0.0};
  const double dt = 0.1;
  gnc::RwBiasResult r;
  for (int k = 0; k < 20000; ++k) {  // 2000 s, ~40 time constants
    double h[4];
    for (int i = 0; i < 4; ++i) {
      h[i] = inertia * speed[i];
    }
    ASSERT_TRUE(servo.update(h, r));
    for (int i = 0; i < 4; ++i) {
      speed[i] += r.torque_nm[i] / inertia * dt;  // ideal rotor
    }
  }
  for (int i = 0; i < 4; ++i) {
    EXPECT_NEAR(inertia * speed[i], c.bias_nms[i], 1e-9) << "wheel " << i;
  }
  // Converged: the residual command is the exponential tail, not a torque.
  double peak = 0.0;
  for (int i = 0; i < 4; ++i) {
    peak = std::max(peak, std::abs(r.torque_nm[i]));
  }
  EXPECT_LT(peak, 1e-9);
  EXPECT_NEAR(r.null_space_nms, 2.0 * 3.0e-3, 1e-9);  // |[b,-b,b,-b]| = 2b
}

TEST(RwBias, ABodyMomentumPartOfThePatternIsNotPursued) {
  gnc::RwBiasConfig c = pyramid();
  const double pattern[4] = {3.0e-3, -3.0e-3, 3.0e-3, 3.0e-3};  // last sign wrong
  for (int i = 0; i < 4; ++i) {
    c.bias_nms[i] = pattern[i];
  }
  gnc::RwBiasServo servo(c);
  // Projection onto [1,-1,1,-1]/2: dot = (3+3+3-3)e-3/2 = 3e-3 -> effective = 1.5e-3*[1,-1,1,-1]
  EXPECT_NEAR(servo.effectiveBiasNms(0), 1.5e-3, 1e-12);
  EXPECT_NEAR(servo.effectiveBiasNms(3), -1.5e-3, 1e-12);
}

TEST(RwBias, ZeroPatternOrZeroGainIsOffAndAThreeWheelArrayIsInert) {
  gnc::RwBiasConfig off = pyramid(0.0);
  gnc::RwBiasServo s_off(off);
  double h[4] = {1e-3, 2e-3, -1e-3, 0.0};
  gnc::RwBiasResult r;
  ASSERT_TRUE(s_off.update(h, r));
  EXPECT_FALSE(r.active);
  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ(r.torque_nm[i], 0.0);
  }
  gnc::RwBiasConfig no_gain = pyramid(3e-3, 0.0, 0.0);
  ASSERT_TRUE(no_gain.isValid());
  gnc::RwBiasServo s_ng(no_gain);
  ASSERT_TRUE(s_ng.update(h, r));
  EXPECT_FALSE(r.active);

  gnc::RwBiasConfig three = pyramid();
  three.wheel_count = 3;
  gnc::RwBiasServo s3(three);
  ASSERT_TRUE(s3.isConfigured());
  EXPECT_EQ(s3.nullDimension(), 0);
  ASSERT_TRUE(s3.update(h, r));
  EXPECT_FALSE(r.active);
  EXPECT_EQ(r.null_space_nms, 0.0);
}

TEST(RwBias, BadConfigsAndInputsAreRefused) {
  gnc::RwBiasConfig c = pyramid();
  c.gain_per_s = 0.02;
  c.max_torque_nm = 0.0;  // a gain with no cap is not a trim
  EXPECT_FALSE(c.isValid());
  c = pyramid();
  c.wheel_count = 2;
  EXPECT_FALSE(c.isValid());
  c = pyramid();
  c.bias_nms[1] = NAN;
  EXPECT_FALSE(c.isValid());
  gnc::RwBiasServo servo(pyramid());
  double h[4] = {0.0, NAN, 0.0, 0.0};
  gnc::RwBiasResult r;
  EXPECT_FALSE(servo.update(h, r));
  EXPECT_EQ(r.refusal, gnc::RwBiasRefusal::kBadInput);
  EXPECT_FALSE(servo.update(nullptr, r));
}

}  // namespace
