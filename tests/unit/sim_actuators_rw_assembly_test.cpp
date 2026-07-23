/// @file Unit tests for the reaction-wheel assembly / W matrix (§7, §8.5).
///
/// W's columns are the wheels' spin axes in body frame. The tests pin the forward
/// map both directions (torque and momentum), that axes are normalised, and the
/// span check that separates a real pyramid from a degenerate collinear set.

#include <gtest/gtest.h>

#include <cmath>
#include <Eigen/Core>
#include <vector>

#include "actuators/rw_assembly.hpp"

namespace act = polaris::sim::actuators;

namespace {

/// The standard body-diagonal 4-wheel pyramid: axes [±1,±1,1]/√3.
std::vector<Eigen::Vector3d> pyramidAxes() {
  const double s = 1.0 / std::sqrt(3.0);
  return {{s, s, s}, {-s, s, s}, {-s, -s, s}, {s, -s, s}};
}

}  // namespace

TEST(RwAssembly, ForwardTorqueMapIsWTimesWheelTorques) {
  // Three orthogonal wheels: the body torque is just the per-axis wheel torques.
  const auto a = act::RwAssembly::fromAxes(
      {Eigen::Vector3d::UnitX(), Eigen::Vector3d::UnitY(), Eigen::Vector3d::UnitZ()});
  ASSERT_EQ(a.size(), 3);
  Eigen::VectorXd tau(3);
  tau << 0.1, -0.2, 0.3;
  const Eigen::Vector3d body = a.bodyTorque(tau);
  EXPECT_NEAR(body.x(), 0.1, 1e-12);
  EXPECT_NEAR(body.y(), -0.2, 1e-12);
  EXPECT_NEAR(body.z(), 0.3, 1e-12);
}

TEST(RwAssembly, MomentumMapMatchesTheTorqueMap) {
  const auto a = act::RwAssembly::fromAxes(pyramidAxes());
  Eigen::VectorXd h(4);
  h << 1.0, 1.0, 1.0, 1.0;
  // Equal momentum on the four symmetric pyramid axes cancels in x and y, sums in z.
  const Eigen::Vector3d body = a.bodyMomentum(h);
  EXPECT_NEAR(body.x(), 0.0, 1e-12);
  EXPECT_NEAR(body.y(), 0.0, 1e-12);
  EXPECT_NEAR(body.z(), 4.0 / std::sqrt(3.0), 1e-12);
}

TEST(RwAssembly, AxesAreNormalised) {
  // A non-unit axis must be scaled to unit — the torque about it is per-axis, so a
  // stray magnitude would silently reweight that wheel.
  const auto a = act::RwAssembly::fromAxes({Eigen::Vector3d(3.0, 0.0, 0.0)});
  ASSERT_EQ(a.size(), 1);
  EXPECT_NEAR(a.matrix().col(0).norm(), 1.0, 1e-12);
  EXPECT_NEAR(a.matrix()(0, 0), 1.0, 1e-12);
}

TEST(RwAssembly, PyramidSpansThreeAxes) {
  const auto a = act::RwAssembly::fromAxes(pyramidAxes());
  EXPECT_EQ(a.size(), 4);
  EXPECT_TRUE(a.spansThreeAxes());
}

TEST(RwAssembly, CollinearWheelsDoNotSpan) {
  // Four wheels all about +z (the old degenerate "pyramid") give no cross-axis
  // authority — exactly the misconfiguration the span check exists to catch.
  const auto a = act::RwAssembly::fromAxes({Eigen::Vector3d::UnitZ(), Eigen::Vector3d::UnitZ(),
                                            Eigen::Vector3d::UnitZ(), Eigen::Vector3d::UnitZ()});
  EXPECT_EQ(a.size(), 4);
  EXPECT_FALSE(a.spansThreeAxes());
}

TEST(RwAssembly, TwoWheelsCannotSpan) {
  const auto a = act::RwAssembly::fromAxes({Eigen::Vector3d::UnitX(), Eigen::Vector3d::UnitY()});
  EXPECT_FALSE(a.spansThreeAxes());  // fewer than 3 wheels
}

TEST(RwAssembly, ZeroAxisRejectsTheWholeSet) {
  const auto a = act::RwAssembly::fromAxes(
      {Eigen::Vector3d::UnitX(), Eigen::Vector3d::Zero(), Eigen::Vector3d::UnitZ()});
  EXPECT_TRUE(a.empty());
}
