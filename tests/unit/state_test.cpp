/// @file Tests for the canonical state structs (EstimatedState / TruthState).
///
/// These pin the §8.0 contract: default validity/mode, the versioned 15-state
/// covariance layout, frame-tagged fields, and — most importantly — the *type
/// distinction* between the onboard and truth products that structurally isolates
/// flight code from truth (REQ-SYS-005).

#include <gtest/gtest.h>

#include <type_traits>

#include "math/frames.hpp"
#include "state/estimated_state.hpp"
#include "state/truth_state.hpp"

namespace ps = polaris::state;
namespace pf = polaris::math::frames;

// The onboard and truth products are distinct, non-interconvertible types: the
// FSW is written against EstimatedState and cannot be handed a TruthState (§2.3).
static_assert(!std::is_same_v<ps::EstimatedState, ps::TruthState>,
              "EstimatedState and TruthState must be distinct types (REQ-SYS-005)");
static_assert(!std::is_convertible_v<ps::TruthState, ps::EstimatedState>,
              "TruthState must not implicitly convert to EstimatedState (REQ-SYS-005)");
static_assert(!std::is_convertible_v<ps::EstimatedState, ps::TruthState>,
              "EstimatedState must not implicitly convert to TruthState (REQ-SYS-005)");

// Fields carry frame tags at the boundary (§3.1): position/velocity are ECI,
// rates/biases are Body. A wrong-frame assignment would not compile.
static_assert(decltype(ps::EstimatedState::position)::frame_name() == pf::ECI::kName,
              "position must be ECI-tagged");
static_assert(decltype(ps::EstimatedState::body_rate)::frame_name() == pf::Body::kName,
              "body_rate must be Body-tagged");

TEST(EstimatedState, DefaultsAreInvalidIdentityZero) {
  RecordProperty("verifies", "REQ-SYS-005");
  const ps::EstimatedState s;
  EXPECT_EQ(s.mode, ps::EstimationMode::Invalid);
  EXPECT_FALSE(s.valid.attitude);
  EXPECT_FALSE(s.valid.position);
  EXPECT_FALSE(s.valid.covariance);
  // Attitude defaults to identity (q0 = 1); vectors and covariance to zero.
  EXPECT_DOUBLE_EQ(s.attitude.core().scalar(), 1.0);
  EXPECT_DOUBLE_EQ(s.position.norm(), 0.0);
  EXPECT_DOUBLE_EQ(s.body_rate.norm(), 0.0);
  EXPECT_DOUBLE_EQ(s.covariance.norm(), 0.0);
  EXPECT_EQ(ps::EstimatedState::kSchemaVersion, 1);
}

TEST(EstimatedState, CovarianceIsFifteenStateWithDocumentedBlocks) {
  RecordProperty("verifies", "REQ-SYS-005");
  EXPECT_EQ(ps::ErrorState::kDim, 15);
  EXPECT_EQ(ps::ErrorState::kAttitude, 0);
  EXPECT_EQ(ps::ErrorState::kGyroBias, 3);
  EXPECT_EQ(ps::ErrorState::kPosition, 6);
  EXPECT_EQ(ps::ErrorState::kVelocity, 9);
  EXPECT_EQ(ps::ErrorState::kAccelBias, 12);
  EXPECT_EQ(ps::Covariance::RowsAtCompileTime, 15);
  EXPECT_EQ(ps::Covariance::ColsAtCompileTime, 15);

  // A populated block round-trips at its documented offset.
  ps::EstimatedState s;
  s.covariance(ps::ErrorState::kPosition, ps::ErrorState::kPosition) = 4.0;
  EXPECT_DOUBLE_EQ(s.covariance(6, 6), 4.0);
}

TEST(EstimatedState, HoldsTaggedFieldsAndFlags) {
  RecordProperty("verifies", "REQ-SYS-005");
  ps::EstimatedState s;
  s.position = polaris::math::Vec3<pf::ECI>(7.0e6, 0.0, 0.0);
  s.velocity = polaris::math::Vec3<pf::ECI>(0.0, 7.5e3, 0.0);
  s.mode = ps::EstimationMode::Fine;
  s.valid.attitude = true;
  s.valid.position = true;

  EXPECT_DOUBLE_EQ(s.position.x(), 7.0e6);
  EXPECT_DOUBLE_EQ(s.velocity.y(), 7.5e3);
  EXPECT_EQ(s.mode, ps::EstimationMode::Fine);
  EXPECT_TRUE(s.valid.attitude);
}

TEST(EstimatedState, ModeOrdersByFidelity) {
  RecordProperty("verifies", "REQ-SYS-005");
  EXPECT_GT(ps::EstimationMode::Fine, ps::EstimationMode::Coarse);
  EXPECT_GT(ps::EstimationMode::Coarse, ps::EstimationMode::Invalid);
}

TEST(TruthState, HasKinematicsAndVersion) {
  RecordProperty("verifies", "REQ-SYS-005");
  ps::TruthState t;
  t.position = polaris::math::Vec3<pf::ECI>(6.9e6, 0.0, 0.0);
  t.body_rate = polaris::math::Vec3<pf::Body>(0.0, 0.0, 0.01);
  EXPECT_DOUBLE_EQ(t.position.x(), 6.9e6);
  EXPECT_DOUBLE_EQ(t.body_rate.z(), 0.01);
  EXPECT_DOUBLE_EQ(t.attitude.core().scalar(), 1.0);
  EXPECT_EQ(ps::TruthState::kSchemaVersion, 1);
}
