/// @file Unit tests for boundary-tagged Vec3<Frame>.

#include "math/typed_vector.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <type_traits>

#include "math/frames.hpp"

namespace pm = polaris::math;
using pm::Vec3;
using pm::frames::Body;
using pm::frames::ECI;

TEST(TypedVector, ConstructionAndAccess) {
  RecordProperty("verifies", "REQ-SYS-004");
  const Vec3<ECI> v(1.0, 2.0, 3.0);
  EXPECT_DOUBLE_EQ(v.x(), 1.0);
  EXPECT_DOUBLE_EQ(v.y(), 2.0);
  EXPECT_DOUBLE_EQ(v.z(), 3.0);
  EXPECT_DOUBLE_EQ(v.norm(), std::sqrt(14.0));
  EXPECT_STREQ(Vec3<ECI>::frame_name(), "ECI");
}

TEST(TypedVector, FramePreservingArithmetic) {
  RecordProperty("verifies", "REQ-SYS-004");
  const Vec3<Body> a(1.0, 0.0, 0.0);
  const Vec3<Body> b(0.0, 1.0, 0.0);
  EXPECT_DOUBLE_EQ((a + b).norm(), std::sqrt(2.0));
  EXPECT_DOUBLE_EQ((a - b).norm(), std::sqrt(2.0));
  EXPECT_DOUBLE_EQ(a.dot(b), 0.0);
  const Vec3<Body> c = a.cross(b);  // x cross y = z
  EXPECT_DOUBLE_EQ(c.z(), 1.0);
  EXPECT_DOUBLE_EQ((2.0 * a).x(), 2.0);
  EXPECT_DOUBLE_EQ((a * 2.0).x(), 2.0);
  EXPECT_DOUBLE_EQ((-a).x(), -1.0);
}

TEST(TypedVector, NormalizedRejectsDegenerate) {
  RecordProperty("verifies", "REQ-SYS-004");
  const Vec3<ECI> zero;
  Vec3<ECI> out;
  EXPECT_FALSE(zero.normalized(out));

  const Vec3<ECI> v(3.0, 0.0, 0.0);
  ASSERT_TRUE(v.normalized(out));
  EXPECT_DOUBLE_EQ(out.x(), 1.0);
  EXPECT_DOUBLE_EQ(out.norm(), 1.0);
}

TEST(TypedVector, IsFiniteDetectsNonFinite) {
  RecordProperty("verifies", "REQ-SYS-006");
  EXPECT_TRUE(Vec3<ECI>(1.0, 2.0, 3.0).isFinite());
  EXPECT_FALSE(Vec3<ECI>(std::nan(""), 0.0, 0.0).isFinite());
}

// Compile-time frame safety: Vec3<ECI> and Vec3<Body> are distinct types, so
// `eci + body` would not compile. We assert that statically here.
static_assert(!std::is_same_v<Vec3<ECI>, Vec3<Body>>,
              "frame tags must produce distinct vector types");
