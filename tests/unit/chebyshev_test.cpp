/// @file Tests for the onboard Chebyshev ephemeris evaluator (design doc §11.3).
///
/// Validated by exact recovery of a hand-evaluated polynomial (position and its
/// analytic derivative), interval-boundary and out-of-coverage guards, and a
/// finite-difference cross-check of the velocity. The multi-segment table is
/// checked for interval selection, gaps, and fixed-capacity overflow.

#include "ephemeris/chebyshev.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <limits>

#include "ephemeris/ephemeris_table.hpp"
#include "math/frames.hpp"
#include "time/timescales.hpp"

namespace pe = polaris::ephemeris;
namespace pf = polaris::math::frames;

namespace {

constexpr std::int64_t kNsPerSecond = 1'000'000'000;

polaris::time::Tdb TdbAtSeconds(double seconds) {
  return polaris::time::Tdb::fromNanosecondsSinceEpoch(static_cast<std::int64_t>(seconds) *
                                                       kNsPerSecond);
}

// A degree-3 segment centred at t=1e9 s with a half-day radius. Coefficients
// chosen so the values are hand-verifiable at τ = 0.25:
//   x(τ) = 100 + 10 T1 + 4 T2 + 2 T3,   y ≡ 200,   z(τ) = 20 T1.
constexpr double kMid = 1.0e9;
constexpr double kRadius = 40'000.0;

pe::ChebyshevSegment MakeSegment() {
  pe::ChebyshevSegment seg;
  seg.mid_ns = static_cast<std::int64_t>(kMid) * kNsPerSecond;
  seg.radius_seconds = kRadius;
  seg.degree = 3;
  seg.cx[0] = 100.0;
  seg.cx[1] = 10.0;
  seg.cx[2] = 4.0;
  seg.cx[3] = 2.0;
  seg.cy[0] = 200.0;
  seg.cz[1] = 20.0;
  return seg;
}

}  // namespace

TEST(Chebyshev, RecoversHandEvaluatedPolynomial) {
  RecordProperty("verifies", "REQ-CDH-002");
  const pe::ChebyshevSegment seg = MakeSegment();
  // τ = 0.25 => t = mid + 0.25 * radius.
  const auto t = TdbAtSeconds(kMid + 0.25 * kRadius);
  polaris::math::Vec3<pf::ECI> pos;
  ASSERT_TRUE(pe::evaluate(seg, t, pos));
  // T1=0.25, T2=-0.875, T3=-0.6875 => x = 100 + 2.5 - 3.5 - 1.375 = 97.625.
  EXPECT_NEAR(pos.x(), 97.625, 1e-9);
  EXPECT_NEAR(pos.y(), 200.0, 1e-9);  // constant
  EXPECT_NEAR(pos.z(), 5.0, 1e-9);    // 20 * 0.25
}

TEST(Chebyshev, EvaluatesAtIntervalCentreAndBoundaries) {
  RecordProperty("verifies", "REQ-CDH-002");
  const pe::ChebyshevSegment seg = MakeSegment();
  polaris::math::Vec3<pf::ECI> pos;

  // Centre τ=0: T_k(0) = {1,0,-1,0} => x = 100 - 4 = 96.
  ASSERT_TRUE(pe::evaluate(seg, TdbAtSeconds(kMid), pos));
  EXPECT_NEAR(pos.x(), 96.0, 1e-9);

  // Right boundary τ=1: all T_k(1)=1 => x = 100+10+4+2 = 116.
  ASSERT_TRUE(pe::evaluate(seg, TdbAtSeconds(kMid + kRadius), pos));
  EXPECT_NEAR(pos.x(), 116.0, 1e-9);

  // Left boundary τ=-1: T_k(-1)=(-1)^k => x = 100-10+4-2 = 92.
  ASSERT_TRUE(pe::evaluate(seg, TdbAtSeconds(kMid - kRadius), pos));
  EXPECT_NEAR(pos.x(), 92.0, 1e-9);
}

TEST(Chebyshev, AnalyticVelocityMatchesHandValueAndFiniteDifference) {
  RecordProperty("verifies", "REQ-CDH-002");
  const pe::ChebyshevSegment seg = MakeSegment();
  const double t0 = kMid + 0.25 * kRadius;
  polaris::math::Vec3<pf::ECI> pos, vel;
  ASSERT_TRUE(pe::evaluate(seg, TdbAtSeconds(t0), pos, vel));

  // dx/dτ at τ=0.25: 10*T1' + 4*T2' + 2*T3' = 10 + 4(1.0) + 2(-2.25) = 9.5;
  // velocity = (dx/dτ)/radius.
  EXPECT_NEAR(vel.x(), 9.5 / kRadius, 1e-12);
  EXPECT_NEAR(vel.y(), 0.0, 1e-12);
  EXPECT_NEAR(vel.z(), 20.0 / kRadius, 1e-12);

  // Central finite difference cross-check on x.
  const double h = 1.0;
  polaris::math::Vec3<pf::ECI> p_plus, p_minus;
  ASSERT_TRUE(pe::evaluate(seg, TdbAtSeconds(t0 + h), p_plus));
  ASSERT_TRUE(pe::evaluate(seg, TdbAtSeconds(t0 - h), p_minus));
  const double fd = (p_plus.x() - p_minus.x()) / (2.0 * h);
  EXPECT_NEAR(vel.x(), fd, 1e-6);
}

TEST(Chebyshev, OutOfCoverageReturnsFalseAndLeavesOutputUntouched) {
  RecordProperty("verifies", "REQ-CDH-002");
  const pe::ChebyshevSegment seg = MakeSegment();
  polaris::math::Vec3<pf::ECI> pos(-1.0, -2.0, -3.0);
  EXPECT_FALSE(pe::evaluate(seg, TdbAtSeconds(kMid + 1.5 * kRadius), pos));
  EXPECT_FALSE(pe::evaluate(seg, TdbAtSeconds(kMid - 1.5 * kRadius), pos));
  // Output preserved.
  EXPECT_DOUBLE_EQ(pos.x(), -1.0);
  EXPECT_DOUBLE_EQ(pos.y(), -2.0);
  EXPECT_DOUBLE_EQ(pos.z(), -3.0);
}

TEST(Chebyshev, VelocityOverloadLeavesBothOutputsUntouchedOutOfCoverage) {
  RecordProperty("verifies", "REQ-CDH-002");
  const pe::ChebyshevSegment seg = MakeSegment();
  polaris::math::Vec3<pf::ECI> pos(-1.0, -2.0, -3.0), vel(-4.0, -5.0, -6.0);
  EXPECT_FALSE(pe::evaluate(seg, TdbAtSeconds(kMid + 1.5 * kRadius), pos, vel));
  EXPECT_DOUBLE_EQ(pos.x(), -1.0);
  EXPECT_DOUBLE_EQ(pos.z(), -3.0);
  EXPECT_DOUBLE_EQ(vel.x(), -4.0);
  EXPECT_DOUBLE_EQ(vel.z(), -6.0);
}

TEST(Chebyshev, AcceptsMaxDegreeAndRejectsBeyond) {
  RecordProperty("verifies", "REQ-CDH-002");
  polaris::math::Vec3<pf::ECI> pos;
  pe::ChebyshevSegment seg = MakeSegment();
  seg.degree = pe::kMaxChebyshevDegree;  // highest storable index is accepted
  EXPECT_TRUE(pe::evaluate(seg, TdbAtSeconds(kMid), pos));

  seg.degree = pe::kMaxChebyshevDegree + 1;  // one past the array bound is rejected
  EXPECT_FALSE(pe::evaluate(seg, TdbAtSeconds(kMid), pos));
}

TEST(Chebyshev, MalformedSegmentReturnsFalse) {
  RecordProperty("verifies", "REQ-CDH-002");
  polaris::math::Vec3<pf::ECI> pos;
  pe::ChebyshevSegment bad = MakeSegment();
  bad.radius_seconds = 0.0;  // non-positive interval
  EXPECT_FALSE(pe::evaluate(bad, TdbAtSeconds(kMid), pos));

  bad = MakeSegment();
  bad.degree = -1;  // no coefficients
  EXPECT_FALSE(pe::evaluate(bad, TdbAtSeconds(kMid), pos));
}

TEST(Chebyshev, NonFiniteCoefficientReturnsFalse) {
  RecordProperty("verifies", "REQ-CDH-002");
  pe::ChebyshevSegment seg = MakeSegment();
  seg.cx[2] = std::numeric_limits<double>::quiet_NaN();
  polaris::math::Vec3<pf::ECI> pos;
  EXPECT_FALSE(pe::evaluate(seg, TdbAtSeconds(kMid), pos));
}

TEST(EphemerisTable, SelectsCoveringSegmentAndRejectsGaps) {
  RecordProperty("verifies", "REQ-CDH-002");
  pe::EphemerisTable<4> table;

  pe::ChebyshevSegment a = MakeSegment();  // covers [1e9-4e4, 1e9+4e4]
  pe::ChebyshevSegment b = MakeSegment();
  b.mid_ns = static_cast<std::int64_t>(kMid + 4.0 * kRadius) *
             kNsPerSecond;  // a disjoint later interval, leaving a gap
  b.cx[0] = 500.0;

  ASSERT_TRUE(table.addSegment(a));
  ASSERT_TRUE(table.addSegment(b));
  EXPECT_EQ(table.size(), 2u);

  polaris::math::Vec3<pf::ECI> pos;
  ASSERT_TRUE(table.position(TdbAtSeconds(kMid), pos));
  EXPECT_NEAR(pos.x(), 96.0, 1e-9);  // from segment a (τ=0)

  ASSERT_TRUE(table.position(TdbAtSeconds(kMid + 4.0 * kRadius), pos));
  EXPECT_NEAR(pos.x(), 496.0, 1e-9);  // from segment b (500 - 4 at τ=0)

  // A time between the two intervals is covered by neither.
  EXPECT_FALSE(table.position(TdbAtSeconds(kMid + 2.5 * kRadius), pos));
}

TEST(EphemerisTable, RejectsMalformedSegmentsAndOverflow) {
  RecordProperty("verifies", "REQ-CDH-002");
  pe::EphemerisTable<1> table;
  pe::ChebyshevSegment bad = MakeSegment();
  bad.radius_seconds = -1.0;
  EXPECT_FALSE(table.addSegment(bad));  // malformed rejected
  EXPECT_TRUE(table.empty());

  ASSERT_TRUE(table.addSegment(MakeSegment()));
  EXPECT_FALSE(table.addSegment(MakeSegment()));  // capacity 1 -> overflow rejected
  EXPECT_EQ(table.size(), 1u);
}
