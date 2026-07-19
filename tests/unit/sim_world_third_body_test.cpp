/// @file Unit tests for third-body point-mass gravity and the force composite
/// (REQ-SIM-002; design doc §5.2).
///
/// The acceleration is cross-checked against an INDEPENDENT derivation — the
/// first-order tidal expansion `a ≈ (GM_b/|s|^3)(3(r·ŝ)ŝ − r)` valid for
/// |r| << |s| — not a restatement of the implementation's own formula. Plus
/// superposition, realistic Sun/Moon magnitudes, ephemeris-coverage skipping, the
/// TAI→TDB argument conversion, and the CompositeForceModel summation.

#include <gtest/gtest.h>

#include <cmath>
#include <Eigen/Core>
#include <memory>

#include "constants/constants.hpp"
#include "dynamics/force_torque.hpp"
#include "ephemeris/chebyshev.hpp"
#include "ephemeris/ephemeris_table.hpp"
#include "math/frames.hpp"
#include "math/typed_vector.hpp"
#include "state/truth_state.hpp"
#include "time/tdb.hpp"
#include "time/timescales.hpp"
#include "world/third_body.hpp"

namespace world = polaris::sim::world;
namespace dyn = polaris::sim::dynamics;
namespace pm = polaris::math;
namespace pmf = polaris::math::frames;
namespace pt = polaris::time;
namespace pc = polaris::constants;

namespace {

using Eci = pm::Vec3<pmf::ECI>;

polaris::state::TruthState at(const Eigen::Vector3d& r) {
  polaris::state::TruthState s;
  s.position = Eci(r);
  s.epoch = pt::Tai::fromNanosecondsSinceEpoch(1'600'000'000'000'000'000LL);  // ~2020
  return s;
}

/// A resolver that always reports a body fixed at @p pos (geocentric ECI).
world::ThirdBodyGravity::PositionFn fixedAt(const Eigen::Vector3d& pos) {
  return [pos](const pt::Tdb&, Eci& out) {
    out = Eci(pos);
    return true;
  };
}

}  // namespace

TEST(ThirdBody, MatchesFirstOrderTidalExpansion) {
  RecordProperty("verifies", "REQ-SIM-002");
  // Sun far away (+X at 1 AU), satellite at LEO — r/s ~ 5e-5, so the tidal
  // expansion is exact to O((r/s)^2) ~ 2e-9. Independent of the impl's formula.
  const Eigen::Vector3d s(pc::bodies::kAstronomicalUnit, 0.0, 0.0);
  world::ThirdBodyGravity g;
  g.addBody(pc::bodies::kSunGM, fixedAt(s));

  const Eigen::Vector3d r(3.0e6, 5.0e6, 2.0e6);  // |r| ~ 6.2e6 m
  const Eigen::Vector3d a = g.acceleration(at(r)).eigen();

  const Eigen::Vector3d shat = s.normalized();
  const double s3 = std::pow(s.norm(), 3);
  const Eigen::Vector3d tidal = (pc::bodies::kSunGM / s3) * (3.0 * r.dot(shat) * shat - r);
  // The first-order tidal term omits the next multipole (~r^2/s^4), so it agrees
  // with the exact perturbation only to O(r/s) ~ 4e-5 here — tolerance set above
  // that. This confirms the impl reduces to the tidal limit, not the reverse.
  EXPECT_LT((a - tidal).norm(), tidal.norm() * 1e-4);
}

TEST(ThirdBody, VanishesAtGeocenter) {
  RecordProperty("verifies", "REQ-SIM-002");
  // At r = 0 the satellite sits at the geocenter, so the two terms are bit-
  // identical and the geocentric perturbation is exactly zero.
  world::ThirdBodyGravity g;
  g.addBody(pc::bodies::kMoonGM, fixedAt(Eigen::Vector3d(3.84e8, 0.0, 0.0)));
  const Eigen::Vector3d a = g.acceleration(at(Eigen::Vector3d::Zero())).eigen();
  EXPECT_EQ(a, Eigen::Vector3d::Zero());
}

TEST(ThirdBody, SuperposesBodies) {
  RecordProperty("verifies", "REQ-SIM-002");
  const Eigen::Vector3d sun(pc::bodies::kAstronomicalUnit, 0.0, 0.0);
  const Eigen::Vector3d moon(0.0, 3.84e8, 1.0e8);
  const Eigen::Vector3d r(6.6e6, -1.2e6, 3.0e6);

  world::ThirdBodyGravity only_sun;
  only_sun.addBody(pc::bodies::kSunGM, fixedAt(sun));
  world::ThirdBodyGravity only_moon;
  only_moon.addBody(pc::bodies::kMoonGM, fixedAt(moon));
  world::ThirdBodyGravity both;
  both.addBody(pc::bodies::kSunGM, fixedAt(sun));
  both.addBody(pc::bodies::kMoonGM, fixedAt(moon));
  EXPECT_EQ(both.size(), 2u);

  const Eigen::Vector3d a_both = both.acceleration(at(r)).eigen();
  const Eigen::Vector3d a_sum =
      only_sun.acceleration(at(r)).eigen() + only_moon.acceleration(at(r)).eigen();
  EXPECT_LT((a_both - a_sum).norm(), a_both.norm() * 1e-15);
}

TEST(ThirdBody, RealisticSunMoonMagnitudes) {
  RecordProperty("verifies", "REQ-SIM-002");
  // Known LEO third-body perturbations: Moon ~1e-6 m/s^2, Sun ~6e-7 m/s^2 —
  // both ~7 orders below Earth's ~9 m/s^2 surface gravity.
  const Eigen::Vector3d r(6.9e6, 0.0, 0.0);
  world::ThirdBodyGravity moon;
  moon.addBody(pc::bodies::kMoonGM, fixedAt(Eigen::Vector3d(0.0, 3.84e8, 0.0)));
  world::ThirdBodyGravity sun;
  sun.addBody(pc::bodies::kSunGM,
              fixedAt(Eigen::Vector3d(0.0, pc::bodies::kAstronomicalUnit, 0.0)));

  const double a_moon = moon.acceleration(at(r)).eigen().norm();
  const double a_sun = sun.acceleration(at(r)).eigen().norm();
  EXPECT_GT(a_moon, 1e-7);
  EXPECT_LT(a_moon, 1e-5);
  EXPECT_GT(a_sun, 1e-7);
  EXPECT_LT(a_sun, 1e-6);
}

TEST(ThirdBody, SkipsBodyOutsideEphemerisCoverage) {
  RecordProperty("verifies", "REQ-SIM-002");
  const Eigen::Vector3d r(6.9e6, 0.0, 0.0);
  const Eigen::Vector3d moon(0.0, 3.84e8, 0.0);

  world::ThirdBodyGravity g;
  g.addBody(pc::bodies::kMoonGM, fixedAt(moon));                              // covered
  g.addBody(pc::bodies::kSunGM, [](const pt::Tdb&, Eci&) { return false; });  // no coverage

  // The uncovered Sun contributes nothing, so the result equals the Moon alone.
  world::ThirdBodyGravity moon_only;
  moon_only.addBody(pc::bodies::kMoonGM, fixedAt(moon));
  EXPECT_EQ(g.acceleration(at(r)).eigen(), moon_only.acceleration(at(r)).eigen());
}

TEST(ThirdBody, ResolverReceivesTdbNotTai) {
  RecordProperty("verifies", "REQ-SIM-002");
  const polaris::state::TruthState s = at(Eigen::Vector3d(7.0e6, 0.0, 0.0));
  pt::Tdb seen{};
  bool called = false;
  world::ThirdBodyGravity g;
  g.addBody(pc::bodies::kSunGM, [&](const pt::Tdb& t, Eci& out) {
    seen = t;
    called = true;
    out = Eci(Eigen::Vector3d(pc::bodies::kAstronomicalUnit, 0.0, 0.0));
    return true;
  });
  g.acceleration(s);
  ASSERT_TRUE(called);
  EXPECT_EQ(seen, pt::toTdb(pt::toTt(s.epoch)));  // TAI epoch converted to TDB
}

TEST(ThirdBody, PointMassExertsNoTorque) {
  RecordProperty("verifies", "REQ-SIM-002");
  world::ThirdBodyGravity g;
  g.addBody(pc::bodies::kMoonGM, fixedAt(Eigen::Vector3d(0.0, 3.84e8, 0.0)));
  EXPECT_EQ(g.torque(at(Eigen::Vector3d(7.0e6, 0.0, 0.0))).eigen(), Eigen::Vector3d::Zero());
}

TEST(ThirdBody, DrivenByEphemerisTableEvaluator) {
  RecordProperty("verifies", "REQ-SIM-002");
  // The production wiring: the resolver wraps an EphemerisTable (the same
  // evaluator the DE440-fed truth ephemeris uses), not a constant lambda. A
  // degree-0 segment holds a constant body position over its interval.
  const polaris::state::TruthState s = at(Eigen::Vector3d(6.9e6, 1.1e6, -2.0e6));
  const pt::Tdb tdb = pt::toTdb(pt::toTt(s.epoch));
  const Eigen::Vector3d moon(1.0e8, 3.7e8, 5.0e7);

  polaris::ephemeris::ChebyshevSegment seg;
  seg.mid_ns = tdb.nanosecondsSinceEpoch();
  seg.radius_seconds = 86'400.0;  // ±1 day coverage around the epoch
  seg.degree = 0;                 // constant position
  seg.cx[0] = moon.x();
  seg.cy[0] = moon.y();
  seg.cz[0] = moon.z();

  auto table = std::make_shared<polaris::ephemeris::EphemerisTable<4>>();
  ASSERT_TRUE(table->addSegment(seg));

  world::ThirdBodyGravity g;
  g.addBody(pc::bodies::kMoonGM,
            [table](const pt::Tdb& t, Eci& out) { return table->position(t, out); });

  // Same result as feeding the body position directly — proves the table path.
  world::ThirdBodyGravity ref;
  ref.addBody(pc::bodies::kMoonGM, fixedAt(moon));
  EXPECT_LT((g.acceleration(s).eigen() - ref.acceleration(s).eigen()).norm(), 1e-18);
}

TEST(CompositeForce, SumsAccelerationAndTorqueOfComponents) {
  RecordProperty("verifies", "REQ-SIM-002");
  const dyn::TwoBodyGravity earth(pc::wgs84::kGM);
  world::ThirdBodyGravity moon;
  moon.addBody(pc::bodies::kMoonGM, fixedAt(Eigen::Vector3d(0.0, 3.84e8, 0.0)));

  dyn::CompositeForceModel composite;
  composite.add(&earth);
  composite.add(&moon);
  composite.add(nullptr);  // ignored
  EXPECT_EQ(composite.size(), 2u);

  const polaris::state::TruthState s = at(Eigen::Vector3d(6.9e6, 1.0e6, 2.0e6));
  const Eigen::Vector3d expected = earth.acceleration(s).eigen() + moon.acceleration(s).eigen();
  EXPECT_LT((composite.acceleration(s).eigen() - expected).norm(), 1e-18);
  // Both components are torque-free, so the composite is too.
  EXPECT_EQ(composite.torque(s).eigen(), Eigen::Vector3d::Zero());
}
