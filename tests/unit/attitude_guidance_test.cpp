/// @file Unit tests for align/constrain attitude guidance (§8.4; REQ-AGN-004,
/// REQ-AGN-005).
///
/// The claim this file has to support is that *every* pointing mode is the same
/// command with different nouns. So the tests are organised around that: the
/// noun lists resolve correctly one by one, the validation catches the ways a
/// pair of nouns can be nonsense, and the solve turns any valid pair into an
/// attitude — with the well-known modes (nadir hold, sun-safe, ground-station
/// track, satellite track) checked as *instances* of the general machinery
/// rather than as separate features.

#include "gnc/attitude_guidance.hpp"

#include <gtest/gtest.h>

#include <cmath>

#include "constants/constants.hpp"

namespace pc = polaris::constants;
namespace pg = polaris::gnc;
namespace pm = polaris::math;
namespace pmf = polaris::math::frames;
namespace pt = polaris::time;

namespace {

constexpr double kDeg = M_PI / 180.0;

pg::BodyVectorRef body(pg::BodyVectorKind k, int i = 0, bool neg = false) {
  pg::BodyVectorRef r;
  r.kind = k;
  r.index = static_cast<std::uint8_t>(i);
  r.negate = neg;
  return r;
}

pg::PointingTargetRef target(pg::PointingTargetKind k, int i = 0, bool neg = false) {
  pg::PointingTargetRef r;
  r.kind = k;
  r.index = static_cast<std::uint8_t>(i);
  r.negate = neg;
  return r;
}

/// A 600 km circular orbit in the equatorial plane, crossing +X moving +Y.
pg::GuidanceContext leoContext() {
  const double r = pc::wgs84::kSemiMajorAxis + 600.0e3;
  const double v = std::sqrt(pc::gravity::kGM / r);
  pg::GuidanceContext ctx;
  ctx.t = pt::Tai::fromNanosecondsSinceEpoch(1'000'000'000'000'000'000);
  ctx.observer_position_m = pm::Vec3<pmf::ECI>(r, 0.0, 0.0);
  ctx.observer_velocity_m_s = pm::Vec3<pmf::ECI>(0.0, v, 0.0);
  ctx.observer_valid = true;
  ctx.sun_position_m = pm::Vec3<pmf::ECI>(1.496e11, 0.0, 0.0);
  ctx.sun_valid = true;
  ctx.moon_position_m = pm::Vec3<pmf::ECI>(0.0, 3.844e8, 0.0);
  ctx.moon_valid = true;
  ctx.earth_orientation_valid = true;
  return ctx;
}

pg::BodyVectorTable defaultTable() {
  pg::BodyVectorTable t;
  EXPECT_TRUE(t.setSensor(pg::BodyVectorKind::kCamera, 0, pm::Vec3<pmf::Body>(0.0, 0.0, 1.0)));
  EXPECT_TRUE(t.setSensor(pg::BodyVectorKind::kStarTracker, 0, pm::Vec3<pmf::Body>(0.0, 1.0, 0.0)));
  EXPECT_TRUE(t.setSensor(pg::BodyVectorKind::kSunSensor, 0, pm::Vec3<pmf::Body>(1.0, 0.0, 0.0)));
  return t;
}

}  // namespace

// ---------------------------------------------------------------------------
// The body-vector noun list
// ---------------------------------------------------------------------------

TEST(BodyVectorTable, StructuralAxesNeedNoMountingData) {
  const pg::BodyVectorTable t;  // deliberately empty
  pm::Vec3<pmf::Body> v;
  ASSERT_TRUE(t.resolve(body(pg::BodyVectorKind::kBodyX), v));
  EXPECT_EQ(v.eigen(), Eigen::Vector3d(1.0, 0.0, 0.0));
  ASSERT_TRUE(t.resolve(body(pg::BodyVectorKind::kBodyZ, 0, /*neg=*/true), v));
  EXPECT_EQ(v.eigen(), Eigen::Vector3d(0.0, 0.0, -1.0));
}

TEST(BodyVectorTable, AnUninstalledUnitIsRefusedRatherThanZero) {
  // A null vector would flow into the guidance and be reported there as a
  // degenerate axis, blaming the geometry for a missing mounting parameter.
  const pg::BodyVectorTable t;
  pm::Vec3<pmf::Body> v;
  EXPECT_FALSE(t.resolve(body(pg::BodyVectorKind::kStarTracker, 3), v));
  EXPECT_FALSE(t.resolve(body(pg::BodyVectorKind::kCustom, 0), v));
}

TEST(BodyVectorTable, CustomVectorsStoreNormaliseAndClear) {
  pg::BodyVectorTable t;
  EXPECT_FALSE(t.isCustomSet(2));
  ASSERT_TRUE(t.setCustom(2, pm::Vec3<pmf::Body>(3.0, 4.0, 0.0)));
  EXPECT_TRUE(t.isCustomSet(2));
  EXPECT_EQ(t.customCount(), 1);

  pm::Vec3<pmf::Body> v;
  ASSERT_TRUE(t.resolve(body(pg::BodyVectorKind::kCustom, 2), v));
  EXPECT_NEAR(v.norm(), 1.0, 1e-15) << "resolution must return a unit vector";
  EXPECT_NEAR(v.eigen().x(), 0.6, 1e-15);

  ASSERT_TRUE(t.clearCustom(2));
  EXPECT_FALSE(t.resolve(body(pg::BodyVectorKind::kCustom, 2), v));
}

TEST(BodyVectorTable, RejectsOutOfRangeAndDegenerateEntries) {
  pg::BodyVectorTable t;
  EXPECT_FALSE(t.setCustom(-1, pm::Vec3<pmf::Body>(1.0, 0.0, 0.0)));
  EXPECT_FALSE(t.setCustom(pg::kMaxCustomBodyVectors, pm::Vec3<pmf::Body>(1.0, 0.0, 0.0)));
  EXPECT_FALSE(t.setCustom(0, pm::Vec3<pmf::Body>::Zero()));
  EXPECT_FALSE(t.setCustom(0, pm::Vec3<pmf::Body>(std::nan(""), 0.0, 0.0)));
  // Mounting setters take only sensor kinds — a structural axis is not
  // mounting data and accepting one would create a second definition of +X.
  EXPECT_FALSE(t.setSensor(pg::BodyVectorKind::kBodyX, 0, pm::Vec3<pmf::Body>(1.0, 0.0, 0.0)));
}

TEST(BodyVectorTable, HoldsTenCustomVectorsAndEightOfEachSensor) {
  pg::BodyVectorTable t;
  for (int i = 0; i < pg::kMaxCustomBodyVectors; ++i) {
    ASSERT_TRUE(t.setCustom(i, pm::Vec3<pmf::Body>(1.0, static_cast<double>(i) + 1.0, 0.0)));
  }
  EXPECT_EQ(t.customCount(), pg::kMaxCustomBodyVectors);
  for (int i = 0; i < pg::kMaxSensorUnits; ++i) {
    EXPECT_TRUE(t.setSensor(pg::BodyVectorKind::kCamera, i, pm::Vec3<pmf::Body>(0.0, 0.0, 1.0)));
  }
  EXPECT_FALSE(
      t.setSensor(pg::BodyVectorKind::kCamera, pg::kMaxSensorUnits, pm::Vec3<pmf::Body>(0, 0, 1)));
}

// ---------------------------------------------------------------------------
// The ground-point table
// ---------------------------------------------------------------------------

TEST(GroundPointTable, HoldsThirtyStationsAndRejectsNonsense) {
  pg::GroundPointTable g;
  EXPECT_EQ(g.count(), 0);
  for (int i = 0; i < pg::kMaxGroundPoints; ++i) {
    ASSERT_TRUE(g.set(i, 10.0 * kDeg, static_cast<double>(i) * kDeg, 100.0)) << "slot " << i;
  }
  EXPECT_EQ(g.count(), pg::kMaxGroundPoints);
  EXPECT_FALSE(g.set(pg::kMaxGroundPoints, 0.0, 0.0, 0.0));

  // A latitude beyond the pole is a typo, not a site.
  EXPECT_FALSE(g.set(0, 91.0 * kDeg, 0.0, 0.0));
  // The height bound exists to catch a metres-vs-kilometres unit error, which
  // is the mistake that actually happens on a station upload.
  EXPECT_FALSE(g.set(0, 0.0, 0.0, 1500000.0));
  EXPECT_FALSE(g.set(0, std::nan(""), 0.0, 0.0));
}

TEST(GroundPointTable, RejectionLeavesTheStoredStationIntact) {
  pg::GroundPointTable g;
  ASSERT_TRUE(g.set(4, 45.0 * kDeg, -75.0 * kDeg, 80.0));
  EXPECT_FALSE(g.set(4, 200.0 * kDeg, 0.0, 0.0));
  pg::GroundPointTable::Point p;
  ASSERT_TRUE(g.get(4, p));
  EXPECT_NEAR(p.latitude_rad, 45.0 * kDeg, 1e-15) << "a rejected upload overwrote a good station";
}

TEST(GroundPointTable, ClearingMakesTheSlotUnnameableAgain) {
  pg::GroundPointTable g;
  ASSERT_TRUE(g.set(7, 0.0, 0.0, 0.0));
  ASSERT_TRUE(g.clear(7));
  EXPECT_FALSE(g.isSet(7));
  pg::GroundPointTable::Point p;
  EXPECT_FALSE(g.get(7, p));
}

// ---------------------------------------------------------------------------
// The target noun list
// ---------------------------------------------------------------------------

TEST(ResolveTarget, InertialAxesAreFixedAndKnownToBe) {
  const pg::GuidanceContext ctx = leoContext();
  pg::ResolvedDirection d;
  ASSERT_EQ(pg::resolveTarget(target(pg::PointingTargetKind::kJ2000Z), ctx, d),
            pg::GuidanceStatus::kOk);
  EXPECT_EQ(d.unit.eigen(), Eigen::Vector3d(0.0, 0.0, 1.0));
  EXPECT_EQ(d.rate_rad_s.eigen(), Eigen::Vector3d::Zero());
  EXPECT_TRUE(d.rate_known) << "a fixed direction must report that its rate is known to be zero";
}

TEST(ResolveTarget, NadirIsAntiRadialAndTurnsAtTheOrbitRate) {
  const pg::GuidanceContext ctx = leoContext();
  pg::ResolvedDirection d;
  ASSERT_EQ(pg::resolveTarget(target(pg::PointingTargetKind::kNadir), ctx, d),
            pg::GuidanceStatus::kOk);
  EXPECT_NEAR(d.unit.eigen().x(), -1.0, 1e-15);
  // One revolution in ~96 min => ~1.08e-3 rad/s about +Z for this orbit.
  EXPECT_NEAR(d.rate_rad_s.eigen().z(), 1.083e-3, 5e-6);
  EXPECT_TRUE(d.rate_known);
}

TEST(ResolveTarget, NegateFlipsTheDirectionButNotItsRate) {
  // The rate is the angular velocity of the *axis*; -u turns with the same
  // angular velocity as u. A sign applied to both would send the feedforward
  // the wrong way.
  const pg::GuidanceContext ctx = leoContext();
  pg::ResolvedDirection plain;
  pg::ResolvedDirection flipped;
  ASSERT_EQ(pg::resolveTarget(target(pg::PointingTargetKind::kNadir), ctx, plain),
            pg::GuidanceStatus::kOk);
  ASSERT_EQ(pg::resolveTarget(target(pg::PointingTargetKind::kNadir, 0, true), ctx, flipped),
            pg::GuidanceStatus::kOk);
  EXPECT_EQ(flipped.unit.eigen(), -plain.unit.eigen());
  EXPECT_EQ(flipped.rate_rad_s.eigen(), plain.rate_rad_s.eigen());
}

TEST(ResolveTarget, LvlhAxesFollowTheRepositoryDefinition) {
  // z = -r̂ (nadir), y = -ĥ, x = y x z (~ +velocity). Taken from
  // math::lvlhFromEci rather than re-derived, so a pointing command and a
  // covariance display cannot disagree about which way LVLH x points.
  const pg::GuidanceContext ctx = leoContext();
  pg::ResolvedDirection x;
  pg::ResolvedDirection z;
  ASSERT_EQ(pg::resolveTarget(target(pg::PointingTargetKind::kLvlhZ), ctx, z),
            pg::GuidanceStatus::kOk);
  ASSERT_EQ(pg::resolveTarget(target(pg::PointingTargetKind::kLvlhX), ctx, x),
            pg::GuidanceStatus::kOk);
  EXPECT_NEAR(z.unit.eigen().x(), -1.0, 1e-12) << "LVLH z is not nadir";
  EXPECT_NEAR(x.unit.eigen().y(), 1.0, 1e-12) << "LVLH x is not along +velocity here";
}

TEST(ResolveTarget, SunAndMoonAreLinesOfSightAndSayWhetherTheRateIsKnown) {
  pg::GuidanceContext ctx = leoContext();
  pg::ResolvedDirection d;
  ASSERT_EQ(pg::resolveTarget(target(pg::PointingTargetKind::kSun), ctx, d),
            pg::GuidanceStatus::kOk);
  EXPECT_NEAR(d.unit.eigen().x(), 1.0, 1e-6);
  // Without the Sun's own velocity only the parallax term is modelled, and the
  // resolver must say so rather than presenting a partial rate as complete.
  EXPECT_FALSE(d.rate_known);

  ctx.sun_velocity_m_s = pm::Vec3<pmf::ECI>(0.0, 29780.0, 0.0);
  ctx.sun_velocity_valid = true;
  ASSERT_EQ(pg::resolveTarget(target(pg::PointingTargetKind::kSun), ctx, d),
            pg::GuidanceStatus::kOk);
  EXPECT_TRUE(d.rate_known);
}

TEST(ResolveTarget, MissingEphemerisIsReportedSpecifically) {
  pg::GuidanceContext ctx = leoContext();
  ctx.sun_valid = false;
  pg::ResolvedDirection d;
  EXPECT_EQ(pg::resolveTarget(target(pg::PointingTargetKind::kSun), ctx, d),
            pg::GuidanceStatus::kNoEphemeris);
}

TEST(ResolveTarget, OrbitTiedDirectionsNeedAVehicleState) {
  pg::GuidanceContext ctx = leoContext();
  ctx.observer_valid = false;
  pg::ResolvedDirection d;
  for (auto k : {pg::PointingTargetKind::kNadir, pg::PointingTargetKind::kLvlhX,
                 pg::PointingTargetKind::kSun, pg::PointingTargetKind::kEcefPoint}) {
    EXPECT_EQ(pg::resolveTarget(target(k), ctx, d), pg::GuidanceStatus::kNoOrbitState)
        << pg::toString(k) << " resolved without a vehicle state";
  }
  // An inertial axis and a star do not need one, and must keep working.
  EXPECT_EQ(pg::resolveTarget(target(pg::PointingTargetKind::kJ2000X), ctx, d),
            pg::GuidanceStatus::kOk);
}

TEST(ResolveTarget, AStarComesFromRightAscensionAndDeclination) {
  const pg::GuidanceContext ctx = leoContext();
  pg::PointingTargetRef ref = target(pg::PointingTargetKind::kStarJ2000);
  ref.params[0] = 90.0 * kDeg;  // RA
  ref.params[1] = 0.0;          // Dec
  pg::ResolvedDirection d;
  ASSERT_EQ(pg::resolveTarget(ref, ctx, d), pg::GuidanceStatus::kOk);
  EXPECT_NEAR(d.unit.eigen().y(), 1.0, 1e-15);
  EXPECT_TRUE(d.rate_known) << "proper motion is far below anything a control loop resolves";

  ref.params[1] = 100.0 * kDeg;  // impossible declination
  EXPECT_EQ(pg::resolveTarget(ref, ctx, d), pg::GuidanceStatus::kBadParameters);
}

TEST(ResolveTarget, AGroundStationIsRefusedWithoutEarthOrientation) {
  // The one target kind that needs EOP. Everything else keeps working through
  // an outage, so this is refused specifically rather than taking the whole
  // guidance mode down.
  pg::GroundPointTable g;
  ASSERT_TRUE(g.set(0, 0.0, 0.0, 0.0));
  pg::GuidanceContext ctx = leoContext();
  ctx.ground_points = &g;
  ctx.earth_orientation_valid = false;
  pg::ResolvedDirection d;
  EXPECT_EQ(pg::resolveTarget(target(pg::PointingTargetKind::kEcefPoint, 0), ctx, d),
            pg::GuidanceStatus::kNoEarthOrientation);
}

TEST(ResolveTarget, AGroundStationCarriesTheEarthRotationTerm) {
  // The station is fixed in ECEF, so in ECI it moves at up to 465 m/s. Omitting
  // that term leaves a station track lagging by exactly the Earth rate.
  pg::GroundPointTable g;
  ASSERT_TRUE(g.set(0, 0.0, 0.0, 0.0));  // on the equator, under the vehicle
  pg::GuidanceContext ctx = leoContext();
  ctx.ground_points = &g;
  pg::ResolvedDirection d;
  ASSERT_EQ(pg::resolveTarget(target(pg::PointingTargetKind::kEcefPoint, 0), ctx, d),
            pg::GuidanceStatus::kOk);
  EXPECT_NEAR(d.unit.eigen().x(), -1.0, 1e-9) << "the station should be directly below";
  EXPECT_TRUE(d.rate_known);
  EXPECT_GT(d.rate_rad_s.norm(), 1e-4) << "the line of sight to a station is not stationary";
}

TEST(ResolveTarget, AnEmptyGroundSlotIsReportedAsEmpty) {
  pg::GroundPointTable g;
  pg::GuidanceContext ctx = leoContext();
  ctx.ground_points = &g;
  pg::ResolvedDirection d;
  EXPECT_EQ(pg::resolveTarget(target(pg::PointingTargetKind::kEcefPoint, 11), ctx, d),
            pg::GuidanceStatus::kTargetSlotEmpty);
}

// ---------------------------------------------------------------------------
// Validation: the ways a pair of nouns can be nonsense
// ---------------------------------------------------------------------------

TEST(ValidateGuidance, RefusesTheSameBodyAxisTwice) {
  const pg::BodyVectorTable t = defaultTable();
  pg::GuidanceCommand cmd;
  cmd.align_vector = body(pg::BodyVectorKind::kBodyZ);
  cmd.constrain_vector = body(pg::BodyVectorKind::kBodyZ);
  cmd.align_target = target(pg::PointingTargetKind::kNadir);
  cmd.constrain_target = target(pg::PointingTargetKind::kJ2000X);
  EXPECT_EQ(pg::validateGuidanceCommand(cmd, t, nullptr, nullptr),
            pg::GuidanceStatus::kSameBodyAxis);
}

TEST(ValidateGuidance, RefusesTheSameAxisEvenWithOppositeSigns) {
  // +Z aligned and -Z constrained is exactly as unsatisfiable as naming +Z
  // twice: one axis cannot point two ways. A sign-aware comparison would let it
  // through and the failure would surface later as a degenerate solve.
  const pg::BodyVectorTable t = defaultTable();
  pg::GuidanceCommand cmd;
  cmd.align_vector = body(pg::BodyVectorKind::kBodyZ);
  cmd.constrain_vector = body(pg::BodyVectorKind::kBodyZ, 0, /*neg=*/true);
  cmd.align_target = target(pg::PointingTargetKind::kNadir);
  cmd.constrain_target = target(pg::PointingTargetKind::kJ2000X);
  EXPECT_EQ(pg::validateGuidanceCommand(cmd, t, nullptr, nullptr),
            pg::GuidanceStatus::kSameBodyAxis);
}

TEST(ValidateGuidance, RefusesTwoDifferentNamesForOneDirection) {
  // Two distinct names can still be one direction: a camera boresighted along a
  // structural axis, or a custom vector written parallel to one. Only the
  // resolved geometry can see it.
  pg::BodyVectorTable t = defaultTable();
  ASSERT_TRUE(t.setCustom(0, pm::Vec3<pmf::Body>(0.0, 0.0, 5.0)));  // parallel to +Z
  pg::GuidanceCommand cmd;
  cmd.align_vector = body(pg::BodyVectorKind::kBodyZ);
  cmd.constrain_vector = body(pg::BodyVectorKind::kCustom, 0);
  cmd.align_target = target(pg::PointingTargetKind::kNadir);
  cmd.constrain_target = target(pg::PointingTargetKind::kJ2000X);
  EXPECT_EQ(pg::validateGuidanceCommand(cmd, t, nullptr, nullptr),
            pg::GuidanceStatus::kBodyAxesParallel);
}

TEST(ValidateGuidance, RefusesAnUninstalledOrUnstoredBodyVector) {
  const pg::BodyVectorTable t = defaultTable();
  pg::GuidanceCommand cmd;
  cmd.align_vector = body(pg::BodyVectorKind::kStarTracker, 5);  // not installed
  cmd.constrain_vector = body(pg::BodyVectorKind::kBodyX);
  cmd.align_target = target(pg::PointingTargetKind::kNadir);
  cmd.constrain_target = target(pg::PointingTargetKind::kJ2000X);
  EXPECT_EQ(pg::validateGuidanceCommand(cmd, t, nullptr, nullptr),
            pg::GuidanceStatus::kBodyVectorUnknown);
}

TEST(ValidateGuidance, RefusesAnEmptyOrOutOfRangeSlot) {
  const pg::BodyVectorTable t = defaultTable();
  pg::TargetCatalog cat;
  pg::GroundPointTable g;
  pg::GuidanceCommand cmd;
  cmd.align_vector = body(pg::BodyVectorKind::kCamera, 0);
  cmd.constrain_vector = body(pg::BodyVectorKind::kBodyX);
  cmd.constrain_target = target(pg::PointingTargetKind::kJ2000X);

  cmd.align_target = target(pg::PointingTargetKind::kSatTle, 2);
  EXPECT_EQ(pg::validateGuidanceCommand(cmd, t, &cat, &g), pg::GuidanceStatus::kTargetSlotEmpty);

  cmd.align_target = target(pg::PointingTargetKind::kSatTle, 9);
  EXPECT_EQ(pg::validateGuidanceCommand(cmd, t, &cat, &g), pg::GuidanceStatus::kBadParameters);

  // A ground slot beyond thirty is out of range; one inside it but unwritten is
  // empty. The two get different reasons because the operator's fix differs.
  cmd.align_target = target(pg::PointingTargetKind::kEcefPoint, pg::kMaxGroundPoints);
  EXPECT_EQ(pg::validateGuidanceCommand(cmd, t, &cat, &g), pg::GuidanceStatus::kBadParameters);
  cmd.align_target = target(pg::PointingTargetKind::kEcefPoint, 12);
  EXPECT_EQ(pg::validateGuidanceCommand(cmd, t, &cat, &g), pg::GuidanceStatus::kTargetSlotEmpty);
  ASSERT_TRUE(g.set(12, 0.0, 0.0, 0.0));
  EXPECT_EQ(pg::validateGuidanceCommand(cmd, t, &cat, &g), pg::GuidanceStatus::kOk);
}

TEST(ValidateGuidance, DoesNotJudgeTheSkyAtCommandTime) {
  // The timing distinction this design turns on. Collinearity of the two
  // *targets* is a property of the sky at a moment, not of the command: a
  // constraint that is fine now can degenerate an orbit later as the Sun, the
  // target and the vehicle line up. Settling it at command time would either
  // reject a legal command or bless one that fails later, so it is deliberately
  // left to the per-cycle solve.
  const pg::BodyVectorTable t = defaultTable();
  pg::GuidanceCommand cmd;
  cmd.align_vector = body(pg::BodyVectorKind::kBodyZ);
  cmd.constrain_vector = body(pg::BodyVectorKind::kBodyX);
  cmd.align_target = target(pg::PointingTargetKind::kJ2000X);
  cmd.constrain_target = target(pg::PointingTargetKind::kJ2000X);  // identical!
  EXPECT_EQ(pg::validateGuidanceCommand(cmd, t, nullptr, nullptr), pg::GuidanceStatus::kOk);

  // ... and the solve is where it is caught.
  const pg::GuidanceContext ctx = leoContext();
  pm::Quat<pmf::Body, pmf::ECI> q;
  pm::Vec3<pmf::Body> rate;
  EXPECT_EQ(pg::solveGuidanceAttitude(cmd, t, ctx, q, rate),
            pg::GuidanceStatus::kDirectionsCollinear);
}

// ---------------------------------------------------------------------------
// The solve, and the well-known modes as instances of it
// ---------------------------------------------------------------------------

TEST(SolveGuidance, NadirHoldIsJustACommand) {
  // ALIGN -Z with NADIR, CONSTRAIN +X toward LVLH_X. Not a mode in the code.
  const pg::BodyVectorTable t = defaultTable();
  const pg::GuidanceContext ctx = leoContext();
  pg::GuidanceCommand cmd;
  cmd.align_vector = body(pg::BodyVectorKind::kBodyZ, 0, /*neg=*/true);
  cmd.align_target = target(pg::PointingTargetKind::kNadir);
  cmd.constrain_vector = body(pg::BodyVectorKind::kBodyX);
  cmd.constrain_target = target(pg::PointingTargetKind::kLvlhX);
  ASSERT_EQ(pg::validateGuidanceCommand(cmd, t, nullptr, nullptr), pg::GuidanceStatus::kOk);

  pm::Quat<pmf::Body, pmf::ECI> q;
  pm::Vec3<pmf::Body> rate;
  ASSERT_EQ(pg::solveGuidanceAttitude(cmd, t, ctx, q, rate), pg::GuidanceStatus::kOk);

  // -Z body must land on nadir.
  const Eigen::Vector3d nadir = -ctx.observer_position_m.eigen().normalized();
  const Eigen::Vector3d in_body = q.rotate(pm::Vec3<pmf::ECI>(nadir)).eigen();
  EXPECT_LT((in_body - Eigen::Vector3d(0.0, 0.0, -1.0)).norm(), 1e-12);
  // A nadir-holding vehicle turns once per orbit, and the rate must show it.
  EXPECT_NEAR(rate.norm(), 1.083e-3, 5e-6) << "nadir hold must carry the orbit rate";
}

TEST(SolveGuidance, SunSafeIsJustACommand) {
  const pg::BodyVectorTable t = defaultTable();
  const pg::GuidanceContext ctx = leoContext();
  pg::GuidanceCommand cmd;
  cmd.align_vector = body(pg::BodyVectorKind::kSunSensor, 0);  // +X
  cmd.align_target = target(pg::PointingTargetKind::kSun);
  cmd.constrain_vector = body(pg::BodyVectorKind::kBodyZ);
  // Orbit normal rather than nadir: in this fixture the Sun sits along +X and so
  // does the radial direction, which would make the two commanded directions
  // collinear. That is a property of this test's geometry, not of the command
  // shape — and the solver refusing it is exercised in
  // ValidateGuidance.DoesNotJudgeTheSkyAtCommandTime.
  cmd.constrain_target = target(pg::PointingTargetKind::kJ2000Z);
  ASSERT_EQ(pg::validateGuidanceCommand(cmd, t, nullptr, nullptr), pg::GuidanceStatus::kOk);

  pm::Quat<pmf::Body, pmf::ECI> q;
  pm::Vec3<pmf::Body> rate;
  ASSERT_EQ(pg::solveGuidanceAttitude(cmd, t, ctx, q, rate), pg::GuidanceStatus::kOk);
  const Eigen::Vector3d to_sun =
      (ctx.sun_position_m.eigen() - ctx.observer_position_m.eigen()).normalized();
  const Eigen::Vector3d in_body = q.rotate(pm::Vec3<pmf::ECI>(to_sun)).eigen();
  EXPECT_LT((in_body - Eigen::Vector3d(1.0, 0.0, 0.0)).norm(), 1e-9);
}

TEST(SolveGuidance, SatelliteTrackIsJustACommand) {
  pg::TargetCatalog cat;
  const double r = pc::wgs84::kSemiMajorAxis + 620.0e3;
  const double v = std::sqrt(pc::gravity::kGM / r);
  pg::StateVectorSlot slot;
  slot.epoch = leoContext().t;
  slot.position_m = pm::Vec3<pmf::ECI>(r, 40.0e3, 0.0);
  slot.velocity_m_s = pm::Vec3<pmf::ECI>(0.0, v, 0.0);
  slot.sigma_at_epoch_m = 20.0;
  ASSERT_EQ(cat.loadStateVector(1, slot), pg::TargetStatus::kOk);

  pg::GuidanceContext ctx = leoContext();
  ctx.catalog = &cat;
  const pg::BodyVectorTable t = defaultTable();

  pg::GuidanceCommand cmd;
  cmd.align_vector = body(pg::BodyVectorKind::kCamera, 0);  // +Z
  cmd.align_target = target(pg::PointingTargetKind::kSatState, 1);
  cmd.constrain_vector = body(pg::BodyVectorKind::kBodyX);
  cmd.constrain_target = target(pg::PointingTargetKind::kNadir);
  ASSERT_EQ(pg::validateGuidanceCommand(cmd, t, &cat, nullptr), pg::GuidanceStatus::kOk);

  pm::Quat<pmf::Body, pmf::ECI> q;
  pm::Vec3<pmf::Body> rate;
  ASSERT_EQ(pg::solveGuidanceAttitude(cmd, t, ctx, q, rate), pg::GuidanceStatus::kOk);

  pg::TargetState st;
  ASSERT_EQ(cat.positionAt(pg::TargetKind::kStateVector, 1, ctx.t, st), pg::TargetStatus::kOk);
  const Eigen::Vector3d los =
      (st.position_m.eigen() - ctx.observer_position_m.eigen()).normalized();
  const Eigen::Vector3d in_body = q.rotate(pm::Vec3<pmf::ECI>(los)).eigen();
  EXPECT_LT((in_body - Eigen::Vector3d(0.0, 0.0, 1.0)).norm(), 1e-12)
      << "the camera boresight is not on the tracked satellite";
  EXPECT_GT(rate.norm(), 0.0) << "tracking a moving satellite needs a non-zero feedforward";
}

TEST(SolveGuidance, TheConstraintIsOptimalNotMerelyApplied) {
  // "As close as possible" is a claim about optimality, so it is tested as one:
  // no small roll about the aligned axis may do better.
  const pg::BodyVectorTable t = defaultTable();
  const pg::GuidanceContext ctx = leoContext();
  pg::GuidanceCommand cmd;
  cmd.align_vector = body(pg::BodyVectorKind::kBodyZ);
  cmd.align_target = target(pg::PointingTargetKind::kNadir);
  cmd.constrain_vector = body(pg::BodyVectorKind::kBodyX);
  cmd.constrain_target = target(pg::PointingTargetKind::kJ2000Z);

  pm::Quat<pmf::Body, pmf::ECI> q;
  pm::Vec3<pmf::Body> rate;
  ASSERT_EQ(pg::solveGuidanceAttitude(cmd, t, ctx, q, rate), pg::GuidanceStatus::kOk);

  const Eigen::Vector3d align_eci = -ctx.observer_position_m.eigen().normalized();
  const Eigen::Vector3d ref(0.0, 0.0, 1.0);
  const Eigen::Vector3d constrain_eci = q.inverse().rotate(pm::Vec3<pmf::Body>(1, 0, 0)).eigen();
  const double best = std::acos(std::clamp(constrain_eci.dot(ref), -1.0, 1.0));
  for (double droll : {-0.1, -0.02, 0.02, 0.1}) {
    const Eigen::Vector3d perturbed = Eigen::AngleAxisd(droll, align_eci) * constrain_eci;
    const double sep = std::acos(std::clamp(perturbed.dot(ref), -1.0, 1.0));
    EXPECT_LE(best, sep + 1e-12) << "a roll of " << droll << " rad beats the returned solution";
  }
}

TEST(SolveGuidance, TheReturnedAttitudeIsUnitAndCanonical) {
  const pg::BodyVectorTable t = defaultTable();
  const pg::GuidanceContext ctx = leoContext();
  pg::GuidanceCommand cmd;
  cmd.align_vector = body(pg::BodyVectorKind::kBodyZ);
  cmd.align_target = target(pg::PointingTargetKind::kNadir);
  cmd.constrain_vector = body(pg::BodyVectorKind::kBodyX);
  cmd.constrain_target = target(pg::PointingTargetKind::kJ2000Z);
  pm::Quat<pmf::Body, pmf::ECI> q;
  pm::Vec3<pmf::Body> rate;
  ASSERT_EQ(pg::solveGuidanceAttitude(cmd, t, ctx, q, rate), pg::GuidanceStatus::kOk);
  EXPECT_NEAR(q.core().coeffs().norm(), 1.0, 1e-12);
  EXPECT_GE(q.core().w(), 0.0) << "the controller's error term assumes a canonical quaternion";
}

TEST(SolveGuidance, TheFeedforwardRateMatchesAFiniteDifference) {
  // The analytic triad derivative against the thing it replaces. Differencing
  // successive commanded quaternions is what a naive implementation would do; it
  // is avoided in flight because it amplifies a day-old TLE's position noise by
  // 1/dt, but with noiseless inputs it is exactly the right cross-check.
  const pg::BodyVectorTable t = defaultTable();
  pg::GuidanceCommand cmd;
  cmd.align_vector = body(pg::BodyVectorKind::kBodyZ, 0, true);
  cmd.align_target = target(pg::PointingTargetKind::kNadir);
  cmd.constrain_vector = body(pg::BodyVectorKind::kBodyX);
  cmd.constrain_target = target(pg::PointingTargetKind::kLvlhX);

  const pg::GuidanceContext base = leoContext();
  auto contextAt = [&](double dt) {
    // Advance the observer on a two-body arc; the guidance rate must match how
    // the commanded attitude actually moves.
    pg::GuidanceContext c = base;
    const Eigen::Vector3d r = base.observer_position_m.eigen();
    const Eigen::Vector3d v = base.observer_velocity_m_s.eigen();
    const double rn = r.norm();
    const double n = std::sqrt(pc::gravity::kGM / (rn * rn * rn));
    const Eigen::Vector3d h = r.cross(v).normalized();
    const Eigen::AngleAxisd rot(n * dt, h);
    c.observer_position_m = pm::Vec3<pmf::ECI>(rot * r);
    c.observer_velocity_m_s = pm::Vec3<pmf::ECI>(rot * v);
    return c;
  };

  pm::Quat<pmf::Body, pmf::ECI> q0;
  pm::Vec3<pmf::Body> rate;
  ASSERT_EQ(pg::solveGuidanceAttitude(cmd, t, base, q0, rate), pg::GuidanceStatus::kOk);

  const double dt = 0.05;
  pm::Quat<pmf::Body, pmf::ECI> qp;
  pm::Quat<pmf::Body, pmf::ECI> qm;
  pm::Vec3<pmf::Body> ignored;
  ASSERT_EQ(pg::solveGuidanceAttitude(cmd, t, contextAt(dt), qp, ignored), pg::GuidanceStatus::kOk);
  ASSERT_EQ(pg::solveGuidanceAttitude(cmd, t, contextAt(-dt), qm, ignored),
            pg::GuidanceStatus::kOk);

  // Body rate from the finite difference: dq = q(+dt) * q(-dt)^-1 is a small
  // rotation of angle |omega| dt about the body rate axis, over 2 dt.
  const Eigen::Matrix3d Ap = qp.core().toRotationMatrix();
  const Eigen::Matrix3d Am = qm.core().toRotationMatrix();
  const Eigen::Matrix3d dA = (Ap - Am) / (2.0 * dt);
  // A_dot = -[omega]x A  =>  [omega]x = -A_dot A^T
  const Eigen::Matrix3d wx = -dA * q0.core().toRotationMatrix().transpose();
  const Eigen::Vector3d omega_fd(wx(2, 1), wx(0, 2), wx(1, 0));

  EXPECT_LT((omega_fd - rate.eigen()).norm(), 1e-6 * std::max(1e-3, rate.norm()))
      << "analytic feedforward " << rate.eigen().transpose() << " vs finite difference "
      << omega_fd.transpose();
}

TEST(SolveGuidance, AnUnavailableTargetIsReportedNotSubstituted) {
  pg::TargetCatalog cat;
  pg::GuidanceContext ctx = leoContext();
  ctx.catalog = &cat;
  const pg::BodyVectorTable t = defaultTable();
  pg::GuidanceCommand cmd;
  cmd.align_vector = body(pg::BodyVectorKind::kCamera, 0);
  cmd.align_target = target(pg::PointingTargetKind::kSatTle, 3);  // empty slot
  cmd.constrain_vector = body(pg::BodyVectorKind::kBodyX);
  cmd.constrain_target = target(pg::PointingTargetKind::kNadir);
  pm::Quat<pmf::Body, pmf::ECI> q;
  pm::Vec3<pmf::Body> rate;
  EXPECT_EQ(pg::solveGuidanceAttitude(cmd, t, ctx, q, rate), pg::GuidanceStatus::kTargetSlotEmpty);
}
