/// @file Unit tests for the exponential atmosphere and cannon-ball drag models
/// (REQ-SIM-002; design doc §5.2).
///
/// The atmosphere tests pin the properties a naive implementation gets wrong:
/// that altitude is *geodetic* (equator and pole differ by the 21 km flattening,
/// which is several thermospheric scale heights), that it is invariant under
/// rotation about the spin axis (the claim that lets us skip the ECI→ECEF
/// reduction), and that the 28-band piecewise fit is monotone and continuous
/// across every band boundary rather than merely correct at the tabulated points.
///
/// The drag tests check the closed-form magnitude against an INDEPENDENT hand
/// computation, the quadratic speed dependence, and — the one most worth having —
/// that the force opposes the *atmosphere-relative* velocity rather than the
/// inertial one, which is what fails if the co-rotation term is dropped.

#include <gtest/gtest.h>

#include <cmath>
#include <Eigen/Core>
#include <Eigen/Geometry>

#include "constants/constants.hpp"
#include "dynamics/force_torque.hpp"
#include "math/frames.hpp"
#include "math/typed_vector.hpp"
#include "state/truth_state.hpp"
#include "time/timescales.hpp"
#include "world/atmosphere.hpp"
#include "world/drag.hpp"

namespace world = polaris::sim::world;
namespace dyn = polaris::sim::dynamics;
namespace pm = polaris::math;
namespace pmf = polaris::math::frames;
namespace pt = polaris::time;
namespace pc = polaris::constants;

namespace {

using Eci = pm::Vec3<pmf::ECI>;
using Body = pm::Vec3<pmf::Body>;

constexpr double kA = pc::wgs84::kSemiMajorAxis;
constexpr double kB = pc::wgs84::kSemiMinorAxis;
constexpr double kOmega = pc::wgs84::kEarthRate;

polaris::state::TruthState at(const Eigen::Vector3d& r, const Eigen::Vector3d& v) {
  polaris::state::TruthState s;
  s.position = Eci(r);
  s.velocity = Eci(v);
  s.epoch = pt::Tai::fromNanosecondsSinceEpoch(1'600'000'000'000'000'000LL);  // ~2020
  return s;
}

/// A density resolver that reports one fixed value everywhere — isolates the
/// drag formula from the atmosphere model.
world::DensityFn uniform(double rho) {
  return [rho](const pt::Tai&, const Eci&) { return rho; };
}

}  // namespace

// --- Geodetic altitude -------------------------------------------------------

TEST(Atmosphere, AltitudeOnTheEquatorIsMeasuredFromTheSemiMajorAxis) {
  const double h = 400e3;
  EXPECT_NEAR(world::geodeticAltitude(Eci(Eigen::Vector3d(kA + h, 0.0, 0.0))), h, 1e-6);
}

TEST(Atmosphere, AltitudeOverThePoleIsMeasuredFromTheSemiMinorAxis) {
  // The test that fails if geodetic altitude is quietly geocentric: the polar
  // radius is 21 km shorter, and 21 km is several scale heights up there.
  const double h = 400e3;
  EXPECT_NEAR(world::geodeticAltitude(Eci(Eigen::Vector3d(0.0, 0.0, kB + h))), h, 1e-6);
  EXPECT_NEAR(world::geodeticAltitude(Eci(Eigen::Vector3d(0.0, 0.0, -(kB + h)))), h, 1e-6);
}

TEST(Atmosphere, GeodeticAltitudeExceedsGeocentricAwayFromTheEquator) {
  // At mid-latitude the ellipsoid is below the sphere of radius a, so geodetic
  // altitude is the larger of the two. Sign, not magnitude, is the point.
  const Eigen::Vector3d r = Eigen::Vector3d(1.0, 0.0, 1.0).normalized() * (kA + 400e3);
  EXPECT_GT(world::geodeticAltitude(Eci(r)), r.norm() - kA);
}

TEST(Atmosphere, AltitudeIsInvariantUnderRotationAboutTheSpinAxis) {
  // This is the property that lets the model take an ECI position with no
  // ECI->ECEF reduction and no EOP. If it ever fails, that shortcut is invalid.
  const Eigen::Vector3d r(3.0e6, 1.0e6, 5.5e6);
  const double h0 = world::geodeticAltitude(Eci(r));
  for (double angle : {0.3, 1.1, 2.7, 5.9}) {
    const Eigen::Vector3d rot = Eigen::AngleAxisd(angle, Eigen::Vector3d::UnitZ()) * r;
    EXPECT_NEAR(world::geodeticAltitude(Eci(rot)), h0, 1e-7) << "angle " << angle;
  }
}

TEST(Atmosphere, AltitudeRoundTripsFromGeodeticCoordinatesAtEveryLatitude) {
  // The iteration converges slowest where cos(lat) -> 0, so the poles are the
  // only place accuracy is in question. Build points from the EXACT forward
  // transform (geodetic -> cartesian is closed-form) and check the iteration
  // recovers the altitude it started from, right up to the axis.
  const auto from_geodetic = [](double lat, double h) {
    const double e2 = pc::wgs84::kEccentricitySq;
    const double s = std::sin(lat);
    const double n = kA / std::sqrt(1.0 - e2 * s * s);
    return Eigen::Vector3d((n + h) * std::cos(lat), 0.0, (n * (1.0 - e2) + h) * s);
  };

  for (double lat_deg : {0.0, 30.0, 45.0, 60.0, 80.0, 89.0, 89.9, 89.99, 90.0}) {
    const double lat = lat_deg * M_PI / 180.0;
    for (double h : {0.0, 400e3, 800e3, 2000e3, 35786e3}) {
      EXPECT_NEAR(world::geodeticAltitude(Eci(from_geodetic(lat, h))), h, 0.01)
          << "lat " << lat_deg << " deg, alt " << h << " m";
    }
  }
}

TEST(Atmosphere, AltitudeIsNegativeInsideTheEllipsoid) {
  EXPECT_LT(world::geodeticAltitude(Eci(Eigen::Vector3d(kA - 1000.0, 0.0, 0.0))), 0.0);
}

// --- Exponential density -----------------------------------------------------

TEST(Atmosphere, SeaLevelDensityMatchesTheStandardAtmosphere) {
  EXPECT_NEAR(world::exponentialDensity(0.0), 1.225, 1e-9);
}

TEST(Atmosphere, DensityIsStrictlyDecreasingAllTheWayUp) {
  double previous = world::exponentialDensity(0.0);
  for (double h = 1e3; h <= 1200e3; h += 1e3) {
    const double rho = world::exponentialDensity(h);
    ASSERT_LT(rho, previous) << "not decreasing at " << h << " m";
    ASSERT_GT(rho, 0.0) << "non-positive at " << h << " m";
    previous = rho;
  }
}

TEST(Atmosphere, DensityIsContinuousAcrossEveryBandBoundary) {
  // The fit is piecewise; a mis-transcribed base density or scale height shows
  // up as a jump at the seam even though each band is smooth on its own.
  const double boundaries_km[] = {25,  30,  40,  50,  60,  70,  80,  90,  100,
                                  110, 120, 130, 140, 150, 180, 200, 250, 300,
                                  350, 400, 450, 500, 600, 700, 800, 900, 1000};
  for (double km : boundaries_km) {
    const double h = km * 1e3;
    const double below = world::exponentialDensity(h - 1.0);
    const double above = world::exponentialDensity(h + 1.0);
    // Bands are fitted, not analytically matched, so allow a few percent step.
    EXPECT_NEAR(above / below, 1.0, 0.05) << "discontinuity at " << km << " km";
  }
}

TEST(Atmosphere, EveryTabulatedBandMatchesVallado) {
  // The band table is transcribed by hand from Vallado Table 8-4, so it gets a
  // second independent transcription here rather than only structural checks. A
  // scale-height typo that stays inside the continuity tolerance and preserves
  // monotonicity is invisible to every other test in this file.
  //
  // Columns: base altitude [km], base density [kg/m^3], scale height [km].
  struct Row {
    double h0_km;
    double rho0;
    double scale_km;
  };

  const Row table[] = {
      {0, 1.225, 7.249},         {25, 3.899e-2, 6.349},    {30, 1.774e-2, 6.682},
      {40, 3.972e-3, 7.554},     {50, 1.057e-3, 8.382},    {60, 3.206e-4, 7.714},
      {70, 8.770e-5, 6.549},     {80, 1.905e-5, 5.799},    {90, 3.396e-6, 5.382},
      {100, 5.297e-7, 5.877},    {110, 9.661e-8, 7.263},   {120, 2.438e-8, 9.473},
      {130, 8.484e-9, 12.636},   {140, 3.845e-9, 16.149},  {150, 2.070e-9, 22.523},
      {180, 5.464e-10, 29.740},  {200, 2.789e-10, 37.105}, {250, 7.248e-11, 45.546},
      {300, 2.418e-11, 53.628},  {350, 9.518e-12, 53.298}, {400, 3.725e-12, 58.515},
      {450, 1.585e-12, 60.828},  {500, 6.967e-13, 63.822}, {600, 1.454e-13, 71.835},
      {700, 3.614e-14, 88.667},  {800, 1.170e-14, 124.64}, {900, 5.245e-15, 181.05},
      {1000, 3.019e-15, 268.00},
  };

  for (const Row& row : table) {
    const double h0 = row.h0_km * 1e3;
    // At the base of a band the exponential is exactly the base density...
    EXPECT_NEAR(world::exponentialDensity(h0), row.rho0, row.rho0 * 1e-12)
        << "base density at " << row.h0_km << " km";
    // ...and the falloff a kilometre up pins the scale height. One km stays
    // inside every band (the narrowest, 25-30 km, is five wide), so this always
    // exercises the row under test and not its neighbour.
    constexpr double kProbe = 1000.0;
    EXPECT_NEAR(world::exponentialDensity(h0 + kProbe),
                row.rho0 * std::exp(-kProbe / (row.scale_km * 1e3)), row.rho0 * 1e-12)
        << "scale height at " << row.h0_km << " km";
  }
}

TEST(Atmosphere, ThereIsNoAtmosphereBelowTheEllipsoid) {
  EXPECT_EQ(world::exponentialDensity(-1.0), 0.0);
  EXPECT_EQ(world::exponentialDensity(std::nan("")), 0.0);
}

TEST(Atmosphere, LeoDensityIsInTheRightBallpark) {
  // ~1e-12 kg/m^3 at 400 km is the number every LEO mission quotes.
  const double rho = world::exponentialDensity(400e3);
  EXPECT_GT(rho, 1e-13);
  EXPECT_LT(rho, 1e-11);
}

TEST(Atmosphere, ResolverAdaptorAgreesWithTheDirectCall) {
  const Eigen::Vector3d r(kA + 300e3, 0.0, 0.0);
  const pt::Tai epoch = pt::Tai::fromNanosecondsSinceEpoch(0);
  EXPECT_DOUBLE_EQ(world::exponentialAtmosphere(epoch, Eci(r)), world::exponentialDensity(300e3));
}

// --- Drag --------------------------------------------------------------------

TEST(Drag, MagnitudeMatchesTheClosedForm) {
  const double rho = 1e-12, area = 2.0, mass = 500.0, cd = 2.2;
  world::AtmosphericDrag drag(area, mass, cd, uniform(rho));

  // Polar position with a purely +Z velocity: omega x r is zero on the axis, so
  // v_rel == v and the expected value needs no co-rotation bookkeeping.
  const Eigen::Vector3d r(0.0, 0.0, kB + 400e3);
  const Eigen::Vector3d v(0.0, 0.0, 7500.0);
  const double expected = 0.5 * rho * cd * area / mass * 7500.0 * 7500.0;

  EXPECT_NEAR(drag.acceleration(at(r, v)).eigen().norm(), expected, expected * 1e-12);
}

TEST(Drag, OpposesTheAtmosphereRelativeVelocityNotTheInertialOne) {
  // The co-rotation test. A satellite over the equator moving purely radially
  // has zero inertial transverse velocity, but the atmosphere beneath it is
  // sweeping past at ~460 m/s — so the drag force must have a transverse
  // component. Drop `omega x r` and this force comes out purely radial.
  world::AtmosphericDrag drag(2.0, 500.0, 2.2, uniform(1e-11));
  const Eigen::Vector3d r(kA + 300e3, 0.0, 0.0);
  const Eigen::Vector3d v(100.0, 0.0, 0.0);  // straight up, no transverse motion

  const Eigen::Vector3d a = drag.acceleration(at(r, v)).eigen();
  EXPECT_LT(a.x(), 0.0);                        // opposes the climb
  EXPECT_GT(a.y(), 0.0);                        // opposes the -Y relative wind
  EXPECT_GT(std::abs(a.y()), std::abs(a.x()));  // 460 m/s wind dominates 100 m/s climb

  // And it is exactly anti-parallel to v_rel.
  const Eigen::Vector3d v_rel = world::AtmosphericDrag::relativeVelocity(at(r, v));
  EXPECT_NEAR(a.normalized().dot(v_rel.normalized()), -1.0, 1e-12);
}

TEST(Drag, RelativeVelocitySubtractsTheCoRotatingWind) {
  const Eigen::Vector3d r(kA, 0.0, 0.0);
  const Eigen::Vector3d v(0.0, 7500.0, 0.0);
  const Eigen::Vector3d v_rel = world::AtmosphericDrag::relativeVelocity(at(r, v));
  // Wind at the equator is +Y at omega*a; an eastward orbit sees that much less.
  EXPECT_NEAR(v_rel.y(), 7500.0 - kOmega * kA, 1e-9);
  EXPECT_NEAR(v_rel.x(), 0.0, 1e-12);
  EXPECT_NEAR(v_rel.z(), 0.0, 1e-12);
}

TEST(Drag, ScalesQuadraticallyWithRelativeSpeed) {
  world::AtmosphericDrag drag(2.0, 500.0, 2.2, uniform(1e-12));
  const Eigen::Vector3d r(0.0, 0.0, kB + 400e3);  // on-axis: v_rel == v

  const double a1 = drag.acceleration(at(r, Eigen::Vector3d(0, 0, 1000.0))).eigen().norm();
  const double a2 = drag.acceleration(at(r, Eigen::Vector3d(0, 0, 2000.0))).eigen().norm();
  EXPECT_NEAR(a2 / a1, 4.0, 1e-9);
}

TEST(Drag, ScalesLinearlyWithDensityAndBallisticCoefficient) {
  const Eigen::Vector3d r(0.0, 0.0, kB + 400e3);
  const Eigen::Vector3d v(0.0, 0.0, 7500.0);

  world::AtmosphericDrag base(2.0, 500.0, 2.2, uniform(1e-12));
  world::AtmosphericDrag denser(2.0, 500.0, 2.2, uniform(3e-12));
  world::AtmosphericDrag bigger(6.0, 500.0, 2.2, uniform(1e-12));
  world::AtmosphericDrag heavier(2.0, 1000.0, 2.2, uniform(1e-12));

  const double a0 = base.acceleration(at(r, v)).eigen().norm();
  EXPECT_NEAR(denser.acceleration(at(r, v)).eigen().norm() / a0, 3.0, 1e-9);
  EXPECT_NEAR(bigger.acceleration(at(r, v)).eigen().norm() / a0, 3.0, 1e-9);
  EXPECT_NEAR(heavier.acceleration(at(r, v)).eigen().norm() / a0, 0.5, 1e-9);
  EXPECT_NEAR(base.ballisticCoefficient(), 2.2 * 2.0 / 500.0, 1e-15);
}

TEST(Drag, VanishesInVacuum) {
  world::AtmosphericDrag drag(2.0, 500.0, 2.2, uniform(0.0));
  const Eigen::Vector3d r(kA + 800e3, 0.0, 0.0);
  EXPECT_TRUE(drag.acceleration(at(r, Eigen::Vector3d(0, 7500, 0))).eigen().isZero());
}

TEST(Drag, VanishesWithoutADensityResolver) {
  world::AtmosphericDrag drag(2.0, 500.0, 2.2, nullptr);
  const Eigen::Vector3d r(kA + 400e3, 0.0, 0.0);
  EXPECT_TRUE(drag.acceleration(at(r, Eigen::Vector3d(0, 7500, 0))).eigen().isZero());
}

TEST(Drag, ResolverSeesTheStatesEpochAndPosition) {
  const Eigen::Vector3d r(kA + 250e3, 0.0, 0.0);
  const polaris::state::TruthState s = at(r, Eigen::Vector3d(0, 7500, 0));

  bool called = false;
  world::AtmosphericDrag drag(2.0, 500.0, 2.2, [&](const pt::Tai& t, const Eci& p) {
    called = true;
    EXPECT_EQ(t.nanosecondsSinceEpoch(), s.epoch.nanosecondsSinceEpoch());
    EXPECT_TRUE(p.eigen().isApprox(r));
    return 1e-11;
  });
  drag.acceleration(s);
  EXPECT_TRUE(called);
}

TEST(Drag, UsesTheRealAtmosphereEndToEnd) {
  // The whole stack: geodetic altitude -> band lookup -> drag. ~1e-6 m/s^2 at
  // 400 km is the textbook LEO drag magnitude.
  world::AtmosphericDrag drag(2.0, 500.0, 2.2, world::exponentialAtmosphere);
  const Eigen::Vector3d r(kA + 400e3, 0.0, 0.0);
  const double a = drag.acceleration(at(r, Eigen::Vector3d(0, 7670, 0))).eigen().norm();
  EXPECT_GT(a, 1e-8);
  EXPECT_LT(a, 1e-5);
}

// --- Aero disturbance torque -------------------------------------------------

TEST(Drag, IsTorqueFreeWithCenterOfPressureAtCenterOfMass) {
  world::AtmosphericDrag drag(2.0, 500.0, 2.2, uniform(1e-11));
  const Eigen::Vector3d r(kA + 300e3, 0.0, 0.0);
  EXPECT_TRUE(drag.torque(at(r, Eigen::Vector3d(0, 7700, 0))).eigen().isZero());
}

TEST(Drag, CenterOfPressureOffsetProducesPerpendicularDisturbanceTorque) {
  world::AtmosphericDrag drag(2.0, 500.0, 2.2, uniform(1e-11));
  const Eigen::Vector3d r_cp(0.1, 0.0, 0.0);
  drag.setCenterOfPressureOffset(Body(r_cp));

  const Eigen::Vector3d r(kA + 300e3, 0.0, 0.0);
  const polaris::state::TruthState s = at(r, Eigen::Vector3d(0, 7700, 0));
  const Eigen::Vector3d tau = drag.torque(s).eigen();

  ASSERT_FALSE(tau.isZero());
  // tau = r_cp x F is perpendicular to both factors, by construction.
  EXPECT_NEAR(tau.dot(r_cp), 0.0, 1e-20);

  // And its magnitude is |r_cp| |F| sin(angle) with F the body-frame force.
  const Eigen::Matrix3d A = s.attitude.core().toRotationMatrix();
  const Eigen::Vector3d f_body = A * (500.0 * drag.acceleration(s).eigen());
  EXPECT_NEAR(tau.norm(), r_cp.cross(f_body).norm(), 1e-18);
  // Not just the magnitude: the whole vector, against the cross product computed
  // here from the force the model itself reports (design doc §5.3 lever arm).
  EXPECT_LT((tau - r_cp.cross(f_body)).norm(), 1e-18);
}

TEST(Drag, TorqueVanishesInVacuum) {
  world::AtmosphericDrag drag(2.0, 500.0, 2.2, uniform(0.0));
  drag.setCenterOfPressureOffset(Body(Eigen::Vector3d(0.1, 0.0, 0.0)));
  const Eigen::Vector3d r(kA + 800e3, 0.0, 0.0);
  EXPECT_TRUE(drag.torque(at(r, Eigen::Vector3d(0, 7500, 0))).eigen().isZero());
}

TEST(Drag, ComposesWithOtherForceModels) {
  dyn::TwoBodyGravity gravity;
  world::AtmosphericDrag drag(2.0, 500.0, 2.2, uniform(1e-11));
  dyn::CompositeForceModel composite;
  composite.add(&gravity);
  composite.add(&drag);

  const Eigen::Vector3d r(kA + 300e3, 0.0, 0.0);
  const polaris::state::TruthState s = at(r, Eigen::Vector3d(0, 7700, 0));
  const Eigen::Vector3d expected = gravity.acceleration(s).eigen() + drag.acceleration(s).eigen();
  EXPECT_TRUE(composite.acceleration(s).eigen().isApprox(expected));
}
