/// @file Unit tests for the ECI magnetic-field adaptor and the residual-dipole
/// disturbance torque (REQ-SIM-002; design doc §5.2, §5.3).
///
/// The torque tests use a *fabricated* field rather than IGRF, so `tau = m x B`
/// is checkable by hand and a failure points at the torque law rather than at
/// the field model. The adaptor tests use the real committed coefficients,
/// because what they check — that the Earth-rotation reduction actually happens,
/// and that a TAI epoch maps to the right decimal year — is only meaningful
/// against a field with real longitude structure.

#include <gtest/gtest.h>

#include <cmath>
#include <Eigen/Core>
#include <string>

#include "actuators/magnetorquer.hpp"
#include "environment/igrf.hpp"
#include "math/frames.hpp"
#include "math/quaternion.hpp"
#include "math/typed_vector.hpp"
#include "state/truth_state.hpp"
#include "time/leap_seconds.hpp"
#include "time/timescales.hpp"
#include "world/igrf_file.hpp"
#include "world/magnetic_field.hpp"

namespace {

namespace env = polaris::environment;
namespace pm = polaris::math;
namespace pt = polaris::time;
namespace world = polaris::sim::world;

using EcefFromEci = pm::Quat<pm::frames::ECEF, pm::frames::ECI>;

env::IgrfCoefficients committedCoefficients(double year) {
  env::IgrfCoefficients c{};
  std::string error;
  EXPECT_TRUE(world::loadIgrfFile(POLARIS_IGRF_COEFFS, year, c, &error)) << error;
  return c;
}

/// TAI epoch for a UTC calendar instant, via the frozen 37 s offset the other
/// sim tests use.
pt::Tai taiAt(int year, unsigned month, unsigned day, int hour = 0) {
  const std::int64_t days = pt::daysFromCivil(year, month, day);
  const std::int64_t seconds = days * 86400 + static_cast<std::int64_t>(hour) * 3600 + 37;
  return pt::Tai::fromNanosecondsSinceEpoch(seconds * 1000000000LL);
}

/// A uniform field, so the torque is exactly m x B with no geometry to unpick.
world::MagneticFieldFn uniformField(const Eigen::Vector3d& b) {
  return [b](const pt::Tai&, const pm::Vec3<pm::frames::ECI>&, pm::Vec3<pm::frames::ECI>& out) {
    out = pm::Vec3<pm::frames::ECI>(b);
    return true;
  };
}

}  // namespace

// --- Decimal year ------------------------------------------------------------

TEST(DecimalYear, MapsCalendarInstantsToYearFractions) {
  const pt::LeapSecondTable leap = pt::LeapSecondTable::frozen(37);
  double year = 0.0;

  ASSERT_TRUE(world::decimalYear(taiAt(2026, 1, 1), leap, year));
  EXPECT_NEAR(year, 2026.0, 1.0e-9);

  // 2026 is not a leap year: 1 July is day 181 of 365.
  ASSERT_TRUE(world::decimalYear(taiAt(2026, 7, 1), leap, year));
  EXPECT_NEAR(year, 2026.0 + 181.0 / 365.0, 1.0e-9);

  // 2024 is: the same calendar date sits at a slightly different fraction, which
  // is exactly what a fixed 365.25 divisor would get wrong.
  ASSERT_TRUE(world::decimalYear(taiAt(2024, 7, 1), leap, year));
  EXPECT_NEAR(year, 2024.0 + 182.0 / 366.0, 1.0e-9);

  // Time of day advances the fraction by the right amount.
  ASSERT_TRUE(world::decimalYear(taiAt(2026, 1, 1, 12), leap, year));
  EXPECT_NEAR(year, 2026.0 + 0.5 / 365.0, 1.0e-9);
}

TEST(DecimalYear, IsMonotonic) {
  const pt::LeapSecondTable leap = pt::LeapSecondTable::frozen(37);
  double previous = 0.0;
  ASSERT_TRUE(world::decimalYear(taiAt(2025, 12, 31), leap, previous));
  for (int month = 1; month <= 12; ++month) {
    double year = 0.0;
    ASSERT_TRUE(world::decimalYear(taiAt(2026, static_cast<unsigned>(month), 15), leap, year));
    EXPECT_GT(year, previous) << "month " << month;
    previous = year;
  }
}

// --- ECI adaptor -------------------------------------------------------------

TEST(EarthMagneticField, RequiresBothDependencies) {
  world::EarthMagneticField field(committedCoefficients(2026.5));
  const pt::LeapSecondTable leap = pt::LeapSecondTable::frozen(37);
  const pm::Vec3<pm::frames::ECI> r(Eigen::Vector3d(7.0e6, 0.0, 0.0));
  pm::Vec3<pm::frames::ECI> b;

  EXPECT_FALSE(field.good());
  EXPECT_FALSE(field.field(taiAt(2026, 6, 1), r, b));

  field.setEciToEcef([](const pt::Tai&, EcefFromEci& q) {
    q = EcefFromEci::Identity();
    return true;
  });
  EXPECT_FALSE(field.good()) << "leap seconds still missing";

  field.setLeapSeconds(&leap);
  EXPECT_TRUE(field.good());
  EXPECT_TRUE(field.field(taiAt(2026, 6, 1), r, b));
}

TEST(EarthMagneticField, PropagatesAResolverRefusal) {
  world::EarthMagneticField field(committedCoefficients(2026.5));
  const pt::LeapSecondTable leap = pt::LeapSecondTable::frozen(37);
  field.setLeapSeconds(&leap);
  field.setEciToEcef([](const pt::Tai&, EcefFromEci&) { return false; });

  const pm::Vec3<pm::frames::ECI> r(Eigen::Vector3d(7.0e6, 0.0, 0.0));
  pm::Vec3<pm::frames::ECI> b(Eigen::Vector3d(1.0, 2.0, 3.0));
  EXPECT_FALSE(field.field(taiAt(2026, 6, 1), r, b));
  // Untouched on refusal.
  EXPECT_EQ(b.eigen(), Eigen::Vector3d(1.0, 2.0, 3.0));
}

TEST(EarthMagneticField, ActuallyAppliesTheEarthRotation) {
  // Two rotations, same epoch and same ECI position. If the reduction were
  // skipped the field would be identical; because the field is strongly
  // longitude-dependent, rotating the Earth 90 deg underneath a fixed inertial
  // point must land on visibly different field.
  const pt::LeapSecondTable leap = pt::LeapSecondTable::frozen(37);
  const pm::Vec3<pm::frames::ECI> r(Eigen::Vector3d(6.9e6, 0.0, 1.0e6));
  const pt::Tai epoch = taiAt(2026, 6, 1);

  auto evaluate = [&](double rotation_rad) {
    world::EarthMagneticField field(committedCoefficients(2026.5));
    field.setLeapSeconds(&leap);
    field.setEciToEcef([rotation_rad](const pt::Tai&, EcefFromEci& q) {
      q = EcefFromEci(pm::Quaternion::FromAxisAngle(Eigen::Vector3d::UnitZ(), rotation_rad));
      return true;
    });
    pm::Vec3<pm::frames::ECI> b;
    EXPECT_TRUE(field.field(epoch, r, b));
    return b.eigen();
  };

  const Eigen::Vector3d unrotated = evaluate(0.0);
  const Eigen::Vector3d quarter_turn = evaluate(M_PI / 2.0);
  // Both are real fields of the right magnitude...
  EXPECT_GT(unrotated.norm(), 10.0e-6);
  EXPECT_GT(quarter_turn.norm(), 10.0e-6);
  // ...but a quarter turn of the Earth is a different place, so a difference of
  // at least a few microtesla. Equality here would mean the rotation was dropped.
  EXPECT_GT((unrotated - quarter_turn).norm(), 1.0e-6);
}

TEST(EarthMagneticField, RotationPreservesFieldMagnitude) {
  // The reduction is a pure rotation, so whatever the Earth orientation, the ECI
  // magnitude must equal the ECEF magnitude at the corresponding ECEF point.
  const pt::LeapSecondTable leap = pt::LeapSecondTable::frozen(37);
  const double angle = 1.234;
  const EcefFromEci q(pm::Quaternion::FromAxisAngle(Eigen::Vector3d::UnitZ(), angle));

  world::EarthMagneticField field(committedCoefficients(2026.5));
  field.setLeapSeconds(&leap);
  field.setEciToEcef([q](const pt::Tai&, EcefFromEci& out) {
    out = q;
    return true;
  });

  const pm::Vec3<pm::frames::ECI> r_eci(Eigen::Vector3d(5.0e6, 3.0e6, 2.0e6));
  pm::Vec3<pm::frames::ECI> b_eci;
  ASSERT_TRUE(field.field(taiAt(2026, 6, 1), r_eci, b_eci));

  double year = 0.0;
  ASSERT_TRUE(world::decimalYear(taiAt(2026, 6, 1), leap, year));
  pm::Vec3<pm::frames::ECEF> b_ecef;
  ASSERT_TRUE(
      field.igrf().field(pm::Vec3<pm::frames::ECEF>(q.core().rotate(r_eci.eigen())), year, b_ecef));

  EXPECT_NEAR(b_eci.eigen().norm(), b_ecef.eigen().norm(), 1.0e-18);
  // And it is the same vector, just expressed inertially.
  EXPECT_LT((q.core().rotate(b_eci.eigen()) - b_ecef.eigen()).norm(), 1.0e-18);
}

TEST(EarthMagneticField, ResolverAdaptorMatchesTheDirectCall) {
  const pt::LeapSecondTable leap = pt::LeapSecondTable::frozen(37);
  world::EarthMagneticField field(committedCoefficients(2026.5));
  field.setLeapSeconds(&leap);
  field.setEciToEcef([](const pt::Tai&, EcefFromEci& q) {
    q = EcefFromEci::Identity();
    return true;
  });

  const pm::Vec3<pm::frames::ECI> r(Eigen::Vector3d(6.8e6, 1.0e6, -2.0e6));
  const pt::Tai epoch = taiAt(2026, 3, 9, 7);
  pm::Vec3<pm::frames::ECI> direct;
  pm::Vec3<pm::frames::ECI> via_fn;
  ASSERT_TRUE(field.field(epoch, r, direct));
  ASSERT_TRUE(field.fieldFn()(epoch, r, via_fn));
  EXPECT_EQ(direct.eigen(), via_fn.eigen());
}

// --- Residual dipole torque --------------------------------------------------

TEST(ResidualDipoleTorque, IsTheCrossProductOfDipoleAndField) {
  polaris::state::TruthState s{};
  s.epoch = taiAt(2026, 6, 1);
  s.position = pm::Vec3<pm::frames::ECI>(Eigen::Vector3d(7.0e6, 0.0, 0.0));
  s.attitude = pm::Quat<pm::frames::Body, pm::frames::ECI>::Identity();

  const Eigen::Vector3d m(0.1, -0.2, 0.05);            // A·m^2
  const Eigen::Vector3d b(20.0e-6, 5.0e-6, -30.0e-6);  // T
  const world::ResidualDipoleTorque model(pm::Vec3<pm::frames::Body>(m), uniformField(b));

  // Identity attitude, so Body and ECI coincide and the answer is m x B directly.
  const Eigen::Vector3d expected = m.cross(b);
  EXPECT_LT((model.torque(s).eigen() - expected).norm(), 1.0e-18);
  // A dipole feels torque but no net force.
  EXPECT_EQ(model.acceleration(s).eigen(), Eigen::Vector3d::Zero());
}

TEST(ResidualDipoleTorque, TakesTheFieldIntoTheBodyFrame) {
  // Rotate the vehicle 90 deg about z. The dipole is fixed in Body, so the torque
  // must be m x (R b), NOT m x b — the distinction a frame-blind implementation
  // gets wrong.
  const double angle = M_PI / 2.0;
  polaris::state::TruthState s{};
  s.epoch = taiAt(2026, 6, 1);
  s.position = pm::Vec3<pm::frames::ECI>(Eigen::Vector3d(7.0e6, 0.0, 0.0));
  s.attitude = pm::Quat<pm::frames::Body, pm::frames::ECI>(
      pm::Quaternion::FromAxisAngle(Eigen::Vector3d::UnitZ(), angle));

  const Eigen::Vector3d m(0.3, 0.0, 0.0);
  const Eigen::Vector3d b(0.0, 40.0e-6, 0.0);
  const world::ResidualDipoleTorque model(pm::Vec3<pm::frames::Body>(m), uniformField(b));

  const Eigen::Vector3d b_body = s.attitude.core().rotate(b);
  EXPECT_LT((model.torque(s).eigen() - m.cross(b_body)).norm(), 1.0e-18);
  // Sanity: the rotation genuinely changed the answer.
  EXPECT_GT((m.cross(b_body) - m.cross(b)).norm(), 1.0e-9);
}

TEST(ResidualDipoleTorque, IsZeroWhenTheDipoleIsAlignedWithTheField) {
  polaris::state::TruthState s{};
  s.epoch = taiAt(2026, 6, 1);
  s.position = pm::Vec3<pm::frames::ECI>(Eigen::Vector3d(7.0e6, 0.0, 0.0));
  s.attitude = pm::Quat<pm::frames::Body, pm::frames::ECI>::Identity();

  const Eigen::Vector3d b(0.0, 0.0, 45.0e-6);
  const world::ResidualDipoleTorque model(
      pm::Vec3<pm::frames::Body>(Eigen::Vector3d(0.0, 0.0, 2.0)), uniformField(b));
  EXPECT_LT(model.torque(s).eigen().norm(), 1.0e-20);
}

TEST(ResidualDipoleTorque, IsZeroWithoutAField) {
  polaris::state::TruthState s{};
  s.epoch = taiAt(2026, 6, 1);
  s.position = pm::Vec3<pm::frames::ECI>(Eigen::Vector3d(7.0e6, 0.0, 0.0));
  s.attitude = pm::Quat<pm::frames::Body, pm::frames::ECI>::Identity();
  const pm::Vec3<pm::frames::Body> m(Eigen::Vector3d(1.0, 1.0, 1.0));

  // No resolver at all...
  const world::ResidualDipoleTorque unwired(m, nullptr);
  EXPECT_EQ(unwired.torque(s).eigen(), Eigen::Vector3d::Zero());

  // ...and a resolver that declines. Both give zero torque rather than a
  // guessed one, so a half-wired model is obviously inert instead of subtly off.
  const world::ResidualDipoleTorque declined(m, [](const pt::Tai&, const pm::Vec3<pm::frames::ECI>&,
                                                   pm::Vec3<pm::frames::ECI>&) { return false; });
  EXPECT_EQ(declined.torque(s).eigen(), Eigen::Vector3d::Zero());
}

TEST(ResidualDipoleTorque, AgreesWithTheMagnetorquerPath) {
  RecordProperty("verifies", "REQ-SIM-002");
  // The residual dipole and a commanded magnetorquer dipole are the same physics
  // with a different source, so the two paths must produce the same torque for
  // the same m and B. Two independent m x B implementations that disagree by a
  // sign would show as a control loop that works in one mode and diverges in
  // another (design doc §5.3).
  const Eigen::Vector3d m(0.4, -0.15, 0.6);            // A·m^2
  const Eigen::Vector3d b(18.0e-6, -25.0e-6, 9.0e-6);  // T, ECI

  polaris::state::TruthState s{};
  s.epoch = taiAt(2026, 6, 1);
  s.position = pm::Vec3<pm::frames::ECI>(Eigen::Vector3d(6.9e6, 1.0e6, -2.0e6));
  s.attitude = pm::Quat<pm::frames::Body, pm::frames::ECI>(
      pm::Quaternion::FromAxisAngle(Eigen::Vector3d(0.0, 1.0, 1.0).normalized(), 1.1));

  const world::ResidualDipoleTorque residual(pm::Vec3<pm::frames::Body>(m), uniformField(b));

  // The magnetorquer path as `closed_loop.cpp` runs it: an ideal rod (no
  // saturation, no hysteresis, no scale-factor error) commanded to exactly m,
  // crossed with the field in Body.
  polaris::sim::actuators::MagnetorquerSpec spec;
  spec.max_dipole_am2 = 10.0;
  polaris::sim::actuators::Magnetorquer mtq(spec);
  const Eigen::Vector3d m_actual = mtq.commandDipole(pm::Vec3<pm::frames::Body>(m)).eigen();
  ASSERT_LT((m_actual - m).norm(), 1.0e-15) << "ideal rod should reproduce the command";
  const Eigen::Vector3d b_body = s.attitude.core().rotate(b);

  EXPECT_LT((residual.torque(s).eigen() - m_actual.cross(b_body)).norm(), 1.0e-20);
}

TEST(ResidualDipoleTorque, HasAPlausibleMagnitudeInLowEarthOrbit) {
  // A realistic residual dipole for a small satellite is ~0.1 A·m^2 against a
  // ~30 µT LEO field, giving a few µN·m — small, but secular, so it is the term
  // that sizes the magnetorquers. This pins the units: an A·m^2/T mix-up or a
  // nT/T slip would land orders of magnitude away.
  const pt::LeapSecondTable leap = pt::LeapSecondTable::frozen(37);
  world::EarthMagneticField field(committedCoefficients(2026.5));
  field.setLeapSeconds(&leap);
  field.setEciToEcef([](const pt::Tai&, EcefFromEci& q) {
    q = EcefFromEci::Identity();
    return true;
  });

  polaris::state::TruthState s{};
  s.epoch = taiAt(2026, 6, 1);
  s.position = pm::Vec3<pm::frames::ECI>(Eigen::Vector3d(0.0, 0.0, 6871.0e3));  // polar, 500 km
  s.attitude = pm::Quat<pm::frames::Body, pm::frames::ECI>::Identity();

  const world::ResidualDipoleTorque model(
      pm::Vec3<pm::frames::Body>(Eigen::Vector3d(0.1, 0.0, 0.0)), field.fieldFn());
  const double magnitude = model.torque(s).eigen().norm();
  EXPECT_GT(magnitude, 1.0e-7);
  EXPECT_LT(magnitude, 1.0e-5);
}
