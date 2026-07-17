/// @file Tests for the IAU 2006/2000A ECI↔ECEF reduction (REQ-CONV-002, REQ-CONV-001).
///
/// The anchor is the **published ERFA/SOFA `t_c2t06a` test case** — the same
/// epoch, polar motion and expected 3×3 matrix upstream validates `eraC2t06a`
/// against. That is what proves our boundary (TAI → two-part TT/UT1 JD, arcsec →
/// radians, matrix → canonical quaternion) is wired correctly; every other test
/// here would pass just as happily against a consistently-wrong wiring.
///
/// The rest check properties the reference case cannot: that the rotation stays
/// a proper rotation over many epochs, that state round-trips carry velocity as
/// well as position, that the Earth actually turns at the right rate, and that
/// the EOP inputs reach the result at all.

#include "frames/eci_ecef.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <Eigen/Geometry>
#include <limits>

#include "constants/constants.hpp"
#include "frames/eop.hpp"
#include "math/frames.hpp"
#include "math/typed_vector.hpp"
#include "time/civil.hpp"
#include "time/leap_seconds.hpp"
#include "time/timescales.hpp"

namespace pf = polaris::frames;
namespace pm = polaris::math;
namespace pmf = polaris::math::frames;
namespace pt = polaris::time;
namespace pc = polaris::constants;

namespace {

using Eci = pm::Vec3<pmf::ECI>;
using Ecef = pm::Vec3<pmf::ECEF>;

/// TAI at 00:00:00 UTC on a civil date, plus @p offset_sec.
pt::Tai TaiAtUtc(std::int64_t y, unsigned m, unsigned d, double offset_sec = 0.0) {
  const pt::LeapSecondTable leap = pt::LeapSecondTable::historical();
  const std::int64_t tai_sec = pt::daysFromCivil(y, m, d) * 86400 + leap.deltaAtForUtcDate(y, m, d);
  return pt::Tai::fromNanosecondsSinceEpoch(tai_sec * 1'000'000'000) +
         pt::Duration::fromSecondsF(offset_sec);
}

double MjdOf(std::int64_t y, unsigned m, unsigned d) {
  return static_cast<double>(pt::daysFromCivil(y, m, d)) + pf::kMjd1970;
}

/// Representative EOP, quiet epoch (no leap nearby).
pf::EopValue QuietEop() {
  pf::EopValue e;
  e.ut1_minus_tai = -37.1770;  // ΔUT1 = -0.177 s with ΔAT = 37 s
  e.xp_arcsec = 0.0730;
  e.yp_arcsec = 0.2850;
  return e;
}

constexpr double kEquatorialRadius = pc::wgs84::kSemiMajorAxis;

}  // namespace

/// The anchor. Upstream ERFA validates `eraC2t06a` at TT = UT1 = MJD 53736
/// (2006-01-01) with xp = 2.55060238e-7 rad, yp = 1.860359247e-6 rad, and
/// publishes the resulting CRS→TRS matrix to 1e-12 (erfa v2.0.1, `src/t_erfa_c.c`,
/// `t_c2t06a`). Reproducing it through *our* API — TAI in, arcsec in, quaternion
/// out — is the end-to-end proof that the reduction is wired correctly.
///
/// The test case sets UT1 numerically equal to TT, which fixes `UT1 - TAI` at
/// exactly +32.184 s. That is physically absurd (the real value is ≈ -33 s), but
/// it is what reproduces the reference arguments, and it is the reduction's
/// wiring — not the EOP's realism — under test here.
TEST(EciEcef, MatchesPublishedErfaC2t06aReferenceCase) {
  RecordProperty("verifies", "REQ-CONV-002");
  // MJD 53736 = 2006-01-01. Build TAI so that TT lands exactly on 00:00:00.
  const std::int64_t tt_sec = pt::daysFromCivil(2006, 1, 1) * 86400;
  ASSERT_DOUBLE_EQ(MjdOf(2006, 1, 1), 53736.0);
  const pt::Tt tt = pt::Tt::fromNanosecondsSinceEpoch(tt_sec * 1'000'000'000);
  const pt::Tai t = pt::toTai(tt);

  // The two-part JD our boundary hands ERFA must be the reference's (2400000.5, 53736.0).
  const pt::JulianDate jd = pt::julianDate(tt);
  EXPECT_DOUBLE_EQ(jd.day + jd.fraction, 2400000.5 + 53736.0);

  pf::EopValue eop;
  eop.ut1_minus_tai = pc::time::kTtMinusTai;  // UT1 == TT, per the reference case
  eop.xp_arcsec = 2.55060238e-7 / pc::iau::kArcsecToRad;
  eop.yp_arcsec = 1.860359247e-6 / pc::iau::kArcsecToRad;

  pm::Quat<pmf::ECEF, pmf::ECI> q;
  ASSERT_TRUE(pf::ecefFromEci(t, eop, q));
  const Eigen::Matrix3d r = q.core().toRotationMatrix();

  // Expected rc2t from erfa v2.0.1 src/t_erfa_c.c::t_c2t06a (tolerance 1e-12).
  Eigen::Matrix3d expected;
  expected << -0.1810332128305897282, 0.9834769806938592296, 0.6555550962998436505e-4,
      -0.9834768134136214897, -0.1810332203649130832, 0.5749800844905594110e-3,
      0.5773474024748545878e-3, 0.3961816829632690581e-4, 0.9999998325501747785;

  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      EXPECT_NEAR(r(i, j), expected(i, j), 1e-12) << "element (" << i << "," << j << ")";
    }
  }
}

TEST(EciEcef, RotationIsOrthonormalAndProperOverManyEpochs) {
  RecordProperty("verifies", "REQ-CONV-002");
  const pf::EopValue eop = QuietEop();
  // Sample ~3 years at an interval that is not a whole number of days, so the
  // samples sweep all rotation phases rather than repeating one.
  for (int i = 0; i < 200; ++i) {
    const pt::Tai t = TaiAtUtc(2020, 1, 1, static_cast<double>(i) * 5432.1 * 100.0);
    pm::Quat<pmf::ECEF, pmf::ECI> q;
    ASSERT_TRUE(pf::ecefFromEci(t, eop, q)) << "i=" << i;
    const Eigen::Matrix3d r = q.core().toRotationMatrix();
    const Eigen::Matrix3d should_be_identity = r * r.transpose();
    EXPECT_TRUE(should_be_identity.isApprox(Eigen::Matrix3d::Identity(), 1e-13)) << "i=" << i;
    EXPECT_NEAR(r.determinant(), 1.0, 1e-13) << "i=" << i;  // proper: no reflection
    EXPECT_TRUE(q.core().isUnit(1e-12)) << "i=" << i;
    EXPECT_GE(q.core().scalar(), 0.0) << "i=" << i;  // canonical
  }
}

TEST(EciEcef, StateRoundTripsThroughEcefAndBack) {
  RecordProperty("verifies", "REQ-CONV-001");
  const pf::EopValue eop = QuietEop();
  const pt::Tai t = TaiAtUtc(2020, 6, 1, 12345.0);
  const Eci r0(6'524'834.0, 6'862'875.0, 6'448'296.0);  // Vallado-scale LEO state
  const Eci v0(4901.327, 5533.756, -1976.341);

  Ecef r_ecef;
  Ecef v_ecef;
  ASSERT_TRUE(pf::ecefStateFromEci(t, eop, r0, v0, r_ecef, v_ecef));
  Eci r1;
  Eci v1;
  ASSERT_TRUE(pf::eciStateFromEcef(t, eop, r_ecef, v_ecef, r1, v1));

  // Position AND velocity: a rotation-only implementation round-trips position
  // perfectly while getting velocity wrong by ~465 m/s, so velocity is the check
  // that actually bites.
  EXPECT_NEAR((r1 - r0).norm(), 0.0, 1e-6);
  EXPECT_NEAR((v1 - v0).norm(), 0.0, 1e-9);

  // The rotation-only pair must round-trip too, and agree with the state path.
  pm::Quat<pmf::ECEF, pmf::ECI> fwd;
  pm::Quat<pmf::ECI, pmf::ECEF> inv;
  ASSERT_TRUE(pf::ecefFromEci(t, eop, fwd));
  ASSERT_TRUE(pf::eciFromEcef(t, eop, inv));
  EXPECT_NEAR((fwd.rotate(r0) - r_ecef).norm(), 0.0, 1e-6);
  EXPECT_NEAR((inv.rotate(fwd.rotate(r0)) - r0).norm(), 0.0, 1e-6);
}

TEST(EciEcef, VelocityCarriesTheTransportTermNotJustTheRotation) {
  RecordProperty("verifies", "REQ-CONV-001");
  const pf::EopValue eop = QuietEop();
  const pt::Tai t = TaiAtUtc(2020, 6, 1);
  // A point bolted to the equator: zero velocity in ECEF, but ~465 m/s in ECI
  // purely from Earth's rotation. Only the transport term can produce this.
  const Ecef r_ecef(kEquatorialRadius, 0.0, 0.0);
  const Ecef v_ecef(0.0, 0.0, 0.0);

  Eci r_eci;
  Eci v_eci;
  ASSERT_TRUE(pf::eciStateFromEcef(t, eop, r_ecef, v_ecef, r_eci, v_eci));
  EXPECT_NEAR(v_eci.norm(), pc::wgs84::kEarthRate * kEquatorialRadius, 1e-3);
  EXPECT_NEAR(v_eci.norm(), 465.1, 0.1);  // the familiar equatorial ground speed
  // Rotation is rigid: the radius is unchanged, and v ⟂ r for a fixed point.
  EXPECT_NEAR(r_eci.norm(), kEquatorialRadius, 1e-6);
  EXPECT_NEAR(v_eci.dot(r_eci) / (v_eci.norm() * r_eci.norm()), 0.0, 1e-9);
}

TEST(EciEcef, EarthReturnsToTheSameOrientationAfterOneSiderealDay) {
  RecordProperty("verifies", "REQ-CONV-002");
  const pf::EopValue eop = QuietEop();
  // 2π / ω⊕ — one rotation of the Earth relative to the stars.
  const double sidereal_day = 2.0 * M_PI / pc::wgs84::kEarthRate;
  // 2π/ω⊕ with the WGS84 nominal rate = 86164.10 s — ~0.01 s longer than the
  // mean sidereal day (86164.0905 s), since kEarthRate is a rounded nominal value.
  EXPECT_NEAR(sidereal_day, 86164.10, 0.01);

  const Eci r_eci(kEquatorialRadius, 0.0, 0.0);
  const pt::Tai t0 = TaiAtUtc(2020, 6, 1);
  pm::Quat<pmf::ECEF, pmf::ECI> q0;
  pm::Quat<pmf::ECEF, pmf::ECI> q1;
  ASSERT_TRUE(pf::ecefFromEci(t0, eop, q0));
  ASSERT_TRUE(pf::ecefFromEci(t0 + pt::Duration::fromSecondsF(sidereal_day), eop, q1));

  // A sidereal day later the same inertial direction maps back to (nearly) the
  // same Earth-fixed point. Residual ~km: nominal ω⊕ is not exactly the ERA rate,
  // and precession/nutation move the frame slightly over a day.
  const double drift = (q1.rotate(r_eci) - q0.rotate(r_eci)).norm();
  EXPECT_LT(drift, 2000.0);
  // A *half* sidereal day must be nowhere near — proves the bound above is a real
  // constraint and not trivially satisfied.
  pm::Quat<pmf::ECEF, pmf::ECI> q_half;
  ASSERT_TRUE(pf::ecefFromEci(t0 + pt::Duration::fromSecondsF(0.5 * sidereal_day), eop, q_half));
  EXPECT_GT((q_half.rotate(r_eci) - q0.rotate(r_eci)).norm(), 1.0e7);
}

/// Proves the EOP inputs are actually plumbed into the reduction rather than
/// silently dropped: a 1 s error in UT1 is 1 s of extra Earth rotation, which
/// drags an equatorial ground point ≈465 m.
TEST(EciEcef, PerturbingUt1MovesTheGroundTrackByTheExpectedAmount) {
  RecordProperty("verifies", "REQ-CONV-002");
  const pt::Tai t = TaiAtUtc(2020, 6, 1);
  const Eci r_eci(kEquatorialRadius, 0.0, 0.0);

  pf::EopValue a = QuietEop();
  pf::EopValue b = QuietEop();
  b.ut1_minus_tai += 1.0;

  pm::Quat<pmf::ECEF, pmf::ECI> qa;
  pm::Quat<pmf::ECEF, pmf::ECI> qb;
  ASSERT_TRUE(pf::ecefFromEci(t, a, qa));
  ASSERT_TRUE(pf::ecefFromEci(t, b, qb));
  const double shift = (qb.rotate(r_eci) - qa.rotate(r_eci)).norm();
  EXPECT_NEAR(shift, pc::wgs84::kEarthRate * kEquatorialRadius, 1.0);
  EXPECT_NEAR(shift, 465.1, 1.0);

  // Polar motion is a much smaller effect but must still reach the result. In the
  // IAU 2006 reduction xp enters as a small rotation about the ECEF +Y axis (IERS
  // TN36 §5, W = R3(-s')·R2(xp)·R1(yp)), so perturbing it by δ displaces a fixed
  // point p by δ·|ŷ × p| — not δ·Re, which is only the upper bound (|ŷ × p| ≤ Re).
  pf::EopValue c = QuietEop();
  c.xp_arcsec += 0.1;
  pm::Quat<pmf::ECEF, pmf::ECI> qc;
  ASSERT_TRUE(pf::ecefFromEci(t, c, qc));
  const double pm_shift = (qc.rotate(r_eci) - qa.rotate(r_eci)).norm();
  const Eigen::Vector3d p = qa.rotate(r_eci).eigen();
  const double expected_pm = 0.1 * pc::iau::kArcsecToRad * Eigen::Vector3d::UnitY().cross(p).norm();
  EXPECT_GT(pm_shift, 0.5);  // plumbed through, not silently dropped
  EXPECT_NEAR(pm_shift, expected_pm, 0.02);
}

TEST(EciEcef, TableDrivenOverloadsAgreeWithTheResolvedEopCore) {
  RecordProperty("verifies", "REQ-CONV-001");
  const pt::LeapSecondTable leap = pt::LeapSecondTable::historical();
  pf::EopTable<8> table;
  ASSERT_TRUE(table.addEntry({MjdOf(2020, 6, 1), -0.1770, 0.0730, 0.2850}));
  ASSERT_TRUE(table.addEntry({MjdOf(2020, 6, 2), -0.1782, 0.0742, 0.2830}));

  // At the first entry's epoch the interpolant is exactly QuietEop().
  const pt::Tai t = TaiAtUtc(2020, 6, 1);
  pm::Quat<pmf::ECEF, pmf::ECI> from_table;
  pm::Quat<pmf::ECEF, pmf::ECI> from_core;
  ASSERT_TRUE(pf::ecefFromEci(t, table, leap, from_table));
  ASSERT_TRUE(pf::ecefFromEci(t, QuietEop(), from_core));
  EXPECT_LT(from_table.core().angularDistance(from_core.core()), 1e-12);

  // The GNSS ingest path (REQ-CONV-001) round-trips through the table too.
  const Ecef r_ecef(kEquatorialRadius, 0.0, 0.0);
  const Ecef v_ecef(0.0, 0.0, 100.0);
  Eci r_eci;
  Eci v_eci;
  ASSERT_TRUE(pf::eciStateFromEcef(t, table, leap, r_ecef, v_ecef, r_eci, v_eci));
  Ecef r_back;
  Ecef v_back;
  ASSERT_TRUE(pf::ecefStateFromEci(t, table, leap, r_eci, v_eci, r_back, v_back));
  EXPECT_NEAR((r_back - r_ecef).norm(), 0.0, 1e-6);
  EXPECT_NEAR((v_back - v_ecef).norm(), 0.0, 1e-9);
}

TEST(EciEcef, EpochOutsideTheEopTableFailsAndLeavesOutputsUntouched) {
  RecordProperty("verifies", "REQ-CONV-002");
  const pt::LeapSecondTable leap = pt::LeapSecondTable::historical();
  pf::EopTable<8> table;
  ASSERT_TRUE(table.addEntry({MjdOf(2020, 6, 1), -0.1770, 0.0730, 0.2850}));
  ASSERT_TRUE(table.addEntry({MjdOf(2020, 6, 2), -0.1782, 0.0742, 0.2830}));

  auto q = pm::Quat<pmf::ECEF, pmf::ECI>::Identity();
  EXPECT_FALSE(pf::ecefFromEci(TaiAtUtc(2021, 1, 1), table, leap, q));
  EXPECT_DOUBLE_EQ(q.core().scalar(), 1.0);  // untouched

  Ecef r_out(1.0, 2.0, 3.0);
  Ecef v_out(4.0, 5.0, 6.0);
  EXPECT_FALSE(pf::ecefStateFromEci(TaiAtUtc(2019, 1, 1), table, leap, Eci(7.0e6, 0.0, 0.0),
                                    Eci(0.0, 7.5e3, 0.0), r_out, v_out));
  EXPECT_DOUBLE_EQ(r_out.x(), 1.0);
  EXPECT_DOUBLE_EQ(v_out.x(), 4.0);
}

TEST(EciEcef, NonFiniteInputsFailAndLeaveOutputsUntouched) {
  RecordProperty("verifies", "REQ-CONV-002");
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const pt::Tai t = TaiAtUtc(2020, 6, 1);

  auto q = pm::Quat<pmf::ECEF, pmf::ECI>::Identity();
  pf::EopValue bad = QuietEop();
  bad.ut1_minus_tai = nan;
  EXPECT_FALSE(pf::ecefFromEci(t, bad, q));
  bad = QuietEop();
  bad.xp_arcsec = nan;
  EXPECT_FALSE(pf::ecefFromEci(t, bad, q));
  bad = QuietEop();
  bad.yp_arcsec = nan;
  EXPECT_FALSE(pf::ecefFromEci(t, bad, q));
  EXPECT_DOUBLE_EQ(q.core().scalar(), 1.0);

  auto qi = pm::Quat<pmf::ECI, pmf::ECEF>::Identity();
  bad = QuietEop();
  bad.ut1_minus_tai = nan;
  EXPECT_FALSE(pf::eciFromEcef(t, bad, qi));
  EXPECT_DOUBLE_EQ(qi.core().scalar(), 1.0);

  // Non-finite state vectors are rejected on both directions.
  Ecef r_ecef(1.0, 2.0, 3.0);
  Ecef v_ecef(4.0, 5.0, 6.0);
  EXPECT_FALSE(pf::ecefStateFromEci(t, QuietEop(), Eci(nan, 0.0, 0.0), Eci(0.0, 7.5e3, 0.0), r_ecef,
                                    v_ecef));
  EXPECT_FALSE(pf::ecefStateFromEci(t, QuietEop(), Eci(7.0e6, 0.0, 0.0), Eci(nan, 0.0, 0.0), r_ecef,
                                    v_ecef));
  EXPECT_DOUBLE_EQ(r_ecef.x(), 1.0);
  EXPECT_DOUBLE_EQ(v_ecef.x(), 4.0);

  Eci r_eci(1.0, 2.0, 3.0);
  Eci v_eci(4.0, 5.0, 6.0);
  EXPECT_FALSE(
      pf::eciStateFromEcef(t, QuietEop(), Ecef(nan, 0.0, 0.0), Ecef(0.0, 0.0, 0.0), r_eci, v_eci));
  EXPECT_FALSE(pf::eciStateFromEcef(t, QuietEop(), Ecef(7.0e6, 0.0, 0.0), Ecef(nan, 0.0, 0.0),
                                    r_eci, v_eci));
  EXPECT_DOUBLE_EQ(r_eci.x(), 1.0);
  EXPECT_DOUBLE_EQ(v_eci.x(), 4.0);
}

/// The two-part JD split is what keeps the reduction sub-nanosecond at JD ≈ 2.46e6
/// (REQ-CONV-005); a single-double JD would quantize the epoch to ~50 µs, i.e.
/// ~2 cm of ground track. Guard the property directly.
TEST(EciEcef, TwoPartJulianDateKeepsNanosecondResolution) {
  RecordProperty("verifies", "REQ-CONV-005");
  const pt::Tai t = TaiAtUtc(2020, 6, 1, 12345.0);
  const pt::Tai t_plus_ns = t + pt::Duration::fromNanoseconds(1);
  const pt::JulianDate a = pt::julianDate(t);
  const pt::JulianDate b = pt::julianDate(t_plus_ns);

  EXPECT_DOUBLE_EQ(a.day, b.day);     // day number is integral and stable
  EXPECT_NE(a.fraction, b.fraction);  // ...and the ns lands in the fraction
  // The fraction is a fraction-of-day double, so near 0.5 a 1 ns step (≈1.16e-14
  // day) carries a ~9 ps representation floor. 1e-11 s still proves ns resolution
  // is *retained* — a single-double JD would quantize the epoch to ~50 µs here.
  EXPECT_NEAR((b.fraction - a.fraction) * 86400.0, 1e-9, 1e-11);
  EXPECT_GE(a.fraction, 0.0);
  EXPECT_LT(a.fraction, 1.0);
  EXPECT_DOUBLE_EQ(a.day, std::floor(a.day));
}
