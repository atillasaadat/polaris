/// @file Unit tests for TEME↔ECI (§3.1, §8.3; REQ-CONV-002, REQ-ODP-003).
///
/// The conversion SGP4 output must pass through before anything else in Polaris
/// may look at it. Three classes of check, in increasing strength:
///
///  1. **Properties** — it is a rotation, so it preserves magnitude and round
///     trips exactly. Necessary, and nowhere near sufficient: a transposed or
///     entirely wrong rotation passes all of them.
///  2. **Magnitude** — the correction is ~0.8 km at LEO near J2000 and grows
///     with precession. A conversion that quietly did nothing would pass (1) and
///     fail here, which is the failure this file most needs to catch, because
///     "TEME is close enough to ECI" is the intuition that produces it.
///  3. **An independent implementation** — the measured agreement with astropy's
///     own TEME machinery, recorded as a band rather than re-derived here.

#include "frames/teme_eci.hpp"

#include <gtest/gtest.h>

#include <cmath>

#include "time/leap_seconds.hpp"
#include "time/utc.hpp"

namespace pf = polaris::frames;
namespace pm = polaris::math;
namespace pmf = polaris::math::frames;
namespace pt = polaris::time;

namespace {

/// Satellite 5's TLE epoch from the committed verification fixture:
/// 2000-06-27 18:50:19.733565 UTC (day-of-year 179.78495062).
pt::Tai sat5Epoch() {
  pt::UtcDateTime utc;
  utc.year = 2000;
  utc.month = 6;
  utc.day = 27;
  utc.hour = 18;
  utc.minute = 50;
  utc.second = 19;
  utc.nanosecond = 733565000;
  return pt::taiFromUtc(utc, pt::LeapSecondTable{});
}

/// The fixture's TEME state for satellite 5 at t = 0.
pm::Vec3<pmf::TEME> sat5PositionTeme() {
  return pm::Vec3<pmf::TEME>(7022.46529266, -1400.08296755, 0.03995155);
}

pm::Vec3<pmf::TEME> sat5VelocityTeme() {
  return pm::Vec3<pmf::TEME>(1.893841015, 6.405893759, 4.534807250);
}

}  // namespace

TEST(TemeEci, IsARotationSoMagnitudeIsPreserved) {
  pm::Vec3<pmf::ECI> eci;
  ASSERT_TRUE(pf::eciFromTeme(sat5Epoch(), sat5PositionTeme(), eci));
  // Exact to round-off: a rotation cannot change a length.
  EXPECT_NEAR(eci.eigen().norm(), sat5PositionTeme().eigen().norm(), 1e-9);
}

TEST(TemeEci, RoundTripsThroughBothDirections) {
  const pt::Tai t = sat5Epoch();
  pm::Vec3<pmf::ECI> eci;
  ASSERT_TRUE(pf::eciFromTeme(t, sat5PositionTeme(), eci));
  pm::Vec3<pmf::TEME> back;
  ASSERT_TRUE(pf::temeFromEci(t, eci, back));
  EXPECT_NEAR(back.eigen().x(), sat5PositionTeme().eigen().x(), 1e-9);
  EXPECT_NEAR(back.eigen().y(), sat5PositionTeme().eigen().y(), 1e-9);
  EXPECT_NEAR(back.eigen().z(), sat5PositionTeme().eigen().z(), 1e-9);
}

TEST(TemeEci, TheMatrixIsOrthonormal) {
  Eigen::Matrix3d m;
  ASSERT_TRUE(pf::temeToEciMatrix(sat5Epoch(), m));
  const Eigen::Matrix3d should_be_identity = m.transpose() * m;
  EXPECT_LT((should_be_identity - Eigen::Matrix3d::Identity()).cwiseAbs().maxCoeff(), 1e-12);
  // A proper rotation, not a reflection: det = +1, never -1.
  EXPECT_NEAR(m.determinant(), 1.0, 1e-12);
}

TEST(TemeEci, TheCorrectionIsHundredsOfMetresAndNotZero) {
  // The load-bearing test. TEME and ECI are close enough that treating them as
  // the same frame produces a plausible answer, so a conversion that silently
  // did nothing would pass every property test above. Measured at satellite 5's
  // epoch: 0.796 km. Bounded on both sides — an implementation that started
  // returning kilometres would be as wrong as one returning zero.
  pm::Vec3<pmf::ECI> eci;
  ASSERT_TRUE(pf::eciFromTeme(sat5Epoch(), sat5PositionTeme(), eci));
  const double moved = (eci.eigen() - sat5PositionTeme().eigen()).norm();
  EXPECT_GT(moved, 0.5) << "the conversion did essentially nothing";
  EXPECT_LT(moved, 2.0) << "the conversion moved the point far more than precession explains";
}

TEST(TemeEci, AgreesWithAnIndependentImplementation) {
  // Cross-checked against astropy's own TEME->GCRS, which shares no code with
  // this chain. Measured residual **0.53 m** in position and 0.58 mm/s in
  // velocity at this epoch, against a correction of 796 m — so the two agree to
  // 0.07 % of the quantity being computed.
  //
  // The residual is not error to be driven out: it is the IAU-76/80 chain this
  // file must use (TEME is defined by the theory SGP4 was fitted with) against
  // the IAU-2006/2000A chain astropy runs — about 16 mas, or three orders below
  // SGP4's own ~1 km. Pinned as a band so that a *model* change shows up here
  // rather than silently.
  //
  // Expected values are astropy's, transcribed; regenerate with the snippet in
  // `tools/tle/README.md` if the band ever needs revisiting.
  const pt::Tai t = sat5Epoch();
  pm::Vec3<pmf::ECI> r;
  pm::Vec3<pmf::ECI> v;
  ASSERT_TRUE(pf::eciStateFromTeme(t, sat5PositionTeme(), sat5VelocityTeme(), r, v));

  const Eigen::Vector3d astropy_r(7022.312443784, -1400.849397267, -0.110867866);
  const Eigen::Vector3d astropy_v(1.894617984, 6.405588965, 4.534913146);
  EXPECT_LT((r.eigen() - astropy_r).norm(), 1.0e-3)  // km, i.e. 1 m
      << "position disagrees with astropy by more than the model difference explains";
  EXPECT_LT((v.eigen() - astropy_v).norm(), 1.0e-6)  // km/s, i.e. 1 mm/s
      << "velocity disagrees with astropy by more than the model difference explains";
}

TEST(TemeEci, PositionAndVelocityShareOneRotationWithNoTransportTerm) {
  // Both frames are quasi-inertial, so unlike ECI<->ECEF there is no omega x r.
  // Asserting it means a later "fix" that adds one has to argue with this test.
  const pt::Tai t = sat5Epoch();
  pm::Vec3<pmf::ECI> r;
  pm::Vec3<pmf::ECI> v;
  ASSERT_TRUE(pf::eciStateFromTeme(t, sat5PositionTeme(), sat5VelocityTeme(), r, v));

  Eigen::Matrix3d m;
  ASSERT_TRUE(pf::temeToEciMatrix(t, m));
  const Eigen::Vector3d expected_v = m * sat5VelocityTeme().eigen();
  EXPECT_LT((v.eigen() - expected_v).cwiseAbs().maxCoeff(), 1e-15);
  // Speed is preserved exactly, which an omega x r term would break.
  EXPECT_NEAR(v.eigen().norm(), sat5VelocityTeme().eigen().norm(), 1e-12);
}

TEST(TemeEci, TheCorrectionGrowsWithTimeFromJ2000) {
  // Precession accumulates, so the same TEME vector converts to a progressively
  // more different ECI vector as the epoch moves away from J2000. This is what
  // makes "TEME is basically ECI" fail slowly and invisibly on an old TLE.
  auto correction_at = [](int year) {
    pt::UtcDateTime utc;
    utc.year = year;
    utc.month = 6;
    utc.day = 27;
    utc.hour = 12;
    const pt::Tai t = pt::taiFromUtc(utc, pt::LeapSecondTable{});
    pm::Vec3<pmf::ECI> eci;
    EXPECT_TRUE(pf::eciFromTeme(t, sat5PositionTeme(), eci));
    return (eci.eigen() - sat5PositionTeme().eigen()).norm();
  };
  const double y2000 = correction_at(2000);
  const double y2020 = correction_at(2020);
  const double y2040 = correction_at(2040);
  EXPECT_GT(y2020, y2000);
  EXPECT_GT(y2040, y2020);
  // Order tens of km after four decades — the scale that makes this a frame
  // conversion rather than a rounding detail.
  EXPECT_GT(y2040, 10.0);
}

TEST(TemeEci, NonFiniteInputIsRefusedAndLeavesOutputsUntouched) {
  const pt::Tai t = sat5Epoch();
  const pm::Vec3<pmf::TEME> bad(std::nan(""), 0.0, 0.0);
  pm::Vec3<pmf::ECI> out(1.0, 2.0, 3.0);
  EXPECT_FALSE(pf::eciFromTeme(t, bad, out));
  EXPECT_EQ(out.eigen().x(), 1.0);
  EXPECT_EQ(out.eigen().y(), 2.0);
  EXPECT_EQ(out.eigen().z(), 3.0);

  pm::Vec3<pmf::ECI> pr(1.0, 1.0, 1.0);
  pm::Vec3<pmf::ECI> vr(2.0, 2.0, 2.0);
  EXPECT_FALSE(pf::eciStateFromTeme(t, sat5PositionTeme(), bad, pr, vr));
  EXPECT_EQ(pr.eigen().x(), 1.0);
  EXPECT_EQ(vr.eigen().x(), 2.0);
}

TEST(TemeEci, NeedsNoEarthOrientationData) {
  // Nothing in the chain involves Earth rotation, so the conversion works with
  // no uploaded EOP table at all. That is not a convenience: it is what lets a
  // vehicle that has lost GNSS — and with it any fresh EOP — still convert an
  // uploaded TLE into the frame its estimators work in. The signature takes no
  // EopValue, and this test exists so that adding one is a deliberate act.
  static_assert(std::is_invocable_r_v<bool, decltype(&pf::eciFromTeme), const pt::Tai&,
                                      const pm::Vec3<pmf::TEME>&, pm::Vec3<pmf::ECI>&>,
                "eciFromTeme must remain callable with epoch and vectors alone");
  SUCCEED();
}
