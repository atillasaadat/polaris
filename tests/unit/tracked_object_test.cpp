/// @file Truth-side secondary-object propagation (`sim/world/tracked_object`).
///
/// Two things here are worth testing and one is not. The TLE path is a thin
/// adaptor over SGP4 and `eciFromTeme`, both verified elsewhere and verified
/// harder than this file could (Push 81's three-way golden test), so what is
/// checked is that the adaptor calls them and hands back what they said — not
/// that they are right. The state-vector path is real code: an incremental RK4
/// whose answer must not depend on the order it was asked, and that is where
/// the tests are.

#include "world/tracked_object.hpp"

#include <gtest/gtest.h>

#include <cmath>

#include "constants/constants.hpp"
#include "frames/teme_eci.hpp"
#include "gnc/sgp4.hpp"
#include "gnc/tle.hpp"
#include "world/gravity_field.hpp"

namespace polaris::sim::world {
namespace {

namespace pm = polaris::math;
namespace pt = polaris::time;
namespace c = polaris::constants;

constexpr std::int64_t kEpochTaiNs = 1767225637000000000LL;

pt::Tai at(double seconds_from_epoch) {
  return pt::Tai::fromNanosecondsSinceEpoch(kEpochTaiNs +
                                            static_cast<std::int64_t>(seconds_from_epoch * 1.0e9));
}

/// A J2-only field: degree 2, order 0. Deliberately not the scenario's degree 8
/// — this is the one truncation with a closed-form circular-equatorial
/// solution, which is what lets the RK4 be checked against something that is
/// not another integration.
SphericalHarmonicGravity j2OnlyField() {
  GravityCoeffs coeffs;
  coeffs.nmax = 2;
  // Triangular, as the field asserts: row n holds orders 0..n.
  for (int n = 0; n <= 2; ++n) {
    coeffs.C.emplace_back(static_cast<std::size_t>(n) + 1, 0.0);
    coeffs.S.emplace_back(static_cast<std::size_t>(n) + 1, 0.0);
  }
  coeffs.C[0][0] = 1.0;  // the leading GM/r term
  // Normalized Cbar_20 = -J2 / sqrt(5), the same convention earthZonal() uses.
  coeffs.C[2][0] = -c::gravity::kJ2 / std::sqrt(5.0);
  // No ECI->ECEF resolver is installed, and that is deliberate: without one the
  // field is evaluated on the ECI position, which puts the J2 symmetry axis on
  // ECI Z — exactly the assumption the closed form below is derived under. In
  // the sim the scenario installs the real resolver; here the point is to check
  // the integrator against algebra, and a tilted pole would compare it against
  // algebra for a different problem.
  return SphericalHarmonicGravity(std::move(coeffs), 2, 0, c::gravity::kGM,
                                  c::gravity::kReferenceRadius);
}

/// Mean motion of a circular equatorial orbit under two-body + J2 [rad/s].
/// Closed form: with z = 0 the J2 term is purely radial, so the orbit stays
/// circular and equatorial and only the rate changes.
double circularEquatorialRate(double r) {
  return std::sqrt(c::gravity::kGM / (r * r * r) +
                   1.5 * c::gravity::kJ2 * c::gravity::kGM * c::gravity::kReferenceRadius *
                       c::gravity::kReferenceRadius / std::pow(r, 5));
}

// ---------------------------------------------------------------------------

TEST(TrackedObject, RefusesWhatItCannotPropagateRatherThanAnswering) {
  const auto field = j2OnlyField();
  // No gravity model: a two-body fallback would be a silent downgrade of the
  // truth side, so the object is invalid instead.
  EXPECT_FALSE(TrackedObject::fromState("t", at(0.0), Eigen::Vector3d(7.0e6, 0, 0),
                                        Eigen::Vector3d(0, 7500, 0), nullptr)
                   .valid());
  EXPECT_FALSE(TrackedObject::fromState("t", at(0.0), Eigen::Vector3d::Zero(),
                                        Eigen::Vector3d(0, 7500, 0), &field)
                   .valid());
  EXPECT_FALSE(
      TrackedObject::fromTle("t", "1 garbage", "2 garbage", pt::LeapSecondTable::historical())
          .valid());

  // And an invalid object answers nothing, rather than answering the origin.
  pm::Vec3<pm::frames::ECI> r;
  EXPECT_FALSE(TrackedObject().positionAt(at(100.0), r));
}

TEST(TrackedObject, MatchesTheClosedFormCircularEquatorialSolution) {
  const auto field = j2OnlyField();
  const double radius = 8.0e6;
  const double w = circularEquatorialRate(radius);
  const auto obj = TrackedObject::fromState("t", at(0.0), Eigen::Vector3d(radius, 0.0, 0.0),
                                            Eigen::Vector3d(0.0, w * radius, 0.0), &field);
  ASSERT_TRUE(obj.valid());

  // The RK4 is checked against algebra, not against another integration. A
  // wrong J2 sign or magnitude changes the rate and separates the two.
  for (const double t : {60.0, 600.0, 3600.0}) {
    pm::Vec3<pm::frames::ECI> got;
    ASSERT_TRUE(obj.positionAt(at(t), got));
    const double theta = w * t;
    const Eigen::Vector3d want(radius * std::cos(theta), radius * std::sin(theta), 0.0);
    EXPECT_LT((got.eigen() - want).norm(), 1.0)
        << "at t=" << t << " s the integration is " << (got.eigen() - want).norm()
        << " m from the closed form";
  }
}

TEST(TrackedObject, TheAnswerDoesNotDependOnTheOrderItWasAsked) {
  const auto field = j2OnlyField();
  const auto make = [&field] {
    return TrackedObject::fromState("t", at(0.0), Eigen::Vector3d(7.2e6, 0.0, 1.0e5),
                                    Eigen::Vector3d(0.0, 7000.0, 500.0), &field);
  };

  // Forward-only, which is what a sim run does. The sample times are
  // deliberately **not** multiples of the 10 s sub-step: an earlier version of
  // this test walked 100 s multiples, which land exactly on the grid and so
  // agreed with a direct jump no matter how the cursor was implemented. It
  // passed while the cursor stopped wherever it was asked, which made the
  // retained state a function of the sampling rate — 0.78 mm between a 10 Hz
  // and a 1 Hz reader of the same run. Aligned samples cannot see that.
  const auto sequential = make();
  pm::Vec3<pm::frames::ECI> stepped;
  for (const double t : {7.0, 33.0, 100.0, 617.25, 1000.0, 1103.5, 1200.0}) {
    ASSERT_TRUE(sequential.positionAt(at(t), stepped));
  }

  // Straight to the end, from a fresh object.
  const auto direct = make();
  pm::Vec3<pm::frames::ECI> jumped;
  ASSERT_TRUE(direct.positionAt(at(1200.0), jumped));

  // Bit-identical, not merely close: retained state only ever sits on a grid
  // anchored at the seed epoch, so the steps taken to reach a given time are the
  // same whatever was asked before. If this stops being identical the truth side
  // has become sampling-dependent, and a run is no longer bit-reproducible from
  // {config, seed} — the rule sim/CLAUDE.md states.
  EXPECT_EQ(stepped.eigen().x(), jumped.eigen().x());
  EXPECT_EQ(stepped.eigen().y(), jumped.eigen().y());
  EXPECT_EQ(stepped.eigen().z(), jumped.eigen().z());

  // And a backwards request re-seeds rather than integrating with a negative
  // step or returning the stale cursor.
  pm::Vec3<pm::frames::ECI> back;
  ASSERT_TRUE(sequential.positionAt(at(300.0), back));
  pm::Vec3<pm::frames::ECI> reference;
  ASSERT_TRUE(make().positionAt(at(300.0), reference));
  EXPECT_LT((back.eigen() - reference.eigen()).norm(), 1.0e-6);
}

TEST(TrackedObject, TheTleBranchHandsBackWhatSgp4AndTemeToEciSaid) {
  // The AIAA-style synthetic set the SITL guidance row flies, epoched at the
  // fixture epoch so the propagation span is minutes rather than decades.
  const char* l1 = "1 99001U 26001A   26001.00000000  .00000000  00000-0  00000-0 0  9997";
  const char* l2 = "2 99001  51.6000  30.0000 0001000  90.0000 180.0000 11.00000000    07";
  const auto leap = pt::LeapSecondTable::historical();
  const auto obj = TrackedObject::fromTle("t", l1, l2, leap);
  ASSERT_TRUE(obj.valid());
  EXPECT_EQ(obj.kind(), TrackedObject::Kind::kTle);

  gnc::TleElements elements;
  ASSERT_EQ(gnc::parseTle(l1, l2, elements), gnc::TleStatus::kOk);
  pt::Tai tle_epoch;
  ASSERT_TRUE(elements.epochTai(leap, tle_epoch));
  gnc::Sgp4 sgp4;
  ASSERT_EQ(sgp4.initialise(elements), gnc::Sgp4Status::kOk);

  for (const double t : {0.0, 300.0, 900.0}) {
    const pt::Tai when = at(t);
    const double minutes =
        static_cast<double>(when.nanosecondsSinceEpoch() - tle_epoch.nanosecondsSinceEpoch()) *
        1.0e-9 / 60.0;
    gnc::Sgp4::PositionKm p_teme;
    gnc::Sgp4::VelocityKmS v_teme;
    ASSERT_EQ(sgp4.propagate(minutes, p_teme, v_teme), gnc::Sgp4Status::kOk);
    pm::Vec3<pm::frames::ECI> want;
    ASSERT_TRUE(
        frames::eciFromTeme(when, pm::Vec3<pm::frames::TEME>(p_teme.eigen() * 1000.0), want));

    pm::Vec3<pm::frames::ECI> got;
    ASSERT_TRUE(obj.positionAt(when, got));
    // Exact: this path does no arithmetic of its own beyond the km->m scale, so
    // any difference would be the adaptor inventing something.
    EXPECT_DOUBLE_EQ(got.eigen().x(), want.eigen().x());
    EXPECT_DOUBLE_EQ(got.eigen().y(), want.eigen().y());
    EXPECT_DOUBLE_EQ(got.eigen().z(), want.eigen().z());
  }

  // The conversion out of TEME is not decoration: skipping it would leave the
  // target roughly an arcsecond-scale rotation away, which at these ranges is
  // hundreds of metres. Assert the two frames actually differ, so a future
  // "simplification" that drops eciFromTeme fails here.
  gnc::Sgp4::PositionKm p_teme;
  gnc::Sgp4::VelocityKmS v_teme;
  ASSERT_EQ(sgp4.propagate(0.0, p_teme, v_teme), gnc::Sgp4Status::kOk);
  pm::Vec3<pm::frames::ECI> got;
  ASSERT_TRUE(obj.positionAt(at(0.0), got));
  EXPECT_GT((got.eigen() - p_teme.eigen() * 1000.0).norm(), 100.0);
}

}  // namespace
}  // namespace polaris::sim::world
