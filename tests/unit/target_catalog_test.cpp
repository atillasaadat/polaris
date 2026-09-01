/// @file Unit tests for the onboard target catalogue (§8.3; REQ-ODP-002).
///
/// Five TLE slots and five state-vector slots behind one query. The tests that
/// matter are about the *seam*: that a caller gets ECI without knowing which
/// propagator answered, that a rejected upload cannot damage a working slot, and
/// that a failed propagation is reported rather than papered over with the last
/// good position — which would be indistinguishable from tracking.

#include "gnc/target_catalog.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <string>

#include "constants/constants.hpp"
#include "time/leap_seconds.hpp"

namespace pc = polaris::constants;
namespace pg = polaris::gnc;
namespace pm = polaris::math;
namespace pmf = polaris::math::frames;
namespace pt = polaris::time;

namespace {

/// Satellite 5 from the committed AIAA verification set — the same element set
/// the SGP4 golden tests and the three-way cross-validation use, so a
/// disagreement here is about the catalogue and not about SGP4.
constexpr const char* kLine1 =
    "1 00005U 58002B   00179.78495062  .00000023  00000-0  28098-4 0  4753";
constexpr const char* kLine2 =
    "2 00005  34.2682 348.7242 1859667 331.7664  19.3264 10.82419157413667";

pt::LeapSecondTable leap() {
  return pt::LeapSecondTable{};
}

/// The TLE's own epoch, recovered through the parser so the test does not carry
/// a second copy of the epoch arithmetic.
pt::Tai tleEpoch() {
  pg::TleElements e;
  EXPECT_EQ(pg::parseTle(kLine1, kLine2, e, pg::TleChecksumPolicy::kIgnore), pg::TleStatus::kOk);
  pt::Tai t;
  EXPECT_TRUE(e.epochTai(leap(), t));
  return t;
}

pg::StateVectorSlot circularSlot() {
  const double r = pc::wgs84::kSemiMajorAxis + 600.0e3;
  const double v = std::sqrt(pc::gravity::kGM / r);
  pg::StateVectorSlot s;
  s.epoch = tleEpoch();
  s.position_m = pm::Vec3<pmf::ECI>(r, 0.0, 0.0);
  s.velocity_m_s = pm::Vec3<pmf::ECI>(0.0, v, 0.0);
  s.sigma_at_epoch_m = 10.0;
  return s;
}

}  // namespace

TEST(TargetCatalog, StartsEmpty) {
  const pg::TargetCatalog cat;
  EXPECT_EQ(cat.occupiedCount(pg::TargetKind::kTle), 0);
  EXPECT_EQ(cat.occupiedCount(pg::TargetKind::kStateVector), 0);
  pg::TargetState out;
  EXPECT_EQ(cat.positionAt(pg::TargetKind::kTle, 0, tleEpoch(), out), pg::TargetStatus::kEmpty);
}

TEST(TargetCatalog, HoldsFiveOfEachKindIndependently) {
  pg::TargetCatalog cat;
  for (int i = 0; i < pg::TargetCatalog::kMaxSlots; ++i) {
    ASSERT_EQ(cat.loadTle(i, kLine1, kLine2, leap(), pg::TleChecksumPolicy::kIgnore),
              pg::TargetStatus::kOk);
    ASSERT_EQ(cat.loadStateVector(i, circularSlot()), pg::TargetStatus::kOk);
  }
  EXPECT_EQ(cat.occupiedCount(pg::TargetKind::kTle), 5);
  EXPECT_EQ(cat.occupiedCount(pg::TargetKind::kStateVector), 5);

  // The two kinds are separate address spaces: clearing one must not touch the
  // other's slot of the same index.
  ASSERT_EQ(cat.clear(pg::TargetKind::kTle, 2), pg::TargetStatus::kOk);
  EXPECT_FALSE(cat.isOccupied(pg::TargetKind::kTle, 2));
  EXPECT_TRUE(cat.isOccupied(pg::TargetKind::kStateVector, 2));
}

TEST(TargetCatalog, RefusesOutOfRangeSlots) {
  pg::TargetCatalog cat;
  EXPECT_EQ(cat.loadTle(-1, kLine1, kLine2, leap()), pg::TargetStatus::kBadSlot);
  EXPECT_EQ(cat.loadTle(pg::TargetCatalog::kMaxSlots, kLine1, kLine2, leap()),
            pg::TargetStatus::kBadSlot);
  EXPECT_EQ(cat.loadStateVector(5, circularSlot()), pg::TargetStatus::kBadSlot);
  pg::TargetState out;
  EXPECT_EQ(cat.positionAt(pg::TargetKind::kTle, 99, tleEpoch(), out), pg::TargetStatus::kBadSlot);
}

TEST(TargetCatalog, ARejectedUploadLeavesThePreviousTargetInService) {
  // The all-or-nothing rule, and the reason for it: a half-loaded target is
  // worse than a stale one, because a stale one is at least somewhere real.
  pg::TargetCatalog cat;
  ASSERT_EQ(cat.loadTle(0, kLine1, kLine2, leap(), pg::TleChecksumPolicy::kIgnore),
            pg::TargetStatus::kOk);
  pg::TargetState before;
  ASSERT_EQ(cat.positionAt(pg::TargetKind::kTle, 0, tleEpoch(), before), pg::TargetStatus::kOk);

  EXPECT_EQ(cat.loadTle(0, "1 garbage", "2 garbage", leap()), pg::TargetStatus::kPropagationFailed);
  EXPECT_TRUE(cat.isOccupied(pg::TargetKind::kTle, 0));

  pg::TargetState after;
  ASSERT_EQ(cat.positionAt(pg::TargetKind::kTle, 0, tleEpoch(), after), pg::TargetStatus::kOk);
  EXPECT_EQ(after.position_m.eigen(), before.position_m.eigen())
      << "a rejected upload moved the target that was already loaded";
}

TEST(TargetCatalog, ARejectedStateVectorAlsoLeavesTheSlotAlone) {
  pg::TargetCatalog cat;
  ASSERT_EQ(cat.loadStateVector(1, circularSlot()), pg::TargetStatus::kOk);
  pg::StateVectorSlot bad = circularSlot();
  bad.position_m = pm::Vec3<pmf::ECI>(1000.0e3, 0.0, 0.0);  // inside the Earth
  EXPECT_EQ(cat.loadStateVector(1, bad), pg::TargetStatus::kPropagationFailed);
  EXPECT_TRUE(cat.isOccupied(pg::TargetKind::kStateVector, 1));

  pg::TargetState out;
  ASSERT_EQ(cat.positionAt(pg::TargetKind::kStateVector, 1, tleEpoch(), out),
            pg::TargetStatus::kOk);
  EXPECT_NEAR(out.position_m.norm(), circularSlot().position_m.norm(), 1e-6);
}

TEST(TargetCatalog, VerifiesTleChecksumsByDefaultSoAFlippedCharacterIsCaught) {
  // The operational concern this default exists for: an uplinked TLE with one
  // corrupted character is not garbage, it is a *plausible orbit somewhere
  // else*, and a catalogue that accepted it would point the instrument
  // confidently at nothing.
  //
  // Satellite 5's own checksums are valid (only 5 of the AIAA fixture's 66 lines
  // carry stale ones), so the corruption is introduced here: one digit of the
  // inclination changed, leaving the line well-formed and the checksum stale.
  std::string corrupted = kLine2;
  corrupted[9] = corrupted[9] == '4' ? '5' : '4';  // 34.2682 -> 44.2682 inclination

  pg::TargetCatalog cat;
  EXPECT_EQ(cat.loadTle(0, kLine1, corrupted, leap()), pg::TargetStatus::kPropagationFailed);
  EXPECT_FALSE(cat.isOccupied(pg::TargetKind::kTle, 0))
      << "a corrupted element set was accepted into the catalogue";

  // The same line is accepted when the operator explicitly waives the check —
  // the policy exists for hand-written element sets, and waiving it must remain
  // possible or the AIAA fixture's own stale lines would be unusable.
  EXPECT_EQ(cat.loadTle(0, kLine1, corrupted, leap(), pg::TleChecksumPolicy::kIgnore),
            pg::TargetStatus::kOk);

  // And an intact element set passes the default, so the check is not simply
  // refusing everything.
  pg::TargetCatalog clean;
  EXPECT_EQ(clean.loadTle(0, kLine1, kLine2, leap()), pg::TargetStatus::kOk);
}

TEST(TargetCatalog, ATleTargetComesBackInEciNotTeme) {
  // The seam this class exists to close. SGP4 answers in TEME; a consumer must
  // never see that. Checked by magnitude of the difference: TEME and ECI differ
  // by ~0.8 km at LEO near J2000, which is far above numerical noise and far
  // below anything else that could move the answer.
  pg::TargetCatalog cat;
  ASSERT_EQ(cat.loadTle(0, kLine1, kLine2, leap(), pg::TleChecksumPolicy::kIgnore),
            pg::TargetStatus::kOk);
  pg::TargetState out;
  ASSERT_EQ(cat.positionAt(pg::TargetKind::kTle, 0, tleEpoch(), out), pg::TargetStatus::kOk);

  // The raw TEME state at epoch, from the same verification fixture.
  const Eigen::Vector3d teme_m(7022.46529266e3, -1400.08296755e3, 0.03995155e3);
  const double moved = (out.position_m.eigen() - teme_m).norm();
  EXPECT_GT(moved, 500.0) << "the catalogue returned TEME: the frame conversion was skipped";
  EXPECT_LT(moved, 2000.0) << "the answer is not the converted TEME state either";
}

TEST(TargetCatalog, AgeIsSignedAndSigmaGrowsWithIt) {
  pg::TargetCatalog cat;
  ASSERT_EQ(cat.loadTle(0, kLine1, kLine2, leap(), pg::TleChecksumPolicy::kIgnore),
            pg::TargetStatus::kOk);
  const pt::Tai epoch = tleEpoch();

  pg::TargetState at_epoch;
  pg::TargetState later;
  pg::TargetState earlier;
  ASSERT_EQ(cat.positionAt(pg::TargetKind::kTle, 0, epoch, at_epoch), pg::TargetStatus::kOk);
  ASSERT_EQ(cat.positionAt(pg::TargetKind::kTle, 0, epoch + pt::Duration::fromSeconds(3600), later),
            pg::TargetStatus::kOk);
  ASSERT_EQ(
      cat.positionAt(pg::TargetKind::kTle, 0, epoch - pt::Duration::fromSeconds(3600), earlier),
      pg::TargetStatus::kOk);

  EXPECT_NEAR(at_epoch.age_s, 0.0, 1e-6);
  EXPECT_NEAR(later.age_s, 3600.0, 1e-6);
  EXPECT_NEAR(earlier.age_s, -3600.0, 1e-6) << "age must be signed; a look-ahead upload is routine";
  EXPECT_GT(later.sigma_m, at_epoch.sigma_m);
  EXPECT_NEAR(later.sigma_m, earlier.sigma_m, 1e-9) << "uncertainty is symmetric in |age|";
}

TEST(TargetCatalog, ATleTargetIsPublishedAsFarLessCertainThanAStateVector) {
  // The number an operator needs to compare the two kinds. A TLE is a kilometre
  // at epoch; a ground OD solution can be metres. If the catalogue published one
  // figure for both, choosing between them would be guesswork.
  pg::TargetCatalog cat;
  ASSERT_EQ(cat.loadTle(0, kLine1, kLine2, leap(), pg::TleChecksumPolicy::kIgnore),
            pg::TargetStatus::kOk);
  ASSERT_EQ(cat.loadStateVector(0, circularSlot()), pg::TargetStatus::kOk);

  pg::TargetState tle_state;
  pg::TargetState sv_state;
  ASSERT_EQ(cat.positionAt(pg::TargetKind::kTle, 0, tleEpoch(), tle_state), pg::TargetStatus::kOk);
  ASSERT_EQ(cat.positionAt(pg::TargetKind::kStateVector, 0, tleEpoch(), sv_state),
            pg::TargetStatus::kOk);
  EXPECT_GT(tle_state.sigma_m, 10.0 * sv_state.sigma_m);
  EXPECT_EQ(tle_state.kind, pg::TargetKind::kTle);
  EXPECT_EQ(sv_state.kind, pg::TargetKind::kStateVector);
}

TEST(TargetCatalog, AFailedPropagationIsReportedRatherThanStale) {
  // A state-vector slot asked beyond its propagator's span. Reported, not
  // answered with the last good position — the failure this whole class is
  // arranged to avoid, because a frozen position looks exactly like tracking.
  pg::TargetCatalog cat;
  ASSERT_EQ(cat.loadStateVector(0, circularSlot()), pg::TargetStatus::kOk);
  pg::TargetState out;
  EXPECT_EQ(cat.positionAt(pg::TargetKind::kStateVector, 0,
                           tleEpoch() + pt::Duration::fromSeconds(200000), out),
            pg::TargetStatus::kPropagationFailed);
}

TEST(TargetCatalog, ClearingIsIdempotentAndEmptiesTheSlot) {
  pg::TargetCatalog cat;
  ASSERT_EQ(cat.loadStateVector(3, circularSlot()), pg::TargetStatus::kOk);
  EXPECT_EQ(cat.clear(pg::TargetKind::kStateVector, 3), pg::TargetStatus::kOk);
  // Idempotent so a recovery sequence need not first ask what is there.
  EXPECT_EQ(cat.clear(pg::TargetKind::kStateVector, 3), pg::TargetStatus::kOk);
  EXPECT_FALSE(cat.isOccupied(pg::TargetKind::kStateVector, 3));
  pg::TargetState out;
  EXPECT_EQ(cat.positionAt(pg::TargetKind::kStateVector, 3, tleEpoch(), out),
            pg::TargetStatus::kEmpty);
}

TEST(TargetCatalog, QueriesAreOrderIndependentAcrossSlots) {
  // Both propagators are order-independent individually; this asserts the
  // catalogue does not reintroduce coupling between slots by sharing state.
  pg::TargetCatalog cat;
  ASSERT_EQ(cat.loadTle(0, kLine1, kLine2, leap(), pg::TleChecksumPolicy::kIgnore),
            pg::TargetStatus::kOk);
  ASSERT_EQ(cat.loadStateVector(0, circularSlot()), pg::TargetStatus::kOk);
  const pt::Tai t = tleEpoch() + pt::Duration::fromSeconds(1200);

  pg::TargetState tle_first;
  pg::TargetState sv_first;
  ASSERT_EQ(cat.positionAt(pg::TargetKind::kTle, 0, t, tle_first), pg::TargetStatus::kOk);
  ASSERT_EQ(cat.positionAt(pg::TargetKind::kStateVector, 0, t, sv_first), pg::TargetStatus::kOk);

  pg::TargetState sv_second;
  pg::TargetState tle_second;
  ASSERT_EQ(cat.positionAt(pg::TargetKind::kStateVector, 0, t, sv_second), pg::TargetStatus::kOk);
  ASSERT_EQ(cat.positionAt(pg::TargetKind::kTle, 0, t, tle_second), pg::TargetStatus::kOk);

  EXPECT_EQ(tle_first.position_m.eigen(), tle_second.position_m.eigen());
  EXPECT_EQ(sv_first.position_m.eigen(), sv_second.position_m.eigen());
}
