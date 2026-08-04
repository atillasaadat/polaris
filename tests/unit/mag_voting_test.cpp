/// @file
/// @brief Fault-tolerant multi-magnetometer voting (design doc §8.2, §9.2;
/// REQ-ADET-011).
///
/// The shared redundancy policy is exercised by `tests/unit/imu_voting_test.cpp`
/// against the same core (`gnc::unit_voting`), so what is pinned here is what is
/// **magnetometer-specific** — the IGRF-magnitude plausibility band and the
/// attitude-quality gate on the identification reference — plus one pass of the
/// policy through this wrapper, so a wiring error in the wrapper cannot hide
/// behind the other suite's coverage.
///
/// The cases deliberately mirror the IMU suite's names and shapes, including the
/// absurd-reference, tie, flap and re-admission ones, because the two votes have
/// to behave the same way and a divergence is easiest to see side by side.

#include "gnc/mag_voting.hpp"

#include <gtest/gtest.h>

#include <cmath>

namespace {

namespace gnc = polaris::gnc;
namespace pm = polaris::math;
using Body = pm::frames::Body;

/// A LEO field: ~30 µT, the magnitude the band is taken against.
constexpr double kFieldT = 30.0e-6;

gnc::MagVoteConfig defaultConfig() {
  gnc::MagVoteConfig cfg{};
  cfg.min_field_ratio = 0.5;
  cfg.max_field_ratio = 1.6;
  cfg.disagreement_tesla = 5.0e-6;
  cfg.max_attitude_sigma_rad = 0.02;
  cfg.readmit_cycles = 3;
  cfg.identify_confirm_cycles = 2;
  return cfg;
}

pm::Vec3<Body> field(double x, double y, double z) {
  return pm::Vec3<Body>(Eigen::Vector3d(x, y, z));
}

/// The truth field this suite votes on: 30 µT along +X.
pm::Vec3<Body> nominal() {
  return field(kFieldT, 0.0, 0.0);
}

gnc::MagVoteReference goodReference() {
  gnc::MagVoteReference r{};
  r.field_tesla = nominal();
  r.attitude_valid = true;
  r.attitude_sigma_rad = 1.0e-3;
  return r;
}

}  // namespace

TEST(MagVoting, UnconfiguredVoterRefusesEveryCycle) {
  gnc::MagVoter inert{gnc::MagVoteConfig{}};
  EXPECT_FALSE(inert.isConfigured());
  gnc::MagVoteInput units[2];
  units[0].field_tesla = nominal();
  units[0].present = true;
  gnc::MagVoteResult out;
  const gnc::MagVoteReference reference = goodReference();
  EXPECT_FALSE(inert.vote(units, 2, kFieldT, &reference, out));
  EXPECT_FALSE(out.valid);
  EXPECT_EQ(out.published_index, -1);

  // The band must straddle the modelled magnitude: one that excluded it would
  // reject every healthy unit on every cycle, which is a configuration error
  // rather than a tight gate.
  gnc::MagVoteConfig bad = defaultConfig();
  bad.min_field_ratio = 1.1;
  EXPECT_FALSE(bad.isValid());
  bad = defaultConfig();
  bad.max_field_ratio = 0.9;
  EXPECT_FALSE(bad.isValid());
  bad = defaultConfig();
  bad.max_attitude_sigma_rad = 0.0;
  EXPECT_FALSE(bad.isValid());
}

TEST(MagVoting, RefusesACycleWithNoModelledField) {
  // Not "runs with the gate disabled". Without a modelled magnitude the
  // plausibility band has nothing to compare against, and a vote whose only
  // surviving gate is finiteness would admit the railed reading the band exists to
  // catch — straight into the pair, where a mean would then split the difference.
  gnc::MagVoter voter(defaultConfig());
  gnc::MagVoteInput units[2];
  for (int i = 0; i < 2; ++i) {
    units[i].field_tesla = nominal();
    units[i].present = true;
  }
  gnc::MagVoteResult out;
  const gnc::MagVoteReference reference = goodReference();
  EXPECT_FALSE(voter.vote(units, 2, 0.0, &reference, out));
  EXPECT_FALSE(voter.vote(units, 2, std::numeric_limits<double>::quiet_NaN(), &reference, out));
  EXPECT_FALSE(voter.vote(units, 2, -kFieldT, &reference, out));
  // A refusal is not a fault of the units: nothing is latched, so a mis-called
  // vote cannot manufacture an FDIR event.
  EXPECT_EQ(out.exclusion_mask, 0u);
  EXPECT_FALSE(voter.isExcluded(0));
}

TEST(MagVoting, AgreeingPairAveragesAndNamesThePublishedUnit) {
  gnc::MagVoter voter(defaultConfig());
  gnc::MagVoteInput units[2];
  units[0].field_tesla = nominal();
  units[0].present = true;
  units[1].field_tesla = field(kFieldT + 1.0e-6, 0.0, 0.0);  // inside the gate
  units[1].present = true;

  gnc::MagVoteResult out;
  const gnc::MagVoteReference reference = goodReference();
  ASSERT_TRUE(voter.vote(units, 2, kFieldT, &reference, out));
  EXPECT_EQ(out.status, gnc::MagVoteStatus::kPair);
  EXPECT_EQ(out.contributing, 2);
  EXPECT_NEAR(out.field_tesla.eigen().x(), kFieldT + 0.5e-6, 1.0e-15);
  // The published index is what a downstream per-unit calibration follows, so it
  // has to name a unit that actually contributed — and 255/-1 when none did.
  EXPECT_EQ(out.published_index, 0);
  // Slots past `count` read absent rather than the zero-initialised
  // "contributing", so a caller scanning the whole array cannot mistake an
  // unpopulated slot for a healthy unit.
  EXPECT_EQ(out.reason[5], gnc::MagVoteReason::kAbsent);
}

TEST(MagVoting, FieldMagnitudeBandExcludesADeadOrSaturatedUnit) {
  gnc::MagVoter voter(defaultConfig());
  gnc::MagVoteInput units[2];
  units[0].field_tesla = nominal();
  units[0].present = true;
  units[1].field_tesla = field(0.1 * kFieldT, 0.0, 0.0);  // a dead/unpowered sensor
  units[1].present = true;

  gnc::MagVoteResult out;
  const gnc::MagVoteReference reference = goodReference();
  ASSERT_TRUE(voter.vote(units, 2, kFieldT, &reference, out));
  EXPECT_EQ(out.status, gnc::MagVoteStatus::kSingle);
  EXPECT_EQ(out.reason[1], gnc::MagVoteReason::kOutOfRange);
  EXPECT_TRUE(out.newly_excluded[1]);
  EXPECT_EQ(out.exclusion_mask, 0x2u);
  // The survivor is passed through unchanged: one unit is the vehicle's only
  // knowledge of the field, and refusing it would cost the magnetic pair on no
  // evidence of a fault in *that* unit.
  EXPECT_TRUE(out.field_tesla.eigen() == nominal().eigen());
  EXPECT_EQ(out.published_index, 0);

  // The band tracks the model rather than a fixed full scale, which is the whole
  // reason it beats a saturation check: the same reading is *plausible* where the
  // field really is that weak.
  gnc::MagVoter fresh(defaultConfig());
  gnc::MagVoteResult low;
  ASSERT_TRUE(fresh.vote(units, 2, 0.1 * kFieldT, &reference, low));
  EXPECT_EQ(low.reason[0], gnc::MagVoteReason::kOutOfRange) << "the 30 uT unit is now the outlier";
  EXPECT_EQ(low.reason[1], gnc::MagVoteReason::kContributing);

  // Saturation, at the other end.
  gnc::MagVoter high_voter(defaultConfig());
  units[1].field_tesla = field(3.0 * kFieldT, 0.0, 0.0);
  gnc::MagVoteResult high;
  ASSERT_TRUE(high_voter.vote(units, 2, kFieldT, &reference, high));
  EXPECT_EQ(high.reason[1], gnc::MagVoteReason::kOutOfRange);
}

TEST(MagVoting, NonFiniteReadingIsReportedAsSuchNotAsOutOfRange) {
  // The gate name is what the EVR carries, and "the data path is broken" and "the
  // field magnitude is wrong" call for different ground actions.
  gnc::MagVoter voter(defaultConfig());
  gnc::MagVoteInput units[2];
  units[0].field_tesla = nominal();
  units[0].present = true;
  units[1].field_tesla = field(std::numeric_limits<double>::quiet_NaN(), 0.0, 0.0);
  units[1].present = true;

  gnc::MagVoteResult out;
  const gnc::MagVoteReference reference = goodReference();
  ASSERT_TRUE(voter.vote(units, 2, kFieldT, &reference, out));
  EXPECT_EQ(out.reason[1], gnc::MagVoteReason::kNotFinite);
  EXPECT_TRUE(out.field_tesla.isFinite());
}

TEST(MagVoting, DisagreementIsIdentifiedByTheModelledField) {
  gnc::MagVoter voter(defaultConfig());
  gnc::MagVoteInput units[2];
  units[0].field_tesla = nominal();  // agrees with the model
  units[0].present = true;
  // A 10 µT offset: twice the disagreement gate, well inside the magnitude band,
  // so **no per-unit gate can see it** and only the comparison can. That is the
  // fault a two-unit suite exists to handle.
  units[1].field_tesla = field(kFieldT, 10.0e-6, 0.0);
  units[1].present = true;

  const gnc::MagVoteReference reference = goodReference();
  gnc::MagVoteResult out;
  // Confirmation takes `identify_confirm_cycles`; the winner is published
  // throughout, so the cost is detection latency and not the magnetic pair.
  ASSERT_TRUE(voter.vote(units, 2, kFieldT, &reference, out));
  EXPECT_EQ(out.status, gnc::MagVoteStatus::kIdentified);
  EXPECT_EQ(out.reason[1], gnc::MagVoteReason::kOutvoted);
  EXPECT_FALSE(out.newly_excluded[1]) << "one sample must not buy a permanent latch";
  EXPECT_TRUE(out.field_tesla.eigen() == nominal().eigen());

  ASSERT_TRUE(voter.vote(units, 2, kFieldT, &reference, out));
  EXPECT_TRUE(out.newly_excluded[1]);
  EXPECT_TRUE(voter.isExcluded(1));
  EXPECT_EQ(out.exclusion_mask, 0x2u);
}

TEST(MagVoting, AnAbsurdReferenceIsTreatedAsNoReference) {
  // The guard that stops an upstream fault latching out the *healthy* unit. The
  // reference comes from the estimator's own published attitude, so a bad attitude
  // makes both residuals large and would award the identification to whichever
  // unit happened to sit nearer to nonsense.
  gnc::MagVoter voter(defaultConfig());
  gnc::MagVoteInput units[2];
  units[0].field_tesla = nominal();
  units[0].present = true;
  units[1].field_tesla = field(kFieldT, 10.0e-6, 0.0);
  units[1].present = true;

  gnc::MagVoteResult out;

  // (a) No attitude at all.
  gnc::MagVoteReference invalid = goodReference();
  invalid.attitude_valid = false;
  EXPECT_FALSE(voter.vote(units, 2, kFieldT, &invalid, out));
  EXPECT_EQ(out.status, gnc::MagVoteStatus::kAmbiguous);
  EXPECT_EQ(out.contributing, 0);

  // (b) An attitude that is *valid* but not good enough — the quality gate, and
  // the case a validity flag alone cannot distinguish. At 10° a 30 µT field is
  // mispredicted by ~5 µT, which is the disagreement gate itself.
  gnc::MagVoteReference coarse = goodReference();
  coarse.attitude_sigma_rad = 0.18;
  EXPECT_FALSE(voter.vote(units, 2, kFieldT, &coarse, out));
  EXPECT_EQ(out.status, gnc::MagVoteStatus::kAmbiguous);

  // (c) A non-finite reference.
  gnc::MagVoteReference broken = goodReference();
  broken.field_tesla = field(std::numeric_limits<double>::quiet_NaN(), 0.0, 0.0);
  EXPECT_FALSE(voter.vote(units, 2, kFieldT, &broken, out));
  EXPECT_EQ(out.status, gnc::MagVoteStatus::kAmbiguous);

  // (d) Nothing at all.
  EXPECT_FALSE(voter.vote(units, 2, kFieldT, nullptr, out));
  EXPECT_EQ(out.status, gnc::MagVoteStatus::kAmbiguous);

  // Through all four, nothing is latched: detection is not attribution, and
  // excluding a unit on a guess would spend the remaining redundancy.
  EXPECT_EQ(out.exclusion_mask, 0u);
  EXPECT_FALSE(voter.isExcluded(0));
  EXPECT_FALSE(voter.isExcluded(1));
}

TEST(MagVoting, AnIndecisiveComparisonIsAmbiguous) {
  // Both units wrong by similar amounts — a common-mode drift, which is exactly
  // where an ordering-only rule is a coin flip. The margin gate requires the
  // reference to agree with one reading *and* disagree with the other, which
  // disposes of the exact tie for free.
  gnc::MagVoter voter(defaultConfig());
  gnc::MagVoteInput units[2];
  units[0].field_tesla = field(kFieldT, 8.0e-6, 0.0);
  units[0].present = true;
  units[1].field_tesla = field(kFieldT, -8.0e-6, 0.0);
  units[1].present = true;

  const gnc::MagVoteReference reference = goodReference();
  gnc::MagVoteResult out;
  EXPECT_FALSE(voter.vote(units, 2, kFieldT, &reference, out));
  EXPECT_EQ(out.status, gnc::MagVoteStatus::kAmbiguous);
  EXPECT_EQ(out.exclusion_mask, 0u);

  // The exact tie, which the same gate has to handle: two units symmetric about
  // the reference, both outside it.
  gnc::MagVoter tie_voter(defaultConfig());
  units[0].field_tesla = field(kFieldT, 8.0e-6, 0.0);
  units[1].field_tesla = field(kFieldT, 0.0, 8.0e-6);
  gnc::MagVoteResult tie;
  EXPECT_FALSE(tie_voter.vote(units, 2, kFieldT, &reference, tie));
  EXPECT_EQ(tie.status, gnc::MagVoteStatus::kAmbiguous);
}

TEST(MagVoting, AVerdictThatFlipsNeverConfirms) {
  // A disagreement the reference genuinely cannot resolve must not latch either
  // unit, however long it lasts: a flipping verdict resets both counts.
  gnc::MagVoter voter(defaultConfig());
  const gnc::MagVoteReference reference = goodReference();
  gnc::MagVoteResult out;
  for (int cycle = 0; cycle < 20; ++cycle) {
    gnc::MagVoteInput units[2];
    const int loser = cycle % 2;
    for (int i = 0; i < 2; ++i) {
      units[i].present = true;
      units[i].field_tesla = (i == loser) ? field(kFieldT, 10.0e-6, 0.0) : nominal();
    }
    ASSERT_TRUE(voter.vote(units, 2, kFieldT, &reference, out));
    EXPECT_EQ(out.status, gnc::MagVoteStatus::kIdentified);
    ASSERT_EQ(out.exclusion_mask, 0u) << "a flipping verdict latched a unit at cycle " << cycle;
  }
}

TEST(MagVoting, ExcludedUnitIsReadmittedOnTheCriterionThatExcludedIt) {
  gnc::MagVoter voter(defaultConfig());
  const gnc::MagVoteReference reference = goodReference();
  gnc::MagVoteResult out;

  // Out on the magnitude band.
  gnc::MagVoteInput units[2];
  units[0].field_tesla = nominal();
  units[0].present = true;
  units[1].field_tesla = field(0.05 * kFieldT, 0.0, 0.0);
  units[1].present = true;
  ASSERT_TRUE(voter.vote(units, 2, kFieldT, &reference, out));
  ASSERT_TRUE(voter.isExcluded(1));

  // Back in range: three consecutive plausible cycles earn it back, and not two.
  units[1].field_tesla = nominal();
  for (int cycle = 0; cycle < 2; ++cycle) {
    ASSERT_TRUE(voter.vote(units, 2, kFieldT, &reference, out));
    EXPECT_TRUE(voter.isExcluded(1)) << "re-admitted after only " << cycle + 1 << " cycles";
    EXPECT_EQ(out.reason[1], gnc::MagVoteReason::kExcluded);
  }
  ASSERT_TRUE(voter.vote(units, 2, kFieldT, &reference, out));
  EXPECT_FALSE(voter.isExcluded(1));
  EXPECT_TRUE(out.newly_readmitted[1]);
}

TEST(MagVoting, AnOutvotedUnitDoesNotFlap) {
  // The one place the re-admission policy could quietly invert itself. An
  // identified unit was *plausible* by construction — it passed every per-unit
  // gate and lost a comparison — so counting plausibility towards its
  // re-admission would return it unconditionally and it would be outvoted again
  // the next cycle: a permanent exclude/re-admit flap with an FDIR event per lap.
  gnc::MagVoter voter(defaultConfig());
  const gnc::MagVoteReference reference = goodReference();
  gnc::MagVoteInput units[2];
  units[0].field_tesla = nominal();
  units[0].present = true;
  units[1].field_tesla = field(kFieldT, 10.0e-6, 0.0);
  units[1].present = true;

  gnc::MagVoteResult out;
  int events = 0;
  for (int cycle = 0; cycle < 40; ++cycle) {
    ASSERT_TRUE(voter.vote(units, 2, kFieldT, &reference, out));
    events += (out.newly_excluded[1] ? 1 : 0) + (out.newly_readmitted[1] ? 1 : 0);
  }
  EXPECT_EQ(events, 1) << "the persistently-faulted unit produced " << events
                       << " transition events over 40 cycles; it must produce exactly one";
  EXPECT_TRUE(voter.isExcluded(1));

  // It comes back only by *agreeing with the combination*, which is the criterion
  // that put it out.
  units[1].field_tesla = nominal();
  for (int cycle = 0; cycle < 3; ++cycle) {
    ASSERT_TRUE(voter.vote(units, 2, kFieldT, &reference, out));
  }
  EXPECT_FALSE(voter.isExcluded(1));
}

TEST(MagVoting, AbsenceIsNotImplausibility) {
  // A dropout says nothing about whether the unit is lying, so it neither latches
  // an exclusion nor lets an excluded unit serve out its sentence by going quiet.
  gnc::MagVoter voter(defaultConfig());
  const gnc::MagVoteReference reference = goodReference();
  gnc::MagVoteInput units[2];
  units[0].field_tesla = nominal();
  units[0].present = true;
  units[1].present = false;  // stale / dropped out

  gnc::MagVoteResult out;
  ASSERT_TRUE(voter.vote(units, 2, kFieldT, &reference, out));
  EXPECT_EQ(out.status, gnc::MagVoteStatus::kSingle);
  EXPECT_EQ(out.reason[1], gnc::MagVoteReason::kAbsent);
  EXPECT_FALSE(out.newly_excluded[1]);
  EXPECT_EQ(out.exclusion_mask, 0u);

  // An excluded unit going quiet does not accrue re-admission credit.
  units[1].field_tesla = field(0.05 * kFieldT, 0.0, 0.0);
  units[1].present = true;
  ASSERT_TRUE(voter.vote(units, 2, kFieldT, &reference, out));
  ASSERT_TRUE(voter.isExcluded(1));
  units[1].present = false;
  for (int cycle = 0; cycle < 10; ++cycle) {
    ASSERT_TRUE(voter.vote(units, 2, kFieldT, &reference, out));
  }
  EXPECT_TRUE(voter.isExcluded(1)) << "an absent unit served out its exclusion by going quiet";
}

TEST(MagVoting, ThreeOrMoreUnitsTakeTheMedianAndNeedNoReference) {
  // The rung a heavier vehicle inherits already tested. The median tolerates one
  // arbitrary fault by construction and needs no external reference at all, which
  // is exactly what the two-unit vehicle cannot have.
  gnc::MagVoter voter(defaultConfig());
  gnc::MagVoteInput units[3];
  units[0].field_tesla = field(kFieldT, 1.0e-6, 0.0);
  units[1].field_tesla = field(kFieldT, -1.0e-6, 0.0);
  units[2].field_tesla = field(kFieldT, 20.0e-6, 0.0);  // out of family but in band
  for (int i = 0; i < 3; ++i) {
    units[i].present = true;
  }

  gnc::MagVoteResult out;
  ASSERT_TRUE(voter.vote(units, 3, kFieldT, nullptr, out));
  EXPECT_EQ(out.status, gnc::MagVoteStatus::kMedian);
  EXPECT_EQ(out.contributing, 3);
  // The median of {1, −1, 20} µT on the y axis is 1 µT: the faulted unit moved the
  // answer by nothing, where a mean would have moved it by 6.7 µT.
  EXPECT_NEAR(out.field_tesla.eigen().y(), 1.0e-6, 1.0e-15);
}

TEST(MagVoting, ClearExclusionsIsTheCommandedRecoveryPath) {
  gnc::MagVoter voter(defaultConfig());
  const gnc::MagVoteReference reference = goodReference();
  gnc::MagVoteInput units[2];
  units[0].field_tesla = nominal();
  units[0].present = true;
  units[1].field_tesla = field(0.05 * kFieldT, 0.0, 0.0);
  units[1].present = true;
  gnc::MagVoteResult out;
  ASSERT_TRUE(voter.vote(units, 2, kFieldT, &reference, out));
  ASSERT_TRUE(voter.isExcluded(1));

  voter.clearExclusions();
  EXPECT_FALSE(voter.isExcluded(1));
  // A unit that is really failed re-excludes on its next reading, which costs one
  // EVR and tells the ground the fault is persistent rather than latched.
  ASSERT_TRUE(voter.vote(units, 2, kFieldT, &reference, out));
  EXPECT_TRUE(out.newly_excluded[1]);
}
