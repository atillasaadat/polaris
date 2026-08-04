/// @file Fault-tolerant multi-IMU voting (design doc §8.2, §9.2;
/// REQ-ADET-008, REQ-ADET-009).
///
/// What these cases pin is the property the module exists for: a single unit
/// reporting an arbitrary value must not move the combined rate by an arbitrary
/// amount. Each fault mode the design doc names — railed high, non-finite,
/// stale/absent — is exercised against a three-unit set, and the surviving
/// combination is compared against the *same* set with the fault removed.
/// The comparison is deliberately against a fault-free reference rather than
/// against a fixed tolerance: it is the statement "the fault cost nothing",
/// which is what single-fault survival means.

#include "gnc/imu_voting.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <limits>

namespace {

namespace pm = polaris::math;
namespace gnc = polaris::gnc;
using Body = pm::frames::Body;

/// Reference-vehicle tuning, mirroring `flight.attitudeEstimator.imu*` in
/// config/spacecraft/leo_smallsat.yaml. Kept literal here rather than derived,
/// so a change to the flight tuning shows up as a test that has to be re-read.
gnc::ImuVoteConfig refConfig() {
  gnc::ImuVoteConfig c;
  c.max_rate_radps = 0.5236;      // 30 °/s vehicle rate limit
  c.disagreement_radps = 0.0087;  // 0.5 °/s pairwise gate
  c.readmit_cycles = 10;
  c.identify_confirm_cycles = 5;
  return c;
}

/// Drive @p voter through @p cycles identical cycles of a disagreeing pair, so a
/// test can walk past the identification-confirmation window without repeating
/// the loop. Returns the last result.
gnc::ImuVoteResult confirmIdentification(gnc::ImuVoter& voter, const gnc::ImuVoteInput* units,
                                         const pm::Vec3<Body>& reference, unsigned cycles) {
  gnc::ImuVoteResult out;
  for (unsigned i = 0; i < cycles; ++i) {
    voter.vote(units, 2, &reference, out);
  }
  return out;
}

gnc::ImuVoteInput present(const Eigen::Vector3d& rate) {
  gnc::ImuVoteInput u;
  u.rate = pm::Vec3<Body>(rate);
  u.present = true;
  return u;
}

/// A healthy three-unit set around @p truth with small, distinct per-unit
/// offsets — enough that the median is a real selection rather than three
/// identical numbers.
void healthyTriad(const Eigen::Vector3d& truth, gnc::ImuVoteInput* units) {
  units[0] = present(truth + Eigen::Vector3d(1e-4, -2e-4, 3e-4));
  units[1] = present(truth + Eigen::Vector3d(-3e-4, 1e-4, -1e-4));
  units[2] = present(truth + Eigen::Vector3d(2e-4, 3e-4, 2e-4));
}

const Eigen::Vector3d kTruth(0.004, -0.002, 0.003);

}  // namespace

// ── Configuration gating ────────────────────────────────────────────────────

TEST(ImuVoting, UnconfiguredVoterRefusesEveryCycle) {
  gnc::ImuVoter voter;  // default-constructed: no tuning
  EXPECT_FALSE(voter.isConfigured());

  gnc::ImuVoteInput units[3];
  healthyTriad(kTruth, units);
  gnc::ImuVoteResult out;
  EXPECT_FALSE(voter.vote(units, 3, nullptr, out));
  EXPECT_FALSE(out.valid);
  // A refusal must not manufacture FDIR evidence about units it never looked at.
  EXPECT_EQ(0u, out.exclusion_mask);
}

TEST(ImuVoting, ConfigRangeGatesRejectEachMissingValue) {
  gnc::ImuVoteConfig c = refConfig();
  EXPECT_TRUE(c.isValid());

  c = refConfig();
  c.max_rate_radps = 0.0;
  EXPECT_FALSE(c.isValid());

  c = refConfig();
  c.disagreement_radps = -1.0;
  EXPECT_FALSE(c.isValid());

  c = refConfig();
  c.readmit_cycles = 0;  // would make the exclusion latch a no-op
  EXPECT_FALSE(c.isValid());

  c = refConfig();
  c.identify_confirm_cycles = 0;  // one sample would buy a permanent latch
  EXPECT_FALSE(c.isValid());

  c = refConfig();
  c.max_rate_radps = std::numeric_limits<double>::quiet_NaN();
  EXPECT_FALSE(c.isValid());
}

// ── The median itself ───────────────────────────────────────────────────────

TEST(ImuVoting, MedianIsPerAxisAndOrderIndependent) {
  pm::Vec3<Body> rates[3] = {pm::Vec3<Body>(1.0, 9.0, -5.0), pm::Vec3<Body>(2.0, 8.0, -1.0),
                             pm::Vec3<Body>(3.0, 7.0, -3.0)};
  pm::Vec3<Body> out;
  ASSERT_TRUE(gnc::medianRate(rates, 3, out));
  // Each axis is chosen independently, so the answer is not any one input row.
  EXPECT_DOUBLE_EQ(2.0, out.x());
  EXPECT_DOUBLE_EQ(8.0, out.y());
  EXPECT_DOUBLE_EQ(-3.0, out.z());

  pm::Vec3<Body> shuffled[3] = {rates[2], rates[0], rates[1]};
  pm::Vec3<Body> out2;
  ASSERT_TRUE(gnc::medianRate(shuffled, 3, out2));
  EXPECT_EQ(out.eigen(), out2.eigen());
}

TEST(ImuVoting, EvenCountAveragesTheTwoCentralOrderStatistics) {
  pm::Vec3<Body> rates[4] = {pm::Vec3<Body>(1.0, 0.0, 0.0), pm::Vec3<Body>(2.0, 0.0, 0.0),
                             pm::Vec3<Body>(4.0, 0.0, 0.0), pm::Vec3<Body>(100.0, 0.0, 0.0)};
  pm::Vec3<Body> out;
  ASSERT_TRUE(gnc::medianRate(rates, 4, out));
  EXPECT_DOUBLE_EQ(3.0, out.x());  // (2 + 4)/2, not (1+2+4+100)/4 = 26.75
}

TEST(ImuVoting, MedianRefusesAnOutOfRangeCount) {
  pm::Vec3<Body> rates[1] = {pm::Vec3<Body>(1.0, 2.0, 3.0)};
  pm::Vec3<Body> out(9.0, 9.0, 9.0);
  EXPECT_FALSE(gnc::medianRate(rates, 0, out));
  EXPECT_FALSE(gnc::medianRate(rates, gnc::kMaxImuUnits + 1, out));
  EXPECT_FALSE(gnc::medianRate(nullptr, 1, out));
  EXPECT_EQ(Eigen::Vector3d(9.0, 9.0, 9.0), out.eigen());  // untouched on refusal
}

// ── Single-fault survival at three units (REQ-ADET-008) ─────────────────────

TEST(ImuVoting, HealthyTriadTakesTheMedian) {
  gnc::ImuVoter voter(refConfig());
  gnc::ImuVoteInput units[3];
  healthyTriad(kTruth, units);

  gnc::ImuVoteResult out;
  ASSERT_TRUE(voter.vote(units, 3, nullptr, out));
  EXPECT_EQ(gnc::ImuVoteStatus::kMedian, out.status);
  EXPECT_EQ(3, out.contributing);
  EXPECT_EQ(0u, out.exclusion_mask);
  EXPECT_LT((out.rate.eigen() - kTruth).norm(), 1e-3);
}

/// The headline case. A unit railed at its measurement full scale is the fault
/// an average cannot survive: at 100 °/s against a 0.3 °/s truth the mean would
/// land two orders of magnitude out. The median puts the output bit-identical to
/// the two-healthy-unit answer.
TEST(ImuVoting, RailedUnitIsGatedAndCostsNothing) {
  gnc::ImuVoter voter(refConfig());
  gnc::ImuVoteInput units[3];
  healthyTriad(kTruth, units);
  const Eigen::Vector3d healthy0 = units[0].rate.eigen();
  const Eigen::Vector3d healthy1 = units[1].rate.eigen();

  units[2] = present(Eigen::Vector3d(1.745, 0.0, 0.0));  // 100 °/s: past the 30 °/s limit

  gnc::ImuVoteResult out;
  ASSERT_TRUE(voter.vote(units, 3, nullptr, out));
  EXPECT_EQ(gnc::ImuVoteStatus::kPair, out.status);
  EXPECT_EQ(2, out.contributing);
  EXPECT_EQ(gnc::ImuVoteReason::kOutOfRange, out.reason[2]);
  EXPECT_TRUE(out.newly_excluded[2]);
  EXPECT_EQ(1u << 2, out.exclusion_mask);
  EXPECT_TRUE(voter.isExcluded(2));

  // "Cost nothing": the surviving pair's answer, exactly.
  EXPECT_EQ(Eigen::Vector3d(0.5 * (healthy0 + healthy1)), out.rate.eigen());
  // And still within a whisker of truth, which the mean of all three would not be.
  EXPECT_LT((out.rate.eigen() - kTruth).norm(), 1e-3);
}

TEST(ImuVoting, NonFiniteUnitIsGatedBeforeItPoisonsTheCombination) {
  gnc::ImuVoter voter(refConfig());
  gnc::ImuVoteInput units[3];
  healthyTriad(kTruth, units);
  units[1] = present(Eigen::Vector3d(std::numeric_limits<double>::quiet_NaN(), 0.0, 0.0));

  gnc::ImuVoteResult out;
  ASSERT_TRUE(voter.vote(units, 3, nullptr, out));
  EXPECT_EQ(gnc::ImuVoteReason::kNotFinite, out.reason[1]);
  EXPECT_TRUE(out.newly_excluded[1]);
  EXPECT_TRUE(out.rate.isFinite());
  EXPECT_LT((out.rate.eigen() - kTruth).norm(), 1e-3);
}

/// A stale or dropped-out unit is *absent*, not implausible, and the distinction
/// is load-bearing: a dropout says nothing about whether the unit is lying, so
/// it must not latch an exclusion the ground then has to reason about.
TEST(ImuVoting, StaleUnitIsAbsentRatherThanExcluded) {
  gnc::ImuVoter voter(refConfig());
  gnc::ImuVoteInput units[3];
  healthyTriad(kTruth, units);
  units[0].present = false;  // the caller's staleness gate closed

  gnc::ImuVoteResult out;
  ASSERT_TRUE(voter.vote(units, 3, nullptr, out));
  EXPECT_EQ(gnc::ImuVoteReason::kAbsent, out.reason[0]);
  EXPECT_FALSE(out.newly_excluded[0]);
  EXPECT_EQ(0u, out.exclusion_mask);
  EXPECT_EQ(gnc::ImuVoteStatus::kPair, out.status);
}

/// An excluded unit cannot serve out its sentence by going quiet: absence
/// neither advances nor resets the re-admission streak.
TEST(ImuVoting, AbsenceDoesNotAdvanceReadmission) {
  gnc::ImuVoteConfig cfg = refConfig();
  cfg.readmit_cycles = 3;
  gnc::ImuVoter voter(cfg);
  gnc::ImuVoteInput units[3];
  healthyTriad(kTruth, units);

  units[2] = present(Eigen::Vector3d(1.745, 0.0, 0.0));
  gnc::ImuVoteResult out;
  ASSERT_TRUE(voter.vote(units, 3, nullptr, out));
  ASSERT_TRUE(voter.isExcluded(2));

  healthyTriad(kTruth, units);
  units[2].present = false;
  for (int i = 0; i < 10; ++i) {
    ASSERT_TRUE(voter.vote(units, 3, nullptr, out));
  }
  EXPECT_TRUE(voter.isExcluded(2));
}

// ── Exclusion latch and re-admission policy (REQ-ADET-009) ──────────────────

TEST(ImuVoting, ReadmissionNeedsConsecutivePlausibleCycles) {
  gnc::ImuVoteConfig cfg = refConfig();
  cfg.readmit_cycles = 3;
  gnc::ImuVoter voter(cfg);

  gnc::ImuVoteInput units[3];
  gnc::ImuVoteResult out;

  healthyTriad(kTruth, units);
  units[0] = present(Eigen::Vector3d(1.745, 0.0, 0.0));
  ASSERT_TRUE(voter.vote(units, 3, nullptr, out));
  ASSERT_TRUE(out.newly_excluded[0]);

  // Two plausible cycles are not enough.
  healthyTriad(kTruth, units);
  for (int i = 0; i < 2; ++i) {
    ASSERT_TRUE(voter.vote(units, 3, nullptr, out));
    EXPECT_EQ(gnc::ImuVoteReason::kExcluded, out.reason[0]);
    EXPECT_FALSE(out.newly_readmitted[0]);
    EXPECT_TRUE(voter.isExcluded(0));
    EXPECT_EQ(2, out.contributing);
  }

  // The third re-admits, once and on the edge.
  ASSERT_TRUE(voter.vote(units, 3, nullptr, out));
  EXPECT_TRUE(out.newly_readmitted[0]);
  EXPECT_EQ(gnc::ImuVoteReason::kContributing, out.reason[0]);
  EXPECT_FALSE(voter.isExcluded(0));
  EXPECT_EQ(3, out.contributing);
  EXPECT_EQ(0u, out.exclusion_mask);

  ASSERT_TRUE(voter.vote(units, 3, nullptr, out));
  EXPECT_FALSE(out.newly_readmitted[0]);
}

/// The hysteresis the policy exists for: a unit that misbehaves once inside its
/// probation restarts the count instead of flapping back in.
TEST(ImuVoting, OneRelapseRestartsTheReadmissionCount) {
  gnc::ImuVoteConfig cfg = refConfig();
  cfg.readmit_cycles = 3;
  gnc::ImuVoter voter(cfg);

  gnc::ImuVoteInput units[3];
  gnc::ImuVoteResult out;
  healthyTriad(kTruth, units);
  units[0] = present(Eigen::Vector3d(1.745, 0.0, 0.0));
  ASSERT_TRUE(voter.vote(units, 3, nullptr, out));

  healthyTriad(kTruth, units);
  ASSERT_TRUE(voter.vote(units, 3, nullptr, out));  // streak 1
  ASSERT_TRUE(voter.vote(units, 3, nullptr, out));  // streak 2

  units[0] = present(Eigen::Vector3d(1.745, 0.0, 0.0));
  ASSERT_TRUE(voter.vote(units, 3, nullptr, out));
  EXPECT_FALSE(out.newly_excluded[0]);  // already excluded: no repeated FDIR event

  healthyTriad(kTruth, units);
  ASSERT_TRUE(voter.vote(units, 3, nullptr, out));  // streak 1 again
  ASSERT_TRUE(voter.vote(units, 3, nullptr, out));  // streak 2
  EXPECT_TRUE(voter.isExcluded(0));
  ASSERT_TRUE(voter.vote(units, 3, nullptr, out));  // streak 3
  EXPECT_TRUE(out.newly_readmitted[0]);
}

TEST(ImuVoting, ClearExclusionsIsTheCommandedReadmission) {
  gnc::ImuVoter voter(refConfig());
  gnc::ImuVoteInput units[3];
  gnc::ImuVoteResult out;
  healthyTriad(kTruth, units);
  units[1] = present(Eigen::Vector3d(1.745, 0.0, 0.0));
  ASSERT_TRUE(voter.vote(units, 3, nullptr, out));
  ASSERT_TRUE(voter.isExcluded(1));

  voter.clearExclusions();
  EXPECT_FALSE(voter.isExcluded(1));

  healthyTriad(kTruth, units);
  ASSERT_TRUE(voter.vote(units, 3, nullptr, out));
  EXPECT_EQ(3, out.contributing);
  // Re-admitted by command, so no automatic-recovery edge is reported.
  EXPECT_FALSE(out.newly_readmitted[1]);
}

// ── Two units: detection without attribution ────────────────────────────────

TEST(ImuVoting, AgreeingPairIsAveraged) {
  gnc::ImuVoter voter(refConfig());
  gnc::ImuVoteInput units[2] = {present(kTruth + Eigen::Vector3d(1e-4, 0.0, 0.0)),
                                present(kTruth - Eigen::Vector3d(1e-4, 0.0, 0.0))};
  gnc::ImuVoteResult out;
  ASSERT_TRUE(voter.vote(units, 2, nullptr, out));
  EXPECT_EQ(gnc::ImuVoteStatus::kPair, out.status);
  EXPECT_LT((out.rate.eigen() - kTruth).norm(), 1e-9);
}

/// Both readings are individually plausible — neither trips the rate limit — so
/// only the *pair* reveals the fault. With nothing to attribute it to, the
/// answer is no rate: the deliberate non-monotonicity against the single-unit
/// case, and the reason it is deliberate is that two disagreeing units are
/// positive evidence one is lying while one unit is no evidence at all.
TEST(ImuVoting, DisagreeingPairWithNoReferenceYieldsNoRate) {
  gnc::ImuVoter voter(refConfig());
  gnc::ImuVoteInput units[2] = {present(kTruth), present(kTruth + Eigen::Vector3d(0.05, 0.0, 0.0))};
  gnc::ImuVoteResult out;
  EXPECT_FALSE(voter.vote(units, 2, nullptr, out));
  EXPECT_EQ(gnc::ImuVoteStatus::kAmbiguous, out.status);
  EXPECT_FALSE(out.valid);
  EXPECT_EQ(0, out.contributing);
  // Detection is not attribution: nothing is latched, because latching the wrong
  // unit is worse than carrying an unattributed disagreement.
  EXPECT_EQ(0u, out.exclusion_mask);
}

TEST(ImuVoting, ReferenceRateIdentifiesTheDisagreeingUnit) {
  gnc::ImuVoter voter(refConfig());
  gnc::ImuVoteInput units[2] = {present(kTruth), present(kTruth + Eigen::Vector3d(0.05, 0.0, 0.0))};
  // The MEKF's propagated rate: close to truth, so unit 1 is the one its own
  // dynamics disbelieve.
  const pm::Vec3<Body> reference(kTruth + Eigen::Vector3d(1e-4, 0.0, 0.0));

  // Before confirmation: the winner is already published (so the estimator keeps
  // a rate, and keeps producing the reference this branch needs) but nothing is
  // latched yet and no FDIR edge has been claimed.
  gnc::ImuVoteResult out;
  ASSERT_TRUE(voter.vote(units, 2, &reference, out));
  EXPECT_EQ(gnc::ImuVoteStatus::kIdentified, out.status);
  EXPECT_EQ(1, out.contributing);
  EXPECT_EQ(kTruth, out.rate.eigen());
  EXPECT_FALSE(out.newly_excluded[1]);
  EXPECT_FALSE(voter.isExcluded(1));
  EXPECT_EQ(0u, out.exclusion_mask);

  // The confirming cycle latches, once.
  out = confirmIdentification(voter, units, reference, refConfig().identify_confirm_cycles - 1);
  EXPECT_EQ(gnc::ImuVoteReason::kOutvoted, out.reason[1]);
  EXPECT_TRUE(out.newly_excluded[1]);
  EXPECT_TRUE(voter.isExcluded(1));
  // Identification puts the loser under the same latch as a gate failure, so
  // FDIR has one path to reason about.
  EXPECT_EQ(1u << 1, out.exclusion_mask);

  // ...and only once: the edge is an edge, however long the fault persists.
  out = confirmIdentification(voter, units, reference, 20);
  EXPECT_FALSE(out.newly_excluded[1]);
}

/// C1. An outvoted unit is plausible **by construction** — it passed every
/// per-unit gate and lost a comparison — so counting plausibility towards its
/// re-admission would return it unconditionally, and it would lose the same
/// comparison on the next cycle. That is a permanent exclude/re-admit flap at
/// the re-admission period, with an FDIR event on every lap, and it is exactly
/// the contract ("one event on the transition") that it violates.
TEST(ImuVoting, OutvotedUnitDoesNotReadmitWhileItStillDisagrees) {
  gnc::ImuVoter voter(refConfig());
  gnc::ImuVoteInput units[2] = {present(kTruth), present(kTruth + Eigen::Vector3d(0.05, 0.0, 0.0))};
  const pm::Vec3<Body> reference(kTruth);

  int exclusions = 0;
  int readmissions = 0;
  gnc::ImuVoteResult out;
  for (int i = 0; i < 200; ++i) {  // 20 s at 10 Hz, well past readmit_cycles
    voter.vote(units, 2, &reference, out);
    exclusions += out.newly_excluded[1] ? 1 : 0;
    readmissions += out.newly_readmitted[1] ? 1 : 0;
  }
  EXPECT_EQ(1, exclusions) << "the sustained fault was re-reported: the unit flapped";
  EXPECT_EQ(0, readmissions) << "a unit still disagreeing re-admitted itself";
  EXPECT_TRUE(voter.isExcluded(1));
}

/// ...and the converse: once it *agrees* again it comes back, because agreement
/// is the criterion that excluded it.
TEST(ImuVoting, OutvotedUnitReadmitsOnceItAgreesAgain) {
  gnc::ImuVoteConfig cfg = refConfig();
  cfg.readmit_cycles = 3;
  gnc::ImuVoter voter(cfg);

  gnc::ImuVoteInput units[2] = {present(kTruth), present(kTruth + Eigen::Vector3d(0.05, 0.0, 0.0))};
  const pm::Vec3<Body> reference(kTruth);
  gnc::ImuVoteResult out =
      confirmIdentification(voter, units, reference, cfg.identify_confirm_cycles);
  ASSERT_TRUE(voter.isExcluded(1));

  units[1] = present(kTruth + Eigen::Vector3d(1e-4, 0.0, 0.0));  // healed
  for (unsigned i = 0; i < cfg.readmit_cycles - 1; ++i) {
    voter.vote(units, 2, &reference, out);
    EXPECT_TRUE(voter.isExcluded(1)) << "re-admitted before serving the window";
  }
  voter.vote(units, 2, &reference, out);
  EXPECT_TRUE(out.newly_readmitted[1]);
  EXPECT_FALSE(voter.isExcluded(1));
}

/// C2. The reference is the estimator's own published rate, so an upstream fault
/// can hand this function nonsense. Unchecked, an absurd reference makes *both*
/// residuals enormous and awards the identification to whichever unit is nearer
/// to nonsense — latching out the healthy one.
TEST(ImuVoting, AbsurdReferenceIsRefusedRatherThanBelieved) {
  gnc::ImuVoter voter(refConfig());
  gnc::ImuVoteInput units[2] = {present(kTruth), present(kTruth + Eigen::Vector3d(0.05, 0.0, 0.0))};
  // Finite, and 1e6 rad/s — six orders past anything the vehicle can do.
  const pm::Vec3<Body> absurd(1.0e6, 0.0, 0.0);

  gnc::ImuVoteResult out;
  for (int i = 0; i < 50; ++i) {
    EXPECT_FALSE(voter.vote(units, 2, &absurd, out));
    EXPECT_EQ(gnc::ImuVoteStatus::kAmbiguous, out.status);
  }
  EXPECT_EQ(0u, out.exclusion_mask) << "an absurd reference latched a unit out";
}

/// C3a. An exact tie carries no information about which unit is wrong, and the
/// margin form disposes of it without a special case: equal residuals can never
/// be both above and below the gate.
TEST(ImuVoting, ExactResidualTieIsAmbiguous) {
  gnc::ImuVoter voter(refConfig());
  // Symmetric about the reference: residuals identical to the last bit.
  const Eigen::Vector3d offset(0.05, 0.0, 0.0);
  gnc::ImuVoteInput units[2] = {present(kTruth + offset), present(kTruth - offset)};
  const pm::Vec3<Body> reference(kTruth);

  gnc::ImuVoteResult out;
  EXPECT_FALSE(voter.vote(units, 2, &reference, out));
  EXPECT_EQ(gnc::ImuVoteStatus::kAmbiguous, out.status);
  EXPECT_EQ(0u, out.exclusion_mask);
}

/// C3a. The margin is about more than ties: when the reference disagrees with
/// *both* units — a common-mode drift, or a stale reference — there is no winner
/// to speak of, and ordering alone would still name one.
TEST(ImuVoting, CommonModeDriftIsAmbiguousRatherThanIdentified) {
  gnc::ImuVoter voter(refConfig());
  // Both units well outside the gate from the reference, one slightly nearer.
  gnc::ImuVoteInput units[2] = {present(kTruth + Eigen::Vector3d(0.040, 0.0, 0.0)),
                                present(kTruth + Eigen::Vector3d(0.055, 0.0, 0.0))};
  const pm::Vec3<Body> reference(kTruth);

  gnc::ImuVoteResult out;
  for (int i = 0; i < 50; ++i) {
    EXPECT_FALSE(voter.vote(units, 2, &reference, out));
    EXPECT_EQ(gnc::ImuVoteStatus::kAmbiguous, out.status);
  }
  EXPECT_EQ(0u, out.exclusion_mask) << "a drift both units share latched one of them out";
}

/// C3b. A verdict that flips between units never confirms, so a noisy sample —
/// or a drift crossing the reference — cannot buy a permanent latch on its own.
TEST(ImuVoting, FlippingVerdictNeverConfirms) {
  gnc::ImuVoter voter(refConfig());
  const pm::Vec3<Body> reference(kTruth);
  const Eigen::Vector3d far(0.05, 0.0, 0.0);
  const Eigen::Vector3d close(1e-4, 0.0, 0.0);

  gnc::ImuVoteResult out;
  for (int i = 0; i < 200; ++i) {
    // Alternate which unit is the outlier: each verdict resets the other's count.
    gnc::ImuVoteInput units[2] = {present(kTruth + (i % 2 == 0 ? far : close)),
                                  present(kTruth + (i % 2 == 0 ? close : far))};
    voter.vote(units, 2, &reference, out);
    EXPECT_FALSE(out.newly_excluded[0]);
    EXPECT_FALSE(out.newly_excluded[1]);
  }
  EXPECT_EQ(0u, out.exclusion_mask) << "an alternating verdict latched a unit out";
}

TEST(ImuVoting, NonFiniteReferenceLeavesTheDisagreementAmbiguous) {
  gnc::ImuVoter voter(refConfig());
  gnc::ImuVoteInput units[2] = {present(kTruth), present(kTruth + Eigen::Vector3d(0.05, 0.0, 0.0))};
  const pm::Vec3<Body> reference(std::numeric_limits<double>::quiet_NaN(), 0.0, 0.0);
  gnc::ImuVoteResult out;
  EXPECT_FALSE(voter.vote(units, 2, &reference, out));
  EXPECT_EQ(gnc::ImuVoteStatus::kAmbiguous, out.status);
}

// ── Degenerate sets ─────────────────────────────────────────────────────────

TEST(ImuVoting, SingleUnitPassesThroughGated) {
  gnc::ImuVoter voter(refConfig());
  gnc::ImuVoteInput units[1] = {present(kTruth)};
  gnc::ImuVoteResult out;
  ASSERT_TRUE(voter.vote(units, 1, nullptr, out));
  EXPECT_EQ(gnc::ImuVoteStatus::kSingle, out.status);
  EXPECT_EQ(kTruth, out.rate.eigen());

  // ...but still gated: a railed single unit is refused, not published.
  units[0] = present(Eigen::Vector3d(1.745, 0.0, 0.0));
  EXPECT_FALSE(voter.vote(units, 1, nullptr, out));
  EXPECT_EQ(gnc::ImuVoteStatus::kNoValue, out.status);
  EXPECT_TRUE(out.newly_excluded[0]);
}

TEST(ImuVoting, EveryUnitFailingLeavesNoRate) {
  gnc::ImuVoter voter(refConfig());
  gnc::ImuVoteInput units[3];
  for (auto& unit : units) {
    unit = present(Eigen::Vector3d(1.745, 0.0, 0.0));
  }
  gnc::ImuVoteResult out;
  EXPECT_FALSE(voter.vote(units, 3, nullptr, out));
  EXPECT_EQ(gnc::ImuVoteStatus::kNoValue, out.status);
  EXPECT_EQ(0, out.contributing);
  EXPECT_EQ(0x7u, out.exclusion_mask);
}

TEST(ImuVoting, MalformedCallIsRefusedWithoutSideEffects) {
  gnc::ImuVoter voter(refConfig());
  gnc::ImuVoteInput units[3];
  healthyTriad(kTruth, units);
  gnc::ImuVoteResult out;

  EXPECT_FALSE(voter.vote(nullptr, 3, nullptr, out));
  EXPECT_FALSE(voter.vote(units, -1, nullptr, out));
  EXPECT_FALSE(voter.vote(units, gnc::kMaxImuUnits + 1, nullptr, out));
  for (int i = 0; i < gnc::kMaxImuUnits; ++i) {
    EXPECT_FALSE(voter.isExcluded(i));
  }
  EXPECT_FALSE(voter.isExcluded(-1));
  EXPECT_FALSE(voter.isExcluded(gnc::kMaxImuUnits));

  // Zero units is a well-formed call with nothing to combine — reported as
  // kNoValue rather than as a malformed one, so a vehicle whose whole IMU set has
  // dropped out is distinguishable from a caller bug.
  EXPECT_FALSE(voter.vote(units, 0, nullptr, out));
  EXPECT_EQ(gnc::ImuVoteStatus::kNoValue, out.status);
  EXPECT_FALSE(out.valid);
}

// ── The property, stated directly ───────────────────────────────────────────

/// Breakdown: sweep one unit's reading across four decades of magnitude and
/// assert the combined rate never moves more than the healthy pair's own spread.
/// This is the statement REQ-ADET-008 is written on, and it is what an average
/// fails by construction.
TEST(ImuVoting, OutputIsBoundedRegardlessOfOneUnitsValue) {
  gnc::ImuVoteInput units[3];
  healthyTriad(kTruth, units);
  const Eigen::Vector3d pair_answer = 0.5 * (units[0].rate.eigen() + units[1].rate.eigen());

  for (double magnitude = 1.0e-2; magnitude < 1.0e5; magnitude *= 10.0) {
    // A fresh voter per point, so this measures the gate rather than a latch
    // left over from the previous magnitude.
    gnc::ImuVoter voter(refConfig());
    healthyTriad(kTruth, units);
    units[2] = present(Eigen::Vector3d(magnitude, -magnitude, magnitude));

    gnc::ImuVoteResult out;
    ASSERT_TRUE(voter.vote(units, 3, nullptr, out)) << "magnitude " << magnitude;
    // Either the fault was gated (pair answer) or it was small enough to be a
    // legitimate reading and the median absorbed it; in both cases the output
    // stays inside the healthy units' own spread of the truth.
    EXPECT_LT((out.rate.eigen() - kTruth).norm(), 1.0e-2) << "magnitude " << magnitude;
    if (magnitude > refConfig().max_rate_radps) {
      EXPECT_EQ(pair_answer, out.rate.eigen()) << "magnitude " << magnitude;
    }
  }
}
