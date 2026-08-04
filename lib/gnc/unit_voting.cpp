/// @file
/// @brief Shared redundancy-vote policy (design doc §8.2, §9.2). See
/// unit_voting.hpp for the argument this implements; the physical gates belong to
/// the wrappers (gnc/imu_voting.hpp, gnc/mag_voting.hpp).

#include "gnc/unit_voting.hpp"

#include <cmath>

namespace polaris::gnc {

namespace {

/// Insertion sort of at most @ref kMaxVoteUnits doubles, in place. Bounded loops,
/// no allocation, no recursion.
void sortSmall(double* values, int count) {
  for (int i = 1; i < count; ++i) {
    const double key = values[i];
    int j = i - 1;
    while (j >= 0 && values[j] > key) {
      values[j + 1] = values[j];
      --j;
    }
    values[j + 1] = key;
  }
}

}  // namespace

bool UnitVoteConfig::isValid() const {
  return std::isfinite(disagreement) && disagreement > 0.0 && readmit_cycles > 0 &&
         identify_confirm_cycles > 0;
}

bool medianVector(const Eigen::Vector3d* values, int count, Eigen::Vector3d& out) {
  if (values == nullptr || count < 1 || count > kMaxVoteUnits) {
    return false;
  }
  Eigen::Vector3d median = Eigen::Vector3d::Zero();
  for (int axis = 0; axis < 3; ++axis) {
    double column[kMaxVoteUnits] = {};
    for (int i = 0; i < count; ++i) {
      column[i] = values[i][axis];
    }
    sortSmall(column, count);
    // Odd count: the central order statistic. Even count: the mean of the two
    // central ones — the standard definition, continuous in the data, and with
    // the same breakdown point as the lower median.
    median[axis] =
        (count % 2 == 1) ? column[count / 2] : 0.5 * (column[count / 2 - 1] + column[count / 2]);
  }
  out = median;
  return true;
}

UnitVoter::UnitVoter(const UnitVoteConfig& config) {
  if (config.isValid()) {
    config_ = config;
    configured_ = true;
  }
}

void UnitVoter::clearExclusions() {
  for (int i = 0; i < kMaxVoteUnits; ++i) {
    excluded_[i] = false;
    exclusion_reason_[i] = VoteReason::kContributing;
    plausible_streak_[i] = 0;
    identify_streak_[i] = 0;
  }
}

std::uint32_t UnitVoter::exclusionMask() const {
  std::uint32_t mask = 0;
  for (int i = 0; i < kMaxVoteUnits; ++i) {
    if (excluded_[i]) {
      mask |= (1u << i);
    }
  }
  return mask;
}

bool UnitVoter::isExcluded(int index) const {
  if (index < 0 || index >= kMaxVoteUnits) {
    return false;
  }
  return excluded_[index];
}

bool UnitVoter::vote(const UnitVoteInput* units, int count, const Eigen::Vector3d* reference,
                     UnitVoteResult& out) {
  out = UnitVoteResult{};
  if (!configured_ || units == nullptr || count < 0 || count > kMaxVoteUnits) {
    // A refusal, not a fault of the units: nothing is latched and no edge is
    // reported, so a mis-called vote cannot manufacture an FDIR event.
    return false;
  }
  // Slots past `count` read kAbsent rather than the zero-initialised
  // kContributing, so a caller scanning the whole array — telemetry, or a search
  // for the published unit — cannot mistake an unpopulated slot for a healthy
  // unit. Costs one bounded loop over eight entries.
  for (int i = 0; i < kMaxVoteUnits; ++i) {
    out.reason[i] = VoteReason::kAbsent;
  }

  // --- Stage 1: per-unit plausibility gates --------------------------------
  // Applied before any combination, so the robust step never has to absorb a
  // value orders of magnitude out of family. A unit that fails a gate is latched
  // out; a unit that passes accumulates towards re-admission.
  //
  // **Re-admission is judged against the criterion that excluded the unit.** A
  // gate failure is undone by passing that gate; an *identification* (kOutvoted)
  // is not, because a unit outvoted by the reference was plausible by
  // construction — it passed every per-unit gate and lost a comparison.
  // Advancing its streak on plausibility alone would re-admit it unconditionally
  // and it would be outvoted again on the next cycle: a permanent
  // exclude/re-admit flap at the re-admission period, and one FDIR event per
  // flap. Those units are held here and judged in stage 3 against the
  // combination, which is the criterion that put them out.
  Eigen::Vector3d survivor_value[kMaxVoteUnits];
  int survivor_index[kMaxVoteUnits] = {};
  int survivors = 0;
  int probation[kMaxVoteUnits] = {};  // excluded kOutvoted units, plausible this cycle
  int probation_count = 0;

  for (int i = 0; i < count; ++i) {
    const UnitVoteInput& unit = units[i];
    if (!unit.present) {
      // Absence is not implausibility. A dropout says nothing about whether the
      // unit is lying, so it neither latches an exclusion nor advances the
      // re-admission streak — an excluded unit cannot serve out its sentence by
      // going quiet.
      out.reason[i] = VoteReason::kAbsent;
      continue;
    }

    VoteReason gate = VoteReason::kContributing;
    if (!unit.value.allFinite()) {
      gate = VoteReason::kNotFinite;
    } else if (!unit.in_range) {
      gate = VoteReason::kOutOfRange;
    }

    if (gate != VoteReason::kContributing) {
      out.reason[i] = gate;
      plausible_streak_[i] = 0;
      if (!excluded_[i]) {
        excluded_[i] = true;
        exclusion_reason_[i] = gate;
        out.newly_excluded[i] = true;
      }
      continue;
    }

    // Plausible this cycle. An excluded unit still has to earn its way back, on
    // the criterion that excluded it.
    if (excluded_[i]) {
      if (exclusion_reason_[i] == VoteReason::kOutvoted) {
        out.reason[i] = VoteReason::kExcluded;
        probation[probation_count] = i;
        ++probation_count;
        continue;
      }
      ++plausible_streak_[i];
      if (plausible_streak_[i] < config_.readmit_cycles) {
        out.reason[i] = VoteReason::kExcluded;
        continue;
      }
      excluded_[i] = false;
      plausible_streak_[i] = 0;
      out.newly_readmitted[i] = true;
    }

    out.reason[i] = VoteReason::kContributing;
    survivor_value[survivors] = unit.value;
    survivor_index[survivors] = i;
    ++survivors;
  }

  // --- Stage 2: robust combination -----------------------------------------
  out.contributing = survivors;
  if (survivors == 0) {
    out.status = VoteStatus::kNoValue;
    out.exclusion_mask = exclusionMask();
    return false;
  }

  if (survivors == 1) {
    // No redundancy left, so no cross-check is possible — the reading has already
    // passed the plausibility gates and that is all the evidence there is.
    // Passing it through is right: a single unit is the vehicle's only knowledge
    // of the quantity, and refusing it would cost that knowledge on no evidence
    // of a fault.
    out.status = VoteStatus::kSingle;
    out.value = survivor_value[0];
  } else if (survivors == 2) {
    const Eigen::Vector3d difference = survivor_value[0] - survivor_value[1];
    const bool reference_usable = reference != nullptr && reference->allFinite();
    if (difference.norm() <= config_.disagreement) {
      // Agreeing pair: their mean. Averaging is safe *here* and nowhere else —
      // both readings have passed the gates and each other, so there is no
      // unbounded value left for the mean to be dragged by.
      out.status = VoteStatus::kPair;
      out.value = 0.5 * (survivor_value[0] + survivor_value[1]);
    } else if (reference_usable) {
      // Detected, and identified by a third information source. The unit the
      // reference disbelieves loses — but only on a *decisive* comparison, and
      // only after the same verdict has repeated.
      const double residual0 = (survivor_value[0] - *reference).norm();
      const double residual1 = (survivor_value[1] - *reference).norm();
      const int winner = (residual0 <= residual1) ? 0 : 1;
      const int loser = 1 - winner;
      const double winner_residual = (winner == 0) ? residual0 : residual1;
      const double loser_residual = (winner == 0) ? residual1 : residual0;

      // **Margin, not just ordering.** Comparing the two residuals alone makes
      // the verdict a coin flip whenever they are close — and they are close in
      // exactly the case that matters, a slow common-mode drift where both units
      // are wrong by similar amounts. Requiring the loser to be *outside* the
      // disagreement gate while the winner is *inside* it means the reference
      // agrees with one reading and disagrees with the other, which is what
      // "identified" should mean. It also disposes of the exact tie for free:
      // equal residuals can never satisfy both halves.
      const bool decisive =
          loser_residual > config_.disagreement && winner_residual <= config_.disagreement;
      if (!decisive) {
        out.status = VoteStatus::kAmbiguous;
        out.contributing = 0;
        identify_streak_[survivor_index[0]] = 0;
        identify_streak_[survivor_index[1]] = 0;
        out.exclusion_mask = exclusionMask();
        return false;
      }

      // **Confirmation.** A latch is permanent until re-admission earns it back,
      // so one sample must not buy one. The same unit has to lose on
      // `identify_confirm_cycles` consecutive cycles; a verdict that flips
      // between units resets both counts and never confirms, which is the correct
      // outcome for a drift the reference cannot resolve.
      //
      // The winner is published *during* confirmation rather than withheld. Where
      // the reference is derived from this vote's own output — the IMU case, whose
      // reference is the filter's published rate — withholding it would starve the
      // next cycle of the reference this branch needs and the streak could never
      // accumulate. The margin gate above has already established that the
      // reference agrees with the published reading and disagrees with the
      // withheld one.
      identify_streak_[survivor_index[winner]] = 0;
      ++identify_streak_[survivor_index[loser]];

      out.status = VoteStatus::kIdentified;
      out.value = survivor_value[winner];
      out.contributing = 1;
      out.reason[survivor_index[loser]] = VoteReason::kOutvoted;

      if (identify_streak_[survivor_index[loser]] >= config_.identify_confirm_cycles &&
          !excluded_[survivor_index[loser]]) {
        // Confirmed. The loser is latched like any other failed unit, so the FDIR
        // event, the validity flag and the re-admission policy are the same
        // however the fault was found.
        excluded_[survivor_index[loser]] = true;
        exclusion_reason_[survivor_index[loser]] = VoteReason::kOutvoted;
        plausible_streak_[survivor_index[loser]] = 0;
        out.newly_excluded[survivor_index[loser]] = true;
        // A unit re-admitted earlier in *this* cycle and immediately outvoted
        // never really came back; reporting both edges would be an incoherent pair
        // for the ground to reconcile.
        out.newly_readmitted[survivor_index[loser]] = false;
      }
    } else {
      // Detected but not identified, and nothing to break the tie. **No value.**
      // The pair is positive evidence that one of the two is lying, and there is
      // no basis to choose; publishing either would propagate a possibly-railed
      // reading into the estimator, whereas withholding it lands the caller in the
      // dropout behaviour it is already designed for. That is the §9.2
      // conservative response for an unattributable disagreement. Note the
      // deliberate non-monotonicity against the single-unit case above: one unit
      // carries no evidence of a fault, two disagreeing units carry evidence and
      // no attribution.
      out.status = VoteStatus::kAmbiguous;
      out.contributing = 0;
      out.exclusion_mask = exclusionMask();
      return false;
    }
  } else {
    out.status = VoteStatus::kMedian;
    if (!medianVector(survivor_value, survivors, out.value)) {
      out.status = VoteStatus::kNoValue;
      out.contributing = 0;
      out.exclusion_mask = exclusionMask();
      return false;
    }
  }

  // Output finiteness guard. It cannot trip on finite inputs — every survivor
  // passed the finiteness gate — and it is here because an estimator handed a NaN
  // does not fail loudly, it propagates a NaN behind a validity flag that still
  // reads true.
  if (!out.value.allFinite()) {
    out.value.setZero();
    out.status = VoteStatus::kNoValue;
    out.contributing = 0;
    out.exclusion_mask = exclusionMask();
    return false;
  }

  // --- Stage 3: probation for units excluded by identification -------------
  // A kOutvoted unit is re-admitted on **agreement**, not on plausibility: the
  // criterion that put it out was disagreement with the combination, so that is
  // the criterion it has to pass. Judged here rather than in stage 1 because the
  // combination it is compared against does not exist until stage 2.
  for (int p = 0; p < probation_count; ++p) {
    const int i = probation[p];
    if ((units[i].value - out.value).norm() <= config_.disagreement) {
      ++plausible_streak_[i];
      if (plausible_streak_[i] >= config_.readmit_cycles) {
        excluded_[i] = false;
        exclusion_reason_[i] = VoteReason::kContributing;
        plausible_streak_[i] = 0;
        identify_streak_[i] = 0;
        out.newly_readmitted[i] = true;
      }
    } else {
      plausible_streak_[i] = 0;
    }
  }

  out.exclusion_mask = exclusionMask();
  out.valid = true;
  return true;
}

}  // namespace polaris::gnc
