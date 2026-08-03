#ifndef POLARIS_GNC_UNIT_VOTING_HPP
#define POLARIS_GNC_UNIT_VOTING_HPP

/// @file
/// @brief Fault-tolerant combination of N redundant 3-vector measurements — the
/// machinery shared by the multi-IMU and multi-magnetometer votes (design doc
/// §8.2, §9.1, §9.2; REQ-ADET-008, REQ-ADET-009, REQ-ADET-011).
///
/// This is the *policy*, with no idea what it is voting on. Everything physical
/// — what makes one unit's reading implausible, and what makes an independent
/// reference usable — is the caller's, because those are the only two things
/// that differ between a gyro triad and a magnetometer pair. @ref ImuVoter
/// (`gnc/imu_voting.hpp`) and @ref MagVoter (`gnc/mag_voting.hpp`) are the two
/// thin wrappers that supply them; a third sensor type is another wrapper rather
/// than another copy of the state machine below.
///
/// **Why not the mean.** The obvious way to use three of something is to average
/// them. The obvious way is wrong, and it is wrong in the one case redundancy is
/// carried for: the arithmetic mean has a **breakdown point of zero**
/// [rousseeuw1987, §1.2] — a single unit reporting an arbitrary value drags the
/// combination by an arbitrary amount. A railed sensor reading full scale does
/// not degrade the answer, it destroys it, and it does so *while that unit's own
/// validity flag still reads true*, because a unit that has failed high has no
/// way of knowing. Averaging therefore converts a survivable single-unit fault
/// into a vehicle-level one: the fault is spread across the output instead of
/// being isolated to the failed unit. This is the classical redundancy-management
/// result for strapdown inertial arrays [gilmore1972], and nothing in it is
/// specific to gyros.
///
/// **The ladder, in order** (§8.2):
///
///  1. **Per-unit plausibility gates.** Finiteness here; the physical range gate
///     in the caller, reported through @ref UnitVoteInput::in_range. These reject
///     the obviously railed reading *before* any combination, so the robust step
///     never has to absorb a value orders of magnitude out of family.
///  2. **Robust combination.** Per-axis **median** for ≥ 3 surviving units: the
///     median tolerates one arbitrary fault by construction (breakdown point
///     ⌊(n−1)/2⌋/n, so 1/3 at n = 3). With exactly **2** units a disagreement can
///     be *detected* but not *attributed* — the pair says one of us is lying and
///     nothing more — so identification needs a third information source, which
///     the caller supplies as @p reference. The unit the reference disbelieves
///     loses, but only on a **decisive** comparison (the loser outside the
///     disagreement gate, the winner inside it) repeated over
///     @ref UnitVoteConfig::identify_confirm_cycles, so neither an exact tie nor
///     a single noisy sample can latch a unit out. Without a usable reference a
///     detected 2-unit disagreement yields **no value at all** rather than a coin
///     flip: see @ref VoteStatus::kAmbiguous.
///  3. **Exclusion is an FDIR event.** A unit that fails a gate is latched out,
///     reported (`reason`, `newly_excluded`) so the caller can raise the EVR and
///     the validity flag, and re-admitted only after
///     @ref UnitVoteConfig::readmit_cycles consecutive cycles passing **the
///     criterion that excluded it**.
///
/// Weighted fusion of the surviving set is deliberately **not** done here.
/// Robustness before optimality: on a homogeneous redundant set — which is what a
/// vehicle flying one part number has, whatever the count — equal weights are
/// already the optimal weights, and a weighting that could be skewed by a unit
/// the gates let through would give back the property this module exists for.
///
/// **Why the median and not a parity-space residual test.** Parity space
/// [gilmore1972] is the sharper instrument — with n ≥ 4 non-coplanar sensitive
/// axes it *isolates* the failed axis rather than merely out-voting it, and it
/// detects faults far below the plausibility limit. It also needs the array's
/// geometry (each unit's mounting) and a threshold per parity direction, neither
/// of which this vehicle's configuration carries, and it degenerates at n = 3
/// exactly where the per-axis median is already correct. The median is the
/// smaller correct thing; parity space is the upgrade path when a skewed or
/// dissimilar array arrives.
///
/// **Re-admission is automatic, with hysteresis** (§9.2). A unit is re-admitted
/// after `readmit_cycles` consecutive good cycles rather than by ground command,
/// because the dominant real cause of a single implausible sample is transient —
/// a bus glitch, a dropped frame, a shock past the vehicle's modelled limit — and
/// permanently spending a unit of redundancy on a transient is the more expensive
/// error. The consecutive-cycle count is the hysteresis that stops a marginal
/// unit from flapping in and out; a truly dead unit simply never accumulates the
/// count. A commanded re-admission (@ref UnitVoter::clearExclusions) exists as
/// well, and is what `RESET_ESTIMATOR` uses.
///
/// **Frames, units, conventions.** The voted quantity is a raw
/// `Eigen::Vector3d`, deliberately untyped: the wrappers own the frame tag, and
/// a policy that is identical for rad/s and tesla should not be written twice to
/// carry two tags. SI throughout. Time does not appear — staleness is a property
/// of the caller's measurement port, so the caller reports it by leaving
/// @ref UnitVoteInput::present false.
///
/// **Flight path.** Fixed-size storage, no heap, no exceptions, no recursion,
/// bounded loops, every return code checked, finiteness checks on the output.
/// No F´ types and no I/O.
///
/// References:
///  - Rousseeuw & Leroy, *Robust Regression and Outlier Detection*, Wiley 1987,
///    §1.2 (breakdown point; the mean's is 0, the median's is 50%).
///    [rousseeuw1987]
///  - Gilmore & McKern, "A Redundant Strapdown Inertial Reference Unit (SIRU)",
///    *J. Spacecraft and Rockets* 9(1):39-47, 1972 (redundancy management and
///    failure detection/isolation for redundant sensors). [gilmore1972]
///  - Design doc §8.2 (multi-IMU and multi-magnetometer fusion, user decisions
///    2026-08-02), §9.2 (FDIR monitor → isolation → response).

#include <cstdint>
#include <Eigen/Core>

namespace polaris::gnc {

/// Largest unit count any vote carries. Matches `flight.GncMaxUnits`, so every
/// unit a measurement port array can deliver has a voting slot.
inline constexpr int kMaxVoteUnits = 8;

/// Voter tuning. No defaults: an unconfigured voter refuses every cycle, the same
/// rule the rest of `lib/gnc` follows (§19.3 — there are no flight defaults, and
/// a disagreement gate on an invented threshold is worse than none).
struct UnitVoteConfig {
  /// Pairwise disagreement threshold on the difference magnitude, in the units of
  /// the voted quantity (rad/s for rates, T for fields). Used in the 2-unit case
  /// and as the re-admission criterion for an identified unit. Set it above the
  /// pair's combined noise and repeatability by a comfortable factor: a false
  /// disagreement costs the measurement for that cycle, so this gate should fire
  /// on faults, not on tails.
  double disagreement = 0.0;

  /// Consecutive cycles passing **the criterion that excluded it** that re-admit a
  /// unit. 0 is not allowed — it would make the exclusion latch a no-op, and a
  /// unit that alternates in and out every cycle is a fault the FDIR log cannot
  /// read.
  ///
  /// The criterion is per exclusion kind, and that is not a detail: a gate failure
  /// is undone by passing that gate, but an *identified* unit
  /// (@ref VoteReason::kOutvoted) was plausible by construction — it passed every
  /// per-unit gate and lost a comparison. Counting plausibility for it would
  /// re-admit it unconditionally, and it would be outvoted again on the next
  /// cycle: a permanent flap at this period, one FDIR event apiece. So an
  /// identified unit accrues credit only on cycles where it **agrees with the
  /// combination** to within @ref disagreement.
  std::uint32_t readmit_cycles = 0;

  /// Consecutive cycles the *same* unit must lose the pairwise identification
  /// before it is latched out. 0 is not allowed.
  ///
  /// A latch is permanent until re-admission earns it back, so one sample must not
  /// buy one — particularly under a slow common-mode drift, where the residual
  /// ordering can flip cycle to cycle. A verdict that flips resets both units'
  /// counts and never confirms, which is the correct outcome for a disagreement
  /// the reference genuinely cannot resolve. The winner is still published during
  /// confirmation (see @ref UnitVoter::vote), so this costs detection latency, not
  /// measurement availability.
  std::uint32_t identify_confirm_cycles = 0;

  /// Range gate, applied at construction. All three values must be present and
  /// sane; the voter stays inert otherwise.
  bool isValid() const;
};

/// Why a unit is not contributing this cycle. Reported per unit so the caller's
/// EVR names the gate that closed rather than "a unit went away" — the ground
/// needs to know whether to expect recovery.
enum class VoteReason : std::uint8_t {
  kContributing = 0,  ///< the unit's reading is in the combination
  kAbsent = 1,        ///< no fresh valid sample this cycle (caller's gate: dropout/staleness)
  kNotFinite = 2,     ///< NaN or Inf in the reported vector
  kOutOfRange = 3,    ///< the caller's physical plausibility gate rejected it
  kExcluded = 4,      ///< plausible now, but still serving out its exclusion latch
  kOutvoted = 5       ///< identified as the disagreeing unit of a pair by the reference
};

/// What the combination could conclude.
enum class VoteStatus : std::uint8_t {
  kNoValue = 0,     ///< no unit survived: the caller has no measurement this cycle
  kSingle = 1,      ///< exactly one surviving unit, passed through gated (no redundancy)
  kPair = 2,        ///< two surviving units that agree; their mean is the output
  kIdentified = 3,  ///< two units disagreed and the reference identified the loser
  kAmbiguous = 4,   ///< two units disagreed and nothing could identify which: no value
  kMedian = 5       ///< three or more surviving units, per-axis median
};

/// One unit's contribution to a vote.
struct UnitVoteInput {
  /// The vector this unit reports, in the caller's units. Read only when
  /// @ref present.
  Eigen::Vector3d value{Eigen::Vector3d::Zero()};

  /// The caller has a fresh, flagged-valid sample from this unit. False covers
  /// every gate the voter cannot apply itself: the unit's own validity flag,
  /// staleness against the master clock, and a port that nothing is connected to.
  bool present = false;

  /// The caller's **physical** plausibility verdict — a rate below the vehicle's
  /// credible maximum, a field magnitude inside a band around the modelled one.
  /// False latches the unit out with @ref VoteReason::kOutOfRange. It is the
  /// caller's because it is the only part of the ladder that knows what is being
  /// measured; finiteness is checked here, since that one is universal.
  bool in_range = true;
};

/// A vote's outcome.
struct UnitVoteResult {
  /// The combined vector. Meaningful only when @ref valid.
  Eigen::Vector3d value{Eigen::Vector3d::Zero()};

  /// The combination is usable. False for @ref VoteStatus::kNoValue and
  /// @ref VoteStatus::kAmbiguous, and for a combination that came out non-finite
  /// (which cannot happen from finite inputs, and is checked anyway because an
  /// estimator fed a NaN fails silently).
  bool valid = false;

  VoteStatus status = VoteStatus::kNoValue;

  /// Units in the combination.
  int contributing = 0;

  /// Per-unit disposition, indexed as the input array. Slots past `count` are left
  /// at @ref VoteReason::kAbsent.
  VoteReason reason[kMaxVoteUnits] = {};

  /// This unit crossed into exclusion **this cycle**: the edge the caller turns
  /// into an FDIR event. A unit that stays excluded does not re-raise it.
  bool newly_excluded[kMaxVoteUnits] = {};

  /// This unit was re-admitted this cycle (the recovery edge).
  bool newly_readmitted[kMaxVoteUnits] = {};

  /// Bit i set means unit i is currently latched out. Telemetered as one word so
  /// the ground sees the whole array's health without N channels.
  std::uint32_t exclusion_mask = 0;
};

/// The voter. Holds the exclusion latch and the re-admission counters across
/// cycles, so it is an object rather than a free function; everything else about a
/// vote is stateless.
class UnitVoter {
 public:
  UnitVoter() = default;

  /// Build with @p config. An invalid config leaves the voter **inert**: every call
  /// to @ref vote refuses and returns nothing. Check @ref isConfigured.
  explicit UnitVoter(const UnitVoteConfig& config);

  bool isConfigured() const { return configured_; }

  const UnitVoteConfig& config() const { return config_; }

  /// Combine one cycle's readings.
  ///
  /// @param units per-unit readings, index-aligned with the caller's port array.
  /// @param count number of populated entries. A count outside
  ///        `[0, kMaxVoteUnits]` is refused (not clamped, and not an assert).
  /// @param reference an independent estimate of the measured quantity, used
  ///        **only** to break a 2-unit disagreement. Pass nullptr when none is
  ///        available or when the caller's own validity gate on it failed — an
  ///        ambiguous disagreement is then reported as such rather than resolved
  ///        arbitrarily. A non-finite reference is treated as absent here too.
  /// @param out receives the combination and the per-unit dispositions.
  /// @return true when @p out carries a usable value. False is a normal condition
  ///         (every unit absent, an ambiguous pair), never a fault of this call.
  bool vote(const UnitVoteInput* units, int count, const Eigen::Vector3d* reference,
            UnitVoteResult& out);

  /// Drop every exclusion latch and re-admission counter — the commanded
  /// re-admission path (`RESET_ESTIMATOR`). Deliberately separate from the
  /// automatic policy: an operator saying "start over" is different information
  /// from a unit having behaved for N cycles.
  void clearExclusions();

  /// Unit @p index is currently latched out. Out-of-range indices read false.
  bool isExcluded(int index) const;

 private:
  UnitVoteConfig config_{};
  bool configured_ = false;

  /// Exclusion latch, **why** each unit was excluded (which decides what it has to
  /// do to come back), the count towards re-admission, and the count towards
  /// confirming a pairwise identification. Per unit.
  bool excluded_[kMaxVoteUnits] = {};
  VoteReason exclusion_reason_[kMaxVoteUnits] = {};
  std::uint32_t plausible_streak_[kMaxVoteUnits] = {};
  std::uint32_t identify_streak_[kMaxVoteUnits] = {};

  std::uint32_t exclusionMask() const;
};

/// Per-axis median of @p count vectors, the robust combination @ref UnitVoter::vote
/// applies at three or more units. Exposed because it is the piece worth testing
/// directly and because a caller with its own gating may want it alone.
///
/// For an even @p count the two central order statistics are averaged, which is
/// the standard definition and keeps the estimator continuous in the data; the
/// breakdown point is unaffected. Copies at most @ref kMaxVoteUnits values per
/// axis into fixed storage and insertion-sorts them — a bounded O(n²) on n ≤ 8,
/// which beats any allocation-free heap sort at this size and has no branches
/// worth mispredicting.
///
/// @param values the surviving units' vectors.
/// @param count number of entries, in `[1, kMaxVoteUnits]`.
/// @param out receives the per-axis median.
/// @return false (leaving @p out untouched) for a count outside that range.
bool medianVector(const Eigen::Vector3d* values, int count, Eigen::Vector3d& out);

}  // namespace polaris::gnc

#endif  // POLARIS_GNC_UNIT_VOTING_HPP
