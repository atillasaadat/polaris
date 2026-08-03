#ifndef POLARIS_GNC_IMU_VOTING_HPP
#define POLARIS_GNC_IMU_VOTING_HPP

/// @file
/// @brief Fault-tolerant combination of N redundant IMU rate measurements
/// (design doc §8.2, §9.1, §9.2; REQ-ADET-008, REQ-ADET-009).
///
/// **Why not the mean.** The obvious way to use three gyros is to average them.
/// The obvious way is wrong, and it is wrong in the one case redundancy is
/// carried for: the arithmetic mean has a **breakdown point of zero**
/// [rousseeuw1987, §1.2] — a single unit reporting an arbitrary value drags the
/// combined rate by an arbitrary amount. A railed gyro reading full scale does
/// not degrade the answer, it destroys it, and it does so *while every unit's
/// own validity flag still reads true*, because a unit that has failed high has
/// no way of knowing it. Averaging therefore converts a survivable single-unit
/// fault into a vehicle-level one: the fault is spread across the output instead
/// of being isolated to the failed unit. This is the classical redundancy-
/// management result for strapdown inertial arrays [gilmore1972].
///
/// **The committed design, in order** (§8.2):
///
///  1. **Per-unit plausibility gates.** Finiteness, then rate magnitude against
///     a configured physical limit for the vehicle. These reject the obviously
///     railed reading *before* any combination, so the robust step never has to
///     absorb a value 10^3 out of family.
///  2. **Robust combination.** Per-axis **median** for ≥ 3 surviving units: the
///     median tolerates one arbitrary fault by construction (breakdown point
///     ⌊(n−1)/2⌋/n, so 1/3 at n = 3), which is exactly the single-fault survival
///     the requirement asks for. With exactly **2** units a disagreement can be
///     *detected* but not *attributed* — the pair says one of us is lying and
///     nothing more — so identification needs a third information source, and
///     the one the vehicle has is the MEKF's own propagated rate (@ref
///     ImuVoteInput::reference_rate). The unit the filter's dynamics disbelieve
///     loses — but only on a *decisive* comparison (the loser outside the
///     disagreement gate, the winner inside it) repeated over
///     @ref ImuVoteConfig::identify_confirm_cycles, so neither an exact tie nor
///     a single noisy sample can latch a unit out. Without a usable reference a
///     detected 2-unit disagreement yields **no rate at all** rather than a coin
///     flip: see @ref ImuVoteStatus::kAmbiguous. This is the branch the
///     reference vehicle flies, because it carries two IMUs and not three.
///
///     The residual is against the *filter's* rate, which is bias-corrected —
///     the MEKF estimates gyro bias as part of its state — so a slow common-mode
///     drift shared by both units is partly absorbed into that estimate rather
///     than appearing in both residuals. That helps and is worth stating, but it
///     is not the guard: bias observability depends on the vector measurements
///     the filter is getting, so it cannot be relied on during eclipse or a
///     measurement outage. The margin and confirmation gates are what make the
///     identification safe without it.
///  3. **Exclusion is an FDIR event.** A unit that fails a gate is latched out,
///     reported (`reason`, `newly_excluded`) so the caller can raise the EVR and
///     the validity flag, and re-admitted only after @ref
///     ImuVoteConfig::readmit_cycles consecutive plausible cycles.
///
/// Weighted fusion of the surviving set is deliberately **not** done here.
/// Robustness before optimality: on a homogeneous redundant set — which is what
/// a vehicle flying one gyro model has, whatever the count — equal weights are
/// already the optimal weights, and a weighting that could be skewed by a unit
/// the gates let through would give back the property this module exists for.
///
/// **Why the median and not a parity-space residual test.** Parity space
/// [gilmore1972] is the sharper instrument — with n ≥ 4 non-coplanar sensitive
/// axes it *isolates* the failed axis rather than merely out-voting it, and it
/// detects faults far below the plausibility limit. It also needs the array's
/// geometry (each unit's mounting) and a threshold per parity direction, neither
/// of which this vehicle's configuration carries, and it degenerates at n = 3
/// skewed-triad-of-triads exactly where the per-axis median is already correct.
/// The median is the smaller correct thing; parity space is the upgrade path
/// when a dissimilar or skewed IMU array arrives.
///
/// **Re-admission is automatic, with hysteresis** (§9.2). A unit is re-admitted
/// after `readmit_cycles` consecutive plausible cycles rather than by ground
/// command, because the dominant real cause of a single implausible sample is
/// transient — a bus glitch, a dropped frame, a mechanical shock past the
/// vehicle's modelled rate limit — and permanently spending a unit of redundancy
/// on a transient is the more expensive error. The consecutive-cycle count is
/// the hysteresis that stops a marginal unit from flapping in and out; a truly
/// dead unit simply never accumulates the count. A commanded re-admission
/// (@ref ImuVoter::clearExclusions) exists as well, and is what
/// `RESET_ESTIMATOR` uses.
///
/// **Frames, units, conventions.** Rates are `Vec3<Body>` in rad/s; SI
/// throughout. Time does not appear: staleness is a property of the caller's
/// measurement port (it needs the master clock), so the caller reports it by
/// leaving @ref ImuVoteInput::present false. Everything else is here, so the
/// gates and the combination cannot be split across two owners.
///
/// **Flight path.** Fixed-size storage, no heap, no exceptions, no recursion,
/// bounded loops, every return code checked, finiteness checks on the output.
/// No F´ types and no I/O — the `AttitudeEstimator` component wraps this.
///
/// References:
///  - Rousseeuw & Leroy, *Robust Regression and Outlier Detection*, Wiley 1987,
///    §1.2 (breakdown point; the mean's is 0, the median's is 50%).
///    [rousseeuw1987]
///  - Gilmore & McKern, "A Redundant Strapdown Inertial Reference Unit (SIRU)",
///    *J. Spacecraft and Rockets* 9(1):39-47, 1972 (redundancy management and
///    failure detection/isolation for redundant inertial sensors).
///    [gilmore1972]
///  - Design doc §8.2 (multi-IMU fusion, user decision 2026-08-02), §9.2 (FDIR
///    monitor → isolation → response).

#include <cstdint>

#include "math/frames.hpp"
#include "math/typed_vector.hpp"

namespace polaris::gnc {

/// Largest IMU count the voter carries. Matches `flight.GncMaxUnits`, so every
/// unit the measurement port array can deliver has a voting slot.
inline constexpr int kMaxImuUnits = 8;

/// Voter tuning. No defaults: an unconfigured voter refuses every cycle, the
/// same rule the rest of `lib/gnc` follows (§19.3 — there are no flight
/// defaults, and a plausibility gate on an invented limit is worse than none).
struct ImuVoteConfig {
  /// Physical body-rate magnitude limit for this vehicle [rad/s]. A reading
  /// above it is not a rate the vehicle can be at, so it is a failed unit. Set
  /// it from the vehicle's own worst credible rate (separation tip-off,
  /// detumble entry) with margin — **not** from the gyro's measurement range,
  /// which is what a railed unit reports.
  double max_rate_radps = 0.0;

  /// Pairwise disagreement threshold on the rate difference magnitude [rad/s],
  /// used only in the 2-unit case. Set it above the pair's combined noise and
  /// bias-repeatability by a comfortable factor: a false disagreement costs the
  /// rate for that cycle (the ambiguous case below), so this gate should fire
  /// on faults, not on tails.
  double disagreement_radps = 0.0;

  /// Consecutive cycles passing **the criterion that excluded it** that re-admit
  /// a unit. 0 is not allowed — it would make the exclusion latch a no-op, and a
  /// unit that alternates in and out every cycle is a fault the FDIR log cannot
  /// read.
  ///
  /// The criterion is per exclusion kind, and that is not a detail: a gate
  /// failure is undone by passing that gate, but an *identified* unit
  /// (@ref ImuVoteReason::kOutvoted) was plausible by construction — it passed
  /// every per-unit gate and lost a comparison. Counting plausibility for it
  /// would re-admit it unconditionally, and it would be outvoted again on the
  /// next cycle: a permanent flap at this period, one FDIR event apiece. So an
  /// identified unit accrues credit only on cycles where it **agrees with the
  /// combination** to within @ref disagreement_radps.
  std::uint32_t readmit_cycles = 0;

  /// Consecutive cycles the *same* unit must lose the pairwise identification
  /// before it is latched out. 0 is not allowed.
  ///
  /// A latch is permanent until re-admission earns it back, so one sample must
  /// not buy one — particularly under a slow common-mode drift, where the
  /// residual ordering can flip cycle to cycle. A verdict that flips resets both
  /// units' counts and never confirms, which is the correct outcome for a
  /// disagreement the reference genuinely cannot resolve. The winner is still
  /// published during confirmation (see @ref ImuVoter::vote), so this costs
  /// detection latency, not rate availability.
  std::uint32_t identify_confirm_cycles = 0;

  /// Range gate, applied at construction. All three values must be present and
  /// sane; the voter stays inert otherwise.
  bool isValid() const;
};

/// Why a unit is not contributing this cycle. Reported per unit so the caller's
/// EVR names the gate that closed rather than "a unit went away" — the ground
/// needs to know whether to expect recovery.
enum class ImuVoteReason : std::uint8_t {
  kContributing = 0,  ///< the unit's reading is in the combination
  kAbsent = 1,        ///< no fresh valid sample this cycle (caller's gate: dropout/staleness)
  kNotFinite = 2,     ///< NaN or Inf in the reported rate
  kRateLimit = 3,     ///< magnitude above ImuVoteConfig::max_rate_radps
  kExcluded = 4,      ///< plausible now, but still serving out its exclusion latch
  kOutvoted = 5       ///< identified as the disagreeing unit of a pair by the reference rate
};

/// What the combination could conclude.
enum class ImuVoteStatus : std::uint8_t {
  kNoRate = 0,      ///< no unit survived: the caller has no body rate this cycle
  kSingle = 1,      ///< exactly one surviving unit, passed through gated (no redundancy)
  kPair = 2,        ///< two surviving units that agree; their mean is the output
  kIdentified = 3,  ///< two units disagreed and the reference rate identified the loser
  kAmbiguous = 4,   ///< two units disagreed and nothing could identify which: no rate
  kMedian = 5       ///< three or more surviving units, per-axis median
};

/// One unit's contribution to a vote.
struct ImuVoteInput {
  /// Body rate [rad/s] this unit reports. Read only when @ref present.
  math::Vec3<math::frames::Body> rate{};

  /// The caller has a fresh, flagged-valid sample from this unit. False covers
  /// every gate the voter cannot apply itself: the unit's own validity flag, a
  /// non-positive accumulation interval, staleness against the master clock, and
  /// a port that nothing is connected to.
  bool present = false;
};

/// A vote's outcome.
struct ImuVoteResult {
  /// The combined body rate [rad/s]. Meaningful only when @ref valid.
  math::Vec3<math::frames::Body> rate{};

  /// The rate is usable. False for @ref ImuVoteStatus::kNoRate and
  /// @ref ImuVoteStatus::kAmbiguous, and for a combination that came out
  /// non-finite (which cannot happen from finite inputs, and is checked anyway
  /// because an estimator fed a NaN rate fails silently).
  bool valid = false;

  ImuVoteStatus status = ImuVoteStatus::kNoRate;

  /// Units in the combination.
  int contributing = 0;

  /// Per-unit disposition, indexed as the input array. Slots past @p count are
  /// left at @ref ImuVoteReason::kAbsent.
  ImuVoteReason reason[kMaxImuUnits] = {};

  /// This unit crossed into exclusion **this cycle**: the edge the caller turns
  /// into an FDIR event. A unit that stays excluded does not re-raise it.
  bool newly_excluded[kMaxImuUnits] = {};

  /// This unit was re-admitted this cycle (the recovery edge).
  bool newly_readmitted[kMaxImuUnits] = {};

  /// Bit i set means unit i is currently latched out. Telemetered as one word so
  /// the ground sees the whole array's health without N channels.
  std::uint32_t exclusion_mask = 0;
};

/// The voter. Holds the exclusion latch and the re-admission counters across
/// cycles, so it is an object rather than a free function; everything else about
/// a vote is stateless.
///
/// Usage: construct with the tuning, call @ref vote once per estimation cycle
/// with every unit's reading, and gate the estimator on
/// @ref ImuVoteResult::valid.
class ImuVoter {
 public:
  ImuVoter() = default;

  /// Build with @p config. An invalid config leaves the voter **inert**: every
  /// call to @ref vote refuses and returns no rate. Check @ref isConfigured.
  explicit ImuVoter(const ImuVoteConfig& config);

  bool isConfigured() const { return configured_; }

  const ImuVoteConfig& config() const { return config_; }

  /// Combine one cycle's readings.
  ///
  /// @param units per-unit readings, index-aligned with the caller's port array.
  /// @param count number of populated entries. A count outside
  ///        `[0, kMaxImuUnits]` is refused (not clamped, and not an assert).
  /// @param reference_rate the MEKF's propagated body rate [rad/s], used **only**
  ///        to break a 2-unit disagreement. Pass nullptr when no filter solution
  ///        is available — an ambiguous disagreement is then reported as such
  ///        rather than resolved arbitrarily.
  /// @param out receives the combination and the per-unit dispositions.
  /// @return true when @p out carries a usable rate. False is a normal
  ///         condition (every unit absent, an ambiguous pair), never a fault of
  ///         this call.
  bool vote(const ImuVoteInput* units, int count,
            const math::Vec3<math::frames::Body>* reference_rate, ImuVoteResult& out);

  /// Drop every exclusion latch and re-admission counter — the commanded
  /// re-admission path (`RESET_ESTIMATOR`). Deliberately separate from the
  /// automatic policy: an operator saying "start over" is different information
  /// from a unit having behaved for N cycles.
  void clearExclusions();

  /// Unit @p index is currently latched out. Out-of-range indices read false.
  bool isExcluded(int index) const;

 private:
  ImuVoteConfig config_{};
  bool configured_ = false;

  /// Exclusion latch, **why** each unit was excluded (which decides what it has
  /// to do to come back), the count towards re-admission, and the count towards
  /// confirming a pairwise identification. Per unit.
  bool excluded_[kMaxImuUnits] = {};
  ImuVoteReason exclusion_reason_[kMaxImuUnits] = {};
  std::uint32_t plausible_streak_[kMaxImuUnits] = {};
  std::uint32_t identify_streak_[kMaxImuUnits] = {};

  std::uint32_t exclusionMask() const;

  /// A reference rate is usable only if it is finite **and** inside the same
  /// physical rate limit the measurements are held to.
  bool referenceUsable(const math::Vec3<math::frames::Body>* reference_rate) const;
};

/// Per-axis median of @p count body rates [rad/s], the robust combination
/// @ref ImuVoter::vote applies at three or more units. Exposed because it is the
/// piece worth testing directly and because a caller with its own gating may
/// want it alone.
///
/// For an even @p count the two central order statistics are averaged, which is
/// the standard definition and keeps the estimator continuous in the data; the
/// breakdown point is unaffected. Copies at most @ref kMaxImuUnits values per
/// axis into fixed storage and insertion-sorts them — a bounded O(n²) on n ≤ 8,
/// which beats any allocation-free heap sort at this size and has no branches
/// worth mispredicting.
///
/// @param rates the surviving units' rates.
/// @param count number of entries, in `[1, kMaxImuUnits]`.
/// @param out receives the per-axis median.
/// @return false (leaving @p out untouched) for a count outside that range.
bool medianRate(const math::Vec3<math::frames::Body>* rates, int count,
                math::Vec3<math::frames::Body>& out);

}  // namespace polaris::gnc

#endif  // POLARIS_GNC_IMU_VOTING_HPP
