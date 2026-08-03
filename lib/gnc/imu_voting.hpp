#ifndef POLARIS_GNC_IMU_VOTING_HPP
#define POLARIS_GNC_IMU_VOTING_HPP

/// @file
/// @brief Fault-tolerant combination of N redundant IMU rate measurements
/// (design doc §8.2, §9.1, §9.2; REQ-ADET-008, REQ-ADET-009).
///
/// The gyro-specific half of the redundancy vote. **The policy — the plausibility
/// ladder, the median, the pairwise identification with its margin and
/// confirmation gates, the exclusion latch and the criterion-matched re-admission
/// — lives in `gnc/unit_voting.hpp`, and that is where its argument is written
/// out.** It is shared with the magnetometer vote (`gnc/mag_voting.hpp`) because
/// nothing in it is about gyros: the mean's breakdown point is zero whatever it is
/// averaging [rousseeuw1987, §1.2], and a railed unit destroys the answer while
/// its own validity flag still reads true [gilmore1972].
///
/// What is gyro-specific, and therefore what this file is:
///
///  - **The plausibility gate is a physical body-rate limit.** Not the gyro's
///    measurement range — that is what a railed unit reports — but the fastest
///    this vehicle can credibly be turning (@ref ImuVoteConfig::max_rate_radps).
///  - **The identification reference is the MEKF's propagated body rate.** With
///    exactly two units — the reference vehicle's configuration — a disagreement
///    can be *detected* but not *attributed*, so the tie-break comes from the
///    filter's own dynamics: the unit the filter disbelieves loses. That reference
///    is **range-checked against the same rate limit**, because it comes from the
///    estimator's own published state and an upstream fault can hand the vote an
///    absurd value; an unchecked one makes *both* residuals enormous and awards
///    the identification to whichever unit happens to sit nearer to nonsense,
///    latching out the healthy unit. A failing reference is treated as no
///    reference, which is the honest ambiguous outcome.
///
///    The residual is against the *filter's* rate, which is bias-corrected — the
///    MEKF estimates gyro bias as part of its state — so a slow common-mode drift
///    shared by both units is partly absorbed into that estimate rather than
///    appearing in both residuals. That helps and is worth stating, but it is not
///    the guard: bias observability depends on the vector measurements the filter
///    is getting, so it cannot be relied on during eclipse or a measurement
///    outage. The margin and confirmation gates in the core are what make the
///    identification safe without it.
///
/// **Frames, units, conventions.** Rates are `Vec3<Body>` in rad/s; SI throughout.
/// Time does not appear: staleness is a property of the caller's measurement port
/// (it needs the master clock), so the caller reports it by leaving
/// @ref ImuVoteInput::present false.
///
/// **Flight path.** Fixed-size storage, no heap, no exceptions, no recursion,
/// bounded loops, every return code checked. No F´ types and no I/O — the
/// `AttitudeEstimator` component wraps this.
///
/// References: as `gnc/unit_voting.hpp` ([rousseeuw1987], [gilmore1972]), plus
/// design doc §8.2 (multi-IMU fusion, user decision 2026-08-02).

#include <cstdint>

#include "gnc/unit_voting.hpp"
#include "math/frames.hpp"
#include "math/typed_vector.hpp"

namespace polaris::gnc {

/// Largest IMU count the voter carries. Matches `flight.GncMaxUnits`, so every
/// unit the measurement port array can deliver has a voting slot.
inline constexpr int kMaxImuUnits = kMaxVoteUnits;

/// The vote's per-unit disposition and overall verdict are the shared ones — one
/// enumeration per concept, so an FDIR consumer reasoning about "a unit was
/// excluded for being out of range" does not need to know which sensor type
/// raised it. `kOutOfRange` is the rate-limit gate here.
using ImuVoteReason = VoteReason;
using ImuVoteStatus = VoteStatus;

/// Voter tuning. No defaults: an unconfigured voter refuses every cycle, the same
/// rule the rest of `lib/gnc` follows (§19.3 — there are no flight defaults, and a
/// plausibility gate on an invented limit is worse than none).
struct ImuVoteConfig {
  /// Physical body-rate magnitude limit for this vehicle [rad/s]. A reading above
  /// it is not a rate the vehicle can be at, so it is a failed unit. Set it from
  /// the vehicle's own worst credible rate (separation tip-off, detumble entry)
  /// with margin — **not** from the gyro's measurement range, which is what a
  /// railed unit reports.
  double max_rate_radps = 0.0;

  /// Pairwise disagreement threshold on the rate difference magnitude [rad/s],
  /// used only in the 2-unit case (@ref UnitVoteConfig::disagreement).
  double disagreement_radps = 0.0;

  /// Consecutive cycles passing the criterion that excluded it that re-admit a
  /// unit (@ref UnitVoteConfig::readmit_cycles). 0 is not allowed.
  std::uint32_t readmit_cycles = 0;

  /// Consecutive cycles the same unit must lose the pairwise identification before
  /// it is latched out (@ref UnitVoteConfig::identify_confirm_cycles). 0 is not
  /// allowed.
  std::uint32_t identify_confirm_cycles = 0;

  /// Range gate, applied at construction. All four values must be present and
  /// sane; the voter stays inert otherwise.
  bool isValid() const;
};

/// One unit's contribution to a vote.
struct ImuVoteInput {
  /// Body rate [rad/s] this unit reports. Read only when @ref present.
  math::Vec3<math::frames::Body> rate{};

  /// The caller has a fresh, flagged-valid sample from this unit. False covers
  /// every gate the voter cannot apply itself: the unit's own validity flag, a
  /// non-positive accumulation interval, staleness against the master clock, and a
  /// port that nothing is connected to.
  bool present = false;
};

/// A vote's outcome. Mirrors @ref UnitVoteResult with the rate carrying its frame
/// tag, which is what the estimator consumes.
struct ImuVoteResult {
  /// The combined body rate [rad/s]. Meaningful only when @ref valid.
  math::Vec3<math::frames::Body> rate{};

  /// The rate is usable. False for @ref ImuVoteStatus::kNoValue and
  /// @ref ImuVoteStatus::kAmbiguous.
  bool valid = false;

  ImuVoteStatus status = ImuVoteStatus::kNoValue;

  /// Units in the combination.
  int contributing = 0;

  /// Per-unit disposition, indexed as the input array.
  ImuVoteReason reason[kMaxImuUnits] = {};

  /// This unit crossed into exclusion **this cycle**: the edge the caller turns
  /// into an FDIR event.
  bool newly_excluded[kMaxImuUnits] = {};

  /// This unit was re-admitted this cycle (the recovery edge).
  bool newly_readmitted[kMaxImuUnits] = {};

  /// Bit i set means unit i is currently latched out.
  std::uint32_t exclusion_mask = 0;
};

/// The IMU voter: the shared @ref UnitVoter plus the two gyro-specific gates.
/// Holds the exclusion latch and the re-admission counters across cycles, so it is
/// an object rather than a free function.
///
/// Usage: construct with the tuning, call @ref vote once per estimation cycle with
/// every unit's reading, and gate the estimator on @ref ImuVoteResult::valid.
class ImuVoter {
 public:
  ImuVoter() = default;

  /// Build with @p config. An invalid config leaves the voter **inert**: every call
  /// to @ref vote refuses and returns no rate. Check @ref isConfigured.
  explicit ImuVoter(const ImuVoteConfig& config);

  bool isConfigured() const { return core_.isConfigured(); }

  const ImuVoteConfig& config() const { return config_; }

  /// Combine one cycle's readings.
  ///
  /// @param units per-unit readings, index-aligned with the caller's port array.
  /// @param count number of populated entries. A count outside `[0, kMaxImuUnits]`
  ///        is refused (not clamped, and not an assert).
  /// @param reference_rate the MEKF's propagated body rate [rad/s], used **only**
  ///        to break a 2-unit disagreement, and only if it is finite and inside
  ///        @ref ImuVoteConfig::max_rate_radps. Pass nullptr when no filter
  ///        solution is available — an ambiguous disagreement is then reported as
  ///        such rather than resolved arbitrarily.
  /// @param out receives the combination and the per-unit dispositions.
  /// @return true when @p out carries a usable rate. False is a normal condition
  ///         (every unit absent, an ambiguous pair), never a fault of this call.
  bool vote(const ImuVoteInput* units, int count,
            const math::Vec3<math::frames::Body>* reference_rate, ImuVoteResult& out);

  /// Drop every exclusion latch and re-admission counter — the commanded
  /// re-admission path (`RESET_ESTIMATOR`).
  void clearExclusions() { core_.clearExclusions(); }

  /// Unit @p index is currently latched out. Out-of-range indices read false.
  bool isExcluded(int index) const { return core_.isExcluded(index); }

 private:
  ImuVoteConfig config_{};
  UnitVoter core_{};
};

/// Per-axis median of @p count body rates [rad/s] — @ref medianVector with the
/// frame tag on. Exposed because it is the piece worth testing directly and
/// because a caller with its own gating may want it alone.
///
/// @param rates the surviving units' rates.
/// @param count number of entries, in `[1, kMaxImuUnits]`.
/// @param out receives the per-axis median.
/// @return false (leaving @p out untouched) for a count outside that range.
bool medianRate(const math::Vec3<math::frames::Body>* rates, int count,
                math::Vec3<math::frames::Body>& out);

}  // namespace polaris::gnc

#endif  // POLARIS_GNC_IMU_VOTING_HPP
