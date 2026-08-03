#ifndef POLARIS_GNC_MAG_VOTING_HPP
#define POLARIS_GNC_MAG_VOTING_HPP

/// @file
/// @brief Fault-tolerant combination of N redundant magnetometer field
/// measurements (design doc §8.2, §9.1, §9.2; REQ-ADET-011).
///
/// The magnetometer-specific half of the redundancy vote. **The policy — the
/// plausibility ladder, the median, the pairwise identification with its margin
/// and confirmation gates, the exclusion latch and the criterion-matched
/// re-admission — lives in `gnc/unit_voting.hpp`**, shared with the multi-IMU vote
/// (`gnc/imu_voting.hpp`). Nothing in that policy is about gyros or about
/// magnetometers: the mean's breakdown point is zero whatever it is averaging
/// [rousseeuw1987, §1.2], so a second magnetometer averaged in is a second way to
/// lose the field rather than a redundancy.
///
/// What is magnetometer-specific, and therefore what this file is:
///
///  - **The plausibility gate is the field magnitude against the modelled one.**
///    This is the natural physical gate for a magnetometer and it needs no new
///    knowledge: the vehicle already evaluates IGRF-14 at its own position every
///    cycle to build the magnetic reference (§8.1), and `‖m‖` has to sit in a band
///    around `‖B_IGRF‖` whatever the attitude is. That is what makes it *better*
///    than a fixed full-scale check — it tracks the field from 22 µT to 52 µT over
///    an orbit instead of admitting everything under saturation — and it is
///    attitude-free, so it still works in the coarse and cold-start cases where
///    there is no attitude to rotate a reference with. The band is a **ratio**
///    rather than an absolute pair (@ref MagVoteConfig::min_field_ratio,
///    @ref MagVoteConfig::max_field_ratio), so it does not have to be re-derived
///    when the orbit changes.
///
///    The vote is only ever called on a cycle that has a modelled field, because
///    without one the estimator has no magnetic pair to form either — the
///    component gates both on the same `have_mag_ref`. A call with a non-positive
///    or non-finite modelled magnitude is refused rather than run with the gate
///    disabled.
///
///  - **The identification reference is the modelled field rotated into body
///    axes** by the current attitude solution. With exactly two units — the
///    reference vehicle's configuration — a disagreement can be *detected* but not
///    *attributed*, so the tie-break comes from the one independent statement the
///    vehicle has about what the field *should* read.
///
///    That reference is only as good as the attitude behind it, so it carries its
///    own validity condition and it is a **quality** condition, not merely a
///    validity flag: @ref MagVoteReference::attitude_sigma_rad must be finite and
///    at or below @ref MagVoteConfig::max_attitude_sigma_rad. A validity flag
///    cannot tell a 0.5° solution from a 10° one, and a 10° attitude error moves
///    the predicted field by `σ·‖B‖` — about 5 µT on a 30 µT field, which is
///    several times any sane disagreement gate. Identifying on that would hand the
///    verdict to whichever unit happened to sit nearer to a badly rotated
///    prediction. Derive the bound from the gate: the reference is worth using
///    while `σ·‖B‖` is comfortably inside @ref MagVoteConfig::disagreement_tesla.
///    A failing reference is treated as no reference, which is the honest
///    ambiguous outcome and the same response the IMU vote gives.
///
/// **What an unattributable disagreement costs, and why that is acceptable.** No
/// magnetic pair that cycle: the estimator's existing dropout behaviour (§8.1) —
/// the coarse chain refuses TRIAD and coasts on the gyro, the MEKF simply skips
/// the magnetic update. That is strictly cheaper than the IMU equivalent, which
/// costs the body rate itself, and it is why the response is the same refusal
/// rather than a guess.
///
/// **Frames, units, conventions.** Fields are `Vec3<Body>` in tesla; SI
/// throughout. Time does not appear: staleness is a property of the caller's
/// measurement port, so the caller reports it by leaving
/// @ref MagVoteInput::present false. The correction from a commanded
/// hard/soft-iron calibration (`gnc/mag_calibration.hpp`) is applied **after** the
/// vote, not before: the vote's job is to decide which unit's raw reading the
/// vehicle believes, and a calibration fitted for one unit must never be used to
/// judge another.
///
/// **Flight path.** Fixed-size storage, no heap, no exceptions, no recursion,
/// bounded loops, every return code checked. No F´ types and no I/O — the
/// `AttitudeEstimator` component wraps this.
///
/// References: as `gnc/unit_voting.hpp` ([rousseeuw1987], [gilmore1972]), plus
/// design doc §8.2 (dual magnetometers with voting, user decision 2026-08-02).

#include <cstdint>

#include "gnc/unit_voting.hpp"
#include "math/frames.hpp"
#include "math/typed_vector.hpp"

namespace polaris::gnc {

/// Largest magnetometer count the voter carries. Matches `flight.GncMaxUnits`.
inline constexpr int kMaxMagUnits = kMaxVoteUnits;

/// Shared with the IMU vote — one enumeration per concept, so an FDIR consumer
/// reasoning about "a unit was excluded for being out of range" does not need to
/// know which sensor type raised it. `kOutOfRange` is the field-magnitude band
/// here.
using MagVoteReason = VoteReason;
using MagVoteStatus = VoteStatus;

/// Voter tuning. No defaults, the same rule as the rest of `lib/gnc` (§19.3).
struct MagVoteConfig {
  /// Lower and upper bounds of the accepted `‖m‖ / ‖B_IGRF‖` ratio
  /// [dimensionless]. Both must be positive with `min < 1 < max`: a band that
  /// excluded the modelled value itself would be a configuration error, not a
  /// tight gate. Size it from the installed hard-iron term plus scale error with
  /// margin — it is a fault gate, not an accuracy gate.
  double min_field_ratio = 0.0;
  double max_field_ratio = 0.0;

  /// Pairwise disagreement threshold on the field difference magnitude [T], used
  /// only in the 2-unit case (@ref UnitVoteConfig::disagreement). Set it above the
  /// pair's combined noise and installed hard-iron spread by a comfortable factor:
  /// a false disagreement costs the magnetic pair for that cycle.
  double disagreement_tesla = 0.0;

  /// Largest attitude-error 1σ [rad] at which the rotated reference is still worth
  /// identifying on. See the file header — this is a *quality* gate, and it is the
  /// difference between using the reference and being misled by it.
  double max_attitude_sigma_rad = 0.0;

  /// Consecutive cycles passing the criterion that excluded it that re-admit a
  /// unit (@ref UnitVoteConfig::readmit_cycles). 0 is not allowed.
  std::uint32_t readmit_cycles = 0;

  /// Consecutive cycles the same unit must lose the pairwise identification before
  /// it is latched out (@ref UnitVoteConfig::identify_confirm_cycles). 0 is not
  /// allowed.
  std::uint32_t identify_confirm_cycles = 0;

  /// Range gate, applied at construction. Every value must be present and sane;
  /// the voter stays inert otherwise.
  bool isValid() const;
};

/// One unit's contribution to a vote.
struct MagVoteInput {
  /// Measured field, body frame [T], **raw** — before any applied hard/soft-iron
  /// correction. Read only when @ref present.
  math::Vec3<math::frames::Body> field_tesla{};

  /// The caller has a fresh, flagged-valid sample from this unit. False covers the
  /// gates the voter cannot apply itself: the unit's own validity flag, staleness
  /// against the master clock, a port that nothing is connected to, and (when the
  /// §7 MTQ/MAG interlock lands) a magnetorquer-contaminated sampling window.
  bool present = false;
};

/// The independent reference used to attribute a two-unit disagreement, and the
/// quality condition that decides whether it may be used at all.
struct MagVoteReference {
  /// Modelled field in **body** axes [T]: the onboard IGRF-14 evaluation rotated
  /// through the current attitude solution.
  math::Vec3<math::frames::Body> field_tesla{};

  /// The attitude used for that rotation is valid.
  bool attitude_valid = false;

  /// 1σ of that attitude's error [rad] — the per-axis figure from the published
  /// covariance, not its trace.
  double attitude_sigma_rad = 0.0;
};

/// A vote's outcome. Mirrors @ref UnitVoteResult with the field carrying its frame
/// tag, plus the index of the unit whose reading was published — which the caller
/// needs, because a per-unit calibration is applied downstream and must follow the
/// unit the vote actually chose.
struct MagVoteResult {
  /// The combined field [T], body frame. Meaningful only when @ref valid.
  math::Vec3<math::frames::Body> field_tesla{};

  /// The field is usable. False for @ref MagVoteStatus::kNoValue and
  /// @ref MagVoteStatus::kAmbiguous.
  bool valid = false;

  MagVoteStatus status = MagVoteStatus::kNoValue;

  /// Units in the combination.
  int contributing = 0;

  /// Port index of the **lowest-indexed contributing** unit, or -1 when none
  /// contributed. On a combination of several units this is a representative
  /// rather than the sole source, which is stated rather than hidden: a per-unit
  /// correction applied to a combined reading is an approximation, and the
  /// reference vehicle's two-unit suite only ever combines an *agreeing* pair, for
  /// which it is a good one.
  int published_index = -1;

  /// Per-unit disposition, indexed as the input array.
  MagVoteReason reason[kMaxMagUnits] = {};

  /// This unit crossed into exclusion **this cycle**: the edge the caller turns
  /// into an FDIR event.
  bool newly_excluded[kMaxMagUnits] = {};

  /// This unit was re-admitted this cycle (the recovery edge).
  bool newly_readmitted[kMaxMagUnits] = {};

  /// Bit i set means unit i is currently latched out.
  std::uint32_t exclusion_mask = 0;
};

/// The magnetometer voter: the shared @ref UnitVoter plus the field-magnitude band
/// and the reference-quality gate. Holds the exclusion latch and the re-admission
/// counters across cycles.
class MagVoter {
 public:
  MagVoter() = default;

  /// Build with @p config. An invalid config leaves the voter **inert**: every call
  /// to @ref vote refuses. Check @ref isConfigured.
  explicit MagVoter(const MagVoteConfig& config);

  bool isConfigured() const { return core_.isConfigured(); }

  const MagVoteConfig& config() const { return config_; }

  /// Combine one cycle's readings.
  ///
  /// @param units per-unit readings, index-aligned with the caller's port array.
  /// @param count number of populated entries. A count outside `[0, kMaxMagUnits]`
  ///        is refused (not clamped, and not an assert).
  /// @param modelled_magnitude_tesla `‖B_IGRF‖` at this cycle's position and epoch
  ///        [T], which sets the plausibility band. Must be finite and positive —
  ///        a call without a modelled field is refused rather than run with the
  ///        gate silently disabled.
  /// @param reference the rotated modelled field and its attitude quality, or
  ///        nullptr when no attitude solution exists. Used **only** to break a
  ///        2-unit disagreement.
  /// @param out receives the combination and the per-unit dispositions.
  /// @return true when @p out carries a usable field. False is a normal condition
  ///         (every unit absent, an ambiguous pair), never a fault of this call.
  bool vote(const MagVoteInput* units, int count, double modelled_magnitude_tesla,
            const MagVoteReference* reference, MagVoteResult& out);

  /// Drop every exclusion latch and re-admission counter — the commanded
  /// re-admission path (`RESET_ESTIMATOR`).
  void clearExclusions() { core_.clearExclusions(); }

  /// Unit @p index is currently latched out. Out-of-range indices read false.
  bool isExcluded(int index) const { return core_.isExcluded(index); }

 private:
  MagVoteConfig config_{};
  UnitVoter core_{};
};

}  // namespace polaris::gnc

#endif  // POLARIS_GNC_MAG_VOTING_HPP
