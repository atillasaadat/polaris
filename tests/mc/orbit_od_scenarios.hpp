#ifndef POLARIS_TESTS_MC_ORBIT_OD_SCENARIOS_HPP
#define POLARIS_TESTS_MC_ORBIT_OD_SCENARIOS_HPP

/// @file
/// @brief The GNSS fault timelines the orbit-OD Monte Carlo campaign flies
/// (design doc §9.2, §13; `tests/mc/orbit_od_mc.cpp`).
///
/// Data, not logic. Each scenario is a name plus a list of timed events, and the
/// driver applies them to the receiver model — never to the filter. That
/// direction is the whole point: the filter must discover a fault from the data
/// it is handed, exactly as it would in flight, and a harness that told the
/// filter what was happening would be testing the harness (a defect class the
/// review-lessons catalogue records twice).
///
/// **Why these scenarios and not others.** Each one targets a *different*
/// defence, and they are deliberately not interchangeable:
///
///  - **Outages** exercise the coast horizon and the re-acquisition policy, at
///    durations either side of it. The 280 s case coasts and recovers on the
///    same solution; the 1800 s case must be *dropped* and re-acquired whole.
///    Getting either wrong is invisible in a nominal run.
///  - **Spoofing** is the case where every validity flag reads true. A spoof is
///    caught, if at all, by the NIS gate on the innovation — so the ramp rate
///    matters: a step is easy, and a slow drag is the one that walks the filter
///    off without ever tripping a gate. Both are flown.
///  - **Jamming** is geographic and repeats every orbit at the same longitudes,
///    which is a different signature from a random dropout: the outages
///    correlate with orbital phase, so any recovery transient lands in the same
///    place every time rather than averaging out across a campaign.
///  - **A GEO jump** is the wire-data trust boundary (§9.1). It is not a
///    plausible measurement at all, and it must be refused on the radius band
///    *before* it reaches the filter, because a fix that far off would otherwise
///    poison the propagation while the fix's own validity flag still reads true.
///  - **A clock jump** attacks the time tag rather than the position, which
///    reaches the filter as a monotonicity or latency violation rather than as
///    an innovation. It is the fault most likely to be silently absorbed by a
///    filter that trusts its epochs.
///  - **Degraded sigmas** are the honest-receiver failure: the fix is real but
///    the receiver says it is bad. The filter must de-weight it rather than
///    reject it, and a campaign that only ever sees nominal sigmas never checks
///    that `R` is actually read per fix.
///
/// Every scenario also runs its nominal stretches, so one run yields both the
/// steady-state statistics and the fault response; the driver records which
/// regime each sample belongs to so the analysis can separate them.

#include <cstdint>
#include <string>
#include <vector>

namespace polaris::mc::od {

/// What a timed event does to the receiver.
enum class FaultKind : std::uint8_t {
  /// Total loss of fix for `duration_s` — the receiver reports invalid, and on
  /// recovery its own reacquisition delay applies before valid fixes resume.
  kOutage,
  /// A persistent ECEF position offset that leaves the fix **valid**. Ramped
  /// linearly from zero to `magnitude_m` over `ramp_s`, then held for the rest
  /// of `duration_s`. `ramp_s = 0` is a step.
  kSpoof,
  /// Geographic jamming: the committed region map is bound for `duration_s`, so
  /// fixes drop out wherever the sub-satellite point enters a region.
  kJam,
  /// A step in the receiver clock of `magnitude_s`, applied to the time tag and
  /// held for `duration_s`.
  kClockJump,
  /// One fix displaced to `magnitude_m` of geocentric radius — a GEO-scale jump.
  /// Instantaneous by construction: `duration_s` bounds how long the injected
  /// offset stays armed, but the scenario intends a single bad fix.
  kRadiusJump,
  /// The receiver keeps reporting, but multiplies its reported sigmas by
  /// `magnitude` for `duration_s`. Not a fault in the fix — a fault in its
  /// quality, which the filter must absorb through `R`.
  kSigmaDegrade,
};

/// One timed event on a receiver.
struct FaultEvent {
  FaultKind kind{FaultKind::kOutage};
  double start_s{0.0};
  double duration_s{0.0};
  /// Metres for @ref FaultKind::kSpoof and @ref FaultKind::kRadiusJump, seconds
  /// for @ref FaultKind::kClockJump, dimensionless for
  /// @ref FaultKind::kSigmaDegrade, unused otherwise.
  double magnitude{0.0};
  /// Ramp time for @ref FaultKind::kSpoof [s]; zero is a step.
  double ramp_s{0.0};
};

/// A named fault timeline.
struct Scenario {
  std::string name;
  std::string intent;  ///< one line, carried into the JSONL so the report can quote it
  std::vector<FaultEvent> events;
};

/// Seconds, for readability in the table below.
inline constexpr double kMinute = 60.0;
inline constexpr double kHour = 3600.0;
inline constexpr double kDay = 86400.0;

/// The campaign's scenarios.
///
/// Event times are spread across the 7-day arc rather than clustered, so each
/// fault lands at a different point in the orbit's precession and in the
/// day/night cycle, and so recovery transients do not overlap. A run shorter
/// than 7 days simply sees the prefix of this timeline that fits, which is what
/// makes the `--duration-s` smoke case meaningful rather than a different test.
inline std::vector<Scenario> scenarios() {
  return {
      {"nominal",
       "Steady-state performance with no faults: the baseline every other scenario is "
       "read against.",
       {}},

      {"outage_short",
       "Dropouts well inside the 300 s coast horizon, repeated. The solution must coast and "
       "recover without being dropped, and the covariance must grow and re-shrink.",
       {
           {FaultKind::kOutage, 2.0 * kHour, 30.0, 0.0, 0.0},
           {FaultKind::kOutage, 8.0 * kHour, 60.0, 0.0, 0.0},
           {FaultKind::kOutage, 20.0 * kHour, 120.0, 0.0, 0.0},
           {FaultKind::kOutage, 1.5 * kDay, 60.0, 0.0, 0.0},
           {FaultKind::kOutage, 3.0 * kDay, 30.0, 0.0, 0.0},
           {FaultKind::kOutage, 5.0 * kDay, 120.0, 0.0, 0.0},
       }},

      {"outage_horizon",
       "Dropouts straddling the 300 s horizon — 280 s must coast through, 320 s must drop the "
       "solution and re-acquire whole. The pair is the point; either alone would pass a filter "
       "with the wrong policy.",
       {
           {FaultKind::kOutage, 3.0 * kHour, 280.0, 0.0, 0.0},
           {FaultKind::kOutage, 12.0 * kHour, 320.0, 0.0, 0.0},
           {FaultKind::kOutage, 2.0 * kDay, 280.0, 0.0, 0.0},
           {FaultKind::kOutage, 4.0 * kDay, 320.0, 0.0, 0.0},
       }},

      {"outage_long",
       "Losses far past the horizon, up to six hours. The solution must be declared invalid and "
       "dropped, stay dropped, and re-acquire from the first fix back rather than blending "
       "against a prior that has stopped meaning anything.",
       {
           {FaultKind::kOutage, 6.0 * kHour, 30.0 * kMinute, 0.0, 0.0},
           {FaultKind::kOutage, 1.0 * kDay, 2.0 * kHour, 0.0, 0.0},
           {FaultKind::kOutage, 3.5 * kDay, 6.0 * kHour, 0.0, 0.0},
       }},

      {"spoof_step",
       "A 5 km position step that stays valid. Large enough that the NIS gate should reject it "
       "outright; the measurement of interest is how many fixes are rejected and whether the "
       "solution survives untouched.",
       {
           {FaultKind::kSpoof, 5.0 * kHour, 20.0 * kMinute, 5000.0, 0.0},
           {FaultKind::kSpoof, 2.5 * kDay, 20.0 * kMinute, 5000.0, 0.0},
       }},

      {"spoof_ramp",
       "The hard one: 2 km walked in over an hour, ~0.55 m/s. Each successive innovation is small "
       "enough to pass a gate sized for one fix's noise, so this is the case that walks a filter "
       "off its true trajectory without ever tripping an alarm. What is measured is how far it "
       "gets and whether it is caught at all.",
       {
           {FaultKind::kSpoof, 10.0 * kHour, 2.0 * kHour, 2000.0, 1.0 * kHour},
           {FaultKind::kSpoof, 4.5 * kDay, 2.0 * kHour, 2000.0, 1.0 * kHour},
       }},

      {"jamming",
       "The committed geographic jamming map, armed for a day at a time. Dropouts recur at the "
       "same longitudes every orbit, so the recovery transients correlate with orbital phase "
       "instead of averaging out — a different signature from a random dropout of the same "
       "duty cycle.",
       {
           {FaultKind::kJam, 12.0 * kHour, 1.0 * kDay, 0.0, 0.0},
           {FaultKind::kJam, 4.0 * kDay, 1.0 * kDay, 0.0, 0.0},
       }},

      {"bad_data",
       "Wire data that is not a plausible measurement at all: single fixes displaced to GEO "
       "radius, and a stretch where the receiver's own reported sigmas degrade by 50x. The first "
       "must be refused on the §9.1 radius band before the filter sees it; the second must be "
       "*absorbed* through R rather than rejected, since it is a real fix from an honest "
       "receiver having a bad day.",
       {
           {FaultKind::kRadiusJump, 4.0 * kHour, 30.0, 4.2164e7, 0.0},
           {FaultKind::kSigmaDegrade, 16.0 * kHour, 2.0 * kHour, 50.0, 0.0},
           {FaultKind::kRadiusJump, 2.2 * kDay, 30.0, 4.2164e7, 0.0},
           {FaultKind::kClockJump, 3.2 * kDay, 10.0 * kMinute, 5.0, 0.0},
           {FaultKind::kRadiusJump, 5.5 * kDay, 30.0, 4.2164e7, 0.0},
           {FaultKind::kSigmaDegrade, 6.0 * kDay, 2.0 * kHour, 50.0, 0.0},
       }},
  };
}

/// True when @p event is armed at @p t_s.
inline bool active(const FaultEvent& event, double t_s) {
  return t_s >= event.start_s && t_s < event.start_s + event.duration_s;
}

/// A short label for the regime a sample belongs to, so the analysis can
/// separate steady state from fault response without re-deriving the timeline.
inline const char* kindName(FaultKind kind) {
  switch (kind) {
    case FaultKind::kOutage:
      return "outage";
    case FaultKind::kSpoof:
      return "spoof";
    case FaultKind::kJam:
      return "jam";
    case FaultKind::kClockJump:
      return "clock_jump";
    case FaultKind::kRadiusJump:
      return "radius_jump";
    case FaultKind::kSigmaDegrade:
      return "sigma_degrade";
  }
  return "unknown";
}

}  // namespace polaris::mc::od

#endif  // POLARIS_TESTS_MC_ORBIT_OD_SCENARIOS_HPP
