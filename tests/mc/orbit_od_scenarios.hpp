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
///  - **Burns** (Push 70) accelerate the *truth* along-track and either hand the
///    filter the acceleration or not: the fed/blind pair measures what the
///    non-gravitational acceleration input buys through an outage.
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
///  - **Fix latency** is not a fault at all — it is the receiver behaving exactly
///    as specified, and it is the largest error the filter can carry (~7.6 m of
///    along-track position per millisecond in LEO). It gets its own scenario
///    because it is the only one whose *cadence* differs: at the campaign's 10 s
///    cycle a 50 ms latency is not resolvable, so it is flown at 50 Hz over a
///    short arc where it is.
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
  /// A finite burn (§17): the *truth* accelerates along-track at `magnitude`
  /// [m/s²] for `duration_s`. Not a receiver fault at all — an unmodelled
  /// force, which is what the filter's non-gravitational acceleration input
  /// exists for (Push 70). `tell_filter` says whether the filter is handed the
  /// acceleration (as the burn executor would) or flies blind through it.
  kThrust,
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
  /// @ref FaultKind::kThrust only: hand the acceleration to the filter (true,
  /// the burn executor's path) or leave it blind (false, the paper's baseline).
  bool tell_filter{true};
};

/// A named fault timeline.
struct Scenario {
  std::string name;
  std::string intent;  ///< one line, carried into the JSONL so the report can quote it
  std::vector<FaultEvent> events;

  /// GNC cycle period [s]; zero takes the driver's campaign default. This is the
  /// rate the filter is propagated *and* the receiver polled at, because in
  /// flight they are the same cycle.
  ///
  /// It is per-scenario because the receiver's fix latency is only observable at
  /// a cadence that can resolve it: the delay-line model delivers the newest
  /// solution at least one latency old, so polling slower than the latency makes
  /// the realised delay a whole *poll* rather than the datasheet's. See
  /// @ref fix_latency_s.
  double cycle_period_s{0.0};

  /// Receiver fix latency [s]; see `GnssSpec::fix_latency_s`. Left at zero on
  /// the long arcs, whose 10 s cadence cannot resolve 50 ms, and set to the
  /// datasheet value only on a scenario whose @ref cycle_period_s can.
  double fix_latency_s{0.0};

  /// Cap on arc length [s]; zero means the campaign duration. A fast-cadence
  /// scenario is bounded here rather than by the campaign flag, so `--duration-s
  /// 7d` does not silently turn a 50 Hz scenario into 30 million samples.
  double max_duration_s{0.0};
};

/// Seconds, for readability in the table below.
inline constexpr double kMinute = 60.0;
inline constexpr double kHour = 3600.0;
inline constexpr double kDay = 86400.0;

/// The campaign's scenarios.
///
/// Event times are spread across each scenario's arc rather than clustered, so
/// every fault lands at a different point in the orbit's precession and in the
/// day/night cycle, and so recovery transients do not overlap. A run shorter
/// than the arc simply sees the prefix of this timeline that fits, which is what
/// makes the `--duration-s` smoke case meaningful rather than a different test.
///
/// Only `nominal` flies the full campaign duration
/// ---------------------------------------------------
/// The long nominal arc exists to answer one question — does the estimate stay
/// bounded, and the covariance honest, over a long coast against a truth model
/// the filter does not carry — and that question is asked of the un-faulted
/// filter. It is also the only scenario the consistency and ensemble statistics
/// are measured on (`analysis/od/statistics.py`), for the separate reason that a
/// fault stretch is not drawn from the distribution the covariance describes.
///
/// The fault scenarios ask a different kind of question: does the right layer
/// refuse, does the coast policy hold at its boundary, how far does a slow spoof
/// walk the estimate. Those resolve within a fault and its recovery transient —
/// minutes to hours — and what more arc buys is repetitions, not new
/// information. So each is capped at a day and its events compressed into it,
/// preserving the count, the magnitudes and the spacing that keeps transients
/// apart. A day is 15 orbits at the reference vehicle's 5677 s period, so the
/// phase diversity the spreading exists for survives the compression.
///
/// This is worth 4x the campaign: 56 spacecraft-days per run became 14, and the
/// 30-run campaign went from ~170 CPU-hours to ~43. The cost is fewer samples in
/// the fault regimes' error distributions, which no criterion is short of — the
/// band-refusal check wants refusals, and a day supplies them.
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
           {FaultKind::kOutage, 5.0 * kHour, 60.0, 0.0, 0.0},
           {FaultKind::kOutage, 9.0 * kHour, 120.0, 0.0, 0.0},
           {FaultKind::kOutage, 13.0 * kHour, 60.0, 0.0, 0.0},
           {FaultKind::kOutage, 17.0 * kHour, 30.0, 0.0, 0.0},
           {FaultKind::kOutage, 21.0 * kHour, 120.0, 0.0, 0.0},
       },
       /*cycle_period_s=*/0.0,
       /*fix_latency_s=*/0.0,
       /*max_duration_s=*/1.0 * kDay},

      {"outage_horizon",
       "Dropouts straddling the 300 s horizon — 280 s must coast through, 320 s must drop the "
       "solution and re-acquire whole. The pair is the point; either alone would pass a filter "
       "with the wrong policy.",
       {
           {FaultKind::kOutage, 3.0 * kHour, 280.0, 0.0, 0.0},
           {FaultKind::kOutage, 8.0 * kHour, 320.0, 0.0, 0.0},
           {FaultKind::kOutage, 14.0 * kHour, 280.0, 0.0, 0.0},
           {FaultKind::kOutage, 20.0 * kHour, 320.0, 0.0, 0.0},
       },
       /*cycle_period_s=*/0.0,
       /*fix_latency_s=*/0.0,
       /*max_duration_s=*/1.0 * kDay},

      {"outage_long",
       "Losses far past the horizon, up to six hours. Past the 300 s fine horizon the solution "
       "stands as DEGRADED with its grown covariance (Push 70); past the 30 min degraded horizon "
       "it is dropped, stays dropped, and re-acquires from the first fix back rather than "
       "blending against a prior that has stopped meaning anything.",
       {
           {FaultKind::kOutage, 2.0 * kHour, 30.0 * kMinute, 0.0, 0.0},
           {FaultKind::kOutage, 7.0 * kHour, 2.0 * kHour, 0.0, 0.0},
           {FaultKind::kOutage, 14.0 * kHour, 6.0 * kHour, 0.0, 0.0},
       },
       /*cycle_period_s=*/0.0,
       /*fix_latency_s=*/0.0,
       /*max_duration_s=*/1.0 * kDay},

      {"spoof_step",
       "A 5 km position step that stays valid. Large enough that the NIS gate should reject it "
       "outright; the measurement of interest is how many fixes are rejected and whether the "
       "solution survives untouched.",
       {
           {FaultKind::kSpoof, 5.0 * kHour, 20.0 * kMinute, 5000.0, 0.0},
           {FaultKind::kSpoof, 16.0 * kHour, 20.0 * kMinute, 5000.0, 0.0},
       },
       /*cycle_period_s=*/0.0,
       /*fix_latency_s=*/0.0,
       /*max_duration_s=*/1.0 * kDay},

      {"spoof_ramp",
       "The hard one: 2 km walked in over an hour, ~0.55 m/s. Each successive innovation is small "
       "enough to pass a gate sized for one fix's noise, so this is the case that walks a filter "
       "off its true trajectory without ever tripping an alarm. What is measured is how far it "
       "gets and whether it is caught at all.",
       {
           {FaultKind::kSpoof, 6.0 * kHour, 2.0 * kHour, 2000.0, 1.0 * kHour},
           {FaultKind::kSpoof, 16.0 * kHour, 2.0 * kHour, 2000.0, 1.0 * kHour},
       },
       /*cycle_period_s=*/0.0,
       /*fix_latency_s=*/0.0,
       /*max_duration_s=*/1.0 * kDay},

      {"jamming",
       "The committed geographic jamming map, armed for eight hours at a time. Dropouts recur at "
       "the same longitudes every orbit, so the recovery transients correlate with orbital phase "
       "instead of averaging out — a different signature from a random dropout of the same "
       "duty cycle. Eight hours is five orbits, so the map is revisited enough times for that "
       "correlation to be the thing the record shows.",
       {
           {FaultKind::kJam, 1.0 * kHour, 8.0 * kHour, 0.0, 0.0},
           {FaultKind::kJam, 14.0 * kHour, 8.0 * kHour, 0.0, 0.0},
       },
       /*cycle_period_s=*/0.0,
       /*fix_latency_s=*/0.0,
       /*max_duration_s=*/1.0 * kDay},

      {"bad_data",
       "Wire data that is not a plausible measurement at all: single fixes displaced to GEO "
       "radius, and a stretch where the receiver's own reported sigmas degrade by 50x. The first "
       "must be refused on the §9.1 radius band before the filter sees it; the second must be "
       "*absorbed* through R rather than rejected, since it is a real fix from an honest "
       "receiver having a bad day.",
       {
           {FaultKind::kRadiusJump, 2.0 * kHour, 30.0, 4.2164e7, 0.0},
           {FaultKind::kSigmaDegrade, 5.0 * kHour, 2.0 * kHour, 50.0, 0.0},
           {FaultKind::kRadiusJump, 9.0 * kHour, 30.0, 4.2164e7, 0.0},
           {FaultKind::kClockJump, 12.0 * kHour, 10.0 * kMinute, 5.0, 0.0},
           {FaultKind::kSigmaDegrade, 15.0 * kHour, 2.0 * kHour, 50.0, 0.0},
           {FaultKind::kRadiusJump, 19.0 * kHour, 30.0, 4.2164e7, 0.0},
       },
       /*cycle_period_s=*/0.0,
       /*fix_latency_s=*/0.0,
       /*max_duration_s=*/1.0 * kDay},

      {"burn_tracked",
       "A 120 s, 0.03 m/s^2 along-track burn while GNSS is nominal, the filter fed the "
       "acceleration as the burn executor would (Push 70). Measured: fixes rejected through the "
       "burn and the position error at its end — a filter propagating blind lags the truth by "
       "1/2 a t^2 (216 m at 120 s) and rejects the fixes as outliers; a fed filter should reject "
       "none and stay at the receiver's noise.",
       {
           {FaultKind::kThrust, 3.0 * kHour, 120.0, 0.03, 0.0, true},
           {FaultKind::kThrust, 15.0 * kHour, 120.0, 0.03, 0.0, true},
       },
       /*cycle_period_s=*/1.0,
       /*fix_latency_s=*/0.0,
       /*max_duration_s=*/1.0 * kDay},

      {"burn_outage_fed",
       "The same 120 s burn inside a 15 min outage, the filter fed the acceleration: the "
       "coasted error at the outage's end is the number the paper measured (5 km fed against "
       "9 km blind, on a J2 model). Read against burn_outage_blind.",
       {
           {FaultKind::kOutage, 3.0 * kHour, 15.0 * kMinute, 0.0, 0.0},
           {FaultKind::kThrust, 3.0 * kHour + 5.0 * kMinute, 120.0, 0.03, 0.0, true},
           {FaultKind::kOutage, 15.0 * kHour, 15.0 * kMinute, 0.0, 0.0},
           {FaultKind::kThrust, 15.0 * kHour + 5.0 * kMinute, 120.0, 0.03, 0.0, true},
       },
       /*cycle_period_s=*/1.0,
       /*fix_latency_s=*/0.0,
       /*max_duration_s=*/1.0 * kDay},

      {"burn_outage_blind",
       "burn_outage_fed with the filter told nothing: it coasts on gravity and drag through a "
       "burn it cannot see. The difference between the two scenarios' outage-end errors is what "
       "the acceleration input buys.",
       {
           {FaultKind::kOutage, 3.0 * kHour, 15.0 * kMinute, 0.0, 0.0},
           {FaultKind::kThrust, 3.0 * kHour + 5.0 * kMinute, 120.0, 0.03, 0.0, false},
           {FaultKind::kOutage, 15.0 * kHour, 15.0 * kMinute, 0.0, 0.0},
           {FaultKind::kThrust, 15.0 * kHour + 5.0 * kMinute, 120.0, 0.03, 0.0, false},
       },
       /*cycle_period_s=*/1.0,
       /*fix_latency_s=*/0.0,
       /*max_duration_s=*/1.0 * kDay},

      {"latency_fast",
       "The only scenario that exercises the fix-latency correction. Polls at 50 Hz over two "
       "minutes with the OEM7600's 50 ms latency armed, so the delivered fix is genuinely ~50 ms "
       "behind the filter's own epoch and the latent-fix branch fires on every update. The long "
       "arcs cannot do this: at their 10 s cadence the delivered fix is a whole poll old, which "
       "models a 10 s latency rather than the receiver's.",
       {},
       /*cycle_period_s=*/0.02,
       /*fix_latency_s=*/0.05,
       // Two minutes, not ten. The correction fires on *every* update here, so
       // the arc buys repetitions of one branch and nothing else — 6000 firings
       // establish it as well as 30000 do. At 50 Hz the scenario was a fifth of
       // the whole campaign's cycle count while covering 0.02% of its flight
       // time, which is the wrong place to spend an hour of every campaign.
       /*max_duration_s=*/2.0 * kMinute},
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
    case FaultKind::kThrust:
      return "thrust";
  }
  return "unknown";
}

}  // namespace polaris::mc::od

#endif  // POLARIS_TESTS_MC_ORBIT_OD_SCENARIOS_HPP
