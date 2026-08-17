#ifndef POLARIS_SIM_SENSORS_GNSS_HPP
#define POLARIS_SIM_SENSORS_GNSS_HPP

/// @file
/// @brief GNSS receiver truth model — PVT (position/velocity/time) fix (§6.2, §8.3).
///
/// A GNSS receiver's interface is a **navigation solution**, not raw signals: it
/// reports an ECEF position and velocity stamped in **GPS time**, having already
/// run its own acquisition, pseudorange processing and Kalman filter inside the
/// unit. So — like the star tracker and the digital sun sensor — the model
/// reproduces the *specified output accuracy* rather than inventing pseudoranges
/// and a constellation and then undoing them with an algorithm that is not the
/// vendor's. The pseudorange/constellation-geometry path (DOP, per-satellite
/// error) belongs with the onboard OD work (§8.3, Phase 6) and needs committed
/// GNSS ephemeris; this Phase-2 model is the fix the receiver hands the FSW.
///
/// **Frames.** A real receiver outputs ECEF/GPS-time, so the model takes the truth
/// state already in ECEF (the caller rotates ECI→ECEF through `frames::eci_ecef`,
/// the one reduction in the repo, REQ-CONV-002) and stamps GPS time. The FSW's
/// job on ingest is the reverse — ECEF/GPS → ECI/TAI before the inertial filter
/// runs (REQ-CONV-001) — which is exactly why the truth model must present the
/// awkward frame rather than a convenient ECI one.
///
/// **Error stack.** Per-axis white Gaussian position error (split horizontal vs
/// vertical in the local geodetic frame, because datasheets and geometry both do),
/// per-axis white velocity error, and a receiver-clock bias on the time tag. This
/// is a **first-order model**: real GNSS position error is strongly correlated over
/// minutes (common-mode ionosphere/orbit/clock error across the visible
/// constellation), which white noise understates for an estimator that averages
/// successive fixes. The datasheet quotes only an RMS, so the model reproduces the
/// RMS; a correlated component is the upgrade path when OD fidelity needs it.
/// The realised σ is carried on every measurement so an estimator can be fed the
/// noise the fix actually had, not a headline.
///
/// **Timing.** The receiver produces fixes at a bounded rate (100 Hz on the
/// OEM7600); polling faster returns the previous fix with `fresh = false`. A
/// cold start withholds fixes for the time-to-first-fix; an outage withholds them
/// and, on recovery, imposes the reacquisition delay before valid fixes resume —
/// a receiver does not snap back the instant signal returns, and an FDIR test of
/// graceful coasting (§9.2) needs that delay to be real.
///
/// **Fault injection is first-class** (§9): loss of fix (outage), a spoofed
/// position offset that stays *valid* so the innovation checks must catch it, and
/// a clock jump — the GNSS FDIR failure modes of §9.2/§6.2.
///
/// Each instance owns an independent, seed-derived noise stream (§3.5), so it is
/// bit-reproducible from `{config, seed}` and adding another sensor does not
/// perturb it.

#include <cstdint>
#include <Eigen/Core>
#include <map>
#include <string>

#include "math/frames.hpp"
#include "math/typed_vector.hpp"
#include "random/rng.hpp"
#include "sensors/gnss_jamming.hpp"
#include "time/timescales.hpp"

namespace polaris::sim::sensors {

/// GNSS receiver performance spec, in SI. Built by `fromParams` from the
/// datasheet-native keys in `config/hardware/gnss/*.yaml`.
struct GnssSpec {
  /// Per-axis horizontal position error, 1σ [m]. Datasheets quote a 2D
  /// horizontal RMS (DRMS = √(σ_e²+σ_n²)); `fromParams` divides by √2 to get the
  /// per-axis σ used here.
  double position_sigma_h_m = 0.0;
  /// Vertical (up) position error, 1σ [m]. A single axis, so its RMS *is* the σ.
  /// Defaults to 1.5× the horizontal σ when the datasheet omits it — the usual
  /// VDOP/HDOP ratio, since vertical geometry is always poorer than horizontal.
  double position_sigma_v_m = 0.0;
  /// Per-axis velocity error, 1σ [m/s].
  double velocity_sigma_m_s = 0.0;
  /// Receiver-clock / time-tag error, 1σ [s].
  double time_sigma_s = 0.0;

  /// Minimum interval between genuinely new fixes [s] — the reciprocal of the
  /// maximum position rate. Reading faster repeats the last fix (`fresh=false`).
  double sample_period_s = 0.0;
  /// Native maximum position rate [Hz]; informational (the §2.4 buffer gates).
  double max_rate_hz = 0.0;

  /// **Fix latency** [s]: the interval between the epoch a solution is *valid
  /// at* and the epoch it reaches the FSW. A receiver does not publish its PVT
  /// instantly — it correlates, solves, formats and clocks the message out — and
  /// the bus and the scheduler add their own delay on top.
  ///
  /// This is a first-class error term, not a detail. At 7.6 km/s a fix delivered
  /// 50 ms late describes a position 380 m behind where the vehicle now is, which
  /// dwarfs the receiver's own ~1 m accuracy. Modelling it is what lets the
  /// onboard filter's latency correction (`lib/gnc/orbit_od.hpp`, "Fix latency")
  /// be tested against the thing it exists for; with the tag stamped at the
  /// sample epoch, as this model originally did, the error is invisible and the
  /// correction untestable.
  ///
  /// Realised as a delay line: `sample()` returns the **newest** buffered fix at
  /// least this old, tagged at **its own** measurement epoch. Zero delivers each
  /// fix immediately, which is the old behaviour.
  ///
  /// "Newest" is the rule, not "oldest": when more than one solution has come due
  /// since the last poll, the older ones are superseded and discarded rather than
  /// queued up to be handed over late. That is what a receiver's output register
  /// does, and it is what a consumer wants — a stale fix carries strictly less
  /// information than the fresh one behind it, and delivering both would hand the
  /// filter two measurements it must then order and de-duplicate. A caller
  /// polling at or above the fix rate sees every solution; one polling slower
  /// sees the latest, which is the correct answer to the question it asked.
  ///
  /// **Caution: a caller polling more slowly than the latency gets a whole poll
  /// of delay, not one latency.** The delay line can only deliver what it has
  /// been given, so if `sample()` is called every 10 s with a 0.05 s latency,
  /// the newest solution that is at least 0.05 s old is the one from the
  /// *previous* call. The model is behaving correctly — it was never handed the
  /// intermediate solutions a real 100 Hz receiver would have produced — but the
  /// realised latency is the poll period. Either poll at something approaching
  /// the fix rate, or set this to zero and verify the latency somewhere that
  /// can resolve it — never leave a datasheet value armed at a cadence that
  /// cannot see it. `tests/mc/orbit_od_mc.cpp` does both: its long arcs poll
  /// every 10 s and pass zero, and its `latency_fast` scenario polls at 50 Hz
  /// with the real value. Flying the datasheet latency on the 10 s arcs was
  /// measured as a constant 76.7 km along-track offset — a 10 s delay, tracked
  /// perfectly by a filter that was never wrong about anything except which
  /// epoch it was answering for. That artifact is what made this trap visible.
  double fix_latency_s = 0.0;

  /// **Latency jitter** [s, 1σ]: each fix's delivery delay is
  /// `max(0, fix_latency_s + N(0, jitter))` — the bus and scheduler do not
  /// deliver every message the same number of milliseconds late (Ceresoli et
  /// al. 2025 measured 15 ± 7.5 ms on CubeSat buses). The fix's **time tag is
  /// unaffected**: it is the measurement epoch, which is what makes the
  /// onboard latency correction exact per fix rather than a mean-delay guess.
  /// What jitter changes is *when* a fix comes due, so a poll can see one fix
  /// early and the next late; the flight side must be indifferent to that.
  /// Zero (default) is the fixed-latency delay line.
  double fix_latency_jitter_s = 0.0;

  /// Time-to-first-fix from a cold start [s] — the receiver is invalid for this
  /// long after its first sample.
  double cold_start_s = 0.0;
  /// Time-to-first-fix from a hot start [s]; informational (a warm reboot is not
  /// modelled at the sensor level, but the figure is carried for the scenario).
  double hot_start_s = 0.0;
  /// Delay before valid fixes resume after an outage clears [s].
  double reacquisition_s = 0.0;

  /// Master noise switch (a scenario knob, not a datasheet value). When false the
  /// receiver reports truth-exact position/velocity/time — for bring-up and debug
  /// runs where GNSS error would only obscure what is being tested. The
  /// datasheet σ are still carried on the measurement; they are simply not drawn.
  bool noise_enabled = true;

  /// Build a spec from datasheet-native params. Missing keys default to 0 (that
  /// term disabled). See gnss.cpp for the key list.
  static GnssSpec fromParams(const std::map<std::string, double>& params);
};

/// Everything one fix needs from the truth state, in the frame the receiver
/// reports in. The caller converts the canonical ECI truth state to ECEF through
/// `frames::eci_ecef` (§3.1) — there is one ECI↔ECEF reduction in the repo.
struct GnssInput {
  math::Vec3<math::frames::ECEF> position_m{};
  math::Vec3<math::frames::ECEF> velocity_m_s{};
};

/// One GNSS fix.
struct GnssMeasurement {
  math::Vec3<math::frames::ECEF> position_m{};
  math::Vec3<math::frames::ECEF> velocity_m_s{};
  /// Fix time tag in **GPS time**, including the receiver-clock bias — the
  /// receiver's own estimate of the epoch, which is what it stamps.
  time::Gps time_tag{};
  /// The clock bias in the time tag [s], carried separately so an OD filter that
  /// estimates receiver clock can be scored against the truth it was given.
  double clock_bias_s = 0.0;

  /// Realised 1σ error the fix was drawn with, carried so an estimator can be fed
  /// the noise it actually had. Horizontal is per-axis; vertical and velocity as
  /// in the spec.
  double position_sigma_h_m = 0.0;
  double position_sigma_v_m = 0.0;
  double velocity_sigma_m_s = 0.0;
  double time_sigma_s = 0.0;

  /// False when the sample period had not elapsed: this fix is the previous one
  /// repeated. An estimator treating repeated fixes as independent grows
  /// overconfident, so it is flagged.
  bool fresh = true;
  /// False during a cold-start acquisition, an outage, jamming, or the
  /// reacquisition delay after one. Downstream consumers must respect this
  /// (§9.1); a spoofed fix is deliberately *valid* — catching it is the
  /// innovation check's job, not a flag.
  bool valid = false;
  /// True when the sub-satellite point is inside a config-defined jamming region
  /// (§9.2). Separated from `valid` so telemetry can tell a geographic jam from a
  /// commanded outage or a cold start.
  bool jammed = false;
  /// The name of the jamming region, when `jammed`; empty otherwise.
  std::string jamming_region;
};

/// A GNSS receiver. Construct with its spec and a per-source stream id under the
/// run's master seed.
///
/// **PVT error model.** Position error is white Gaussian, split horizontal vs
/// vertical in the local geodetic east/north/up basis \f$(\hat{e},\hat{n},\hat{u})\f$
/// at the truth ECEF position, plus the spoof offset; velocity error is per-axis
/// white in ECEF:
/// \f[
///   \tilde{r} = r + \sigma_h\,g_e\,\hat{e} + \sigma_h\,g_n\,\hat{n} + \sigma_v\,g_u\,\hat{u} +
///   \Delta r^{\mathrm{flt}}, \qquad \tilde{v} = v + \sigma_v^{\!vel}\,(g_x,\,g_y,\,g_z)^\top,
/// \f]
/// with per-axis \f$\sigma_h\f$ (`position_sigma_h_m`, = DRMS\f$/\sqrt2\f$),
/// vertical \f$\sigma_v\f$ (`position_sigma_v_m`), velocity
/// \f$\sigma_v^{\!vel}\f$ (`velocity_sigma_m_s`), each \f$g\sim\mathcal{N}(0,1)\f$.
/// The receiver-clock bias biases the GPS time tag:
/// \f[
///   b = \sigma_t\,g_t + \Delta t^{\mathrm{flt}}, \qquad
///   \tilde{t} = t_{\mathrm{gps}} + b,
/// \f]
/// \f$\sigma_t\f$ = `time_sigma_s`. With `noise_enabled = false` every
/// \f$\sigma\f$-scaled draw is skipped (truth-exact PVT), but the spoof offset
/// \f$\Delta r^{\mathrm{flt}}\f$ and clock jump \f$\Delta t^{\mathrm{flt}}\f$ still
/// apply — they are faults, not measurement noise.
///
/// **Validity decision.** A fix is reported `valid` only when
/// \f[
///   \lnot\,\text{outage} \ \land\ t_{\mathrm{gps}} \ge t_{\mathrm{valid}},
/// \f]
/// where \f$\text{outage} = \text{fault\_outage} \lor (\text{sub-satellite point}
/// \in \text{jamming region})\f$, and \f$t_{\mathrm{valid}}\f$ is set on the first
/// sample to \f$t_{\mathrm{gps}} + t_{\mathrm{cold}}\f$ (`cold_start_s`) and re-armed
/// on each outage falling edge to \f$t_{\mathrm{gps}} + t_{\mathrm{reacq}}\f$
/// (`reacquisition_s`). A spoofed fix stays `valid` by design. A poll within one
/// `sample_period_s` of the last genuine fix repeats it with `fresh = false`.
class Gnss {
 public:
  /// The spec this unit was built from (fix rate drives the §2.4 loop).
  const GnssSpec& spec() const { return spec_; }

  Gnss(const GnssSpec& spec, std::uint64_t master_seed, std::uint64_t stream_id)
      : spec_(spec), rng_(random::streamRng(master_seed, stream_id)) {}

  /// Produce the fix delivered at truth time @p epoch for truth state @p input.
  ///
  /// With @ref GnssSpec::fix_latency_s set, the returned fix is **not** the
  /// solution for @p input: it is an earlier solution, tagged at its own
  /// measurement epoch, that has now finished making its way out of the
  /// receiver. Until the first one has, the fix is invalid — a receiver that has
  /// not yet published anything has nothing to report, and that is the same
  /// state a cold start leaves it in.
  GnssMeasurement sample(const time::Tai& epoch, const GnssInput& input);

  /// Pending fixes discarded because the delay line was full. Non-zero means the
  /// configured latency and rate together overrun @ref kMaxPending; a scenario
  /// asserting graceful behaviour should assert this is zero.
  std::uint64_t pendingDropped() const { return pending_dropped_; }

  // --- Fault injection (§9) --------------------------------------------------

  /// Loss of fix: every subsequent sample is invalid until cleared, and on
  /// recovery the reacquisition delay applies before valid fixes resume.
  void setOutage(bool out) { fault_outage_ = out; }

  /// Spoofing / meaconing: add a persistent ECEF position offset [m] to the
  /// reported fix, which stays **valid** — the point of a spoof is that it looks
  /// like a real measurement, so FDIR must reject it on innovation/consistency
  /// (§9.2), not on a flag. Replaces any previous offset (does not accumulate).
  void injectPositionOffset(const math::Vec3<math::frames::ECEF>& delta_m) {
    fault_pos_offset_ = delta_m.eigen();
  }

  /// A step in the receiver clock [s], added to the time-tag bias until cleared.
  void injectClockJump(double delta_s) { fault_clock_jump_s_ = delta_s; }

  /// Clear all injected faults, returning the receiver to nominal.
  void clearFaults() {
    fault_outage_ = false;
    fault_pos_offset_.setZero();
    fault_clock_jump_s_ = 0.0;
  }

  /// Bind the config-defined jamming map (§9.2). Not owned — the caller keeps it
  /// alive for the run. Passing nullptr (the default) disables geographic
  /// jamming. When the sub-satellite point is inside a region the fix goes
  /// invalid, and leaving it imposes the reacquisition delay, exactly like an
  /// outage — a jammed receiver does not reacquire the instant it clears the zone.
  void setJammingRegions(const JammingRegions* regions) { jamming_ = regions; }

 private:
  GnssSpec spec_;
  random::SplitMix64 rng_;

  bool has_sampled_ = false;
  bool prev_outage_ = false;
  time::Gps last_fix_time_{};
  GnssMeasurement last_fix_{};

  /// Delay line for @ref GnssSpec::fix_latency_s. A fixed-capacity ring: the
  /// entries are solutions already computed against truth at their own epoch and
  /// waiting to be delivered. Sized so the deepest configured latency at the
  /// fastest configured rate still fits — 100 Hz against a 0.25 s latency is 25
  /// entries — with headroom. Overrunning it drops the *oldest* pending fix,
  /// which is the honest failure (a receiver that far behind has lost the fix),
  /// and `pendingDropped()` counts it so a mis-sized buffer cannot hide.
  static constexpr std::size_t kMaxPending = 64;
  GnssMeasurement pending_[kMaxPending]{};
  std::int64_t pending_epoch_ns_[kMaxPending]{};  ///< GPS ns the fix comes due (measured + latency)
  std::size_t pending_count_ = 0;
  std::uint64_t pending_dropped_ = 0;
  /// GPS-time nanoseconds before which fixes stay invalid (cold-start acquisition
  /// or post-outage reacquisition). Set on the first sample and on each outage
  /// falling edge.
  std::int64_t valid_from_gps_ns_ = 0;

  Eigen::Vector3d fault_pos_offset_ = Eigen::Vector3d::Zero();
  double fault_clock_jump_s_ = 0.0;
  bool fault_outage_ = false;
  const JammingRegions* jamming_ = nullptr;
};

}  // namespace polaris::sim::sensors

#endif  // POLARIS_SIM_SENSORS_GNSS_HPP
