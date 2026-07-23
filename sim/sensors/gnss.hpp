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
class Gnss {
 public:
  Gnss(const GnssSpec& spec, std::uint64_t master_seed, std::uint64_t stream_id)
      : spec_(spec), rng_(random::streamRng(master_seed, stream_id)) {}

  /// Produce the fix for truth state @p input at truth time @p epoch.
  GnssMeasurement sample(const time::Tai& epoch, const GnssInput& input);

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
