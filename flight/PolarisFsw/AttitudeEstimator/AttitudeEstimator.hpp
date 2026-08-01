// ======================================================================
// \title  AttitudeEstimator.hpp
// \brief  Coarse attitude estimator component (§8.1, §10; REQ-ADET-002,
//         REQ-ADET-003, REQ-ADET-004)
//
// The F´ wrapper around polaris::gnc::CoarseAttitudeEstimator: gathers sun,
// magnetometer, gyro and GNSS measurements off its port arrays, builds the
// inertial references (Sun from OnboardTables, geomagnetic field from the
// onboard IGRF-14 snapshot), runs one estimation cycle per rate-group call, and
// publishes the §8.0 estimate plus health telemetry. No estimation math and no
// I/O live here. Flight rules — no heap after init, no exceptions, fixed-size
// storage, every return code checked.
// ======================================================================

#ifndef FLIGHT_POLARISFSW_ATTITUDEESTIMATOR_HPP
#define FLIGHT_POLARISFSW_ATTITUDEESTIMATOR_HPP

#include <cstdint>

#include "environment/igrf.hpp"
#include "flight/PolarisFsw/AttitudeEstimator/AttitudeEstimatorComponentAc.hpp"
#include "gnc/coarse_attitude.hpp"
#include "math/frames.hpp"
#include "math/typed_vector.hpp"
#include "state/estimated_state.hpp"
#include "time/leap_seconds.hpp"
#include "time/timescales.hpp"

namespace flight {

class AttitudeEstimator final : public AttitudeEstimatorComponentBase {
 public:
  //! Longest configurable IGRF coefficient path held (fixed, no heap).
  static constexpr FwSizeType kMaxPathLength = 256;

  //! Construct AttitudeEstimator object
  explicit AttitudeEstimator(const char* const compName);

  //! Destroy AttitudeEstimator object
  ~AttitudeEstimator();

  //! Load the onboard IGRF-14 snapshot from the verbatim IAGA coefficient file
  //! at @p igrfPath, for mission epoch @p decimalYear (called at topology setup,
  //! before the rate groups start).
  //!
  //! @p decimalYear selects the bracketing IAGA epoch pair the snapshot is
  //! collapsed from; the secular variation is then applied per cycle at that
  //! cycle's own epoch, so any value inside the same bracket yields identical
  //! field evaluations. It is deliberately **not** read from the master clock:
  //! at setup that clock has not been served a sim epoch yet (and on a cold
  //! hardware boot the RTC may not be set), which would silently snapshot the
  //! 1970 bracket and extrapolate it half a century. Which snapshot to hold is a
  //! ground decision, like the snapshot itself — see @ref kMaxIgrfEpochGapYears
  //! for what happens if the vehicle outlives it.
  //!
  //! Emits IgrfLoaded or IgrfLoadFailed. Returns true on success; on failure the
  //! estimator still runs and still publishes body rate (Safe-mode rate damping
  //! needs it) but can never form the magnetic pair, so it cannot acquire
  //! attitude.
  bool configureIgrf(const char* igrfPath, double decimalYear);

 private:
  // ----------------------------------------------------------------------
  // Handler implementations for typed input ports
  // ----------------------------------------------------------------------

  //! Rate-group entry: run one estimation cycle.
  void run_handler(FwIndexType portNum, U32 context) override;

  //! Latch one IMU's increments for the next cycle.
  void imuIn_handler(FwIndexType portNum, const ImuMeas& meas) override;

  //! Latch one sun sensor's unit vector for the next cycle.
  void sunSensorIn_handler(FwIndexType portNum, const SunSensorMeas& meas) override;

  //! Latch one magnetometer's field measurement for the next cycle.
  void magnetometerIn_handler(FwIndexType portNum, const MagnetometerMeas& meas) override;

  //! Latch one GNSS fix for the next cycle (position only is consumed).
  void gnssIn_handler(FwIndexType portNum, const GnssMeas& meas) override;

  //! Count one star-tracker solution. Not fused: the coarse mode must stay
  //! tracker-independent to remain the Safe-mode floor (fine mode is the MEKF).
  void starTrackerIn_handler(FwIndexType portNum, const StarTrackerMeas& meas) override;

  // ----------------------------------------------------------------------
  // Command and parameter handlers
  // ----------------------------------------------------------------------

  //! RESET_ESTIMATOR: drop the solution and re-acquire from the next TRIAD.
  void RESET_ESTIMATOR_cmdHandler(FwOpcodeType opCode, U32 cmdSeq) override;

  //! A parameter changed: re-read the whole set on the next cycle.
  void parameterUpdated(FwPrmIdType id) override;

  // ----------------------------------------------------------------------
  // Helpers
  // ----------------------------------------------------------------------

  //! Read every parameter from ParameterDb into a config and rebuild the
  //! estimator with it. Emits ConfigInvalid (edge-gated) and leaves the
  //! estimator unconfigured if any parameter is missing or out of range — there
  //! are no flight defaults (§19.3). Returns true when configured.
  bool refreshConfig();

  //! Emit ConfigInvalid(@p detail) if not already flagged, and leave the
  //! estimator inert. Never substitutes a value (§19.3).
  void failConfig(const char* detail);

  //! First valid, fresh unit of each type on its port array, or nullptr. "Fresh"
  //! is |now - timeTag| <= MaxMeasAgeSec (§9.1 staleness gate). Bounded loop
  //! over the array; index order is vehicle build order, so this is a
  //! deterministic priority, not a fusion — §8.2 owns fusion.
  const ImuMeas* selectImu(I64 nowTaiNs) const;
  const SunSensorMeas* selectSunSensor(I64 nowTaiNs) const;
  const MagnetometerMeas* selectMagnetometer(I64 nowTaiNs) const;
  const GnssMeas* selectGnss(I64 nowTaiNs) const;

  //! Publish the estimate on estimateOut (if connected) and write the solution
  //! telemetry channels, stamping the product with @p epoch.
  void publish(const polaris::gnc::CoarseAttitudeOutput& out, const polaris::time::Tai& epoch);

  //! Emit ReferenceDegraded/ReferenceRecovered for @p domain on a transition of
  //! its served grade, then latch @p grade into @p last. Per-domain so the alert
  //! names which reference degraded, following the OnboardTables pattern.
  void noteReferenceGrade(TableDomain::T domain, TableGrade::T grade, TableGrade::T& last,
                          bool& have);

  //! Emit the AttitudeLost edge if a valid solution is being given up, and clear
  //! the latch. Called from every path that drops the solution — coast expiry,
  //! configuration failure, and a tuning change that rebuilds the estimator —
  //! so a consumer sees the same edge however the attitude went away.
  void noteAttitudeLost();

  //! Current master-clock epoch as TAI nanoseconds.
  I64 currentTaiNs();

  // ----------------------------------------------------------------------
  // State
  // ----------------------------------------------------------------------

  //! The estimator proper. Starts inert (default config fails validation) and is
  //! rebuilt by refreshConfig() once every parameter is present and in range.
  polaris::gnc::CoarseAttitudeEstimator estimator_;

  //! Onboard IGRF-14 snapshot. Inert until configureIgrf() succeeds.
  polaris::environment::IgrfField igrf_;

  //! Base epoch [decimal years] of the loaded snapshot (telemetry/EVR context).
  double igrf_epoch_year_{0.0};

  //! Decimal year past which the loaded snapshot stops being the published
  //! model; the magnetic reference is refused beyond it.
  double igrf_valid_until_year_{0.0};

  //! In-code IERS leap-second record, for the TAI->decimal-year reduction the
  //! field model is parameterised by. Independent of any upload, like the
  //! OnboardTables ΔAT path (Push 38), so the coarse chain stays table-independent.
  polaris::time::LeapSecondTable leap_;

  //! Canonical onboard state (§8.0) written by gnc::writeToEstimatedState each
  //! cycle; the published port struct is filled from it.
  polaris::state::EstimatedState state_{};

  //! Latched measurements, one slot per port. Default-constructed slots have
  //! `valid == false`, so a never-connected unit is simply never selected.
  ImuMeas imu_[NUM_IMUIN_INPUT_PORTS]{};
  SunSensorMeas sun_[NUM_SUNSENSORIN_INPUT_PORTS]{};
  MagnetometerMeas mag_[NUM_MAGNETOMETERIN_INPUT_PORTS]{};
  GnssMeas gnss_[NUM_GNSSIN_INPUT_PORTS]{};

  char igrf_path_[kMaxPathLength]{};

  //! Staleness gate [s], cached from the parameter set with the rest of the config.
  F64 max_meas_age_s_{0.0};

  //! §9.1 range gate on a GNSS fix [m], geocentric radius. Zero until configured,
  //! which rejects every fix — the estimator is inert without tuning anyway.
  F64 min_position_radius_m_{0.0};
  F64 max_position_radius_m_{0.0};

  U32 triad_accepted_{0};
  U32 triad_rejected_{0};
  U32 cycles_refused_{0};
  U32 star_tracker_count_{0};

  I64 last_epoch_tai_ns_{0};
  bool have_epoch_{false};

  //! Set at construction and by parameterUpdated(): the next cycle re-reads the
  //! whole parameter set before estimating.
  bool params_dirty_{true};

  //! Edge gates: each alert fires on the transition into its state, not once per
  //! cycle in it (the Push 38 TableDegraded pattern).
  bool attitude_valid_{false};
  bool config_invalid_flagged_{false};
  bool position_unavailable_flagged_{false};
  bool igrf_stale_flagged_{false};
  TableGrade::T last_ephem_grade_{TableGrade::PRECISE};
  TableGrade::T last_eop_grade_{TableGrade::PRECISE};
  bool have_ephem_grade_{false};
  bool have_eop_grade_{false};

  //! Solution age [s] of the last cycle, so the AttitudeLost edge can report it
  //! from paths that have no estimator output in hand (config failure).
  double last_age_s_{0.0};
};

}  // namespace flight

#endif
