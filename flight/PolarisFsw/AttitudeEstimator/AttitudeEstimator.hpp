// ======================================================================
// \title  AttitudeEstimator.hpp
// \brief  Attitude estimator component: coarse chain + MEKF fine mode with
//         arbitration (§8.1, §10; REQ-ADET-002, REQ-ADET-003, REQ-ADET-004)
//
// The F´ wrapper around polaris::gnc::CoarseAttitudeEstimator and
// polaris::gnc::Mekf: gathers sun, magnetometer, gyro and GNSS measurements off
// its port arrays, builds the inertial references (Sun from OnboardTables,
// geomagnetic field from the onboard IGRF-14 snapshot), runs one coarse cycle
// per rate-group call, arbitrates fine vs coarse, and publishes whichever
// solution is active plus health telemetry. No estimation math and no I/O live
// here. Flight rules — no heap after init, no exceptions, fixed-size storage,
// every return code checked.
//
// The coarse chain runs every cycle regardless of mode, so the fallback under a
// demotion is a live solution rather than one that has to re-acquire from cold.
// ======================================================================

#ifndef FLIGHT_POLARISFSW_ATTITUDEESTIMATOR_HPP
#define FLIGHT_POLARISFSW_ATTITUDEESTIMATOR_HPP

#include <cstdint>

#include "environment/igrf.hpp"
#include "flight/PolarisFsw/AttitudeEstimator/AttitudeEstimatorComponentAc.hpp"
#include "gnc/coarse_attitude.hpp"
#include "gnc/davenport.hpp"
#include "gnc/mekf.hpp"
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

  //! Count one star-tracker solution. Not fused yet — the §8.2 layer feeds it to
  //! the MEKF; the coarse chain must stay tracker-independent regardless, to
  //! remain the Safe-mode floor.
  void starTrackerIn_handler(FwIndexType portNum, const StarTrackerMeas& meas) override;

  // ----------------------------------------------------------------------
  // Command and parameter handlers
  // ----------------------------------------------------------------------

  //! RESET_ESTIMATOR: drop both solutions and re-acquire from cold — the next
  //! TRIAD for coarse, a fresh Davenport seed for fine.
  void RESET_ESTIMATOR_cmdHandler(FwOpcodeType opCode, U32 cmdSeq) override;

  //! A parameter changed: re-read the whole set on the next cycle.
  void parameterUpdated(FwPrmIdType id) override;

  // ----------------------------------------------------------------------
  // Helpers
  // ----------------------------------------------------------------------

  //! Read the coarse-chain parameters from ParameterDb into a config and
  //! rebuild the estimator with it. Emits ConfigInvalid (edge-gated) and leaves
  //! the estimator unconfigured if any parameter is missing or out of range —
  //! there are no flight defaults (§19.3). Returns true when configured.
  bool refreshCoarseConfig();

  //! Read the fine-mode parameters and rebuild the MEKF with them. Independent
  //! of refreshCoarseConfig(): a missing fine parameter costs the fine mode
  //! (FineConfigInvalid, coarse-only operation), not the whole estimator.
  //! Returns true when the filter is configured.
  bool refreshFineConfig();

  //! Emit ConfigInvalid(@p detail) if not already flagged, and leave the
  //! estimator inert. Never substitutes a value (§19.3).
  void failConfig(const char* detail);

  //! Emit FineConfigInvalid(@p detail) if not already flagged, demote fine mode
  //! if it was engaged, and leave the MEKF inert. The coarse chain is untouched.
  void failFineConfig(const char* detail);

  //! First valid, fresh unit of each type on its port array, or nullptr. "Fresh"
  //! is |now - timeTag| <= MaxMeasAgeSec (§9.1 staleness gate). Bounded loop
  //! over the array; index order is vehicle build order, so this is a
  //! deterministic priority, not a fusion — §8.2 owns fusion.
  const ImuMeas* selectImu(I64 nowTaiNs) const;
  const SunSensorMeas* selectSunSensor(I64 nowTaiNs) const;
  const MagnetometerMeas* selectMagnetometer(I64 nowTaiNs) const;
  const GnssMeas* selectGnss(I64 nowTaiNs) const;

  //! One cycle of fine-mode arbitration, after the coarse cycle has run and
  //! with the same measurement set. Engages, steps, or demotes fine mode; never
  //! touches the coarse solution. @p coarse is the cycle's coarse product, whose
  //! validity gates a promotion. Returns the cycle's NIS for telemetry, or NaN
  //! when no update ran — so the channel is written exactly once per cycle.
  double arbitrateFineMode(const polaris::time::Tai& epoch,
                           const polaris::gnc::CoarseAttitudeInput& in,
                           const polaris::gnc::CoarseAttitudeOutput& coarse);

  //! Propagate and update the engaged filter on this cycle's measurements,
  //! maintaining the refusal and NIS-rejection streaks and demoting when a
  //! streak, the fine coast horizon, or an internal fault says to. Returns the
  //! largest NIS seen this cycle (NaN if no update ran).
  double stepFineMode(const polaris::time::Tai& epoch, const polaris::gnc::CoarseAttitudeInput& in);

  //! Try to seed the MEKF from a Davenport solve over this cycle's vector pairs.
  //! Emits FineModeEngaged on success, edge-gated FineInitFailed on refusal.
  void tryPromoteFineMode(const polaris::time::Tai& epoch,
                          const polaris::gnc::CoarseAttitudeInput& in);

  //! Give up the fine solution for @p reason: emit FineModeDemoted, drop the
  //! filter state (a solution no longer trusted is not worth carrying), and
  //! clear the streaks. No-op when fine mode is not engaged.
  void demoteFineMode(FineDemotionReason::T reason);

  //! Publish the coarse product on estimateOut (if connected) and write the
  //! solution telemetry channels, stamping the product with @p epoch.
  void publishCoarse(const polaris::gnc::CoarseAttitudeOutput& out,
                     const polaris::time::Tai& epoch);

  //! Publish the fine (MEKF) product the same way.
  void publishFine(const polaris::time::Tai& epoch);

  //! Fill and emit the AttitudeEstimate port struct and the solution telemetry
  //! from `state_`, which the caller has already written from its estimator.
  //! @p cov is the attitude-error covariance to report [rad²] and @p age_s the
  //! solution age; both belong to the active solution.
  void emitEstimate(const Eigen::Matrix3d& cov, double age_s);

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

  //! The fine estimator. Starts inert on the same rule, and is rebuilt by
  //! refreshFineConfig(); an inert filter simply means coarse-only operation.
  polaris::gnc::Mekf mekf_;

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

  //! Per-source 1-sigma handed to the MEKF and the Davenport seed [rad]: the
  //! white and systematic parts of the same budget, root-sum-squared. The filter
  //! treats R as white, so the systematic part has to be inflated into sigma or
  //! the covariance it reports converges below the true error (gnc/mekf.hpp).
  F64 sigma_sun_total_rad_{0.0};
  F64 sigma_mag_total_rad_{0.0};

  //! Fine-mode tuning cached from the parameter set alongside the MekfConfig.
  F64 seed_min_observability_{0.0};
  F64 bias_sigma_init_{0.0};
  U32 refusal_streak_limit_{0};
  U32 nis_streak_limit_{0};

  U32 triad_accepted_{0};
  U32 triad_rejected_{0};
  U32 cycles_refused_{0};
  U32 star_tracker_count_{0};
  U32 fine_demotions_{0};

  //! NIS-gate rejections accumulated over filters this component has already
  //! given up on. `Mekf::reset` clears the filter's own count by contract (a
  //! commanded reset means "start over"), which would erase the FDIR signal at
  //! the exact moment a NIS_STREAK or FILTER_FAULT demotion created it. Only
  //! RESET_ESTIMATOR clears this; telemetry reports it plus the live count.
  U32 fine_rejected_total_{0};

  //! Consecutive cycles the filter refused a call / the NIS gate rejected a
  //! measurement. Cleared by a clean cycle and by a demotion.
  U32 refusal_streak_{0};
  U32 nis_streak_{0};

  I64 last_epoch_tai_ns_{0};
  bool have_epoch_{false};

  //! Set at construction and by parameterUpdated(): the next cycle re-reads the
  //! whole parameter set before estimating.
  bool params_dirty_{true};

  //! Edge gates: each alert fires on the transition into its state, not once per
  //! cycle in it (the Push 38 TableDegraded pattern).
  bool attitude_valid_{false};
  bool config_invalid_flagged_{false};
  bool fine_config_invalid_flagged_{false};
  bool fine_init_failed_flagged_{false};
  bool position_unavailable_flagged_{false};
  bool igrf_stale_flagged_{false};
  TableGrade::T last_ephem_grade_{TableGrade::PRECISE};
  TableGrade::T last_eop_grade_{TableGrade::PRECISE};
  bool have_ephem_grade_{false};
  bool have_eop_grade_{false};

  //! Solution age [s] of the last cycle, so the AttitudeLost edge can report it
  //! from paths that have no estimator output in hand (config failure).
  double last_age_s_{0.0};

  //! Fine mode is engaged: the MEKF is seeded and its solution is what
  //! estimateOut carries. False means the coarse chain is the published product.
  bool fine_active_{false};
};

}  // namespace flight

#endif
