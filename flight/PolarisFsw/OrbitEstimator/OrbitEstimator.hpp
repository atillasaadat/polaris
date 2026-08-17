// ======================================================================
// \title  OrbitEstimator.hpp
// \brief  Onboard orbit determination component: the F´ seam around
//         polaris::gnc::OrbitOd (§8.3, §9.2; REQ-ODP-001, -005, -006, -009)
//
// Latches GNSS fixes from the receiver port array, runs the 6-state
// position/velocity filter on the 10 Hz GNC cycle — propagate to now, fold in
// the freshest fix, publish — and serves the orbit half of the §8.0 state to
// the attitude estimator and every other consumer on `orbitStateOut`. No filter
// math lives here.
//
// Push 71 adds the NESC navigation-filter usability practices at this seam
// (NASA/TP-2018-219822 Ch. 7 and Ch. 9; NESC TB 20-03 items d-g): re-tune on
// parameter upload without losing the solution, covariance re-initialisation
// on command, the per-measurement accept/inhibit/force policy, a propagated
// backup ephemeris with a restart command and a divergence comparator, and a
// per-cycle covariance definiteness check.
//
// Flight rules — no heap after init, no exceptions, fixed-size storage, every
// return code checked, finiteness guards on the published solution.
// ======================================================================

#ifndef FLIGHT_POLARISFSW_ORBITESTIMATOR_HPP
#define FLIGHT_POLARISFSW_ORBITESTIMATOR_HPP

#include <cstdint>

#include "flight/PolarisFsw/OrbitEstimator/OrbitEstimatorComponentAc.hpp"
#include "gnc/orbit_od.hpp"

namespace flight {

class OrbitEstimator final : public OrbitEstimatorComponentBase {
 public:
  using OdRefusal = OrbitEstimator_OdRefusal;

  explicit OrbitEstimator(const char* compName);
  ~OrbitEstimator() override = default;

  //! SITL/bench only: run OD_RESET's body on GNC cycle @p cycle (1-based;
  //! 0 = never). Exists so the reset-and-reseed path can be flown mid-run in a
  //! deployment with no ground link. It runs the command handler's own body
  //! from inside the guarded run cycle (the command port shares the mutex, so
  //! it cannot be dispatched from there) — only the uplink is skipped.
  void commandResetAtCycle(U32 cycle);

  //! SITL/bench only: ignore the non-gravitational acceleration input (§8.3)
  //! regardless of what arrives on accelIn, so the same burn can be flown with
  //! the filter told and blind and the difference measured. On is the flight
  //! behaviour; this only ever switches it off.
  void setAccelInputAtStartup(bool enable) { this->accel_input_enabled_ = enable; }

 private:
  // ----------------------------------------------------------------------
  // Port handlers
  // ----------------------------------------------------------------------

  //! Latch one fix for the next cycle and mark the slot fresh.
  void gnssIn_handler(FwIndexType portNum, const GnssMeas& meas) override;

  //! Latch the burn executor's non-gravitational acceleration for the next
  //! propagate.
  void accelIn_handler(FwIndexType portNum, const NonGravAccel& accel) override;

  //! One GNC cycle: propagate to now, ingest the freshest fix, publish.
  void run_handler(FwIndexType portNum, U32 context) override;

  //! Re-read the tuning and apply it to the **running** filter: the solution,
  //! its covariance and its age are kept (NESC TB 20-03 item g; TP §9.3). Only
  //! a set that fails validation leaves the old one in force.
  void parameterUpdated(FwPrmIdType id) override;

  // ----------------------------------------------------------------------
  // Commands
  // ----------------------------------------------------------------------

  void OD_RESET_cmdHandler(FwOpcodeType opCode, U32 cmdSeq) override;
  void OD_SEED_STATE_cmdHandler(FwOpcodeType opCode, U32 cmdSeq, I64 epochTaiNs, F64 posEciX,
                                F64 posEciY, F64 posEciZ, F64 velEciX, F64 velEciY, F64 velEciZ,
                                F64 posSigmaM, F64 velSigmaMps) override;
  void OD_REINIT_COV_cmdHandler(FwOpcodeType opCode, U32 cmdSeq, F64 posSigmaM,
                                F64 velSigmaMps) override;
  void OD_RESTART_FROM_BACKUP_cmdHandler(FwOpcodeType opCode, U32 cmdSeq) override;

  // ----------------------------------------------------------------------
  // Helpers
  // ----------------------------------------------------------------------

  //! Read every parameter; on a missing or invalid set, warn once and leave the
  //! filter as it was (inert if it never had a valid set — a running filter is
  //! never made inert by a bad upload). Returns true when a valid set is in
  //! force.
  bool applyParameters();

  //! Backup ephemeris (TP §9.2): propagate the copy alongside the solution and
  //! re-seed it from a FINE solution every BackupPeriodS. Called once per cycle
  //! after the solution's own propagate/ingest.
  void serviceBackup(I64 nowNs, const polaris::frames::EopValue* eopNow,
                     const polaris::gnc::NonGravAccelInput* accel);

  //! OD_RESTART_FROM_BACKUP's body: the backup replaces the solution.
  //! Returns false (with the EVR) when there is no backup.
  bool restartFromBackup();

  //! Look the EOP up at @p taiNs through OnboardTables. False (and an edge-gated
  //! EVR) when the port is unconnected or the epoch is uncovered.
  bool eopAt(I64 taiNs, polaris::frames::EopValue& out);

  //! Slot index of the freshest valid fix, or -1.
  FwIndexType selectFix() const;

  //! Record a refusal: counters, telemetry, edge-gated EVR.
  void noteRefusal(U8 unit, polaris::gnc::OrbitOdRefusal refusal);

  //! Publish `orbitStateOut` and the telemetry for this cycle.
  void publish(I64 nowNs);

  I64 currentTaiNs() const;

  // ----------------------------------------------------------------------
  // State — all touched under the component mutex (every port is guarded)
  // ----------------------------------------------------------------------

  //! Latched fixes and their fresh marks, one per receiver slot.
  GnssMeas gnss_[NUM_GNSSIN_INPUT_PORTS]{};
  bool gnss_fresh_[NUM_GNSSIN_INPUT_PORTS]{};

  //! The filter. Constructed inert (invalid config); brought up by the first
  //! valid parameter set and re-tuned in place by every later one.
  polaris::gnc::OrbitOd od_;
  //! The backup ephemeris: a copy of od_ taken from a FINE solution and then
  //! only ever propagated (TP §9.2 — "unaltered by measurement updates since
  //! initialization"). A plain value copy, which is why the library needs no
  //! notion of it.
  polaris::gnc::OrbitOd backup_;
  bool backup_valid_{false};
  I64 backup_seeded_ns_{0};   //!< when the backup was last seeded from od_
  F64 backup_period_s_{0.0};  //!< 0 = no backup kept
  //! Per-measurement editing policy (TP §9.1), from the parameters; the values
  //! last reported through MeasurementPolicyChanged.
  polaris::gnc::GnssMeasurementPolicy policy_{};
  bool policy_reported_{false};
  bool cov_indefinite_alerted_{false};
  bool configured_{false};
  bool tuning_alerted_{false};
  bool eop_alerted_{false};

  //! Horizons and windows as configured, for the EVRs and the seed gate.
  F64 max_coast_s_{0.0};
  F64 max_degraded_coast_s_{0.0};
  F64 max_accel_age_s_{0.0};
  F64 max_fix_latency_s_{0.0};
  U32 status_period_cycles_{0};
  //! Latched non-gravitational acceleration and the applied/cleared edge.
  NonGravAccel accel_{};
  bool accel_applied_{false};
  //! The quality published last cycle, for the degraded/dropped edges.
  polaris::gnc::OrbitOdQuality last_quality_{polaris::gnc::OrbitOdQuality::kNone};
  //! Epoch of the previous cycle, for the drop EVR's age figure.
  I64 last_run_ns_{0};

  //! Diagnostics of the last ingest, for telemetry.
  polaris::gnc::OrbitOdResult last_result_{};
  U32 fixes_accepted_{0};
  U32 fixes_refused_{0};
  //! OD_RESET's body: drop the solution and the counters, report it.
  void resetFilter();

  bool accel_input_enabled_{true};  //!< SITL/bench override (setAccelInputAtStartup)
  U32 cycle_{0};                    //!< GNC cycles run so far (for the armed reset)
  U32 reset_at_cycle_{0};           //!< 0 = no reset armed
  polaris::gnc::OrbitOdRefusal last_refusal_{polaris::gnc::OrbitOdRefusal::kNone};
  polaris::gnc::OrbitOdRefusal last_alerted_refusal_{polaris::gnc::OrbitOdRefusal::kNone};
};

}  // namespace flight

#endif
