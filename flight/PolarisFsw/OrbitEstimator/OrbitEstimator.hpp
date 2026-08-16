// ======================================================================
// \title  OrbitEstimator.hpp
// \brief  Onboard orbit determination component: the F´ seam around
//         polaris::gnc::OrbitOd (§8.3, §9.2; REQ-ODP-001, -005, -006)
//
// Latches GNSS fixes from the receiver port array, runs the 6-state
// position/velocity filter on the 10 Hz GNC cycle — propagate to now, fold in
// the freshest fix, publish — and serves the orbit half of the §8.0 state to
// the attitude estimator and every other consumer on `orbitStateOut`. No filter
// math lives here.
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

 private:
  // ----------------------------------------------------------------------
  // Port handlers
  // ----------------------------------------------------------------------

  //! Latch one fix for the next cycle and mark the slot fresh.
  void gnssIn_handler(FwIndexType portNum, const GnssMeas& meas) override;

  //! One GNC cycle: propagate to now, ingest the freshest fix, publish.
  void run_handler(FwIndexType portNum, U32 context) override;

  //! Re-read the tuning; the filter is rebuilt (and its solution dropped) on
  //! any change, since a covariance built under the old q_a means nothing under
  //! the new one.
  void parameterUpdated(FwPrmIdType id) override;

  // ----------------------------------------------------------------------
  // Commands
  // ----------------------------------------------------------------------

  void OD_RESET_cmdHandler(FwOpcodeType opCode, U32 cmdSeq) override;

  // ----------------------------------------------------------------------
  // Helpers
  // ----------------------------------------------------------------------

  //! Read every parameter; on a missing or invalid set, warn once and leave the
  //! filter inert. Returns true when a valid filter was (re)built.
  bool applyParameters();

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

  //! The filter. Constructed inert (invalid config) and rebuilt by
  //! applyParameters(); a class with a config-only constructor and no default
  //! is rebuilt by assignment, which is what OrbitOd supports.
  polaris::gnc::OrbitOd od_;
  bool configured_{false};
  bool tuning_alerted_{false};
  bool eop_alerted_{false};

  //! Coast horizon as configured, for the drop EVR.
  F64 max_coast_s_{0.0};
  //! Epoch of the previous cycle, for the drop EVR's age figure.
  I64 last_run_ns_{0};

  //! Diagnostics of the last ingest, for telemetry.
  polaris::gnc::OrbitOdResult last_result_{};
  U32 fixes_accepted_{0};
  U32 fixes_refused_{0};
  polaris::gnc::OrbitOdRefusal last_refusal_{polaris::gnc::OrbitOdRefusal::kNone};
  polaris::gnc::OrbitOdRefusal last_alerted_refusal_{polaris::gnc::OrbitOdRefusal::kNone};
};

}  // namespace flight

#endif
