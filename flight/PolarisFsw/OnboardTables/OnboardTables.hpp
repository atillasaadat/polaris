// ======================================================================
// \title  OnboardTables.hpp
// \brief  Onboard time/EOP/ephemeris table provider (§11.3, §22; REQ-CDH-002)
//
// Loads the leap-second, IERS EOP, and Sun/Moon Chebyshev tables at setup and
// serves point queries to the Phase-4 GNC stack. A thin F´ wrapper over the
// flight-safe lib holder (polaris::onboard::TableStore): the query ports are
// lock-free reads of the active double-buffer slot; RELOAD_TABLES restages and
// swaps on success. Flight rules — no heap after init, no exceptions.
// ======================================================================

#ifndef FLIGHT_POLARISFSW_ONBOARDTABLES_HPP
#define FLIGHT_POLARISFSW_ONBOARDTABLES_HPP

#include <atomic>
#include <cstdint>

#include "flight/PolarisFsw/OnboardTables/OnboardTablesComponentAc.hpp"
#include "onboard/tables.hpp"

namespace flight {

class OnboardTables final : public OnboardTablesComponentBase {
 public:
  //! Longest configurable table path held (fixed, no heap).
  static constexpr FwSizeType kMaxPathLength = 256;

  //! Construct OnboardTables object
  explicit OnboardTables(const char* const compName);

  //! Destroy OnboardTables object
  ~OnboardTables();

  //! Set the on-disk table paths and load them (called at topology setup). The
  //! same paths are reused by RELOAD_TABLES. Emits per-table load EVRs and
  //! writes the health telemetry. Returns true if every table loaded.
  bool configureAndLoad(const char* eopPath, const char* ephemPath);

 private:
  // ----------------------------------------------------------------------
  // Handler implementations for typed input ports
  // ----------------------------------------------------------------------

  //! EOP at a TAI epoch.
  bool getEopAt_handler(FwIndexType portNum, I64 taiNs, EopSample& sample) override;

  //! Geocentric ECI position [m] of a body at a TAI epoch.
  bool getBodyPosition_handler(FwIndexType portNum, const OnboardBody& body, I64 taiNs,
                               PosEciMeters& posEciM) override;

  //! TAI - UTC (delta-AT) [s] at a TAI epoch.
  bool getTaiUtcOffset_handler(FwIndexType portNum, I64 taiNs, I32& deltaAtSec) override;

  //! Rate-group entry: coverage-expiry check against the current time.
  void run_handler(FwIndexType portNum, U32 context) override;

  // ----------------------------------------------------------------------
  // Command handler implementations
  // ----------------------------------------------------------------------

  //! RELOAD_TABLES: restage and swap the tables from the configured paths.
  void RELOAD_TABLES_cmdHandler(FwOpcodeType opCode, U32 cmdSeq) override;

  // ----------------------------------------------------------------------
  // Helpers
  // ----------------------------------------------------------------------

  //! Load from the stored paths; emit EVRs and write telemetry. @p reload marks
  //! an operator reload (emits TablesReloaded) vs the setup-time load.
  bool load(bool reload);

  //! Write the health telemetry channels from the active tables.
  void writeTelemetry();

  //! Emit a degrade/recover EVR (and re-arm on recovery) when @p current differs
  //! from @p last for @p domain, then update @p last. Called from the watchdog.
  void noteGrade(TableDomain domain, polaris::onboard::Quality current,
                 std::atomic<polaris::onboard::Quality>& last);

  //! Current master-clock epoch as TAI nanoseconds (Fw::Time from timeCaller).
  I64 currentTaiNs();

  // ----------------------------------------------------------------------
  // State
  // ----------------------------------------------------------------------

  polaris::onboard::TableStore store_;
  char eop_path_[kMaxPathLength]{};
  char ephem_path_[kMaxPathLength]{};
  //! Atomics, not a guarded port: this passive component's handlers run on
  //! three threads (run on rateGroup3's, RELOAD_TABLES on the dispatcher's,
  //! the query ports on the caller's), and the table payload is already
  //! protected by the store's seqlock. Holding the component mutex across a
  //! RELOAD_TABLES parse would stall the rate group, so the residual scalar
  //! state is made atomic instead — the same lock-free intent the component
  //! was built around. (P46 lesson class: cross-thread scalars beside a
  //! correctly-guarded payload.)
  std::atomic<U32> reload_count_{0};
  std::atomic<bool> coverage_warned_{false};  //!< de-bounce the once-per-cycle coverage EVR
  //! Last served grade per domain (watchdog transition detection). Seeded to
  //! kPrecise so a coarse first cycle (tables missing at setup) flags each
  //! degraded domain exactly once, while a precise first cycle stays silent.
  //! TableDegraded is unthrottled — this edge-gating is the only rate limit,
  //! and it is per-domain, so both domains degrading each get their alert.
  std::atomic<polaris::onboard::Quality> last_eop_grade_{polaris::onboard::Quality::kPrecise};
  std::atomic<polaris::onboard::Quality> last_ephem_grade_{polaris::onboard::Quality::kPrecise};
};

}  // namespace flight

#endif
