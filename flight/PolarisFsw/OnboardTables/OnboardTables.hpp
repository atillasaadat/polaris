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

  //! Current master-clock epoch as TAI nanoseconds (Fw::Time from timeCaller).
  I64 currentTaiNs();

  // ----------------------------------------------------------------------
  // State
  // ----------------------------------------------------------------------

  polaris::onboard::TableStore store_;
  char eop_path_[kMaxPathLength]{};
  char ephem_path_[kMaxPathLength]{};
  U32 reload_count_{0};
  bool coverage_warned_{false};  //!< de-bounce the once-per-cycle coverage EVR
};

}  // namespace flight

#endif
