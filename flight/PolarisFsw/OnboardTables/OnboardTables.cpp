// ======================================================================
// \title  OnboardTables.cpp
// \brief  Onboard time/EOP/ephemeris table provider (§11.3, §22; REQ-CDH-002)
// ======================================================================

#include "flight/PolarisFsw/OnboardTables/OnboardTables.hpp"

#include <cstring>

#include "Fw/Log/LogString.hpp"
#include "Fw/Types/Assert.hpp"

namespace flight {

namespace {
constexpr I64 kNsPerSecond = 1000000000LL;
constexpr I64 kNsPerMicrosecond = 1000LL;

//! Map the lib source-quality grade to the port/telemetry enum.
TableGrade toTableGrade(polaris::onboard::Quality q) {
  switch (q) {
    case polaris::onboard::Quality::kPrecise:
      return TableGrade::PRECISE;
    case polaris::onboard::Quality::kCoarse:
      return TableGrade::COARSE;
    default:
      return TableGrade::UNAVAILABLE;
  }
}
}  // namespace

// ----------------------------------------------------------------------
// Component construction and destruction
// ----------------------------------------------------------------------

OnboardTables ::OnboardTables(const char* const compName) : OnboardTablesComponentBase(compName) {}

OnboardTables ::~OnboardTables() {}

// ----------------------------------------------------------------------
// Configuration and loading
// ----------------------------------------------------------------------

bool OnboardTables ::configureAndLoad(const char* eopPath, const char* ephemPath) {
  FW_ASSERT(eopPath != nullptr);
  FW_ASSERT(ephemPath != nullptr);
  std::strncpy(this->eop_path_, eopPath, kMaxPathLength - 1);
  this->eop_path_[kMaxPathLength - 1] = '\0';
  std::strncpy(this->ephem_path_, ephemPath, kMaxPathLength - 1);
  this->ephem_path_[kMaxPathLength - 1] = '\0';
  return this->load(false);
}

bool OnboardTables ::load(bool reload) {
  polaris::onboard::LoadReport report;
  const bool ok = this->store_.load(this->eop_path_, this->ephem_path_, report);

  // Per-table EVRs: successes first, then the failure reason (if any).
  if (report.leap_ok) {
    this->log_ACTIVITY_HI_LeapTableLoaded(static_cast<U32>(report.leap_entries));
  }
  if (report.ephem_ok) {
    this->log_ACTIVITY_HI_EphemerisTableLoaded(
        static_cast<U32>(report.sun_segments), static_cast<U32>(report.moon_segments),
        report.ephem_span.start_tai_s, report.ephem_span.end_tai_s);
  }
  if (report.eop_ok) {
    this->log_ACTIVITY_HI_EopTableLoaded(static_cast<U32>(report.eop_entries),
                                         report.eop_span.start_tai_s, report.eop_span.end_tai_s);
  }
  if (!ok) {
    const Fw::LogStringArg reason(report.reason);
    this->log_WARNING_HI_TableLoadFailed(reason);
  } else if (reload) {
    ++this->reload_count_;
    this->log_ACTIVITY_HI_TablesReloaded(this->reload_count_);
    this->coverage_warned_ = false;  // new coverage: re-arm the expiry EVR
    this->log_WARNING_HI_CoverageExpiring_ThrottleClear();
  }

  this->writeTelemetry();
  return ok;
}

void OnboardTables ::writeTelemetry() {
  this->tlmWrite_TablesValid(this->store_.ready());
  this->tlmWrite_LeapEntries(static_cast<U32>(this->store_.leapEntries()));
  this->tlmWrite_EopEntries(static_cast<U32>(this->store_.eopEntries()));
  this->tlmWrite_SunSegments(static_cast<U32>(this->store_.sunSegments()));
  this->tlmWrite_MoonSegments(static_cast<U32>(this->store_.moonSegments()));
  const polaris::onboard::TableSpan eop = this->store_.eopSpan();
  const polaris::onboard::TableSpan eph = this->store_.ephemSpan();
  this->tlmWrite_EopStartTai(eop.start_tai_s);
  this->tlmWrite_EopEndTai(eop.end_tai_s);
  this->tlmWrite_EphStartTai(eph.start_tai_s);
  this->tlmWrite_EphEndTai(eph.end_tai_s);
  this->tlmWrite_ReloadCount(this->reload_count_);
}

// ----------------------------------------------------------------------
// Handler implementations for typed input ports
// ----------------------------------------------------------------------

bool OnboardTables ::getEopAt_handler(FwIndexType portNum, I64 taiNs, EopSample& sample) {
  static_cast<void>(portNum);
  polaris::frames::EopValue v;
  const polaris::onboard::Quality q = this->store_.eopAt(taiNs, v);
  if (q == polaris::onboard::Quality::kUnavailable) {
    return false;
  }
  sample = EopSample(v.ut1_minus_tai, v.xp_arcsec, v.yp_arcsec, toTableGrade(q));
  return true;
}

bool OnboardTables ::getBodyPosition_handler(FwIndexType portNum, const OnboardBody& body,
                                             I64 taiNs, PosEciMeters& posEciM) {
  static_cast<void>(portNum);
  const polaris::onboard::Body b =
      (body.e == OnboardBody::MOON) ? polaris::onboard::Body::Moon : polaris::onboard::Body::Sun;
  polaris::math::Vec3<polaris::math::frames::ECI> pos;
  const polaris::onboard::Quality q = this->store_.bodyPositionEci(b, taiNs, pos);
  if (q == polaris::onboard::Quality::kUnavailable) {
    return false;
  }
  posEciM = PosEciMeters(pos.x(), pos.y(), pos.z(), toTableGrade(q));
  return true;
}

bool OnboardTables ::getTaiUtcOffset_handler(FwIndexType portNum, I64 taiNs, I32& deltaAtSec) {
  static_cast<void>(portNum);
  std::int32_t delta = 0;
  const polaris::onboard::Quality q = this->store_.taiUtcOffset(taiNs, delta);
  if (q == polaris::onboard::Quality::kUnavailable) {
    return false;
  }
  deltaAtSec = static_cast<I32>(delta);
  return true;
}

void OnboardTables ::run_handler(FwIndexType portNum, U32 context) {
  static_cast<void>(portNum);
  static_cast<void>(context);
  using polaris::onboard::Quality;

  // Evaluate the grade actually being served at the current time, per domain.
  // Precise iff the uploaded table covers now; otherwise the coarse fallback is
  // what a query would return (analytic ephemeris / zero-EOP). This holds even
  // when nothing is loaded — coverageAt returns false and both grades are coarse.
  bool eop_ok = false;
  bool ephem_ok = false;
  const bool covered = this->store_.coverageAt(this->currentTaiNs(), eop_ok, ephem_ok);
  const Quality eop_grade = (covered && eop_ok) ? Quality::kPrecise : Quality::kCoarse;
  const Quality ephem_grade = (covered && ephem_ok) ? Quality::kPrecise : Quality::kCoarse;

  this->tlmWrite_EopGrade(toTableGrade(eop_grade));
  this->tlmWrite_EphemGrade(toTableGrade(ephem_grade));

  this->noteGrade(TableDomain::EOP, eop_grade, this->last_eop_grade_);
  this->noteGrade(TableDomain::EPHEMERIS, ephem_grade, this->last_ephem_grade_);

  // Coverage-expiring warning: only meaningful when tables are loaded (the not-
  // loaded case is already flagged by the load-failure EVR and by TableDegraded).
  if (this->store_.ready()) {
    const bool expiring = !eop_ok || !ephem_ok;
    if (expiring && !this->coverage_warned_) {
      // throttle 1 in the FPP also bounds this to a single downlink; the flag
      // stops the handler re-issuing every cycle until a reload re-arms it.
      this->log_WARNING_HI_CoverageExpiring(!eop_ok, !ephem_ok);
      this->coverage_warned_ = true;
    }
  }
}

void OnboardTables ::noteGrade(TableDomain domain, polaris::onboard::Quality current,
                               std::atomic<polaris::onboard::Quality>& last) {
  using polaris::onboard::Quality;
  // One load, compared and stored once: the watchdog is the only writer of
  // `last`, so this is not a CAS race — the atomic is for the readers on the
  // other threads, not for contention here.
  const Quality previous = last.load();
  if (current == previous) {
    return;
  }
  if (current == Quality::kCoarse && previous == Quality::kPrecise) {
    this->log_WARNING_HI_TableDegraded(domain, toTableGrade(current));
  } else if (current == Quality::kPrecise && previous == Quality::kCoarse) {
    // TableDegraded is unthrottled (edge-gated here per domain), so recovery
    // needs no throttle-clear; the next genuine degrade always emits.
    this->log_ACTIVITY_HI_TableRecovered(domain);
  }
  last.store(current);
}

// ----------------------------------------------------------------------
// Command handler implementations
// ----------------------------------------------------------------------

void OnboardTables ::RELOAD_TABLES_cmdHandler(FwOpcodeType opCode, U32 cmdSeq) {
  const bool ok = this->load(true);
  this->cmdResponse_out(opCode, cmdSeq,
                        ok ? Fw::CmdResponse::OK : Fw::CmdResponse::EXECUTION_ERROR);
}

// ----------------------------------------------------------------------
// Helpers
// ----------------------------------------------------------------------

I64 OnboardTables ::currentTaiNs() {
  const Fw::Time now = this->getTime();
  return static_cast<I64>(now.getSeconds()) * kNsPerSecond +
         static_cast<I64>(now.getUSeconds()) * kNsPerMicrosecond;
}

}  // namespace flight
