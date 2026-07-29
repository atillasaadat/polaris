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
  polaris::frames::EopValue v;
  if (!this->store_.eopAt(taiNs, v)) {
    return false;
  }
  sample = EopSample(v.ut1_minus_tai, v.xp_arcsec, v.yp_arcsec);
  return true;
}

bool OnboardTables ::getBodyPosition_handler(FwIndexType portNum, const OnboardBody& body,
                                             I64 taiNs, PosEciMeters& posEciM) {
  const polaris::onboard::Body b =
      (body.e == OnboardBody::MOON) ? polaris::onboard::Body::Moon : polaris::onboard::Body::Sun;
  polaris::math::Vec3<polaris::math::frames::ECI> pos;
  if (!this->store_.bodyPositionEci(b, taiNs, pos)) {
    return false;
  }
  posEciM = PosEciMeters(pos.x(), pos.y(), pos.z());
  return true;
}

bool OnboardTables ::getTaiUtcOffset_handler(FwIndexType portNum, I64 taiNs, I32& deltaAtSec) {
  std::int32_t delta = 0;
  if (!this->store_.taiUtcOffset(taiNs, delta)) {
    return false;
  }
  deltaAtSec = static_cast<I32>(delta);
  return true;
}

void OnboardTables ::run_handler(FwIndexType portNum, U32 context) {
  if (!this->store_.ready()) {
    return;  // nothing loaded; the load-failure EVR already fired
  }
  bool eop_ok = false;
  bool ephem_ok = false;
  (void)this->store_.coverageAt(this->currentTaiNs(), eop_ok, ephem_ok);
  const bool expiring = !eop_ok || !ephem_ok;
  if (expiring && !this->coverage_warned_) {
    // throttle 1 in the FPP also bounds this to a single downlink; the flag
    // stops the handler re-issuing every cycle until a reload re-arms it.
    this->log_WARNING_HI_CoverageExpiring(!eop_ok, !ephem_ok);
    this->coverage_warned_ = true;
  }
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
