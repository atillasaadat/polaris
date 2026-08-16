// ======================================================================
// \title  SitlTime.cpp
// \brief  FSW time source: sim time under SITL, wall clock otherwise (§2.4, §3.2)
// ======================================================================

#include "flight/PolarisFsw/SitlTime/SitlTime.hpp"

#include <chrono>

#include "Fw/Types/Assert.hpp"

namespace flight {

namespace {
constexpr std::int64_t kNsPerSecond = 1000000000LL;
constexpr std::int64_t kNsPerMicrosecond = 1000LL;
}  // namespace

// ----------------------------------------------------------------------
// Component construction and destruction
// ----------------------------------------------------------------------

SitlTime ::SitlTime(const char* const compName) : SitlTimeComponentBase(compName) {}

SitlTime ::~SitlTime() {}

void SitlTime ::setSitlActive() {
  this->sitl_active_.store(true, std::memory_order_relaxed);
}

// ----------------------------------------------------------------------
// Handler implementations for typed input ports
// ----------------------------------------------------------------------

void SitlTime ::timeSetIn_handler(FwIndexType portNum, I64 epochTaiNs) {
  static_cast<void>(portNum);
  this->sim_ns_.store(epochTaiNs, std::memory_order_relaxed);
}

void SitlTime ::timeGetPort_handler(FwIndexType portNum, Fw::Time& time) {
  static_cast<void>(portNum);
  if (this->sitl_active_.load(std::memory_order_relaxed)) {
    // Sim time: the last pushed macro-step epoch, TAI. The onboard master clock
    // is TAI (§3.2); F´ has no TAI TimeBase, so it rides as spacecraft time.
    const std::int64_t ns = this->sim_ns_.load(std::memory_order_relaxed);
    FW_ASSERT(ns >= 0, static_cast<FwAssertArgType>(ns));
    time.set(TimeBase::TB_SC_TIME, static_cast<U32>(ns / kNsPerSecond),
             static_cast<U32>((ns % kNsPerSecond) / kNsPerMicrosecond));
    return;
  }
  // Wall clock — identical to the stock Svc::ChronoTime this component replaces.
  const auto now = std::chrono::system_clock::now().time_since_epoch();
  time.set(TimeBase::TB_WORKSTATION_TIME,
           static_cast<U32>(std::chrono::duration_cast<std::chrono::seconds>(now).count()),
           static_cast<U32>(std::chrono::duration_cast<std::chrono::microseconds>(now).count() %
                            1000000));
}

}  // namespace flight
