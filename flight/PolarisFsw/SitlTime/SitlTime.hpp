// ======================================================================
// \title  SitlTime.hpp
// \brief  FSW time source: sim time under SITL, wall clock otherwise (§2.4, §3.2)
//
// The deployment's single time-get source. SITL off → workstation wall clock,
// byte-identical to the stock Svc::ChronoTime it replaces. SITL on → the sim
// epoch pushed by SitlBridge each macro step, so timestamps key off sim time and
// a run is bit-reproducible. Flight rules apply — no heap, no exceptions; the
// shared epoch is a lock-free atomic because the wall-clock rate groups read the
// time port on their own threads while SitlBridge writes it on the SITL task.
// ======================================================================

#ifndef FLIGHT_POLARISFSW_SITLTIME_HPP
#define FLIGHT_POLARISFSW_SITLTIME_HPP

#include <atomic>
#include <cstdint>

#include "flight/PolarisFsw/SitlTime/SitlTimeComponentAc.hpp"

namespace flight {

class SitlTime final : public SitlTimeComponentBase {
 public:
  //! Construct SitlTime object
  explicit SitlTime(const char* const compName);

  //! Destroy SitlTime object
  ~SitlTime();

  //! Switch the time source to sim time (called at topology setup when a SITL
  //! port is given). Until then, and always with SITL off, the wall clock is the
  //! source. One-way: a SITL run stays sim-timed for its life.
  void setSitlActive();

 private:
  // ----------------------------------------------------------------------
  // Handler implementations for typed input ports
  // ----------------------------------------------------------------------

  //! Latch the macro-step sim epoch (TAI ns) the time port returns while active.
  void timeSetIn_handler(FwIndexType portNum, I64 epochTaiNs) override;

  //! Time port: sim epoch (SITL active) or wall clock (otherwise) as Fw::Time.
  void timeGetPort_handler(FwIndexType portNum, Fw::Time& time) override;

  // ----------------------------------------------------------------------
  // State
  // ----------------------------------------------------------------------

  std::atomic<bool> sitl_active_{false};  //!< True after setSitlActive()
  std::atomic<std::int64_t> sim_ns_{0};   //!< Last pushed macro-step sim epoch (TAI ns)

  // The time port is called from flight rate-group tasks: a locking atomic
  // would be an unbounded blocking call in a flight path (JPL rule 3).
  static_assert(std::atomic<std::int64_t>::is_always_lock_free,
                "sim epoch atomic must be lock-free on the flight target");
};

}  // namespace flight

#endif
