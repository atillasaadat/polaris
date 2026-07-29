// ======================================================================
// \title  ScriptedCmdSource.hpp
// \brief  Phase-4 placeholder actuator commander on the SITL rate group (§2.4)
//
// Stands in for the not-yet-built GNC control component: driven once per macro
// step by the SITL PassiveRateGroup, reads sim time from its time port, and
// emits a deterministic profile (lib/sitl/scripted_profile.hpp) to SitlBridge.
// Disabled (zero commands) unless setEnabled(true). Flight rules apply — no heap,
// no exceptions, fixed-size F´ array command types. Deleted when real GNC lands.
// ======================================================================

#ifndef FLIGHT_POLARISFSW_SCRIPTEDCMDSOURCE_HPP
#define FLIGHT_POLARISFSW_SCRIPTEDCMDSOURCE_HPP

#include "flight/PolarisFsw/ScriptedCmdSource/ScriptedCmdSourceComponentAc.hpp"

namespace flight {

class ScriptedCmdSource final : public ScriptedCmdSourceComponentBase {
 public:
  //! Construct ScriptedCmdSource object
  explicit ScriptedCmdSource(const char* const compName);

  //! Destroy ScriptedCmdSource object
  ~ScriptedCmdSource();

  //! Enable/disable the nonzero scripted profile (called at topology setup).
  //! Disabled → the source emits all-zero commands, so the zero-command SITL
  //! gate is unaffected. Emits the Configured event.
  void setEnabled(bool enabled);

 private:
  // ----------------------------------------------------------------------
  // Handler implementations for typed input ports
  // ----------------------------------------------------------------------

  //! Rate-group cycle: compute this step's commands from sim time and emit them.
  void run_handler(FwIndexType portNum, U32 context) override;

  // ----------------------------------------------------------------------
  // State
  // ----------------------------------------------------------------------

  bool enabled_ = false;  //!< Emit the nonzero profile? Set by setEnabled()
  U32 cycles_ = 0;        //!< Rate-group cycles executed (telemetry)
};

}  // namespace flight

#endif
