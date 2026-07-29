// ======================================================================
// \title  ScriptedCmdSource.cpp
// \brief  Phase-4 placeholder actuator commander on the SITL rate group (§2.4)
// ======================================================================

#include "flight/PolarisFsw/ScriptedCmdSource/ScriptedCmdSource.hpp"

#include "sitl/scripted_profile.hpp"

namespace flight {

namespace {
constexpr I64 kNsPerSecond = 1000000000LL;
constexpr I64 kNsPerMicrosecond = 1000LL;
}  // namespace

// ----------------------------------------------------------------------
// Component construction and destruction
// ----------------------------------------------------------------------

ScriptedCmdSource ::ScriptedCmdSource(const char* const compName)
    : ScriptedCmdSourceComponentBase(compName) {}

ScriptedCmdSource ::~ScriptedCmdSource() {}

void ScriptedCmdSource ::setEnabled(bool enabled) {
  this->enabled_ = enabled;
  this->tlmWrite_Enabled(enabled);
  this->log_ACTIVITY_HI_Configured(enabled);
}

// ----------------------------------------------------------------------
// Handler implementations for typed input ports
// ----------------------------------------------------------------------

void ScriptedCmdSource ::run_handler(FwIndexType portNum, U32 context) {
  // Sim epoch (TAI ns) for this macro step. SitlTime serves whole-microsecond
  // sim time under SITL, so this reconstruction is exact for the 10 Hz grid; the
  // profile is a pure function of it, matching the sim-side reference bit for bit.
  const Fw::Time now = this->getTime();
  const I64 epoch_tai_ns = static_cast<I64>(now.getSeconds()) * kNsPerSecond +
                           static_cast<I64>(now.getUSeconds()) * kNsPerMicrosecond;

  WheelTorqueSet wheels;  // default-constructed: all zero (torque mode)
  MtqDipoleSet mtqs;      // default-constructed: all zero
  if (this->enabled_) {
    for (U32 i = 0; i < WheelTorqueSet::SIZE; ++i) {
      wheels[i] = polaris::sitl::scriptedWheelTorque(epoch_tai_ns, i);
    }
    for (U32 i = 0; i < MtqDipoleSet::SIZE; ++i) {
      double dipole[3];
      polaris::sitl::scriptedMtqDipole(epoch_tai_ns, i, dipole);
      mtqs[i][0] = dipole[0];
      mtqs[i][1] = dipole[1];
      mtqs[i][2] = dipole[2];
    }
  }

  this->wheelCmdOut_out(0, wheels);
  this->mtqCmdOut_out(0, mtqs);

  this->cycles_++;
  this->tlmWrite_CyclesRun(this->cycles_);
}

}  // namespace flight
