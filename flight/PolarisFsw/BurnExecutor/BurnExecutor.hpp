// ======================================================================
// \title  BurnExecutor.hpp
// \brief  Finite-burn executor: throttle out, commanded acceleration to the
//         orbit filter (design doc §17, §8.3; REQ-MAN-001)
//
// Flight rules — no heap after init, no exceptions, fixed-size storage, every
// return code checked, finiteness guards on the published acceleration.
// ======================================================================

#ifndef FLIGHT_POLARISFSW_BURNEXECUTOR_HPP
#define FLIGHT_POLARISFSW_BURNEXECUTOR_HPP

#include <Eigen/Core>

#include "flight/PolarisFsw/BurnExecutor/BurnExecutorComponentAc.hpp"

namespace flight {

class BurnExecutor final : public BurnExecutorComponentBase {
 public:
  using BurnState = BurnExecutor_BurnState;
  using BurnRefusal = BurnExecutor_BurnRefusal;

  explicit BurnExecutor(const char* compName);
  ~BurnExecutor() override = default;

  //! SITL/bench only: start a burn of @p durationS at @p throttle on GNC cycle
  //! @p cycle (1-based; 0 = never). Runs the command handler's body from inside
  //! the guarded run cycle (the command port shares the mutex) — only the
  //! uplink is skipped.
  void commandBurnAtCycle(U32 cycle, F64 durationS, F64 throttle);

 private:
  void run_handler(FwIndexType portNum, U32 context) override;
  void attitudeIn_handler(FwIndexType portNum, const AttitudeEstimate& estimate) override;
  void parameterUpdated(FwPrmIdType id) override;
  void BURN_START_cmdHandler(FwOpcodeType opCode, U32 cmdSeq, F64 durationS,
                             F64 throttleFrac) override;
  void BURN_ABORT_cmdHandler(FwOpcodeType opCode, U32 cmdSeq) override;

  bool applyParameters();
  I64 nowTaiNs() const;
  //! The command's body; false with the reason when refused.
  bool startBurn(F64 durationS, F64 throttle, BurnRefusal::T& reason);
  void endBurn(bool completed, BurnRefusal::T reason);
  bool attitudeUsable(I64 nowNs) const;
  void publishIdle(I64 nowNs);

  static constexpr U32 kMaxThrusters = 8;  // GncMaxUnits
  bool configured_{false};
  bool config_alerted_{false};
  U32 count_{0};
  Eigen::Vector3d axis_[kMaxThrusters]{};
  F64 thrust_n_[kMaxThrusters]{};
  F64 isp_s_[kMaxThrusters]{};
  F64 mass_kg_{0.0};
  F64 knowledge_frac_{0.0};
  F64 max_duration_s_{0.0};
  F64 max_att_age_s_{0.0};

  AttitudeEstimate attitude_{};
  bool attitude_seen_{false};
  BurnState::T state_{BurnState::IDLE};
  F64 throttle_{0.0};
  F64 remaining_s_{0.0};
  F64 delta_v_mps_{0.0};
  I64 last_run_ns_{0};

  U32 cycle_{0};
  U32 armed_cycle_{0};
  F64 armed_duration_s_{0.0};
  F64 armed_throttle_{0.0};
};

}  // namespace flight

#endif  // FLIGHT_POLARISFSW_BURNEXECUTOR_HPP
