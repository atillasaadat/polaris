// ======================================================================
// \title  BurnExecutorTester.hpp
// \brief  Component unit tests for BurnExecutor (design doc §23.1)
// ======================================================================

#ifndef FLIGHT_POLARISFSW_BURNEXECUTOR_TESTER_HPP
#define FLIGHT_POLARISFSW_BURNEXECUTOR_TESTER_HPP

#include <Eigen/Core>

#include "BurnExecutorGTestBase.hpp"
#include "flight/PolarisFsw/BurnExecutor/BurnExecutor.hpp"
#include "math/quaternion.hpp"

namespace flight {

class BurnExecutorTester : public BurnExecutorGTestBase {
 public:
  static const U32 MAX_HISTORY_SIZE = 4000;
  static const FwEnumStoreType TEST_INSTANCE_ID = 0;

  BurnExecutorTester();
  ~BurnExecutorTester();

  //! No parameters: ConfigInvalid once, idle outputs (throttle 0, accel
  //! invalid) every cycle, BURN_START refused UNCONFIGURED.
  void testRefusesWithoutParameters();
  //! Duration, throttle, attitude and already-burning refusals, each by name.
  void testRefusalPaths();
  //! A burn at a known attitude publishes the hand-computed ECI acceleration
  //! and sigma, the throttle while burning and zero after, and completes with
  //! the accumulated delta-v; the mass estimate depletes at F/(Isp g0).
  void testBurnAccelerationAndDepletion();
  //! BURN_ABORT mid-burn: throttle off this cycle, ABORTED state, accel invalid.
  void testAbortMidBurn();
  //! The attitude going stale mid-burn aborts it with ATTITUDE.
  void testStaleAttitudeAbortsTheBurn();
  //! The -b hook fires the burn on the armed cycle.
  void testArmedBurnFiresOnItsCycle();

  void from_thrusterCmdOut_handler(FwIndexType portNum, const ThrusterThrottleSet& cmds) override;
  void from_accelOut_handler(FwIndexType portNum, const NonGravAccel& accel) override;

 private:
  void connectPorts();
  void initComponents();
  void setValidParameters();
  void setAttitude(const polaris::math::Quaternion& qBodyEci, I64 taiNs, bool valid = true);
  void runCycleAt(I64 taiNs);

  BurnExecutor component;
  ThrusterThrottleSet last_cmds_{};
  NonGravAccel last_accel_{};
  U32 accel_count_{0};
};

}  // namespace flight

#endif
