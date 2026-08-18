// ======================================================================
// \title  OrbitEstimatorTester.hpp
// \brief  Component unit tests for OrbitEstimator (design doc §23.1)
// ======================================================================

#ifndef FLIGHT_POLARISFSW_ORBITESTIMATOR_TESTER_HPP
#define FLIGHT_POLARISFSW_ORBITESTIMATOR_TESTER_HPP

#include <Eigen/Core>

#include "flight/PolarisFsw/OrbitEstimator/OrbitEstimator.hpp"
#include "gnc/orbit_od.hpp"
#include "OrbitEstimatorGTestBase.hpp"

namespace flight {

class OrbitEstimatorTester : public OrbitEstimatorGTestBase {
 public:
  static const U32 MAX_HISTORY_SIZE = 4000;
  static const FwEnumStoreType TEST_INSTANCE_ID = 0;

  OrbitEstimatorTester();
  ~OrbitEstimatorTester();

  // ----------------------------------------------------------------------
  // Tests
  // ----------------------------------------------------------------------

  //! No parameters: warns once, publishes an invalid solution every cycle,
  //! and a fix that arrived is not folded in (§19.3).
  void testRefusesWithoutParameters();

  //! The first valid fix seeds the filter (OrbitSeeded), the product goes
  //! valid at the fix's state, and 1 Hz fixes over 60 s hold the estimate on
  //! the truth with an accepted count that follows the fixes.
  void testSeedsAndTracks();

  //! With fixes withheld the solution stays valid on the propagated state up
  //! to MaxCoastS, then is dropped with OrbitSolutionDropped, and the next fix
  //! re-seeds whole.
  void testCoastsThenDropsAtHorizon();
  //! A fix inside the degraded band updates on the grown covariance: no re-seed.
  void testDegradedReacquiresByUpdateNotSeed();
  //! The burn executor's acceleration is propagated with only when valid and
  //! no older than MaxAccelAgeS; edges reported once.
  void testNonGravAccelIsAppliedOnlyWhenFreshAndValid();
  //! OD_SEED_STATE: refused outside the epoch window or the plausibility band,
  //! accepted otherwise, and the next fix updates rather than seeds.
  void testGroundSeedAcceptedAndRefused();

  //! A fix at an implausible radius is refused with FIX_IMPLAUSIBLE, the
  //! refusal is edge-gated, and the solution is untouched.
  void testRefusesImplausibleFix();

  //! An EOP query that cannot answer refuses the cycle (EopUnavailable, edge
  //! gated) and no solution is published valid.
  void testEopUnavailable();

  //! OD_RESET drops the solution and clears the counters; the next fix
  //! re-seeds.
  void testResetDropsSolution();

  //! NESC TB 20-03 item (g) / TP §9.3: a parameter upload re-tunes the running
  //! filter and keeps the solution; a bad upload leaves the last valid set in
  //! force.
  void testTuningUploadKeepsTheSolution();

  //! TB 20-03 items (d) and (f) / TP §9.1-9.2: OD_REINIT_COV re-opens the
  //! covariance around the same state; the measurement policy inhibits and
  //! forces, and says so in telemetry and EVRs.
  void testCovarianceReinitAndMeasurementPolicy();

  //! TB 20-03 item (e) / TP §9.2: the backup ephemeris is seeded from a FINE
  //! solution, propagated alongside it, and OD_RESTART_FROM_BACKUP brings a
  //! dropped solution back from it without an uplink.
  void testBackupEphemerisRestart();

  //! TP §2.1 / §2.2.3 (Push 73): the SMA and flight-path-angle sigmas are
  //! telemetered from the solution's covariance; the DMC states engage from
  //! parameters and report their estimate and sigma; a correlation time under
  //! ten sub-steps is refused as tuning.
  void testCovarianceMetricsAndDmcParameters();

  // ----------------------------------------------------------------------
  // Port handlers
  // ----------------------------------------------------------------------

  bool from_getEopAt_handler(FwIndexType portNum, I64 taiNs, EopSample& sample) override;
  void from_orbitStateOut_handler(FwIndexType portNum, const OrbitEstimate& estimate) override;

 private:
  void connectPorts();
  void initComponents();

  //! Load the reference tuning (the campaign's filterConfig()).
  void setValidParameters();

  //! Seed the truth propagator at the circular reference orbit.
  void startTruthAt(I64 taiNs);

  //! Truth state at @p taiNs, propagated on the same force model.
  void truthAt(I64 taiNs, Eigen::Vector3d& r_eci, Eigen::Vector3d& v_eci);

  //! Present the truth at @p taiNs as a receiver fix on unit 0 (ECEF, GPS time,
  //! the reference receiver's sigmas), optionally scaled in radius.
  void sendFixAt(I64 taiNs, double radiusScale = 1.0, bool velocityValid = true);

  //! Advance the component's clock and run one cycle.
  void runCycleAt(I64 taiNs);

  OrbitEstimator component;

  polaris::gnc::OrbitOd truth_;
  bool eop_available_{true};
  OrbitEstimate last_estimate_{};
  U32 estimate_count_{0};
  U32 cycles_run_{0};
};

}  // namespace flight

#endif
