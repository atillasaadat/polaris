// ======================================================================
// \title  AttitudeEstimatorTester.hpp
// \brief  Test harness for the AttitudeEstimator component (§8.1, §23.1)
//
// The estimation algorithm itself is pinned in tests/unit/coarse_attitude_test
// and tests/unit/triad_test against lib/gnc. What this harness tests is the
// *component*: that measurements off the port arrays and references off the
// OnboardTables ports are assembled into the right frames, that stale or
// invalid inputs are excluded (§9.1), that missing tuning refuses the cycle
// instead of inventing one, and that mode/health telemetry and the acquisition
// and loss EVR edges say what actually happened (REQ-ADET-004).
//
// The harness stubs the two query ports the component calls out on, so the
// references it consumes are fully controlled by the test.
// ======================================================================

#ifndef FLIGHT_POLARISFSW_ATTITUDEESTIMATOR_TESTER_HPP
#define FLIGHT_POLARISFSW_ATTITUDEESTIMATOR_TESTER_HPP

#include <optional>

#include "AttitudeEstimatorGTestBase.hpp"
#include "flight/PolarisFsw/AttitudeEstimator/AttitudeEstimator.hpp"
#include "math/frames.hpp"
#include "math/quaternion.hpp"
#include "math/typed_vector.hpp"

namespace flight {

class AttitudeEstimatorTester : public AttitudeEstimatorGTestBase {
 public:
  static const U32 MAX_HISTORY_SIZE = 100;
  static const FwEnumStoreType TEST_INSTANCE_ID = 0;

  AttitudeEstimatorTester();
  ~AttitudeEstimatorTester();

  // ----------------------------------------------------------------------
  // Tests
  // ----------------------------------------------------------------------

  //! A cycle with no parameters in ParameterDb refuses, warns once, and
  //! telemeters INVALID — it never substitutes a tuning value (§19.3).
  void testRefusesWithoutParameters();

  //! Sun + magnetometer + gyro consistent with a known attitude: the component
  //! acquires it, telemeters COARSE, and publishes the estimate.
  void testAcquiresFromSyntheticMeasurements();

  //! Eclipse (no sun in view) coasts on the gyro, expires past MaxCoastSec with
  //! one AttitudeLost, and re-acquires whole when the sun returns.
  void testCoastsThroughEclipseAndReacquires();

  //! A measurement older than MaxMeasAgeSec is excluded exactly as an invalid
  //! one is, and its validity channel says so.
  void testStaleMeasurementsAreExcluded();

  //! The reference grade served by OnboardTables is telemetered and its
  //! precise->coarse transition warns exactly once.
  void testReferenceGradeIsCarriedAndAlerted();

  //! Without a GNSS fix there is no field model, hence no TRIAD: the component
  //! warns once and gyro-coasts rather than guessing a position.
  void testPositionLossBlocksTheMagneticPair();

  //! A position that is non-finite or nowhere a spacecraft can be is excluded by
  //! the §9.1 range gate, exactly as an invalid fix is — it must not reach the
  //! field model or the sun reference.
  void testImplausiblePositionIsRejected();

  //! RESET_ESTIMATOR drops the solution and re-arms *every* edge-gated alert, so
  //! a still-faulted vehicle reports each fault again rather than staying quiet.
  void testResetReArmsEveryAlert();

  //! Past the loaded IGRF snapshot's published horizon the magnetic reference is
  //! refused rather than extrapolated, with one warning.
  void testExpiredIgrfSnapshotRefusesTheMagneticReference();

 private:
  // ----------------------------------------------------------------------
  // Stubbed query ports (the component's outputs, this harness's inputs)
  // ----------------------------------------------------------------------

  bool from_getBodyPosition_handler(FwIndexType portNum, const OnboardBody& body, I64 taiNs,
                                    PosEciMeters& posEciM) override;
  bool from_getEopAt_handler(FwIndexType portNum, I64 taiNs, EopSample& sample) override;
  void from_estimateOut_handler(FwIndexType portNum, const AttitudeEstimate& estimate) override;

  void connectPorts();
  void initComponents();

  // ----------------------------------------------------------------------
  // Helpers
  // ----------------------------------------------------------------------

  //! Load a valid tuning set into the tester's parameter table.
  void setValidParameters();

  //! Load the onboard IGRF snapshot from the committed IAGA file, taken at
  //! @p decimalYear (default: the era the other tests run in).
  void loadIgrf(double decimalYear = 0.0);

  //! Set the master clock (and the epoch measurements are stamped with) to
  //! @p taiNs, then run one estimation cycle.
  void runCycleAt(I64 taiNs);

  //! Feed one cycle's worth of measurements for truth attitude @p q_bi at
  //! @p taiNs: gyro @p rate_body, the GNSS position, and the sun/magnetic body
  //! vectors obtained by rotating the same references the component builds.
  //! @p sunInView false models eclipse (the pair is dropped).
  void feedMeasurements(
      I64 taiNs,
      const polaris::math::Quat<polaris::math::frames::Body, polaris::math::frames::ECI>& q_bi,
      const Eigen::Vector3d& rate_body, bool sunInView);

  //! The sun reference the component will build at @p taiNs, unit, ECI.
  polaris::math::Vec3<polaris::math::frames::ECI> expectedSunRef(I64 taiNs) const;

  //! The magnetic reference the component will build at @p taiNs, ECI [T].
  polaris::math::Vec3<polaris::math::frames::ECI> expectedMagRef(I64 taiNs) const;

  // ----------------------------------------------------------------------
  // Variables
  // ----------------------------------------------------------------------

  AttitudeEstimator component;

  //! Source grade the stubbed queries report; tests move it to check the alert.
  TableGrade::T stub_grade_{TableGrade::PRECISE};

  //! Whether the GNSS fix fed by feedMeasurements() is valid.
  bool gnss_valid_{true};

  //! Position fed instead of the nominal one, for the range-gate test.
  std::optional<Eigen::Vector3d> position_override_{};

  //! Time tag offset applied to the fed measurements [ns] — negative values age
  //! them for the staleness test.
  I64 meas_time_offset_ns_{0};

  //! Last estimate seen on estimateOut.
  AttitudeEstimate last_estimate_{};
  U32 estimate_count_{0};
};

}  // namespace flight

#endif
