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
  //! The calibration tests run a full 100-sample collection window plus its
  //! before/after cycles, so the bounded history has to hold a few hundred
  //! entries per channel. Long runs (the fine-mode convergence tests) still call
  //! clearHistory() rather than sizing this to them.
  static const U32 MAX_HISTORY_SIZE = 1000;
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

  //! With the fine tuning present the component promotes to FINE off a Davenport
  //! seed, converges the injected gyro bias, and reports a covariance well below
  //! the single-frame seed — which is the whole reason to run a filter.
  void testPromotesToFineAndEstimatesGyroBias();

  //! A sun measurement persistently inconsistent with the filter is rejected by
  //! the NIS gate every cycle; past the configured streak the fine solution is
  //! given up and the published product falls back to the live coarse chain.
  void testNisStreakDemotesToCoarse();

  //! A measurement the filter cannot use at all — finite but unnormalisable, so
  //! it clears the component's gates and is refused rather than gate-rejected —
  //! demotes on the refusal streak, which is a different fault from an outlier
  //! stream and must not be counted as one.
  void testRefusalStreakDemotesToCoarse();

  //! Losing *both* vector sources past the fine coast horizon demotes to coarse
  //! — without an AttitudeLost, because nothing was lost: the coarse solution
  //! was running underneath the whole time.
  void testCoastDemotesFineMode();

  //! A missing fine-mode parameter costs the fine mode, not the estimator: one
  //! FineConfigInvalid, no ConfigInvalid, and the vehicle keeps a coarse
  //! attitude (§10 Safe-mode floor).
  void testMissingFineTuningLeavesCoarseRunning();

  //! RESET_ESTIMATOR drops the fine solution as well as the coarse one, and the
  //! component re-promotes through a fresh seed rather than a resumed filter.
  void testResetDropsFineMode();

  //! The whole commanded flow on a magnetometer carrying a known hard/soft iron:
  //! MAG_CAL_START, a tumbling collection window, an accepted fit, and — the
  //! assertion that matters — the *estimator's* attitude error collapses, which
  //! it can only do if the correction is applied to the vector the estimator
  //! consumes rather than merely stored.
  void testMagCalCollectsFitsAndAppliesTheCorrection();

  //! MAG_CAL_ABORT closes a window without fitting: no calibration is applied,
  //! the sample count goes back to zero, and the estimator is untouched.
  void testMagCalAbortDiscardsTheWindow();

  //! MAG_CAL_CLEAR drops an applied calibration and every consumer goes back to
  //! the raw field — the attitude error returns to its uncalibrated value.
  void testMagCalClearRevertsToRaw();

  //! A window collected over a nearly-fixed attitude spans too narrow a cone of
  //! field directions: the fit is refused with COVERAGE, nothing is applied, and
  //! the live coverage channel showed it coming.
  void testMagCalRejectsNarrowCoverage();

  //! RESET_ESTIMATOR is a *full* reset: it abandons a window in progress and
  //! clears an applied calibration, putting the magnetometer back on raw.
  void testMagCalResetAbortsAndClears();

  //! MAG_CAL_START with no MagCal* tuning in ParameterDb refuses the command and
  //! says CONFIG — the first thing a real vehicle hits, since the calibration
  //! set is delivered separately from the flight tuning — while leaving the
  //! estimator running exactly as before.
  void testMagCalStartRefusedWithoutParameters();

  //! A sample count below MagCalMinSamples or above kMaxCalSamples is refused at
  //! the command rather than after a whole wasted window.
  void testMagCalStartRejectsOutOfRangeCounts();

  //! Collection is a tap, not a mode: an estimator running with a window open
  //! produces bit-identical mode and attitude telemetry to one running the same
  //! measurements with no window at all.
  void testEstimatorUndisturbedDuringCollection();

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

  //! Load a valid tuning set into the tester's parameter table. @p withFine adds
  //! the seven fine-mode parameters; without them the component runs coarse-only
  //! (one FineConfigInvalid), which is what the coarse-behaviour tests want.
  void setValidParameters(bool withFine = false);

  //! Add the seven MagCal* parameters to the tester's table and re-load. Kept
  //! out of setValidParameters() so the existing tests keep proving the
  //! estimator runs with no calibration tuning at all — which is the design:
  //! a missing MagCal* costs only the MAG_CAL_START command.
  void setMagCalParameters();

  //! Feed @p count tumbling cycles from @p t at the 10 Hz rate, advancing @p t.
  //! The attitude sweeps two incommensurate axes so the body-frame field
  //! direction covers the sphere — a single-axis tumble traces a cone and would
  //! be refused on coverage, which is what testMagCalRejectsNarrowCoverage uses.
  void runTumbleCycles(int count, I64& t);

  //! Run @p count tumble cycles and return the **largest** published attitude
  //! error over them [rad]. The maximum, not the last: an uncorrected hard iron
  //! tilts the field by an amount that depends on where the field points in body
  //! axes, so a single cycle can land on a geometry where the iron happens not
  //! to matter, and comparing two single cycles compares two geometries rather
  //! than two calibrations.
  double maxPublishedErrorOverCycles(int count, I64& t);

  //! Angle [rad] between the last published attitude and @p truth.
  double publishedErrorRad(const polaris::math::Quat<polaris::math::frames::Body,
                                                     polaris::math::frames::ECI>& truth) const;

  //! Truth attitude of tumble step @p k — the sequence runTumbleCycles walks.
  static polaris::math::Quat<polaris::math::frames::Body, polaris::math::frames::ECI> tumbleAt(
      int k);

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

  //! Where the vehicle is [m, ECEF]: what the GNSS fix reports *and* what both
  //! inertial references are built at, so the two cannot disagree. Fixed by
  //! default (the geometry only has to be a real place); runTumbleCycles walks it
  //! along a polar orbit, which the calibration needs — see that function.
  Eigen::Vector3d position_ecef_{7.0e6, 0.0, 0.0};

  //! Position fed on the GNSS port *instead of* position_ecef_, for the
  //! range-gate test: the references stay where they were, so the test sees the
  //! gate reject the fix rather than the field model quietly following it.
  std::optional<Eigen::Vector3d> position_override_{};

  //! Time tag offset applied to the fed measurements [ns] — negative values age
  //! them for the staleness test.
  I64 meas_time_offset_ns_{0};

  //! Constant gyro bias [rad/s] added to the reported delta-angle while the sun
  //! and magnetic vectors keep following the *true* attitude: the error the MEKF
  //! exists to estimate and the coarse chain cannot.
  Eigen::Vector3d gyro_bias_{Eigen::Vector3d::Zero()};

  //! Angle [rad] the fed sun body vector is rotated by, away from the direction
  //! the true attitude implies. Large values are the implausible measurement
  //! stream the NIS gate is supposed to reject.
  double sun_body_error_rad_{0.0};

  //! Feed a zero-length (but finite) sun body vector: it clears the component's
  //! finiteness gate and is then refused by the filter as unnormalisable, which
  //! is a *refusal* rather than a gate rejection.
  bool sun_body_degenerate_{false};

  //! Magnetometer error injected into the fed field measurement:
  //! `m = S·B_body + b`, the model the ellipsoid fit inverts. Zero offset and
  //! identity S (the defaults) feed a perfect magnetometer. `S` is kept
  //! **symmetric** in the tests because magnitude data constrains the soft iron
  //! only up to a left rotation — an antisymmetric part is a mounting error, not
  //! an iron error, and belongs to the alignment calibration (§8.1).
  Eigen::Vector3d mag_hard_iron_t_{Eigen::Vector3d::Zero()};
  Eigen::Matrix3d mag_soft_iron_{Eigen::Matrix3d::Identity()};

  //! Tumble step counter, so a test can pause and resume a tumble across
  //! commands without the attitude jumping back to the start.
  int tumble_step_{0};

  //! Last estimate seen on estimateOut.
  AttitudeEstimate last_estimate_{};
  U32 estimate_count_{0};
};

}  // namespace flight

#endif
