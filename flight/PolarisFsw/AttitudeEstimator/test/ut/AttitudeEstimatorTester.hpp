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

#include <array>
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

  //! An orbit solution that is flagged valid but non-finite is refused exactly
  //! as an invalid one is — it must not reach the field model or the sun
  //! reference (§9.1: wire data is gated, whatever its flag says).
  void testNonFinitePositionIsRejected();

  //! The orbit solution is consumed once per cycle: a producer that stops
  //! publishing leaves position unavailable, never a stale vector reused.
  void testOrbitSolutionIsConsumedOnce();

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

  //! The albedo correction runs where the geometry supports it, is telemetered,
  //! and drags the reported covariance down with it via the per-cycle sigma
  //! selection — including the cold-start cycle, where there is no attitude yet
  //! and it must not run.
  void testAlbedoCorrectionAppliesAndTightensTheCovariance();

  //! Each way the geometry can be missing — no position fix, Earth out of the
  //! sensor's field — leaves the measurement uncorrected rather than corrected
  //! on a guess.
  void testAlbedoCorrectionSkippedWithoutGeometry();

  //! The corrected cycle's sun sigma carries the `A·σ_att/2` term, so it
  //! inflates while the attitude is poorly known and relaxes as the covariance
  //! converges — never reaching the uncorrected budget.
  void testAlbedoSigmaInflatesWithTheAttitudeUncertainty();

  //! A sun sensor whose per-unit boresight slot is the zero vector — not
  //! installed, or mounting not characterised — is never albedo corrected:
  //! applying a neighbour's boresight fails silently rather than loudly.
  void testAlbedoSkippedForAnUncharacterisedSunSensor();

  //! ...and the other half: a unit on another port *with* a configured boresight
  //! is corrected, with its own geometry. Without this the §8.2 handoff would
  //! silently drop back to the uncorrected sun budget.
  void testAlbedoFollowsTheSelectedUnitsBoresight();

  //! Sun-sensor selection takes the unit reporting the smallest realised sigma,
  //! whichever port it arrives on, with ties keeping the lower index (§8.2).
  void testSunSelectionTakesTheBestIlluminatedUnit();

  //! A non-positive or non-finite reported sigma is gated out of the selection
  //! rather than winning it by being the smallest number in the array.
  void testSunSelectionRejectsAnUnusableSigma();

  //! REQ-ADET-008 through the component: one railed IMU among three leaves the
  //! published attitude bit-identical to the all-healthy run, with one
  //! ImuUnitExcluded and the exclusion mask set.
  void testRailedImuIsExcludedAndCostsNothing();

  //! A NaN gyro reading is gated before it can propagate a NaN quaternion behind
  //! a validity flag that still reads true.
  void testNonFiniteImuIsExcludedNotPropagated();

  //! A stale unit is *absent*, not implausible: no exclusion is latched and no
  //! FDIR event is raised (§9.1 vs §9.2).
  void testStaleImuIsAbsentNotExcluded();

  //! REQ-ADET-009: an excluded unit comes back after ImuReadmitCycles
  //! consecutive plausible cycles, with the recovery edge reported once.
  void testExcludedImuIsReadmittedAfterRecovery();

  //! Two plausible units disagreeing with nothing to attribute it leaves the
  //! vehicle with **no** body rate, and latches nothing.
  void testTwoImuDisagreementLeavesNoRate();

  //! The reference vehicle's branch: two units disagree, the MEKF's propagated
  //! rate identifies the offender, it is excluded as OUTVOTED, and the surviving
  //! unit carries the solution with no loss and no demotion.
  void testTwoImuDisagreementIsIdentifiedByTheFilter();

  //! An outvoted unit does not re-admit itself while it still disagrees: over a
  //! sustained fault the exclusion is reported once, not once per re-admission
  //! window (the flap C1 named).
  void testOutvotedImuDoesNotFlap();

  //! A persistent unattributable disagreement escalates at the configured
  //! horizon and repeats at that bounded cadence, while the vehicle keeps a
  //! TRIAD attitude throughout.
  void testPersistentAmbiguityEscalates();

  //! RESET_ESTIMATOR is the commanded re-admission: every exclusion latch drops
  //! at once, without serving out the automatic policy.
  void testResetClearsImuExclusions();

  //! A negative sun sigma on the good side of either pair is refused: the hypot
  //! composition would otherwise absorb the sign silently.
  void testNegativeSunSigmaIsRefused();

  //! The sun systematic follows the served ephemeris grade: a PRECISE-graded
  //! cycle (DE440 tables covering the epoch) reports a tighter covariance floor
  //! than an otherwise identical analytic-fallback cycle.
  void testSunSigmaFollowsTheEphemerisGrade();

  //! A missing albedo parameter costs the correction and nothing else: one
  //! edge-gated alert, and a vehicle still acquiring attitude on the wider
  //! uncorrected budget.
  void testMissingAlbedoTuningLeavesTheEstimatorRunning();

  //! A valid tracker takes the §8.2 ladder to STAR_TRACKER: the sun and magnetic
  //! pairs stop being fused and become residual monitors, and the transition is
  //! reported as a source change rather than as a demotion.
  void testStarTrackerTakesTheLadderToItsTopRung();

  //! Losing every tracker walks the ladder back down to SUN_MAG without losing
  //! the attitude, the filter state, or emitting a demotion.
  void testStarTrackerLossFallsBackToSunAndMagnetometer();

  //! A sun sensor that has drifted is *observable* once it is a monitor rather
  //! than a measurement: the alert fires after MonitorAlertCycles and clears on
  //! recovery.
  void testDriftedSunSensorRaisesTheResidualMonitor();

  //! Missing star-tracker tuning costs the top rung and nothing else: one
  //! StConfigInvalid, the vehicle still in SS+MAG fine mode.
  void testMissingStarTrackerTuningCapsTheLadder();

  //! A tracker seeds the filter with no sun/field pair and no coarse solution —
  //! the eclipse and cold-start case a Davenport seed structurally cannot cover.
  void testStarTrackerSeedsFineModeWithoutTheVectorPairs();

  //! The commanded inter-tracker alignment: window, fit, apply, and the measured
  //! improvement in the second unit's agreement with the king.
  void testInterTrackerAlignmentCollectsFitsAndApplies();

  //! ST_ALIGN_CAL_START refuses the king (a rotation that is zero by definition),
  //! an out-of-range unit, an uncharacterised one, and a bad sample count.
  void testInterTrackerAlignmentRefusesTheKingAndBadCommands();

  //! ST_ALIGN_CAL_ABORT discards a window without fitting; ST_ALIGN_CAL_CLEAR
  //! reverts one unit to as-mounted; RESET_ESTIMATOR does both.
  void testInterTrackerAlignmentAbortAndClear();

  //! A magnetometer outside the IGRF-magnitude band is latched out and the
  //! surviving unit carries the pair — the published attitude is unchanged.
  void testImplausibleMagnetometerIsExcludedAndCostsNothing();

  //! Two plausible magnetometers disagreeing with no usable attitude reference
  //! costs the magnetic pair for that cycle and latches nothing.
  void testTwoMagnetometerDisagreementLeavesNoMagneticPair();

  //! The sun cross-unit consistency check: a confidently-wrong selected unit is
  //! caught by the runner-up and overridden against the fine solution.
  void testSunCrossUnitCheckAlertsAndOverrides();

  //! The C1+C2 regression together: with the king healthy and the second tracker
  //! persistently rejected by the NIS gate, the fine mode **stays engaged on the
  //! king** and the bad unit is latched out with its own EVR — where a
  //! cycle-global NIS streak would have demoted the mode, dropped the filter and
  //! re-promoted off the same bad unit at the streak period.
  void testBadStarTrackerIsIsolatedWithoutDemotingTheMode();

  //! An uncalibrated non-king tracker is not fused at all: its as-mounted reading
  //! carries the two units' bias difference, and fusing it at the configured
  //! sigma would sell a systematic as white noise. Fused once the alignment is
  //! fitted, dropped again when it is cleared.
  void testUncalibratedSecondTrackerIsNotFused();

  //! The §9.2 off-rung tracker arbitration (REQ-FDIR-013). A tracker the filter
  //! persistently rejects while the solution is *not* tracker-sourced is judged
  //! against the coarse solution instead, in the Mahalanobis metric at
  //! χ²₃(0.999): inside, the filter is the outlier and is re-seeded from the
  //! tracker; outside, the unit is refused — and **not** latched out, since the
  //! criterion that would convict it is not the one that would re-admit it.
  //!
  //! Two branches of `arbitrateRejectedTrackers` are deliberately not covered
  //! here because they are **unreachable by construction** rather than merely
  //! awkward: an invalid coarse solution under an active fine mode cannot occur
  //! while `MekfMaxCoastSec` < `MaxCoastSec` (the filter coasts out first, and it
  //! does on the flight tuning too — 300 s against 2400 s), and a refused
  //! `Mekf::initialize` needs a non-finite or indefinite `R`, which
  //! `refreshStConfig` rejects before a tracker is ever fused. Both are guard
  //! clauses that return without acting; building harness hooks to force them
  //! would test the hooks.
  //! @{

  //! Arm the arbitration: promote on the vector pairs with no tracker in view,
  //! then bring one in disagreeing by @p trackerErrorRad, so the filter rejects
  //! it for `MekfNisStreak` consecutive cycles. Advances @p t.
  void armOffRungArbitration(
      I64& t,
      const polaris::math::Quat<polaris::math::frames::Body, polaris::math::frames::ECI>& truth,
      double trackerErrorRad);

  //! Inside the gate: adopted, reported with the statistic it was decided on,
  //! and the ladder is on its top rung from that cycle.
  void testOffRungTrackerVerdictFollowsTheCoarseGate();

  //! Outside the gate: refused, nothing latched, the mode untouched, and the
  //! refusal repeated at a bounded cadence rather than once or per cycle.
  void testOffRungTrackerOutsideTheGateIsRefusedNotLatched();

  //! Two eligible trackers disagreeing off the rung: the one the coarse fix
  //! supports is adopted and the other is left alone, not blamed.
  void testOffRungArbitrationPicksTheTrackerTheCoarseFixSupports();
  //! @}

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
  //! @p withAlbedo adds the three Earth-albedo parameters; without them the
  //! correction never runs (one AlbedoConfigInvalid) and every cycle is weighted
  //! at SigmaSunAlbedoUncorrRad — the default, so the pre-existing tests keep
  //! proving the estimator works with no albedo tuning at all.
  //! @p withStarTracker adds the star-tracker fusion set (king unit, the two
  //! sigmas, the per-unit boresights and the three residual-monitor thresholds);
  //! without them no tracker is fused (one StConfigInvalid) and the §8.2 ladder is
  //! capped at SS+MAG — the default, so every pre-existing case keeps proving the
  //! estimator works with no tracker tuning at all.
  void setValidParameters(bool withFine = false, bool withAlbedo = false,
                          bool withStarTracker = false);

  //! Add the three StAlign* parameters and re-load. Kept out of
  //! setValidParameters() for the same reason the MagCal set is: a missing one
  //! costs only the ST_ALIGN_CAL_START command.
  void setStAlignParameters();

  //! Per-unit albedo boresights staged into `SunAlbedoBoresightsBody`, flattened
  //! three at a time in port order. Default: slot 0 is body +Z (the reference
  //! vehicle's solar-array normal) and every other slot is the zero vector, i.e.
  //! "not installed" — so a test that moves the sun measurement to another port
  //! sees the correction skipped unless it also writes that slot's boresight.
  //! Only read when `setValidParameters(withAlbedo = true)` is called.
  std::array<F64, 3 * AttitudeEstimator::NUM_SUNSENSORIN_INPUT_PORTS> sun_boresights_{
      {0.0, 0.0, 1.0}};

  //! Add the seven MagCal* parameters to the tester's table and re-load. Kept
  //! out of setValidParameters() so the existing tests keep proving the
  //! estimator runs with no calibration tuning at all — which is the design:
  //! a missing MagCal* costs only the MAG_CAL_START command.
  void setMagCalParameters();

  //! The attitude that puts nadir on the sun sensor's boresight (body +Z) at
  //! @p taiNs — an Earth-pointing vehicle, the geometry the albedo correction
  //! exists for and the only one where it has anything to remove.
  polaris::math::Quat<polaris::math::frames::Body, polaris::math::frames::ECI>
  earthInTheSunSensorField(I64 taiNs) const;

  //! The Sun's ECI direction the stubbed ephemeris reports at @p taiNs.
  Eigen::Vector3d sunDirectionEci(I64 taiNs) const;

  //! Port index the sun-sensor measurement is fed on. 0 for most tests; the §8.2
  //! selection and per-unit-boresight cases move it, and the handoff case feeds
  //! several ports at once through @ref sun_extra_.
  FwIndexType sun_port_index_{0};

  //! One extra sun-sensor unit fed alongside @ref sun_port_index_, so a test can
  //! exercise the best-illuminated *selection* rather than a single candidate.
  //! `index < 0` (the default) feeds nothing extra. The measurement is the same
  //! true direction; only the reported sigma differs, which is exactly the
  //! discriminator the component selects on.
  struct ExtraSunUnit {
    FwIndexType index{-1};
    double sigma_rad{0.0};
  };

  ExtraSunUnit sun_extra_{};

  //! The extra sun unit reports the **true** direction while @ref
  //! sun_body_error_rad_ is applied to the selected one only. That is the
  //! §8.2 cross-unit case: the selected unit is confidently wrong (it reports the
  //! smaller sigma and wins), and only the runner-up can say so.
  bool sun_extra_truthful_{false};

  //! Number of IMU ports fed by feedMeasurements(). **Two** by default, matching
  //! the reference vehicle, so every test exercises the pairwise branch of the
  //! §8.2 vote — the one that actually flies — rather than a single-unit special
  //! case. The median branch is still covered, by the cases that raise this to
  //! three explicitly. Units report identical readings unless @ref imu_fault_
  //! says otherwise, and identical readings combine to themselves either way.
  FwIndexType imu_unit_count_{2};

  //! Fault injected into one IMU unit, for the single-fault-survival cases.
  enum class ImuFault { kNone, kRailed, kNotFinite, kStale, kOffset };

  struct ImuFaultInjection {
    FwIndexType index{-1};  //!< which unit; < 0 injects nothing
    ImuFault kind{ImuFault::kNone};
    //! Rate reported by a kRailed unit, or the offset added by a kOffset one
    //! [rad/s]. kOffset stays inside the plausibility limit on purpose: it is the
    //! fault only a *comparison* between units can find.
    Eigen::Vector3d value{Eigen::Vector3d::Zero()};
  };

  ImuFaultInjection imu_fault_{};

  //! Number of magnetometer ports fed by feedMeasurements(). **Two** by default,
  //! matching the reference vehicle, so every test exercises the pairwise branch
  //! of the §8.2 magnetometer vote — the one that actually flies. Units report
  //! identical readings unless @ref mag_fault_index_ says otherwise, and identical
  //! readings combine to themselves, so the published solution is unchanged.
  FwIndexType mag_unit_count_{2};

  //! Field offset [T] added to magnetometer unit @ref mag_fault_index_ (< 0
  //! injects nothing). Small values stay inside the magnitude band and are the
  //! fault only a *comparison* between units can find; large ones trip the band.
  //! Scale and offset applied to magnetometer unit @ref mag_fault_index_ as
  //! `scale * field + offset` (< 0 index injects nothing). The scale models a dead
  //! or unpowered sensor, which the IGRF-magnitude band catches whatever direction
  //! the field happens to point; the offset models a plausible-magnitude
  //! disagreement, the fault only a comparison between units can find.
  FwIndexType mag_fault_index_{-1};
  double mag_fault_scale_{1.0};
  Eigen::Vector3d mag_fault_offset_t_{Eigen::Vector3d::Zero()};

  //! Number of star-tracker ports fed by feedMeasurements(). **Zero** by default,
  //! so every case that predates §8.2 tracker fusion runs exactly as it did — the
  //! ladder never leaves SS+MAG unless a test asks it to.
  FwIndexType star_unit_count_{0};

  //! Per-unit small-angle error added to the fed tracker attitude [rad, body
  //! axes], composed as δq(θ) ⊗ q_true. Slot 0 stands in for the king's own bias —
  //! which nothing removes, because it is what defines the frame — and slot 1 for
  //! the second unit's mounting misalignment, which is what ST_ALIGN_CAL
  //! estimates.
  //!
  //! **Zeroed explicitly in the constructor, not by `{}`.** A fixed-size Eigen
  //! type's default constructor leaves its storage *uninitialized*, so
  //! `Eigen::Vector3d a[N]{}` value-initialises the array by calling that
  //! constructor N times and produces garbage — which reads as zero on a fresh
  //! stack and as the previous test's data inside a full suite run. That is a
  //! test that passes alone and fails in CI, and it cost an afternoon here.
  Eigen::Vector3d star_error_[AttitudeEstimator::NUM_STARTRACKERIN_INPUT_PORTS];

  //! Which tracker units report `valid` this cycle. All true by default; a test
  //! drops one to walk the ladder back down to SS+MAG.
  bool star_valid_[AttitudeEstimator::NUM_STARTRACKERIN_INPUT_PORTS]{true, true, true, true,
                                                                     true, true, true, true};

  //! Per-unit star-tracker boresights staged into `StBoresightsBody`. Default:
  //! the reference vehicle's two units at (∓1, 0, −1)/√2, 90° apart; the rest are
  //! the zero vector, i.e. "not installed".
  std::array<F64, 3 * AttitudeEstimator::NUM_STARTRACKERIN_INPUT_PORTS> st_boresights_{
      {-0.7071067811865476, 0.0, -0.7071067811865476, 0.7071067811865476, 0.0,
       -0.7071067811865476}};

  //! Put the Sun 45 degrees from the radial direction rather than square to the
  //! field: the geometry the albedo term peaks in (its magnitude goes as
  //! cos*sin of that angle). Default false, so every test that predates the
  //! correction sees the Sun exactly where it always did.
  bool sun_at_45_from_nadir_{false};

  //! Trace of the published attitude-error covariance [rad^2]. Read rather than
  //! asserted so a test can compare two cycles' confidence against each other.
  double publishedCovTrace() const;

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

  //! Whether the orbit solution fed by feedMeasurements() is flagged valid, and
  //! whether one is fed at all (a producer that did not run this cycle).
  bool orbit_valid_{true};
  bool orbit_published_{true};

  //! Where the vehicle is [m, ECEF]: what the fed orbit solution states (in ECI,
  //! rotated with the stubbed EOP) *and* what both inertial references are
  //! built at, so the two cannot disagree. Fixed by
  //! default (the geometry only has to be a real place); runTumbleCycles walks it
  //! along a polar orbit, which the calibration needs — see that function.
  Eigen::Vector3d position_ecef_{7.0e6, 0.0, 0.0};

  //! Position fed on the orbit port *instead of* position_ecef_ (stated in ECI
  //! as-is), for the non-finite gate test: the references stay where they were,
  //! so the test sees the gate reject the solution rather than the field model
  //! quietly following it.
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
