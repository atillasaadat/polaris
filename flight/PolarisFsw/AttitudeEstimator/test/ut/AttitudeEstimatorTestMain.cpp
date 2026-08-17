// ======================================================================
// \title  AttitudeEstimatorTestMain.cpp
// \brief  Entry point for the AttitudeEstimator component tests (§23.1)
// ======================================================================

#include "AttitudeEstimatorTester.hpp"

TEST(AttitudeEstimator, RefusesWithoutParameters) {
  RecordProperty("verifies", "REQ-ADET-004");
  flight::AttitudeEstimatorTester tester;
  tester.testRefusesWithoutParameters();
}

TEST(AttitudeEstimator, AcquiresFromSyntheticMeasurements) {
  RecordProperty("verifies", "REQ-ADET-002");
  flight::AttitudeEstimatorTester tester;
  tester.testAcquiresFromSyntheticMeasurements();
}

TEST(AttitudeEstimator, CoastsThroughEclipseAndReacquires) {
  RecordProperty("verifies", "REQ-ADET-003");
  flight::AttitudeEstimatorTester tester;
  tester.testCoastsThroughEclipseAndReacquires();
}

TEST(AttitudeEstimator, StaleMeasurementsAreExcluded) {
  flight::AttitudeEstimatorTester tester;
  tester.testStaleMeasurementsAreExcluded();
}

TEST(AttitudeEstimator, ReferenceGradeIsCarriedAndAlerted) {
  RecordProperty("verifies", "REQ-ADET-004");
  flight::AttitudeEstimatorTester tester;
  tester.testReferenceGradeIsCarriedAndAlerted();
}

TEST(AttitudeEstimator, PositionLossBlocksTheMagneticPair) {
  flight::AttitudeEstimatorTester tester;
  tester.testPositionLossBlocksTheMagneticPair();
}

TEST(AttitudeEstimator, NonFinitePositionIsRejected) {
  flight::AttitudeEstimatorTester tester;
  tester.testNonFinitePositionIsRejected();
}

TEST(AttitudeEstimator, DegradedOrbitSolutionIsUsedUpToTheSigmaTolerance) {
  flight::AttitudeEstimatorTester tester;
  tester.testDegradedOrbitSolutionIsUsedUpToTheSigmaTolerance();
}

TEST(AttitudeEstimator, OrbitSolutionIsConsumedOnce) {
  flight::AttitudeEstimatorTester tester;
  tester.testOrbitSolutionIsConsumedOnce();
}

TEST(AttitudeEstimator, ResetReArmsEveryAlert) {
  flight::AttitudeEstimatorTester tester;
  tester.testResetReArmsEveryAlert();
}

TEST(AttitudeEstimator, ExpiredIgrfSnapshotRefusesTheMagneticReference) {
  flight::AttitudeEstimatorTester tester;
  tester.testExpiredIgrfSnapshotRefusesTheMagneticReference();
}

TEST(AttitudeEstimator, PromotesToFineAndEstimatesGyroBias) {
  RecordProperty("verifies", "REQ-ADET-004");
  flight::AttitudeEstimatorTester tester;
  tester.testPromotesToFineAndEstimatesGyroBias();
}

TEST(AttitudeEstimator, NisStreakDemotesToCoarse) {
  RecordProperty("verifies", "REQ-ADET-004");
  flight::AttitudeEstimatorTester tester;
  tester.testNisStreakDemotesToCoarse();
}

TEST(AttitudeEstimator, RefusalStreakDemotesToCoarse) {
  RecordProperty("verifies", "REQ-ADET-004");
  flight::AttitudeEstimatorTester tester;
  tester.testRefusalStreakDemotesToCoarse();
}

TEST(AttitudeEstimator, CoastDemotesFineMode) {
  RecordProperty("verifies", "REQ-ADET-004");
  flight::AttitudeEstimatorTester tester;
  tester.testCoastDemotesFineMode();
}

TEST(AttitudeEstimator, MissingFineTuningLeavesCoarseRunning) {
  flight::AttitudeEstimatorTester tester;
  tester.testMissingFineTuningLeavesCoarseRunning();
}

TEST(AttitudeEstimator, ResetDropsFineMode) {
  RecordProperty("verifies", "REQ-ADET-004");
  flight::AttitudeEstimatorTester tester;
  tester.testResetDropsFineMode();
}

TEST(AttitudeEstimator, FineTuningUploadKeepsTheSolution) {
  RecordProperty("verifies", "REQ-ADET-014");
  flight::AttitudeEstimatorTester tester;
  tester.testFineTuningUploadKeepsTheSolution();
}

TEST(AttitudeEstimator, FineCovarianceReinitAndMeasurementPolicy) {
  RecordProperty("verifies", "REQ-ADET-014");
  flight::AttitudeEstimatorTester tester;
  tester.testFineCovarianceReinitAndMeasurementPolicy();
}

TEST(AttitudeEstimator, MagCalCollectsFitsAndAppliesTheCorrection) {
  RecordProperty("verifies", "REQ-ADET-005");
  flight::AttitudeEstimatorTester tester;
  tester.testMagCalCollectsFitsAndAppliesTheCorrection();
}

TEST(AttitudeEstimator, MagCalAbortDiscardsTheWindow) {
  flight::AttitudeEstimatorTester tester;
  tester.testMagCalAbortDiscardsTheWindow();
}

TEST(AttitudeEstimator, MagCalClearRevertsToRaw) {
  flight::AttitudeEstimatorTester tester;
  tester.testMagCalClearRevertsToRaw();
}

TEST(AttitudeEstimator, MagCalRejectsNarrowCoverage) {
  RecordProperty("verifies", "REQ-ADET-005");
  flight::AttitudeEstimatorTester tester;
  tester.testMagCalRejectsNarrowCoverage();
}

TEST(AttitudeEstimator, MagCalResetAbortsAndClears) {
  flight::AttitudeEstimatorTester tester;
  tester.testMagCalResetAbortsAndClears();
}

TEST(AttitudeEstimator, MagCalStartRefusedWithoutParameters) {
  flight::AttitudeEstimatorTester tester;
  tester.testMagCalStartRefusedWithoutParameters();
}

TEST(AttitudeEstimator, MagCalStartRejectsOutOfRangeCounts) {
  flight::AttitudeEstimatorTester tester;
  tester.testMagCalStartRejectsOutOfRangeCounts();
}

TEST(AttitudeEstimator, EstimatorUndisturbedDuringCollection) {
  RecordProperty("verifies", "REQ-ADET-002");
  flight::AttitudeEstimatorTester tester;
  tester.testEstimatorUndisturbedDuringCollection();
}

TEST(AttitudeEstimator, AlbedoCorrectionAppliesAndTightensTheCovariance) {
  RecordProperty("verifies", "REQ-ADET-006");
  flight::AttitudeEstimatorTester tester;
  tester.testAlbedoCorrectionAppliesAndTightensTheCovariance();
}

TEST(AttitudeEstimator, AlbedoCorrectionSkippedWithoutGeometry) {
  flight::AttitudeEstimatorTester tester;
  tester.testAlbedoCorrectionSkippedWithoutGeometry();
}

TEST(AttitudeEstimator, AlbedoSigmaInflatesWithTheAttitudeUncertainty) {
  RecordProperty("verifies", "REQ-ADET-006");
  flight::AttitudeEstimatorTester tester;
  tester.testAlbedoSigmaInflatesWithTheAttitudeUncertainty();
}

TEST(AttitudeEstimator, AlbedoSkippedForAnUncharacterisedSunSensor) {
  flight::AttitudeEstimatorTester tester;
  tester.testAlbedoSkippedForAnUncharacterisedSunSensor();
}

TEST(AttitudeEstimator, AlbedoFollowsTheSelectedUnitsBoresight) {
  flight::AttitudeEstimatorTester tester;
  tester.testAlbedoFollowsTheSelectedUnitsBoresight();
}

// ── §8.2 multi-unit selection and fault-tolerant voting ─────────────────────

TEST(AttitudeEstimator, SunSelectionTakesTheBestIlluminatedUnit) {
  flight::AttitudeEstimatorTester tester;
  tester.testSunSelectionTakesTheBestIlluminatedUnit();
}

TEST(AttitudeEstimator, SunSelectionRejectsAnUnusableSigma) {
  flight::AttitudeEstimatorTester tester;
  tester.testSunSelectionRejectsAnUnusableSigma();
}

TEST(AttitudeEstimator, RailedImuIsExcludedAndCostsNothing) {
  RecordProperty("verifies", "REQ-ADET-008");
  flight::AttitudeEstimatorTester tester;
  tester.testRailedImuIsExcludedAndCostsNothing();
}

TEST(AttitudeEstimator, NonFiniteImuIsExcludedNotPropagated) {
  RecordProperty("verifies", "REQ-ADET-008");
  flight::AttitudeEstimatorTester tester;
  tester.testNonFiniteImuIsExcludedNotPropagated();
}

TEST(AttitudeEstimator, StaleImuIsAbsentNotExcluded) {
  RecordProperty("verifies", "REQ-ADET-009");
  flight::AttitudeEstimatorTester tester;
  tester.testStaleImuIsAbsentNotExcluded();
}

TEST(AttitudeEstimator, ExcludedImuIsReadmittedAfterRecovery) {
  RecordProperty("verifies", "REQ-ADET-009");
  flight::AttitudeEstimatorTester tester;
  tester.testExcludedImuIsReadmittedAfterRecovery();
}

TEST(AttitudeEstimator, TwoImuDisagreementLeavesNoRate) {
  RecordProperty("verifies", "REQ-ADET-008");
  flight::AttitudeEstimatorTester tester;
  tester.testTwoImuDisagreementLeavesNoRate();
}

TEST(AttitudeEstimator, TwoImuDisagreementIsIdentifiedByTheFilter) {
  RecordProperty("verifies", "REQ-ADET-008");
  flight::AttitudeEstimatorTester tester;
  tester.testTwoImuDisagreementIsIdentifiedByTheFilter();
}

TEST(AttitudeEstimator, OutvotedImuDoesNotFlap) {
  RecordProperty("verifies", "REQ-ADET-009");
  flight::AttitudeEstimatorTester tester;
  tester.testOutvotedImuDoesNotFlap();
}

TEST(AttitudeEstimator, PersistentAmbiguityEscalates) {
  RecordProperty("verifies", "REQ-ADET-008");
  flight::AttitudeEstimatorTester tester;
  tester.testPersistentAmbiguityEscalates();
}

TEST(AttitudeEstimator, ResetClearsImuExclusions) {
  RecordProperty("verifies", "REQ-ADET-009");
  flight::AttitudeEstimatorTester tester;
  tester.testResetClearsImuExclusions();
}

TEST(AttitudeEstimator, NegativeSunSigmaIsRefused) {
  flight::AttitudeEstimatorTester tester;
  tester.testNegativeSunSigmaIsRefused();
}

TEST(AttitudeEstimator, SunSigmaFollowsTheEphemerisGrade) {
  RecordProperty("verifies", "REQ-ADET-006");
  flight::AttitudeEstimatorTester tester;
  tester.testSunSigmaFollowsTheEphemerisGrade();
}

TEST(AttitudeEstimator, MissingAlbedoTuningLeavesTheEstimatorRunning) {
  flight::AttitudeEstimatorTester tester;
  tester.testMissingAlbedoTuningLeavesTheEstimatorRunning();
}

// ── §8.2 star-tracker fusion and the mode ladder ────────────────────────────

TEST(AttitudeEstimator, StarTrackerTakesTheLadderToItsTopRung) {
  RecordProperty("verifies", "REQ-ADET-012");
  flight::AttitudeEstimatorTester tester;
  tester.testStarTrackerTakesTheLadderToItsTopRung();
}

TEST(AttitudeEstimator, StarTrackerLossFallsBackToSunAndMagnetometer) {
  RecordProperty("verifies", "REQ-ADET-012");
  flight::AttitudeEstimatorTester tester;
  tester.testStarTrackerLossFallsBackToSunAndMagnetometer();
}

TEST(AttitudeEstimator, DriftedSunSensorRaisesTheResidualMonitor) {
  RecordProperty("verifies", "REQ-ADET-012");
  flight::AttitudeEstimatorTester tester;
  tester.testDriftedSunSensorRaisesTheResidualMonitor();
}

TEST(AttitudeEstimator, MissingStarTrackerTuningCapsTheLadder) {
  RecordProperty("verifies", "REQ-ADET-012");
  flight::AttitudeEstimatorTester tester;
  tester.testMissingStarTrackerTuningCapsTheLadder();
}

TEST(AttitudeEstimator, StarTrackerSeedsFineModeWithoutTheVectorPairs) {
  RecordProperty("verifies", "REQ-ADET-007");
  flight::AttitudeEstimatorTester tester;
  tester.testStarTrackerSeedsFineModeWithoutTheVectorPairs();
}

TEST(AttitudeEstimator, InterTrackerAlignmentCollectsFitsAndApplies) {
  RecordProperty("verifies", "REQ-ADET-013");
  flight::AttitudeEstimatorTester tester;
  tester.testInterTrackerAlignmentCollectsFitsAndApplies();
}

TEST(AttitudeEstimator, InterTrackerAlignmentRefusesTheKingAndBadCommands) {
  RecordProperty("verifies", "REQ-ADET-013");
  flight::AttitudeEstimatorTester tester;
  tester.testInterTrackerAlignmentRefusesTheKingAndBadCommands();
}

TEST(AttitudeEstimator, InterTrackerAlignmentAbortAndClear) {
  RecordProperty("verifies", "REQ-ADET-013");
  flight::AttitudeEstimatorTester tester;
  tester.testInterTrackerAlignmentAbortAndClear();
}

// ── §8.2 multi-magnetometer voting ──────────────────────────────────────────

TEST(AttitudeEstimator, ImplausibleMagnetometerIsExcludedAndCostsNothing) {
  RecordProperty("verifies", "REQ-ADET-011");
  flight::AttitudeEstimatorTester tester;
  tester.testImplausibleMagnetometerIsExcludedAndCostsNothing();
}

TEST(AttitudeEstimator, TwoMagnetometerDisagreementLeavesNoMagneticPair) {
  RecordProperty("verifies", "REQ-ADET-011");
  flight::AttitudeEstimatorTester tester;
  tester.testTwoMagnetometerDisagreementLeavesNoMagneticPair();
}

TEST(AttitudeEstimator, SunCrossUnitCheckAlertsAndOverrides) {
  RecordProperty("verifies", "REQ-ADET-012");
  flight::AttitudeEstimatorTester tester;
  tester.testSunCrossUnitCheckAlertsAndOverrides();
}

TEST(AttitudeEstimator, BadStarTrackerIsIsolatedWithoutDemotingTheMode) {
  RecordProperty("verifies", "REQ-ADET-012");
  flight::AttitudeEstimatorTester tester;
  tester.testBadStarTrackerIsIsolatedWithoutDemotingTheMode();
}

TEST(AttitudeEstimator, UncalibratedSecondTrackerIsNotFused) {
  RecordProperty("verifies", "REQ-ADET-013");
  flight::AttitudeEstimatorTester tester;
  tester.testUncalibratedSecondTrackerIsNotFused();
}

TEST(AttitudeEstimator, OffRungTrackerVerdictFollowsTheCoarseGate) {
  RecordProperty("verifies", "REQ-FDIR-013");
  flight::AttitudeEstimatorTester tester;
  tester.testOffRungTrackerVerdictFollowsTheCoarseGate();
}

TEST(AttitudeEstimator, OffRungTrackerOutsideTheGateIsRefusedNotLatched) {
  RecordProperty("verifies", "REQ-FDIR-013");
  flight::AttitudeEstimatorTester tester;
  tester.testOffRungTrackerOutsideTheGateIsRefusedNotLatched();
}

TEST(AttitudeEstimator, OffRungArbitrationPicksTheTrackerTheCoarseFixSupports) {
  RecordProperty("verifies", "REQ-FDIR-013");
  flight::AttitudeEstimatorTester tester;
  tester.testOffRungArbitrationPicksTheTrackerTheCoarseFixSupports();
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
