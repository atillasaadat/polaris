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

TEST(AttitudeEstimator, ImplausiblePositionIsRejected) {
  flight::AttitudeEstimatorTester tester;
  tester.testImplausiblePositionIsRejected();
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

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
