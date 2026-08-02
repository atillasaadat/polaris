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

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
