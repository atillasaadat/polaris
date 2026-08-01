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

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
