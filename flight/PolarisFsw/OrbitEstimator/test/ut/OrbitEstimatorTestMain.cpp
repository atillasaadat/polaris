// ======================================================================
// \title  OrbitEstimatorTestMain.cpp
// \brief  gtest entry for the OrbitEstimator component tests
// ======================================================================

#include "OrbitEstimatorTester.hpp"

TEST(OrbitEstimator, RefusesWithoutParameters) {
  flight::OrbitEstimatorTester tester;
  tester.testRefusesWithoutParameters();
}

TEST(OrbitEstimator, SeedsAndTracks) {
  RecordProperty("verifies", "REQ-ODP-001;REQ-ODP-006");
  flight::OrbitEstimatorTester tester;
  tester.testSeedsAndTracks();
}

TEST(OrbitEstimator, CoastsThenDropsAtHorizon) {
  RecordProperty("verifies", "REQ-ODP-001");
  flight::OrbitEstimatorTester tester;
  tester.testCoastsThenDropsAtHorizon();
}

TEST(OrbitEstimator, RefusesImplausibleFix) {
  RecordProperty("verifies", "REQ-ODP-001");
  flight::OrbitEstimatorTester tester;
  tester.testRefusesImplausibleFix();
}

TEST(OrbitEstimator, EopUnavailable) {
  flight::OrbitEstimatorTester tester;
  tester.testEopUnavailable();
}

TEST(OrbitEstimator, DegradedReacquiresByUpdateNotSeed) {
  flight::OrbitEstimatorTester tester;
  tester.testDegradedReacquiresByUpdateNotSeed();
}

TEST(OrbitEstimator, NonGravAccelIsAppliedOnlyWhenFreshAndValid) {
  flight::OrbitEstimatorTester tester;
  tester.testNonGravAccelIsAppliedOnlyWhenFreshAndValid();
}

TEST(OrbitEstimator, GroundSeedAcceptedAndRefused) {
  flight::OrbitEstimatorTester tester;
  tester.testGroundSeedAcceptedAndRefused();
}

TEST(OrbitEstimator, ResetDropsSolution) {
  flight::OrbitEstimatorTester tester;
  tester.testResetDropsSolution();
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
