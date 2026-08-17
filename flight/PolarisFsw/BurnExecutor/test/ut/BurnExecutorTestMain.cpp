// ======================================================================
// \title  BurnExecutorTestMain.cpp
// \brief  gtest entry for the BurnExecutor component tests
// ======================================================================

#include "BurnExecutorTester.hpp"

TEST(BurnExecutor, RefusesWithoutParameters) {
  flight::BurnExecutorTester tester;
  tester.testRefusesWithoutParameters();
}

TEST(BurnExecutor, RefusalPaths) {
  RecordProperty("verifies", "REQ-MAN-001");
  flight::BurnExecutorTester tester;
  tester.testRefusalPaths();
}

TEST(BurnExecutor, BurnAccelerationAndDepletion) {
  RecordProperty("verifies", "REQ-MAN-001");
  flight::BurnExecutorTester tester;
  tester.testBurnAccelerationAndDepletion();
}

TEST(BurnExecutor, AbortMidBurn) {
  flight::BurnExecutorTester tester;
  tester.testAbortMidBurn();
}

TEST(BurnExecutor, StaleAttitudeAbortsTheBurn) {
  RecordProperty("verifies", "REQ-MAN-001");
  flight::BurnExecutorTester tester;
  tester.testStaleAttitudeAbortsTheBurn();
}

TEST(BurnExecutor, ArmedBurnFiresOnItsCycle) {
  flight::BurnExecutorTester tester;
  tester.testArmedBurnFiresOnItsCycle();
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
