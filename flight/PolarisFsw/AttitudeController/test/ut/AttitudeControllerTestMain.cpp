// ======================================================================
// \title  AttitudeControllerTestMain.cpp
// \brief  Entry point for the AttitudeController component tests (§23.1)
// ======================================================================

#include "AttitudeControllerTester.hpp"

TEST(AttitudeController, RefusesWithoutParameters) {
  RecordProperty("verifies", "REQ-ACTL-002");
  flight::AttitudeControllerTester tester;
  tester.testRefusesWithoutParameters();
}

TEST(AttitudeController, PointRefusalPaths) {
  RecordProperty("verifies", "REQ-ACTL-002");
  flight::AttitudeControllerTester tester;
  tester.testPointRefusalPaths();
}

TEST(AttitudeController, PointEngagesAndReducesError) {
  RecordProperty("verifies", "REQ-ACTL-002");
  flight::AttitudeControllerTester tester;
  tester.testPointEngagesAndReducesError();
}

TEST(AttitudeController, DetumbleCommandsOpposingDipole) {
  RecordProperty("verifies", "REQ-ACTL-001");
  flight::AttitudeControllerTester tester;
  tester.testDetumbleCommandsOpposingDipole();
}

TEST(AttitudeController, DutyCycleScheduleInvariants) {
  RecordProperty("verifies", "REQ-ACTL-004");
  flight::AttitudeControllerTester tester;
  tester.testDutyCycleScheduleInvariants();
}

TEST(AttitudeController, StuckOnMonitorLatchesAndClears) {
  RecordProperty("verifies", "REQ-ACTL-005");
  flight::AttitudeControllerTester tester;
  tester.testStuckOnMonitorLatchesAndClears();
}

TEST(AttitudeController, StuckOnAttribution) {
  RecordProperty("verifies", "REQ-ACTL-005");
  flight::AttitudeControllerTester tester;
  tester.testStuckOnAttribution();
}

TEST(AttitudeController, ResetClearsState) {
  flight::AttitudeControllerTester tester;
  tester.testResetClearsState();
}

TEST(AttitudeController, DesatEngagesAndDisengagesInPoint) {
  RecordProperty("verifies", "REQ-ACTL-010");
  flight::AttitudeControllerTester tester;
  tester.testDesatEngagesAndDisengagesInPoint();
}

TEST(AttitudeController, DesatExcludedFromDetumbleAndIdle) {
  RecordProperty("verifies", "REQ-ACTL-010");
  flight::AttitudeControllerTester tester;
  tester.testDesatExcludedFromDetumbleAndIdle();
}

TEST(AttitudeController, DesatGroundOverride) {
  RecordProperty("verifies", "REQ-ACTL-010");
  flight::AttitudeControllerTester tester;
  tester.testDesatGroundOverride();
}

TEST(AttitudeController, WheelFrictionFeedforward) {
  RecordProperty("verifies", "REQ-ACTL-010");
  flight::AttitudeControllerTester tester;
  tester.testWheelFrictionFeedforward();
}

TEST(AttitudeController, WheelCapacityMonitorSeesNullSpaceMomentum) {
  flight::AttitudeControllerTester tester;
  tester.testWheelCapacityMonitorSeesNullSpaceMomentum();
}

TEST(AttitudeController, MomentumEnvelopeAndWheelDropout) {
  RecordProperty("verifies", "REQ-ACTL-009");
  flight::AttitudeControllerTester tester;
  tester.testMomentumEnvelopeAndWheelDropout();
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
