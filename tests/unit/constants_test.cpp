/// @file Unit tests for the shared physical-constants registry.

#include "constants/constants.hpp"

#include <gtest/gtest.h>

namespace pc = polaris::constants;

TEST(Constants, Wgs84EllipsoidRelationships) {
  RecordProperty("verifies", "REQ-CONV-003");
  EXPECT_NEAR(pc::wgs84::kEccentricitySq, pc::wgs84::kFlattening * (2.0 - pc::wgs84::kFlattening),
              1e-18);
  EXPECT_NEAR(pc::wgs84::kSemiMinorAxis, pc::wgs84::kSemiMajorAxis * (1.0 - pc::wgs84::kFlattening),
              1e-6);
  EXPECT_GT(pc::wgs84::kSemiMajorAxis, pc::wgs84::kSemiMinorAxis);
  EXPECT_NEAR(pc::wgs84::kSemiMajorAxis, 6378137.0, 1e-6);
  EXPECT_GT(pc::wgs84::kGM, 3.9e14);
  EXPECT_LT(pc::wgs84::kGM, 4.0e14);
  EXPECT_GT(pc::wgs84::kEarthRate, 7.0e-5);
}

TEST(Constants, TimeOffsets) {
  RecordProperty("verifies", "REQ-SYS-001");
  EXPECT_DOUBLE_EQ(pc::time::kTaiMinusGps, 19.0);
  EXPECT_DOUBLE_EQ(pc::time::kTtMinusTai, 32.184);
  EXPECT_DOUBLE_EQ(pc::time::kSecondsPerDay, 86400.0);
}

TEST(Constants, PhysicalExactValues) {
  RecordProperty("verifies", "REQ-SYS-002");
  EXPECT_DOUBLE_EQ(pc::physical::kSpeedOfLight, 299792458.0);
  EXPECT_DOUBLE_EQ(pc::physical::kStandardGravity, 9.80665);
}
