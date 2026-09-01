/// @file Unit tests for the NORAD two-line element parser (§8.3, REQ-ODP-003).
///
/// The propagator's acceptance test is the official verification set
/// (`tests/golden/sgp4_verification_golden_test.cpp`); this file covers the
/// *input* half, where the failure modes are textual rather than numerical and
/// a golden comparison would not reach them. The three encodings worth pinning
/// are the assumed decimal point, the assumed exponent, and the two-digit year
/// window — each of which produces a plausible wrong orbit rather than an error
/// when it is got wrong.

#include "gnc/tle.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <string>

#include "time/leap_seconds.hpp"
#include "time/utc.hpp"

namespace gnc = polaris::gnc;
namespace pt = polaris::time;

namespace {

/// ISS-like element set with valid checksums, used as the well-formed baseline.
/// Mirrors the format exactly; the numbers themselves are not asserted against
/// a catalogue, only the parse.
constexpr const char* kLine1 =
    "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927";
constexpr const char* kLine2 =
    "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537";

/// One of the verification set's first cases (satellite 5), whose eccentricity
/// and drag term exercise both assumed-format fields with non-trivial values.
constexpr const char* kSat5Line1 =
    "1 00005U 58002B   00179.78495062  .00000023  00000-0  28098-4 0  4753";
constexpr const char* kSat5Line2 =
    "2 00005  34.2682 348.7242 1859667 331.7664  19.3264 10.82419157413667";

}  // namespace

TEST(Tle, ParsesAWellFormedElementSet) {
  gnc::TleElements e;
  ASSERT_EQ(gnc::parseTle(kLine1, kLine2, e), gnc::TleStatus::kOk);
  EXPECT_EQ(e.satellite_number, 25544);
  EXPECT_EQ(e.classification, 'U');
  EXPECT_EQ(e.epoch_year, 2008);
  EXPECT_NEAR(e.epoch_day, 264.51782528, 1e-9);
  EXPECT_NEAR(e.inclination_deg, 51.6416, 1e-9);
  EXPECT_NEAR(e.raan_deg, 247.4627, 1e-9);
  EXPECT_NEAR(e.arg_perigee_deg, 130.5360, 1e-9);
  EXPECT_NEAR(e.mean_anomaly_deg, 325.0288, 1e-9);
  EXPECT_NEAR(e.mean_motion_rev_per_day, 15.72125391, 1e-8);
  EXPECT_EQ(e.revolution_number, 56353);
}

TEST(Tle, EccentricityCarriesAnAssumedLeadingDecimalPoint) {
  // `0006703` is 0.0006703, not 6703. Reading it as an integer would produce a
  // hyperbolic "orbit"; reading it with the point one column off produces a
  // perfectly plausible wrong one, which is the dangerous case.
  gnc::TleElements e;
  ASSERT_EQ(gnc::parseTle(kLine1, kLine2, e), gnc::TleStatus::kOk);
  EXPECT_NEAR(e.eccentricity, 0.0006703, 1e-12);

  gnc::TleElements s5;
  ASSERT_EQ(gnc::parseTle(kSat5Line1, kSat5Line2, s5), gnc::TleStatus::kOk);
  EXPECT_NEAR(s5.eccentricity, 0.1859667, 1e-12);
}

TEST(Tle, DragTermCarriesAnAssumedExponent) {
  // `-11606-4` is -0.11606e-4; `28098-4` is +0.28098e-4. The sign of the
  // mantissa and the sign of the exponent are separate fields and both matter.
  gnc::TleElements e;
  ASSERT_EQ(gnc::parseTle(kLine1, kLine2, e), gnc::TleStatus::kOk);
  EXPECT_NEAR(e.bstar, -0.11606e-4, 1e-16);

  gnc::TleElements s5;
  ASSERT_EQ(gnc::parseTle(kSat5Line1, kSat5Line2, s5), gnc::TleStatus::kOk);
  EXPECT_NEAR(s5.bstar, 0.28098e-4, 1e-16);
  // An all-zero field is zero, not a parse failure.
  EXPECT_EQ(s5.nddot, 0.0);
}

TEST(Tle, TwoDigitYearUsesTheSpaceTrackWindow) {
  // 57-99 -> 1957-1999, 00-56 -> 2000-2056. Satellite 5's epoch year `00` is
  // 2000, and the 1958-launched object proves the window is not "launch year".
  gnc::TleElements s5;
  ASSERT_EQ(gnc::parseTle(kSat5Line1, kSat5Line2, s5), gnc::TleStatus::kOk);
  EXPECT_EQ(s5.epoch_year, 2000);

  // Hand-build a `98` epoch and confirm it lands in 1998.
  std::string l1 = kLine1;
  l1[18] = '9';
  l1[19] = '8';
  // Fix the checksum for the edited line so the edit itself is not what fails.
  const int want = gnc::tleChecksum(l1);
  l1[68] = static_cast<char>('0' + want);
  gnc::TleElements e;
  ASSERT_EQ(gnc::parseTle(l1, kLine2, e), gnc::TleStatus::kOk);
  EXPECT_EQ(e.epoch_year, 1998);
}

TEST(Tle, EpochConvertsToTaiThroughUtc) {
  gnc::TleElements s5;
  ASSERT_EQ(gnc::parseTle(kSat5Line1, kSat5Line2, s5), gnc::TleStatus::kOk);
  const pt::LeapSecondTable leap;
  pt::Tai epoch;
  ASSERT_TRUE(s5.epochTai(leap, epoch));
  // Day 179.78495062 of 2000 is 2000-06-27, a little under 19 hours in.
  // Checked as a round trip rather than against a transcribed constant.
  const pt::UtcDateTime utc = pt::utcFromTai(epoch, leap);
  EXPECT_EQ(utc.year, 2000);
  EXPECT_EQ(utc.month, 6u);
  EXPECT_EQ(utc.day, 27u);
  EXPECT_EQ(utc.hour, 18u);
  EXPECT_EQ(utc.minute, 50u);
}

TEST(Tle, DeepSpaceIsDecidedOnThe225MinutePeriod) {
  gnc::TleElements e;
  ASSERT_EQ(gnc::parseTle(kLine1, kLine2, e), gnc::TleStatus::kOk);
  // The ISS at ~15.7 rev/day is a 91.6 minute orbit: near-Earth.
  EXPECT_NEAR(e.periodMinutes(), 1440.0 / 15.72125391, 1e-6);
  EXPECT_FALSE(e.isDeepSpace());

  // A 12-hour orbit is deep space.
  std::string l2 = kLine2;
  const std::string mm = " 2.00000000";
  for (std::size_t i = 0; i < mm.size(); ++i) {
    l2[52 + i] = mm[i];
  }
  l2[68] = static_cast<char>('0' + gnc::tleChecksum(l2));
  gnc::TleElements deep;
  ASSERT_EQ(gnc::parseTle(kLine1, l2, deep), gnc::TleStatus::kOk);
  EXPECT_TRUE(deep.isDeepSpace());
}

TEST(Tle, ChecksumIsVerifiedByDefaultAndCanBeWaived) {
  // A single flipped digit is exactly what the checksum exists to catch, and it
  // is also exactly what would otherwise produce a plausible wrong orbit.
  std::string corrupted = kLine2;
  corrupted[10] = (corrupted[10] == '1') ? '2' : '1';  // perturb the inclination
  gnc::TleElements e;
  EXPECT_EQ(gnc::parseTle(kLine1, corrupted, e), gnc::TleStatus::kBadChecksum);
  // ...and a caller who knows the element set was hand-written may say so.
  EXPECT_EQ(gnc::parseTle(kLine1, corrupted, e, gnc::TleChecksumPolicy::kIgnore),
            gnc::TleStatus::kOk);
}

TEST(Tle, ChecksumCountsMinusSignsAsOne) {
  // The one rule in the checksum that is not "sum the digits". Line 1 above has
  // two minus signs, so getting this wrong shifts the sum by two.
  EXPECT_EQ(gnc::tleChecksum(kLine1), kLine1[68] - '0');
  EXPECT_EQ(gnc::tleChecksum(kLine2), kLine2[68] - '0');
}

TEST(Tle, MismatchedSatelliteNumbersAreRefused) {
  // Two lines from different objects is a splice, and it produces an element set
  // that is individually well-formed and physically meaningless.
  std::string l2 = kLine2;
  l2[6] = '3';
  l2[68] = static_cast<char>('0' + gnc::tleChecksum(l2));
  gnc::TleElements e;
  EXPECT_EQ(gnc::parseTle(kLine1, l2, e), gnc::TleStatus::kSatelliteMismatch);
}

TEST(Tle, StructurallyBrokenInputIsNamedNotGuessed) {
  gnc::TleElements e;
  EXPECT_EQ(gnc::parseTle("1 25544U", kLine2, e), gnc::TleStatus::kLineTooShort);
  EXPECT_EQ(gnc::parseTle(kLine2, kLine1, e), gnc::TleStatus::kBadLineNumber);
}

TEST(Tle, PhysicallyImpossibleElementsAreRefused) {
  // The checksum passes on these by construction, so the range gate is the only
  // thing standing between them and a propagator that would happily run.
  std::string l2 = kLine2;
  const std::string ecc = "9999999";  // e = 0.9999999 is legal; 1.0 would not be
  for (std::size_t i = 0; i < ecc.size(); ++i) {
    l2[26 + i] = ecc[i];
  }
  l2[68] = static_cast<char>('0' + gnc::tleChecksum(l2));
  gnc::TleElements e;
  EXPECT_EQ(gnc::parseTle(kLine1, l2, e), gnc::TleStatus::kOk);

  // Zero mean motion has no period at all.
  std::string zero_mm = kLine2;
  const std::string mm = " 0.00000000";
  for (std::size_t i = 0; i < mm.size(); ++i) {
    zero_mm[52 + i] = mm[i];
  }
  zero_mm[68] = static_cast<char>('0' + gnc::tleChecksum(zero_mm));
  EXPECT_EQ(gnc::parseTle(kLine1, zero_mm, e), gnc::TleStatus::kOutOfRange);
}

TEST(Tle, TrailingDriverFieldsAndCarriageReturnsAreIgnored) {
  // The committed verification fixture appends start/stop/step to line 2, and
  // files arrive with CRLF. Neither is part of the format and neither may
  // change the parse.
  gnc::TleElements plain;
  ASSERT_EQ(gnc::parseTle(kSat5Line1, kSat5Line2, plain), gnc::TleStatus::kOk);

  const std::string annotated = std::string(kSat5Line2) + "     0.00      4320.0        360.00";
  const std::string crlf = std::string(kSat5Line1) + "\r";
  gnc::TleElements decorated;
  ASSERT_EQ(gnc::parseTle(crlf, annotated, decorated), gnc::TleStatus::kOk);
  EXPECT_EQ(decorated.satellite_number, plain.satellite_number);
  EXPECT_EQ(decorated.mean_motion_rev_per_day, plain.mean_motion_rev_per_day);
  EXPECT_EQ(decorated.eccentricity, plain.eccentricity);
}

TEST(Tle, ARefusedParseLeavesTheOutputUntouched) {
  // A caller that ignores the status must not find a half-populated element set
  // that looks usable (§3.6).
  gnc::TleElements e;
  ASSERT_EQ(gnc::parseTle(kLine1, kLine2, e), gnc::TleStatus::kOk);
  const gnc::TleElements before = e;
  EXPECT_EQ(gnc::parseTle("1 short", kLine2, e), gnc::TleStatus::kLineTooShort);
  EXPECT_EQ(e.satellite_number, before.satellite_number);
  EXPECT_EQ(e.mean_motion_rev_per_day, before.mean_motion_rev_per_day);
}
