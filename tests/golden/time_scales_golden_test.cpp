/// @file Golden-fixture comparison for the time scales (design doc §23.1, REQ-VV-002).
///
/// Loads the committed reference fixture `tests/golden/time_scales.json` and checks
/// `lib/time` (Push 2b) against it within each quantity's documented tolerance
/// band. The fixture values are published constants (IERS leap seconds; the IAU
/// definitional TT−TAI / TAI−GPS offsets), independent of the Polaris code they
/// verify, and are regenerable from GMAT via `tools/gmat/` — this test never runs
/// GMAT, it only reads the checked-in fixture (REQ-SYS-010, REQ-VV-002).

#include <gtest/gtest.h>

#include <cstdint>
#include <fstream>
#include <nlohmann/json.hpp>
#include <string>

#include "time/leap_seconds.hpp"
#include "time/timescales.hpp"
#include "time/utc.hpp"

namespace pt = polaris::time;
using json = nlohmann::json;

namespace {

// Returns a null json if the fixture is missing, or a discarded json if it is
// malformed; the caller hard-stops on either. Avoids letting a failed stream
// reach nlohmann and surface as an opaque parse exception instead of the path.
json LoadFixture(const std::string& name) {
  const std::string path = std::string(GOLDEN_DIR) + "/" + name;
  std::ifstream file(path);
  if (!file.is_open()) {
    ADD_FAILURE() << "cannot open golden fixture: " << path;
    return json{};
  }
  return json::parse(file, /*cb=*/nullptr, /*allow_exceptions=*/false);
}

pt::UtcDateTime ParseUtc(const json& u) {
  pt::UtcDateTime utc;
  utc.year = u.at("year").get<std::int64_t>();
  utc.month = u.at("month").get<unsigned>();
  utc.day = u.at("day").get<unsigned>();
  utc.hour = u.at("hour").get<unsigned>();
  utc.minute = u.at("minute").get<unsigned>();
  utc.second = u.at("second").get<unsigned>();
  utc.nanosecond = u.at("nanosecond").get<std::int32_t>();
  return utc;
}

// Compute one named scale offset [s] from lib/time for the given UTC epoch.
double ComputeQuantity(const std::string& name, const pt::UtcDateTime& utc) {
  const pt::LeapSecondTable historical = pt::LeapSecondTable::historical();
  const pt::Tai tai = pt::taiFromUtc(utc, historical);

  if (name == "tai_minus_utc_s") {
    // TAI−UTC is the leap offset: the same calendar fields mapped with a zero-ΔAT
    // table give the naive linear instant, so the difference is the offset.
    const pt::Tai tai_zero = pt::taiFromUtc(utc, pt::LeapSecondTable::frozen(0));
    return (tai - tai_zero).seconds();
  }
  if (name == "tt_minus_tai_s") {
    return static_cast<double>(pt::toTt(tai).nanosecondsSinceEpoch() -
                               tai.nanosecondsSinceEpoch()) /
           1.0e9;
  }
  if (name == "tai_minus_gps_s") {
    return static_cast<double>(tai.nanosecondsSinceEpoch() -
                               pt::toGps(tai).nanosecondsSinceEpoch()) /
           1.0e9;
  }
  ADD_FAILURE() << "unknown golden quantity: " << name;
  return 0.0;
}

}  // namespace

TEST(TimeScalesGolden, MatchesGmatFixtureWithinTolerance) {
  RecordProperty("verifies", "REQ-VV-002");
  const json fixture = LoadFixture("time_scales.json");
  ASSERT_FALSE(fixture.is_null() || fixture.is_discarded())
      << "golden fixture missing or malformed";
  ASSERT_EQ(fixture.at("schema_version").get<std::string>(), "1.0");

  int checked = 0;
  for (const auto& c : fixture.at("cases")) {
    const pt::UtcDateTime utc = ParseUtc(c.at("utc"));
    const std::string case_name = c.at("name").get<std::string>();
    for (const auto& [qname, spec] : c.at("quantities").items()) {
      const double expected = spec.at("expected").get<double>();
      const double tol = spec.at("tol_abs").get<double>();
      const double actual = ComputeQuantity(qname, utc);
      EXPECT_NEAR(actual, expected, tol) << "case '" << case_name << "' quantity '" << qname << "'";
      ++checked;
    }
  }
  EXPECT_GT(checked, 0) << "fixture contributed no comparisons";
}
