/// @file Unit tests for the CelesTrak space-weather loader (REQ-SIM-002; §3.7).
///
/// Two kinds of check. The synthetic-CSV tests pin the driver *conventions* that
/// a wrong parse would silently get right on magnitude but wrong on meaning: the
/// F10.7 the model reads is the **previous** day's, the 81-day average is the
/// current day's centered value, and only the daily Ap is used. The committed-
/// fixture test pins the CSV *column layout* — a first historical row whose values
/// never change — so a CelesTrak reformat that shifted a column fails loudly here
/// rather than as subtly-wrong drag.

#include <gtest/gtest.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "time/leap_seconds.hpp"
#include "time/timescales.hpp"
#include "time/utc.hpp"
#include "world/space_weather_file.hpp"

namespace world = polaris::sim::world;
namespace pt = polaris::time;

namespace {

/// Build one 31-column CelesTrak row (the full schema width), setting only the
/// fields the loader reads: date@0, AP_AVG@20, F10.7_OBS@24, F10.7_OBS_CENTER81@27.
std::string row(const std::string& date, const std::string& ap, const std::string& f107,
                const std::string& center81) {
  std::vector<std::string> f(31, "0");
  f[0] = date;
  f[20] = ap;
  f[24] = f107;
  f[27] = center81;
  std::string line = f[0];
  for (std::size_t i = 1; i < f.size(); ++i) {
    line += "," + f[i];
  }
  return line;
}

/// A minimal SW-All.csv: header + three daily rows then a monthly-prediction tail
/// whose blank Ap must stop the parse.
std::string writeTempCsv() {
  // mkstemps rather than tmpnam: name minted and file created atomically, so
  // no other process can claim the path in between.
  std::string path = (std::filesystem::temp_directory_path() / "polaris_sw_XXXXXX.csv").string();
  const int fd = mkstemps(path.data(), 4);  // 4 = strlen(".csv")
  EXPECT_GE(fd, 0) << "mkstemps failed for " << path;
  ::close(fd);
  std::ofstream out(path);
  out << "DATE,BSRN,ND,KP1,KP2,KP3,KP4,KP5,KP6,KP7,KP8,KP_SUM,"
         "AP1,AP2,AP3,AP4,AP5,AP6,AP7,AP8,AP_AVG,CP,C9,ISN,F10.7_OBS,F10.7_ADJ,"
         "F10.7_DATA_TYPE,F10.7_OBS_CENTER81,F10.7_OBS_LAST81,F10.7_ADJ_CENTER81,F10.7_ADJ_"
         "LAST81\n";
  out << row("2024-06-14", "10", "100", "110") << "\n";
  out << row("2024-06-15", "20", "200", "120") << "\n";
  out << row("2024-06-16", "30", "300", "130") << "\n";
  out << row("2041-09-01", "", "69", "70") << "\n";  // blank Ap -> parse stops here
  return path;
}

pt::Tai taiAtUtcNoon(int y, unsigned mo, unsigned d) {
  pt::UtcDateTime utc;
  utc.year = y;
  utc.month = mo;
  utc.day = d;
  utc.hour = 12;
  return pt::taiFromUtc(utc, pt::LeapSecondTable::historical());
}

}  // namespace

TEST(SpaceWeatherFile, ParsesDailyRowsAndStopsAtBlankAp) {
  const std::string path = writeTempCsv();
  std::vector<world::SpaceWeatherRecord> rows;
  std::string error;
  ASSERT_TRUE(world::parseSpaceWeatherCsv(path, 0, -1, 0, rows, &error)) << error;
  std::remove(path.c_str());

  // Three daily rows; the blank-Ap monthly tail is not parsed.
  ASSERT_EQ(rows.size(), 3u);
  EXPECT_EQ(rows[0].ap_daily, 10.0);
  EXPECT_EQ(rows[0].f107_obs, 100.0);
  EXPECT_EQ(rows[0].f107_center81, 110.0);
  EXPECT_EQ(rows[2].ap_daily, 30.0);
}

TEST(SpaceWeatherFile, ResolvesMsisDriverConventions) {
  const std::string path = writeTempCsv();
  const pt::LeapSecondTable leap = pt::LeapSecondTable::historical();
  world::SpaceWeatherTable table;
  std::string error;
  ASSERT_TRUE(table.load(path, leap, taiAtUtcNoon(2024, 6, 15), taiAtUtcNoon(2024, 6, 16), &error))
      << error;
  std::remove(path.c_str());

  double f107 = 0.0;
  double f107a = 0.0;
  double ap = 0.0;
  ASSERT_TRUE(table.at(taiAtUtcNoon(2024, 6, 15), f107, f107a, ap));
  // F10.7 lags one day (2024-06-14's 100), F10.7A and Ap are the current day's.
  EXPECT_EQ(f107, 100.0);
  EXPECT_EQ(f107a, 120.0);
  EXPECT_EQ(ap, 20.0);
}

TEST(SpaceWeatherFile, OutOfCoverageEpochFailsClosed) {
  const std::string path = writeTempCsv();
  const pt::LeapSecondTable leap = pt::LeapSecondTable::historical();
  world::SpaceWeatherTable table;
  std::string error;
  ASSERT_TRUE(table.load(path, leap, taiAtUtcNoon(2024, 6, 15), taiAtUtcNoon(2024, 6, 15), &error))
      << error;
  std::remove(path.c_str());

  // A day well outside the loaded 3-day window resolves nothing rather than
  // extrapolating a fabricated atmosphere.
  double f107 = 0.0;
  double f107a = 0.0;
  double ap = 0.0;
  EXPECT_FALSE(table.at(taiAtUtcNoon(2020, 1, 1), f107, f107a, ap));
}

TEST(SpaceWeatherFile, CommittedFixtureColumnLayoutIsStable) {
  // The first row of SW-All.csv is 1957-10-01, a historical record whose values
  // are fixed; assert them so a shifted CSV column is caught here.
  std::vector<world::SpaceWeatherRecord> rows;
  std::string error;
  ASSERT_TRUE(world::parseSpaceWeatherCsv(POLARIS_SW_FIXTURE, 0, -1, 0, rows, &error)) << error;
  ASSERT_FALSE(rows.empty());
  const world::SpaceWeatherRecord& first = rows.front();
  // 1957-10-01 = MJD 36112.
  EXPECT_EQ(first.mjd_day, 36112);
  EXPECT_EQ(first.ap_daily, 21.0);
  EXPECT_EQ(first.f107_obs, 269.3);
  EXPECT_EQ(first.f107_center81, 266.6);
}
