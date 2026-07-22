#include "world/space_weather_file.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>

#include "time/civil.hpp"
#include "world/eop_file.hpp"  // mjdUtcOf

namespace polaris::sim::world {

namespace {

/// MJD of 1970-01-01 (the epoch `daysFromCivil` counts from).
constexpr long kMjdUnixEpoch = 40587;

// 0-indexed CSV field positions in CelesTrak SW-All.csv (see the fetch tool).
constexpr std::size_t kDate = 0;
constexpr std::size_t kApAvg = 20;
constexpr std::size_t kF107Obs = 24;
constexpr std::size_t kF107Center81 = 27;
// The full CelesTrak schema is 31 columns. Requiring all of them (not just the
// 28 we read) rejects a truncated row outright rather than letting the last
// column we consume slide onto the line's trailing field — which under the file's
// CRLF endings would carry a stray '\r'. F10.7_OBS_CENTER81 (idx 27) is interior
// to a 31-field row, so it never does today; this keeps it that way.
constexpr std::size_t kMinFields = 31;

bool fail(std::string* error, const std::string& message) {
  if (error != nullptr) {
    *error = message;
  }
  return false;
}

/// Parse "YYYY-MM-DD" to an integer MJD day. Returns false on a malformed date.
bool mjdDayOfDate(const std::string& date, long& out) {
  int year = 0;
  unsigned month = 0;
  unsigned day = 0;
  if (std::sscanf(date.c_str(), "%d-%u-%u", &year, &month, &day) != 3 || month < 1 || month > 12 ||
      day < 1 || day > 31) {
    return false;
  }
  out = static_cast<long>(time::daysFromCivil(year, month, day)) + kMjdUnixEpoch;
  return true;
}

}  // namespace

bool parseSpaceWeatherCsv(const std::string& path, long start_day, long end_day, long margin_days,
                          std::vector<SpaceWeatherRecord>& out, std::string* error) {
  std::ifstream file(path);
  if (!file.good()) {
    return fail(error, "cannot open space-weather file: " + path);
  }

  const bool windowed = end_day >= start_day;
  const long lower = start_day - margin_days;
  const long upper = end_day + margin_days;

  std::string line;
  std::getline(file, line);  // discard the header row
  std::vector<std::string> fields;
  while (std::getline(file, line)) {
    fields.clear();
    std::stringstream ss(line);
    std::string field;
    while (std::getline(ss, field, ',')) {
      fields.push_back(field);
    }
    if (fields.size() < kMinFields) {
      continue;
    }
    // A blank daily Ap marks the end of the daily record (the monthly-prediction
    // tail carries F10.7 only). Stop rather than fabricating geomagnetic activity.
    if (fields[kApAvg].empty()) {
      break;
    }

    SpaceWeatherRecord rec;
    if (!mjdDayOfDate(fields[kDate], rec.mjd_day)) {
      continue;
    }
    if (windowed) {
      // The product is chronological, so once past the window there is nothing
      // left to find — stop rather than scanning the remaining ~25k daily rows.
      if (rec.mjd_day > upper) {
        break;
      }
      if (rec.mjd_day < lower) {
        continue;
      }
    }
    try {
      rec.ap_daily = std::stod(fields[kApAvg]);
      rec.f107_obs = std::stod(fields[kF107Obs]);
      rec.f107_center81 = std::stod(fields[kF107Center81]);
    } catch (const std::exception&) {
      continue;  // a partially-filled row: skip rather than abort the whole load
    }
    out.push_back(rec);
  }

  if (out.empty()) {
    return fail(error, "no usable space-weather records in window: " + path);
  }
  std::sort(out.begin(), out.end(), [](const SpaceWeatherRecord& a, const SpaceWeatherRecord& b) {
    return a.mjd_day < b.mjd_day;
  });
  return true;
}

bool SpaceWeatherTable::load(const std::string& path, const time::LeapSecondTable& leap,
                             const time::Tai& start, const time::Tai& end, std::string* error) {
  // Daily granularity: the previous-day F10.7 needs one day before the span, and
  // a couple of days of slack keeps a run ending just past midnight in-window.
  constexpr long kMarginDays = 2;
  const long start_day = static_cast<long>(std::floor(mjdUtcOf(start, leap)));
  const long end_day = static_cast<long>(std::floor(mjdUtcOf(end, leap)));
  records_.clear();
  if (!parseSpaceWeatherCsv(path, start_day, end_day, kMarginDays, records_, error)) {
    return false;
  }
  leap_ = &leap;
  return true;
}

const SpaceWeatherRecord* SpaceWeatherTable::find(long mjd_day) const {
  const auto it =
      std::lower_bound(records_.begin(), records_.end(), mjd_day,
                       [](const SpaceWeatherRecord& rec, long day) { return rec.mjd_day < day; });
  if (it != records_.end() && it->mjd_day == mjd_day) {
    return &*it;
  }
  return nullptr;
}

bool SpaceWeatherTable::at(const time::Tai& epoch, double& f107, double& f107a,
                           double& ap_daily) const {
  if (records_.empty() || leap_ == nullptr) {
    return false;
  }
  const long day = static_cast<long>(std::floor(mjdUtcOf(epoch, *leap_)));
  const SpaceWeatherRecord* today = find(day);
  if (today == nullptr) {
    return false;
  }
  f107a = today->f107_center81;
  ap_daily = today->ap_daily;
  // MSIS convention: F10.7 lags one day. At the very start of the file (no prior
  // day in-window) fall back to the same day rather than reporting no atmosphere.
  const SpaceWeatherRecord* prev = find(day - 1);
  f107 = (prev != nullptr) ? prev->f107_obs : today->f107_obs;
  return true;
}

}  // namespace polaris::sim::world
