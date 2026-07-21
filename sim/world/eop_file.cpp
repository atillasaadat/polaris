#include "world/eop_file.hpp"

#include <cmath>
#include <fstream>

#include "time/civil.hpp"
#include "time/utc.hpp"

namespace polaris::sim::world {
namespace {

/// MJD of the Unix epoch, 1970-01-01.
constexpr double kMjd1970 = 40587.0;
constexpr double kSecondsPerDay = 86400.0;

/// Bulletin A fixed-width column offsets, matching `tools/eop/finals.py`.
/// A record is usable only once the prediction flag and value are both present;
/// the file's tail has date lines with empty value fields, and parsing those as
/// zeros would inject a hard step to zero Earth orientation at the end of
/// coverage — far worse than stopping.
constexpr std::size_t kMjdOffset = 7;
constexpr std::size_t kMjdWidth = 8;
constexpr std::size_t kXpOffset = 18;
constexpr std::size_t kYpOffset = 37;
constexpr std::size_t kFieldWidth = 9;
constexpr std::size_t kDut1Offset = 58;
constexpr std::size_t kDut1Width = 10;
constexpr std::size_t kMinLineLength = kDut1Offset + kDut1Width;

/// Parse a fixed-width numeric field. False if it is blank or not a number.
bool field(const std::string& line, std::size_t offset, std::size_t width, double& out) {
  if (line.size() < offset + width) {
    return false;
  }
  const std::string text = line.substr(offset, width);
  if (text.find_first_not_of(" \t") == std::string::npos) {
    return false;
  }
  try {
    std::size_t consumed = 0;
    out = std::stod(text, &consumed);
    return consumed > 0;
  } catch (const std::exception&) {
    return false;
  }
}

}  // namespace

double mjdUtcOf(const time::Tai& t, const time::LeapSecondTable& leap) {
  const time::UtcDateTime utc = time::utcFromTai(t, leap);
  const double days = static_cast<double>(time::daysFromCivil(utc.year, utc.month, utc.day));
  const double seconds = static_cast<double>(utc.hour) * 3600.0 +
                         static_cast<double>(utc.minute) * 60.0 + static_cast<double>(utc.second);
  return kMjd1970 + days + seconds / kSecondsPerDay;
}

bool parseFinals(const std::string& path, double start_mjd, double end_mjd, double margin_days,
                 std::vector<FinalsRow>& out, std::string* error) {
  std::ifstream file(path);
  if (!file.good()) {
    if (error != nullptr) {
      *error = "cannot open EOP file: " + path;
    }
    return false;
  }

  const bool windowed = end_mjd >= start_mjd;
  const double lower = start_mjd - margin_days;
  const double upper = end_mjd + margin_days;

  out.clear();
  std::string line;
  while (std::getline(file, line)) {
    if (line.size() < kMinLineLength) {
      continue;
    }
    FinalsRow row;
    // dUT1 is the field that runs out first: records past it are date-only
    // placeholders, so a row missing it is skipped rather than turned into a
    // zero-filled entry (which would be a hard step to zero Earth orientation).
    // Skipping, rather than stopping at the first blank, means an interleaved
    // provisional gap — which IERS does occasionally publish — still yields the
    // valid rows on both sides instead of silently truncating coverage there.
    if (!field(line, kMjdOffset, kMjdWidth, row.mjd_utc) ||
        !field(line, kDut1Offset, kDut1Width, row.dut1_s) ||
        !field(line, kXpOffset, kFieldWidth, row.xp_arcsec) ||
        !field(line, kYpOffset, kFieldWidth, row.yp_arcsec)) {
      continue;
    }
    if (windowed && (row.mjd_utc < lower || row.mjd_utc > upper)) {
      continue;
    }
    out.push_back(row);
  }

  if (out.size() < 2) {
    if (error != nullptr) {
      *error = "EOP file " + path + " yielded fewer than two usable records" +
               (windowed ? " in the requested window (scenario epoch outside coverage?)" : "");
    }
    return false;
  }
  return true;
}

}  // namespace polaris::sim::world
