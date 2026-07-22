#ifndef POLARIS_SIM_WORLD_SPACE_WEATHER_FILE_HPP
#define POLARIS_SIM_WORLD_SPACE_WEATHER_FILE_HPP

/// @file
/// @brief Loader for the verbatim CelesTrak `SW-All.csv` space-weather product.
///
/// Reads `tests/golden/SW-All.csv` — committed byte-for-byte as CelesTrak
/// publishes it (design doc §3.7) — into a small per-day table, which is what
/// lets NRLMSIS run against the real solar cycle and geomagnetic history instead
/// of a single hard-coded "moderate activity" snapshot. Without it the truth
/// atmosphere cannot resolve the order-of-magnitude thermospheric swing between
/// solar minimum and maximum, which above ~200 km dominates the drag error
/// budget.
///
/// **What NRLMSIS actually consumes.** Polaris leaves the model's storm-time
/// switches off (`msisinit` defaults), so only the *daily* Ap is read — the eight
/// 3-hourly Ap values in the CSV are not used. The drivers per the standard MSIS
/// convention are: F10.7 for the **previous** day, the centered 81-day average of
/// F10.7, and the daily Ap for the current day. All observed (not 1-AU-adjusted)
/// flux, matching the 81-day column we read.
///
/// **Loading is windowed**, like the EOP loader: `SW-All.csv` spans 1957 to a
/// month or so ahead (~25 000 daily rows) while a scenario needs the few days it
/// covers. Filtering at load keeps the table small.
///
/// Ground/sim-side: file I/O, heap, exceptions-free error returns.

#include <cstddef>
#include <string>
#include <vector>

#include "math/frames.hpp"
#include "math/typed_vector.hpp"
#include "time/leap_seconds.hpp"
#include "time/timescales.hpp"

namespace polaris::sim::world {

/// One parsed daily record, in the units NRLMSIS consumes. Only the fields the
/// model reads are kept; the 3-hourly Ap columns are dropped (see file header).
struct SpaceWeatherRecord {
  long mjd_day{0};            ///< integer Modified Julian Date (00:00 UTC)
  double ap_daily{0.0};       ///< daily Ap index
  double f107_obs{0.0};       ///< observed daily F10.7 [sfu]
  double f107_center81{0.0};  ///< centered 81-day average of observed F10.7 [sfu]
};

/// Parse `SW-All.csv`, keeping only daily records within @p margin_days of the
/// integer-MJD window [@p start_day, @p end_day]. Pass @p end_day < @p start_day
/// to keep everything. Records are returned sorted by day.
///
/// @return false if the file cannot be opened or no usable record survives the
///         window — the latter usually means the scenario epoch is outside the
///         product's coverage, worth failing on rather than propagating with a
///         fabricated atmosphere.
bool parseSpaceWeatherCsv(const std::string& path, long start_day, long end_day, long margin_days,
                          std::vector<SpaceWeatherRecord>& out, std::string* error = nullptr);

/// A windowed per-day table that resolves the MSIS drivers at an arbitrary epoch.
class SpaceWeatherTable {
 public:
  /// Load the records covering [@p start, @p end] (with a small day margin).
  /// @return false on read/parse failure or if no record covers the span.
  bool load(const std::string& path, const time::LeapSecondTable& leap, const time::Tai& start,
            const time::Tai& end, std::string* error = nullptr);

  /// Resolve the drivers at @p epoch: @p f107 = previous day's observed F10.7,
  /// @p f107a = the day's centered 81-day average, @p ap_daily = the day's Ap.
  /// @return false if the day containing @p epoch is outside the loaded window.
  bool at(const time::Tai& epoch, double& f107, double& f107a, double& ap_daily) const;

  bool empty() const { return records_.empty(); }

 private:
  const SpaceWeatherRecord* find(long mjd_day) const;

  std::vector<SpaceWeatherRecord> records_;  ///< sorted by mjd_day
  const time::LeapSecondTable* leap_{nullptr};
};

}  // namespace polaris::sim::world

#endif  // POLARIS_SIM_WORLD_SPACE_WEATHER_FILE_HPP
