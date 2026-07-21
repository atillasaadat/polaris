#ifndef POLARIS_SIM_WORLD_EOP_FILE_HPP
#define POLARIS_SIM_WORLD_EOP_FILE_HPP

/// @file
/// @brief Loader for the verbatim IERS `finals.all.iau2000` product.
///
/// Reads `tests/golden/finals.all.iau2000.txt` — committed byte-for-byte as IERS
/// publishes it (design doc §3.7) — into a `frames::EopTable`, which is what
/// makes the ECI↔ECEF reduction (and therefore the ECEF-frame gravity, NRLMSIS,
/// and IGRF paths) work against real Earth orientation instead of an identity
/// rotation.
///
/// The Bulletin A column spec was previously transcribed in two places —
/// `tools/eop/finals.py` on the ground side and inline in the EOP golden test.
/// This is the C++ one, and the golden test now uses it rather than keeping a
/// private copy: a fixed-width column spec duplicated across files is exactly
/// the kind of thing that gets fixed in one copy and not the other.
///
/// **Loading is windowed.** `finals.all` carries every day from 1973 to roughly a
/// year ahead — about 20 000 records — while a scenario needs the handful of days
/// it actually spans. Filtering at load keeps the fixed-capacity table small
/// enough to hold as a member instead of forcing a 20 000-entry allocation onto
/// every consumer. The window is widened by a margin so interpolation at the
/// endpoints still has neighbours.
///
/// Ground/sim-side: file I/O, heap, exceptions-free error returns.

#include <cstddef>
#include <string>
#include <vector>

#include "frames/eop.hpp"
#include "time/leap_seconds.hpp"
#include "time/timescales.hpp"

namespace polaris::sim::world {

/// One parsed Bulletin A record. Fixed-width columns of `finals.all.iau2000`;
/// mirrors `tools/eop/finals.py`.
struct FinalsRow {
  double mjd_utc{0.0};
  double dut1_s{0.0};     ///< UT1 − UTC [s]
  double xp_arcsec{0.0};  ///< polar motion x
  double yp_arcsec{0.0};  ///< polar motion y
};

/// Parse the file, keeping only records within @p margin_days of
/// [@p start, @p end]. Pass an empty range to keep everything.
///
/// @return false if the file cannot be opened or no usable record survives the
///         window — the latter usually means the scenario epoch is outside the
///         product's coverage, which is worth failing on rather than silently
///         propagating with no Earth orientation.
bool parseFinals(const std::string& path, double start_mjd, double end_mjd, double margin_days,
                 std::vector<FinalsRow>& out, std::string* error = nullptr);

/// Modified Julian Date (UTC) of a TAI instant, for building a load window.
double mjdUtcOf(const time::Tai& t, const time::LeapSecondTable& leap);

/// Load into @p table the records covering [@p start, @p end] with margin.
///
/// @return false on a read/parse failure, if no record covers the span, or if
///         the window does not fit in @p Capacity.
template <std::size_t Capacity>
bool loadEopFile(const std::string& path, const time::LeapSecondTable& leap, const time::Tai& start,
                 const time::Tai& end, frames::EopTable<Capacity>& table,
                 std::string* error = nullptr) {
  // Bulletin A is daily, so a few days either side is plenty for the linear
  // interpolation the table does.
  constexpr double kMarginDays = 5.0;
  std::vector<FinalsRow> rows;
  if (!parseFinals(path, mjdUtcOf(start, leap), mjdUtcOf(end, leap), kMarginDays, rows, error)) {
    return false;
  }
  if (rows.size() > Capacity) {
    if (error != nullptr) {
      *error = "EOP window needs " + std::to_string(rows.size()) + " entries but capacity is " +
               std::to_string(Capacity);
    }
    return false;
  }
  for (const FinalsRow& r : rows) {
    if (!table.addEntry({r.mjd_utc, r.dut1_s, r.xp_arcsec, r.yp_arcsec})) {
      if (error != nullptr) {
        *error = "EOP table rejected an entry (out of order or full)";
      }
      return false;
    }
  }
  return true;
}

}  // namespace polaris::sim::world

#endif  // POLARIS_SIM_WORLD_EOP_FILE_HPP
