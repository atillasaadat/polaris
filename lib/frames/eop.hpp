#ifndef POLARIS_FRAMES_EOP_HPP
#define POLARIS_FRAMES_EOP_HPP

/// @file
/// @brief Onboard IERS Earth Orientation Parameter table (REQ-CONV-002, REQ-CDH-002).
///
/// The measured/predicted Earth-orientation quantities the IAU 2006/2000A
/// ECI↔ECEF reduction needs beyond theory: **UT1 - TAI** (Earth's actual
/// rotation phase, which no model predicts) and the **polar motion** `x_p, y_p`
/// (the wander of the rotation pole in the crust). IERS publishes them daily in
/// `finals2000A.all`; `tools/eop/` trims that to the committed JSON fixture.
///
/// Onboard this is an **uploaded** low-rate table (design doc §11.3, REQ-CDH-002),
/// not a file the FSW reads: entries are pushed in via `addEntry()`, exactly as
/// the ephemeris and leap-second tables are. Fixed capacity, no heap, no
/// exceptions (flight-safe, §3.6); lookups are a bounded linear scan over a small
/// ascending table.
///
/// **UT1 - TAI, not UT1 - UTC — this is the whole trick.** IERS publishes
/// ΔUT1 = UT1 - UTC, which *steps by a full second* at every leap second, because
/// UTC steps and UT1 does not. Linearly interpolating ΔUT1 across a leap-second
/// boundary therefore smears a 1 s error across the surrounding days — a ~465 m/s
/// × 1 s ≈ 465 m ground-track error at its worst, and a classic bug. UT1 - TAI is
/// continuous (TAI never steps), so this table stores and interpolates *that*,
/// reconstructing it per entry as `(UT1 - UTC) + ΔAT(entry date)` at load time.
/// The interpolation argument is TAI for the same reason: UTC is not a uniform
/// argument across a leap either. See `EopTable::lookup()`.
///
/// Interpolation is **linear** between daily entries — the standard treatment for
/// the daily IERS series (IERS Conventions §5.5.1; the sub-daily tidal/libration
/// terms it omits are ~0.1 mas / ~10 µs, far below our error budget). Epochs
/// outside the table's span are **rejected** (`lookup()` returns false); the
/// series is not extrapolated, since UT1 drift is unmodellable and a silent
/// extrapolation would be a wrong answer rather than a detectable failure.
///
/// References:
///  - IERS Conventions (2010), IERS TN 36, §5 (Earth orientation, EOP usage).
///    [iers2010]
///  - Vallado, *Fundamentals of Astrodynamics and Applications*, 4th ed., §3.7
///    (EOP, ΔUT1, polar motion). [vallado2013]

#include <cmath>
#include <cstddef>

#include "constants/constants.hpp"
#include "time/civil.hpp"
#include "time/leap_seconds.hpp"
#include "time/timescales.hpp"

namespace polaris::frames {

/// One daily IERS EOP record, in the units IERS publishes them.
struct EopEntry {
  double mjd_utc{0.0};    ///< UTC Modified Julian Date of the record (integral, 00:00 UTC)
  double dut1{0.0};       ///< UT1 - UTC [s] (Bulletin A); stepped at leap seconds
  double xp_arcsec{0.0};  ///< polar motion x [arcsec]
  double yp_arcsec{0.0};  ///< polar motion y [arcsec]
};

/// EOP interpolated to an epoch, in the units the reduction consumes.
struct EopValue {
  double ut1_minus_tai{0.0};  ///< UT1 - TAI [s] — continuous across leap seconds
  double xp_arcsec{0.0};      ///< polar motion x [arcsec]
  double yp_arcsec{0.0};      ///< polar motion y [arcsec]
};

/// UT1 as an `Instant`, from the master clock and the table's `UT1 - TAI`. The
/// only route to UT1 (see `time::scale::Ut1`).
inline time::Ut1 ut1FromTai(const time::Tai& tai, double ut1_minus_tai) {
  return time::Ut1::fromNanosecondsSinceEpoch(
      tai.nanosecondsSinceEpoch() + time::Duration::fromSecondsF(ut1_minus_tai).nanoseconds());
}

/// MJD offset of the uniform-scale epoch: MJD 40587 = 1970-01-01.
inline constexpr double kMjd1970 =
    constants::time::kJulianDate1970 - constants::time::kJulianDateToMjd;

/// An ascending, fixed-capacity EOP series (design doc §11.3, REQ-CDH-002).
/// @tparam Capacity maximum number of daily records (compile-time; no heap).
template <std::size_t Capacity>
class EopTable {
 public:
  static_assert(Capacity >= 2, "interpolation needs room for at least two records");

  /// Append a record. Records must be added in **ascending MJD order**. Returns
  /// false (and ignores the record) if the table is full, any field is
  /// non-finite, or the MJD is not strictly after the previous one — so a lookup
  /// never has to re-validate structure, and an out-of-order upload cannot
  /// silently corrupt the series.
  bool addEntry(const EopEntry& e) {
    if (count_ >= Capacity) {
      return false;
    }
    if (!std::isfinite(e.mjd_utc) || !std::isfinite(e.dut1) || !std::isfinite(e.xp_arcsec) ||
        !std::isfinite(e.yp_arcsec)) {
      return false;
    }
    if (count_ > 0 && !(e.mjd_utc > entries_[count_ - 1].mjd_utc)) {
      return false;
    }
    entries_[count_] = e;
    ++count_;
    return true;
  }

  /// Number of records currently held.
  std::size_t size() const { return count_; }

  /// True if no lookup can succeed (fewer than two records to interpolate).
  bool empty() const { return count_ < 2; }

  /// EOP at @p t, linearly interpolated between the bracketing daily records.
  /// Returns false — leaving @p out untouched — if the table holds fewer than two
  /// records or @p t lies outside their span (§3.6: no extrapolation, no throw).
  ///
  /// Both the interpolated quantity (`UT1 - TAI`) and the interpolation argument
  /// (TAI) are continuous across a leap second; @p leap supplies the ΔAT that
  /// converts each record's published UTC/ΔUT1 to that continuous form. See the
  /// file header for why interpolating the raw ΔUT1 against UTC would be wrong.
  [[nodiscard]] bool lookup(const time::Tai& t, const time::LeapSecondTable& leap,
                            EopValue& out) const {
    if (count_ < 2) {
      return false;
    }
    const double t_tai = taiSecondsOf(t);
    if (!std::isfinite(t_tai)) {
      return false;
    }
    // Records are ascending in MJD_UTC and ΔAT is non-decreasing, so the derived
    // TAI epochs are ascending too — a linear scan finds the bracket. Daily tables
    // spanning a mission upload window are small and bounded.
    for (std::size_t i = 0; i + 1 < count_; ++i) {
      const double t0 = taiSecondsOf(entries_[i], leap);
      const double t1 = taiSecondsOf(entries_[i + 1], leap);
      if (t_tai < t0) {
        return false;  // before the table's span
      }
      if (t_tai > t1) {
        continue;
      }
      const double w = (t1 > t0) ? (t_tai - t0) / (t1 - t0) : 0.0;
      out.ut1_minus_tai =
          lerp(ut1MinusTaiOf(entries_[i], leap), ut1MinusTaiOf(entries_[i + 1], leap), w);
      out.xp_arcsec = lerp(entries_[i].xp_arcsec, entries_[i + 1].xp_arcsec, w);
      out.yp_arcsec = lerp(entries_[i].yp_arcsec, entries_[i + 1].yp_arcsec, w);
      return true;
    }
    return false;  // after the table's span
  }

 private:
  static double lerp(double a, double b, double w) { return a + (b - a) * w; }

  /// ΔAT [s] in effect on the UTC date of record @p e.
  static double deltaAtOf(const EopEntry& e, const time::LeapSecondTable& leap) {
    const auto day = static_cast<std::int64_t>(std::floor(e.mjd_utc - kMjd1970));
    const time::CivilDate d = time::civilFromDays(day);
    return static_cast<double>(leap.deltaAtForUtcDate(d.year, d.month, d.day));
  }

  /// `UT1 - TAI` [s] of record @p e: the published `UT1 - UTC` less the ΔAT in
  /// effect on that date. Continuous across leap seconds by construction.
  static double ut1MinusTaiOf(const EopEntry& e, const time::LeapSecondTable& leap) {
    return e.dut1 - deltaAtOf(e, leap);
  }

  /// TAI seconds since the TAI epoch at record @p e's nominal UTC epoch.
  static double taiSecondsOf(const EopEntry& e, const time::LeapSecondTable& leap) {
    return (e.mjd_utc - kMjd1970) * constants::time::kSecondsPerDay + deltaAtOf(e, leap);
  }

  /// TAI seconds since the TAI epoch, keeping the sub-second part.
  static double taiSecondsOf(const time::Tai& t) {
    const time::TwoPart tp = t.twoPart();
    return static_cast<double>(tp.seconds) + tp.fraction;
  }

  EopEntry entries_[Capacity]{};
  std::size_t count_{0};
};

}  // namespace polaris::frames

#endif  // POLARIS_FRAMES_EOP_HPP
