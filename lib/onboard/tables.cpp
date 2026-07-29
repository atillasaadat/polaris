/// @file
/// @brief Onboard table store implementation. See `tables.hpp` for the contract.
///
/// Flight-safe file reading: `<cstdio>` line reads into a fixed buffer, C string
/// scanning, no `std::string`/`std::vector`, no exceptions. The one non-owned
/// allocation is whatever the libc stream buffers internally on `fopen`; a
/// hardware target routes this through `Os::File`/`Drv` instead, but the parse
/// logic is unchanged. ponytail: cstdio at load/reload only, never steady state.

#include "onboard/tables.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>

#include "ephemeris/chebyshev.hpp"
#include "time/civil.hpp"
#include "time/tdb.hpp"

namespace polaris::onboard {
namespace {

constexpr double kMjd1970 = 40587.0;
constexpr double kSecondsPerDay = 86400.0;

/// TT − TAI [ns] (32.184 s). TDB − TT is ≤ ~2 ms and ignored when reducing a TDB
/// ephemeris epoch to a coarse TAI coverage bound — negligible against the
/// day-scale coverage margins.
constexpr std::int64_t kTtMinusTaiNs = 32184000000LL;

/// EOP load window half-width beyond the ephemeris span, so endpoint
/// interpolation still has neighbours. Mirrors the sim loader's margin.
constexpr double kEopMarginDays = 5.0;

/// Bulletin A fixed-width column spec, matching `sim/world/eop_file.cpp` and
/// `tools/eop/finals.py`. Kept in sync with those; a fixed-width product only
/// ever gets re-columned upstream, not per reader.
constexpr std::size_t kMjdOffset = 7;
constexpr std::size_t kMjdWidth = 8;
constexpr std::size_t kXpOffset = 18;
constexpr std::size_t kYpOffset = 37;
constexpr std::size_t kFieldWidth = 9;
constexpr std::size_t kDut1Offset = 58;
constexpr std::size_t kDut1Width = 10;
constexpr std::size_t kMinFinalsLine = kDut1Offset + kDut1Width;

void setReason(LoadReport& report, const char* msg) {
  std::snprintf(report.reason, sizeof(report.reason), "%s", msg);
}

/// Parse a fixed-width numeric field from @p line into @p out. False if the
/// field is off the end of the line, blank, or not a number.
bool finalsField(const char* line, std::size_t len, std::size_t offset, std::size_t width,
                 double& out) {
  if (len < offset + width) {
    return false;
  }
  char buf[32];
  if (width >= sizeof(buf)) {
    return false;
  }
  std::memcpy(buf, line + offset, width);
  buf[width] = '\0';
  char* end = nullptr;
  const double v = std::strtod(buf, &end);
  if (end == buf) {
    return false;  // blank or non-numeric
  }
  out = v;
  return true;
}

/// MJD (UTC) of a segment-derived TDB nanosecond count, coarse (see kTtMinusTaiNs).
double mjdOfTdbNs(std::int64_t tdb_ns) {
  const double tai_s = static_cast<double>(tdb_ns - kTtMinusTaiNs) / 1.0e9;
  return kMjd1970 + tai_s / kSecondsPerDay;
}

/// TAI seconds at 00:00 UTC on the day of @p mjd_utc, including that day's ΔAT.
std::int64_t taiSecondsOfMjd(double mjd_utc, const time::LeapSecondTable& leap) {
  const auto day = static_cast<std::int64_t>(std::floor(mjd_utc - kMjd1970));
  const time::CivilDate d = time::civilFromDays(day);
  const std::int32_t delta = leap.deltaAtForUtcDate(d.year, d.month, d.day);
  return static_cast<std::int64_t>(std::llround((mjd_utc - kMjd1970) * kSecondsPerDay)) + delta;
}

}  // namespace

bool TableStore::load(const char* eop_path, const char* ephem_path, LoadReport& report) {
  const int inactive = 1 - active_.load(std::memory_order_relaxed);
  // Seqlock writer: odd = slot being rewritten. A reader still latched onto
  // this slot from before the previous flip detects the change and retries on
  // the fresh slot instead of serving a torn read.
  gen_[inactive].fetch_add(1, std::memory_order_acq_rel);
  const bool ok = loadInto(slots_[inactive], eop_path, ephem_path, report);
  gen_[inactive].fetch_add(1, std::memory_order_release);  // even = stable again
  if (ok) {
    // Publish the freshly loaded slot. Release pairs with the acquire in
    // readConsistent(): a reader that sees the new index sees the full slot.
    active_.store(inactive, std::memory_order_release);
    return true;
  }
  return false;  // active tables untouched
}

bool TableStore::loadInto(TableSet& set, const char* eop_path, const char* ephem_path,
                          LoadReport& report) {
  set = TableSet{};
  report = LoadReport{};

  // --- Leap seconds: the committed in-code IERS record (no uploaded file yet) -
  set.leap = time::LeapSecondTable::historical();
  set.leap_entries = set.leap.size();
  report.leap_entries = set.leap_entries;
  report.leap_ok = set.leap_entries > 0;

  // --- Ephemeris: the committed Chebyshev fixture (Sun/Moon only) -------------
  if (ephem_path == nullptr) {
    setReason(report, "no ephemeris path");
    return false;
  }
  std::FILE* ef = std::fopen(ephem_path, "r");
  if (ef == nullptr) {
    setReason(report, "cannot open ephemeris file");
    return false;
  }
  char line[kMaxLineLength];
  std::int64_t eph_start_ns = 0;
  std::int64_t eph_end_ns = 0;
  bool eph_span_set = false;
  bool ephem_bad = false;
  while (std::fgets(line, sizeof(line), ef) != nullptr) {
    if (line[0] == '#' || line[0] == '\n' || line[0] == '\0') {
      continue;
    }
    char tag[8];
    char body[16];
    long long mid_ns = 0;
    double radius = 0.0;
    int degree = -1;
    int consumed = 0;
    if (std::sscanf(line, "%7s %15s %lld %lf %d%n", tag, body, &mid_ns, &radius, &degree,
                    &consumed) != 5) {
      setReason(report, "malformed ephemeris segment header");
      ephem_bad = true;
      break;
    }
    if (std::strcmp(tag, "seg") != 0) {
      setReason(report, "unexpected ephemeris record tag");
      ephem_bad = true;
      break;
    }
    const bool is_sun = std::strcmp(body, "sun") == 0;
    const bool is_moon = std::strcmp(body, "moon") == 0;
    if (!is_sun && !is_moon) {
      continue;  // planets and any other body: not carried onboard
    }
    if (degree < 0 || degree > ephemeris::kMaxChebyshevDegree) {
      setReason(report, "ephemeris degree out of range");
      ephem_bad = true;
      break;
    }
    ephemeris::ChebyshevSegment seg;
    seg.mid_ns = mid_ns;
    seg.radius_seconds = radius;
    seg.degree = degree;
    double* comps[3] = {seg.cx, seg.cy, seg.cz};
    const char* cursor = line + consumed;
    bool coeff_ok = true;
    for (int c = 0; c < 3 && coeff_ok; ++c) {
      for (int i = 0; i <= degree; ++i) {
        char* end = nullptr;
        comps[c][i] = std::strtod(cursor, &end);
        // Reject non-finite coefficients at the trust boundary (same rule the
        // EOP path applies): a nan/inf upload must fail the load with a clear
        // reason, not "succeed" into a table whose every query returns false.
        if (end == cursor || !std::isfinite(comps[c][i])) {
          coeff_ok = false;
          break;
        }
        cursor = end;
      }
    }
    if (!coeff_ok) {
      setReason(report, "truncated or non-finite ephemeris coefficients");
      ephem_bad = true;
      break;
    }
    ephemeris::EphemerisTable<kEphCapacity>& table = is_sun ? set.sun : set.moon;
    if (!table.addSegment(seg)) {
      setReason(report, "ephemeris capacity exceeded or segment rejected");
      ephem_bad = true;
      break;
    }
    const auto radius_ns = static_cast<std::int64_t>(std::llround(radius * 1.0e9));
    const std::int64_t s0 = mid_ns - radius_ns;
    const std::int64_t s1 = mid_ns + radius_ns;
    if (!eph_span_set || s0 < eph_start_ns) {
      eph_start_ns = s0;
    }
    if (!eph_span_set || s1 > eph_end_ns) {
      eph_end_ns = s1;
    }
    eph_span_set = true;
  }
  std::fclose(ef);
  if (ephem_bad) {
    return false;
  }
  if (set.sun.empty() || set.moon.empty()) {
    setReason(report, "ephemeris fixture missing Sun or Moon segments");
    return false;
  }
  set.ephem_span.start_tai_s = (eph_start_ns - kTtMinusTaiNs) / 1'000'000'000LL;
  set.ephem_span.end_tai_s = (eph_end_ns - kTtMinusTaiNs) / 1'000'000'000LL;
  set.ephem_span.valid = true;
  report.sun_segments = set.sun.size();
  report.moon_segments = set.moon.size();
  report.ephem_span = set.ephem_span;
  report.ephem_ok = true;

  // --- EOP: IERS finals.all, windowed to the ephemeris span -------------------
  const double win_lo = mjdOfTdbNs(eph_start_ns) - kEopMarginDays;
  const double win_hi = mjdOfTdbNs(eph_end_ns) + kEopMarginDays;
  if (eop_path == nullptr) {
    setReason(report, "no EOP path");
    return false;
  }
  std::FILE* pf = std::fopen(eop_path, "r");
  if (pf == nullptr) {
    setReason(report, "cannot open EOP file");
    return false;
  }
  double first_mjd = 0.0;
  double last_mjd = 0.0;
  bool eop_overflow = false;
  bool eop_at_capacity = false;
  while (std::fgets(line, sizeof(line), pf) != nullptr) {
    const std::size_t len = std::strlen(line);
    if (len < kMinFinalsLine) {
      continue;
    }
    double mjd = 0.0;
    double dut1 = 0.0;
    double xp = 0.0;
    double yp = 0.0;
    // dUT1 runs out first; a row missing it is a date-only placeholder, skipped
    // rather than zero-filled (which would step Earth orientation to zero).
    if (!finalsField(line, len, kMjdOffset, kMjdWidth, mjd) ||
        !finalsField(line, len, kDut1Offset, kDut1Width, dut1) ||
        !finalsField(line, len, kXpOffset, kFieldWidth, xp) ||
        !finalsField(line, len, kYpOffset, kFieldWidth, yp)) {
      continue;
    }
    if (mjd < win_lo || mjd > win_hi) {
      continue;
    }
    if (!set.eop.addEntry({mjd, dut1, xp, yp})) {
      eop_overflow = true;
      // addEntry rejects on capacity or a non-ascending MJD; tell the operator
      // which, so a corrupted/reordered upload isn't misdiagnosed as too large.
      eop_at_capacity = set.eop.size() >= kEopCapacity;
      break;
    }
    if (set.eop.size() == 1) {
      first_mjd = mjd;
    }
    last_mjd = mjd;
  }
  std::fclose(pf);
  if (eop_overflow) {
    setReason(report, eop_at_capacity ? "EOP window exceeds capacity"
                                      : "EOP row rejected: non-ascending MJD");
    return false;
  }
  if (set.eop.size() < 2) {
    setReason(report, "EOP file yielded fewer than two records in the ephemeris window");
    return false;
  }
  set.eop_span.start_tai_s = taiSecondsOfMjd(first_mjd, set.leap);
  set.eop_span.end_tai_s = taiSecondsOfMjd(last_mjd, set.leap);
  set.eop_span.valid = true;
  report.eop_entries = set.eop.size();
  report.eop_span = set.eop_span;
  report.eop_ok = true;

  set.valid = true;
  return true;
}

// Every query computes into locals inside the seqlock read and assigns the
// caller's out-params only after the snapshot proved consistent — so "returns
// false with out untouched" holds even when an attempt raced a reload.

bool TableStore::eopAt(std::int64_t tai_ns, frames::EopValue& out) const {
  frames::EopValue v{};
  const bool ok = readConsistent([&](const TableSet& s) {
    return s.valid && s.eop.lookup(time::Tai::fromNanosecondsSinceEpoch(tai_ns), s.leap, v);
  });
  if (ok) {
    out = v;
  }
  return ok;
}

bool TableStore::bodyPositionEci(Body body, std::int64_t tai_ns,
                                 math::Vec3<math::frames::ECI>& out) const {
  math::Vec3<math::frames::ECI> v;
  const bool ok = readConsistent([&](const TableSet& s) {
    if (!s.valid) {
      return false;
    }
    const time::Tdb tdb = time::toTdb(time::toTt(time::Tai::fromNanosecondsSinceEpoch(tai_ns)));
    const ephemeris::EphemerisTable<kEphCapacity>& table = (body == Body::Sun) ? s.sun : s.moon;
    return table.position(tdb, v);
  });
  if (ok) {
    out = v;
  }
  return ok;
}

bool TableStore::taiUtcOffset(std::int64_t tai_ns, std::int32_t& out) const {
  std::int32_t v = 0;
  const bool ok = readConsistent([&](const TableSet& s) {
    if (!s.valid) {
      return false;
    }
    v = s.leap.deltaAtForTaiSeconds(tai_ns / 1'000'000'000LL);
    return true;
  });
  if (ok) {
    out = v;
  }
  return ok;
}

bool TableStore::coverageAt(std::int64_t tai_ns, bool& eop_ok, bool& ephem_ok) const {
  bool eop_v = false;
  bool ephem_v = false;
  const bool ok = readConsistent([&](const TableSet& s) {
    if (!s.valid) {
      return false;
    }
    const std::int64_t tai_s = tai_ns / 1'000'000'000LL;
    eop_v = s.eop_span.contains(tai_s);
    ephem_v = s.ephem_span.contains(tai_s);
    return true;
  });
  if (ok) {
    eop_ok = eop_v;
    ephem_ok = ephem_v;
  }
  return ok;
}

}  // namespace polaris::onboard
