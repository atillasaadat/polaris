#ifndef POLARIS_ONBOARD_TABLES_HPP
#define POLARIS_ONBOARD_TABLES_HPP

/// @file
/// @brief Onboard time/EOP/ephemeris table store (design doc §11.3, §22;
/// REQ-CDH-002).
///
/// The flight-safe holder the `flight.OnboardTables` F´ component wraps. It owns
/// the three onboard reference tables the Phase-4 GNC stack consumes —
///  - the **leap-second** (ΔAT) table (`time::LeapSecondTable`),
///  - the **IERS EOP** table (`frames::EopTable`), and
///  - the **Chebyshev ephemeris** of the Sun and Moon
///    (`ephemeris::EphemerisTable`) —
/// loads the file-backed ones from disk (the leap-second table is compiled in),
/// validates them, and answers point queries against them. It only ever *wraps* the `lib/` tables
/// and evaluators; the parsing here fills them through their public `addEntry`/`addSegment`
/// contracts.
///
/// Flight discipline (§3.6): no heap and no exceptions in steady state.
/// Fixed-capacity tables, fixed-size line buffers, `<cstdio>` for the file read
/// (see @ref TableStore::load), and return codes throughout — a malformed file
/// is a rejected load, never a throw or an assert.
///
/// **Double-buffered, lock-free (seqlock-guarded).** Two `TableSet` slots back
/// an atomic active index. A load parses into the *inactive* slot and flips the
/// index only on full success (`TableStore::load`), so a query never observes a
/// half-loaded table and a failed reload leaves the previous tables in service —
/// the "stage then swap" the upload→activate path (§22) needs. Two slots alone
/// guard one swap, not two: a reader latched onto the retired slot while two
/// back-to-back reloads run would race the second reload's writes. Each slot
/// therefore carries a seqlock generation counter — the writer makes it odd
/// before writing and even after; a reader retries (bounded) if the count was
/// odd or changed across its read, so a torn read is always detected and
/// discarded, never served. Queries stay wait-free for readers whenever no
/// reload is mid-write on their slot.
///
/// References:
///  - IERS Conventions (2010), IERS TN 36, §5 (EOP). [iers2010]
///  - Newhall, "Numerical representation of planetary ephemerides",
///    *Celestial Mechanics* 45 (1989). [newhall1989]

#include <atomic>
#include <cstddef>
#include <cstdint>

#include "ephemeris/ephemeris_table.hpp"
#include "frames/eop.hpp"
#include "math/frames.hpp"
#include "math/typed_vector.hpp"
#include "time/leap_seconds.hpp"
#include "time/timescales.hpp"

namespace polaris::onboard {

/// Bodies the onboard ephemeris carries. The fixture also holds the planets, but
/// the onboard consumers (sun-vector reference, third-body) need only these two,
/// so planet segments are skipped at load.
enum class Body { Sun, Moon };

/// Source quality of a query answer (design doc §8.1, §11.3 coarse-fallback
/// note; REQ-CDH-002). Callers gate on this so table loss degrades to coarse
/// operation instead of "no answer":
///  - `kPrecise` — served from the uploaded table (DE440 Chebyshev / IERS EOP).
///  - `kCoarse`  — served from a table-independent fallback (analytic Vallado
///    Sun/Moon, or zero-EOP with UT1 ≈ UTC and zero polar motion). This is what
///    guarantees the coarse sun-pointing Safe-mode floor is table-independent.
///  - `kUnavailable` — no answer at all (reserved; in practice the coarse path
///    always produces a finite value, so ephemeris/EOP queries never return it).
enum class Quality : std::uint8_t { kUnavailable = 0, kCoarse = 1, kPrecise = 2 };

/// Daily-EOP capacity: a mission-arc window (about a year) of daily records with
/// margin. `finals.all` holds ~20 000 days; the load keeps only the window
/// around the ephemeris span (see @ref TableStore::load), so this bounds that
/// window, not the whole product.
inline constexpr std::size_t kEopCapacity = 512;

/// Chebyshev segment capacity per body. The committed one-year fixture is 46
/// Sun and 92 Moon segments; 128 leaves margin before a fixture regenerated with
/// shorter intervals would overflow.
inline constexpr std::size_t kEphCapacity = 128;

/// Longest table-file line handled by the fixed read buffer. A `.cheb` segment
/// line is the driver: `degree ≤ 15` ⇒ 16 coefficients × 3 components, each
/// ≤ 24 characters, plus the header ≈ 1.2 kB. 4 kB is comfortable margin; a
/// longer line is treated as a malformed file (rejected, not truncated).
inline constexpr std::size_t kMaxLineLength = 4096;

/// Longest failure reason recorded for an EVR (fixed, no heap).
inline constexpr std::size_t kMaxReasonLength = 96;

/// Coverage span of one table, as TAI seconds since the TAI epoch. `valid` is
/// false until at least one record/segment is loaded.
struct TableSpan {
  std::int64_t start_tai_s{0};
  std::int64_t end_tai_s{0};
  bool valid{false};

  /// True if @p tai_s lies within [start, end] (inclusive) and the span is set.
  bool contains(std::int64_t tai_s) const {
    return valid && tai_s >= start_tai_s && tai_s <= end_tai_s;
  }
};

/// Outcome of a load/reload: per-table success, counts, spans, and a fixed-size
/// reason on the first failure. Everything the component telemeters or EVRs.
struct LoadReport {
  bool leap_ok{false};
  bool eop_ok{false};
  bool ephem_ok{false};

  std::size_t leap_entries{0};
  std::size_t eop_entries{0};
  std::size_t sun_segments{0};
  std::size_t moon_segments{0};

  TableSpan eop_span{};
  TableSpan ephem_span{};

  char reason[kMaxReasonLength]{};  ///< first failure, empty on full success

  /// True only when all three tables loaded and validated.
  bool ok() const { return leap_ok && eop_ok && ephem_ok; }
};

/// One immutable snapshot of every onboard table plus its coverage metadata.
struct TableSet {
  time::LeapSecondTable leap{};
  frames::EopTable<kEopCapacity> eop{};
  ephemeris::EphemerisTable<kEphCapacity> sun{};
  ephemeris::EphemerisTable<kEphCapacity> moon{};

  std::size_t leap_entries{0};
  TableSpan eop_span{};
  TableSpan ephem_span{};
  bool valid{false};  ///< true once a full load has populated this slot
};

/// Owns the onboard tables and answers point queries. See the file header for
/// the double-buffer/atomic-swap contract.
class TableStore {
 public:
  TableStore() = default;

  // Non-copyable: holds an atomic and two large table sets; there is one per
  // component and it is never copied.
  TableStore(const TableStore&) = delete;
  TableStore& operator=(const TableStore&) = delete;

  /// Load leap (`historical()`), the Chebyshev fixture at @p ephem_path, and the
  /// IERS `finals.all` product at @p eop_path (windowed to the ephemeris span)
  /// into the inactive slot; flip the active slot only if all three succeed.
  ///
  /// @return true iff every table loaded — @p report carries per-table status,
  ///         counts, spans, and, on failure, the reason. On failure the active
  ///         tables are unchanged.
  ///
  /// Runs at init and on operator `RELOAD_TABLES`. Uses `<cstdio>` file reads
  /// into a fixed line buffer; no dynamic allocation of our own.
  bool load(const char* eop_path, const char* ephem_path, LoadReport& report);

  /// True once a full load has succeeded (queries can answer).
  bool ready() const { return active().valid; }

  /// EOP at @p tai_ns, always answered (out is always written). `kPrecise` when
  /// the uploaded EOP table covers the epoch; otherwise the **zero-EOP** coarse
  /// fallback (UT1−TAI = −ΔAT from the in-code leap table, i.e. UT1 ≈ UTC; polar
  /// motion zero) and `kCoarse`. Error budget of the fallback: |ΔUT1| ≤ 0.9 s (by
  /// leap-second scheduling) and |polar motion| ≤ ~0.4 arcsec (≈ 12 m ground
  /// projection). Never `kUnavailable`.
  [[nodiscard]] Quality eopAt(std::int64_t tai_ns, frames::EopValue& out) const;

  /// Geocentric ECI position [m] of @p body at @p tai_ns (TAI converted to the
  /// TDB the fit uses). `kPrecise` when the uploaded Chebyshev fit covers the
  /// epoch; otherwise the analytic Vallado Sun/Moon fallback and `kCoarse`
  /// (out always written). `kUnavailable` only if the analytic value is non-finite
  /// (does not occur for finite epochs).
  [[nodiscard]] Quality bodyPositionEci(Body body, std::int64_t tai_ns,
                                        math::Vec3<math::frames::ECI>& out) const;

  /// ΔAT = TAI − UTC [s] at @p tai_ns. Always `kPrecise`: the leap table is an
  /// in-code IERS record held independently of the uploaded tables, so it answers
  /// even before any load and never degrades.
  [[nodiscard]] Quality taiUtcOffset(std::int64_t tai_ns, std::int32_t& out) const;

  /// Whether @p tai_ns is inside the EOP and ephemeris coverage. Coarse (span
  /// endpoints only); an actual query is authoritative. Used by the scheduler
  /// coverage-expiry check. Returns false if no tables are loaded.
  [[nodiscard]] bool coverageAt(std::int64_t tai_ns, bool& eop_ok, bool& ephem_ok) const;

  // --- Telemetry accessors (seqlock-consistent; zero/invalid if unreadable) ---
  std::size_t leapEntries() const {
    return snapshotSize([](const TableSet& s) { return s.leap_entries; });
  }

  std::size_t eopEntries() const {
    return snapshotSize([](const TableSet& s) { return s.eop.size(); });
  }

  std::size_t sunSegments() const {
    return snapshotSize([](const TableSet& s) { return s.sun.size(); });
  }

  std::size_t moonSegments() const {
    return snapshotSize([](const TableSet& s) { return s.moon.size(); });
  }

  TableSpan eopSpan() const {
    TableSpan span{};
    (void)readConsistent([&span](const TableSet& s) {
      span = s.eop_span;
      return true;
    });
    return span;
  }

  TableSpan ephemSpan() const {
    TableSpan span{};
    (void)readConsistent([&span](const TableSet& s) {
      span = s.ephem_span;
      return true;
    });
    return span;
  }

 private:
  const TableSet& active() const { return slots_[active_.load(std::memory_order_acquire)]; }

  /// Consistent scalar snapshot for the telemetry accessors.
  template <typename Fn>
  std::size_t snapshotSize(Fn&& fn) const {
    std::size_t v = 0;
    (void)readConsistent([&v, &fn](const TableSet& s) {
      v = fn(s);
      return true;
    });
    return v;
  }

  /// Seqlock read: run @p fn against a consistent slot snapshot. @p fn is
  /// invoked with the latched slot and must confine all side effects to locals
  /// it owns (it may run against a slot that a concurrent reload is rewriting;
  /// such an attempt is detected and its result discarded). Bounded retries
  /// (JPL rule 2); returns false if no consistent read was obtained.
  template <typename Fn>
  bool readConsistent(Fn&& fn) const {
    for (int attempt = 0; attempt < kMaxReadAttempts; ++attempt) {
      const int idx = active_.load(std::memory_order_acquire);
      const std::uint32_t g1 = gen_[idx].load(std::memory_order_acquire);
      if ((g1 & 1u) != 0u) {
        continue;  // slot mid-write; the flip to the fresh slot is imminent
      }
      const bool ok = fn(slots_[idx]);
      std::atomic_thread_fence(std::memory_order_acquire);
      if (gen_[idx].load(std::memory_order_relaxed) == g1) {
        return ok;
      }
    }
    return false;  // retries exhausted: report unavailable rather than torn
  }

  /// Fill @p set from files. Static so it cannot touch the atomic mid-parse.
  static bool loadInto(TableSet& set, const char* eop_path, const char* ephem_path,
                       LoadReport& report);

  static constexpr int kMaxReadAttempts = 3;

  /// In-code IERS leap-second record, held independently of the uploaded tables
  /// so ΔAT (and the zero-EOP fallback's UT1−TAI = −ΔAT) answer even before any
  /// load — the table-independent piece of the coarse Safe-mode floor.
  time::LeapSecondTable leap_{time::LeapSecondTable::historical()};

  TableSet slots_[2]{};
  std::atomic<int> active_{0};
  /// Per-slot seqlock generation: odd while `load` writes the slot, even when
  /// stable. Readers pair with it in `readConsistent`.
  std::atomic<std::uint32_t> gen_[2]{};

  static_assert(std::atomic<int>::is_always_lock_free &&
                    std::atomic<std::uint32_t>::is_always_lock_free,
                "table-swap atomics must be lock-free on the flight target");
};

}  // namespace polaris::onboard

#endif  // POLARIS_ONBOARD_TABLES_HPP
