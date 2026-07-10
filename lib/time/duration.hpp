#ifndef POLARIS_TIME_DURATION_HPP
#define POLARIS_TIME_DURATION_HPP

/// @file
/// @brief Signed time interval on a uniform timescale (design doc §3.2).
///
/// `Duration` is the difference between two `Instant`s: a signed count of
/// integer nanoseconds. It is the arithmetic currency of the time library —
/// `Instant`s are points, `Duration`s are the gaps between them. Value type,
/// `constexpr`, no heap, no exceptions (flight-safe, §3.6).
///
/// Range: int64 nanoseconds spans ~292 years, far beyond any mission arc.
///
/// References:
///  - Vallado, *Fundamentals of Astrodynamics and Applications*, 4th ed., §3
///    (time systems). [vallado2013]

#include <cmath>
#include <cstdint>

namespace polaris::time {

/// Signed interval as an integer nanosecond count.
class Duration {
 public:
  /// Zero interval.
  constexpr Duration() = default;

  /// Named constructor from an exact nanosecond count.
  static constexpr Duration fromNanoseconds(std::int64_t ns) { return Duration(ns); }

  /// Named constructor from whole seconds (exact for |s| < ~9.2e9).
  static constexpr Duration fromSeconds(std::int64_t s) { return Duration(s * kNsPerSecond); }

  /// Largest fractional-seconds magnitude representable without int64-ns
  /// overflow: 9e9 s -> 9e18 ns, comfortably below `INT64_MAX` (~9.22e18).
  static constexpr double kMaxSeconds = 9.0e9;

  /// Named constructor from fractional seconds (rounded to the nearest ns).
  ///
  /// Defensive per §3.6: a non-finite input (NaN/Inf) yields a **zero** interval,
  /// and a magnitude beyond `kMaxSeconds` **saturates** to `±kMaxSeconds` — so
  /// this never invokes the undefined behavior `std::llround` has on non-finite
  /// or out-of-range arguments. Callers needing to *detect* such inputs (rather
  /// than absorb them) must range/finiteness-check upstream, since a named
  /// constructor has no status channel. Not `constexpr`: `std::llround`
  /// (round-half-away-from-zero) is not usable in a constant expression.
  static Duration fromSecondsF(double s) {
    if (!std::isfinite(s)) {
      return Duration(0);
    }
    const double clamped = (s > kMaxSeconds) ? kMaxSeconds : (s < -kMaxSeconds ? -kMaxSeconds : s);
    return Duration(std::llround(clamped * kNsPerSecondF));
  }

  /// Exact nanosecond count.
  constexpr std::int64_t nanoseconds() const { return ns_; }

  /// Interval in seconds (may lose precision for large magnitudes).
  constexpr double seconds() const { return static_cast<double>(ns_) / kNsPerSecondF; }

  /// @name Arithmetic (interval algebra)
  /// @{
  constexpr Duration operator+(const Duration& o) const { return Duration(ns_ + o.ns_); }

  constexpr Duration operator-(const Duration& o) const { return Duration(ns_ - o.ns_); }

  constexpr Duration operator-() const { return Duration(-ns_); }

  constexpr Duration operator*(std::int64_t k) const { return Duration(ns_ * k); }

  /// @}

  /// @name Comparisons
  /// @{
  constexpr bool operator==(const Duration& o) const { return ns_ == o.ns_; }

  constexpr bool operator!=(const Duration& o) const { return ns_ != o.ns_; }

  constexpr bool operator<(const Duration& o) const { return ns_ < o.ns_; }

  constexpr bool operator<=(const Duration& o) const { return ns_ <= o.ns_; }

  constexpr bool operator>(const Duration& o) const { return ns_ > o.ns_; }

  constexpr bool operator>=(const Duration& o) const { return ns_ >= o.ns_; }

  /// @}

  /// Nanoseconds in one SI second.
  static constexpr std::int64_t kNsPerSecond = 1'000'000'000;

 private:
  explicit constexpr Duration(std::int64_t ns) : ns_(ns) {}

  static constexpr double kNsPerSecondF = 1'000'000'000.0;

  std::int64_t ns_{0};
};

}  // namespace polaris::time

#endif  // POLARIS_TIME_DURATION_HPP
