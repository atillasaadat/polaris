#ifndef POLARIS_GNC_TLE_HPP
#define POLARIS_GNC_TLE_HPP

/// @file
/// @brief NORAD two-line element set: parsing and validation (design doc §8.3,
/// REQ-ODP-003).
///
/// A TLE is the input half of SGP4, and it is worth being precise about what it
/// is: **not a state vector in a coordinate frame**, but a set of mean elements
/// that only mean anything when fed to the propagator they were fitted with.
/// The values here are therefore kept in the units the format carries them in
/// (revolutions per day, degrees, earth radii) and converted inside
/// @ref polaris::gnc::Sgp4 rather than at parse time — converting early would
/// invite treating them as osculating elements, which they are not.
///
/// ## Format
///
/// The columns are fixed-width and defined by NORAD/Space-Track; this parser
/// follows Vallado's `twoline2rv` [vallado2006revisiting] and the layout in
/// [vallado2013] §3.6. Three encodings are easy to get wrong and are handled
/// explicitly below:
///
/// - **Assumed decimal point.** Eccentricity is written without its leading
///   `0.` — `1859667` is 0.1859667.
/// - **Assumed exponent.** The drag term and the second derivative use a
///   trailing signed exponent with an implied decimal point: `28098-4` is
///   \f$0.28098 \times 10^{-4}\f$. A `+` or a blank sign both mean positive.
/// - **Two-digit year.** Windowed as Space-Track defines it: `57`–`99` are
///   1957–1999, `00`–`56` are 2000–2056. This is a real cliff, not a
///   convention we may reinterpret — a TLE issued in 2057 will break it, and
///   that is the format's problem rather than one this parser may paper over.
///
/// ## Checksum
///
/// Each line ends in a modulo-10 checksum over the preceding 68 characters,
/// digits at face value and every `-` counted as 1. It is **verified, not
/// ignored**: a TLE reaches this code over a command uplink or a file, and a
/// single flipped character in a mean motion produces a perfectly plausible
/// orbit somewhere else entirely. The checksum is weak — it catches no
/// transposition — so it is a necessary and very much not sufficient gate, and
/// the range checks below carry the rest.
///
/// ## Flight standard
///
/// No heap, no exceptions, no `std::string`. Input is a pair of
/// `std::string_view` over caller-owned storage; the result is a status enum
/// (§3.6). Every failure is named, because "the TLE was bad" is not an
/// actionable event on a spacecraft.

#include <cstdint>
#include <string_view>

#include "time/leap_seconds.hpp"
#include "time/timescales.hpp"

namespace polaris::gnc {

/// Why a TLE was refused. `kOk` is the only accepting value.
enum class TleStatus : std::uint8_t {
  kOk = 0,
  kLineTooShort,       ///< a line is under the 69 columns the format defines
  kBadLineNumber,      ///< line 1 does not start with '1 ', or line 2 with '2 '
  kSatelliteMismatch,  ///< the two lines carry different catalog numbers
  kBadChecksum,        ///< modulo-10 checksum failed on one of the lines
  kMalformedField,     ///< a numeric field did not parse
  kOutOfRange,         ///< a field parsed but is not physically usable
};

/// Human-readable name of @p status, for events and telemetry. Never null.
const char* toString(TleStatus status);

/// One parsed element set, in the format's own units.
///
/// Field names follow Vallado's so the SGP4 source reads against the paper
/// rather than against a private renaming.
struct TleElements {
  std::int32_t satellite_number{0};  ///< NORAD catalog number
  char classification{'U'};          ///< 'U' unclassified, 'C', 'S'
  std::int32_t epoch_year{0};        ///< full 4-digit year, already windowed
  double epoch_day{0.0};             ///< day of year + fraction, 1.0 = Jan 1 00:00

  /// First derivative of mean motion / 2 [rev/day^2]. Unused by SGP4 itself
  /// (it is a legacy SGP term) but parsed, because a consumer comparing against
  /// Space-Track's own text needs it and a silently dropped field is a field
  /// that disagrees.
  double ndot{0.0};
  double nddot{0.0};  ///< second derivative / 6 [rev/day^3]; also SGP-legacy
  double bstar{0.0};  ///< drag term [1 / earth radii]

  std::int32_t element_number{0};

  double inclination_deg{0.0};
  double raan_deg{0.0};         ///< right ascension of the ascending node
  double eccentricity{0.0};     ///< [0, 1)
  double arg_perigee_deg{0.0};  ///< argument of perigee
  double mean_anomaly_deg{0.0};
  double mean_motion_rev_per_day{0.0};
  std::int32_t revolution_number{0};

  /// Orbital period implied by the mean motion [min], or 0 when mean motion is
  /// not positive. The deep-space cut is stated on this rather than on the mean
  /// motion so the threshold reads as the 225-minute rule it is.
  double periodMinutes() const;

  /// True when SGP4 must run its deep-space (SDP4) branch: period >= 225 min.
  bool isDeepSpace() const;

  /// Epoch as TAI, using @p leap for the UTC->TAI offset (TLE epochs are UTC).
  ///
  /// Returns false and leaves @p out untouched if the epoch fields do not form
  /// a real date — a day-of-year of 367, say. Separate from parsing because the
  /// conversion needs a leap-second table and parsing must not.
  bool epochTai(const time::LeapSecondTable& leap, time::Tai& out) const;
};

/// Whether to enforce the line checksums.
///
/// Not a convenience knob — the two callers genuinely differ. A TLE arriving
/// over an uplink or out of a file gets @ref kVerify, because a single flipped
/// character produces a perfectly plausible orbit somewhere else entirely and
/// the checksum is the only integrity signal the format carries. A TLE that was
/// *constructed by hand* — the official verification fixture's `3333x` edge
/// cases, 5 of whose 66 lines carry stale checksums, and any test or analysis
/// element set written by a person — gets @ref kIgnore, because its checksum
/// says nothing about a transmission that never happened.
///
/// The reference implementation does not check at all. Defaulting to @ref
/// kVerify is therefore a deliberate departure, and it is the right one for the
/// path that matters: refusing a corrupted uplink is worth making the fixture
/// say so explicitly.
enum class TleChecksumPolicy : std::uint8_t {
  kVerify = 0,  ///< refuse a line whose checksum does not match
  kIgnore,      ///< parse regardless — hand-written element sets
};

/// Parse a two-line element set from @p line1 and @p line2 into @p out.
///
/// Lines may be longer than 69 columns and the excess is ignored: the committed
/// verification fixture (`tests/golden/SGP4-VER.TLE`) appends start/stop/step
/// times to line 2, which is a driver convention and not part of the format.
/// Trailing `\r` and whitespace are tolerated for the same reason.
///
/// @p out is written only on success (`kOk`), so a refused TLE cannot leave a
/// half-populated element set behind for a caller that ignored the status.
TleStatus parseTle(std::string_view line1, std::string_view line2, TleElements& out,
                   TleChecksumPolicy checksum = TleChecksumPolicy::kVerify);

/// The modulo-10 checksum of @p line's first 68 columns: digits at face value,
/// `-` as 1, everything else as 0. Exposed for tests and for tools that need to
/// *write* a TLE.
int tleChecksum(std::string_view line);

}  // namespace polaris::gnc

#endif  // POLARIS_GNC_TLE_HPP
