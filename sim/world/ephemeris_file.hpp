#ifndef POLARIS_SIM_WORLD_EPHEMERIS_FILE_HPP
#define POLARIS_SIM_WORLD_EPHEMERIS_FILE_HPP

/// @file
/// @brief Loader for the committed Chebyshev ephemeris fixture (REQ-CDH-002).
///
/// Reads the `.cheb` fit produced by `tools/ephem` into `EphemerisTable`s and
/// adapts them to the shared `BodyPositionFn` contract (`body_position.hpp`) —
/// which is what finally drives `ThirdBodyGravity` and `SolarRadiationPressure`
/// from real DE440 positions instead of hand-fed test values.
///
/// **The DE440 kernel itself is not committed.** It is a ~32 MB binary fetch
/// input, like the full EGM2008 `.gfc`; what ships is the derived Chebyshev fit,
/// which §3.7 sanctions as an "uploadable table" computed *from* the original.
/// The fixture's header records the source URL and the kernel's SHA-256, so the
/// derivation is reproducible by re-download.
///
/// Ground/sim-side: this does file I/O and uses heap and exceptions-free error
/// returns. The `EphemerisTable` it fills is the same fixed-capacity container
/// flight uses, so the sim exercises the flight data structure rather than a
/// parallel one — the onboard difference is that flight receives segments by
/// upload (`addSegment`) instead of reading a file (design doc §11.3).
///
/// Format (see `tools/ephem/writer.py`): `#` comment lines, then one line per
/// segment,
///
///     seg <body> <mid_ns> <radius_seconds> <degree> <cx...> <cy...> <cz...>
///
/// with `degree + 1` coefficients per component, geocentric ECI metres, and
/// `mid_ns` in TDB nanoseconds since 1970-01-01T00:00:00.

#include <array>
#include <cstddef>
#include <string>

#include "ephemeris/ephemeris_table.hpp"
#include "world/body_position.hpp"

namespace polaris::sim::world {

/// Segment capacity per body. One year of Moon fits at the default 4-day
/// interval is 92 segments, so this leaves room for roughly triple that before a
/// fixture has to be regenerated with longer intervals. Sized for the sim, not
/// for flight: an `EphemerisSet` is ~200 kB, so hold it as a member or on the
/// heap rather than as a large stack temporary.
inline constexpr std::size_t kEphemerisCapacity = 256;

using SimEphemerisTable = ephemeris::EphemerisTable<kEphemerisCapacity>;

/// Planet names a fixture may carry, in a fixed order (Mercury→Neptune). These
/// are the *barycenter/system* positions and pair with the system GMs in
/// `constants::bodies` — the right point mass seen from Earth orbit, where a
/// planet and its moons are unresolved.
inline constexpr std::array<const char*, 7> kPlanetNames = {
    "mercury", "venus", "mars", "jupiter", "saturn", "uranus", "neptune"};

/// The bodies one fixture provides. Sun and Moon are always fitted; planets are
/// present in fixtures regenerated since the planetary-third-body push (their
/// tables are simply empty when loading an older Sun/Moon-only fixture).
struct EphemerisSet {
  SimEphemerisTable sun;
  SimEphemerisTable moon;
  /// Planetary barycenters, indexed parallel to @ref kPlanetNames.
  std::array<SimEphemerisTable, kPlanetNames.size()> planets;

  /// The table for @p name ("sun", "moon", or a @ref kPlanetNames entry), or
  /// nullptr for an unknown name.
  const SimEphemerisTable* find(const std::string& name) const;
  SimEphemerisTable* find(const std::string& name);
};

/// Load a `.cheb` fixture into @p out.
///
/// @param path  Fixture path (e.g. `tests/golden/de440_bodies.cheb`).
/// @param out   Filled on success; partially filled on failure, so do not use it
///              unless this returned true.
/// @param error If non-null, receives a human-readable reason on failure.
/// @return false if the file cannot be read, a line is malformed, a body is
///         unknown, or a table overflows its capacity. Malformed input is a
///         hard failure rather than a silent skip: a half-loaded ephemeris would
///         show up as an intermittent coverage gap mid-propagation, which is far
///         harder to diagnose than a refusal at load time.
bool loadEphemerisFile(const std::string& path, EphemerisSet& out, std::string* error = nullptr);

/// Adapt a table to the shared resolver contract. The returned callable holds a
/// pointer to @p table, which must outlive it.
BodyPositionFn bodyPositionFn(const SimEphemerisTable& table);

}  // namespace polaris::sim::world

#endif  // POLARIS_SIM_WORLD_EPHEMERIS_FILE_HPP
