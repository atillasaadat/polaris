#ifndef POLARIS_SIM_WORLD_IGRF_FILE_HPP
#define POLARIS_SIM_WORLD_IGRF_FILE_HPP

/// @file
/// @brief Loader for the verbatim IAGA IGRF-14 coefficient file.
///
/// Reads `tests/golden/igrf14coeffs.txt` — the IAGA product **byte-for-byte as
/// published** (design doc §3.7) — and collapses it into the single
/// `environment::IgrfCoefficients` snapshot the flight-side evaluator consumes.
///
/// IAGA publishes coefficients on a 5-year grid (1900.0 … 2025.0) plus one
/// secular-variation column covering the final interval (2025–2030). The model's
/// own definition is: linear interpolation between bracketing epochs inside the
/// grid, and linear extrapolation by the SV column past the last one. Both cases
/// are a base epoch plus a rate, so this loader picks whichever applies and hands
/// the evaluator one `(epoch, ġ)` pair. Inside the grid the derived rate is
/// `(g(t₁) − g(t₀)) / (t₁ − t₀)`, which reproduces the interpolation exactly
/// rather than approximating it — so the evaluator never needs to know that a
/// grid existed. That keeps the whole 26-epoch table off the flight side, which
/// matters: onboard, an `IgrfCoefficients` arrives as an upload, and uploading
/// one snapshot instead of the full history is ~26× less to push through the
/// link (design doc §11.3, §19.3).
///
/// Ground/sim side: this does file I/O and returns errors by value. Nothing here
/// is on a flight path.
///
/// Format (IAGA `igrf14coeffs.txt`): `#` comment lines, a `c/s deg ord …` type
/// row, a `g/h n m <epoch> … <sv>` header row naming each epoch, then one row per
/// coefficient,
///
///     g|h  <n>  <m>  <value at each epoch…>  <secular variation>
///
/// with main-field values in nT and secular variation in nT/yr.

#include <string>

#include "environment/igrf.hpp"

namespace polaris::sim::world {

/// Load the IAGA coefficient file and produce the snapshot valid at
/// @p decimal_year.
///
/// @param path         Path to the verbatim IAGA file.
/// @param decimal_year Epoch of interest, e.g. 2026.5.
/// @param out          Filled on success; left partially written on failure, so
///                     do not use it unless this returned true.
/// @param error        If non-null, receives a human-readable reason on failure.
/// @return false if the file cannot be read, the header is unrecognisable, a row
///         is malformed, a degree is out of range, or @p decimal_year falls
///         before the first tabulated epoch. Malformed input is a hard failure
///         rather than a partial load: a field model silently missing its high
///         harmonics looks like a plausible field, not like a bug, and would
///         quietly corrupt every magnetometer comparison downstream.
///
/// Extrapolating past the last epoch by the SV column is *allowed* — that is the
/// model's defined behaviour through 2030 — and is not reported as an error.
/// Extrapolating backwards before 1900.0 is refused, since IGRF says nothing
/// there.
bool loadIgrfFile(const std::string& path, double decimal_year, environment::IgrfCoefficients& out,
                  std::string* error = nullptr);

}  // namespace polaris::sim::world

#endif  // POLARIS_SIM_WORLD_IGRF_FILE_HPP
