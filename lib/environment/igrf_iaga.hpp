#ifndef POLARIS_ENVIRONMENT_IGRF_IAGA_HPP
#define POLARIS_ENVIRONMENT_IGRF_IAGA_HPP

/// @file
/// @brief Flight-safe reader for the verbatim IAGA IGRF coefficient file
/// (design doc §3.7, §6.2, §11.3).
///
/// Reads the IAGA product **byte-for-byte as published** (`igrf14coeffs.txt`)
/// and collapses it into the single @ref IgrfCoefficients snapshot the evaluator
/// consumes. Both sides of the repo use this one parser: the truth sim through
/// `sim/world/igrf_file.hpp` (which is now a thin `std::string` façade over it),
/// and the FSW directly — the `AttitudeEstimator` component needs the same
/// onboard IGRF-14 that the magnetometer consistency check and coarse (sun +
/// mag) attitude reference are defined against (§8.1).
///
/// IAGA publishes coefficients on a 5-year grid (1900.0 … 2025.0) plus one
/// secular-variation column covering the final interval. The model's own
/// definition is linear interpolation between bracketing epochs inside the grid
/// and linear extrapolation by the SV column past the last one; both are a base
/// epoch plus a rate, so this loader picks whichever applies and hands the
/// evaluator one `(epoch, ġ)` pair. Inside the grid the derived rate is
/// `(g(t₁) − g(t₀)) / (t₁ − t₀)`, which reproduces the interpolation exactly,
/// so the evaluator never needs to know a grid existed — and only one snapshot,
/// not the 26-epoch history, has to reach the vehicle.
///
/// **Flight discipline (§3.6).** Like `lib/onboard/tables.cpp`, the file read is
/// `<cstdio>` into fixed buffers with C string scanning: no `std::string`, no
/// `std::vector`, no exceptions, bounded loops, return codes. It runs at
/// initialisation (or on an upload), never in the 10 Hz frame. A malformed file
/// is a **rejected load** with a reason string, never a partial one: a field
/// model silently missing its high harmonics looks like a plausible field rather
/// than like a bug, and would quietly corrupt every magnetometer comparison
/// downstream.
///
/// Format (IAGA `igrf14coeffs.txt`): `#` comment lines, a `c/s deg ord …` type
/// row, a `g/h n m <epoch> … <sv>` header row naming each epoch, then one row per
/// coefficient,
///
///     g|h  <n>  <m>  <value at each epoch…>  <secular variation>
///
/// with main-field values in nT and secular variation in nT/yr.
///
/// References:
///  - Alken et al., *International Geomagnetic Reference Field: the thirteenth
///    generation*, Earth Planets Space 73:49, 2021. [alken2021]

#include <cstddef>

#include "environment/igrf.hpp"

namespace polaris::environment {

/// Most epoch columns handled. IGRF-14 tabulates 26 (1900.0 … 2025.0); 40 leaves
/// room for several more generations before a wider file must be re-bounded.
inline constexpr std::size_t kIgrfMaxEpochs = 40;

/// Longest coefficient-file line handled by the fixed read buffer. A row is one
/// value per epoch plus the SV column, each ≤ ~16 characters — comfortably under
/// this. A longer line is a malformed file (rejected, never truncated).
inline constexpr std::size_t kIgrfMaxLineLength = 2048;

/// Longest rejection reason written (fixed, no heap).
inline constexpr std::size_t kIgrfMaxReasonLength = 96;

/// Interval [decimal years] the IAGA secular-variation column is published for,
/// following the last tabulated epoch. Sets @ref IgrfCoefficients::valid_until_year
/// for a snapshot taken on that column.
inline constexpr double kIgrfSvIntervalYears = 5.0;

/// Load the IAGA coefficient file at @p path and produce the snapshot valid at
/// @p decimal_year.
///
/// @param path          Path to the verbatim IAGA file.
/// @param decimal_year  Epoch of interest, e.g. 2026.5.
/// @param out           Filled on success — including
///                      @ref IgrfCoefficients::valid_until_year, the horizon past
///                      which this snapshot stops being the published model —
///                      and left partially written on failure, so do not use it
///                      unless this returned true.
/// @param reason        If non-null, receives a NUL-terminated reason on failure
///                      (truncated to @p reason_cap).
/// @param reason_cap    Capacity of @p reason.
/// @return false if the file cannot be opened, the header is unrecognisable, a
///         row is malformed, a degree is out of range, the file tabulates more
///         than @ref kIgrfMaxEpochs epochs, or @p decimal_year falls before the
///         first tabulated epoch.
///
/// Extrapolating past the last epoch by the SV column is *allowed* — that is the
/// model's defined behaviour through the end of the generation — and is not
/// reported as an error. Extrapolating backwards before the first epoch is
/// refused, since IGRF says nothing there.
bool loadIgrfIaga(const char* path, double decimal_year, IgrfCoefficients& out,
                  char* reason = nullptr, std::size_t reason_cap = 0);

}  // namespace polaris::environment

#endif  // POLARIS_ENVIRONMENT_IGRF_IAGA_HPP
