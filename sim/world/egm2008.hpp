#ifndef POLARIS_SIM_WORLD_EGM2008_HPP
#define POLARIS_SIM_WORLD_EGM2008_HPP

/// @file
/// @brief Loader for EGM2008 spherical-harmonic coefficients in ICGEM `.gfc`
/// format (REQ-SIM-002; design doc §5.2, §3.7).
///
/// The EGM2008 geopotential model is distributed by ICGEM (icgem.gfz-potsdam.de)
/// in the `.gfc` text format, committed **verbatim** as ground/sim reference data
/// (§3.7) and parsed here as-is — no bespoke intermediate. The coefficients are
/// already **fully-normalized** `Cbar_nm / Sbar_nm`, exactly the convention
/// `GravityCoeffs` stores, so loading is a near-1:1 fill of the triangular table.
///
/// This is sim/ground code (truth uses a high-degree field; the onboard model is a
/// lower-order uploaded set — `sim/CLAUDE.md`), so heap / iostream / exceptions are
/// permitted here.
///
/// `.gfc` layout: a header block terminated by `end_of_head` carrying
/// `earth_gravity_constant`, `radius`, `max_degree`, `norm fully_normalized`; then
/// data lines `gfc  L  M  Cbar  Sbar  [sigmaC sigmaS]`. Some distributions use a
/// Fortran `D` exponent — normalized to `E` on read (robust parse, not a rewrite).
///
/// References:
///  - Pavlis et al., "The development and evaluation of EGM2008", JGR 117, 2012.
///    [pavlis2012]
///  - Barthelmes & Förste, "The ICGEM `gfc` data format", GFZ. [icgemformat]

#include <istream>
#include <string>

#include "world/gravity_field.hpp"

namespace polaris::sim::world {

/// Header scalars read from a `.gfc` model (the model's own `GM` and reference
/// radius, which differ slightly from WGS84 and should be passed to the
/// `SphericalHarmonicGravity` ctor for a self-consistent field).
struct Egm2008Header {
  double gm = 0.0;      ///< earth_gravity_constant [m^3/s^2]
  double radius = 0.0;  ///< reference radius Re [m]
  int max_degree = 0;   ///< model's max degree as declared in the header
};

/// Parse an ICGEM `.gfc` stream into fully-normalized `GravityCoeffs`, truncated
/// to @p max_degree (clamped to the model's own max degree). `Cbar_00` is forced
/// to 1 (point-mass term) regardless of the file. On success @p header, if given,
/// receives the model's GM / radius / declared max degree. Throws
/// `std::runtime_error` if the stream has no valid coefficients (sim-side; §3.6
/// return-code discipline is a flight rule, not a sim one).
GravityCoeffs loadEgm2008Gfc(std::istream& in, int max_degree, Egm2008Header* header = nullptr);

/// Convenience overload: open @p path and parse it. Throws `std::runtime_error`
/// if the file cannot be opened.
GravityCoeffs loadEgm2008Gfc(const std::string& path, int max_degree,
                             Egm2008Header* header = nullptr);

}  // namespace polaris::sim::world

#endif  // POLARIS_SIM_WORLD_EGM2008_HPP
