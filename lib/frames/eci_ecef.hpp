#ifndef POLARIS_FRAMES_ECI_ECEF_HPP
#define POLARIS_FRAMES_ECI_ECEF_HPP

/// @file
/// @brief ECI↔ECEF reduction, IAU 2006/2000A (design doc §3.1, REQ-CONV-002).
///
/// The single transform between the inertial frame the filter and propagator work
/// in (**ECI** = GCRS/J2000) and the Earth-fixed frame GNSS reports in and the
/// geopotential is defined in (**ECEF** = ITRS). "No ad-hoc rotations" — every
/// ECI↔ECEF conversion in Polaris routes through here (REQ-CONV-002).
///
/// The full reduction is the chain
///   `ITRS = W(x_p, y_p) · R(ERA(UT1)) · Q(TT)  ·  GCRS`
/// — polar motion `W`, Earth rotation `R` about the CIP through the Earth Rotation
/// Angle, and the CIO-based precession/nutation `Q` of the IAU 2006/2000A model.
/// It is not re-derived here: it is `eraC2t06a` from **ERFA**, the BSD-licensed
/// translation of the IAU's own SOFA reference implementation, which *is* the
/// standard. Re-deriving a ~1400-term nutation series would be strictly worse
/// than calling the reference. Our job is the boundary: TAI → the TT/UT1 two-part
/// JD arguments ERFA wants, EOP in radians, and a canonical `Quat<ECEF, ECI>` out.
///
/// **A rotation alone cannot transform a state.** ECEF is rotating, so velocity
/// picks up the transport term `ω⊕ × r`; applying only `R` to a velocity is a
/// ~465 m/s error at the equator. Use the `*State*` functions for position+velocity
/// — `ecefFromEci`/`eciFromEcef` give the orientation only (attitude, geopotential
/// direction), never a velocity.
///
/// ECEF→ECI is the direction REQ-CONV-001 needs on **every GNSS fix**, before the
/// inertial-frame filter and propagator run; ECI→ECEF serves ground-track,
/// geodetic and tesseral-geopotential work.
///
/// Flight-safe (§3.6): no heap, no exceptions, fixed-size Eigen; the ERFA routines
/// in this chain are stack-only C. Failures — non-finite input, or an epoch the
/// uploaded EOP table does not cover — return `false` and leave outputs untouched.
///
/// References:
///  - IERS Conventions (2010), IERS TN 36, §5 (transformation between the CRS and
///    the TRS; IAU 2006/2000A precession-nutation). [iers2010]
///  - Wallace & Capitaine, "Precession-nutation procedures consistent with IAU
///    2006 resolutions", A&A 459, 981 (2006) (the C2T06A chain). [wallace2006]
///  - IAU SOFA / ERFA, `eraC2t06a` (`c2t06a.c`). [erfa2021]
///  - Vallado, *Fundamentals of Astrodynamics and Applications*, 4th ed., §3.7
///    (IAU 2006 reduction; ECEF↔ECI state transformation and the ω⊕ × r term).
///    [vallado2013]

#include <cstddef>

#include "frames/eop.hpp"
#include "math/frames.hpp"
#include "math/quaternion.hpp"
#include "math/typed_vector.hpp"
#include "time/leap_seconds.hpp"
#include "time/timescales.hpp"

namespace polaris::frames {

/// @name Reduction core (EOP already interpolated)
///
/// The table-driven overloads below are the normal entry points; these take a
/// pre-resolved `EopValue` so the ERFA dependency stays out of this header and
/// so a caller doing many transforms at one epoch can look the EOP up once.
/// @{

/// Orientation **ECI → ECEF** at @p t. On success writes the canonical @p out and
/// returns true; returns false, leaving @p out untouched, if @p t or @p eop is
/// non-finite. REQ-CONV-002.
[[nodiscard]] bool ecefFromEci(const time::Tai& t, const EopValue& eop,
                               math::Quat<math::frames::ECEF, math::frames::ECI>& out);

/// Orientation **ECEF → ECI** at @p t — the inverse of ecefFromEci(). REQ-CONV-001.
[[nodiscard]] bool eciFromEcef(const time::Tai& t, const EopValue& eop,
                               math::Quat<math::frames::ECI, math::frames::ECEF>& out);

/// Full state **ECI → ECEF** at @p t: `r_ecef = R r_eci`, and
/// `v_ecef = R v_eci − ω⊕ × r_ecef` — the transport term for the rotating frame.
///
/// ω⊕ is the WGS84 nominal mean rate about the ECEF Z (polar) axis
/// (`constants::wgs84::kEarthRate`). The EOP table's length-of-day correction is
/// deliberately **not** applied: LOD perturbs ω⊕ by ~1e-8 relative, i.e. ~5 µm/s
/// on a LEO velocity — orders below the navigation error budget, and omitting it
/// keeps ω⊕ identical to the value the sim's dynamics already use.
///
/// Returns false and leaves both outputs untouched on non-finite input.
/// REQ-CONV-002.
[[nodiscard]] bool ecefStateFromEci(const time::Tai& t, const EopValue& eop,
                                    const math::Vec3<math::frames::ECI>& r_eci,
                                    const math::Vec3<math::frames::ECI>& v_eci,
                                    math::Vec3<math::frames::ECEF>& r_ecef,
                                    math::Vec3<math::frames::ECEF>& v_ecef);

/// Full state **ECEF → ECI** at @p t: `r_eci = Rᵀ r_ecef`, and
/// `v_eci = Rᵀ (v_ecef + ω⊕ × r_ecef)`. The exact inverse of ecefStateFromEci().
///
/// This is the REQ-CONV-001 ingest path: every GNSS fix arrives as an ECEF state
/// and must land here before the inertial-frame filter and propagator run.
[[nodiscard]] bool eciStateFromEcef(const time::Tai& t, const EopValue& eop,
                                    const math::Vec3<math::frames::ECEF>& r_ecef,
                                    const math::Vec3<math::frames::ECEF>& v_ecef,
                                    math::Vec3<math::frames::ECI>& r_eci,
                                    math::Vec3<math::frames::ECI>& v_eci);

/// @}

/// @name Table-driven overloads
///
/// Resolve the EOP for @p t from the uploaded table, then apply the core above.
/// Return false — outputs untouched — if @p t lies outside the table's span (the
/// table does not extrapolate; see `EopTable::lookup()`) or the core fails.
/// @{

template <std::size_t Capacity>
[[nodiscard]] inline bool ecefFromEci(const time::Tai& t, const EopTable<Capacity>& eop,
                                      const time::LeapSecondTable& leap,
                                      math::Quat<math::frames::ECEF, math::frames::ECI>& out) {
  EopValue e;
  return eop.lookup(t, leap, e) && ecefFromEci(t, e, out);
}

template <std::size_t Capacity>
[[nodiscard]] inline bool eciFromEcef(const time::Tai& t, const EopTable<Capacity>& eop,
                                      const time::LeapSecondTable& leap,
                                      math::Quat<math::frames::ECI, math::frames::ECEF>& out) {
  EopValue e;
  return eop.lookup(t, leap, e) && eciFromEcef(t, e, out);
}

template <std::size_t Capacity>
[[nodiscard]] inline bool ecefStateFromEci(const time::Tai& t, const EopTable<Capacity>& eop,
                                           const time::LeapSecondTable& leap,
                                           const math::Vec3<math::frames::ECI>& r_eci,
                                           const math::Vec3<math::frames::ECI>& v_eci,
                                           math::Vec3<math::frames::ECEF>& r_ecef,
                                           math::Vec3<math::frames::ECEF>& v_ecef) {
  EopValue e;
  return eop.lookup(t, leap, e) && ecefStateFromEci(t, e, r_eci, v_eci, r_ecef, v_ecef);
}

template <std::size_t Capacity>
[[nodiscard]] inline bool eciStateFromEcef(const time::Tai& t, const EopTable<Capacity>& eop,
                                           const time::LeapSecondTable& leap,
                                           const math::Vec3<math::frames::ECEF>& r_ecef,
                                           const math::Vec3<math::frames::ECEF>& v_ecef,
                                           math::Vec3<math::frames::ECI>& r_eci,
                                           math::Vec3<math::frames::ECI>& v_eci) {
  EopValue e;
  return eop.lookup(t, leap, e) && eciStateFromEcef(t, e, r_ecef, v_ecef, r_eci, v_eci);
}

/// @}

}  // namespace polaris::frames

#endif  // POLARIS_FRAMES_ECI_ECEF_HPP
