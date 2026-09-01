#ifndef POLARIS_FRAMES_TEME_ECI_HPP
#define POLARIS_FRAMES_TEME_ECI_HPP

/// @file
/// @brief TEME↔ECI, the frame boundary SGP4 sits behind (design doc §3.1, §8.3;
/// REQ-CONV-002, REQ-ODP-003).
///
/// SGP4 works in **True Equator, Mean Equinox** and nothing else in Polaris
/// does. TEME is not J2000/GCRS: it shares the true-of-date equator but places
/// its origin of right ascension at a *mean* equinox, so the two differ by the
/// **equation of the equinoxes** on top of the full precession-nutation chain.
///
/// The size of that difference is the reason this file exists rather than an
/// assignment. Near a TLE's own epoch it is tens of metres — small enough to
/// look like noise, large enough to bias a pointing solution — and it grows with
/// precession to order 100 km over decades. A TEME position used directly as ECI
/// therefore produces a **plausible wrong answer**, which is exactly the class
/// Golden Rule 4 exists to make a compile error: `Vec3<TEME>` will not bind to a
/// `Vec3<ECI>` parameter, so the conversion cannot be skipped by accident.
///
/// ## The chain
///
/// Following [vallado2013] §3.7 (`teme2eci`), with the rotations supplied by
/// **ERFA** — the BSD-licensed translation of the IAU's SOFA reference — rather
/// than re-derived, for the same reason `eci_ecef.hpp` gives:
///
///     GCRS --P(TT)--> MOD --N(TT)--> TOD --R3(Eq_eq)--> TEME
///
/// so the direction we want is the transpose of that product. `P` is the IAU-76
/// precession (`eraPmat76`), `N` the IAU-80 nutation (`eraNutm80`), and `Eq_eq`
/// the IAU-1994 equation of the equinoxes **including its complementary terms**
/// (`eraEqeq94`). The 1976/1980 models are deliberate and not an oversight:
/// TEME is defined by the theory SGP4 was fitted with, so pairing it with the
/// modern IAU 2006/2000A chain that `eci_ecef.hpp` uses would be a *different*
/// frame. The two files disagreeing on precession model is correct.
///
/// ## Velocity
///
/// Unlike ECI↔ECEF, there is **no transport term**. Both frames are
/// quasi-inertial and the rotation between them varies on a precession
/// timescale, so the same matrix carries position and velocity. Adding an
/// `ω × r` here would be the mirror of the mistake `eci_ecef.hpp` warns about.
///
/// ## Accuracy and what it does not need
///
/// The conversion is exact to the models named; it takes **no EOP** and no UT1,
/// because nothing in the chain involves Earth rotation. That matters
/// operationally: a TLE can be propagated and converted with no uploaded EOP
/// table at all, which is precisely the situation a long GNSS outage leaves the
/// vehicle in. (The IAU-76/80 models themselves differ from IAU 2006/2000A by
/// well under a metre at LEO — far inside SGP4's own ~1 km, so the model choice
/// is not the error budget here.)
///
/// Flight-safe (§3.6): no heap, no exceptions, fixed-size Eigen; the ERFA
/// routines used are stack-only C. Non-finite inputs return `false` and leave
/// outputs untouched.

#include "math/frames.hpp"
#include "math/typed_vector.hpp"
#include "time/timescales.hpp"

namespace polaris::frames {

/// The TEME→ECI direction-cosine matrix at @p t, as an Eigen matrix.
///
/// Exposed for callers that must rotate several vectors at one epoch (a state
/// plus a covariance, say) without recomputing a ~100-term nutation series each
/// time. Returns false on a non-finite epoch.
bool temeToEciMatrix(const time::Tai& t, Eigen::Matrix3d& out);

/// Rotate a TEME position (or any single TEME direction) into ECI.
bool eciFromTeme(const time::Tai& t, const math::Vec3<math::frames::TEME>& teme,
                 math::Vec3<math::frames::ECI>& out);

/// Rotate an ECI vector into TEME — the inverse, for building a TLE-frame
/// comparison against a state the vehicle holds in ECI.
bool temeFromEci(const time::Tai& t, const math::Vec3<math::frames::ECI>& eci,
                 math::Vec3<math::frames::TEME>& out);

/// Convert a full TEME state (position **and** velocity) to ECI.
///
/// The same rotation serves both — see the file header on why there is no
/// transport term. Provided as one call so a caller cannot rotate position and
/// forget velocity, which is the asymmetry that makes the ECEF case dangerous.
bool eciStateFromTeme(const time::Tai& t, const math::Vec3<math::frames::TEME>& position_teme,
                      const math::Vec3<math::frames::TEME>& velocity_teme,
                      math::Vec3<math::frames::ECI>& position_eci,
                      math::Vec3<math::frames::ECI>& velocity_eci);

}  // namespace polaris::frames

#endif  // POLARIS_FRAMES_TEME_ECI_HPP
