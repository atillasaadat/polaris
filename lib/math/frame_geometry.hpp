#ifndef POLARIS_MATH_FRAME_GEOMETRY_HPP
#define POLARIS_MATH_FRAME_GEOMETRY_HPP

/// @file
/// @brief Geometric orbit-relative frames built from position/velocity (§3.1).
///
/// Constructs the two orbit-relative frames Polaris uses as **derived views** of
/// the canonical `EstimatedState` (design doc §3.1, §8.0, REQ-SYS-013): the Hill
/// frame **RIC** (Radial / In-track / Cross-track; RTN is a synonym) used for
/// relative state and covariance display, and **LVLH** (local-vertical /
/// local-horizontal) used for nadir/orbit-relative attitude pointing.
///
/// Both are returned as boundary-tagged passive rotations from ECI
/// (`Quat<RIC, ECI>`, `Quat<LVLH, ECI>`): `v_target = q.rotate(v_eci)`. They are
/// pure functions of the ECI position/velocity — no EOP, no time — so they are
/// exact and available whenever a valid orbit state exists. The full,
/// time-dependent ECI↔ECEF reduction (IAU 2006/2000A) is a separate transform
/// (REQ-CONV-002), shipped in `lib/frames/eci_ecef.cpp`.
///
/// Flight-safe: no heap, no exceptions, fixed-size Eigen. Degenerate inputs
/// (non-finite, zero radius, or radial-parallel-to-velocity so the orbit normal
/// vanishes) cannot define a frame; the builders then return `false` and leave
/// the output untouched — the §3.6 return-code discipline, not a throw.
///
/// Conventions (right-handed triads; rows of the passive DCM are the target-frame
/// axes expressed in ECI):
///  - **RIC**: R = r̂, C = ĥ = (r×v)/|r×v|, I = C×R (≈ +velocity, near-circular).
///  - **LVLH**: ẑ = −r̂ (nadir), ŷ = −ĥ (−orbit-normal), x̂ = ŷ×ẑ (≈ +velocity).
///
/// References:
///  - Vallado, *Fundamentals of Astrodynamics and Applications*, 4th ed., §3.4
///    (RSW/RIC and orbit-relative frames). [vallado2013]
///  - Wertz (ed.), *Spacecraft Attitude Determination and Control*, 1978, §2
///    (LVLH / orbit reference frame). [wertz1978]

#include "math/frames.hpp"
#include "math/quaternion.hpp"
#include "math/typed_vector.hpp"

namespace polaris::math {

/// Build the passive rotation **ECI → RIC** (Hill/RTN) from an inertial orbit
/// state: R = r̂, C = (r×v)̂, I = C×R. On success writes @p out (canonical,
/// `q0 >= 0`) and returns `true`. Returns `false` — leaving @p out unchanged —
/// if @p r_eci / @p v_eci are non-finite, the radius is zero, or r∥v (the orbit
/// normal vanishes and the frame is undefined). REQ-SYS-013.
[[nodiscard]] bool ricFromEci(const Vec3<frames::ECI>& r_eci, const Vec3<frames::ECI>& v_eci,
                              Quat<frames::RIC, frames::ECI>& out);

/// Build the passive rotation **ECI → LVLH** (nadir pointing) from an inertial
/// orbit state: ẑ = −r̂, ŷ = −(r×v)̂, x̂ = ŷ×ẑ. On success writes @p out
/// (canonical) and returns `true`; returns `false` under the same degenerate
/// conditions as ricFromEci(). REQ-SYS-013.
[[nodiscard]] bool lvlhFromEci(const Vec3<frames::ECI>& r_eci, const Vec3<frames::ECI>& v_eci,
                               Quat<frames::LVLH, frames::ECI>& out);

}  // namespace polaris::math

#endif  // POLARIS_MATH_FRAME_GEOMETRY_HPP
