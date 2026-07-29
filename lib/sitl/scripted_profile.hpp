#ifndef POLARIS_SITL_SCRIPTED_PROFILE_HPP
#define POLARIS_SITL_SCRIPTED_PROFILE_HPP

/// @file
/// @brief Phase-4 placeholder actuator command profile for the SITL rate group.
///
/// Until the Phase-4 GNC estimator/controller components exist, the FSW's SITL
/// rate group runs a `ScriptedCmdSource` that commands actuators from this
/// deterministic profile — a pure function of the macro-step **sim epoch** (TAI
/// nanoseconds) and the unit index. It is intentionally trivial (a slow, small
/// sinusoid): the point is a *non-zero, reproducible* command crossing the §2.4
/// barrier so the closed loop is exercised end to end, not a control law.
///
/// **Shared, header-only, and pure** for the same reason `wire.hpp` is: the
/// flight `ScriptedCmdSource` and the in-process reference callback in the
/// integration test both call these functions, so a two-process run and an
/// in-process run apply byte-identical commands and produce a bitwise-identical
/// truth trace. Determinism follows from purity — identical `int64` sim epoch in
/// → identical `double` out on both sides — not from the profile's shape.
///
/// **Flight-safe:** no heap, no exceptions, no recursion, `<cmath>` only.
/// When real GNC lands this header and `ScriptedCmdSource` are both deleted.

#include <cmath>
#include <cstdint>

namespace polaris::sitl {

/// 2*pi as an exact-enough literal (M_PI is not portable without feature macros).
inline constexpr double kTwoPi = 6.283185307179586476925286766559;

/// Scripted reaction-wheel torque command [N*m], torque mode, for @p wheel_index
/// at macro-step sim epoch @p epoch_tai_ns. Bounded to 1 mN*m — small enough not
/// to saturate a representative wheel, large enough to move the plant.
inline double scriptedWheelTorque(std::int64_t epoch_tai_ns, std::uint32_t wheel_index) {
  const double t_s = static_cast<double>(epoch_tai_ns) * 1.0e-9;
  return 1.0e-3 * std::sin(kTwoPi * t_s / 10.0 + static_cast<double>(wheel_index));
}

/// Scripted magnetorquer dipole command [A*m^2], body frame, for @p rod_index at
/// macro-step sim epoch @p epoch_tai_ns; writes the 3 components into @p out.
/// Bounded to 10 mA*m^2. Placeholder like the wheel profile above.
inline void scriptedMtqDipole(std::int64_t epoch_tai_ns, std::uint32_t rod_index, double out[3]) {
  const double t_s = static_cast<double>(epoch_tai_ns) * 1.0e-9;
  const double phase = kTwoPi * t_s / 20.0 + static_cast<double>(rod_index);
  out[0] = 1.0e-2 * std::sin(phase);
  out[1] = 1.0e-2 * std::cos(phase);
  out[2] = 0.0;
}

}  // namespace polaris::sitl

#endif  // POLARIS_SITL_SCRIPTED_PROFILE_HPP
