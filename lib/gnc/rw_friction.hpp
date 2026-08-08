#ifndef POLARIS_GNC_RW_FRICTION_HPP
#define POLARIS_GNC_RW_FRICTION_HPP

/// @file
/// @brief Reaction-wheel drive friction feedforward (design doc §8.5;
/// REQ-ACTL-010, tightening REQ-ACTL-002).
///
/// **The problem this exists for.** A torque-mode wheel drive is commanded a
/// motor torque, but the rotor obeys \f$I\dot\omega = \tau_m + \tau_f\f$ with
/// bearing friction \f$\tau_f = -\mathrm{sgn}(\omega)(\tau_c + b|\omega|)\f$.
/// The reaction the *body* feels is \f$-I\dot\omega = -(\tau_m + \tau_f)\f$,
/// so the §8.5 allocation — which solves \f$A\mathbf u = \boldsymbol\tau\f$ on
/// the assumption that commanding \f$u_i\f$ delivers \f$-u_i\f$ about wheel
/// \f$i\f$'s axis — is wrong by exactly \f$-\tau_f\f$ per wheel. On a *loaded*
/// array that error is secular, not zero-mean: every spinning wheel's Coulomb
/// reaction points the same way for as long as its speed keeps its sign, and on
/// the reference vehicle the four-wheel pyramid's \f$\tau_c = 10^{-4}\f$ N·m
/// reactions sum to 2.3e-4 N·m on the body — more than twice the pointing PID
/// integrator's entire authority. The measured cost was 2.9° of steady-state
/// pointing error against REQ-ACTL-002's 1.0°.
///
/// This module supplies the missing term. Applied **after** the allocation
/// produces per-wheel demands and **before** they are commanded, it adds
/// \f$-\tau_f\f$ to each wheel so the *net* rotor torque is the one the
/// allocation asked for and the delivered body torque is the commanded one. The
/// allocation itself is untouched: friction is a property of a drive, not of the
/// array geometry, and putting it in the allocation would make the pseudo-inverse
/// a function of wheel speed.
///
/// **The zero crossing, and what the deadband costs.** \f$\mathrm{sgn}(\omega)\f$
/// is discontinuous at \f$\omega = 0\f$, and a feedforward built on it chatters:
/// a wheel dithering about zero speed sees the compensation flip sign every
/// sample, which injects a square wave of amplitude \f$2\tau_c\f$ into the body
/// and can sustain a limit cycle at exactly the operating point (near-empty
/// wheels) where the vehicle is otherwise quietest. The compensation therefore
/// runs the sign through a **bounded linear blend** over a configured deadband
/// speed \f$\omega_{db}\f$:
/// \f[
///   s(\omega) = \mathrm{clamp}(\omega/\omega_{db},\,-1,\,+1),\qquad
///   c = k\,\bigl(\tau_c\,s(\omega) + b\,\omega\bigr).
/// \f]
/// This is Karnopp's zero-velocity band [karnopp1985] in its feedforward form,
/// and the honest statement of the trade is: **inside \f$|\omega| <
/// \omega_{db}\f$ the friction is deliberately under-compensated**, linearly in
/// speed, down to nothing at \f$\omega = 0\f$. Nothing is claimed there. That is
/// the price of not chattering, and it is the right price: the residual error
/// inside the band is bounded by \f$\tau_c\f$ and is exactly the error the vehicle
/// already lived with, whereas a chattering feedforward is a new disturbance the
/// vehicle did not have. \f$\omega_{db}\f$ should be set from the tachometer's
/// own resolution and noise — below it, the *sign* of the speed is not a
/// measurement — not from the control loop.
///
/// The viscous term \f$b\,\omega\f$ needs no blend: it is continuous through zero
/// and vanishes there on its own. It is compensated for completeness rather than
/// necessity (on the reference wheels it is ~5e-6 N·m at 1 rad/s, two orders
/// below the Coulomb term), because leaving a modelled, continuous, exactly-known
/// term out would be arbitrary.
///
/// The catalog's third rundown term, aero friction \f$\propto \omega^2\f$, is
/// **not** compensated. It is zero on every wheel in the committed catalog (a
/// wheel in vacuum), and compensating a term nobody has a value for would be a
/// guess dressed as a model. A wheel with a non-zero aero coefficient is
/// therefore under-compensated at high speed by that term, which is safe in the
/// direction described next.
///
/// **Never over-compensate, and why the direction matters.** Feedforward against
/// a *modelled* disturbance is monotonically helpful only while the model does
/// not exceed the truth: compensating a fraction \f$k \le 1\f$ of a disturbance
/// leaves \f$(1-k)\f$ of it — smaller, same sign, never worse. Over-compensating
/// (\f$k > 1\f$) reverses the sign of the residual and, past \f$k = 2\f$, makes it
/// larger than the uncompensated one. That asymmetry is why the per-wheel scale
/// factor @ref RwFrictionConfig::scale carries a **policy of \f$k \le 1\f$**: a
/// flight campaign that measures the real rundown trims *down* toward the truth,
/// and only a campaign that has measured the friction to be *larger* than the
/// catalog says has any business going above 1. The code does not clamp it —
/// silently rewriting a commanded trim would hide exactly the case an operator
/// needs to see — it validates it as finite and non-negative and states the
/// policy here. With \f$k \le 1\f$ the guarantee is exact and unconditional:
/// \f$|c| \le k(\tau_c + b|\omega|) \le |\tau_f(\omega)|\f$.
///
/// **What that open-loop argument does not cover, measured.** "Smaller residual
/// disturbance" is not the same as "better pointing", and on this vehicle the two
/// part company: the friction being removed was also passively *damping* the
/// stored wheel momentum, and a near-empty array runs at the speeds where the
/// compensation's sign is least trustworthy. Flown end to end (SITL rows
/// `DesaturationDumpsMomentumWhilePointingHolds` and
/// `InertialHoldConvergesUnderThePointingBound`), the worst pointing in degrees
/// went 2.77 / 0.12 uncompensated, to 1.38 / 0.25 at \f$k = 0.5\f$, to 1.52 /
/// 1.09 at \f$k = 1.0\f$ — loaded array first, near-empty second. Full
/// compensation is *worse than half* on both. So the trim is a genuine
/// closed-loop tuning parameter, not merely a safety margin on a model, and
/// \f$k = 1\f$ is not the obvious right answer even when the model is exactly
/// right. The reference vehicle flies 0.5.
///
/// **Saturation cannot eat the control demand.** The compensated command is
/// clamped to the wheel's torque box, and the clamp is applied so that what it
/// removes is the *compensation*: the allocation's demand \f$d_i\f$ (already
/// inside the box, and re-asserted here rather than trusted) is preserved and the
/// applied compensation is \f$u_i - d_i\f$, which has the same sign as \f$c_i\f$
/// and magnitude at most \f$|c_i|\f$. A wheel commanded at its limit therefore
/// gets no friction compensation at all and keeps every newton-metre of the
/// control demand — the correct priority, since the demand is the closed loop and
/// the compensation is an open-loop refinement. Truncation is reported in
/// @ref RwFrictionResult::saturated so the caller can alert on it rather than
/// discover it as unexplained error.
///
/// **A wheel with no usable tachometer is not compensated.** The sign of the
/// friction is the sign of the speed, so without a speed there is no
/// feedforward — that wheel passes its demand through unchanged and says so in
/// @ref RwFrictionResult::compensated. Guessing a sign here would apply a torque
/// in the wrong direction half the time, which is strictly worse than the
/// uncompensated vehicle this module exists to improve on.
///
/// **Frames, units, conventions.** Per-wheel torques and speeds, indexed as the
/// §8.5 allocation's columns; N·m and rad/s. Speeds are the rotor speeds about
/// each wheel's own spin axis, sign positive along that axis — the same
/// convention `gnc/momentum.hpp` reads the tachometers in. SI throughout. No
/// frames appear: this layer is per-wheel scalars on both sides. Memoryless —
/// there is no state to reset.
///
/// **Flight path.** Fixed-size storage, no heap, no exceptions, no recursion,
/// bounded loops, finiteness-checked output, no F´ types and no I/O.
///
/// References:
///  - Armstrong-Hélouvry, Dupont & Canudas de Wit, *A survey of models, analysis
///    tools and compensation methods for the control of machines with friction*,
///    Automatica 30(7), 1994, §2 (the Coulomb + viscous model) and §5.1
///    (model-based friction compensation and its over/under-compensation
///    asymmetry) [armstrong1994].
///  - Karnopp, *Computer simulation of stick-slip friction in mechanical dynamic
///    systems*, J. Dyn. Sys. Meas. Control 107(1), 1985 — the zero-velocity band
///    that replaces the discontinuous sign [karnopp1985].
///  - Wie, *Space Vehicle Dynamics and Control*, 2nd ed., 2008, §7.4 — the
///    reaction-wheel array and the per-wheel torque box this layer clamps to;
///    the same section `gnc/rw_allocation.hpp` is written on [wie2008].
///  - Design doc §8.5 (control, allocation and the wheel command path); the
///    friction model being inverted is `sim/actuators/reaction_wheel.cpp`
///    `ReactionWheel::frictionTorque`, whose coefficients come from
///    `config/hardware/reaction_wheel/*.yaml`.

#include <cstdint>

#include "gnc/rw_allocation.hpp"

namespace polaris::gnc {

/// Why a `RwFrictionCompensator::compensate` produced no command.
enum class RwFrictionRefusal : std::uint8_t {
  kNone = 0,         ///< a compensated wheel-torque set was produced
  kUnconfigured,     ///< built with an invalid config
  kBadInput,         ///< a non-finite demand, or a speed flagged valid that is not finite
  kNonFiniteOutput,  ///< the compensated torques failed their finiteness guard
};

/// Drive friction model and trim, per wheel. No in-code defaults (§19.3): every
/// coefficient is a hardware-catalog fact about the installed units and reaches
/// the flight component through the config compiler.
struct RwFrictionConfig {
  /// Number of installed wheels, in `[1, kMaxWheels]`. Matches the allocation's
  /// `wheel_count`; entries past it are ignored.
  int wheel_count = 0;

  /// Coulomb (dry) friction torque magnitude [N·m], the `dry_friction_nm` of the
  /// wheels' catalog entry. One value because the reference vehicle flies four
  /// identical wheels; a mixed array needs this per-unit, exactly as the axes
  /// already are.
  double dry_friction_nm = 0.0;

  /// Viscous friction coefficient [N·m/(rad/s)], the catalog's
  /// `viscous_friction_nm_s`.
  double viscous_friction_nm_s = 0.0;

  /// Blend half-width \f$\omega_{db}\f$ [rad/s]. The Coulomb term ramps linearly
  /// from zero at \f$\omega = 0\f$ to full magnitude at \f$|\omega| =
  /// \omega_{db}\f$. Set it from the tachometer's resolution and noise: below
  /// that speed the *sign* of \f$\omega\f$ is not a measurement, so a
  /// full-magnitude compensation there would be a coin flip applied at
  /// \f$\tau_c\f$. Must be strictly positive — a zero deadband is the
  /// discontinuous `sign()` this module exists to avoid, and is refused rather
  /// than accepted as "no blending".
  double deadband_radps = 0.0;

  /// Per-wheel commanded-torque limit [N·m], the same box the allocation clamps
  /// to. Each must be positive for an installed wheel.
  double max_torque_nm[kMaxWheels] = {};

  /// Per-wheel trim on the modelled friction, dimensionless. **Policy: \f$\le
  /// 1\f$** — see the file comment on why over-compensation is the one direction
  /// that can make the vehicle worse. Not clamped: a trim above 1 is a decision
  /// an operator may have data for, and silently rewriting it would hide it.
  /// Zero disables compensation on that wheel, which is the correct value for a
  /// unit whose drive is not in torque mode. Must be finite and non-negative.
  double scale[kMaxWheels] = {};

  /// Count in range, coefficients finite and non-negative, deadband positive,
  /// every installed limit positive, every installed scale finite and
  /// non-negative.
  bool isValid() const;
};

/// One compensation's product.
struct RwFrictionResult {
  /// Compensated per-wheel commands [N·m] — what the drives are told. Entries
  /// past `wheel_count` are zero. Meaningful only when @ref valid.
  double torque_nm[kMaxWheels] = {};

  /// What was actually added to each demand [N·m], **after** the torque-box
  /// clamp. Zero on a wheel with no usable tachometer.
  double compensation_nm[kMaxWheels] = {};

  /// A usable speed was available and the wheel was compensated. False means the
  /// demand passed through untouched, which is the uncompensated behaviour and
  /// not a failure.
  bool compensated[kMaxWheels] = {};

  /// The torque box truncated the compensation on at least one wheel. The
  /// control demand is intact on every wheel regardless; this says the open-loop
  /// refinement was the part that did not fit.
  bool saturated = false;

  /// Largest \f$|c_i|\f$ actually applied [N·m], for telemetry.
  double max_compensation_nm = 0.0;

  /// The commands are usable.
  bool valid = false;

  /// Why not, when @ref valid is false.
  RwFrictionRefusal refusal = RwFrictionRefusal::kUnconfigured;
};

/// The drive-level friction feedforward. Stateless; an invalid config leaves it
/// **inert** and every call refuses, which the caller must read as "command the
/// uncompensated demand", never as "command nothing".
class RwFrictionCompensator {
 public:
  RwFrictionCompensator() = default;

  explicit RwFrictionCompensator(const RwFrictionConfig& config);

  bool isConfigured() const { return configured_; }

  const RwFrictionConfig& config() const { return config_; }

  /// The blend \f$s(\omega) = \mathrm{clamp}(\omega/\omega_{db}, -1, +1)\f$.
  /// Exposed because it is the one piece of this module worth testing on its own
  /// (continuity at zero, boundedness, oddness) and worth plotting when a
  /// deadband is chosen.
  double blend(double speed_radps) const;

  /// Add the friction feedforward to @p demand_nm.
  ///
  /// @param demand_nm   per-wheel torques from the §8.5 allocation [N·m].
  /// @param speed_radps per-wheel rotor speeds [rad/s], signed along each spin
  ///                    axis. Read only where @p speed_valid is set.
  /// @param speed_valid per-wheel tachometer admissibility — the caller's
  ///                    validity **and** staleness verdict (§9.1), not the raw
  ///                    sensor flag.
  /// @param out         receives the compensated commands and the diagnostics.
  /// @return true when @p out carries usable commands; false leaves it zeroed
  ///         and the caller must fall back to the uncompensated demand.
  bool compensate(const double* demand_nm, const double* speed_radps, const bool* speed_valid,
                  RwFrictionResult& out) const;

 private:
  RwFrictionConfig config_{};
  bool configured_ = false;
};

}  // namespace polaris::gnc

#endif  // POLARIS_GNC_RW_FRICTION_HPP
