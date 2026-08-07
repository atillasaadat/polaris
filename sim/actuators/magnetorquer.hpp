#ifndef POLARIS_SIM_ACTUATORS_MAGNETORQUER_HPP
#define POLARIS_SIM_ACTUATORS_MAGNETORQUER_HPP

/// @file
/// @brief Magnetorquer (magnetic torque rod) truth model (design doc §7).
///
/// Commanded-dipole-in → actual-dipole-out, per axis (a three-rod set, one rod
/// per body axis). The produced dipole differs from the command through three
/// real effects of a ferromagnetic-core rod:
///  - **Dipole limit**: the command saturates at the rod's rated moment.
///  - **Linearity error**: a scale-factor error across the operating range (the
///    NSS Taurus datasheet quotes ±5%).
///  - **Residual moment + hysteresis**: the core's B-H loop means the moment lags
///    the command and, when de-energized, retains a remanent moment. Both come
///    from one mechanism, modelled as a play (backlash) operator whose half-width
///    is the residual moment: commanding zero after saturating leaves exactly the
///    residual, and the moment "sticks" until the command moves past the loop.
///
/// The rod only produces a **dipole**; the resulting torque τ = m × B is computed
/// by the environment's magnetic-field path (`ResidualDipoleTorque`, §5.2), so
/// this model is deliberately field-agnostic. Power scales with the square of the
/// dipole (I²R, dipole ∝ current).
///
/// Fault injection is first-class (§9): stuck-on holds the last command; a dropout
/// collapses the moment to the (uncontrollable) residual.
///
/// Implements REQ-SIM-003 (actuator truth models with full error stacks) and
/// REQ-SIM-005 (scriptable fault injection).

#include <cmath>
#include <Eigen/Core>
#include <map>
#include <string>

#include "math/frames.hpp"
#include "math/typed_vector.hpp"

namespace polaris::sim::actuators {

/// Magnetorquer parameters, SI. Per-axis identical (a symmetric three-rod set).
struct MagnetorquerSpec {
  double max_dipole_am2 = 0.0;       ///< rated magnetic moment per rod
  double residual_dipole_am2 = 0.0;  ///< remanent moment = hysteresis half-width
  double linearity = 0.0;            ///< scale-factor error, fractional (±0.05 bound)
  double power_max_w = 0.0;          ///< power at full dipole (scales with dipole²)
  /// Time [s] after the drive is removed by which the rod's field has decayed to
  /// insignificance — the coil's L/R current decay plus the core's relaxation.
  /// This is the number the §7 MTQ/MAG interlock's quiet window is sized from, so
  /// it is a per-model catalog value, not a global constant. The model decays the
  /// switched-off moment as `exp(-t/τ)` with `τ = settle_time_s/3`, i.e. ~95 % of
  /// the transient is gone at `settle_time_s`. Zero means no transient (the
  /// pre-Push-54 behaviour: the moment vanishes the instant the drive does).
  double settle_time_s = 0.0;

  /// Build a spec from hardware-library params (the keys used by the
  /// `config/hardware/magnetorquer/*.yaml` entries). See magnetorquer.cpp.
  static MagnetorquerSpec fromParams(const std::map<std::string, double>& params);
};

/// The near-field a dipole @p dipole_am2 sitting at @p source_m produces at
/// @p observer_m, all in body axes [T] — the standard static magnetic dipole
/// field
/// \f[
///   \mathbf B = \frac{\mu_0}{4\pi r^3}\big(3(\mathbf m\cdot\hat{\mathbf r})
///   \hat{\mathbf r} - \mathbf m\big).
/// \f]
///
/// This is what makes the §7 interlock *visible* rather than assumed: an
/// amp-turn-metre-class rod a decimetre from a magnetometer puts hundreds of
/// microtesla on it, an order of magnitude above the ~30 µT ambient, so a sample
/// taken while the rod is energised is not a measurement of the geomagnetic
/// field at all. Truth-side only — no flight component knows this geometry, which
/// is the point: the FSW must protect itself with the duty-cycle schedule, not
/// with a correction model.
///
/// Co-located source and observer (`r = 0`) return zero rather than diverging:
/// the dipole approximation has no meaning inside the source, and a run whose
/// layout is uncharacterised should not be handed an infinite field.
///
/// Reference: Jackson, *Classical Electrodynamics*, 3rd ed., §5.6 [jackson1999].
math::Vec3<math::frames::Body> dipoleNearField(const math::Vec3<math::frames::Body>& dipole_am2,
                                               const Eigen::Vector3d& source_m,
                                               const Eigen::Vector3d& observer_m);

/// A three-axis magnetorquer set. Each body axis is one rod with its own
/// hysteresis state, so the residual/hysteresis history is tracked per axis.
///
/// **Per-axis model.** For a commanded moment \f$m_c\f$ with rated moment
/// \f$m_{\max}\f$, linearity error \f$\epsilon\f$, and residual moment \f$r\f$:
/// \f[
///   x = (1+\epsilon)\,\operatorname{clamp}(m_c,\,-m_{\max},\,m_{\max}), \qquad
///   s \leftarrow \operatorname{clamp}(s,\,x - r,\,x + r), \qquad
///   m = \operatorname{clamp}(s,\,-m_{\max},\,m_{\max})
/// \f]
/// where \f$s\f$ is the per-axis play (backlash) state — the standard scalar
/// play operator, whose memory reproduces both the B-H lag on command reversal
/// and the remanent moment \f$\pm r\f$ at zero command.
///
/// The rod produces a dipole only; the environment applies
/// \f$\boldsymbol{\tau} = \mathbf{m} \times \mathbf{B}\f$. Bus power is
/// \f$P = P_{\max} \sum_i (m_i/m_{\max})^2\f$ (\f$P \propto I^2R\f$ with
/// \f$m \propto I\f$; each rod carries its own winding, so full three-axis
/// drive draws \f$3P_{\max}\f$).
class Magnetorquer {
 public:
  explicit Magnetorquer(const MagnetorquerSpec& spec) : spec_(spec) {}

  /// Command a body-frame dipole [A·m²] and return the actual dipole the rods
  /// produce, advancing the per-axis hysteresis state.
  math::Vec3<math::frames::Body> commandDipole(const math::Vec3<math::frames::Body>& dipole_cmd);

  /// Remove the drive at the end of the §7 on-window: command zero (advancing
  /// the hysteresis, which is what leaves the remanent moment behind) while
  /// latching the moment the rods were carrying, so @ref settlingDipole can
  /// interpolate between them.
  ///
  /// Physically this is the *second* command of every duty-cycled period — the
  /// rod really is driven and then de-energised each control period — so it
  /// advances the play operator a second time rather than pretending the period
  /// was one long command.
  void deenergize();

  /// The moment the rods carry @p seconds_since_off after @ref deenergize —
  /// \f$\mathbf m_\mathrm{off} + (\mathbf m_\mathrm{on} - \mathbf
  /// m_\mathrm{off})\,e^{-3t/T_s}\f$ with \f$T_s\f$ the catalog settle time — so
  /// the transient is ~95 % gone at \f$T_s\f$, exactly the pre-off moment at
  /// \f$t=0\f$, and asymptotically the remanent moment the play operator left.
  /// The residual is what it decays *toward*, not part of the transient: the core
  /// keeps that indefinitely. A zero settle time makes this a step, which is the
  /// model before Push 54.
  math::Vec3<math::frames::Body> settlingDipole(double seconds_since_off) const;

  /// The **time-average** of @ref settlingDipole over `[t0, t0 + dt]`, i.e.
  /// \f$\mathbf m_\mathrm{off} + (\mathbf m_\mathrm{on} - \mathbf
  /// m_\mathrm{off})\,\frac{\tau}{\Delta t}\big(e^{-t_0/\tau} -
  /// e^{-(t_0+\Delta t)/\tau}\big)\f$ with \f$\tau = T_s/3\f$.
  ///
  /// This is what a *torque* over that span must be computed from. The plant
  /// takes one held wrench per integration span (§2.4), and the span covering
  /// the quiet window is a whole macro period while the transient decays in
  /// ~\f$T_s\f$ — so holding the instantaneous value at the span's start would
  /// apply the full driven moment across the entire quiet window, tens of times
  /// the real post-off impulse. The mean gives the exact impulse for a field
  /// that is constant over the span, which over 100 ms it is. Sensing wants the
  /// instantaneous value instead: a magnetometer reads the field at its sample
  /// epoch, not an average.
  math::Vec3<math::frames::Body> settlingDipoleMean(double t0_s, double dt_s) const;

  /// Bus power [W] for the last produced dipole (∝ dipole²).
  double busPower() const;

  const math::Vec3<math::frames::Body>& dipole() const { return actual_; }

  // --- Fault injection (§9) --------------------------------------------------

  /// Stuck-on: ignore new commands, hold the last produced dipole until cleared.
  ///
  /// Note the timing this implies — a rod that is *off* when the fault is
  /// injected is stuck **off**, which is a dropout, not a stuck-on. Injecting a
  /// stuck-on before a run (as the FDIR scenarios do) therefore needs the moment
  /// stated: see @ref injectStuckOnAt.
  void setStuckOn(bool stuck) { fault_stuck_ = stuck; }

  /// Stuck-on **at a stated moment** [A·m²]: the drive latched at @p dipole_am2
  /// and no command can move it. The deterministic form of the fault for the §9
  /// integration suite, which injects before the run starts and so cannot catch
  /// the rod mid-on-window (review-lessons: "a fault injected on the macro-step
  /// seam misses the first sample"). Sets the produced moment, the hysteresis
  /// state and the settle anchor together, so every later query — including
  /// @ref settlingDipole through the quiet window — returns exactly this.
  void injectStuckOnAt(const math::Vec3<math::frames::Body>& dipole_am2) {
    actual_ = dipole_am2;
    pre_off_ = dipole_am2;
    hysteresis_state_ = dipole_am2.eigen();
    fault_stuck_ = true;
  }

  /// Dropout: collapse the controllable moment; only the residual remains.
  void setDropout(bool dropped) { fault_dropout_ = dropped; }

  void clearFaults() {
    fault_stuck_ = false;
    fault_dropout_ = false;
  }

 private:
  /// Play (backlash) hysteresis on one axis: y = clamp(y_prev, x-r, x+r), with the
  /// result clamped to the rated moment. The remanence when x=0 is ±r.
  double applyAxis(double cmd, double& state) const;

  MagnetorquerSpec spec_;
  math::Vec3<math::frames::Body> actual_{Eigen::Vector3d::Zero()};
  Eigen::Vector3d hysteresis_state_ = Eigen::Vector3d::Zero();
  /// Moment carried immediately before the last @ref deenergize — the start of
  /// the settle transient.
  math::Vec3<math::frames::Body> pre_off_{Eigen::Vector3d::Zero()};
  bool fault_stuck_ = false;
  bool fault_dropout_ = false;
};

}  // namespace polaris::sim::actuators

#endif  // POLARIS_SIM_ACTUATORS_MAGNETORQUER_HPP
