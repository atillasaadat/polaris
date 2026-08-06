#ifndef POLARIS_GNC_MOMENTUM_HPP
#define POLARIS_GNC_MOMENTUM_HPP

/// @file
/// @brief Wheel-momentum management and magnetic desaturation (design doc §8.5,
/// §7, §9; REQ-ACTL-009, REQ-ACTL-010, REQ-ACTL-011).
///
/// Two pieces, in the order the vehicle uses them:
///
///  1. @ref MomentumManager — what the wheel array is storing, measured from the
///     wheel tachometers: \f$\bar{\mathbf h} = W\,(I_w\boldsymbol\omega_w)\f$ with
///     \f$W\f$ the array matrix whose columns are the wheels' spin axes (§7), the
///     error \f$\Delta\mathbf h = \bar{\mathbf h} - \bar{\mathbf h}_\mathrm{tgt}\f$
///     against a configured target, a **desaturation predicate** with
///     entry/exit hysteresis, and the **envelope** flag §9 watches.
///  2. @ref mtqDesaturation — the cross-product magnetic unloading law that
///     removes that error.
///
/// **The desaturation law.** With the wheels holding attitude, the whole
/// vehicle's angular momentum obeys \f$\dot{\mathbf H} = \boldsymbol\tau_{ext}\f$,
/// and at a held attitude (\f$\boldsymbol\omega\approx0\f$) every external torque
/// lands in the wheels: \f$\dot{\bar{\mathbf h}} = \boldsymbol\tau_{ext}\f$
/// (Wie §7 [wie2008]). A magnetic dipole produces
/// \f$\boldsymbol\tau = \mathbf m\times\mathbf B\f$, which has no component along
/// \f$\hat{\mathbf B}\f$ — so no instantaneous choice of \f$\mathbf m\f$ can dump
/// the momentum parallel to the field, and what removes it is the field direction
/// turning over the orbit. The classical **cross-product law**
/// \f[
///   \mathbf m \;=\; \frac{k_d}{\|\mathbf B\|^2}\,\bigl(\Delta\mathbf h \times
///   \mathbf B\bigr)
/// \f]
/// (Camillo & Markley [camillo1980]; Markley & Crassidis §7.5 [markley2014])
/// gives \f$\boldsymbol\tau = -k_d\,\Delta\mathbf h_\perp\f$ and therefore
/// \f$\tfrac{d}{dt}\|\Delta\mathbf h\|^2 = -2k_d\|\Delta\mathbf h_\perp\|^2 \le 0\f$
/// — the orbit-averaged unloading Camillo & Markley analyse, with \f$k_d\f$ in
/// \f$\mathrm{s^{-1}}\f$ setting the perpendicular decay time constant
/// \f$1/k_d\f$ (the duty division below is exactly what keeps the duty factor
/// out of the decay rate).
///
/// **Field normalisation, for the same reason B-dot normalises.** Dividing by
/// \f$\|\mathbf B\|^2\f$ makes the *torque* — and hence the momentum decay rate —
/// independent of field strength, so the law does not do four times as much at
/// the poles as at the equator (`gnc/bdot.hpp` carries the same argument).
///
/// **Duty factor, and saturation.** The rods are energised only during the §7
/// on-window, so the demand is divided by the duty factor before it leaves this
/// law and the *average* dipole over a control period is what \f$k_d\f$ asked for.
/// The demand is returned **unclamped**, exactly as B-dot's is: a rated moment is
/// a limit in the rod basis, and the caller clamps there, componentwise, once.
///
/// **The dissipativity argument transfers to that clamp, and here is why.** Write
/// \f$\mathbf v = \mathbf B \times \Delta\mathbf h\f$, so the law is
/// \f$\mathbf m = -(k_d/\|\mathbf B\|^2)\,\mathbf v\f$ and
/// \f$\tfrac{d}{dt}\|\Delta\mathbf h\|^2 = 2\,\Delta\mathbf h\cdot(\mathbf m\times
/// \mathbf B) = 2\,\mathbf m\cdot\mathbf v = \sum_i 2\,m_i v_i\f$ by the scalar
/// triple product. Every term of that sum is separately non-positive, because
/// \f$m_i\f$ carries the opposite sign to \f$v_i\f$ by construction. A
/// **componentwise** clamp preserves each sign, so every term stays non-positive
/// and a saturated desaturation still reduces \f$\|\Delta\mathbf h\|\f$ — the
/// identical argument B-dot's per-rod clamp rests on. It holds because the rod
/// triad the caller clamps in is **orthonormal** (`flight::AttitudeController`
/// refuses any other), which is what makes the componentwise decomposition of the
/// triple product legitimate in that basis. A direction-preserving scale would
/// also be dissipative, at lower authority; a clamp in a *skewed* basis would not
/// be, and is refused rather than approximated.
///
/// **Frames, units, conventions.** Momentum is `Vec3<Body>` in N·m·s, field
/// `Vec3<Body>` in tesla, dipole `Vec3<Body>` in A·m², wheel speeds rad/s; time
/// tags are TAI nanoseconds (§3.2). SI throughout.
///
/// **Flight path.** Fixed-size Eigen, no heap, no exceptions, no recursion,
/// bounded loops, finiteness-checked output, no F´ types and no I/O.
///
/// References:
///  - Camillo & Markley, "Orbit-Averaged Behavior of Magnetic Control Laws for
///    Momentum Unloading," J. Guidance and Control 3(6):563–568, 1980
///    [camillo1980] — the cross-product law and its orbit-averaged convergence.
///  - Markley & Crassidis, *Fundamentals of Spacecraft Attitude Determination
///    and Control*, 2014, §7.5 (magnetic control) [markley2014].
///  - Wie, *Space Vehicle Dynamics and Control*, 2nd ed., 2008, §7 (momentum
///    exchange, wheel arrays and unloading) [wie2008].
///  - Design doc §8.5 (momentum management), §7 (the W matrix, the interlock).

#include <cstdint>
#include <Eigen/Core>

#include "gnc/bdot.hpp"
#include "gnc/rw_allocation.hpp"
#include "math/frames.hpp"
#include "math/typed_vector.hpp"

namespace polaris::gnc {

/// Why a @ref mtqDesaturation call produced no dipole.
enum class DesatRefusal : std::uint8_t {
  kNone = 0,         ///< a dipole was produced
  kUnconfigured,     ///< the config failed @ref MtqDesatConfig::isValid
  kBadInput,         ///< non-finite momentum error or field, or a null field
  kNonFiniteOutput,  ///< the computed dipole failed its finiteness guard
};

/// Desaturation-law tuning. No in-code defaults (§19.3).
struct MtqDesatConfig {
  /// Unloading gain \f$k_d\f$ [1/s]. The perpendicular momentum error decays with
  /// time constant \f$1/k_d\f$ while the rods are unsaturated (the duty division
  /// makes the *average* torque what the gain asked for, so the duty factor
  /// cancels out of the decay rate); raising it buys decay rate until the rods
  /// saturate, above which the law is simply full-authority.
  double gain_per_s = 0.0;

  /// On-window fraction of the control period, in (0, 1] (§7). The demand is
  /// divided by it so the *average* dipole is the one the gain asked for — the
  /// same treatment `gnc::BdotConfig::duty_factor` gets, and for the same reason.
  double duty_factor = 0.0;

  /// Both values present and in range.
  bool isValid() const;
};

/// One desaturation evaluation's product.
struct MtqDesatResult {
  /// Commanded dipole [A·m²] during the on-window, already divided by the duty
  /// factor. **Unclamped** — the caller clamps per rod (see the file comment).
  math::Vec3<math::frames::Body> dipole_am2{};

  /// The body torque this dipole would produce, \f$\mathbf m\times\mathbf B\f$
  /// [N·m]. A diagnostic: it is what the wheels will have to take up, so it is
  /// the number that says whether the unloading is competing with the pointing
  /// law for authority.
  math::Vec3<math::frames::Body> torque_nm{};

  /// @ref dipole_am2 is usable.
  bool valid = false;

  /// Why not, when @ref valid is false.
  DesatRefusal refusal = DesatRefusal::kUnconfigured;
};

/// Cross-product magnetic desaturation. Stateless — the whole law is one cross
/// product of this cycle's inputs, so there is nothing to hold across cycles and
/// no object to build.
///
/// @param config             the tuning.
/// @param momentum_error_nms \f$\Delta\mathbf h\f$, body frame [N·m·s] — the
///                           wheel-array momentum less its target
///                           (@ref MomentumManager::update produces it).
/// @param field_tesla        the voted, interlock-gated quiet-window field [T].
/// @param out                receives the dipole and the diagnostics.
/// @return true when @p out carries a usable dipole; false leaves it zeroed,
///         which is the safe command.
bool mtqDesaturation(const MtqDesatConfig& config,
                     const math::Vec3<math::frames::Body>& momentum_error_nms,
                     const math::Vec3<math::frames::Body>& field_tesla, MtqDesatResult& out);

/// Why a @ref MomentumManager::update produced no state.
enum class MomentumRefusal : std::uint8_t {
  kNone = 0,      ///< a momentum state was produced
  kUnconfigured,  ///< the config failed @ref MomentumConfig::isValid
  kWheelInvalid,  ///< a wheel reported no usable speed — see the note on `update`
  kBadInput,      ///< a non-finite wheel speed
};

/// Momentum-manager configuration: the array geometry, the target, and the two
/// thresholds. No in-code defaults (§19.3).
struct MomentumConfig {
  /// Installed wheels, in `[3, kMaxWheels]` — the same count and order the
  /// allocation layer uses, since both describe one array.
  int wheel_count = 0;

  /// Column `i` is wheel `i`'s **spin axis** in body frame — the sim's `W`
  /// matrix (§7), *not* the negated torque-authority axes
  /// `gnc::RwAllocationConfig::axes` carries. The sign difference is real and
  /// deliberate: a wheel spinning along \f$+\hat a\f$ stores momentum along
  /// \f$+\hat a\f$ while its motor torque reacts on the body along \f$-\hat a\f$.
  Eigen::Matrix<double, 3, kMaxWheels> spin_axes = Eigen::Matrix<double, 3, kMaxWheels>::Zero();

  /// Rotor inertia \f$I_w\f$ [kg·m²], one value for an array of identical wheels
  /// (which is what the reference vehicle flies). A mixed array needs a per-unit
  /// value here, and the momentum sum below is already written per wheel.
  double rotor_inertia_kgm2 = 0.0;

  /// Momentum bias to hold [N·m·s], body frame. Zero for a zero-momentum vehicle;
  /// a non-zero target is what a momentum-biased design would carry — and is the
  /// case the §8.5 SISO validity boundary explicitly does not cover.
  Eigen::Vector3d target_nms = Eigen::Vector3d::Zero();

  /// \f$\|\Delta\mathbf h\|\f$ [N·m·s] above which desaturation is required.
  double desat_enter_nms = 0.0;

  /// \f$\|\Delta\mathbf h\|\f$ [N·m·s] below which, held for
  /// @ref desat_confirm_cycles consecutive cycles, desaturation is complete. Must
  /// be below @ref desat_enter_nms: the deadband is what stops the demand
  /// chattering at the threshold, and the confirmation count is what makes the
  /// latch provably clearable rather than merely revocable in principle.
  double desat_exit_nms = 0.0;

  /// Consecutive cycles under @ref desat_exit_nms that end a desaturation.
  std::uint32_t desat_confirm_cycles = 0;

  /// Stored-momentum ceiling [N·m·s] the §9 envelope monitor watches — the bound
  /// the vehicle is *analysed* at, not the wheels' physical capacity. On the
  /// reference vehicle it comes from the §8.5 SISO validity boundary: above the
  /// momentum at which the gyroscopic coupling ratio reaches its limit, the
  /// per-axis margins the pointing loop is certified on stop describing the
  /// vehicle, so exceeding it is a condition FDIR must see even though no wheel
  /// is near its own limit. Must be at or above @ref desat_enter_nms — an
  /// envelope inside the desat threshold would fire before the law that fixes it.
  double envelope_nms = 0.0;

  /// Count in range, every installed axis finite and non-zero, inertia positive,
  /// thresholds positive and ordered, confirmation count non-zero.
  bool isValid() const;
};

/// One cycle's momentum state.
struct MomentumState {
  /// Stored wheel-array momentum in body axes [N·m·s].
  math::Vec3<math::frames::Body> stored_nms{};

  /// \f$\Delta\mathbf h\f$ = @ref stored_nms less the configured target [N·m·s].
  math::Vec3<math::frames::Body> error_nms{};

  /// \f$\|\bar{\mathbf h}\|\f$ [N·m·s].
  double stored_norm_nms = 0.0;

  /// \f$\|\Delta\mathbf h\|\f$ [N·m·s] — the quantity both thresholds gate.
  double error_norm_nms = 0.0;

  /// The desaturation predicate: true from the cycle the error first exceeds
  /// `desat_enter_nms` until it has been under `desat_exit_nms` for
  /// `desat_confirm_cycles` consecutive cycles.
  bool desat_required = false;

  /// @ref stored_norm_nms is above `envelope_nms` (§9). Reported on the *stored*
  /// momentum rather than the error, because the analysis the envelope comes from
  /// is about what the array is holding, not about how far that is from a target.
  bool envelope_exceeded = false;

  /// The state is usable.
  bool valid = false;

  /// Why not, when @ref valid is false.
  MomentumRefusal refusal = MomentumRefusal::kUnconfigured;

  /// The wheel that caused a per-wheel refusal (`kWheelInvalid`, `kBadInput`),
  /// in the array's build order; -1 when the refusal is not attributable to one
  /// wheel. Exists so the refusal EVR can name the unit instead of the caller
  /// guessing.
  int refused_wheel = -1;
};

/// Wheel-array momentum accounting and the desaturation predicate.
///
/// Stateful only in the hysteresis, which is why it is an object. The hysteresis
/// itself is `gnc::RateHysteresis` — the same enter/exit-with-confirmation state
/// machine B-dot's completion predicate uses, reused rather than re-implemented;
/// only the units at this interface differ, which is why the config above names
/// its thresholds in N·m·s.
class MomentumManager {
 public:
  MomentumManager() = default;

  /// Build with @p config. An invalid config leaves the manager **inert**: every
  /// @ref update refuses with @ref MomentumRefusal::kUnconfigured.
  explicit MomentumManager(const MomentumConfig& config);

  bool isConfigured() const { return configured_; }

  const MomentumConfig& config() const { return config_; }

  /// Fold in one set of wheel tachometer readings.
  ///
  /// @param wheel_speeds_radps rotor speeds [rad/s], `config.wheel_count` entries
  ///        in the array's build order.
  /// @param wheel_valid per-wheel validity (§9.1), same indexing.
  /// @param out receives the momentum state.
  /// @return true when @p out is usable.
  ///
  /// **A wheel with no usable speed refuses the whole cycle.** The stored
  /// momentum is a sum over the array, so one missing term is not a small error
  /// in the answer — it is an unknown vector of that wheel's full magnitude, and
  /// treating it as zero would understate the momentum by exactly the amount a
  /// runaway wheel is putting in. The hysteresis is held across a refusal rather
  /// than reset, so a dropout does not silently end a desaturation in progress.
  bool update(const double* wheel_speeds_radps, const bool* wheel_valid, MomentumState& out);

  /// Return to the "not desaturating" state (mode entry, commanded reset).
  void reset();

 private:
  MomentumConfig config_{};
  bool configured_ = false;
  /// True while a desaturation is in progress; the `RateHysteresis` below decides
  /// when it starts and — via the confirmation count — when it may end.
  RateHysteresis hysteresis_{};
};

}  // namespace polaris::gnc

#endif  // POLARIS_GNC_MOMENTUM_HPP
