#ifndef POLARIS_GNC_RW_BIAS_HPP
#define POLARIS_GNC_RW_BIAS_HPP

/// @file
/// @brief Reaction-wheel speed bias: a null-space servo that keeps every wheel
/// off zero without putting momentum on the bus (design doc §8.5; REQ-ACTL-012).
///
/// **The problem.** A zero-momentum vehicle holds its wheels near zero speed,
/// and near zero a wheel is a poor actuator: the drive's torque LSB is a
/// larger fraction of the command, Coulomb friction flips sign under the
/// rotor (a step of \f$2\tau_c\f$ on the body at every crossing), and inside
/// the stiction band a sub-breakaway command moves nothing at all
/// (`sim/actuators/reaction_wheel.hpp`). The friction feedforward
/// (`gnc/rw_friction.hpp`) deliberately gives up inside its deadband for the
/// same reason. The operational fix every wheel-based ACS carries is a
/// **speed bias**: hold the wheels at a non-zero speed so the operating point
/// stays out of the crossing region altogether.
///
/// **Where the bias goes.** On a redundant array the bias lives in the
/// **null space** of the torque map \f$A\f$ (`gnc/rw_allocation.hpp`): a
/// wheel-momentum vector \f$\mathbf h_w\f$ with \f$A\mathbf h_w = 0\f$ spins
/// the wheels and stores *no* body momentum, so it costs neither pointing nor
/// desaturation. On the reference pyramid the null vector is
/// \f$[+1,-1,+1,-1]/2\f$: two wheels forward, two back. This module is that
/// servo: given a target per-wheel momentum pattern \f$\mathbf h_b\f$ it
/// commands the wheel torque
/// \f[
///   \boldsymbol\tau_b = \mathrm{sat}_{\tau_{max}}\!\big(k\,N N^{\top}
///                        (\mathbf h_b - \mathbf h_w)\big),
/// \f]
/// with \f$N\f$ an orthonormal basis of \f$\ker A\f$. The projection is what
/// makes it safe: whatever pattern is configured, only its null-space part is
/// pursued, so the body torque of \f$\boldsymbol\tau_b\f$ is identically zero
/// and a mis-set pattern degrades to a smaller bias, never to a disturbance. A
/// three-wheel array has no null space and the servo is inert by
/// construction; the bus-level alternative there is the momentum manager's
/// `target` (`gnc/momentum.hpp`), which holds a *body* momentum bias the
/// desaturation then defends.
///
/// **Sizing.** The bias magnitude is a fraction of a wheel's capacity — ~10 %
/// on the reference vehicle, 600 rpm of 6000 — chosen so the operating point
/// is far outside the stiction/deadband region and inside the per-wheel
/// capacity with the pointing envelope on top. The gain sets a slow time
/// constant (\f$1/k\f$, tens to hundreds of seconds): the servo corrects the
/// drift the L-∞ allocation and friction asymmetry inject into the null space,
/// it does not race the pointing loop. The torque cap keeps it a trim.
///
/// **Toggle.** A zero pattern (or a zero gain) is "off": the servo returns a
/// zero torque and reports itself inactive, and the vehicle flies
/// zero-momentum wheels as before. This is a parameter, so it is a ground
/// decision that survives a reboot with `ParameterDb`.
///
/// **Frames, units, conventions.** Wheel momenta and torques are per-wheel
/// scalars about each wheel's own spin axis [N·m·s], [N·m], in the same
/// (spin-axis) sense the momentum manager uses; the torque map is the
/// allocation's (negated spin axes) and only its null space is used, which is
/// the same for either sign convention.
///
/// **Flight path.** Fixed-size Eigen, no heap, no exceptions, no recursion,
/// bounded loops, finiteness-checked output, no F´ types and no I/O.
///
/// References:
///  - Markley & Crassidis, *Fundamentals of Spacecraft Attitude Determination
///    and Control*, 2014, §7.5 (reaction-wheel null motion) [markley2014].
///  - Karnopp, "Computer Simulation of Stick-Slip Friction in Mechanical
///    Dynamic Systems," J. Dyn. Sys. Meas. Control 107(1):100–103, 1985
///    [karnopp1985].

#include <cstdint>
#include <Eigen/Core>

#include "gnc/rw_allocation.hpp"

namespace polaris::gnc {

/// Why an `RwBiasServo::update` produced no torque.
enum class RwBiasRefusal : std::uint8_t {
  kNone = 0,      ///< a torque was produced (possibly zero: inactive)
  kUnconfigured,  ///< invalid config, or an array with no null space
  kBadInput,      ///< a non-finite wheel momentum
};

/// Null-space bias servo tuning. No in-code defaults (§19.3).
struct RwBiasConfig {
  /// Installed wheels, `[3, kMaxWheels]`.
  int wheel_count = 0;
  /// Body torque per unit wheel torque, as `RwAllocationConfig::axes`. Only its
  /// null space is used.
  Eigen::Matrix<double, 3, kMaxWheels> axes = Eigen::Matrix<double, 3, kMaxWheels>::Zero();
  /// Target per-wheel momentum pattern [N·m·s]. All zero = off.
  double bias_nms[kMaxWheels] = {};
  /// Servo gain [1/s]: torque per unit momentum error. Zero = off.
  double gain_per_s = 0.0;
  /// Cap on the servo's per-wheel torque [N·m]; keeps it a trim.
  double max_torque_nm = 0.0;

  /// Finite; count in range; gain and cap non-negative (zero = off); torque cap
  /// positive when the gain is.
  bool isValid() const;
};

struct RwBiasResult {
  /// Per-wheel servo torque [N·m], zero past `wheel_count`.
  double torque_nm[kMaxWheels] = {};
  /// Null-space content of the wheel momentum before this cycle's command [N·m·s]
  /// — what the servo sees and steers.
  double null_space_nms = 0.0;
  /// The servo commanded a non-zero torque this cycle.
  bool active = false;
  bool valid = false;
  RwBiasRefusal refusal = RwBiasRefusal::kUnconfigured;
};

/// Null-space wheel-speed servo. Stateless per cycle; the null basis is built
/// once at construction.
class RwBiasServo {
 public:
  explicit RwBiasServo(const RwBiasConfig& config);

  bool isConfigured() const { return configured_; }

  /// Dimension of the array's null space (0 on a three-wheel array).
  int nullDimension() const { return null_dim_; }

  /// The bias pattern actually pursued: the null-space projection of the
  /// configured one [N·m·s]. What a mis-set pattern degrades to.
  double effectiveBiasNms(int wheel) const;

  /// One cycle from the measured per-wheel momenta [N·m·s].
  bool update(const double* wheel_momentum_nms, RwBiasResult& out) const;

 private:
  RwBiasConfig config_{};
  bool configured_ = false;
  int null_dim_ = 0;
  /// Orthonormal null basis, columns 0..null_dim_-1 valid.
  Eigen::Matrix<double, kMaxWheels, kMaxWheels> null_basis_ =
      Eigen::Matrix<double, kMaxWheels, kMaxWheels>::Zero();
  Eigen::Matrix<double, kMaxWheels, 1> effective_bias_ =
      Eigen::Matrix<double, kMaxWheels, 1>::Zero();
};

}  // namespace polaris::gnc

#endif  // POLARIS_GNC_RW_BIAS_HPP
