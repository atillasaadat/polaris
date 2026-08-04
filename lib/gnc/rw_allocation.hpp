#ifndef POLARIS_GNC_RW_ALLOCATION_HPP
#define POLARIS_GNC_RW_ALLOCATION_HPP

/// @file
/// @brief Reaction-wheel torque allocation, L2 and L-∞ (design doc §8.5;
/// REQ-ACTL-002).
///
/// The swappable half of the §8.5 actuator abstraction: control produces one
/// commanded **body torque**, and this layer distributes it across the N-wheel
/// array. The array is described by its axes matrix \f$A\in\mathbb R^{3\times N}\f$
/// — column \f$i\f$ is the body-frame torque wheel \f$i\f$ delivers per unit of
/// commanded wheel torque — so the allocation problem is the underdetermined
/// system \f$A\mathbf u = \boldsymbol\tau\f$ with \f$N>3\f$ and a null space to
/// spend.
///
/// **Sign, stated once.** The sim's assembly matrix \f$W\f$ (`sim/actuators/
/// rw_assembly`) has columns equal to the wheels' **spin axes**, and a wheel's
/// reaction on the body is \f$-I\dot\omega\f$ — the body torque per unit of
/// *commanded motor torque* is therefore \f$-W\f$, and that is what
/// `RwAllocationConfig::axes` must hold. Building it from the spin axes without the
/// negation gives a sign-inverted, perfectly plausible controller, so the caller
/// does the negation once, where the config is read.
///
/// **Two allocations, both wanted (§8.5).**
///  - **Minimum norm (L2):** \f$\mathbf u = A^{+}\boldsymbol\tau\f$ with
///    \f$A^{+} = A^{\top}(AA^{\top})^{-1}\f$, the least-squares minimum-effort
///    solution (Markley & Crassidis §7.3 [markley2014]). It minimises total wheel
///    torque, hence roughly the dissipation, and the pseudo-inverse is a constant
///    of the geometry so it is factorised once at construction and every cycle is
///    one fixed-size matrix–vector product.
///  - **Minimum–maximum (L-∞):** minimise \f$\max_i|u_i|\f$ subject to the same
///    equality. This is the allocation that matters when a wheel is near its
///    torque box: the L2 solution can put one wheel at its limit while others idle,
///    and the L-∞ solution spreads the demand so the array saturates as late as
///    possible. The solution set is \f$\mathbf u_p + \alpha\mathbf n\f$ with
///    \f$\mathbf u_p\f$ the L2 solution and \f$\mathbf n\f$ spanning the null
///    space; for a **four-wheel** array the null space is one-dimensional and
///    \f$f(\alpha)=\max_i|u_{p,i}+\alpha n_i|\f$ is a convex piecewise-linear
///    function of a scalar, minimised exactly by evaluating it at the breakpoints
///    of its upper envelope — a bounded, branch-free-enough search over
///    \f$O(N^2)\f$ candidates, with \f$\alpha=0\f$ always among them so the L-∞
///    result can never be worse than the L2 one.
///
///    **Ceiling, stated rather than hidden:** an array of five or more wheels has
///    a null space of dimension ≥ 2 and its exact min–max is a linear program.
///    This module does **not** solve that; `RwAllocator::allocate` refuses
///    `RwAllocationMethod::kMinMax` with `RwAllocationRefusal::kMinMaxUnsupported`
///    and the caller falls back to L2 (and says so). The upgrade is a bounded
///    active-set/simplex over N ≤ 8 columns; the reference vehicle flies four
///    wheels, so nothing on it is blocked.
///
/// **Saturation preserves the torque direction.** When any \f$|u_i|\f$ exceeds its
/// wheel's limit the *whole* vector is scaled by the single worst ratio, so the
/// delivered torque is \f$s\,\boldsymbol\tau\f$ with \f$0<s<1\f$ — the same
/// direction, less of it. Clipping the offending component instead would deliver a
/// torque pointing somewhere the control law never asked for, which on a
/// three-axis vehicle shows up as a slew about the wrong axis, and silently.
///
/// **Frames, units, conventions.** Torques are `Vec3<Body>` and per-wheel N·m; the
/// axes matrix is body-frame. SI throughout. No time appears — allocation is
/// memoryless.
///
/// **Flight path.** Fixed-size Eigen only, no heap, no exceptions, no recursion,
/// bounded loops, finiteness-checked output, no F´ types and no I/O.
///
/// References:
///  - Markley & Crassidis, *Fundamentals of Spacecraft Attitude Determination
///    and Control*, 2014, §7.3 (actuator/momentum distribution) [markley2014].
///  - Wie, *Space Vehicle Dynamics and Control*, 2nd ed., 2008, §7.4 (reaction
///    wheel configurations and torque distribution) [wie2008].
///  - Design doc §7 (the W matrix), §8.5 (control and allocation).

#include <cstdint>
#include <Eigen/Core>

#include "math/frames.hpp"
#include "math/typed_vector.hpp"

namespace polaris::gnc {

/// Largest wheel array the allocator carries. Matches `flight.SitlMaxUnits` /
/// `flight.GncMaxUnits`, so every wheel the command port array can carry has a
/// column.
inline constexpr int kMaxWheels = 8;

/// Which allocation to run.
enum class RwAllocationMethod : std::uint8_t {
  kMinNorm = 0,  ///< L2 pseudo-inverse (minimum total wheel torque)
  kMinMax = 1,   ///< L-∞ (minimum largest wheel torque); four wheels or fewer
};

/// Why an `RwAllocator::allocate` produced no command.
enum class RwAllocationRefusal : std::uint8_t {
  kNone = 0,           ///< a wheel-torque set was produced
  kUnconfigured,       ///< built with an invalid config (bad count, degenerate array)
  kBadInput,           ///< non-finite commanded torque
  kMinMaxUnsupported,  ///< L-∞ asked for on an array with a null space of dimension > 1
  kNonFiniteOutput,    ///< the computed wheel torques failed their finiteness guard
};

/// Array geometry and per-wheel limits. No in-code defaults (§19.3): the axes and
/// the torque box are vehicle configuration.
struct RwAllocationConfig {
  /// Number of installed wheels, in `[3, kMaxWheels]`. Fewer than three cannot
  /// span the three body axes and is rejected.
  int wheel_count = 0;

  /// Column `i` is the body-frame torque wheel `i` produces per unit commanded
  /// wheel torque — the **negated** spin axis (see the file comment). Columns
  /// past `wheel_count` are ignored. Need not be unit norm; a non-unit column
  /// simply scales that wheel's contribution.
  Eigen::Matrix<double, 3, kMaxWheels> axes = Eigen::Matrix<double, 3, kMaxWheels>::Zero();

  /// Per-wheel commanded-torque limit [N·m], indexed as the columns. Each must be
  /// positive for an installed wheel.
  double max_torque_nm[kMaxWheels] = {};

  /// Smallest acceptable \f$\lambda_{\min}/\lambda_{\max}\f$ of \f$AA^{\top}\f$ —
  /// the three-axis-span gate. Written as an eigenvalue ratio because that is
  /// **scale-free**: it gates the array's *geometry* rather than how large the
  /// wheels are, so re-sizing the units cannot silently change the verdict (the
  /// same argument the observability gates in `gnc/davenport.hpp` are written on).
  /// A collinear or coplanar array fails it and leaves the allocator inert rather
  /// than producing a pseudo-inverse built on a near-singular Gram matrix.
  double min_conditioning = 0.0;

  /// Count in range, every installed column finite and non-zero, every installed
  /// limit positive, `min_conditioning` positive, and the array spanning three
  /// axes to within `min_conditioning`.
  bool isValid() const;
};

/// One allocation's product.
struct RwAllocationResult {
  /// Commanded torque per wheel [N·m], indexed as the config columns. Entries
  /// past `wheel_count` are zero. Meaningful only when @ref valid.
  double torque_nm[kMaxWheels] = {};

  /// The body torque these wheel commands actually deliver, \f$A\mathbf u\f$ —
  /// equal to the commanded torque unless @ref saturated.
  math::Vec3<math::frames::Body> achieved_torque_nm{};

  /// Largest \f$|u_i|\f$ in the solution [N·m]. The quantity L-∞ minimises.
  double max_wheel_torque_nm = 0.0;

  /// Fraction of the commanded torque delivered, in (0, 1]. Below 1 only when
  /// @ref saturated.
  double scale = 1.0;

  /// The array could not deliver the commanded torque and it was scaled down.
  bool saturated = false;

  /// The wheel commands are usable.
  bool valid = false;

  /// Why not, when @ref valid is false.
  RwAllocationRefusal refusal = RwAllocationRefusal::kUnconfigured;
};

/// The allocation layer. Stateless per cycle; the pseudo-inverse and the null
/// direction are factorised once at construction, so `allocate` is bounded
/// arithmetic with no decomposition in the loop.
class RwAllocator {
 public:
  RwAllocator() = default;

  /// Build from @p config, factorising \f$A^{+}\f$ and (for four wheels) the null
  /// direction. An invalid config leaves the allocator **inert**: every
  /// @ref allocate refuses.
  explicit RwAllocator(const RwAllocationConfig& config);

  bool isConfigured() const { return configured_; }

  const RwAllocationConfig& config() const { return config_; }

  /// L-∞ allocation is available for this array (null space of dimension ≤ 1).
  bool supportsMinMax() const { return configured_ && config_.wheel_count <= 4; }

  /// Distribute @p torque_cmd across the wheels.
  ///
  /// @param torque_cmd commanded body torque [N·m].
  /// @param method     which allocation to run.
  /// @param out        receives the per-wheel torques and the diagnostics.
  /// @return true when @p out carries a usable command; false leaves zeros.
  bool allocate(const math::Vec3<math::frames::Body>& torque_cmd, RwAllocationMethod method,
                RwAllocationResult& out) const;

 private:
  /// Body torque delivered by wheel commands @p u.
  Eigen::Vector3d achieved(const Eigen::Matrix<double, kMaxWheels, 1>& u) const;

  RwAllocationConfig config_{};
  bool configured_ = false;
  /// \f$A^{+}\f$, rows past `wheel_count` zero.
  Eigen::Matrix<double, kMaxWheels, 3> pinv_ = Eigen::Matrix<double, kMaxWheels, 3>::Zero();
  /// Unit null-space direction of \f$A\f$ for a four-wheel array; zero otherwise.
  Eigen::Matrix<double, kMaxWheels, 1> null_ = Eigen::Matrix<double, kMaxWheels, 1>::Zero();
  bool has_null_ = false;
};

}  // namespace polaris::gnc

#endif  // POLARIS_GNC_RW_ALLOCATION_HPP
