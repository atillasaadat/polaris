#ifndef POLARIS_SIM_DYNAMICS_RIGID_BODY_HPP
#define POLARIS_SIM_DYNAMICS_RIGID_BODY_HPP

/// @file
/// @brief Coupled translational + rotational 6DOF rigid-body EOM (REQ-SIM-001).
///
/// The truth plant. The 13-element state is
///   `[ r_ECI(3) | v_ECI(3) | q_Body<-ECI(4, scalar-first) | omega_Body(3) ]`.
/// Its time derivative is
///   dr/dt = v
///   dv/dt = a_ext                              (from the force/torque model, ECI)
///   dq/dt = 1/2 [0, omega] (x) q               (JPL scalar-first, Body-rate)
///   dw/dt = J^-1 ( tau_ext - omega x (J omega) ) (Euler's equation, Body)
/// integrated with the adaptive RK8(9) stepper. The quaternion is renormalized
/// after each accepted step so it stays on the unit manifold.
///
/// References:
///  - Markley & Crassidis, *Fundamentals of Spacecraft Attitude Determination
///    and Control*, 2014, §3.1 (quaternion kinematics). [markley2014]
///  - Hughes, *Spacecraft Attitude Dynamics*, 1986, §4 (Euler's equation).
///  - Montenbruck & Gill, *Satellite Orbits*, 2000. [montenbruck2000]

#include <Eigen/Core>
#include <Eigen/LU>  // Matrix3d::inverse

#include "dynamics/force_torque.hpp"
#include "dynamics/integrator.hpp"
#include "math/quaternion.hpp"
#include "state/truth_state.hpp"
#include "time/duration.hpp"
#include "time/timescales.hpp"

namespace polaris::sim::dynamics {

/// 6DOF rigid-body truth propagator. Holds the (constant) inertia tensor and a
/// reference to the external force/torque model; not the owner of either.
///
/// **Equations of motion.** The 13-element state
/// \f$\mathbf{y} = [\mathbf{r}_I,\ \mathbf{v}_I,\ \bar{q}_{B\leftarrow I},
/// \ \boldsymbol{\omega}_B]\f$ (position and velocity in ECI, attitude quaternion
/// Body\f$\leftarrow\f$ECI in JPL scalar-first order, body rate in Body) evolves as
/// \f[
///   \dot{\mathbf{r}}_I = \mathbf{v}_I, \qquad
///   \dot{\mathbf{v}}_I = \mathbf{a}_\mathrm{ext}, \qquad
///   \dot{\bar q} = \tfrac{1}{2}\,\begin{bmatrix} 0 \\ \boldsymbol{\omega}_B \end{bmatrix}
///     \otimes \bar q, \qquad
///   \dot{\boldsymbol{\omega}}_B = J^{-1}\!\left(\boldsymbol{\tau}_\mathrm{ext}
///     - \boldsymbol{\omega}_B \times J\,\boldsymbol{\omega}_B\right),
/// \f]
/// with \f$\mathbf{a}_\mathrm{ext}\f$ (ECI) and \f$\boldsymbol{\tau}_\mathrm{ext}\f$
/// (Body) supplied by the force/torque model, \f$J\f$ the body-frame inertia
/// tensor, and \f$\otimes\f$ the JPL quaternion product.
///
/// The kinematic term uses **left** multiplication by the rate quaternion
/// \f$[0,\boldsymbol{\omega}_B]\f$ because \f$\boldsymbol{\omega}_B\f$ is expressed
/// in the Body frame — the body-referenced form \f$\dot{\bar q} =
/// \tfrac{1}{2}\,\Omega(\boldsymbol{\omega}_B)\,\bar q\f$ of Trawny & Roumeliotis
/// [trawny2005] Eq. (106). It is equivalent to \f$\dot A = -[\boldsymbol{\omega}_B
/// \times]\,A\f$ for the attitude matrix \f$A = A(\bar q_{B\leftarrow I})\f$
/// (Markley & Crassidis §3.1 [markley2014]). The rotational term is Euler's
/// equation. After each accepted integrator step the quaternion sub-vector is
/// renormalised, \f$\bar q \leftarrow \bar q / \lVert \bar q \rVert\f$, to hold it
/// on the unit manifold.
class RigidBody6Dof {
 public:
  static constexpr int kStateDim = 13;
  using State = Eigen::Matrix<double, kStateDim, 1>;

  /// @param inertia Body-frame inertia tensor J [kg·m^2], symmetric positive
  ///        definite (invertibility is the caller's responsibility).
  /// @param model  external acceleration/torque source (must outlive this).
  RigidBody6Dof(const Eigen::Matrix3d& inertia, const ForceTorqueModel& model)
      : inertia_(inertia), inertia_inv_(inertia.inverse()), model_(&model) {}

  /// Flatten a `TruthState` into the 13-element integration vector.
  static State pack(const state::TruthState& s) {
    State y;
    y.segment<3>(0) = s.position.eigen();
    y.segment<3>(3) = s.velocity.eigen();
    y.segment<4>(6) = s.attitude.core().coeffs();  // [q0,q1,q2,q3]
    y.segment<3>(10) = s.body_rate.eigen();
    return y;
  }

  /// Rebuild a `TruthState` at TAI @p epoch from an integration vector.
  static state::TruthState unpack(const State& y, const time::Tai& epoch) {
    state::TruthState s;
    s.epoch = epoch;
    s.position = math::Vec3<math::frames::ECI>(Eigen::Vector3d(y.segment<3>(0)));
    s.velocity = math::Vec3<math::frames::ECI>(Eigen::Vector3d(y.segment<3>(3)));
    s.attitude = math::Quat<math::frames::Body, math::frames::ECI>(
        math::Quaternion(Eigen::Vector4d(y.segment<4>(6))));
    s.body_rate = math::Vec3<math::frames::Body>(Eigen::Vector3d(y.segment<3>(10)));
    return s;
  }

  /// State derivative at TAI @p epoch. The model sees the full truth state so a
  /// later time-varying environment can use epoch/attitude; the static models in
  /// this push do not. ponytail: pass epoch + t once time-varying models land.
  State derivative(const time::Tai& epoch, const State& y) const {
    const state::TruthState s = unpack(y, epoch);
    const Eigen::Vector3d w = y.segment<3>(10);

    // Rotational: Euler's equation J w' = tau - w x (J w).
    const Eigen::Vector3d tau = model_->torque(s).eigen();
    const Eigen::Vector3d w_dot = inertia_inv_ * (tau - w.cross(inertia_ * w));

    // Attitude kinematics: q' = 1/2 [0, w] (x) q (JPL scalar-first). The rate is
    // Body-frame, so the incremental rotation multiplies on the LEFT — that gives
    // A' = -[w x] A (v_body' = -w x v_body), the correct body-referenced relation.
    // (Right multiply q (x) [0,w] would instead treat w as an ECI-frame rate.)
    const math::Quaternion q(Eigen::Vector4d(y.segment<4>(6)));
    const math::Quaternion w_quat(0.0, w.x(), w.y(), w.z());
    const math::Quaternion q_dot = w_quat * q;

    State dy;
    dy.segment<3>(0) = y.segment<3>(3);                  // dr/dt = v
    dy.segment<3>(3) = model_->acceleration(s).eigen();  // dv/dt = a_ext
    dy.segment<4>(6) = 0.5 * q_dot.coeffs();             // dq/dt
    dy.segment<3>(10) = w_dot;                           // dw/dt
    return dy;
  }

  /// Propagate @p s0 forward by @p dt seconds (dt >= 0) with adaptive RK8(9).
  state::TruthState propagate(const state::TruthState& s0, double dt,
                              const StepControl& ctl = {}) const {
    // Boundary guard (§3.6): this plant only propagates forward. A non-positive
    // dt is a no-op returning s0 unchanged — never a state advanced against a
    // rewound epoch (which would be an internally inconsistent TruthState).
    if (dt <= 0.0) {
      return s0;
    }
    // The derivative is evaluated at the RUNNING epoch, s0.epoch + t, not at the
    // fixed start epoch. Every environment model that actually varies with time
    // depends on this: the ephemeris resolvers move the Sun and Moon, the
    // Earth-rotation reduction turns the geomagnetic and atmospheric fields
    // underneath the vehicle, and eclipse geometry follows the Sun. Freezing the
    // epoch across a step would hold all of them still — over a single call
    // spanning an orbit that is a gross error, and even over a short step it
    // biases the result in a way that shrinks with step size and so hides as
    // "integration error" rather than showing up as a wrong model.
    const time::Tai epoch = s0.epoch;
    auto f = [this, &epoch](double t, const State& y) {
      return derivative(epoch + time::Duration::fromSecondsF(t), y);
    };
    const State y1 = integrate<kStateDim>(verner89(), f, 0.0, dt, pack(s0), ctl, QuatProjector{});
    return unpack(y1, epoch + time::Duration::fromSecondsF(dt));
  }

 private:
  /// Renormalizes the quaternion sub-vector after each accepted step so numeric
  /// drift off the unit manifold cannot accumulate.
  struct QuatProjector {
    void operator()(State& y) const {
      const double n = y.segment<4>(6).norm();
      if (n > 1.0e-12) {
        y.segment<4>(6) /= n;
      } else {
        // Degenerate (near-zero) quaternion: reset to identity rather than let a
        // meaningless zero rotation propagate. Unreachable in normal use.
        y.segment<4>(6) << 1.0, 0.0, 0.0, 0.0;
      }
    }
  };

  Eigen::Matrix3d inertia_;
  Eigen::Matrix3d inertia_inv_;
  const ForceTorqueModel* model_;
};

}  // namespace polaris::sim::dynamics

#endif  // POLARIS_SIM_DYNAMICS_RIGID_BODY_HPP
