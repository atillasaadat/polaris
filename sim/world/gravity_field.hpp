#ifndef POLARIS_SIM_WORLD_GRAVITY_FIELD_HPP
#define POLARIS_SIM_WORLD_GRAVITY_FIELD_HPP

/// @file
/// @brief Spherical-harmonic gravity + gravity-gradient torque (REQ-SIM-002).
///
/// `SphericalHarmonicGravity` is a `ForceTorqueModel`: it returns the geopotential
/// acceleration (ECI) and the gravity-gradient torque (Body). The acceleration is
/// the gradient of the harmonic potential
///   U = (GM/Re) sum_{n,m} (Re/r)^{n+1} P_{nm}(sin phi) [C_{nm} cos m.lambda + S_{nm} sin m.lambda]
/// evaluated with the Cunningham/Montenbruck-Gill V/W recursion (M&G §3.2.4), so
/// the same code serves point-mass, J2, or a full EGM2008 field once coefficients
/// are loaded. Degree and order are settable per run (truth is deliberately a
/// higher fidelity than the onboard model — `sim/CLAUDE.md`).
///
/// Frame caveat: the recursion is defined in the Earth-fixed frame, but the field
/// is evaluated here directly on the ECI position. That is exact for the **zonal**
/// (axisymmetric, m=0) field — which is longitude-independent and shares Earth's
/// pole/Z axis — and the embedded coefficient set is zonal (J2..J6). Tesseral
/// (m>0) terms additionally require the ECI->ECEF rotation (EOP), which lands with
/// EGM2008 file loading; the recursion already supports order>0 so only the frame
/// rotation is missing. ponytail: unnormalized V/W is numerically clean to ~degree
/// 40; switch to normalized Gottlieb when high-degree EGM2008 lands.
///
/// References:
///  - Montenbruck & Gill, *Satellite Orbits*, 2000, §3.2 (V/W recursion).
///    [montenbruck2000]
///  - Vallado, *Fundamentals of Astrodynamics and Applications*, 4th ed., 2013,
///    §8 (zonal coefficients, gravity-gradient torque). [vallado2013]

#include <Eigen/Core>
#include <vector>

#include "constants/constants.hpp"
#include "dynamics/force_torque.hpp"
#include "math/frames.hpp"
#include "math/typed_vector.hpp"
#include "state/truth_state.hpp"

namespace polaris::sim::world {

/// Unnormalized zonal harmonic coefficients J_n (Vallado 2013, Table; EGM/JGM).
/// C_{n,0} = -J_n. Only these are embedded until an EGM2008 file loader lands.
inline constexpr double kJ2 = 1.082'626'683'5e-3;
inline constexpr double kJ3 = -2.532'656'485'3e-6;
inline constexpr double kJ4 = -1.619'621'591'4e-6;
inline constexpr double kJ5 = -2.272'721'801'1e-7;
inline constexpr double kJ6 = 5.406'815'991'0e-7;

/// Dense unnormalized coefficient table: `C[n][m]`, `S[n][m]` for n in [0,nmax],
/// m in [0,n]. `C[0][0] = 1` is the point-mass term.
struct GravityCoeffs {
  int nmax = 0;
  std::vector<std::vector<double>> C;
  std::vector<std::vector<double>> S;

  /// Point mass only (`C[0][0] = 1`).
  static GravityCoeffs pointMass();

  /// Earth zonal field through degree 6 (J2..J6), tesserals zero.
  static GravityCoeffs earthZonal();
};

/// Spherical-harmonic gravity with the matching gravity-gradient torque.
class SphericalHarmonicGravity : public dynamics::ForceTorqueModel {
 public:
  /// @param coeffs  harmonic coefficients (owned).
  /// @param inertia Body inertia J [kg·m^2] — needed for the gravity-gradient
  ///        torque; pass the same tensor the plant integrates.
  /// @param degree  max harmonic degree n to evaluate (clamped to `coeffs.nmax`).
  /// @param order   max harmonic order m to evaluate (clamped to `degree`).
  /// @param mu      gravitational parameter [m^3/s^2].
  /// @param ref_radius reference radius Re [m] the coefficients are scaled to.
  SphericalHarmonicGravity(GravityCoeffs coeffs, const Eigen::Matrix3d& inertia, int degree,
                           int order, double mu = constants::wgs84::kGM,
                           double ref_radius = constants::wgs84::kSemiMajorAxis);

  math::Vec3<math::frames::ECI> acceleration(const state::TruthState& s) const override;

  /// Gravity-gradient torque tau = 3 (GM/r^3) c_hat x (J c_hat), c_hat the Body
  /// nadir unit vector (Wertz/Hughes). Uses the point-mass term (standard; the
  /// harmonic contribution to GG torque is negligible).
  math::Vec3<math::frames::Body> torque(const state::TruthState& s) const override;

  /// Geopotential U at ECI position [m^2/s^2]; the acceleration is grad U. Exposed
  /// for finite-difference validation of the recursion (all orders).
  double potential(const Eigen::Vector3d& r) const;

  int degree() const { return degree_; }

  int order() const { return order_; }

 private:
  /// Geopotential gradient (acceleration) via the M&G V/W recursion.
  Eigen::Vector3d gradient(const Eigen::Vector3d& r) const;

  GravityCoeffs coeffs_;
  Eigen::Matrix3d inertia_;
  int degree_;
  int order_;
  double mu_;
  double re_;

  static constexpr double kMinRadius_ = 1.0;  ///< [m] singular-radius guard
};

}  // namespace polaris::sim::world

#endif  // POLARIS_SIM_WORLD_GRAVITY_FIELD_HPP
