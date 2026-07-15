#ifndef POLARIS_SIM_WORLD_GRAVITY_FIELD_HPP
#define POLARIS_SIM_WORLD_GRAVITY_FIELD_HPP

/// @file
/// @brief Spherical-harmonic gravity + gravity-gradient torque (REQ-SIM-002).
///
/// `SphericalHarmonicGravity` is a `ForceTorqueModel`: it returns the geopotential
/// acceleration (ECI) and the gravity-gradient torque (Body). The field uses
/// FULLY-NORMALIZED coefficients Cbar_nm/Sbar_nm and the singularity-free
/// **normalized Gottlieb** recursion, ported from NASA/TP-2016-218604 Appendix
/// C.9 (`gottliebnorm.m`). This is numerically stable to high degree/order
/// (verified at 200x200, incl. the poles), unlike the old unnormalized
/// Cunningham V/W whose sectoral term grows like (2n-1)!! and overflows past
/// ~degree 40. Two INDEPENDENT engines cross-validate one another:
///   - `potential()`   — normalized-ALF sum of the geopotential (Eq. 1.20),
///                       via the Holmes-Featherstone forward-column recursion.
///   - `gradient()`    — Gottlieb's direct Cartesian acceleration assembly.
/// So `a = grad U` checks the two against each other by finite difference.
/// The same code serves point-mass, J2, or a full EGM2008 field once
/// coefficients are loaded. Degree and order are settable per run (truth is
/// deliberately a higher fidelity than the onboard model — `sim/CLAUDE.md`).
///
/// Potential (Eq. 1.20): with sin(phi)=z/r, lambda=atan2(y,x),
///   U = (GM/r) sum_{n>=0} (Re/r)^n sum_{m=0..n} Pbar_nm(sin phi)
///         [Cbar_nm cos(m.lambda) + Sbar_nm sin(m.lambda)],
/// and Cbar_00 = 1 supplies the leading GM/r; a = grad U.
///
/// Frame caveat: the recursion is defined in the Earth-fixed frame, but the field
/// is evaluated here directly on the ECI position (rnp = I). That is exact for the
/// **zonal** (axisymmetric, m=0) field — which is longitude-independent and shares
/// Earth's pole/Z axis — and the embedded coefficient set is zonal (J2..J6).
/// Tesseral (m>0) terms additionally require the ECI->ECEF rotation (EOP), which
/// lands with EGM2008 file loading; the recursion already supports arbitrary
/// order>0 so only the frame rotation is missing.
///
/// References:
///  - Eckman, Brown & Adamo, *Normalization and Implementation of Three
///    Gravitational Acceleration Models*, NASA/TP-2016-218604, 2016, Ch. 3 +
///    Appendix C.9 (normalized Gottlieb). [eckman2016]
///  - Gottlieb, *Fast Gravity...*, NASA CR-188243, 1993. [gottlieb1993]
///  - Holmes & Featherstone, *A unified approach...*, J. Geodesy 76, 2002
///    (normalized ALF forward-column recursion for `potential()`). [holmes2002]
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

/// Unnormalized physical zonal harmonic coefficients J_n (Vallado 2013, Table;
/// EGM/JGM). The unnormalized zonal C_{n,0} = -J_n; `earthZonal()` converts these
/// to the FULLY-NORMALIZED Cbar_{n,0} = -J_n / sqrt(2n+1) actually stored. Only
/// these are embedded until an EGM2008 file loader lands.
inline constexpr double kJ2 = 1.082'626'683'5e-3;
inline constexpr double kJ3 = -2.532'656'485'3e-6;
inline constexpr double kJ4 = -1.619'621'591'4e-6;
inline constexpr double kJ5 = -2.272'721'801'1e-7;
inline constexpr double kJ6 = 5.406'815'991'0e-7;

/// Dense FULLY-NORMALIZED coefficient table: `C[n][m]` = Cbar_{n,m}, `S[n][m]` =
/// Sbar_{n,m} for n in [0,nmax], m in [0,n]. `C[0][0] = 1` is the point-mass term.
/// These are the overbarred coefficients EGM models ship; the recursion in
/// `gradient()`/`potential()` expects them normalized (see @file).
struct GravityCoeffs {
  int nmax = 0;
  std::vector<std::vector<double>> C;
  std::vector<std::vector<double>> S;

  /// Point mass only (`C[0][0] = Cbar_00 = 1`).
  static GravityCoeffs pointMass();

  /// Earth zonal field through degree 6 (J2..J6), tesserals zero. Stores the
  /// normalized Cbar_{n,0} = -J_n / sqrt(2n+1).
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
  /// Geopotential gradient (acceleration) via the normalized Gottlieb recursion
  /// (NASA/TP-2016-218604 App. C.9). Independent of `potential()`.
  Eigen::Vector3d gradient(const Eigen::Vector3d& r) const;

  /// Precompute the degree/order-only normalization ratios used by `gradient()`
  /// so the per-call recursion holds no sqrt() (they depend on n,m alone).
  void buildNormTables();

  GravityCoeffs coeffs_;
  Eigen::Matrix3d inertia_;
  int degree_;
  int order_;
  double mu_;
  double re_;

  // Gottlieb normalization ratio tables, indexed by degree n (and order m),
  // sized [degree_+2]. See NASA/TP-2016-218604 App. C.9. [eckman2016]
  std::vector<double> norm1_, norm2_, norm11_, normn10_;
  std::vector<std::vector<double>> norm1m_, norm2m_, normn1_;

  static constexpr double kMinRadius_ = 1.0;  ///< [m] singular-radius guard
};

}  // namespace polaris::sim::world

#endif  // POLARIS_SIM_WORLD_GRAVITY_FIELD_HPP
