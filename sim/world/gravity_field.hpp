#ifndef POLARIS_SIM_WORLD_GRAVITY_FIELD_HPP
#define POLARIS_SIM_WORLD_GRAVITY_FIELD_HPP

/// @file
/// @brief Spherical-harmonic gravity (REQ-SIM-002).
///
/// `SphericalHarmonicGravity` is a `ForceTorqueModel` returning the geopotential
/// acceleration (ECI); it is torque-free, because the gravity-gradient couple is
/// its own §5.3 provider (`world/gravity_gradient.hpp`). The field uses
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
/// Frame: the recursion is defined in the Earth-fixed frame, so an ECI->ECEF
/// resolver must be installed (`setEciToEcef`) for the field to be physically
/// correct — at EVERY order, including a purely zonal one.
///
/// An earlier version applied the rotation only for order > 0, on the reasoning
/// that a zonal (m=0) field is axisymmetric and therefore frame-invariant. That
/// is half true and the wrong half was assumed: a zonal field is invariant under
/// rotation ABOUT ITS OWN AXIS — Earth's diurnal spin, harmlessly — but ECI is
/// J2000 mean equator while ECEF follows the TRUE pole, and precession plus
/// nutation separate the two by ~0.37 deg by 2026. Evaluating zonally in ECI
/// therefore tilts the J2 bulge by that angle, worth ~100 m per revolution in
/// LEO. Because a mis-axed field is still perfectly conservative, it conserves
/// energy and angular momentum exactly and no self-consistency test can see it;
/// cross-validation against GMAT is what exposed it (design doc §23.1).
///
/// Without a resolver the evaluation falls back to ECI, which is an
/// approximation at every order — wrong in longitude for tesserals and wrong in
/// pole orientation for zonals. It is a deliberate low-fidelity/test mode, not
/// an exact path.
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
#include <functional>
#include <utility>
#include <vector>

#include "constants/constants.hpp"
#include "dynamics/force_torque.hpp"
#include "math/frames.hpp"
#include "math/quaternion.hpp"
#include "math/typed_vector.hpp"
#include "state/truth_state.hpp"
#include "time/timescales.hpp"

namespace polaris::sim::world {

/// Unnormalized physical zonal harmonic coefficients J_n (Vallado 2013, Table;
/// EGM/JGM). The unnormalized zonal C_{n,0} = -J_n; `earthZonal()` converts these
/// to the FULLY-NORMALIZED Cbar_{n,0} = -J_n / sqrt(2n+1) actually stored. Only
/// these are embedded until an EGM2008 file loader lands.
///
/// They are **aliases**, not copies: the values live in the shared constants
/// registry (`constants::gravity`) because the onboard orbit filter's coarse
/// force model (§8.3) needs J2 too, and a geopotential coefficient written twice
/// is a number that can drift between the truth model and the flight model with
/// nothing to catch it.
inline constexpr double kJ2 = constants::gravity::kJ2;
inline constexpr double kJ3 = constants::gravity::kJ3;
inline constexpr double kJ4 = constants::gravity::kJ4;
inline constexpr double kJ5 = constants::gravity::kJ5;
inline constexpr double kJ6 = constants::gravity::kJ6;

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

/// Spherical-harmonic gravity.
///
/// **Model.** The geopotential is the fully-normalized spherical-harmonic sum
/// (\f$\sin\phi = z/r\f$, \f$\lambda = \operatorname{atan2}(y,x)\f$, \f$\bar
/// C_{00}=1\f$ supplying the leading \f$GM/r\f$):
/// \f[
///   U(\mathbf r) = \frac{GM}{r}\sum_{n=0}^{N}\left(\frac{R_e}{r}\right)^{\!n}
///     \sum_{m=0}^{n}\bar P_{nm}(\sin\phi)
///       \left[\bar C_{nm}\cos m\lambda + \bar S_{nm}\sin m\lambda\right],
///   \qquad \mathbf a = \nabla U,
/// \f]
/// with \f$\bar P_{nm}\f$ the fully-normalized associated Legendre functions and
/// \f$\bar C_{nm},\bar S_{nm}\f$ the overbarred coefficients EGM models ship. Two
/// independent engines evaluate it so \f$\mathbf a = \nabla U\f$ cross-checks by
/// finite difference: `potential()` sums \f$U\f$ with the Holmes-Featherstone
/// forward-column ALF recursion [holmes2002], while `gradient()` assembles
/// \f$\mathbf a\f$ directly in Cartesian form via the singularity-free normalized
/// Gottlieb recursion [eckman2016; gottlieb1993] (the sectoral \f$\cos^m\phi\f$
/// factor is carried by a direction-cosine recursion, so no \f$1/\cos\phi\f$ ever
/// appears and the field is stable to high degree and at the poles). The
/// recursion is defined in the Earth-fixed frame; see @ref setEciToEcef for the
/// ECI\f$\to\f$ECEF reduction it requires at every order, zonal included.
class SphericalHarmonicGravity : public dynamics::ForceTorqueModel {
 public:
  /// @param coeffs  harmonic coefficients (owned).
  /// @param degree  max harmonic degree n to evaluate (clamped to `coeffs.nmax`).
  /// @param order   max harmonic order m to evaluate (clamped to `degree`).
  /// @param mu      gravitational parameter [m^3/s^2].
  /// @param ref_radius reference radius Re [m] the coefficients are scaled to.
  SphericalHarmonicGravity(GravityCoeffs coeffs, int degree, int order,
                           double mu = constants::wgs84::kGM,
                           double ref_radius = constants::wgs84::kSemiMajorAxis);

  math::Vec3<math::frames::ECI> acceleration(const state::TruthState& s) const override;

  /// Torque-free. The gravity-gradient couple is its own provider
  /// (`world/gravity_gradient.hpp`, design doc §5.3) rather than a side effect of
  /// this one: it needs the point-mass term only, it applies equally to a
  /// point-mass or free-drift scenario where no harmonic field is built, and it
  /// carries its own enable switch. Returning it from here as well would
  /// double-count it in the composite whenever both are wired.
  math::Vec3<math::frames::Body> torque(const state::TruthState&) const override {
    return math::Vec3<math::frames::Body>::Zero();
  }

  /// Geopotential U at ECI position [m^2/s^2]; the acceleration is grad U. Exposed
  /// for finite-difference validation of the recursion (all orders).
  double potential(const Eigen::Vector3d& r) const;

  int degree() const { return degree_; }

  int order() const { return order_; }

  /// Resolver for the ECI->ECEF rotation at a TAI epoch, used to evaluate tesseral
  /// (m>0) terms in the Earth-fixed frame where they are physically defined.
  using EciToEcefFn =
      std::function<bool(const time::Tai&, math::Quat<math::frames::ECEF, math::frames::ECI>&)>;

  /// Install the ECI->ECEF resolver (typically wrapping `frames::ecefFromEci` over
  /// an EOP table). Required at every order, zonal included — see the file header
  /// for why a zonal field is NOT frame-invariant under this transform. Without a
  /// resolver the field is evaluated on the ECI position, which mis-orients the
  /// pole for zonals and is additionally wrong in longitude for tesserals.
  /// Sim-side wiring, e.g.:
  ///   g.setEciToEcef([&](const time::Tai& t,
  ///                      math::Quat<math::frames::ECEF, math::frames::ECI>& q) {
  ///     return frames::ecefFromEci(t, eop_table, leap, q);
  ///   });
  void setEciToEcef(EciToEcefFn fn) { eci_to_ecef_ = std::move(fn); }

  /// Gravitational parameter the field was built with [m^3/s^2]. Must be the
  /// value the COEFFICIENTS were solved with, not a generic WGS84 constant.
  double mu() const { return mu_; }

  /// Reference radius the coefficients are scaled to [m]. Same caveat as mu().
  double referenceRadius() const { return re_; }

 private:
  /// Geopotential gradient (acceleration) via the normalized Gottlieb recursion
  /// (NASA/TP-2016-218604 App. C.9). Independent of `potential()`.
  Eigen::Vector3d gradient(const Eigen::Vector3d& r) const;

  /// Precompute the degree/order-only normalization ratios used by `gradient()`
  /// so the per-call recursion holds no sqrt() (they depend on n,m alone).
  void buildNormTables();

  GravityCoeffs coeffs_;
  int degree_;
  int order_;
  double mu_;
  double re_;

  // Gottlieb normalization ratio tables, indexed by degree n (and order m),
  // sized [degree_+2]. See NASA/TP-2016-218604 App. C.9. [eckman2016]
  std::vector<double> norm1_, norm2_, norm11_, normn10_;
  std::vector<std::vector<double>> norm1m_, norm2m_, normn1_;

  EciToEcefFn eci_to_ecef_;  ///< optional; see setEciToEcef()

  static constexpr double kMinRadius_ = 1.0;  ///< [m] singular-radius guard
};

}  // namespace polaris::sim::world

#endif  // POLARIS_SIM_WORLD_GRAVITY_FIELD_HPP
