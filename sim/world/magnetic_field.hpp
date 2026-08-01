#ifndef POLARIS_SIM_WORLD_MAGNETIC_FIELD_HPP
#define POLARIS_SIM_WORLD_MAGNETIC_FIELD_HPP

/// @file
/// @brief IGRF field in ECI, and the residual-dipole disturbance torque
/// (design doc §5.2, §5.3; REQ-SIM-002).
///
/// Two things live here, in the same relationship `atmosphere.hpp` has with
/// `drag.hpp`: an *environment* (where is the field, in the frame the plant
/// works in) and a *disturbance* that consumes it.
///
///  - `EarthMagneticField` wraps the flight-side `environment::IgrfField` and
///    does the two conversions the plant needs and flight does not: TAI to the
///    decimal year IGRF is parameterised by, and ECEF to ECI. The Earth-rotation
///    reduction is mandatory here, unlike geodetic altitude in `atmosphere.hpp`
///    which is rotation-invariant about the spin axis — the field is strongly
///    longitude-dependent, so evaluating it on an unrotated ECI position would
///    smear the whole field pattern around the Earth once per day.
///
///  - `ResidualDipoleTorque` is the disturbance every real spacecraft carries:
///    leftover magnetisation in the structure and current loops in the harness
///    give the body a residual magnetic moment m, and the ambient field exerts
///    tau = m x B on it. It is small (typically micro-N·m) but secular rather
///    than oscillatory, so it dominates the long-term momentum budget in LEO and
///    sets the magnetorquer sizing.
///
/// The field is exposed as a `MagneticFieldFn` for the same reason every other
/// environment model here takes a resolver: models compose and test without any
/// external data on disk. Hand a fake field to `ResidualDipoleTorque` and its
/// torque is exactly checkable by hand.
///
/// Sim-side, so heap, exceptions-free error returns, and `std::function` are all
/// fine (`sim/CLAUDE.md`).

#include <functional>

#include "dynamics/force_torque.hpp"
#include "environment/igrf.hpp"
#include "frames/eci_ecef.hpp"
#include "math/frames.hpp"
#include "math/typed_vector.hpp"
#include "time/leap_seconds.hpp"
#include "time/timescales.hpp"

namespace polaris::sim::world {

/// Resolver for the ambient magnetic field: given an epoch and an ECI position,
/// write the field in ECI [T] and return true. False means "no field available"
/// — out of model coverage, or a missing dependency — and leaves @p out alone.
using MagneticFieldFn = std::function<bool(const time::Tai&, const math::Vec3<math::frames::ECI>&,
                                           math::Vec3<math::frames::ECI>&)>;

/// Resolver for the ECI->ECEF rotation at a TAI epoch. Same contract the gravity
/// field and NRLMSIS use, so one wiring serves all three.
using EciToEcefFn =
    std::function<bool(const time::Tai&, math::Quat<math::frames::ECEF, math::frames::ECI>&)>;

/// IGRF-14 evaluated in ECI at a TAI epoch.
///
/// **Model.** Wraps `environment::IgrfField` (the Schmidt semi-normalized IGRF
/// expansion; see `lib/environment/igrf.hpp`) and applies the two conversions the
/// plant needs: TAI to the decimal year the field is parameterised by, and the
/// Earth-fixed field to ECI. The IGRF harmonics are defined in ECEF, so the
/// position is rotated in and the field vector rotated back with the same
/// rotation \f$R \equiv R_{\mathrm{ECEF}\leftarrow\mathrm{ECI}}\f$:
/// \f[
///   \mathbf r_{\mathrm{ecef}} = R\,\mathbf r_{\mathrm{eci}},
///   \qquad
///   \mathbf B_{\mathrm{eci}} = R^{-1}\,\mathbf B_{\mathrm{ecef}}
///     = R^{\mathsf T}\,\mathbf B_{\mathrm{ecef}}
/// \f]
/// (a pure rotation, no translation — the frames share an origin, and \f$\mathbf B\f$
/// is a free vector). The rotation is mandatory here because the field is strongly
/// longitude-dependent; evaluating it on an unrotated ECI position would smear the
/// field pattern around the Earth once per day.
///
/// Both dependencies are injected rather than constructed: the Earth-rotation
/// resolver because it needs EOP, and the leap-second table because TAI->UTC is
/// not computable without one. Missing either yields no field rather than a
/// guessed one, so a half-wired model produces an obviously-zero torque instead
/// of a plausibly-wrong one.
class EarthMagneticField {
 public:
  explicit EarthMagneticField(const environment::IgrfCoefficients& coefficients)
      : field_(coefficients) {}

  /// Supply the ECI->ECEF rotation. Required.
  void setEciToEcef(EciToEcefFn fn) { eci_to_ecef_ = std::move(fn); }

  /// Supply the leap-second table used for TAI->UTC. Required; must outlive this.
  void setLeapSeconds(const time::LeapSecondTable* table) { leap_ = table; }

  /// False if the coefficients were rejected or a dependency is unset.
  bool good() const { return field_.good() && static_cast<bool>(eci_to_ecef_) && leap_ != nullptr; }

  const environment::IgrfField& igrf() const { return field_; }

  /// Field in ECI [T] at @p epoch and ECI position @p r_eci.
  /// @return false, leaving @p out untouched, if @ref good() is false or the
  ///         Earth-rotation resolver declines the epoch.
  bool field(const time::Tai& epoch, const math::Vec3<math::frames::ECI>& r_eci,
             math::Vec3<math::frames::ECI>& out) const;

  /// Adapt to the resolver contract. The returned callable holds a pointer to
  /// this object, which must outlive it.
  MagneticFieldFn fieldFn() const;

 private:
  environment::IgrfField field_;
  EciToEcefFn eci_to_ecef_;
  const time::LeapSecondTable* leap_{nullptr};
};

/// Convert a TAI epoch to the decimal year IGRF is parameterised by.
///
/// Sim-side alias for `time::decimalYear` (lib/time/utc.hpp), where the
/// conversion now lives so the FSW's onboard IGRF reference (§8.1) shares it.
/// Kept because it is worth testing directly: the year fraction must account for
/// leap years, and getting it wrong is a silent sub-year epoch offset that shows
/// up only as a small secular-variation bias.
///
/// @return false if @p epoch cannot be converted with @p leap.
bool decimalYear(const time::Tai& epoch, const time::LeapSecondTable& leap, double& out);

/// Torque from the spacecraft's residual magnetic dipole.
///
/// **Model.** A magnetic moment \f$\mathbf m\f$ in an ambient field \f$\mathbf B\f$
/// feels the torque
/// \f[
///   \boldsymbol\tau = \mathbf m \times \mathbf B,
/// \f]
/// evaluated in the Body frame — the field is rotated from ECI into Body (via the
/// attitude) and crossed with the Body-fixed dipole, since \f$\mathbf m\f$ is a
/// property of the structure and rotates with the vehicle.
///
/// Contributes no acceleration — a dipole in a uniform field feels a torque but
/// no net force. (The force from the field *gradient* is smaller by the ratio of
/// the vehicle size to Earth's radius and is not modelled.)
///
/// The dipole is expressed in the Body frame and is constant there, which is the
/// physically right choice: residual magnetisation is a property of the
/// structure, so it rotates with the vehicle.
class ResidualDipoleTorque : public dynamics::ForceTorqueModel {
 public:
  /// @param dipole_body Residual magnetic moment in the Body frame [A·m^2].
  /// @param field       Ambient-field resolver; if empty or declining, the torque
  ///                    is zero.
  ResidualDipoleTorque(const math::Vec3<math::frames::Body>& dipole_body, MagneticFieldFn field)
      : dipole_body_(dipole_body), field_(std::move(field)) {}

  math::Vec3<math::frames::ECI> acceleration(const state::TruthState&) const override {
    return math::Vec3<math::frames::ECI>::Zero();
  }

  math::Vec3<math::frames::Body> torque(const state::TruthState& s) const override;

  const math::Vec3<math::frames::Body>& dipole() const { return dipole_body_; }

 private:
  math::Vec3<math::frames::Body> dipole_body_;
  MagneticFieldFn field_;
};

}  // namespace polaris::sim::world

#endif  // POLARIS_SIM_WORLD_MAGNETIC_FIELD_HPP
