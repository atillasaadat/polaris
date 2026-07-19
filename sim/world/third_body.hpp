#ifndef POLARIS_SIM_WORLD_THIRD_BODY_HPP
#define POLARIS_SIM_WORLD_THIRD_BODY_HPP

/// @file
/// @brief Third-body point-mass gravitational perturbation (REQ-SIM-002, §5.2).
///
/// The perturbing acceleration on an Earth-orbiting satellite from one or more
/// distant point masses (Sun, Moon, planets). For a body at geocentric position
/// `s` and a satellite at geocentric `r` (both ECI/ICRF), the geocentric
/// perturbation is the direct attraction toward the body minus Earth's own
/// attraction toward it (Montenbruck & Gill Eq. 3.37; Vallado §8.6):
///
///   a = GM_b [ (s - r)/|s - r|^3  -  s/|s|^3 ].
///
/// Body positions come from an **injected resolver** (`PositionFn`) — the model
/// is decoupled from the ephemeris source, mirroring the `setEciToEcef` pattern
/// in `gravity_field.hpp`. In the truth sim that resolver wraps the DE440-fed
/// `ephemeris::EphemerisTable`; onboard would use the uploaded Chebyshev fits
/// (REQ-CDH-002). The resolver takes TDB (the ephemeris argument); this model
/// converts the state's TAI epoch via `toTdb(toTt(epoch))`.
///
/// A point mass exerts no net torque on the rigid body, so `torque()` is zero
/// (the third-body gravity-gradient torque is a separate, negligible term).
///
/// Sim-side (`sim/CLAUDE.md`): heap / std::function / virtual dispatch are fine.
///
/// References:
///  - Montenbruck & Gill, *Satellite Orbits*, 2000, §3.3 (third-body
///    perturbation). [montenbruck2000]
///  - Vallado, *Fundamentals of Astrodynamics and Applications*, 4th ed., §8.6.
///    [vallado2013]
///  - Park et al., "The JPL Planetary and Lunar Ephemerides DE440 and DE441",
///    AJ 161, 2021 (GM values, ephemeris source). [park2021]

#include <cstddef>
#include <functional>
#include <vector>

#include "dynamics/force_torque.hpp"
#include "math/frames.hpp"
#include "math/typed_vector.hpp"
#include "state/truth_state.hpp"
#include "time/timescales.hpp"

namespace polaris::sim::world {

/// Sum of third-body point-mass perturbations from injected body ephemerides.
class ThirdBodyGravity : public dynamics::ForceTorqueModel {
 public:
  /// Geocentric ECI position [m] of a body at a TDB epoch. Returns false —
  /// leaving @p out untouched — if the epoch is outside the ephemeris coverage.
  using PositionFn = std::function<bool(const time::Tdb&, math::Vec3<math::frames::ECI>& out)>;

  /// Add one perturbing body: its gravitational parameter @p gm [m^3/s^2] and a
  /// position resolver. A body whose resolver has no coverage at an epoch is
  /// simply skipped for that epoch.
  void addBody(double gm, PositionFn position);

  /// Number of perturbing bodies registered.
  std::size_t size() const { return bodies_.size(); }

  math::Vec3<math::frames::ECI> acceleration(const state::TruthState& s) const override;

  /// Point masses exert no net torque (§ file header).
  math::Vec3<math::frames::Body> torque(const state::TruthState&) const override {
    return math::Vec3<math::frames::Body>::Zero();
  }

 private:
  struct Body {
    double gm;
    PositionFn position;
  };

  std::vector<Body> bodies_;

  static constexpr double kMinRadius_ = 1.0;  ///< [m] singular-separation guard
};

}  // namespace polaris::sim::world

#endif  // POLARIS_SIM_WORLD_THIRD_BODY_HPP
