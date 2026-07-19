/// @file
/// @brief Third-body point-mass perturbation. See third_body.hpp.

#include "world/third_body.hpp"

#include <Eigen/Core>
#include <utility>

#include "time/tdb.hpp"

namespace polaris::sim::world {

void ThirdBodyGravity::addBody(double gm, PositionFn position) {
  bodies_.push_back(Body{gm, std::move(position)});
}

math::Vec3<math::frames::ECI> ThirdBodyGravity::acceleration(const state::TruthState& s) const {
  const Eigen::Vector3d r = s.position.eigen();
  // Ephemerides are argued in TDB; convert the state's TAI epoch once.
  const time::Tdb tdb = time::toTdb(time::toTt(s.epoch));

  Eigen::Vector3d a = Eigen::Vector3d::Zero();
  for (const Body& b : bodies_) {
    math::Vec3<math::frames::ECI> body_pos;
    if (!b.position(tdb, body_pos)) {
      continue;  // body ephemeris has no coverage at this epoch -> skip
    }
    const Eigen::Vector3d sb = body_pos.eigen();  // body, geocentric ECI
    const Eigen::Vector3d d = sb - r;             // satellite -> body
    const double sb_n = sb.norm();
    const double d_n = d.norm();
    if (sb_n < kMinRadius_ || d_n < kMinRadius_) {
      continue;  // degenerate separation -> no finite contribution (§3.6)
    }
    // a = GM_b [ (s - r)/|s - r|^3 - s/|s|^3 ]  (Montenbruck & Gill Eq. 3.37).
    // ponytail: the direct subtraction loses ~log10(|s|/|r|) ~ 4 digits to
    // cancellation for the Sun (~11 remain, far above the truth error budget).
    // Swap in Battin's f(q) formulation if third-body precision ever dominates.
    a += b.gm * (d / (d_n * d_n * d_n) - sb / (sb_n * sb_n * sb_n));
  }
  return math::Vec3<math::frames::ECI>(a);
}

}  // namespace polaris::sim::world
