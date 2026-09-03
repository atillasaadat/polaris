/// @file Truth-side propagation of the *other* satellites (§2.3, §8.4.1).
///
/// A pointing row that aims a camera at `SAT_TLE_n` or `SAT_STATE_n` has a
/// problem no other row has: the thing being pointed at **does not exist in the
/// plant**. The vehicle is simulated; the target is a catalogue entry the
/// flight software propagates onboard, and nothing outside the flight software
/// has an opinion about where it is.
///
/// That is fine for an assertion — a test can compute its own reference — and
/// useless for a viewer, which needs something to draw. Worse, drawing the
/// *onboard* position would make the visualization circular in the same way an
/// assertion computed from `solveGuidanceAttitude` is circular: the camera
/// would appear to track the target perfectly no matter how wrong the onboard
/// propagation was, because both halves of the picture came from it.
///
/// So the truth side propagates the targets itself, independently, and better:
///
///  - **TLE** → SGP4 and `frames::eciFromTeme`. Closed form at any epoch. This
///    is the same lineage the flight side runs, and it is the honest limit of
///    what a truth side can offer for a TLE: SGP4 *is* the definition of where
///    a TLE says an object is, so an independent implementation would be
///    verifying the implementation rather than the position. That verification
///    is Push 81's three-way golden test and does not belong here.
///  - **State vector** → RK4 over the sim's own `SphericalHarmonicGravity` at
///    the scenario's degree, which is **not** the onboard model. The flight
///    side runs the onboard 8x8 truncation (`lib/gnc/target_propagator`); this runs
///    the full field the vehicle itself flies through. The difference is real
///    and it is the point: it is measured at ~13 m over 900 s for an 8 000 km
///    target and ~1.1 km over the propagator's 12 h horizon for a 500 km one,
///    so a viewer showing both the truth position and where the camera is
///    aimed shows the onboard model error rather than hiding it.
///
/// Truth-only, output-only: nothing here feeds the plant, the flight software,
/// or any assertion the flight software could influence. It exists to be drawn
/// and to be compared against.

#ifndef POLARIS_SIM_WORLD_TRACKED_OBJECT_HPP
#define POLARIS_SIM_WORLD_TRACKED_OBJECT_HPP

#include <Eigen/Dense>
#include <memory>
#include <string>
#include <vector>

#include "frames/eop.hpp"
#include "gnc/sgp4.hpp"
#include "gnc/tle.hpp"
#include "math/typed_vector.hpp"
#include "time/leap_seconds.hpp"
#include "time/timescales.hpp"
#include "world/gravity_field.hpp"

namespace polaris::sim::world {

/// One secondary object the truth side knows the position of.
///
/// Constructed through the two factories; a default-constructed instance is
/// deliberately unusable (`valid()` false) rather than silently at the origin,
/// because an object drawn at the centre of the Earth is a picture of a bug
/// that looks like a picture of a scene.
class TrackedObject {
 public:
  enum class Kind { kTle, kStateVector };

  TrackedObject() = default;

  /// A catalogue object from its two-line element set.
  ///
  /// Returns an invalid object when the lines do not parse or SGP4 cannot be
  /// initialised from them — the same refusals the flight catalogue applies,
  /// for the same reason: a target that cannot be propagated must not be
  /// answered with a position.
  static TrackedObject fromTle(std::string name, const std::string& line1, const std::string& line2,
                               const time::LeapSecondTable& leap);

  /// An osculating ECI state, propagated over @p gravity.
  ///
  /// @p gravity is held by pointer and must outlive this object; it is the
  /// scenario's own field, so the target flies the same gravity the vehicle
  /// does. Passing nullptr yields an invalid object rather than a two-body
  /// fallback: silently dropping to a weaker model is how a truth side stops
  /// being a truth side.
  static TrackedObject fromState(std::string name, const time::Tai& epoch,
                                 const Eigen::Vector3d& position_m,
                                 const Eigen::Vector3d& velocity_m_s,
                                 const SphericalHarmonicGravity* gravity);

  bool valid() const { return valid_; }

  const std::string& name() const { return name_; }

  Kind kind() const { return kind_; }

  /// Position in ECI at @p t [m]. Returns false when the object is invalid or
  /// the propagation refuses the epoch (a decayed TLE past decay, say).
  ///
  /// For a state-vector object the integration is **incremental**: a forward
  /// request continues from the last answer, and a request at or before the
  /// last epoch re-seeds from the upload. That makes a sequential sim run cheap
  /// and a random-access query correct, which is the only combination worth
  /// having.
  bool positionAt(const time::Tai& t, math::Vec3<math::frames::ECI>& position_m) const;

 private:
  /// Fixed-step RK4 over the gravity field. 10 s matches the onboard
  /// propagator's step so a difference between the two is the *model*, not the
  /// integrator; the field is what changes.
  static constexpr double kStepSec = 10.0;

  void stepTo(const time::Tai& t) const;

  /// Position and velocity together — the two halves of one state, which is
  /// why they are returned as one thing rather than one returned and one
  /// written through an out-parameter (C++ Core Guidelines F.20/F.21).
  struct Step {
    Eigen::Vector3d position_m;
    Eigen::Vector3d velocity_m_s;
  };

  /// One RK4 step of @p h from @p from at @p t0. Shared by the grid walk and
  /// the unretained tail step so the two cannot drift apart.
  Step rk4(const time::Tai& t0, double h, const Step& from) const;

  std::string name_;
  Kind kind_ = Kind::kStateVector;
  bool valid_ = false;

  // TLE branch.
  gnc::Sgp4 sgp4_;
  time::Tai tle_epoch_;

  // State-vector branch. Mutable because propagation is a cache: `positionAt`
  // is logically const and physically advances the integration.
  const SphericalHarmonicGravity* gravity_ = nullptr;
  time::Tai seed_epoch_;
  Eigen::Vector3d seed_position_m_ = Eigen::Vector3d::Zero();
  Eigen::Vector3d seed_velocity_m_s_ = Eigen::Vector3d::Zero();
  mutable bool cursor_valid_ = false;
  mutable int cursor_step_ = 0;
  mutable time::Tai cursor_;
  mutable Eigen::Vector3d position_m_ = Eigen::Vector3d::Zero();
  mutable Eigen::Vector3d velocity_m_s_ = Eigen::Vector3d::Zero();
};

/// The objects a scenario is tracking, in the order they were added.
using TrackedObjectSet = std::vector<TrackedObject>;

}  // namespace polaris::sim::world

#endif  // POLARIS_SIM_WORLD_TRACKED_OBJECT_HPP
