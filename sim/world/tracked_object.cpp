#include "world/tracked_object.hpp"

#include <cmath>
#include <utility>

#include "frames/teme_eci.hpp"

namespace polaris::sim::world {

namespace pm = polaris::math;
namespace pt = polaris::time;

namespace {

/// Seconds between two TAI instants, as a double. The spans here are hours at
/// most, so the nanosecond integer converts without losing anything that
/// matters against a metre-level truth.
double secondsBetween(const pt::Tai& a, const pt::Tai& b) {
  return static_cast<double>(a.nanosecondsSinceEpoch() - b.nanosecondsSinceEpoch()) * 1.0e-9;
}

pt::Tai advanced(const pt::Tai& t, double seconds) {
  return pt::Tai::fromNanosecondsSinceEpoch(t.nanosecondsSinceEpoch() +
                                            static_cast<std::int64_t>(seconds * 1.0e9));
}

}  // namespace

TrackedObject TrackedObject::fromTle(std::string name, const std::string& line1,
                                     const std::string& line2, const pt::LeapSecondTable& leap) {
  TrackedObject o;
  o.name_ = std::move(name);
  o.kind_ = Kind::kTle;

  gnc::TleElements elements;
  if (gnc::parseTle(line1, line2, elements) != gnc::TleStatus::kOk) {
    return o;  // invalid; see the header on refusing rather than answering
  }
  if (!elements.epochTai(leap, o.tle_epoch_)) {
    return o;
  }
  if (o.sgp4_.initialise(elements) != gnc::Sgp4Status::kOk) {
    return o;
  }
  o.valid_ = true;
  return o;
}

TrackedObject TrackedObject::fromState(std::string name, const pt::Tai& epoch,
                                       const Eigen::Vector3d& position_m,
                                       const Eigen::Vector3d& velocity_m_s,
                                       const SphericalHarmonicGravity* gravity) {
  TrackedObject o;
  o.name_ = std::move(name);
  o.kind_ = Kind::kStateVector;
  if (gravity == nullptr || !position_m.allFinite() || !velocity_m_s.allFinite() ||
      position_m.norm() <= 0.0) {
    return o;
  }
  o.gravity_ = gravity;
  o.seed_epoch_ = epoch;
  o.seed_position_m_ = position_m;
  o.seed_velocity_m_s_ = velocity_m_s;
  o.cursor_ = epoch;
  o.position_m_ = position_m;
  o.velocity_m_s_ = velocity_m_s;
  o.valid_ = true;
  return o;
}

void TrackedObject::stepTo(const pt::Tai& t) const {
  // A request at or before where the integration stands re-seeds, so a
  // backwards or repeated query is correct rather than merely cheap. Forward
  // requests continue, which is what a sim run does for every sample after the
  // first.
  if (secondsBetween(t, cursor_) <= 0.0) {
    cursor_ = seed_epoch_;
    position_m_ = seed_position_m_;
    velocity_m_s_ = seed_velocity_m_s_;
    if (secondsBetween(t, cursor_) <= 0.0) {
      return;
    }
  }

  // The gravity model answers on a TruthState; only epoch and position are read
  // for an acceleration, so the attitude and rate fields stay at their defaults.
  const auto accel = [this](const pt::Tai& at, const Eigen::Vector3d& r) {
    state::TruthState probe;
    probe.epoch = at;
    probe.position = pm::Vec3<pm::frames::ECI>(r);
    return gravity_->acceleration(probe).eigen();
  };

  double remaining = secondsBetween(t, cursor_);
  while (remaining > 0.0) {
    const double h = remaining < kStepSec ? remaining : kStepSec;
    const Eigen::Vector3d& r = position_m_;
    const Eigen::Vector3d& v = velocity_m_s_;
    const pt::Tai t0 = cursor_;
    const pt::Tai th = advanced(t0, 0.5 * h);
    const pt::Tai t1 = advanced(t0, h);

    const Eigen::Vector3d k1r = v;
    const Eigen::Vector3d k1v = accel(t0, r);
    const Eigen::Vector3d k2r = v + 0.5 * h * k1v;
    const Eigen::Vector3d k2v = accel(th, r + 0.5 * h * k1r);
    const Eigen::Vector3d k3r = v + 0.5 * h * k2v;
    const Eigen::Vector3d k3v = accel(th, r + 0.5 * h * k2r);
    const Eigen::Vector3d k4r = v + h * k3v;
    const Eigen::Vector3d k4v = accel(t1, r + h * k3r);

    position_m_ += (h / 6.0) * (k1r + 2.0 * k2r + 2.0 * k3r + k4r);
    velocity_m_s_ += (h / 6.0) * (k1v + 2.0 * k2v + 2.0 * k3v + k4v);
    cursor_ = t1;
    remaining -= h;
  }
}

bool TrackedObject::positionAt(const pt::Tai& t, pm::Vec3<pm::frames::ECI>& position_m) const {
  if (!valid_) {
    return false;
  }
  if (kind_ == Kind::kTle) {
    const double minutes = secondsBetween(t, tle_epoch_) / 60.0;
    gnc::Sgp4::PositionKm p_teme;
    gnc::Sgp4::VelocityKmS v_teme;
    if (sgp4_.propagate(minutes, p_teme, v_teme) != gnc::Sgp4Status::kOk) {
      return false;
    }
    // SGP4 works in TEME and everything drawn is ECI, so the conversion is not
    // optional dressing — it is the same one the flight side has to make, and
    // omitting it would put the target ~20 arcsec from where it is.
    return frames::eciFromTeme(t, pm::Vec3<pm::frames::TEME>(p_teme.eigen() * 1000.0), position_m);
  }
  stepTo(t);
  if (!position_m_.allFinite()) {
    return false;
  }
  position_m = pm::Vec3<pm::frames::ECI>(position_m_);
  return true;
}

}  // namespace polaris::sim::world
