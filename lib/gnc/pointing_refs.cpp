#include "gnc/pointing_refs.hpp"

#include <cmath>

namespace polaris::gnc {

const char* toString(BodyVectorKind kind) {
  switch (kind) {
    case BodyVectorKind::kBodyX:
      return "BODY_X";
    case BodyVectorKind::kBodyY:
      return "BODY_Y";
    case BodyVectorKind::kBodyZ:
      return "BODY_Z";
    case BodyVectorKind::kStarTracker:
      return "STAR_TRACKER";
    case BodyVectorKind::kSunSensor:
      return "SUN_SENSOR";
    case BodyVectorKind::kCamera:
      return "CAMERA";
    case BodyVectorKind::kCustom:
      return "CUSTOM_BODY_VEC";
  }
  return "UNKNOWN";
}

const char* toString(PointingTargetKind kind) {
  switch (kind) {
    case PointingTargetKind::kSun:
      return "SUN";
    case PointingTargetKind::kMoon:
      return "MOON";
    case PointingTargetKind::kNadir:
      return "NADIR";
    case PointingTargetKind::kEcefPoint:
      return "ECEF_TARGET";
    case PointingTargetKind::kStarJ2000:
      return "STAR_J2000";
    case PointingTargetKind::kJ2000X:
      return "J2000_X";
    case PointingTargetKind::kJ2000Y:
      return "J2000_Y";
    case PointingTargetKind::kJ2000Z:
      return "J2000_Z";
    case PointingTargetKind::kLvlhX:
      return "LVLH_X";
    case PointingTargetKind::kLvlhY:
      return "LVLH_Y";
    case PointingTargetKind::kLvlhZ:
      return "LVLH_Z";
    case PointingTargetKind::kSatTle:
      return "SAT_TLE";
    case PointingTargetKind::kSatState:
      return "SAT_STATE";
  }
  return "UNKNOWN";
}

namespace {

bool usable(const math::Vec3<math::frames::Body>& v) {
  return v.isFinite() && v.norm() > 0.0;
}

}  // namespace

bool BodyVectorTable::setSensor(BodyVectorKind kind, int index,
                                const math::Vec3<math::frames::Body>& v) {
  if (!validSensorIndex(index) || !usable(v)) {
    return false;
  }
  switch (kind) {
    case BodyVectorKind::kStarTracker:
      star_tracker_[index] = Slot{v, true};
      return true;
    case BodyVectorKind::kSunSensor:
      sun_sensor_[index] = Slot{v, true};
      return true;
    case BodyVectorKind::kCamera:
      camera_[index] = Slot{v, true};
      return true;
    default:
      // The structural axes and the custom slots are not mounting data.
      return false;
  }
}

bool BodyVectorTable::setCustom(int index, const math::Vec3<math::frames::Body>& v) {
  if (!validCustomIndex(index) || !usable(v)) {
    return false;
  }
  custom_[index] = Slot{v, true};
  return true;
}

bool BodyVectorTable::clearCustom(int index) {
  if (!validCustomIndex(index)) {
    return false;
  }
  custom_[index] = Slot{};
  return true;
}

bool BodyVectorTable::isCustomSet(int index) const {
  return validCustomIndex(index) && custom_[index].valid;
}

int BodyVectorTable::customCount() const {
  int n = 0;
  for (int i = 0; i < kMaxCustomBodyVectors; ++i) {
    if (custom_[i].valid) {
      ++n;
    }
  }
  return n;
}

bool BodyVectorTable::resolve(const BodyVectorRef& ref, math::Vec3<math::frames::Body>& out) const {
  math::Vec3<math::frames::Body> v;
  switch (ref.kind) {
    case BodyVectorKind::kBodyX:
      v = math::Vec3<math::frames::Body>(1.0, 0.0, 0.0);
      break;
    case BodyVectorKind::kBodyY:
      v = math::Vec3<math::frames::Body>(0.0, 1.0, 0.0);
      break;
    case BodyVectorKind::kBodyZ:
      v = math::Vec3<math::frames::Body>(0.0, 0.0, 1.0);
      break;
    case BodyVectorKind::kStarTracker:
    case BodyVectorKind::kSunSensor:
    case BodyVectorKind::kCamera: {
      const int i = static_cast<int>(ref.index);
      if (!validSensorIndex(i)) {
        return false;
      }
      const Slot& s = (ref.kind == BodyVectorKind::kStarTracker) ? star_tracker_[i]
                      : (ref.kind == BodyVectorKind::kSunSensor) ? sun_sensor_[i]
                                                                 : camera_[i];
      // An uninstalled unit is refused rather than resolving to zero: a null
      // vector would propagate into the guidance as a degenerate axis and be
      // reported there, blaming the geometry instead of the missing mounting.
      if (!s.valid) {
        return false;
      }
      v = s.v;
      break;
    }
    case BodyVectorKind::kCustom: {
      const int i = static_cast<int>(ref.index);
      if (!validCustomIndex(i) || !custom_[i].valid) {
        return false;
      }
      v = custom_[i].v;
      break;
    }
    default:
      return false;
  }

  const double n = v.norm();
  if (!(n > 0.0)) {
    return false;
  }
  const double sign = ref.negate ? -1.0 : 1.0;
  out = math::Vec3<math::frames::Body>(v.eigen() * (sign / n));
  return true;
}

namespace {

/// Wide enough to admit any real site, tight enough to catch a metres-versus-
/// kilometres unit error on upload — which is the mistake that actually happens.
constexpr double kMinHeightM = -1000.0;
constexpr double kMaxHeightM = 100000.0;

}  // namespace

bool GroundPointTable::set(int index, double latitude_rad, double longitude_rad, double height_m) {
  if (!validIndex(index)) {
    return false;
  }
  if (!std::isfinite(latitude_rad) || !std::isfinite(longitude_rad) || !std::isfinite(height_m)) {
    return false;
  }
  constexpr double kHalfPi = 1.570796326794896619231321691639751442;
  if (std::fabs(latitude_rad) > kHalfPi + 1e-12) {
    return false;
  }
  if (height_m < kMinHeightM || height_m > kMaxHeightM) {
    return false;
  }
  points_[index] = Point{latitude_rad, longitude_rad, height_m, true};
  return true;
}

bool GroundPointTable::clear(int index) {
  if (!validIndex(index)) {
    return false;
  }
  points_[index] = Point{};
  return true;
}

bool GroundPointTable::isSet(int index) const {
  return validIndex(index) && points_[index].valid;
}

bool GroundPointTable::get(int index, Point& out) const {
  if (!isSet(index)) {
    return false;
  }
  out = points_[index];
  return true;
}

int GroundPointTable::count() const {
  int n = 0;
  for (int i = 0; i < kMaxGroundPoints; ++i) {
    if (points_[i].valid) {
      ++n;
    }
  }
  return n;
}

}  // namespace polaris::gnc
