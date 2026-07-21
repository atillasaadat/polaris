#include "world/magnetic_field.hpp"

#include <cmath>
#include <cstdint>

#include "time/civil.hpp"
#include "time/utc.hpp"

namespace polaris::sim::world {
namespace {

/// True if @p year is a leap year on the proleptic Gregorian calendar.
constexpr bool isLeapYear(std::int64_t year) {
  return (year % 4 == 0 && year % 100 != 0) || year % 400 == 0;
}

constexpr double kSecondsPerDay = 86400.0;

}  // namespace

bool decimalYear(const time::Tai& epoch, const time::LeapSecondTable& leap, double& out) {
  const time::UtcDateTime utc = time::utcFromTai(epoch, leap);
  const std::int64_t year_start = time::daysFromCivil(utc.year, 1, 1);
  const std::int64_t next_year_start = time::daysFromCivil(utc.year + 1, 1, 1);
  const double days_in_year = static_cast<double>(next_year_start - year_start);
  if (!(days_in_year > 0.0)) {
    return false;
  }

  const double day_of_year =
      static_cast<double>(time::daysFromCivil(utc.year, utc.month, utc.day) - year_start);
  const double seconds_of_day =
      static_cast<double>(utc.hour) * 3600.0 + static_cast<double>(utc.minute) * 60.0 +
      static_cast<double>(utc.second) + static_cast<double>(utc.nanosecond) * 1e-9;

  // Divide by the actual length of *this* year rather than a fixed 365.25, so a
  // date lands at the same fraction the IGRF grid means by it. The difference is
  // under a day — negligible against a field that drifts tens of nT per year —
  // but a fixed divisor would also make 1 January drift off 0.0 in leap years,
  // which is the kind of thing that is confusing to see in a log.
  out = static_cast<double>(utc.year) +
        (day_of_year + seconds_of_day / kSecondsPerDay) / days_in_year;
  // Guard the constant above against silent divergence from the calendar helper.
  static_assert(isLeapYear(2024) && !isLeapYear(2023) && !isLeapYear(1900) && isLeapYear(2000),
                "leap-year rule");
  return std::isfinite(out);
}

bool EarthMagneticField::field(const time::Tai& epoch, const math::Vec3<math::frames::ECI>& r_eci,
                               math::Vec3<math::frames::ECI>& out) const {
  if (!good()) {
    return false;
  }

  math::Quat<math::frames::ECEF, math::frames::ECI> q;
  if (!eci_to_ecef_(epoch, q)) {
    return false;
  }

  double year = 0.0;
  if (!decimalYear(epoch, *leap_, year)) {
    return false;
  }

  const math::Vec3<math::frames::ECEF> r_ecef(q.core().rotate(r_eci.eigen()));
  math::Vec3<math::frames::ECEF> b_ecef;
  if (!field_.field(r_ecef, year, b_ecef)) {
    return false;
  }

  // B is a vector field, so it rotates back with the inverse of the same
  // rotation that took the position out — no translation, the frames share an
  // origin.
  out = math::Vec3<math::frames::ECI>(q.core().inverse().rotate(b_ecef.eigen()));
  return true;
}

MagneticFieldFn EarthMagneticField::fieldFn() const {
  return [this](const time::Tai& epoch, const math::Vec3<math::frames::ECI>& r_eci,
                math::Vec3<math::frames::ECI>& out) { return field(epoch, r_eci, out); };
}

math::Vec3<math::frames::Body> ResidualDipoleTorque::torque(const state::TruthState& s) const {
  if (!field_) {
    return math::Vec3<math::frames::Body>::Zero();
  }
  math::Vec3<math::frames::ECI> b_eci;
  if (!field_(s.epoch, s.position, b_eci)) {
    return math::Vec3<math::frames::Body>::Zero();
  }
  // The dipole is fixed in the Body frame, so bring the field to Body rather
  // than the dipole to ECI — the cross product has to happen in one frame and
  // the torque is wanted in Body.
  const Eigen::Vector3d b_body = s.attitude.core().rotate(b_eci.eigen());
  return math::Vec3<math::frames::Body>(Eigen::Vector3d(dipole_body_.eigen().cross(b_body)));
}

}  // namespace polaris::sim::world
