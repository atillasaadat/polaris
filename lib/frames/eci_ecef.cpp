/// @file
/// @brief IAU 2006/2000A ECI↔ECEF reduction (design doc §3.1). See eci_ecef.hpp.

#include "frames/eci_ecef.hpp"

#include <erfa.h>

#include <cmath>
#include <Eigen/Core>
#include <Eigen/Geometry>

#include "constants/constants.hpp"
#include "time/timescales.hpp"

namespace polaris::frames {

namespace {

using math::frames::ECEF;
using math::frames::ECI;

/// Earth's rotation vector in ECEF: ω⊕ about the polar (+Z) axis [rad/s].
/// See the header for why the EOP length-of-day correction is not applied.
Eigen::Vector3d earthRateEcef() {
  return Eigen::Vector3d(0.0, 0.0, constants::wgs84::kEarthRate);
}

/// The CRS→TRS (ECI→ECEF) direction-cosine matrix at @p t, per IAU 2006/2000A.
/// Returns false if @p t or @p eop is non-finite (§3.6 boundary guard).
///
/// ERFA wants two-part JDs so it can keep sub-microsecond resolution at
/// JD ≈ 2.46e6; `time::julianDate()` supplies exactly that split. TT drives the
/// precession/nutation model, UT1 drives the Earth Rotation Angle — feeding one
/// where the other belongs is a ~0.5 mas/ms error, hence the distinct scales.
bool crsToTrsMatrix(const time::Tai& t, const EopValue& eop, Eigen::Matrix3d& out) {
  if (!std::isfinite(eop.ut1_minus_tai) || !std::isfinite(eop.xp_arcsec) ||
      !std::isfinite(eop.yp_arcsec)) {
    return false;
  }
  const time::JulianDate tt = time::julianDate(time::toTt(t));
  const time::JulianDate ut1 = time::julianDate(ut1FromTai(t, eop.ut1_minus_tai));
  if (!std::isfinite(tt.day) || !std::isfinite(ut1.day)) {
    return false;
  }

  double rc2t[3][3];
  eraC2t06a(tt.day, tt.fraction, ut1.day, ut1.fraction,
            eop.xp_arcsec * constants::iau::kArcsecToRad,
            eop.yp_arcsec * constants::iau::kArcsecToRad, rc2t);

  // ERFA writes rc2t[row][col]; Eigen defaults to column-major storage, so copy
  // element-wise rather than mapping the raw buffer.
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      out(i, j) = rc2t[i][j];
    }
  }
  return out.allFinite();
}

/// The ECI→ECEF rotation at @p t as a canonical quaternion.
bool ecefFromEciCore(const time::Tai& t, const EopValue& eop, math::Quaternion& out) {
  Eigen::Matrix3d dcm;
  if (!crsToTrsMatrix(t, eop, dcm)) {
    return false;
  }
  const math::Quaternion q = math::Quaternion::FromRotationMatrix(dcm).canonical();
  if (!q.isFinite()) {
    return false;
  }
  out = q;
  return true;
}

}  // namespace

bool ecefFromEci(const time::Tai& t, const EopValue& eop, math::Quat<ECEF, ECI>& out) {
  math::Quaternion q;
  if (!ecefFromEciCore(t, eop, q)) {
    return false;
  }
  out = math::Quat<ECEF, ECI>(q);
  return true;
}

bool eciFromEcef(const time::Tai& t, const EopValue& eop, math::Quat<ECI, ECEF>& out) {
  math::Quat<ECEF, ECI> fwd;
  if (!ecefFromEci(t, eop, fwd)) {
    return false;
  }
  out = fwd.inverse().canonical();
  return true;
}

bool ecefStateFromEci(const time::Tai& t, const EopValue& eop, const math::Vec3<ECI>& r_eci,
                      const math::Vec3<ECI>& v_eci, math::Vec3<ECEF>& r_ecef,
                      math::Vec3<ECEF>& v_ecef) {
  if (!r_eci.isFinite() || !v_eci.isFinite()) {
    return false;
  }
  Eigen::Matrix3d dcm;
  if (!crsToTrsMatrix(t, eop, dcm)) {
    return false;
  }
  const Eigen::Vector3d r = dcm * r_eci.eigen();
  // v_ecef = R v_eci − ω⊕ × r_ecef: the rotating-frame transport term.
  const Eigen::Vector3d v = dcm * v_eci.eigen() - earthRateEcef().cross(r);
  if (!r.allFinite() || !v.allFinite()) {
    return false;
  }
  r_ecef = math::Vec3<ECEF>(r);
  v_ecef = math::Vec3<ECEF>(v);
  return true;
}

bool eciStateFromEcef(const time::Tai& t, const EopValue& eop, const math::Vec3<ECEF>& r_ecef,
                      const math::Vec3<ECEF>& v_ecef, math::Vec3<ECI>& r_eci,
                      math::Vec3<ECI>& v_eci) {
  if (!r_ecef.isFinite() || !v_ecef.isFinite()) {
    return false;
  }
  Eigen::Matrix3d dcm;
  if (!crsToTrsMatrix(t, eop, dcm)) {
    return false;
  }
  // Exact inverse of ecefStateFromEci: undo the transport term, then rotate back.
  const Eigen::Vector3d r = dcm.transpose() * r_ecef.eigen();
  const Eigen::Vector3d v =
      dcm.transpose() * (v_ecef.eigen() + earthRateEcef().cross(r_ecef.eigen()));
  if (!r.allFinite() || !v.allFinite()) {
    return false;
  }
  r_eci = math::Vec3<ECI>(r);
  v_eci = math::Vec3<ECI>(v);
  return true;
}

}  // namespace polaris::frames
