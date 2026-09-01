#include "frames/teme_eci.hpp"

#include <erfa.h>

#include <cmath>

namespace polaris::frames {
namespace {

/// R3(theta): rotation **of the frame** about +Z, in the sense Vallado's
/// `ROT3` uses — [[c, s, 0], [-s, c, 0], [0, 0, 1]].
Eigen::Matrix3d rot3(double theta) {
  const double c = std::cos(theta);
  const double s = std::sin(theta);
  Eigen::Matrix3d m;
  m << c, s, 0.0, -s, c, 0.0, 0.0, 0.0, 1.0;
  return m;
}

Eigen::Matrix3d toEigen(const double m[3][3]) {
  // ERFA writes [row][col]; Eigen is column-major by default, so copy rather
  // than map the raw buffer (the same trap `eci_ecef.cpp` calls out).
  Eigen::Matrix3d out;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      out(i, j) = m[i][j];
    }
  }
  return out;
}

}  // namespace

bool temeToEciMatrix(const time::Tai& t, Eigen::Matrix3d& out) {
  const time::JulianDate tt = time::julianDate(time::toTt(t));
  if (!std::isfinite(tt.day) || !std::isfinite(tt.fraction)) {
    return false;
  }

  // **Frame bias first.** `eraPmat76` precesses from the FK5 J2000 mean equator
  // and equinox, but this repository's ECI is GCRS (`eci_ecef.hpp`, which runs
  // the IAU 2006/2000A chain). The two differ by the ~23 mas frame bias, which
  // is 0.8 m at LEO — measured against astropy's independent TEME
  // implementation, where omitting it left a 0.687 m residual sitting almost
  // entirely on Z, the bias's own signature.
  //
  // That is far inside SGP4's ~1 km, so it is not an accuracy argument. It is a
  // *consistency* one: a TLE-derived position and the orbit filter's position
  // get differenced — for conjunction range, or for a pointing vector — and a
  // systematic 0.8 m offset between two frames both called "ECI" is exactly the
  // kind of quiet inconsistency this repository's one-transform rule exists to
  // prevent (REQ-CONV-002).
  double rb[3][3];
  double rp_bias[3][3];
  double rbp[3][3];
  eraBp00(tt.day, tt.fraction, rb, rp_bias, rbp);

  // GCRS -> MOD (IAU-76 precession), applied after the bias.
  double rp[3][3];
  eraPmat76(tt.day, tt.fraction, rp);
  // MOD -> TOD (IAU-80 nutation).
  double rn[3][3];
  eraNutm80(tt.day, tt.fraction, rn);
  // TOD -> TEME is a rotation about the pole by the equation of the equinoxes.
  // eraEqeq94 carries the IAU-1994 complementary terms, which is the
  // `eqeterms > 0` branch of Vallado's teme2eci and the one that matches the
  // operational convention TLEs are distributed under.
  const double eqeq = eraEqeq94(tt.day, tt.fraction);
  if (!std::isfinite(eqeq)) {
    return false;
  }

  // J2000 -> TEME, then transpose for the direction we want.
  const Eigen::Matrix3d eci_to_teme = rot3(eqeq) * toEigen(rn) * toEigen(rp) * toEigen(rb);
  Eigen::Matrix3d teme_to_eci = eci_to_teme.transpose();
  if (!teme_to_eci.allFinite()) {
    return false;
  }
  out = teme_to_eci;
  return true;
}

bool eciFromTeme(const time::Tai& t, const math::Vec3<math::frames::TEME>& teme,
                 math::Vec3<math::frames::ECI>& out) {
  if (!teme.isFinite()) {
    return false;
  }
  Eigen::Matrix3d m;
  if (!temeToEciMatrix(t, m)) {
    return false;
  }
  const Eigen::Vector3d r = m * teme.eigen();
  if (!r.allFinite()) {
    return false;
  }
  out = math::Vec3<math::frames::ECI>(r);
  return true;
}

bool temeFromEci(const time::Tai& t, const math::Vec3<math::frames::ECI>& eci,
                 math::Vec3<math::frames::TEME>& out) {
  if (!eci.isFinite()) {
    return false;
  }
  Eigen::Matrix3d m;
  if (!temeToEciMatrix(t, m)) {
    return false;
  }
  const Eigen::Vector3d r = m.transpose() * eci.eigen();
  if (!r.allFinite()) {
    return false;
  }
  out = math::Vec3<math::frames::TEME>(r);
  return true;
}

bool eciStateFromTeme(const time::Tai& t, const math::Vec3<math::frames::TEME>& position_teme,
                      const math::Vec3<math::frames::TEME>& velocity_teme,
                      math::Vec3<math::frames::ECI>& position_eci,
                      math::Vec3<math::frames::ECI>& velocity_eci) {
  if (!position_teme.isFinite() || !velocity_teme.isFinite()) {
    return false;
  }
  Eigen::Matrix3d m;
  if (!temeToEciMatrix(t, m)) {
    return false;
  }
  // One matrix, both vectors: no transport term between two quasi-inertial
  // frames (see the header). Computed once so the two cannot be rotated with
  // different epochs by a later edit.
  const Eigen::Vector3d r = m * position_teme.eigen();
  const Eigen::Vector3d v = m * velocity_teme.eigen();
  if (!r.allFinite() || !v.allFinite()) {
    return false;
  }
  position_eci = math::Vec3<math::frames::ECI>(r);
  velocity_eci = math::Vec3<math::frames::ECI>(v);
  return true;
}

}  // namespace polaris::frames
