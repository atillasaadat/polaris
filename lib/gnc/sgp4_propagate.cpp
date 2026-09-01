#include <cmath>

#include "gnc/sgp4.hpp"
#include "gnc/sgp4_internal.hpp"

/// @file
/// @brief SGP4 propagation from an initialised element set (design doc §8.3).
///
/// The secular/periodic evaluation and the Kepler solve, following
/// [vallado2006revisiting] `sgp4`. Split from initialisation for size; the two
/// are one algorithm.

namespace polaris::gnc {

using namespace sgp4_detail;  // NOLINT(google-build-using-namespace) — file-local by design

namespace {

/// Kilometres per canonical distance unit, per minute -> km/s.
double kmPerSecond() {
  return kEarthRadiusKm * xke() / 60.0;
}

}  // namespace

Sgp4Status Sgp4::propagate(double minutes_since_epoch, PositionKm& position_km,
                           VelocityKmS& velocity_km_s) const {
  const Sgp4Elements& e = elements_;
  if (!e.initialised) {
    return Sgp4Status::kNotInitialised;
  }
  const double t = minutes_since_epoch;

  // --- Secular update for atmospheric drag and gravity --------------------
  const double xmdf = e.mo + e.mdot * t;
  const double argpdf = e.argpo + e.argpdot * t;
  const double nodedf = e.nodeo + e.nodedot * t;
  double argpm = argpdf;
  double mm = xmdf;
  const double t2 = t * t;
  double nodem = nodedf + e.nodecf * t2;
  double tempa = 1.0 - e.cc1 * t;
  double tempe = e.bstar * e.cc4 * t;
  double templ = e.t2cof * t2;

  if (!e.use_simplified_drag) {
    const double delomg = e.omgcof * t;
    const double delmtemp = 1.0 + e.eta * std::cos(xmdf);
    const double delm = e.xmcof * (delmtemp * delmtemp * delmtemp - e.delmo);
    const double temp = delomg + delm;
    mm = xmdf + temp;
    argpm = argpdf - temp;
    const double t3 = t2 * t;
    const double t4 = t3 * t;
    tempa = tempa - e.d2 * t2 - e.d3 * t3 - e.d4 * t4;
    tempe = tempe + e.bstar * e.cc5 * (std::sin(mm) - e.sinmao);
    templ = templ + e.t3cof * t3 + t4 * (e.t4cof + t * e.t5cof);
  }

  double nm = e.no_unkozai;
  double em = e.ecco;
  double inclm = e.inclo;
  if (e.is_deep_space) {
    double dndt = 0.0;
    deepSpaceSecular(e, t, em, argpm, inclm, mm, nm, nodem, dndt);
  }

  if (nm <= 0.0) {
    return Sgp4Status::kMeanMotionNegative;
  }
  double am = std::pow(xke() / nm, 2.0 / 3.0) * tempa * tempa;
  nm = xke() / std::pow(am, 1.5);
  em = em - tempe;

  // The reference's own bounds. An eccentricity slightly below zero is a
  // rounding artefact and is floored; anything further out means the mean
  // elements have left the theory's domain and is refused rather than clamped.
  if (em >= 1.0 || em < -0.001) {
    return Sgp4Status::kMeanElementsDiverged;
  }
  if (em < 1.0e-6) {
    em = 1.0e-6;
  }
  mm = mm + e.no_unkozai * templ;
  double xlm = mm + argpm + nodem;
  nodem = std::fmod(nodem, kTwoPi);
  argpm = std::fmod(argpm, kTwoPi);
  xlm = std::fmod(xlm, kTwoPi);
  mm = std::fmod(xlm - argpm - nodem, kTwoPi);

  // Keep the mean-anomaly-derived angles on a consistent branch; the reference
  // relies on fmod's sign behaviour here and the periodics below assume it.
  const double sinim = std::sin(inclm);
  const double cosim = std::cos(inclm);

  // --- Lunar-solar periodics (deep space only) ----------------------------
  double ep = em;
  double xincp = inclm;
  double argpp = argpm;
  double nodep = nodem;
  double mp = mm;
  double sinip = sinim;
  double cosip = cosim;
  double aycof = e.aycof;
  double xlcof = e.xlcof;
  double con41 = e.con41;
  double x1mth2 = e.x1mth2;
  double x7thm1 = e.x7thm1;

  if (e.is_deep_space) {
    deepSpacePeriodics(e, t, /*init=*/false, ep, xincp, nodep, argpp, mp);
    if (xincp < 0.0) {
      // A negative inclination is the Lyddane formulation's way of expressing a
      // retrograde crossing; fold it back into the conventional range.
      xincp = -xincp;
      nodep = nodep + kPi;
      argpp = argpp - kPi;
    }
    if (ep < 0.0 || ep > 1.0) {
      return Sgp4Status::kPerturbedEccentricity;
    }
    sinip = std::sin(xincp);
    cosip = std::cos(xincp);
    aycof = -0.5 * kJ3OverJ2 * sinip;
    if (std::fabs(cosip + 1.0) > 1.5e-12) {
      xlcof = -0.25 * kJ3OverJ2 * sinip * (3.0 + 5.0 * cosip) / (1.0 + cosip);
    } else {
      xlcof = -0.25 * kJ3OverJ2 * sinip * (3.0 + 5.0 * cosip) / 1.5e-12;
    }
    const double cosisq = cosip * cosip;
    con41 = 3.0 * cosisq - 1.0;
    x1mth2 = 1.0 - cosisq;
    x7thm1 = 7.0 * cosisq - 1.0;
  }

  // --- Long-period periodics ----------------------------------------------
  const double axnl = ep * std::cos(argpp);
  double temp = 1.0 / (am * (1.0 - ep * ep));
  const double aynl = ep * std::sin(argpp) + temp * aycof;
  const double xl = mp + argpp + nodep + temp * xlcof * axnl;

  // --- Kepler's equation ---------------------------------------------------
  // Newton-Raphson with the reference's step clamp and 10-iteration cap. The
  // clamp is what keeps a near-parabolic case from stepping across the root;
  // the cap means a non-converged solve returns the best iterate rather than
  // spinning, which is the correct trade on a flight path.
  const double u = std::fmod(xl - nodep, kTwoPi);
  double eo1 = u;
  double tem5 = 9999.9;
  double sineo1 = 0.0;
  double coseo1 = 0.0;
  for (int ktr = 1; ktr <= 10 && std::fabs(tem5) >= 1.0e-12; ++ktr) {
    sineo1 = std::sin(eo1);
    coseo1 = std::cos(eo1);
    tem5 = 1.0 - coseo1 * axnl - sineo1 * aynl;
    tem5 = (u - aynl * coseo1 + axnl * sineo1 - eo1) / tem5;
    if (std::fabs(tem5) >= 0.95) {
      tem5 = tem5 > 0.0 ? 0.95 : -0.95;
    }
    eo1 = eo1 + tem5;
  }
  sineo1 = std::sin(eo1);
  coseo1 = std::cos(eo1);

  // --- Short-period periodics and the state vector -------------------------
  const double ecose = axnl * coseo1 + aynl * sineo1;
  const double esine = axnl * sineo1 - aynl * coseo1;
  const double el2 = axnl * axnl + aynl * aynl;
  const double pl = am * (1.0 - el2);
  if (pl < 0.0) {
    return Sgp4Status::kSemiLatusRectum;
  }

  const double rl = am * (1.0 - ecose);
  const double rdotl = std::sqrt(am) * esine / rl;
  const double rvdotl = std::sqrt(pl) / rl;
  const double betal = std::sqrt(1.0 - el2);
  temp = esine / (1.0 + betal);
  const double sinu = am / rl * (sineo1 - aynl - axnl * temp);
  const double cosu = am / rl * (coseo1 - axnl + aynl * temp);
  double su = std::atan2(sinu, cosu);
  const double sin2u = (cosu + cosu) * sinu;
  const double cos2u = 1.0 - 2.0 * sinu * sinu;
  temp = 1.0 / pl;
  const double temp1 = 0.5 * kJ2 * temp;
  const double temp2 = temp1 * temp;

  const double mrt = rl * (1.0 - 1.5 * temp2 * betal * con41) + 0.5 * temp1 * x1mth2 * cos2u;
  su = su - 0.25 * temp2 * x7thm1 * sin2u;
  const double xnode = nodep + 1.5 * temp2 * cosip * sin2u;
  const double xinc = xincp + 1.5 * temp2 * cosip * sinip * cos2u;
  const double mvt = rdotl - nm * temp1 * x1mth2 * sin2u / xke();
  const double rvdot = rvdotl + nm * temp1 * (x1mth2 * cos2u + 1.5 * con41) / xke();

  // Orientation vectors.
  const double sinsu = std::sin(su);
  const double cossu = std::cos(su);
  const double snod = std::sin(xnode);
  const double cnod = std::cos(xnode);
  const double sini = std::sin(xinc);
  const double cosi = std::cos(xinc);
  const double xmx = -snod * cosi;
  const double xmy = cnod * cosi;
  const double ux = xmx * sinsu + cnod * cossu;
  const double uy = xmy * sinsu + snod * cossu;
  const double uz = sini * sinsu;
  const double vx = xmx * cossu - cnod * sinsu;
  const double vy = xmy * cossu - snod * sinsu;
  const double vz = sini * cossu;

  // Decay is reported *after* computing the state: the position is still the
  // theory's answer, and a caller screening a decaying object wants both the
  // status and where the theory put it.
  const double km_per_second = kmPerSecond();
  position_km =
      PositionKm(mrt * ux * kEarthRadiusKm, mrt * uy * kEarthRadiusKm, mrt * uz * kEarthRadiusKm);
  velocity_km_s =
      VelocityKmS((mvt * ux + rvdot * vx) * km_per_second, (mvt * uy + rvdot * vy) * km_per_second,
                  (mvt * uz + rvdot * vz) * km_per_second);

  if (mrt < 1.0) {
    return Sgp4Status::kDecayed;
  }
  return Sgp4Status::kOk;
}

}  // namespace polaris::gnc
