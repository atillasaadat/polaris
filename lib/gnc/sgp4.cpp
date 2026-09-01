#include "gnc/sgp4.hpp"

#include <cmath>

#include "gnc/sgp4_internal.hpp"
#include "time/civil.hpp"

namespace polaris::gnc {

using namespace sgp4_detail;  // NOLINT(google-build-using-namespace) — file-local by design

namespace sgp4_detail {

double xke() {
  // xke = 60 / sqrt(Re^3 / mu): one minute expressed in the canonical units the
  // theory works in. Derived rather than tabulated so the WGS-72 dependence of
  // every downstream coefficient stays visible.
  return 60.0 / std::sqrt(kEarthRadiusKm * kEarthRadiusKm * kEarthRadiusKm / kMu);
}

double tumin() {
  return 1.0 / xke();
}

}  // namespace sgp4_detail

namespace {

/// Greenwich sidereal time [rad] from a UT1 Julian date (Vallado `gstime`).
double gstime(double jdut1) {
  const double tut1 = (jdut1 - 2451545.0) / 36525.0;
  double temp = -6.2e-6 * tut1 * tut1 * tut1 + 0.093104 * tut1 * tut1 +
                (876600.0 * 3600.0 + 8640184.812866) * tut1 + 67310.54841;
  temp = std::fmod(temp * kDegToRad / 240.0, kTwoPi);
  if (temp < 0.0) {
    temp += kTwoPi;
  }
  return temp;
}

/// Greenwich sidereal time at the element-set epoch.
///
/// The two operation modes genuinely differ here. AFSPC reproduces the original
/// operational software's series about 1970; the improved mode calls the modern
/// `gstime`. They disagree by a small angle that becomes metres of along-track
/// position, which is why the mode has to match whoever fitted the TLE rather
/// than being a preference.
double greenwichSiderealAtEpoch(double epoch_days_1950, Sgp4OpsMode mode) {
  if (mode == Sgp4OpsMode::kImproved) {
    return gstime(epoch_days_1950 + 2433281.5);
  }
  constexpr double kC1 = 1.72027916940703639e-2;
  constexpr double kThetaGr70 = 1.7321343856509374;
  constexpr double kFk5r = 5.07551419432269442e-15;
  const double ts70 = epoch_days_1950 - 7305.0;
  const double ds70 = std::floor(ts70 + 1.0e-8);
  const double tfrac = ts70 - ds70;
  const double c1p2p = kC1 + kTwoPi;
  double gsto = std::fmod(kThetaGr70 + kC1 * ds70 + c1p2p * tfrac + ts70 * ts70 * kFk5r, kTwoPi);
  if (gsto < 0.0) {
    gsto += kTwoPi;
  }
  return gsto;
}

/// Days from 1949 December 31.0 UT to the element set's epoch — the theory's
/// own day count, which every deep-space term is written against.
///
/// **This deliberately reproduces the reference's arithmetic, round-off
/// included, and is not the most accurate way to compute the quantity.** The
/// direct route — `daysFromCivil(year,1,1) - daysFromCivil(1949,12,31) +
/// (epoch_day - 1)` — is exact to the last bit and was what this function did
/// first. It is *more* accurate by about 2e-10 days (20 microseconds), and it
/// makes this propagator disagree with every other SGP4 in the world.
///
/// The reference goes day-of-year -> month/day/h/m/s -> Julian date, and a JD
/// near 2.45e6 has an ULP of ~5e-10 days, so the round trip loses exactly that
/// much. Since SGP4 is defined by its reference implementation and exists here
/// for **interop** — screening against a catalogue everyone else propagates the
/// same way — bit-compatibility is worth more than 20 microseconds of accuracy
/// the theory's own ~1 km error swamps a billionfold. Measured: with the exact
/// epoch the official verification set's extreme-eccentricity cases sit ~4e-6 km
/// off at t = 0; with this one they are at the reference's own noise floor.
double epochDaysSince1950(const TleElements& tle) {
  // days2mdhms: day-of-year to calendar, with the reference's leap rule (a
  // plain year%4, valid over 1901-2099 and wrong in 2100 — the format's own
  // two-digit year expires long before that matters).
  int lmonth[12] = {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
  if ((tle.epoch_year % 4) == 0) {
    lmonth[1] = 29;
  }
  const auto day_of_year = static_cast<int>(std::floor(tle.epoch_day));
  int i = 1;
  int accumulated = 0;
  while ((day_of_year > accumulated + lmonth[i - 1]) && (i < 12)) {
    accumulated += lmonth[i - 1];
    ++i;
  }
  const int month = i;
  const int day = day_of_year - accumulated;

  double temp = (tle.epoch_day - static_cast<double>(day_of_year)) * 24.0;
  const auto hour = static_cast<int>(std::floor(temp));
  temp = (temp - static_cast<double>(hour)) * 60.0;
  const auto minute = static_cast<int>(std::floor(temp));
  const double second = (temp - static_cast<double>(minute)) * 60.0;

  // jday, in the reference's exact form and grouping.
  const auto year = static_cast<double>(tle.epoch_year);
  double jd = 367.0 * year - std::floor((7.0 * (year + std::floor((month + 9.0) / 12.0))) * 0.25) +
              std::floor(275.0 * month / 9.0) + static_cast<double>(day) + 1721013.5;
  double jd_frac =
      (second + static_cast<double>(minute) * 60.0 + static_cast<double>(hour) * 3600.0) / 86400.0;
  if (std::fabs(jd_frac) > 1.0) {
    const double whole = std::floor(jd_frac);
    jd += whole;
    jd_frac -= whole;
  }
  // Summed before differencing, as the reference stores one Julian date.
  return (jd + jd_frac) - 2433281.5;
}

/// Shared initialisation (`initl`): un-Kozai the mean motion and derive the
/// geometry every later coefficient is written in terms of.
struct Initl {
  double ao{0.0}, con41{0.0}, con42{0.0}, cosio{0.0}, cosio2{0.0};
  double eccsq{0.0}, omeosq{0.0}, posq{0.0}, rp{0.0}, rteosq{0.0}, sinio{0.0};
  double no_unkozai{0.0}, gsto{0.0};
};

Initl initl(const Sgp4Elements& e, double epoch_days_1950) {
  Initl o;
  o.eccsq = e.ecco * e.ecco;
  o.omeosq = 1.0 - o.eccsq;
  o.rteosq = std::sqrt(o.omeosq);
  o.cosio = std::cos(e.inclo);
  o.cosio2 = o.cosio * o.cosio;

  // Un-Kozai: a TLE carries the Kozai mean motion, while the theory runs on the
  // Brouwer one. Skipping this is a classic error worth several kilometres.
  const double ak = std::pow(xke() / e.no_kozai, 2.0 / 3.0);
  const double d1 = 0.75 * kJ2 * (3.0 * o.cosio2 - 1.0) / (o.rteosq * o.omeosq);
  double del = d1 / (ak * ak);
  const double adel = ak * (1.0 - del * del - del * (1.0 / 3.0 + 134.0 * del * del / 81.0));
  del = d1 / (adel * adel);
  o.no_unkozai = e.no_kozai / (1.0 + del);

  o.ao = std::pow(xke() / o.no_unkozai, 2.0 / 3.0);
  o.sinio = std::sin(e.inclo);
  const double po = o.ao * o.omeosq;
  o.con42 = 1.0 - 5.0 * o.cosio2;
  o.con41 = -o.con42 - o.cosio2 - o.cosio2;
  o.posq = po * po;
  o.rp = o.ao * (1.0 - e.ecco);
  o.gsto = greenwichSiderealAtEpoch(epoch_days_1950, e.opsmode);
  return o;
}

}  // namespace

const char* toString(Sgp4Status status) {
  switch (status) {
    case Sgp4Status::kOk:
      return "OK";
    case Sgp4Status::kNotInitialised:
      return "NOT_INITIALISED";
    case Sgp4Status::kBadElements:
      return "BAD_ELEMENTS";
    case Sgp4Status::kMeanElementsDiverged:
      return "MEAN_ELEMENTS_DIVERGED";
    case Sgp4Status::kMeanMotionNegative:
      return "MEAN_MOTION_NEGATIVE";
    case Sgp4Status::kPerturbedEccentricity:
      return "PERTURBED_ECCENTRICITY";
    case Sgp4Status::kSemiLatusRectum:
      return "SEMI_LATUS_RECTUM";
    case Sgp4Status::kDecayed:
      return "DECAYED";
  }
  return "UNKNOWN";
}

Sgp4Status Sgp4::initialise(const TleElements& tle, Sgp4OpsMode mode) {
  elements_ = Sgp4Elements{};
  if (!(tle.mean_motion_rev_per_day > 0.0) || !(tle.eccentricity >= 0.0) ||
      !(tle.eccentricity < 1.0)) {
    return Sgp4Status::kBadElements;
  }

  Sgp4Elements e;
  e.opsmode = mode;
  e.bstar = tle.bstar;
  e.ecco = tle.eccentricity;
  e.argpo = tle.arg_perigee_deg * kDegToRad;
  e.inclo = tle.inclination_deg * kDegToRad;
  e.mo = tle.mean_anomaly_deg * kDegToRad;
  e.nodeo = tle.raan_deg * kDegToRad;
  // rev/day -> rad/min.
  e.no_kozai = tle.mean_motion_rev_per_day * kTwoPi / 1440.0;

  const double epoch_days = epochDaysSince1950(tle);
  const Initl init = initl(e, epoch_days);

  e.no_unkozai = init.no_unkozai;
  e.a = std::pow(e.no_unkozai * tumin(), -2.0 / 3.0);
  e.con41 = init.con41;
  e.gsto = init.gsto;
  e.alta = e.a * (1.0 + e.ecco) - 1.0;
  e.altp = e.a * (1.0 - e.ecco) - 1.0;

  // An orbit whose osculating perigee is inside the Earth is refused rather
  // than propagated: the theory produces numbers for it, and they are not an
  // orbit. `omeosq <= 0` is the same statement for a parabolic/hyperbolic set.
  if (!(init.omeosq > 0.0) || !(init.rp > 0.0)) {
    return Sgp4Status::kBadElements;
  }

  const double ss = 78.0 / kEarthRadiusKm + 1.0;
  const double qzms2ttemp = (120.0 - 78.0) / kEarthRadiusKm;
  const double qzms2t = qzms2ttemp * qzms2ttemp * qzms2ttemp * qzms2ttemp;

  if ((init.omeosq >= 0.0) || (e.no_unkozai >= 0.0)) {
    e.use_simplified_drag = (init.rp < (220.0 / kEarthRadiusKm + 1.0));

    // The atmospheric-density fit is piecewise: below 156 km the reference
    // clamps `sfour` and rescales, which is a fit artefact rather than physics
    // and is reproduced exactly because the elements were fitted against it.
    double sfour = ss;
    double qzms24 = qzms2t;
    const double perige = (init.rp - 1.0) * kEarthRadiusKm;
    if (perige < 156.0) {
      sfour = perige - 78.0;
      if (perige < 98.0) {
        sfour = 20.0;
      }
      const double qzms24temp = (120.0 - sfour) / kEarthRadiusKm;
      qzms24 = qzms24temp * qzms24temp * qzms24temp * qzms24temp;
      sfour = sfour / kEarthRadiusKm + 1.0;
    }
    const double pinvsq = 1.0 / init.posq;

    const double tsi = 1.0 / (init.ao - sfour);
    e.eta = init.ao * e.ecco * tsi;
    const double etasq = e.eta * e.eta;
    const double eeta = e.ecco * e.eta;
    const double psisq = std::fabs(1.0 - etasq);
    const double coef = qzms24 * std::pow(tsi, 4.0);
    const double coef1 = coef / std::pow(psisq, 3.5);
    const double cc2 = coef1 * e.no_unkozai *
                       (init.ao * (1.0 + 1.5 * etasq + eeta * (4.0 + etasq)) +
                        0.375 * kJ2 * tsi / psisq * e.con41 * (8.0 + 3.0 * etasq * (8.0 + etasq)));
    e.cc1 = e.bstar * cc2;

    double cc3 = 0.0;
    if (e.ecco > 1.0e-4) {
      cc3 = -2.0 * coef * tsi * kJ3OverJ2 * e.no_unkozai * init.sinio / e.ecco;
    }
    e.x1mth2 = 1.0 - init.cosio2;
    e.cc4 =
        2.0 * e.no_unkozai * coef1 * init.ao * init.omeosq *
        (e.eta * (2.0 + 0.5 * etasq) + e.ecco * (0.5 + 2.0 * etasq) -
         kJ2 * tsi / (init.ao * psisq) *
             (-3.0 * e.con41 * (1.0 - 2.0 * eeta + etasq * (1.5 - 0.5 * eeta)) +
              0.75 * e.x1mth2 * (2.0 * etasq - eeta * (1.0 + etasq)) * std::cos(2.0 * e.argpo)));
    e.cc5 = 2.0 * coef1 * init.ao * init.omeosq * (1.0 + 2.75 * (etasq + eeta) + eeta * etasq);

    const double cosio4 = init.cosio2 * init.cosio2;
    const double temp1 = 1.5 * kJ2 * pinvsq * e.no_unkozai;
    const double temp2 = 0.5 * temp1 * kJ2 * pinvsq;
    const double temp3 = -0.46875 * kJ4 * pinvsq * pinvsq * e.no_unkozai;
    e.mdot = e.no_unkozai + 0.5 * temp1 * init.rteosq * e.con41 +
             0.0625 * temp2 * init.rteosq * (13.0 - 78.0 * init.cosio2 + 137.0 * cosio4);
    e.argpdot = -0.5 * temp1 * init.con42 +
                0.0625 * temp2 * (7.0 - 114.0 * init.cosio2 + 395.0 * cosio4) +
                temp3 * (3.0 - 36.0 * init.cosio2 + 49.0 * cosio4);
    const double xhdot1 = -temp1 * init.cosio;
    e.nodedot = xhdot1 + (0.5 * temp2 * (4.0 - 19.0 * init.cosio2) +
                          2.0 * temp3 * (3.0 - 7.0 * init.cosio2)) *
                             init.cosio;
    const double xpidot = e.argpdot + e.nodedot;
    e.omgcof = e.bstar * cc3 * std::cos(e.argpo);
    e.xmcof = 0.0;
    if (e.ecco > 1.0e-4) {
      e.xmcof = -2.0 / 3.0 * coef * e.bstar / eeta;
    }
    e.nodecf = 3.5 * init.omeosq * xhdot1 * e.cc1;
    e.t2cof = 1.5 * e.cc1;

    // Lyddane's singularity guard: near i = 180 deg the denominator collapses,
    // and the reference clamps rather than diverging. Reproduced exactly.
    if (std::fabs(init.cosio + 1.0) > 1.5e-12) {
      e.xlcof = -0.25 * kJ3OverJ2 * init.sinio * (3.0 + 5.0 * init.cosio) / (1.0 + init.cosio);
    } else {
      e.xlcof = -0.25 * kJ3OverJ2 * init.sinio * (3.0 + 5.0 * init.cosio) / 1.5e-12;
    }
    e.aycof = -0.5 * kJ3OverJ2 * init.sinio;
    const double delmotemp = 1.0 + e.eta * std::cos(e.mo);
    e.delmo = delmotemp * delmotemp * delmotemp;
    e.sinmao = std::sin(e.mo);
    e.x7thm1 = 7.0 * init.cosio2 - 1.0;

    e.is_deep_space = (kTwoPi / e.no_unkozai) >= 225.0;
    if (e.is_deep_space) {
      // **Deep space always takes the simplified drag branch**, whatever the
      // perigee height says. The higher-order terms d2/d3/d4 (and their t^3/t^4
      // cofactors) are a near-Earth atmospheric fit and are simply not part of
      // SDP4; applying them is silent above the noise only because they carry
      // t^2 and higher, so the error is *exactly zero at epoch* and grows with
      // the drag term. That signature — every case exact at t = 0, high-drag
      // deep-space cases metres off later, low-drag GEO cases fine — is what
      // the verification set caught, and it is why this line is not a tidy-up.
      e.use_simplified_drag = true;
      e.gsto_deep = init.gsto;
      double em = e.ecco;
      double argpm = e.argpo;
      double inclm = e.inclo;
      double mm = e.mo;
      double nm = e.no_unkozai;
      double nodem = e.nodeo;

      const DeepSpaceCommon common = deepSpaceCommon(e, epoch_days, 0.0, e.argpo, e.nodeo);

      // Lunar-solar periodics at epoch, with the Lyddane correction suppressed
      // (`init = true`): at t = 0 the elements are the fitted ones and applying
      // the choice here would move them.
      double ep = e.ecco;
      double inclp = e.inclo;
      double nodep = e.nodeo;
      double argpp = e.argpo;
      double mp = e.mo;
      deepSpacePeriodics(e, 0.0, /*init=*/true, ep, inclp, nodep, argpp, mp);

      deepSpaceInit(e, common, epoch_days, e.argpo, em, argpm, inclm, mm, nm, nodem);
    }

    // Higher-order drag terms, only for the non-simplified branch.
    if (!e.use_simplified_drag) {
      const double cc1sq = e.cc1 * e.cc1;
      e.d2 = 4.0 * init.ao * tsi * cc1sq;
      const double temp = e.d2 * tsi * e.cc1 / 3.0;
      e.d3 = (17.0 * init.ao + sfour) * temp;
      e.d4 = 0.5 * temp * init.ao * tsi * (221.0 * init.ao + 31.0 * sfour) * e.cc1;
      e.t3cof = e.d2 + 2.0 * cc1sq;
      e.t4cof = 0.25 * (3.0 * e.d3 + e.cc1 * (12.0 * e.d2 + 10.0 * cc1sq));
      e.t5cof = 0.2 * (3.0 * e.d4 + 12.0 * e.cc1 * e.d3 + 6.0 * e.d2 * e.d2 +
                       15.0 * cc1sq * (2.0 * e.d2 + cc1sq));
    }
    (void)xpidot;
  }

  e.initialised = true;
  elements_ = e;

  // The reference runs one propagation at t = 0 to settle derived state; doing
  // the same here keeps any epoch-zero special case exercised at init rather
  // than on a caller's first request.
  PositionKm r;
  VelocityKmS v;
  const Sgp4Status at_epoch = propagate(0.0, r, v);
  if (at_epoch != Sgp4Status::kOk && at_epoch != Sgp4Status::kDecayed) {
    elements_.initialised = false;
    return Sgp4Status::kBadElements;
  }
  return Sgp4Status::kOk;
}

}  // namespace polaris::gnc
