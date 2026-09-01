#include <cmath>

#include "gnc/sgp4_internal.hpp"

/// @file
/// @brief SDP4 — the deep-space half of SGP4 (design doc §8.3)
/// [vallado2006revisiting] `dscom`, `dpper`, `dsinit`, `dspace`.
///
/// Everything here exists because above a 225-minute period two effects the
/// near-Earth theory ignores stop being ignorable: **lunar-solar gravitational
/// periodics**, and **resonance** with the Earth's tesseral harmonics for orbits
/// whose period is commensurate with a sidereal day. Twenty-four of the
/// thirty-three official verification cases exercise this file, which is a fair
/// measure of how much of SGP4's real difficulty lives on this side.
///
/// The series coefficients are empirical fits from the original Spacetrack
/// Report #3 lineage. They are reproduced to the digit rather than re-derived
/// or tidied: the TLEs were fitted against *these* numbers, so a "cleaner"
/// constant is simply a different theory that the elements were not fitted for.

namespace polaris::gnc::sgp4_detail {
namespace {

constexpr double kZes = 0.01675;
constexpr double kZel = 0.05490;
constexpr double kC1ss = 2.9864797e-6;
constexpr double kC1l = 4.7968065e-7;
constexpr double kZsinis = 0.39785416;
constexpr double kZcosis = 0.91744867;
constexpr double kZcosgs = 0.1945905;
constexpr double kZsings = -0.98088458;

constexpr double kZns = 1.19459e-5;
constexpr double kZnl = 1.5835218e-4;

/// Earth rotation in the resonance terms' units [rad/min].
constexpr double kRptim = 4.37526908801129966e-3;

}  // namespace

DeepSpaceCommon deepSpaceCommon(Sgp4Elements& e, double epoch_days_1950, double tc, double argpp,
                                double nodep) {
  DeepSpaceCommon c;
  c.nm = e.no_unkozai;
  c.em = e.ecco;
  c.snodm = std::sin(nodep);
  c.cnodm = std::cos(nodep);
  c.sinomm = std::sin(argpp);
  c.cosomm = std::cos(argpp);
  c.sinim = std::sin(e.inclo);
  c.cosim = std::cos(e.inclo);
  c.emsq = c.em * c.em;
  const double betasq = 1.0 - c.emsq;
  c.rtemsq = std::sqrt(betasq);

  e.peo = 0.0;
  e.pinco = 0.0;
  e.plo = 0.0;
  e.pgho = 0.0;
  e.pho = 0.0;

  c.day = epoch_days_1950 + 18261.5 + tc / 1440.0;
  const double xnodce = std::fmod(4.5236020 - 9.2422029e-4 * c.day, kTwoPi);
  const double stem = std::sin(xnodce);
  const double ctem = std::cos(xnodce);
  const double zcosil = 0.91375164 - 0.03568096 * ctem;
  const double zsinil = std::sqrt(1.0 - zcosil * zcosil);
  const double zsinhl = 0.089683511 * stem / zsinil;
  const double zcoshl = std::sqrt(1.0 - zsinhl * zsinhl);
  c.gam = 5.8351514 + 0.0019443680 * c.day;
  double zx = 0.39785416 * stem / zsinil;
  const double zy = zcoshl * ctem + 0.91744867 * zsinhl * stem;
  zx = std::atan2(zx, zy);
  zx = c.gam + zx - xnodce;
  const double zcosgl = std::cos(zx);
  const double zsingl = std::sin(zx);

  double zcosg = kZcosgs;
  double zsing = kZsings;
  double zcosi = kZcosis;
  double zsini = kZsinis;
  double zcosh = c.cnodm;
  double zsinh = c.snodm;
  double cc = kC1ss;
  const double xnoi = 1.0 / c.nm;

  // Pass 1 is solar, pass 2 lunar; the body is identical and only the ecliptic
  // orientation and coefficient change, which is why it is a loop.
  for (int lsflg = 1; lsflg <= 2; ++lsflg) {
    const double a1 = zcosg * zcosh + zsing * zcosi * zsinh;
    const double a3 = -zsing * zcosh + zcosg * zcosi * zsinh;
    const double a7 = -zcosg * zsinh + zsing * zcosi * zcosh;
    const double a8 = zsing * zsini;
    const double a9 = zsing * zsinh + zcosg * zcosi * zcosh;
    const double a10 = zcosg * zsini;
    const double a2 = c.cosim * a7 + c.sinim * a8;
    const double a4 = c.cosim * a9 + c.sinim * a10;
    const double a5 = -c.sinim * a7 + c.cosim * a8;
    const double a6 = -c.sinim * a9 + c.cosim * a10;

    const double x1 = a1 * c.cosomm + a2 * c.sinomm;
    const double x2 = a3 * c.cosomm + a4 * c.sinomm;
    const double x3 = -a1 * c.sinomm + a2 * c.cosomm;
    const double x4 = -a3 * c.sinomm + a4 * c.cosomm;
    const double x5 = a5 * c.sinomm;
    const double x6 = a6 * c.sinomm;
    const double x7 = a5 * c.cosomm;
    const double x8 = a6 * c.cosomm;

    c.z31 = 12.0 * x1 * x1 - 3.0 * x3 * x3;
    c.z32 = 24.0 * x1 * x2 - 6.0 * x3 * x4;
    c.z33 = 12.0 * x2 * x2 - 3.0 * x4 * x4;
    c.z1 = 3.0 * (a1 * a1 + a2 * a2) + c.z31 * c.emsq;
    c.z2 = 6.0 * (a1 * a3 + a2 * a4) + c.z32 * c.emsq;
    c.z3 = 3.0 * (a3 * a3 + a4 * a4) + c.z33 * c.emsq;
    c.z11 = -6.0 * a1 * a5 + c.emsq * (-24.0 * x1 * x7 - 6.0 * x3 * x5);
    c.z12 = -6.0 * (a1 * a6 + a3 * a5) +
            c.emsq * (-24.0 * (x2 * x7 + x1 * x8) - 6.0 * (x3 * x6 + x4 * x5));
    c.z13 = -6.0 * a3 * a6 + c.emsq * (-24.0 * x2 * x8 - 6.0 * x4 * x6);
    c.z21 = 6.0 * a2 * a5 + c.emsq * (24.0 * x1 * x5 - 6.0 * x3 * x7);
    c.z22 = 6.0 * (a4 * a5 + a2 * a6) +
            c.emsq * (24.0 * (x2 * x5 + x1 * x6) - 6.0 * (x4 * x7 + x3 * x8));
    c.z23 = 6.0 * a4 * a6 + c.emsq * (24.0 * x2 * x6 - 6.0 * x4 * x8);
    c.z1 = c.z1 + c.z1 + betasq * c.z31;
    c.z2 = c.z2 + c.z2 + betasq * c.z32;
    c.z3 = c.z3 + c.z3 + betasq * c.z33;
    c.s3 = cc * xnoi;
    c.s2 = -0.5 * c.s3 / c.rtemsq;
    c.s4 = c.s3 * c.rtemsq;
    c.s1 = -15.0 * c.em * c.s4;
    c.s5 = x1 * x3 + x2 * x4;
    c.s6 = x2 * x3 + x1 * x4;
    c.s7 = x2 * x4 - x1 * x3;

    if (lsflg == 1) {
      c.ss1 = c.s1;
      c.ss2 = c.s2;
      c.ss3 = c.s3;
      c.ss4 = c.s4;
      c.ss5 = c.s5;
      c.ss6 = c.s6;
      c.ss7 = c.s7;
      c.sz1 = c.z1;
      c.sz2 = c.z2;
      c.sz3 = c.z3;
      c.sz11 = c.z11;
      c.sz12 = c.z12;
      c.sz13 = c.z13;
      c.sz21 = c.z21;
      c.sz22 = c.z22;
      c.sz23 = c.z23;
      c.sz31 = c.z31;
      c.sz32 = c.z32;
      c.sz33 = c.z33;
      zcosg = zcosgl;
      zsing = zsingl;
      zcosi = zcosil;
      zsini = zsinil;
      zcosh = zcoshl * c.cnodm + zsinhl * c.snodm;
      zsinh = c.snodm * zcoshl - c.cnodm * zsinhl;
      cc = kC1l;
    }
  }

  e.zmol = std::fmod(4.7199672 + 0.22997150 * c.day - c.gam, kTwoPi);
  e.zmos = std::fmod(6.2565837 + 0.017201977 * c.day, kTwoPi);

  // Solar periodic coefficients.
  e.se2 = 2.0 * c.ss1 * c.ss6;
  e.se3 = 2.0 * c.ss1 * c.ss7;
  e.si2 = 2.0 * c.ss2 * c.sz12;
  e.si3 = 2.0 * c.ss2 * (c.sz13 - c.sz11);
  e.sl2 = -2.0 * c.ss3 * c.sz2;
  e.sl3 = -2.0 * c.ss3 * (c.sz3 - c.sz1);
  e.sl4 = -2.0 * c.ss3 * (-21.0 - 9.0 * c.emsq) * kZes;
  e.sgh2 = 2.0 * c.ss4 * c.sz32;
  e.sgh3 = 2.0 * c.ss4 * (c.sz33 - c.sz31);
  e.sgh4 = -18.0 * c.ss4 * kZes;
  e.sh2 = -2.0 * c.ss2 * c.sz22;
  e.sh3 = -2.0 * c.ss2 * (c.sz23 - c.sz21);

  // Lunar periodic coefficients.
  e.ee2 = 2.0 * c.s1 * c.s6;
  e.e3 = 2.0 * c.s1 * c.s7;
  e.xi2 = 2.0 * c.s2 * c.z12;
  e.xi3 = 2.0 * c.s2 * (c.z13 - c.z11);
  e.xl2 = -2.0 * c.s3 * c.z2;
  e.xl3 = -2.0 * c.s3 * (c.z3 - c.z1);
  e.xl4 = -2.0 * c.s3 * (-21.0 - 9.0 * c.emsq) * kZel;
  e.xgh2 = 2.0 * c.s4 * c.z32;
  e.xgh3 = 2.0 * c.s4 * (c.z33 - c.z31);
  e.xgh4 = -18.0 * c.s4 * kZel;
  e.xh2 = -2.0 * c.s2 * c.z22;
  e.xh3 = -2.0 * c.s2 * (c.z23 - c.z21);

  return c;
}

void deepSpacePeriodics(const Sgp4Elements& e, double t, bool init, double& ep, double& inclp,
                        double& nodep, double& argpp, double& mp) {
  // Solar.
  double zm = init ? e.zmos : (e.zmos + kZns * t);
  double zf = zm + 2.0 * kZes * std::sin(zm);
  double sinzf = std::sin(zf);
  double f2 = 0.5 * sinzf * sinzf - 0.25;
  double f3 = -0.5 * sinzf * std::cos(zf);
  const double ses = e.se2 * f2 + e.se3 * f3;
  const double sis = e.si2 * f2 + e.si3 * f3;
  const double sls = e.sl2 * f2 + e.sl3 * f3 + e.sl4 * sinzf;
  const double sghs = e.sgh2 * f2 + e.sgh3 * f3 + e.sgh4 * sinzf;
  const double shs = e.sh2 * f2 + e.sh3 * f3;

  // Lunar.
  zm = init ? e.zmol : (e.zmol + kZnl * t);
  zf = zm + 2.0 * kZel * std::sin(zm);
  sinzf = std::sin(zf);
  f2 = 0.5 * sinzf * sinzf - 0.25;
  f3 = -0.5 * sinzf * std::cos(zf);
  const double sel = e.ee2 * f2 + e.e3 * f3;
  const double sil = e.xi2 * f2 + e.xi3 * f3;
  const double sll = e.xl2 * f2 + e.xl3 * f3 + e.xl4 * sinzf;
  const double sghl = e.xgh2 * f2 + e.xgh3 * f3 + e.xgh4 * sinzf;
  const double shll = e.xh2 * f2 + e.xh3 * f3;

  double pe = ses + sel;
  double pinc = sis + sil;
  double pl = sls + sll;
  double pgh = sghs + sghl;
  double ph = shs + shll;

  if (init) {
    // At epoch the elements are the fitted ones; applying the periodics here
    // would move them off the fit. The reference calls this path for symmetry
    // and discards the result, and so do we.
    return;
  }

  pe -= e.peo;
  pinc -= e.pinco;
  pl -= e.plo;
  pgh -= e.pgho;
  ph -= e.pho;

  inclp = inclp + pinc;
  ep = ep + pe;
  const double sinip = std::sin(inclp);
  const double cosip = std::cos(inclp);

  // **Lyddane's choice.** Above 0.2 rad the straightforward formulation is
  // fine; below it, dividing the node correction by sin(i) is a division by a
  // small number, so the reference switches to a vector formulation that stays
  // conditioned. The threshold is empirical and part of the theory: moving it
  // changes answers for near-equatorial orbits, several of which are in the
  // verification set precisely to pin this branch.
  if (inclp >= 0.2) {
    ph = ph / sinip;
    pgh = pgh - cosip * ph;
    argpp = argpp + pgh;
    nodep = nodep + ph;
    mp = mp + pl;
  } else {
    const double sinop = std::sin(nodep);
    const double cosop = std::cos(nodep);
    double alfdp = sinip * sinop;
    double betdp = sinip * cosop;
    const double dalf = ph * cosop + pinc * cosip * sinop;
    const double dbet = -ph * sinop + pinc * cosip * cosop;
    alfdp = alfdp + dalf;
    betdp = betdp + dbet;
    nodep = std::fmod(nodep, kTwoPi);
    // AFSPC mode keeps the node positive here; the improved mode does not. This
    // is one of the two places the operation mode actually changes an answer.
    if (nodep < 0.0 && e.opsmode == Sgp4OpsMode::kAfspc) {
      nodep = nodep + kTwoPi;
    }
    double xls = mp + argpp + cosip * nodep;
    const double dls = pl + pgh - pinc * nodep * sinip;
    xls = xls + dls;
    const double xnoh = nodep;
    nodep = std::atan2(alfdp, betdp);
    if (nodep < 0.0 && e.opsmode == Sgp4OpsMode::kAfspc) {
      nodep = nodep + kTwoPi;
    }
    if (std::fabs(xnoh - nodep) > kPi) {
      if (nodep < xnoh) {
        nodep = nodep + kTwoPi;
      } else {
        nodep = nodep - kTwoPi;
      }
    }
    mp = mp + pl;
    argpp = xls - mp - cosip * nodep;
  }
}

void deepSpaceInit(Sgp4Elements& e, const DeepSpaceCommon& c, double epoch_days_1950, double argpo,
                   double& em, double& argpm, double& inclm, double& mm, double& nm,
                   double& nodem) {
  (void)epoch_days_1950;
  constexpr double kQ22 = 1.7891679e-6;
  constexpr double kQ31 = 2.1460748e-6;
  constexpr double kQ33 = 2.2123015e-7;
  constexpr double kRoot22 = 1.7891679e-6;
  constexpr double kRoot44 = 7.3636953e-9;
  constexpr double kRoot54 = 2.1765803e-9;
  constexpr double kRoot32 = 3.7393792e-7;
  constexpr double kRoot52 = 1.1428639e-7;
  constexpr double kX2o3 = 2.0 / 3.0;

  e.resonant = false;
  e.synchronous = false;
  // Resonance selection. The 24-hour band is a 1:1 commensurability with the
  // Earth's rotation; the 12-hour band additionally requires e >= 0.5, because
  // below that the tesseral terms do not lock.
  if (nm < 0.0052359877 && nm > 0.0034906585) {
    e.resonant = true;
    e.synchronous = true;
  }
  if (nm >= 8.26e-3 && nm <= 9.24e-3 && em >= 0.5) {
    e.resonant = true;
    e.synchronous = false;
  }

  // Secular rates from the lunar-solar terms.
  const double ses = c.ss1 * kZns * c.ss5;
  const double sis = c.ss2 * kZns * (c.sz11 + c.sz13);
  const double sls = -kZns * c.ss3 * (c.sz1 + c.sz3 - 14.0 - 6.0 * c.emsq);
  const double sghs = c.ss4 * kZns * (c.sz31 + c.sz33 - 6.0);
  double shs = -kZns * c.ss2 * (c.sz21 + c.sz23);
  // Near-equatorial: the node correction is not meaningful and is dropped
  // rather than divided by a vanishing sine.
  if (inclm < 5.2359877e-2 || inclm > kPi - 5.2359877e-2) {
    shs = 0.0;
  }
  if (c.sinim != 0.0) {
    shs = shs / c.sinim;
  }
  const double sgs = sghs - c.cosim * shs;

  e.dedt = ses + c.s1 * kZnl * c.s5;
  e.didt = sis + c.s2 * kZnl * (c.z11 + c.z13);
  e.dmdt = sls - kZnl * c.s3 * (c.z1 + c.z3 - 14.0 - 6.0 * c.emsq);
  const double sghl = c.s4 * kZnl * (c.z31 + c.z33 - 6.0);
  double shll = -kZnl * c.s2 * (c.z21 + c.z23);
  if (inclm < 5.2359877e-2 || inclm > kPi - 5.2359877e-2) {
    shll = 0.0;
  }
  e.domdt = sgs + sghl;
  e.dnodt = shs;
  if (c.sinim != 0.0) {
    e.domdt = e.domdt - c.cosim / c.sinim * shll;
    e.dnodt = e.dnodt + shll / c.sinim;
  }

  const double theta = std::fmod(e.gsto_deep, kTwoPi);
  // t = 0 at initialisation, so the secular advances below are all zero; they
  // are written out anyway so this function reads the same as the reference.
  em = em + e.dedt * 0.0;
  inclm = inclm + e.didt * 0.0;
  argpm = argpm + e.domdt * 0.0;
  nodem = nodem + e.dnodt * 0.0;
  mm = mm + e.dmdt * 0.0;

  if (!e.resonant) {
    return;
  }

  const double aonv = std::pow(nm / xke(), kX2o3);

  if (!e.synchronous) {
    // --- 12-hour (2:1) resonance ------------------------------------------
    const double cosisq = c.cosim * c.cosim;
    const double emo = em;
    const double emsqo = c.emsq;
    // The g-coefficients are fitted against the *TLE's* eccentricity, not the
    // secularly advanced one, so the reference swaps them in and back out.
    const double em_fit = e.ecco;
    const double emsq_fit = e.ecco * e.ecco;
    const double eoc = em_fit * emsq_fit;

    const double g201 = -0.306 - (em_fit - 0.64) * 0.440;
    double g211 = 0.0;
    double g310 = 0.0;
    double g322 = 0.0;
    double g410 = 0.0;
    double g422 = 0.0;
    double g520 = 0.0;
    if (em_fit <= 0.65) {
      g211 = 3.616 - 13.2470 * em_fit + 16.2900 * emsq_fit;
      g310 = -19.302 + 117.3900 * em_fit - 228.4190 * emsq_fit + 156.5910 * eoc;
      g322 = -18.9068 + 109.7927 * em_fit - 214.6334 * emsq_fit + 146.5816 * eoc;
      g410 = -41.122 + 242.6940 * em_fit - 471.0940 * emsq_fit + 313.9530 * eoc;
      g422 = -146.407 + 841.8800 * em_fit - 1629.014 * emsq_fit + 1083.4350 * eoc;
      g520 = -532.114 + 3017.977 * em_fit - 5740.032 * emsq_fit + 3708.2760 * eoc;
    } else {
      g211 = -72.099 + 331.819 * em_fit - 508.738 * emsq_fit + 266.724 * eoc;
      g310 = -346.844 + 1582.851 * em_fit - 2415.925 * emsq_fit + 1246.113 * eoc;
      g322 = -342.585 + 1554.908 * em_fit - 2366.899 * emsq_fit + 1215.972 * eoc;
      g410 = -1052.797 + 4758.686 * em_fit - 7193.992 * emsq_fit + 3651.957 * eoc;
      g422 = -3581.690 + 16178.110 * em_fit - 24462.770 * emsq_fit + 12422.520 * eoc;
      if (em_fit > 0.715) {
        g520 = -5149.66 + 29936.92 * em_fit - 54087.36 * emsq_fit + 31324.56 * eoc;
      } else {
        g520 = 1464.74 - 4664.75 * em_fit + 3763.64 * emsq_fit;
      }
    }
    double g533 = 0.0;
    double g521 = 0.0;
    double g532 = 0.0;
    if (em_fit < 0.7) {
      g533 = -919.22770 + 4988.6100 * em_fit - 9064.7700 * emsq_fit + 5542.21 * eoc;
      g521 = -822.71072 + 4568.6173 * em_fit - 8491.4146 * emsq_fit + 5337.524 * eoc;
      g532 = -853.66600 + 4690.2500 * em_fit - 8624.7700 * emsq_fit + 5341.4 * eoc;
    } else {
      g533 = -37995.780 + 161616.52 * em_fit - 229838.20 * emsq_fit + 109377.94 * eoc;
      g521 = -51752.104 + 218913.95 * em_fit - 309468.16 * emsq_fit + 146349.42 * eoc;
      g532 = -40023.880 + 170470.89 * em_fit - 242699.48 * emsq_fit + 115605.82 * eoc;
    }

    const double sini2 = c.sinim * c.sinim;
    const double f220 = 0.75 * (1.0 + 2.0 * c.cosim + cosisq);
    const double f221 = 1.5 * sini2;
    const double f321 = 1.875 * c.sinim * (1.0 - 2.0 * c.cosim - 3.0 * cosisq);
    const double f322 = -1.875 * c.sinim * (1.0 + 2.0 * c.cosim - 3.0 * cosisq);
    const double f441 = 35.0 * sini2 * f220;
    const double f442 = 39.3750 * sini2 * sini2;
    const double f522 = 9.84375 * c.sinim *
                        (sini2 * (1.0 - 2.0 * c.cosim - 5.0 * cosisq) +
                         0.33333333 * (-2.0 + 4.0 * c.cosim + 6.0 * cosisq));
    const double f523 = c.sinim * (4.92187512 * sini2 * (-2.0 - 4.0 * c.cosim + 10.0 * cosisq) +
                                   6.56250012 * (1.0 + 2.0 * c.cosim - 3.0 * cosisq));
    const double f542 = 29.53125 * c.sinim *
                        (2.0 - 8.0 * c.cosim + cosisq * (-12.0 + 8.0 * c.cosim + 10.0 * cosisq));
    const double f543 = 29.53125 * c.sinim *
                        (-2.0 - 8.0 * c.cosim + cosisq * (12.0 + 8.0 * c.cosim - 10.0 * cosisq));

    const double xno2 = nm * nm;
    const double ainv2 = aonv * aonv;
    double temp1 = 3.0 * xno2 * ainv2;
    double temp = temp1 * kRoot22;
    e.d2201 = temp * f220 * g201;
    e.d2211 = temp * f221 * g211;
    temp1 = temp1 * aonv;
    temp = temp1 * kRoot32;
    e.d3210 = temp * f321 * g310;
    e.d3222 = temp * f322 * g322;
    temp1 = temp1 * aonv;
    temp = 2.0 * temp1 * kRoot44;
    e.d4410 = temp * f441 * g410;
    e.d4422 = temp * f442 * g422;
    temp1 = temp1 * aonv;
    temp = temp1 * kRoot52;
    e.d5220 = temp * f522 * g520;
    e.d5232 = temp * f523 * g532;
    temp = 2.0 * temp1 * kRoot54;
    e.d5421 = temp * f542 * g521;
    e.d5433 = temp * f543 * g533;
    e.xlamo = std::fmod(e.mo + e.nodeo + e.nodeo - theta - theta, kTwoPi);
    e.xfact = e.mdot + e.dmdt + 2.0 * (e.nodedot + e.dnodt - kRptim) - e.no_unkozai;
    (void)emo;
    (void)emsqo;
  } else {
    // --- 24-hour (1:1) synchronous resonance ------------------------------
    const double g200 = 1.0 + c.emsq * (-2.5 + 0.8125 * c.emsq);
    const double g310 = 1.0 + 2.0 * c.emsq;
    const double g300 = 1.0 + c.emsq * (-6.0 + 6.60937 * c.emsq);
    const double f220 = 0.75 * (1.0 + c.cosim) * (1.0 + c.cosim);
    const double f311 = 0.9375 * c.sinim * c.sinim * (1.0 + 3.0 * c.cosim) - 0.75 * (1.0 + c.cosim);
    double f330 = 1.0 + c.cosim;
    f330 = 1.875 * f330 * f330 * f330;
    e.del1 = 3.0 * nm * nm * aonv * aonv;
    e.del2 = 2.0 * e.del1 * f220 * g200 * kQ22;
    e.del3 = 3.0 * e.del1 * f330 * g300 * kQ33 * aonv;
    e.del1 = e.del1 * f311 * g310 * kQ31 * aonv;
    e.xlamo = std::fmod(e.mo + e.nodeo + argpo - theta, kTwoPi);
    const double xpidot = e.argpdot + e.nodedot;
    e.xfact = e.mdot + xpidot - kRptim + e.dmdt + e.domdt + e.dnodt - e.no_unkozai;
  }

  e.xli = e.xlamo;
  e.xni = e.no_unkozai;
  e.atime = 0.0;
  nm = e.no_unkozai;
}

void deepSpaceSecular(const Sgp4Elements& e, double t, double& em, double& argpm, double& inclm,
                      double& mm, double& nm, double& nodem, double& dndt) {
  constexpr double kFasx2 = 0.13130908;
  constexpr double kFasx4 = 2.8843198;
  constexpr double kFasx6 = 0.37448087;
  constexpr double kG22 = 5.7686396;
  constexpr double kG32 = 0.95240898;
  constexpr double kG44 = 1.8014998;
  constexpr double kG52 = 1.0508330;
  constexpr double kG54 = 4.4108898;
  constexpr double kStep = 720.0;
  constexpr double kStep2 = 259200.0;  // 0.5 * step^2

  dndt = 0.0;
  const double theta = std::fmod(e.gsto_deep + t * kRptim, kTwoPi);
  em = em + e.dedt * t;
  inclm = inclm + e.didt * t;
  argpm = argpm + e.domdt * t;
  nodem = nodem + e.dnodt * t;
  mm = mm + e.dmdt * t;

  if (!e.resonant) {
    return;
  }

  // Resonance integration. **Restarted from epoch every call** — the reference
  // caches `atime`/`xli`/`xni` between calls and resumes, which makes its answer
  // depend on the order requests arrived in. Restarting reproduces the cached
  // path exactly (the recurrence is fixed-step, so 0->720->1440 and a resumed
  // 720->1440 give identical iterates) while making `propagate` const and
  // order-independent. The cost is bounded: |t|/720 steps.
  double atime = 0.0;
  double xli = e.xlamo;
  double xni = e.no_unkozai;
  const double delt = (t > 0.0) ? kStep : -kStep;

  double xndt = 0.0;
  double xldot = 0.0;
  double xnddt = 0.0;
  double ft = 0.0;

  for (;;) {
    if (!e.synchronous) {
      const double xomi = e.argpo + e.argpdot * atime;
      const double x2omi = xomi + xomi;
      const double x2li = xli + xli;
      xndt = e.d2201 * std::sin(x2omi + xli - kG22) + e.d2211 * std::sin(xli - kG22) +
             e.d3210 * std::sin(xomi + xli - kG32) + e.d3222 * std::sin(-xomi + xli - kG32) +
             e.d4410 * std::sin(x2omi + x2li - kG44) + e.d4422 * std::sin(x2li - kG44) +
             e.d5220 * std::sin(xomi + xli - kG52) + e.d5232 * std::sin(-xomi + xli - kG52) +
             e.d5421 * std::sin(xomi + x2li - kG54) + e.d5433 * std::sin(-xomi + x2li - kG54);
      xldot = xni + e.xfact;
      xnddt =
          e.d2201 * std::cos(x2omi + xli - kG22) + e.d2211 * std::cos(xli - kG22) +
          e.d3210 * std::cos(xomi + xli - kG32) + e.d3222 * std::cos(-xomi + xli - kG32) +
          e.d5220 * std::cos(xomi + xli - kG52) + e.d5232 * std::cos(-xomi + xli - kG52) +
          2.0 * (e.d4410 * std::cos(x2omi + x2li - kG44) + e.d4422 * std::cos(x2li - kG44) +
                 e.d5421 * std::cos(xomi + x2li - kG54) + e.d5433 * std::cos(-xomi + x2li - kG54));
      xnddt = xnddt * xldot;
    } else {
      xndt = e.del1 * std::sin(xli - kFasx2) + e.del2 * std::sin(2.0 * (xli - kFasx4)) +
             e.del3 * std::sin(3.0 * (xli - kFasx6));
      xldot = xni + e.xfact;
      xnddt = e.del1 * std::cos(xli - kFasx2) + 2.0 * e.del2 * std::cos(2.0 * (xli - kFasx4)) +
              3.0 * e.del3 * std::cos(3.0 * (xli - kFasx6));
      xnddt = xnddt * xldot;
    }

    if (std::fabs(t - atime) < kStep) {
      ft = t - atime;
      break;
    }
    xli = xli + xldot * delt + xndt * kStep2;
    xni = xni + xndt * delt + xnddt * kStep2;
    atime = atime + delt;
  }

  nm = xni + xndt * ft + xnddt * ft * ft * 0.5;
  const double xl = xli + xldot * ft + xndt * ft * ft * 0.5;
  if (!e.synchronous) {
    mm = xl - 2.0 * nodem + 2.0 * theta;
  } else {
    mm = xl - nodem - argpm + theta;
  }
  dndt = nm - e.no_unkozai;
  nm = e.no_unkozai + dndt;
}

}  // namespace polaris::gnc::sgp4_detail
