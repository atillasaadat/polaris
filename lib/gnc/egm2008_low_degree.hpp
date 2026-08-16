#ifndef POLARIS_GNC_EGM2008_LOW_DEGREE_HPP
#define POLARIS_GNC_EGM2008_LOW_DEGREE_HPP

/// @file
/// @brief EGM2008 geopotential truncated to degree/order 8,
/// UNNORMALIZED, for the onboard orbit filter's force model (design doc §8.3).
///
/// GENERATED — do not edit. Regenerate with:
///
///     PYTHONPATH=tools uv run python -m gravity cxx-header
///       --input tests/golden/EGM2008_to200.gfc
///       --max-degree 8
///       --out lib/gnc/egm2008_low_degree.hpp
///
/// (one shell command; the arguments are split across lines here because a
/// trailing backslash inside a `//` comment trips -Wcomment, and this file is
/// compiled with -Werror.)
///
/// Derived from the committed `tests/golden/EGM2008_to200.gfc` (design doc §3.7: the reference
/// datum stays the native ICGEM `.gfc`; this is a derivation *from* it, not a
/// substitute for it). `tests/tools/test_gravity_cxxtable.py` regenerates this
/// file and asserts the coefficients match, so the two cannot drift apart.
///
/// The coefficients are **unnormalized** — de-normalized on the ground by
/// `tools/gravity/cxxtable.py` — because the flight evaluator
/// (@ref polaris::gnc::geopotentialAcceleration) uses the unnormalized
/// Cunningham V/W recursion (Montenbruck & Gill §3.2.4 [montenbruck2000]).
/// `kC[0][0] = 1` is the point-mass term.
///
/// `kGm` and `kReferenceRadius` are the **model's own** values from the `.gfc`
/// header, not WGS84's. They must be used together with these coefficients:
/// the coefficients were solved with this GM and this radius, and pairing them
/// with a generic constant reintroduces an error comparable to the truncation
/// the table exists to remove.

#include <cstddef>

namespace polaris::gnc::egm2008 {

/// Truncation degree and order of the table below.
inline constexpr int kMaxDegree = 8;

/// Gravitational parameter the coefficients were solved with [m^3/s^2].
inline constexpr double kGm = 398600441500000.0;

/// Reference radius the coefficients are scaled to [m].
inline constexpr double kReferenceRadius = 6378136.3;

/// Unnormalized C_nm, row n, column m; entries with m > n are zero.
inline constexpr double kC[kMaxDegree + 1][kMaxDegree + 1] = {
    {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},  // n = 0
    {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},  // n = 1
    {-0.0010826261738522225, -2.667394752374836e-10, 1.5746153257229176e-06, 0.0, 0.0, 0.0, 0.0,
     0.0, 0.0},  // n = 2
    {2.5324105185677225e-06, 2.1931496313133285e-06, 3.0904390039164874e-07, 1.0058351340882277e-07,
     0.0, 0.0, 0.0, 0.0, 0.0},  // n = 3
    {1.6198975999169733e-06, -5.086435604395838e-07, 7.837454574045519e-08, 5.921501776396661e-08,
     -3.9832042487318765e-09, 0.0, 0.0, 0.0, 0.0},  // n = 4
    {2.277535907308362e-07, -5.38824899516695e-08, 1.0552886671559868e-07, -1.4926487654663265e-08,
     -2.299511403504223e-09, 4.304280419791827e-10, 0.0, 0.0, 0.0},  // n = 5
    {-5.406665762838133e-07, -5.973432980334215e-08, 6.052084606346259e-09, 1.1869148534366206e-09,
     -3.2564075033060574e-10, -2.1562081933978544e-10, 2.2064784920641347e-12, 0.0, 0.0},  // n = 6
    {3.50551795713742e-07, 2.0558863962959296e-07, 3.2909423902140844e-08, 3.5279331698276798e-09,
     -5.839578622360583e-10, 5.831682438273027e-13, -2.490410824239854e-11, 2.7964277989424365e-14,
     0.0},  // n = 7
    {2.0399312592988433e-07, 1.5915736860899883e-08, 6.5719178905083104e-09, -1.9587706212708e-10,
     -3.189389549747225e-10, -4.6518686490078914e-12, -1.842311139500083e-12, 3.429469812357841e-13,
     -1.5809971799002101e-13},  // n = 8
};

/// Unnormalized S_nm, row n, column m; entries with m > n (and m = 0) are zero.
inline constexpr double kS[kMaxDegree + 1][kMaxDegree + 1] = {
    {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},                                       // n = 0
    {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},                                       // n = 1
    {0.0, 1.7872706485240428e-09, -9.03872789196567e-07, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},  // n = 2
    {0.0, 2.680870894008977e-07, -2.1143062093348251e-07, 1.9722158183571826e-07, 0.0, 0.0, 0.0,
     0.0, 0.0},  // n = 3
    {0.0, -4.492654321438082e-07, 1.4813503724885995e-07, -1.2009461262296094e-08,
     6.524672871876875e-09, 0.0, 0.0, 0.0, 0.0},  // n = 4
    {0.0, -8.081347491204561e-08, -5.232977296929693e-08, -7.100917272237693e-09,
     3.8781150374611435e-10, -1.648171932690549e-09, 0.0, 0.0, 0.0},  // n = 5
    {0.0, 2.085973408290852e-08, -4.650063963542403e-08, 1.8561001441516656e-10,
     -1.7845687898590347e-09, -4.329851457129141e-10, -5.530528350756143e-11, 0.0, 0.0},  // n = 6
    {0.0, 6.96250561212826e-08, 9.262715849636068e-09, -3.0583074109008355e-09,
     -2.634417371888356e-10, 6.345170628833404e-12, 1.0536278832770712e-11, 4.471954729352735e-13,
     0.0},  // n = 7
    {0.0, 4.047340405073659e-08, 5.361763963487449e-09, -8.690975069756568e-10,
     9.111232566838356e-11, 1.6145695868698155e-11, 8.628471804213623e-12, 3.8175910346257523e-13,
     1.5367516320040298e-13},  // n = 8
};

}  // namespace polaris::gnc::egm2008

#endif  // POLARIS_GNC_EGM2008_LOW_DEGREE_HPP
