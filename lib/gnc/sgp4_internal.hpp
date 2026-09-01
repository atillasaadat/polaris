#ifndef POLARIS_GNC_SGP4_INTERNAL_HPP
#define POLARIS_GNC_SGP4_INTERNAL_HPP

/// @file
/// @brief Shared constants and the deep-space (SDP4) seam, internal to the SGP4
/// implementation (design doc §8.3). Not part of the public API.
///
/// Split from `sgp4.cpp` only because the deep-space half is a separable body of
/// work of comparable size (`sgp4_deep_space.cpp`); the two together are one
/// algorithm and neither is meaningful alone.

#include "gnc/sgp4.hpp"

namespace polaris::gnc::sgp4_detail {

/// WGS-72 gravitational constants. **Not WGS-84** — see the note in `sgp4.hpp`:
/// TLEs are fitted with these, so evaluating them with the vehicle's WGS-84 set
/// would introduce an inconsistency the theory cannot absorb. Values from
/// [vallado2006revisiting] `getgravconst(wgs72)`.
inline constexpr double kMu = 398600.8;             ///< [km^3/s^2]
inline constexpr double kEarthRadiusKm = 6378.135;  ///< [km]
inline constexpr double kJ2 = 0.001082616;
inline constexpr double kJ3 = -0.00000253881;
inline constexpr double kJ4 = -0.00000165597;
inline constexpr double kJ3OverJ2 = kJ3 / kJ2;

inline constexpr double kPi = 3.14159265358979323846;
inline constexpr double kTwoPi = 2.0 * kPi;
inline constexpr double kDegToRad = kPi / 180.0;

/// Earth's rotation rate in the theory's units [rad/min]. This is the value the
/// theory was fitted with, not a modern sidereal rate.
inline constexpr double kEarthRotationRadPerMin = 1.19459e-5;

/// `xke` — sqrt(GM) in earth-radii^{3/2} per minute. Written as the reference
/// derives it so the WGS-72 dependence stays visible.
double xke();

/// One minute in canonical time units.
double tumin();

// --- Deep-space entry points (sgp4_deep_space.cpp) -------------------------

/// Deep-space common quantities, evaluated once at initialisation.
/// Fills the lunar-solar coefficient block of @p e and returns the intermediate
/// terms `dsinit` needs.
struct DeepSpaceCommon {
  double sinim{0.0}, cosim{0.0}, sinomm{0.0}, cosomm{0.0};
  double snodm{0.0}, cnodm{0.0}, day{0.0};
  double em{0.0}, emsq{0.0}, gam{0.0};
  double rtemsq{0.0};
  double s1{0.0}, s2{0.0}, s3{0.0}, s4{0.0}, s5{0.0}, s6{0.0}, s7{0.0};
  double ss1{0.0}, ss2{0.0}, ss3{0.0}, ss4{0.0}, ss5{0.0}, ss6{0.0}, ss7{0.0};
  double sz1{0.0}, sz2{0.0}, sz3{0.0};
  double sz11{0.0}, sz12{0.0}, sz13{0.0};
  double sz21{0.0}, sz22{0.0}, sz23{0.0};
  double sz31{0.0}, sz32{0.0}, sz33{0.0};
  double z1{0.0}, z2{0.0}, z3{0.0};
  double z11{0.0}, z12{0.0}, z13{0.0};
  double z21{0.0}, z22{0.0}, z23{0.0};
  double z31{0.0}, z32{0.0}, z33{0.0};
  double nm{0.0};
};

DeepSpaceCommon deepSpaceCommon(Sgp4Elements& e, double epoch_days_1950, double tc, double argpp,
                                double nodep);

/// Deep-space initialisation: resonance selection and the secular rates.
void deepSpaceInit(Sgp4Elements& e, const DeepSpaceCommon& c, double epoch_days_1950, double argpo,
                   double& em, double& argpm, double& inclm, double& mm, double& nm, double& nodem);

/// Lunar-solar periodics. Applied both at initialisation (@p init true, which
/// suppresses the Lyddane-choice correction) and every propagation step.
void deepSpacePeriodics(const Sgp4Elements& e, double t, bool init, double& ep, double& inclp,
                        double& nodep, double& argpp, double& mp);

/// Deep-space secular update, including the resonance integration.
///
/// Unlike the reference, this **restarts the resonance integration from epoch on
/// every call** rather than caching `atime`/`xli`/`xni` across calls, so the
/// result for a given @p t never depends on the order calls were made in. See
/// the note in `sgp4.hpp`.
void deepSpaceSecular(const Sgp4Elements& e, double t, double& em, double& argpm, double& inclm,
                      double& mm, double& nm, double& nodem, double& dndt);

}  // namespace polaris::gnc::sgp4_detail

#endif  // POLARIS_GNC_SGP4_INTERNAL_HPP
