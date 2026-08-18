// Covariance propagation, checked against things that are true independently of
// this filter.
//
// `orbit_od_test.cpp` covers the filter's behaviour and pins its Jacobian at a
// point against the closed-form two-body gravity gradient. What neither it nor
// the Monte Carlo campaign establishes is whether the *propagated covariance*
// is right over an arc. The campaign's NEES normalises the error by the very
// covariance under test, so a filter wrong about both in the same direction
// passes it; and the header's argument that the `O((‖F‖h)³)` truncation of
// `Φ = I + Fh + ½F²h²` is negligible is an argument about one sub-step, not a
// measurement of what ~300 compositions of it do across a coast horizon.
//
// Two independent references, neither needing an external tool:
//
// **Liouville's theorem.** Two-body and J2 are conservative, so the true state
// transition is symplectic and `det Φ = 1` exactly — phase-space volume is
// preserved. Seeded with `P₀ = I` and **zero process noise**, the filter's own
// covariance is `P(t) = ΦΦᵀ`, so `det P(t)` must be 1 and any departure is
// pure truncation error in the covariance path. No reference implementation is
// involved at all, which is what makes this the sharper of the two: it cannot
// be fooled by a reference that shares an assumption with the code.
//
// **Finite differences of the trajectory.** `Φ` is by definition the sensitivity
// of the propagated state to its initial condition, so differencing the
// propagated trajectory about the nominal recovers it. That reference is
// independent *of the covariance path* — it exercises the RK4 state integration,
// which is separately validated to sub-metre against GMAT in
// `gmat_propagation_golden_test.cpp` — while the covariance path is analytic
// Jacobian → truncated Φ → composition, sharing none of that code. It is also
// the reason this is preferred to hand-writing Battin's closed-form two-body
// STM: a closed form only covers two-body and risks a buggy reference, whereas
// this works with J2 on and reuses code already under test.
//
// Both run with `accel_psd_m2_per_s3 = 0`. That is not a shortcut: `Q` is tuned
// to *this* force model's truncation error and is validated by the consistency
// campaign, which is the right instrument for it. Mixing it in here would
// conflate a dynamics question with a tuning choice.

#include <gtest/gtest.h>

#include <cmath>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <memory>

#include "constants/constants.hpp"
#include "frames/eop.hpp"
#include "gnc/orbit_od.hpp"
#include "math/typed_vector.hpp"
#include "time/leap_seconds.hpp"
#include "time/timescales.hpp"
#include "time/utc.hpp"

namespace pc = polaris::constants;
namespace pf = polaris::frames;
namespace pm = polaris::math;
namespace pmf = polaris::math::frames;
namespace pt = polaris::time;

using polaris::gnc::OrbitOd;
using polaris::gnc::OrbitOdConfig;
using polaris::gnc::OrbitOdRefusal;

namespace {

constexpr double kAltitudeM = 400'000.0;
constexpr double kInclinationRad = 0.9006;  // 51.6 deg, the ISS-like reference

/// The orbital period at the test altitude [s]; the arcs below are quoted in it.
double orbitPeriodS() {
  const double a = pc::gravity::kReferenceRadius + kAltitudeM;
  return 2.0 * M_PI * std::sqrt(a * a * a / pc::gravity::kGM);
}

pt::Tai testEpoch() {
  pt::UtcDateTime utc;
  utc.year = 2026;
  utc.month = 1;
  utc.day = 1;
  utc.hour = 0;
  utc.minute = 0;
  utc.second = 0;
  return pt::taiFromUtc(utc, pt::LeapSecondTable::historical());
}

pt::Tai advance(const pt::Tai& base, double seconds) {
  return base + pt::Duration::fromSecondsF(seconds);
}

/// Zero EOP over the arc. The Earth orientation matters to the J2 pole only, and
/// the reduction's accuracy is pinned elsewhere (`eop_golden_test.cpp`); a
/// synthetic table keeps this file from re-parsing a 3.6 MB fixture per case.
pf::EopValue zeroEop(const pt::Tai& t) {
  pf::EopTable<8> table;
  const double mjd0 = std::floor(static_cast<double>(t.nanosecondsSinceEpoch()) / 1.0e9 /
                                 pc::time::kSecondsPerDay) +
                      pf::kMjd1970 - 1.0;
  for (int i = 0; i < 3; ++i) {
    EXPECT_TRUE(table.addEntry({mjd0 + static_cast<double>(i), 0.0, 0.0, 0.0}));
  }
  pf::EopValue v;
  EXPECT_TRUE(table.lookup(t, pt::LeapSecondTable::historical(), v));
  return v;
}

/// A conservative, process-noise-free configuration.
///
/// `zonal_j2` selects the closed-form path when non-zero; the harmonic field is
/// left off so the two cases below isolate exactly one force term each. Drag is
/// off because it is **not** conservative — Liouville's theorem does not apply to
/// a dissipative system, and including it would make the determinant check
/// measure atmospheric decay rather than truncation.
OrbitOdConfig conservativeConfig(bool with_j2) {
  OrbitOdConfig cfg;
  cfg.mu_m3_per_s2 = pc::gravity::kGM;
  cfg.reference_radius_m = pc::gravity::kReferenceRadius;
  cfg.geopotential_degree = 0;
  cfg.geopotential_order = 0;
  cfg.zonal_j2 = with_j2 ? pc::gravity::kJ2 : 0.0;
  cfg.drag_ballistic_coeff_m2_per_kg = 0.0;
  // Process noise as close to off as the config permits. `isValid` requires a
  // strictly positive PSD and is right to: a filter with literally no process
  // noise drives its covariance to zero and stops listening to measurements, so
  // the flight config must not be able to express it. 1e-30 m²/s³ contributes
  // ~1e-28 to the covariance over the arcs below, twenty orders under the
  // tightest tolerance asserted here, so `P(t) = Φ P₀ Φᵀ` holds to far better
  // than anything being measured. Q is not being tested here — it is tuned to
  // this force model's truncation error and the consistency campaign is the
  // instrument for it.
  cfg.accel_psd_m2_per_s3 = 1.0e-30;
  cfg.position_nis_gate = 16.27;  // chi2(3) at 0.999; unused, no fix is ingested
  cfg.velocity_nis_gate = 16.27;
  cfg.max_coast_s = 1.0e9;  // the coast policy is tested on its own
  cfg.max_degraded_coast_s = 1.0e9;
  cfg.max_dt_s = 10.0;
  cfg.max_step_s = 1.0;
  cfg.min_radius_m = 6.5e6;
  cfg.max_radius_m = 8.0e6;
  return cfg;
}

void circularState(Eigen::Vector3d& r, Eigen::Vector3d& v) {
  const double radius = pc::gravity::kReferenceRadius + kAltitudeM;
  r = Eigen::Vector3d(radius, 0.0, 0.0);
  const double speed = std::sqrt(pc::gravity::kGM / radius);
  v = Eigen::Vector3d(0.0, speed * std::cos(kInclinationRad), speed * std::sin(kInclinationRad));
}

/// Propagate to `epoch0 + t_s` in steps the config accepts.
[[nodiscard]] bool walkTo(OrbitOd& filter, const pt::Tai& epoch0, double t_s,
                          const pf::EopValue& eop) {
  const pt::Tai target = advance(epoch0, t_s);
  while (filter.epoch() < target) {
    const double remaining = (target - filter.epoch()).seconds();
    const pt::Tai next = (remaining <= 10.0) ? target : advance(filter.epoch(), 10.0);
    if (filter.propagate(next, eop) != OrbitOdRefusal::kNone) {
      return false;
    }
  }
  return true;
}

/// The covariance after propagating `P₀ = I` over @p arc_s, i.e. `ΦΦᵀ`.
OrbitOd::Covariance propagatedIdentity(const OrbitOdConfig& cfg, double arc_s) {
  const pt::Tai epoch0 = testEpoch();
  const pf::EopValue eop = zeroEop(epoch0);
  Eigen::Vector3d r;
  Eigen::Vector3d v;
  circularState(r, v);

  OrbitOd filter(cfg);
  EXPECT_EQ(filter.initialize(epoch0, pm::Vec3<pmf::ECI>(r), pm::Vec3<pmf::ECI>(v),
                              OrbitOd::Covariance::Identity()),
            OrbitOdRefusal::kNone);
  EXPECT_TRUE(walkTo(filter, epoch0, arc_s, eop));
  return filter.covariance();
}

/// The propagated state after @p arc_s, started from @p x0. Used to build the
/// finite-difference reference for Φ.
Eigen::Matrix<double, 6, 1> propagatedState(const OrbitOdConfig& cfg,
                                            const Eigen::Matrix<double, 6, 1>& x0, double arc_s) {
  const pt::Tai epoch0 = testEpoch();
  const pf::EopValue eop = zeroEop(epoch0);

  OrbitOd filter(cfg);
  EXPECT_EQ(filter.initialize(epoch0, pm::Vec3<pmf::ECI>(Eigen::Vector3d(x0.head<3>())),
                              pm::Vec3<pmf::ECI>(Eigen::Vector3d(x0.tail<3>())),
                              OrbitOd::Covariance::Identity()),
            OrbitOdRefusal::kNone);
  EXPECT_TRUE(walkTo(filter, epoch0, arc_s, eop));

  Eigen::Matrix<double, 6, 1> out;
  out.head<3>() = filter.position().eigen();
  out.tail<3>() = filter.velocity().eigen();
  return out;
}

/// Φ over @p arc_s by central differences of the propagated trajectory.
///
/// The perturbations are scaled per block — metres on position, millimetres per
/// second on velocity — because a step good for one is a poor step for the
/// other: the two differ by three orders in magnitude and by six in the
/// sensitivity they produce. Both are far above the ~1e-10 m round-off of a
/// 7e6 m coordinate and far below the scale over which the dynamics curve.
Eigen::Matrix<double, 6, 6> finiteDifferenceStm(const OrbitOdConfig& cfg, double arc_s) {
  Eigen::Matrix<double, 6, 1> x0;
  Eigen::Vector3d r;
  Eigen::Vector3d v;
  circularState(r, v);
  x0.head<3>() = r;
  x0.tail<3>() = v;

  Eigen::Matrix<double, 6, 6> stm;
  for (int j = 0; j < 6; ++j) {
    const double step = (j < 3) ? 1.0 : 1.0e-3;
    Eigen::Matrix<double, 6, 1> plus = x0;
    Eigen::Matrix<double, 6, 1> minus = x0;
    plus(j) += step;
    minus(j) -= step;
    stm.col(j) =
        (propagatedState(cfg, plus, arc_s) - propagatedState(cfg, minus, arc_s)) / (2.0 * step);
  }
  return stm;
}

// ---------------------------------------------------------------------------
// Liouville: det Phi = 1, so det(Phi Phi^T) = 1
// ---------------------------------------------------------------------------
//
// Every bound below is the measured value with roughly an order of margin, not
// a target picked in advance. The measurements, J2 on, P0 = I:
//
//     arc        det - 1        sigma_R      sigma_I      sigma_C
//      60 s      2.9e-10        3.6e+03      3.6e+03      3.6e+03
//     150 s      7.8e-10        2.3e+04      2.2e+04      2.2e+04
//     300 s      1.1e-09        9.7e+04      8.7e+04      8.7e+04   <- coast horizon
//     600 s      3.3e-09        4.6e+05      3.3e+05      3.1e+05
//    1200 s      5.4e-08        2.6e+06      2.1e+06      7.5e+05
//    3000 s      2.6e-05        3.5e+07      8.7e+07      5.0e+04
//    5554 s      1.2e-04        2.6e+04      2.8e+08      2.6e+02   <- one orbit
//
// The shape of that first column is the result worth carrying: flat at ~1e-9
// through twice the coast horizon, then climbing by four orders over the next
// factor of nine in arc. The covariance is excellent exactly where the filter
// is allowed to operate and degrades quickly past it, which is an argument for
// `max_coast_s` that did not previously exist in measured form.

TEST(OrbitOdCovariance, CoastHorizonPropagationPreservesPhaseSpaceVolume) {
  // The operational claim: over the longest arc the filter is ever permitted to
  // coast, the covariance's phase-space volume is right to a part in 1e9. This
  // is the case that must not regress.
  const OrbitOd::Covariance p = propagatedIdentity(conservativeConfig(true), 300.0);
  EXPECT_NEAR(p.determinant(), 1.0, 1.0e-8);
}

TEST(OrbitOdCovariance, VolumeHoldsToTwiceTheCoastHorizon) {
  // Margin on the policy rather than a second sample of the same number: the
  // 300 s horizon sits inside the flat region, not on its edge. Raising
  // `max_coast_s` towards an arc where the volume error has begun to climb
  // would fail here first.
  const OrbitOd::Covariance p = propagatedIdentity(conservativeConfig(true), 600.0);
  EXPECT_NEAR(p.determinant(), 1.0, 1.0e-8);
}

TEST(OrbitOdCovariance, TwoBodyVolumeHoldsOverAFullOrbit) {
  // Far outside the flown envelope, and included because it bounds the
  // degradation rather than leaving it unstated: a full orbit of unaided
  // propagation still only inflates the volume by ~1e-4.
  const OrbitOd::Covariance p = propagatedIdentity(conservativeConfig(false), orbitPeriodS());
  EXPECT_NEAR(p.determinant(), 1.0, 1.0e-3);
}

TEST(OrbitOdCovariance, J2VolumeHoldsOverAFullOrbit) {
  // The same with oblateness on, which is what makes these checks bear on the
  // Jacobian of the *flown* model rather than on two-body motion alone.
  const OrbitOd::Covariance p = propagatedIdentity(conservativeConfig(true), orbitPeriodS());
  EXPECT_NEAR(p.determinant(), 1.0, 1.0e-3);
}

TEST(OrbitOdCovariance, TheVolumeErrorIsInTheConservativeDirection) {
  // Sign matters as much as magnitude. The truncation inflates the covariance
  // rather than shrinking it, so the filter's error is towards over-stating its
  // own uncertainty. That is the safe direction: an over-confident filter
  // under-weights good measurements and eventually refuses them, which is the
  // failure mode the campaign's NEES upper bound exists to catch.
  const OrbitOdConfig cfg = conservativeConfig(true);
  EXPECT_GT(propagatedIdentity(cfg, 300.0).determinant(), 1.0);
  EXPECT_GT(propagatedIdentity(cfg, 600.0).determinant(), 1.0);
  // Not asserted at a full orbit: there the deviation is ~1e-4 either way, and
  // the determinant of a 6×6 with entries up to ~1e8 that cancel to 1 sits at
  // the round-off floor — the sign flipped when the covariance became the
  // marginal of a 9-state P (Push 73) with no change to the arithmetic, only
  // to its summation order. The magnitude bound is J2VolumeHoldsOverAFullOrbit.
}

// ---------------------------------------------------------------------------
// Against a finite-difference STM off the GMAT-validated trajectory
// ---------------------------------------------------------------------------

TEST(OrbitOdCovariance, PropagatedCovarianceMatchesTheFiniteDifferenceTransition) {
  // Over the coast horizon, `Phi Phi^T` from the covariance path against the
  // same quantity built by differencing the state path. Compared as the
  // covariance rather than as Phi because that is what the flight software
  // publishes and what a consumer acts on.
  //
  // The worst entry disagrees by 1.35e-4 relative, and that number is the
  // filter's, not the test's: it is unchanged to four digits when the
  // difference step is scaled by 1/4 and by 4, which rules out both truncation
  // in the stencil and round-off in the differences.
  //
  // Read together with the determinant cases, this says the truncation error is
  // almost entirely a *shape* error: the volume is right to 1e-9 while the map
  // is off by 1e-4. That is the expected behaviour of a truncated exponential
  // of a Hamiltonian matrix, and it is why both checks are here — neither one
  // sees what the other does.
  const OrbitOdConfig cfg = conservativeConfig(true);
  const double arc = 300.0;

  const OrbitOd::Covariance measured = propagatedIdentity(cfg, arc);
  const Eigen::Matrix<double, 6, 6> stm = finiteDifferenceStm(cfg, arc);
  const Eigen::Matrix<double, 6, 6> expected = stm * stm.transpose();

  // Scaled per entry by the geometric mean of the two diagonals it sits
  // between: position-position entries are O(1e5) here and velocity-velocity
  // entries ~1e-6 of them, so one absolute tolerance would be either vacuous on
  // one block or unmeetable on the other.
  for (int i = 0; i < 6; ++i) {
    for (int j = 0; j < 6; ++j) {
      const double scale = std::sqrt(std::abs(expected(i, i) * expected(j, j)));
      EXPECT_NEAR(measured(i, j), expected(i, j), 1.0e-3 * std::max(scale, 1.0e-12))
          << "entry (" << i << "," << j << ")";
    }
  }
}

// ---------------------------------------------------------------------------
// The shape the RIC decomposition in the campaign exists to see
// ---------------------------------------------------------------------------

TEST(OrbitOdCovariance, InTrackUncertaintyDominatesOverAnOrbit) {
  // A property of orbital motion rather than of this implementation: a radial or
  // velocity error becomes a period error and therefore a secularly growing
  // along-track error, so an initially isotropic covariance must emerge from a
  // long coast elongated along the track. Measured over one orbit the in-track
  // variance exceeds the radial by four orders.
  //
  // Deliberately *not* asserted at the 300 s coast horizon, where it is false:
  // there the radial variance is still the larger (9.7e4 against 8.7e4). The
  // secular in-track term needs a substantial fraction of an orbit to overtake
  // the periodic radial one, and a test asserting otherwise would have been
  // asserting an intuition rather than the dynamics.
  const OrbitOd::Covariance p = propagatedIdentity(conservativeConfig(true), orbitPeriodS());

  Eigen::Vector3d r;
  Eigen::Vector3d v;
  circularState(r, v);
  const Eigen::Vector3d radial = r.normalized();
  const Eigen::Vector3d cross = r.cross(v).normalized();
  const Eigen::Vector3d in_track = cross.cross(radial);

  const Eigen::Matrix3d p_rr = p.block<3, 3>(OrbitOd::kPosition, OrbitOd::kPosition);
  const double var_radial = radial.transpose() * p_rr * radial;
  const double var_in_track = in_track.transpose() * p_rr * in_track;
  const double var_cross = cross.transpose() * p_rr * cross;

  EXPECT_GT(var_in_track, 100.0 * var_radial);
  EXPECT_GT(var_in_track, 100.0 * var_cross);
}

}  // namespace
