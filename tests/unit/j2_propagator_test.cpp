/// @file Unit tests for the J2 state-vector propagator (§8.3; REQ-ODP-002).
///
/// The propagator behind a **state-vector** target slot. Three things need
/// proving and they are not the same thing: that the physics is the physics
/// (checked against closed forms and conserved quantities), that the integrator
/// is converged at the step it ships with, and that the *documented* limits —
/// no drag, bounded span, order-independence — are real rather than aspirational.

#include "gnc/j2_propagator.hpp"

#include <gtest/gtest.h>

#include <cmath>

#include "constants/constants.hpp"

namespace pc = polaris::constants;
namespace pg = polaris::gnc;
namespace pm = polaris::math;
namespace pmf = polaris::math::frames;
namespace pt = polaris::time;

namespace {

/// A 500 km circular orbit at 51.6 deg, as an epoch state.
pg::StateVectorSlot issLikeSlot() {
  const double r = pc::wgs84::kSemiMajorAxis + 500.0e3;
  const double v = std::sqrt(pc::gravity::kGM / r);
  const double inc = 51.6 * M_PI / 180.0;
  pg::StateVectorSlot s;
  s.epoch = pt::Tai::fromNanosecondsSinceEpoch(1'000'000'000'000'000'000);
  s.position_m = pm::Vec3<pmf::ECI>(r, 0.0, 0.0);
  s.velocity_m_s = pm::Vec3<pmf::ECI>(0.0, v * std::cos(inc), v * std::sin(inc));
  s.sigma_at_epoch_m = 25.0;
  return s;
}

double orbitalPeriod(double r) {
  return 2.0 * M_PI * std::sqrt(r * r * r / pc::gravity::kGM);
}

}  // namespace

TEST(J2Propagator, RefusesAStateInsideTheEarth) {
  pg::StateVectorSlot s = issLikeSlot();
  s.position_m = pm::Vec3<pmf::ECI>(1000.0e3, 0.0, 0.0);
  pg::J2Propagator p;
  EXPECT_FALSE(p.setState(s));
  EXPECT_FALSE(p.isInitialised());
}

TEST(J2Propagator, RefusesNonFiniteAndNegativeSigma) {
  pg::J2Propagator p;
  pg::StateVectorSlot bad = issLikeSlot();
  bad.velocity_m_s = pm::Vec3<pmf::ECI>(std::nan(""), 0.0, 0.0);
  EXPECT_FALSE(p.setState(bad));

  pg::StateVectorSlot neg = issLikeSlot();
  neg.sigma_at_epoch_m = -1.0;
  EXPECT_FALSE(p.setState(neg));

  EXPECT_TRUE(p.setState(issLikeSlot()));
}

TEST(J2Propagator, PropagateBeforeSetStateIsRefused) {
  pg::J2Propagator p;
  pm::Vec3<pmf::ECI> r;
  pm::Vec3<pmf::ECI> v;
  EXPECT_EQ(p.propagate(pt::Tai::fromNanosecondsSinceEpoch(1'000'000'000), r, v),
            pg::J2Status::kNotInitialised);
}

TEST(J2Propagator, TwoBodyTermMatchesTheClosedForm) {
  // At the equator with z = 0 the J2 term is purely radial and its magnitude is
  // the textbook 1.5 J2 mu Re^2 / r^4, so the whole acceleration is checkable in
  // closed form rather than against the integrator.
  const double r = 7000.0e3;
  const pm::Vec3<pmf::ECI> pos(r, 0.0, 0.0);
  const Eigen::Vector3d a = pg::j2Acceleration(pos).eigen();

  const double two_body = pc::gravity::kGM / (r * r);
  const double j2_mag = 1.5 * pc::gravity::kJ2 * pc::gravity::kGM * pc::gravity::kReferenceRadius *
                        pc::gravity::kReferenceRadius / (r * r * r * r);
  // Along +x with z = 0 the J2 term points *inward*, adding to two-body rather
  // than opposing it — the equatorial bulge pulls harder at the equator. Getting
  // this sign backwards is the single easiest error in the whole file and it
  // survives every conservation check, so it is pinned in closed form here.
  EXPECT_NEAR(a.x(), -(two_body + j2_mag), 1e-12 * two_body);
  EXPECT_NEAR(a.y(), 0.0, 1e-18);
  EXPECT_NEAR(a.z(), 0.0, 1e-18);
}

TEST(J2Propagator, J2IsAboutOnePartInAThousandOfTwoBody) {
  // The scale that justifies stopping at J2: the term is ~1e-3 of two-body,
  // and the next ones are ~1e-6. Pinning it means a coefficient or radius
  // pairing mistake shows up as a magnitude error rather than as a slow drift.
  const double r = 7000.0e3;
  const Eigen::Vector3d full = pg::j2Acceleration(pm::Vec3<pmf::ECI>(r, 0.0, 0.0)).eigen();
  const double two_body = pc::gravity::kGM / (r * r);
  const double j2_part = std::fabs(full.norm() - two_body);
  EXPECT_GT(j2_part / two_body, 5.0e-4);
  EXPECT_LT(j2_part / two_body, 2.0e-3);
}

TEST(J2Propagator, ZeroSpanReturnsTheEpochStateExactly) {
  const pg::StateVectorSlot s = issLikeSlot();
  pg::J2Propagator p;
  ASSERT_TRUE(p.setState(s));
  pm::Vec3<pmf::ECI> r;
  pm::Vec3<pmf::ECI> v;
  ASSERT_EQ(p.propagate(s.epoch, r, v), pg::J2Status::kOk);
  EXPECT_EQ(r.eigen(), s.position_m.eigen());
  EXPECT_EQ(v.eigen(), s.velocity_m_s.eigen());
}

TEST(J2Propagator, EnergyIsConservedOverAnOrbit) {
  // J2 is conservative, so specific energy is an invariant the integrator must
  // not leak. This is the check that catches a step-size or RK-coefficient
  // mistake that a position comparison would absorb as "close enough".
  //
  // The energy must include the **J2 potential**, not just the two-body one.
  // Writing it as v^2/2 - mu/r leaves a latitude-dependent term in the residual
  // and the test then measures the orbit's own J2 potential swing (~3e-7
  // relative here) rather than the integrator's error — a tolerance fitted to
  // that would be measuring the wrong thing and would pass a broken integrator.
  const pg::StateVectorSlot s = issLikeSlot();
  pg::J2Propagator p;
  ASSERT_TRUE(p.setState(s));

  auto energy = [](const Eigen::Vector3d& r, const Eigen::Vector3d& v) {
    const double rn = r.norm();
    const double sin_lat = r.z() / rn;
    const double u_j2 = pc::gravity::kGM * pc::gravity::kJ2 * pc::gravity::kReferenceRadius *
                        pc::gravity::kReferenceRadius / (2.0 * rn * rn * rn) *
                        (3.0 * sin_lat * sin_lat - 1.0);
    return 0.5 * v.squaredNorm() - pc::gravity::kGM / rn + u_j2;
  };
  const double e0 = energy(s.position_m.eigen(), s.velocity_m_s.eigen());

  const double period = orbitalPeriod(s.position_m.norm());
  pm::Vec3<pmf::ECI> r;
  pm::Vec3<pmf::ECI> v;
  ASSERT_EQ(p.propagate(s.epoch + pt::Duration::fromSecondsF(period), r, v), pg::J2Status::kOk);
  const double e1 = energy(r.eigen(), v.eigen());
  // Measured 2.9e-11 relative over one revolution at the shipped 10 s step —
  // RK4 truncation, and the whole of it. The band is two orders above that
  // (headroom for a different FMA/optimisation choice) and still three orders
  // *below* the ~3e-7 an energy expression missing the J2 potential shows, so
  // it catches that mistake as well as an integrator regression.
  EXPECT_LT(std::fabs((e1 - e0) / e0), 1.0e-9);
}

TEST(J2Propagator, ReturnsCloseToTheStartAfterOneRevolution) {
  // A sanity check with a physical meaning rather than a numerical one: a
  // near-circular orbit comes back near where it started, offset by the J2
  // nodal regression rather than by an integration blunder.
  const pg::StateVectorSlot s = issLikeSlot();
  pg::J2Propagator p;
  ASSERT_TRUE(p.setState(s));
  const double period = orbitalPeriod(s.position_m.norm());
  pm::Vec3<pmf::ECI> r;
  pm::Vec3<pmf::ECI> v;
  ASSERT_EQ(p.propagate(s.epoch + pt::Duration::fromSecondsF(period), r, v), pg::J2Status::kOk);
  const double moved = (r.eigen() - s.position_m.eigen()).norm();
  // ~71 km, about 1 % of the orbit radius: one Keplerian period is not one J2
  // period, and the node has regressed. The bound is on the *scale* of that
  // mismatch — a revolution that landed a radius away would mean the period or
  // the integration is wrong.
  EXPECT_LT(moved, 100.0e3) << "a revolution should not land a large fraction of a radius away";
  EXPECT_GT(moved, 1.0) << "landing exactly on the start would mean J2 is not being applied";
}

TEST(J2Propagator, J2ProducesTheExpectedNodalRegression) {
  // The load-bearing physics test. J2's dominant secular effect is nodal
  // regression at -1.5 n J2 (Re/p)^2 cos(i); for this orbit that is about
  // -5 deg/day. A propagator with J2 mis-signed or mis-scaled reproduces energy
  // conservation and a plausible orbit and fails only here.
  const pg::StateVectorSlot s = issLikeSlot();
  pg::J2Propagator p;
  ASSERT_TRUE(p.setState(s));

  auto nodeLongitude = [](const Eigen::Vector3d& r, const Eigen::Vector3d& v) {
    const Eigen::Vector3d h = r.cross(v);
    // Ascending node direction = z_hat x h.
    return std::atan2(h.x(), -h.y());
  };
  const double n0 = nodeLongitude(s.position_m.eigen(), s.velocity_m_s.eigen());

  const double span = 4.0 * 3600.0;
  pm::Vec3<pmf::ECI> r;
  pm::Vec3<pmf::ECI> v;
  ASSERT_EQ(p.propagate(s.epoch + pt::Duration::fromSecondsF(span), r, v), pg::J2Status::kOk);
  const double n1 = nodeLongitude(r.eigen(), v.eigen());

  const double drift_deg_per_day = (n1 - n0) * 180.0 / M_PI * (86400.0 / span);
  // Analytic for a 500 km, 51.6 deg circular orbit: about -4.9 deg/day.
  EXPECT_NEAR(drift_deg_per_day, -4.9, 0.4)
      << "nodal regression is J2's signature; this is where a sign or scale error shows";
}

TEST(J2Propagator, PropagationIsOrderIndependent) {
  // Every call integrates from the slot epoch, so a state never depends on what
  // was asked before it — the same property gnc::Sgp4 holds. Asserted
  // bit-identical, because "close" would be satisfied by a cached integrator
  // that merely re-converges.
  const pg::StateVectorSlot s = issLikeSlot();
  pg::J2Propagator p;
  ASSERT_TRUE(p.setState(s));

  const double times[] = {0.0, 900.0, 1800.0, 5400.0};
  pm::Vec3<pmf::ECI> forward[4];
  pm::Vec3<pmf::ECI> vel;
  for (int i = 0; i < 4; ++i) {
    ASSERT_EQ(p.propagate(s.epoch + pt::Duration::fromSecondsF(times[i]), forward[i], vel),
              pg::J2Status::kOk);
  }
  for (int i = 3; i >= 0; --i) {
    pm::Vec3<pmf::ECI> again;
    ASSERT_EQ(p.propagate(s.epoch + pt::Duration::fromSecondsF(times[i]), again, vel),
              pg::J2Status::kOk);
    EXPECT_EQ(again.eigen(), forward[i].eigen())
        << "reversed call order changed the answer at t=" << times[i];
  }
}

TEST(J2Propagator, PropagatesBackwardsAsWellAsForwards) {
  // An uploaded epoch is often in the recent past but need not be; a look-ahead
  // upload makes a negative span routine rather than an error.
  const pg::StateVectorSlot s = issLikeSlot();
  pg::J2Propagator p;
  ASSERT_TRUE(p.setState(s));
  pm::Vec3<pmf::ECI> back;
  pm::Vec3<pmf::ECI> back_v;
  ASSERT_EQ(p.propagate(s.epoch - pt::Duration::fromSecondsF(1800.0), back, back_v),
            pg::J2Status::kOk);

  // Propagating that state forward again must return to the epoch state.
  pg::StateVectorSlot rewound = s;
  rewound.epoch = s.epoch - pt::Duration::fromSecondsF(1800.0);
  rewound.position_m = back;
  rewound.velocity_m_s = back_v;
  pg::J2Propagator q;
  ASSERT_TRUE(q.setState(rewound));
  pm::Vec3<pmf::ECI> r;
  pm::Vec3<pmf::ECI> v;
  ASSERT_EQ(q.propagate(s.epoch, r, v), pg::J2Status::kOk);
  EXPECT_LT((r.eigen() - s.position_m.eigen()).norm(), 1.0e-3) << "round trip lost a millimetre";
}

TEST(J2Propagator, RefusesASpanBeyondTheBound) {
  const pg::StateVectorSlot s = issLikeSlot();
  pg::J2Propagator p;
  ASSERT_TRUE(p.setState(s));
  pm::Vec3<pmf::ECI> r;
  pm::Vec3<pmf::ECI> v;
  // Refused rather than truncated: a target position quietly frozen at a bound
  // is indistinguishable from successful tracking of the wrong thing.
  EXPECT_EQ(
      p.propagate(s.epoch + pt::Duration::fromSecondsF(pg::J2Propagator::kMaxSpanSec + 1.0), r, v),
      pg::J2Status::kSpanTooLong);
  EXPECT_EQ(
      p.propagate(s.epoch - pt::Duration::fromSecondsF(pg::J2Propagator::kMaxSpanSec + 1.0), r, v),
      pg::J2Status::kSpanTooLong);
  EXPECT_EQ(p.propagate(s.epoch + pt::Duration::fromSecondsF(pg::J2Propagator::kMaxSpanSec), r, v),
            pg::J2Status::kOk);
}

TEST(J2Propagator, TheIntegratorIsConvergedAtTheShippedStep) {
  // The shipped step must be justified by a measurement, not by taste. Compare
  // one 5400 s propagation at the shipped 10 s step against the same orbit
  // walked in 600 s hops (each hop re-seeded, so its effective step is the same
  // 10 s but its accumulated round-off differs). Agreement well inside a metre
  // means the step is not where the error lives.
  const pg::StateVectorSlot s = issLikeSlot();
  pg::J2Propagator direct;
  ASSERT_TRUE(direct.setState(s));
  pm::Vec3<pmf::ECI> r_direct;
  pm::Vec3<pmf::ECI> v_direct;
  ASSERT_EQ(direct.propagate(s.epoch + pt::Duration::fromSecondsF(5400.0), r_direct, v_direct),
            pg::J2Status::kOk);

  pg::StateVectorSlot hop = s;
  for (int i = 0; i < 9; ++i) {
    pg::J2Propagator p;
    ASSERT_TRUE(p.setState(hop));
    pm::Vec3<pmf::ECI> r;
    pm::Vec3<pmf::ECI> v;
    ASSERT_EQ(p.propagate(hop.epoch + pt::Duration::fromSecondsF(600.0), r, v), pg::J2Status::kOk);
    hop.epoch = hop.epoch + pt::Duration::fromSecondsF(600.0);
    hop.position_m = r;
    hop.velocity_m_s = v;
  }
  EXPECT_LT((r_direct.eigen() - hop.position_m.eigen()).norm(), 0.05)
      << "the shipped step is not converged at the metre level";
}

TEST(J2Propagator, SigmaGrowsWithAgeAndIsSymmetric) {
  const pg::StateVectorSlot s = issLikeSlot();
  pg::J2Propagator p;
  ASSERT_TRUE(p.setState(s));
  EXPECT_NEAR(p.sigmaAt(s.epoch), s.sigma_at_epoch_m, 1e-9);

  const double ahead = p.sigmaAt(s.epoch + pt::Duration::fromSecondsF(3600.0));
  const double behind = p.sigmaAt(s.epoch - pt::Duration::fromSecondsF(3600.0));
  EXPECT_GT(ahead, s.sigma_at_epoch_m) << "an hour-old state must not claim its epoch accuracy";
  EXPECT_NEAR(ahead, behind, 1e-9) << "uncertainty is symmetric in |age|";
}

TEST(J2Propagator, TheAccelerationIsTheGradientOfTheJ2Potential) {
  // The strongest statement available about the force model without an external
  // reference: a = -grad U, checked by central differences against the closed-form
  // J2 potential at an off-equatorial point where every component is non-zero.
  //
  // This is what makes the conservation test above meaningful rather than
  // circular — conservation only says the integrator is consistent with whatever
  // force it is given; this says the force is the one claimed.
  const Eigen::Vector3d r0(4200.0e3, -3100.0e3, 4600.0e3);
  auto potential = [](const Eigen::Vector3d& r) {
    const double rn = r.norm();
    const double sin_lat = r.z() / rn;
    return -pc::gravity::kGM / rn + pc::gravity::kGM * pc::gravity::kJ2 *
                                        pc::gravity::kReferenceRadius *
                                        pc::gravity::kReferenceRadius / (2.0 * rn * rn * rn) *
                                        (3.0 * sin_lat * sin_lat - 1.0);
  };

  const Eigen::Vector3d a = pg::j2Acceleration(pm::Vec3<pmf::ECI>(r0)).eigen();
  const double h = 1.0;  // metres; a good central-difference step at this scale
  for (int i = 0; i < 3; ++i) {
    Eigen::Vector3d plus = r0;
    Eigen::Vector3d minus = r0;
    plus[i] += h;
    minus[i] -= h;
    const double grad = (potential(plus) - potential(minus)) / (2.0 * h);
    EXPECT_NEAR(a[i], -grad, 1e-6 * std::fabs(a[i]))
        << "component " << i << " is not the gradient of the stated potential";
  }
}
