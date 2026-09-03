/// @file Unit tests for the J2 state-vector propagator (§8.3; REQ-ODP-002).
///
/// The propagator behind a **state-vector** target slot. Three things need
/// proving and they are not the same thing: that the physics is the physics
/// (checked against closed forms and conserved quantities), that the integrator
/// is converged at the step it ships with, and that the *documented* limits —
/// no drag, bounded span, order-independence — are real rather than aspirational.

#include "gnc/target_propagator.hpp"

#include <gtest/gtest.h>

#include <cmath>

#include "constants/constants.hpp"
#include "gnc/geopotential.hpp"

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

/// The J2 setting: degree 2, order 0. Most of the cases below are *about* the
/// J2 model — nodal regression, the closed-form two-body limit, the potential
/// gradient — so they name it rather than riding whatever the default happens
/// to be. The default is 8x8 (Push 82) and has its own cases at the end.
constexpr pg::TargetForceModel kJ2{2, 0};

pm::Vec3<pmf::ECI> j2AccelerationForTest(const pm::Vec3<pmf::ECI>& r) {
  return pg::targetAcceleration(r, kJ2, nullptr);
}

}  // namespace

TEST(TargetPropagator, RefusesAStateInsideTheEarth) {
  pg::StateVectorSlot s = issLikeSlot();
  s.position_m = pm::Vec3<pmf::ECI>(1000.0e3, 0.0, 0.0);
  pg::TargetPropagator p;
  EXPECT_FALSE(p.setState(s));
  EXPECT_FALSE(p.isInitialised());
}

TEST(TargetPropagator, RefusesNonFiniteAndNegativeSigma) {
  pg::TargetPropagator p;
  pg::StateVectorSlot bad = issLikeSlot();
  bad.velocity_m_s = pm::Vec3<pmf::ECI>(std::nan(""), 0.0, 0.0);
  EXPECT_FALSE(p.setState(bad));

  pg::StateVectorSlot neg = issLikeSlot();
  neg.sigma_at_epoch_m = -1.0;
  EXPECT_FALSE(p.setState(neg));

  EXPECT_TRUE(p.setState(issLikeSlot()));

  p.setForceModel(kJ2);
}

TEST(TargetPropagator, PropagateBeforeSetStateIsRefused) {
  pg::TargetPropagator p;
  pm::Vec3<pmf::ECI> r;
  pm::Vec3<pmf::ECI> v;
  EXPECT_EQ(p.propagate(pt::Tai::fromNanosecondsSinceEpoch(1'000'000'000), nullptr, r, v),
            pg::PropagationStatus::kNotInitialised);
}

TEST(TargetPropagator, TwoBodyTermMatchesTheClosedForm) {
  // At the equator with z = 0 the J2 term is purely radial and its magnitude is
  // the textbook 1.5 J2 mu Re^2 / r^4, so the whole acceleration is checkable in
  // closed form rather than against the integrator.
  //
  // The tolerance is 1e-8 relative rather than 1e-12 because the J2 setting now
  // reads C20 out of the **EGM2008 table** the vehicle already carries
  // (`gnc::geopotential`) instead of the standalone `constants::gravity::kJ2`.
  // Those are two sources for one physical quantity and they differ by 6.3e-10
  // relative — a rounding difference, measured, not a defect. Flying the table's
  // value is the right side of that trade: it is the same coefficient the orbit
  // filter uses, so the two propagators cannot disagree about the Earth's
  // oblateness while both claiming to model J2.
  const double r = 7000.0e3;
  const pm::Vec3<pmf::ECI> pos(r, 0.0, 0.0);
  const Eigen::Vector3d a = j2AccelerationForTest(pos).eigen();

  const double two_body = pc::gravity::kGM / (r * r);
  const double j2_mag = 1.5 * pc::gravity::kJ2 * pc::gravity::kGM * pc::gravity::kReferenceRadius *
                        pc::gravity::kReferenceRadius / (r * r * r * r);
  // Along +x with z = 0 the J2 term points *inward*, adding to two-body rather
  // than opposing it — the equatorial bulge pulls harder at the equator. Getting
  // this sign backwards is the single easiest error in the whole file and it
  // survives every conservation check, so it is pinned in closed form here.
  EXPECT_NEAR(a.x(), -(two_body + j2_mag), 1e-8 * two_body);
  EXPECT_NEAR(a.y(), 0.0, 1e-18);
  EXPECT_NEAR(a.z(), 0.0, 1e-18);
}

TEST(TargetPropagator, J2IsAboutOnePartInAThousandOfTwoBody) {
  // The scale that justifies stopping at J2: the term is ~1e-3 of two-body,
  // and the next ones are ~1e-6. Pinning it means a coefficient or radius
  // pairing mistake shows up as a magnitude error rather than as a slow drift.
  const double r = 7000.0e3;
  const Eigen::Vector3d full = j2AccelerationForTest(pm::Vec3<pmf::ECI>(r, 0.0, 0.0)).eigen();
  const double two_body = pc::gravity::kGM / (r * r);
  const double j2_part = std::fabs(full.norm() - two_body);
  EXPECT_GT(j2_part / two_body, 5.0e-4);
  EXPECT_LT(j2_part / two_body, 2.0e-3);
}

TEST(TargetPropagator, ZeroSpanReturnsTheEpochStateExactly) {
  const pg::StateVectorSlot s = issLikeSlot();
  pg::TargetPropagator p;
  ASSERT_TRUE(p.setState(s));
  p.setForceModel(kJ2);
  pm::Vec3<pmf::ECI> r;
  pm::Vec3<pmf::ECI> v;
  ASSERT_EQ(p.propagate(s.epoch, nullptr, r, v), pg::PropagationStatus::kOk);
  EXPECT_EQ(r.eigen(), s.position_m.eigen());
  EXPECT_EQ(v.eigen(), s.velocity_m_s.eigen());
}

TEST(TargetPropagator, EnergyIsConservedOverAnOrbit) {
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
  pg::TargetPropagator p;
  ASSERT_TRUE(p.setState(s));
  p.setForceModel(kJ2);

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
  ASSERT_EQ(p.propagate(s.epoch + pt::Duration::fromSecondsF(period), nullptr, r, v),
            pg::PropagationStatus::kOk);
  const double e1 = energy(r.eigen(), v.eigen());
  // Measured 2.9e-11 relative over one revolution at the shipped 10 s step —
  // RK4 truncation, and the whole of it. The band is two orders above that
  // (headroom for a different FMA/optimisation choice) and still three orders
  // *below* the ~3e-7 an energy expression missing the J2 potential shows, so
  // it catches that mistake as well as an integrator regression.
  EXPECT_LT(std::fabs((e1 - e0) / e0), 1.0e-9);
}

TEST(TargetPropagator, ReturnsCloseToTheStartAfterOneRevolution) {
  // A sanity check with a physical meaning rather than a numerical one: a
  // near-circular orbit comes back near where it started, offset by the J2
  // nodal regression rather than by an integration blunder.
  const pg::StateVectorSlot s = issLikeSlot();
  pg::TargetPropagator p;
  ASSERT_TRUE(p.setState(s));
  p.setForceModel(kJ2);
  const double period = orbitalPeriod(s.position_m.norm());
  pm::Vec3<pmf::ECI> r;
  pm::Vec3<pmf::ECI> v;
  ASSERT_EQ(p.propagate(s.epoch + pt::Duration::fromSecondsF(period), nullptr, r, v),
            pg::PropagationStatus::kOk);
  const double moved = (r.eigen() - s.position_m.eigen()).norm();
  // ~71 km, about 1 % of the orbit radius: one Keplerian period is not one J2
  // period, and the node has regressed. The bound is on the *scale* of that
  // mismatch — a revolution that landed a radius away would mean the period or
  // the integration is wrong.
  EXPECT_LT(moved, 100.0e3) << "a revolution should not land a large fraction of a radius away";
  EXPECT_GT(moved, 1.0) << "landing exactly on the start would mean J2 is not being applied";
}

TEST(TargetPropagator, J2ProducesTheExpectedNodalRegression) {
  // The load-bearing physics test. J2's dominant secular effect is nodal
  // regression at -1.5 n J2 (Re/p)^2 cos(i); for this orbit that is about
  // -5 deg/day. A propagator with J2 mis-signed or mis-scaled reproduces energy
  // conservation and a plausible orbit and fails only here.
  const pg::StateVectorSlot s = issLikeSlot();
  pg::TargetPropagator p;
  ASSERT_TRUE(p.setState(s));
  p.setForceModel(kJ2);

  auto nodeLongitude = [](const Eigen::Vector3d& r, const Eigen::Vector3d& v) {
    const Eigen::Vector3d h = r.cross(v);
    // Ascending node direction = z_hat x h.
    return std::atan2(h.x(), -h.y());
  };
  const double n0 = nodeLongitude(s.position_m.eigen(), s.velocity_m_s.eigen());

  const double span = 4.0 * 3600.0;
  pm::Vec3<pmf::ECI> r;
  pm::Vec3<pmf::ECI> v;
  ASSERT_EQ(p.propagate(s.epoch + pt::Duration::fromSecondsF(span), nullptr, r, v),
            pg::PropagationStatus::kOk);
  const double n1 = nodeLongitude(r.eigen(), v.eigen());

  const double drift_deg_per_day = (n1 - n0) * 180.0 / M_PI * (86400.0 / span);
  // Analytic for a 500 km, 51.6 deg circular orbit: about -4.9 deg/day.
  EXPECT_NEAR(drift_deg_per_day, -4.9, 0.4)
      << "nodal regression is J2's signature; this is where a sign or scale error shows";
}

TEST(TargetPropagator, PropagationIsOrderIndependent) {
  // Every call integrates from the slot epoch, so a state never depends on what
  // was asked before it — the same property gnc::Sgp4 holds. Asserted
  // bit-identical, because "close" would be satisfied by a cached integrator
  // that merely re-converges.
  const pg::StateVectorSlot s = issLikeSlot();
  pg::TargetPropagator p;
  ASSERT_TRUE(p.setState(s));
  p.setForceModel(kJ2);

  const double times[] = {0.0, 900.0, 1800.0, 5400.0};
  pm::Vec3<pmf::ECI> forward[4];
  pm::Vec3<pmf::ECI> vel;
  for (int i = 0; i < 4; ++i) {
    ASSERT_EQ(p.propagate(s.epoch + pt::Duration::fromSecondsF(times[i]), nullptr, forward[i], vel),
              pg::PropagationStatus::kOk);
  }
  for (int i = 3; i >= 0; --i) {
    pm::Vec3<pmf::ECI> again;
    ASSERT_EQ(p.propagate(s.epoch + pt::Duration::fromSecondsF(times[i]), nullptr, again, vel),
              pg::PropagationStatus::kOk);
    EXPECT_EQ(again.eigen(), forward[i].eigen())
        << "reversed call order changed the answer at t=" << times[i];
  }
}

TEST(TargetPropagator, PropagatesBackwardsAsWellAsForwards) {
  // An uploaded epoch is often in the recent past but need not be; a look-ahead
  // upload makes a negative span routine rather than an error.
  const pg::StateVectorSlot s = issLikeSlot();
  pg::TargetPropagator p;
  ASSERT_TRUE(p.setState(s));
  p.setForceModel(kJ2);
  pm::Vec3<pmf::ECI> back;
  pm::Vec3<pmf::ECI> back_v;
  ASSERT_EQ(p.propagate(s.epoch - pt::Duration::fromSecondsF(1800.0), nullptr, back, back_v),
            pg::PropagationStatus::kOk);

  // Propagating that state forward again must return to the epoch state.
  pg::StateVectorSlot rewound = s;
  rewound.epoch = s.epoch - pt::Duration::fromSecondsF(1800.0);
  rewound.position_m = back;
  rewound.velocity_m_s = back_v;
  pg::TargetPropagator q;
  ASSERT_TRUE(q.setState(rewound));
  q.setForceModel(kJ2);
  pm::Vec3<pmf::ECI> r;
  pm::Vec3<pmf::ECI> v;
  ASSERT_EQ(q.propagate(s.epoch, nullptr, r, v), pg::PropagationStatus::kOk);
  EXPECT_LT((r.eigen() - s.position_m.eigen()).norm(), 1.0e-3) << "round trip lost a millimetre";
}

TEST(TargetPropagator, RefusesASpanBeyondTheBound) {
  const pg::StateVectorSlot s = issLikeSlot();
  pg::TargetPropagator p;
  ASSERT_TRUE(p.setState(s));
  p.setForceModel(kJ2);
  pm::Vec3<pmf::ECI> r;
  pm::Vec3<pmf::ECI> v;
  // Refused rather than truncated: a target position quietly frozen at a bound
  // is indistinguishable from successful tracking of the wrong thing.
  EXPECT_EQ(
      p.propagate(s.epoch + pt::Duration::fromSecondsF(pg::TargetPropagator::kMaxSpanSec + 1.0),
                  nullptr, r, v),
      pg::PropagationStatus::kSpanTooLong);
  EXPECT_EQ(
      p.propagate(s.epoch - pt::Duration::fromSecondsF(pg::TargetPropagator::kMaxSpanSec + 1.0),
                  nullptr, r, v),
      pg::PropagationStatus::kSpanTooLong);
  EXPECT_EQ(p.propagate(s.epoch + pt::Duration::fromSecondsF(pg::TargetPropagator::kMaxSpanSec),
                        nullptr, r, v),
            pg::PropagationStatus::kOk);
}

TEST(TargetPropagator, TheIntegratorIsConvergedAtTheShippedStep) {
  // The shipped step must be justified by a measurement, not by taste. Compare
  // one 5400 s propagation at the shipped 10 s step against the same orbit
  // walked in 600 s hops (each hop re-seeded, so its effective step is the same
  // 10 s but its accumulated round-off differs). Agreement well inside a metre
  // means the step is not where the error lives.
  const pg::StateVectorSlot s = issLikeSlot();
  pg::TargetPropagator direct;
  ASSERT_TRUE(direct.setState(s));
  direct.setForceModel(kJ2);
  pm::Vec3<pmf::ECI> r_direct;
  pm::Vec3<pmf::ECI> v_direct;
  ASSERT_EQ(
      direct.propagate(s.epoch + pt::Duration::fromSecondsF(5400.0), nullptr, r_direct, v_direct),
      pg::PropagationStatus::kOk);

  pg::StateVectorSlot hop = s;
  for (int i = 0; i < 9; ++i) {
    pg::TargetPropagator p;
    ASSERT_TRUE(p.setState(hop));
    p.setForceModel(kJ2);
    pm::Vec3<pmf::ECI> r;
    pm::Vec3<pmf::ECI> v;
    ASSERT_EQ(p.propagate(hop.epoch + pt::Duration::fromSecondsF(600.0), nullptr, r, v),
              pg::PropagationStatus::kOk);
    hop.epoch = hop.epoch + pt::Duration::fromSecondsF(600.0);
    hop.position_m = r;
    hop.velocity_m_s = v;
  }
  EXPECT_LT((r_direct.eigen() - hop.position_m.eigen()).norm(), 0.05)
      << "the shipped step is not converged at the metre level";
}

TEST(TargetPropagator, SigmaGrowsWithAgeAndIsSymmetric) {
  const pg::StateVectorSlot s = issLikeSlot();
  pg::TargetPropagator p;
  ASSERT_TRUE(p.setState(s));
  p.setForceModel(kJ2);
  EXPECT_NEAR(p.sigmaAt(s.epoch), s.sigma_at_epoch_m, 1e-9);

  const double ahead = p.sigmaAt(s.epoch + pt::Duration::fromSecondsF(3600.0));
  const double behind = p.sigmaAt(s.epoch - pt::Duration::fromSecondsF(3600.0));
  EXPECT_GT(ahead, s.sigma_at_epoch_m) << "an hour-old state must not claim its epoch accuracy";
  EXPECT_NEAR(ahead, behind, 1e-9) << "uncertainty is symmetric in |age|";
}

TEST(TargetPropagator, TheAccelerationIsTheGradientOfTheJ2Potential) {
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

  const Eigen::Vector3d a = j2AccelerationForTest(pm::Vec3<pmf::ECI>(r0)).eigen();
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

// ======================================================================
// The 8x8 default (Push 82)
// ======================================================================

TEST(TargetPropagator, DefaultsToTheEightByEightFieldTheVehicleAlreadyCarries) {
  pg::TargetPropagator p;
  EXPECT_EQ(p.forceModel().degree, 8);
  EXPECT_EQ(p.forceModel().order, 8);
  EXPECT_TRUE(p.forceModel().needsEarthOrientation());

  // The J2 setting is the one that does not, which is what makes it the
  // fallback when the tables are unavailable rather than merely a coarser mode.
  EXPECT_FALSE(kJ2.needsEarthOrientation());
}

TEST(TargetPropagator, ClampsAnOutOfRangeFieldRatherThanRefusingIt) {
  pg::TargetPropagator p;
  p.setForceModel({99, 99});
  EXPECT_EQ(p.forceModel().degree, pg::kGeopotentialMaxDegree);
  EXPECT_EQ(p.forceModel().order, pg::kGeopotentialMaxDegree);
  p.setForceModel({4, 9});  // order above degree is meaningless
  EXPECT_EQ(p.forceModel().degree, 4);
  EXPECT_EQ(p.forceModel().order, 4);
  p.setForceModel({-1, -1});
  EXPECT_EQ(p.forceModel().degree, 0);
  EXPECT_EQ(p.forceModel().order, 0);
}

TEST(TargetPropagator, TheDefaultFieldMovesTheAnswerAwayFromJ2) {
  // The whole reason for the change: if 8x8 and J2 agreed, the default would be
  // costing Earth-orientation data for nothing.
  const pg::StateVectorSlot s = issLikeSlot();
  const polaris::frames::EopValue eop{0.0, 0.0, 0.0};

  pg::TargetPropagator j2;
  j2.setForceModel(kJ2);
  ASSERT_TRUE(j2.setState(s));
  pg::TargetPropagator full;
  ASSERT_TRUE(full.setState(s));

  pm::Vec3<pmf::ECI> r_j2, v_j2, r_full, v_full;
  const pt::Tai t = s.epoch + pt::Duration::fromSecondsF(900.0);
  ASSERT_EQ(j2.propagate(t, nullptr, r_j2, v_j2), pg::PropagationStatus::kOk);
  ASSERT_EQ(full.propagate(t, &eop, r_full, v_full), pg::PropagationStatus::kOk);

  const double moved = (r_full.eigen() - r_j2.eigen()).norm();
  // Tens of metres over a quarter hour for a LEO target -- the same order the
  // scoping measurement found, and far above numerical noise.
  EXPECT_GT(moved, 1.0) << "8x8 and J2 agree to " << moved << " m, so the default buys nothing";
  EXPECT_LT(moved, 5000.0) << "8x8 and J2 differ by " << moved << " m, which is not a truncation";
}

TEST(TargetPropagator, WithoutEarthOrientationItFallsBackToOrderZeroForThatCall) {
  // Losing the tables must degrade the answer, not withdraw it: a target
  // position that disappears because EOP went stale is a target the operator
  // cannot point at for a reason unrelated to the target.
  const pg::StateVectorSlot s = issLikeSlot();
  pg::TargetPropagator full;
  ASSERT_TRUE(full.setState(s));  // 8x8 by default
  pg::TargetPropagator zonal;
  zonal.setForceModel({8, 0});
  ASSERT_TRUE(zonal.setState(s));

  const pt::Tai t = s.epoch + pt::Duration::fromSecondsF(600.0);
  pm::Vec3<pmf::ECI> r_no_eop, v_no_eop, r_zonal, v_zonal;
  ASSERT_EQ(full.propagate(t, nullptr, r_no_eop, v_no_eop), pg::PropagationStatus::kOk);
  ASSERT_EQ(zonal.propagate(t, nullptr, r_zonal, v_zonal), pg::PropagationStatus::kOk);

  // Identical: the fallback is exactly "evaluate the same degree at order 0",
  // not some third behaviour.
  EXPECT_LT((r_no_eop.eigen() - r_zonal.eigen()).norm(), 1.0e-9);
  // And the configured model is untouched, so fidelity returns with the tables.
  EXPECT_EQ(full.forceModel().order, 8);
}

TEST(TargetPropagator, TheCursorDoesNotMakeTheAnswerDependOnCallOrder) {
  // The invariant the header promises, now that propagation carries state
  // between calls. Bit-identical, not merely close: the retained state sits on a
  // grid anchored at the slot epoch, so the steps taken to reach a given time
  // are the same whatever was asked before it.
  const pg::StateVectorSlot s = issLikeSlot();
  const polaris::frames::EopValue eop{0.0, 0.0, 0.0};
  const pt::Tai target = s.epoch + pt::Duration::fromSecondsF(1234.5);

  pg::TargetPropagator direct;
  ASSERT_TRUE(direct.setState(s));
  pm::Vec3<pmf::ECI> r_direct, v_direct;
  ASSERT_EQ(direct.propagate(target, &eop, r_direct, v_direct), pg::PropagationStatus::kOk);

  pg::TargetPropagator walked;
  ASSERT_TRUE(walked.setState(s));
  pm::Vec3<pmf::ECI> r, v;
  // Negative spans included deliberately. The earlier version of this test used
  // only non-negative times, which never visits the backward branch — and that
  // branch was the asymmetric one: a walk to step -10 followed by a request at
  // -5 integrated forward from -10 instead of re-seeding, and RK4 is not
  // time-reversible. A look-ahead upload makes negative spans routine, so this
  // is not an exotic path.
  for (const double t : {7.0, 33.0, 100.0, 617.25, 1000.0, 1234.5}) {
    ASSERT_EQ(walked.propagate(s.epoch + pt::Duration::fromSecondsF(t), &eop, r, v),
              pg::PropagationStatus::kOk);
  }
  EXPECT_EQ(r.eigen().x(), r_direct.eigen().x());
  EXPECT_EQ(r.eigen().y(), r_direct.eigen().y());
  EXPECT_EQ(r.eigen().z(), r_direct.eigen().z());

  // Including after a backwards excursion, which re-seeds.
  ASSERT_EQ(walked.propagate(s.epoch + pt::Duration::fromSecondsF(50.0), &eop, r, v),
            pg::PropagationStatus::kOk);
  ASSERT_EQ(walked.propagate(target, &eop, r, v), pg::PropagationStatus::kOk);
  EXPECT_EQ(r.eigen().x(), r_direct.eigen().x());
  EXPECT_EQ(r.eigen().z(), r_direct.eigen().z());
}

TEST(TargetPropagator, AMomentaryEopGapDoesNotPoisonEveryLaterAnswer) {
  // The cursor carries integration state forward, and the *effective* model is
  // the configured one only while Earth orientation is available. Continuing a
  // cursor across that change makes the trajectory a history of table
  // availability rather than the result of a model — measured at 0.4 m of
  // permanent offset from a single 10 s cycle without EOP, and it never
  // recovers, because the pollution rides the cursor forward.
  //
  // Both sequences below ask for the same final time with EOP available on that
  // call; only the middle of the run differs.
  const pg::StateVectorSlot s = issLikeSlot();
  const polaris::frames::EopValue eop{0.0, 0.0, 0.0};
  const pt::Tai t_end = s.epoch + pt::Duration::fromSecondsF(600.0);

  pg::TargetPropagator clean;
  ASSERT_TRUE(clean.setState(s));
  pm::Vec3<pmf::ECI> r, v;
  for (double t = 10.0; t <= 600.0; t += 10.0) {
    ASSERT_EQ(clean.propagate(s.epoch + pt::Duration::fromSecondsF(t), &eop, r, v),
              pg::PropagationStatus::kOk);
  }
  const Eigen::Vector3d after_clean = r.eigen();

  pg::TargetPropagator gapped;
  ASSERT_TRUE(gapped.setState(s));
  for (double t = 10.0; t <= 600.0; t += 10.0) {
    const bool have_eop = !(t > 295.0 && t < 305.0);  // one step without tables
    ASSERT_EQ(
        gapped.propagate(s.epoch + pt::Duration::fromSecondsF(t), have_eop ? &eop : nullptr, r, v),
        pg::PropagationStatus::kOk);
  }
  const Eigen::Vector3d after_gap = r.eigen();

  pg::TargetPropagator direct;
  ASSERT_TRUE(direct.setState(s));
  pm::Vec3<pmf::ECI> r_direct, v_direct;
  ASSERT_EQ(direct.propagate(t_end, &eop, r_direct, v_direct), pg::PropagationStatus::kOk);

  // Bit-identical in both cases: the gap re-seeds rather than being carried.
  EXPECT_EQ(after_clean.x(), r_direct.eigen().x());
  EXPECT_EQ(after_clean.z(), r_direct.eigen().z());
  EXPECT_EQ(after_gap.x(), r_direct.eigen().x())
      << "a momentary EOP gap left " << (after_gap - r_direct.eigen()).norm()
      << " m of permanent offset";
  EXPECT_EQ(after_gap.z(), r_direct.eigen().z());
}

TEST(TargetPropagator, BackwardsPropagationIsOrderIndependentToo) {
  const pg::StateVectorSlot s = issLikeSlot();
  const polaris::frames::EopValue eop{0.0, 0.0, 0.0};
  const pt::Tai target = s.epoch - pt::Duration::fromSecondsF(50.0);

  pg::TargetPropagator direct;
  ASSERT_TRUE(direct.setState(s));
  pm::Vec3<pmf::ECI> r_direct, v;
  ASSERT_EQ(direct.propagate(target, &eop, r_direct, v), pg::PropagationStatus::kOk);

  // Deeper into the past first, then back toward the epoch — the ordering that
  // used to integrate forward from the deeper cursor instead of re-seeding.
  pg::TargetPropagator walked;
  ASSERT_TRUE(walked.setState(s));
  pm::Vec3<pmf::ECI> r;
  for (const double t : {-100.0, -50.0, -300.0, -50.0, -7.5, -50.0}) {
    ASSERT_EQ(walked.propagate(s.epoch + pt::Duration::fromSecondsF(t), &eop, r, v),
              pg::PropagationStatus::kOk);
  }
  EXPECT_EQ(r.eigen().x(), r_direct.eigen().x());
  EXPECT_EQ(r.eigen().y(), r_direct.eigen().y());
  EXPECT_EQ(r.eigen().z(), r_direct.eigen().z());

  // And crossing zero: forward, then backward, then forward again.
  pg::TargetPropagator crossing;
  ASSERT_TRUE(crossing.setState(s));
  for (const double t : {120.0, -50.0, 300.0}) {
    ASSERT_EQ(crossing.propagate(s.epoch + pt::Duration::fromSecondsF(t), &eop, r, v),
              pg::PropagationStatus::kOk);
  }
  pg::TargetPropagator plain;
  ASSERT_TRUE(plain.setState(s));
  pm::Vec3<pmf::ECI> r_plain;
  ASSERT_EQ(plain.propagate(s.epoch + pt::Duration::fromSecondsF(300.0), &eop, r_plain, v),
            pg::PropagationStatus::kOk);
  EXPECT_EQ(r.eigen().x(), r_plain.eigen().x());
  EXPECT_EQ(r.eigen().z(), r_plain.eigen().z());
}

TEST(TargetPropagator, ChangingTheFieldResetsTheCursor) {
  // Otherwise a slot would carry a trajectory that is the history of a setting
  // rather than the result of a model.
  const pg::StateVectorSlot s = issLikeSlot();
  const polaris::frames::EopValue eop{0.0, 0.0, 0.0};
  const pt::Tai t = s.epoch + pt::Duration::fromSecondsF(900.0);

  pg::TargetPropagator a;
  ASSERT_TRUE(a.setState(s));
  pm::Vec3<pmf::ECI> warm, v1;
  ASSERT_EQ(a.propagate(t, &eop, warm, v1), pg::PropagationStatus::kOk);
  a.setForceModel(kJ2);
  pm::Vec3<pmf::ECI> after, v2;
  ASSERT_EQ(a.propagate(t, nullptr, after, v2), pg::PropagationStatus::kOk);

  pg::TargetPropagator fresh;
  fresh.setForceModel(kJ2);
  ASSERT_TRUE(fresh.setState(s));
  pm::Vec3<pmf::ECI> cold, v3;
  ASSERT_EQ(fresh.propagate(t, nullptr, cold, v3), pg::PropagationStatus::kOk);
  EXPECT_EQ(after.eigen().x(), cold.eigen().x());
  EXPECT_EQ(after.eigen().y(), cold.eigen().y());
  EXPECT_EQ(after.eigen().z(), cold.eigen().z());
}
