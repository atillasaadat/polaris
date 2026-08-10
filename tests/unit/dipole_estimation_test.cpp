/// @file Unit tests for the §8.5 tier-3 residual-dipole estimator
/// (lib/gnc/dipole_estimation). REQ-ACTL-011.
///
/// What these pin, beyond "the least-squares solve is right":
///
///  * **Observability is tested, not assumed.** The measurement `tau = -[Bx] m`
///    is rank 2 at every instant, so the estimator must refuse while the field
///    has not turned — and must name *which* gate closed, since "keep tumbling"
///    and "keep collecting" are different instructions to the ground.
///  * **The sigma has to be honest about what it covers.** It is checked against
///    the empirical scatter of the fit under a known observer noise, and the
///    non-dipole torque it cannot see is exercised separately: an aero-like
///    body-fixed torque is shown to leak into the fit, which is why the
///    cleanliness bound exists.
///  * **The forgetting factor is a tradeoff, pinned in both directions**: a short
///    memory tracks a power-state step and a long one lags it.
///  * **The convergence time is measured, not asserted from theory** — it is the
///    number that tells the ground how long a calibration pass must run, and on
///    the reference vehicle it is the number that says tier 3 is a *diagnostic*
///    for an anomaly-class dipole rather than a refinement of a clean vehicle's
///    feedforward.

#include "gnc/dipole_estimation.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <Eigen/Geometry>

#include "math/frames.hpp"
#include "math/typed_vector.hpp"
#include "random/rng.hpp"

namespace {

namespace pm = polaris::math;
namespace gnc = polaris::gnc;
using Body = pm::frames::Body;
using Vec3B = pm::Vec3<Body>;

constexpr std::int64_t kNsPerSecond = 1000000000LL;

/// 500 km SSO: period 5677 s, mean motion 1.107e-3 rad/s.
constexpr double kOrbitPeriodS = 5677.0;
constexpr double kMeanMotion = 2.0 * 3.14159265358979323846 / kOrbitPeriodS;

/// The reference vehicle's magnetic-cleanliness allocation,
/// `config/spacecraft/leo_smallsat.yaml: spacecraft.residual_dipole_am2`.
const Eigen::Vector3d kReferenceDipole(0.002, -0.001, 0.0015);

std::int64_t tag(double t_s) {
  return static_cast<std::int64_t>(t_s * 1.0e9);
}

/// A schematic LEO field in inertially-fixed body axes.
///
/// **Direction geometry only.** A tilted-dipole field seen from an inclined
/// circular orbit turns roughly twice per orbit in the orbit plane, with a
/// smaller out-of-plane component at the orbit rate; the magnitude sweeps the
/// IGRF band the flight gate admits (2.2e-5 to 5.2e-5 T). That is exactly the
/// content the estimator's observability depends on, and nothing here needs the
/// field to be an IGRF evaluation — the truth-model version of this question is
/// SITL's, against `sim/world/magnetic_field`.
Vec3B leoField(double t_s) {
  const Eigen::Vector3d u(std::sin(2.0 * kMeanMotion * t_s),
                          0.6 * std::cos(2.0 * kMeanMotion * t_s),
                          0.35 * std::sin(kMeanMotion * t_s));
  const double mag = 3.5e-5 + 1.2e-5 * std::cos(2.0 * kMeanMotion * t_s);
  return Vec3B(mag * u.normalized());
}

/// The reference vehicle's tier-3 tuning.
///
/// `torque_sigma_nm` is the tier-2 observer's *filtered* floor: a 10 Hz
/// difference quotient of `J*omega` carries ~2e-4 N·m of STIM300 rate noise, and
/// the committed `ObserverTauSec` = 200 s low pass brings that to
/// `2e-4*sqrt(0.1/(2*200))` = 3.2e-6 N·m (design doc §8.5, "the anomaly budget is
/// set by the observer's noise floor"). `min_sample_interval_s` is twice that
/// time constant, so consecutive accumulations carry independent noise.
/// `min_information` = 4 is the policy that a published fit must carry
/// `sigma_worst <= sigma_tau/(B_nom*sqrt(4))` = 0.05 A·m², a twentieth of the
/// anomaly-class dipole this estimator can actually resolve.
gnc::DipoleEstimatorConfig referenceConfig() {
  gnc::DipoleEstimatorConfig c;
  c.nominal_field_t = 3.0e-5;
  c.min_field_t = 2.2e-5;
  c.max_field_t = 5.2e-5;
  c.forgetting_time_s = 3.0 * kOrbitPeriodS;
  c.min_sample_interval_s = 400.0;
  c.min_observability = 0.05;
  c.min_information = 4.0;
  c.torque_sigma_nm = 3.2e-6;
  c.max_dipole_am2 = 1.0;
  return c;
}

/// Feed `[0, duration_s)` at `cadence_s`, with `m` the truth dipole and an
/// optional extra torque and per-axis Gaussian observer noise. Returns the last
/// result seen (valid or not).
struct RunOptions {
  double duration_s = 4.0 * kOrbitPeriodS;
  double cadence_s = 400.0;
  double noise_nm = 0.0;
  Eigen::Vector3d extra_torque_nm = Eigen::Vector3d::Zero();
  double start_s = 0.0;
};

gnc::DipoleResult run(gnc::DipoleEstimator& est, const Eigen::Vector3d& m, const RunOptions& opt,
                      polaris::random::SplitMix64* rng = nullptr) {
  gnc::DipoleResult out;
  for (double t = opt.start_s; t < opt.start_s + opt.duration_s; t += opt.cadence_s) {
    const Vec3B field = leoField(t);
    Eigen::Vector3d tau = m.cross(field.eigen()) + opt.extra_torque_nm;
    if (rng != nullptr && opt.noise_nm > 0.0) {
      for (int i = 0; i < 3; ++i) {
        tau[i] += opt.noise_nm * rng->gaussian();
      }
    }
    est.update(Vec3B(tau), field, tag(t), out);
  }
  return out;
}

// ---------------------------------------------------------------------------
// Recovery
// ---------------------------------------------------------------------------

TEST(DipoleEstimation, RecoversAKnownDipoleExactlyFromANoiselessRotatingField) {
  gnc::DipoleEstimator est(referenceConfig());
  const Eigen::Vector3d truth(0.30, -0.15, 0.22);
  const gnc::DipoleResult out = run(est, truth, RunOptions{});

  ASSERT_TRUE(out.valid) << "refusal " << static_cast<int>(out.refusal);
  EXPECT_TRUE(est.hasEstimate());
  // The only error here is round-off: the model is exact and the data noiseless.
  EXPECT_LT((out.dipole_am2.eigen() - truth).norm(), 1.0e-12);
  // The component along B is unobservable instantaneously, so this is only true
  // because the field turned; the next test is the same run without that.
  EXPECT_GT(out.observability, referenceConfig().min_observability);
}

TEST(DipoleEstimation, RecoversTheComponentAlongTheInitialFieldToo) {
  // A dipole deliberately parallel to B(0): the very component a single-instant
  // fit cannot see. If the accumulator ever leaked a pseudo-inverse's null-space
  // choice into the answer, this is the case that would show it.
  gnc::DipoleEstimator est(referenceConfig());
  const Eigen::Vector3d truth = 0.4 * leoField(0.0).eigen().normalized();
  const gnc::DipoleResult out = run(est, truth, RunOptions{});

  ASSERT_TRUE(out.valid);
  EXPECT_LT((out.dipole_am2.eigen() - truth).norm(), 1.0e-12);
}

// ---------------------------------------------------------------------------
// Observability
// ---------------------------------------------------------------------------

TEST(DipoleEstimation, RefusesWhileTheFieldHasNotTurned) {
  gnc::DipoleEstimator est(referenceConfig());
  const Eigen::Vector3d truth(0.30, -0.15, 0.22);
  const Vec3B fixed_field = leoField(0.0);

  gnc::DipoleResult out;
  for (int k = 0; k < 200; ++k) {
    const double t = 400.0 * static_cast<double>(k);
    est.update(Vec3B(truth.cross(fixed_field.eigen())), fixed_field, tag(t), out);
  }
  // 200 samples, so no shortage of evidence — and still rank 2, because every
  // one of them looked in the same direction.
  EXPECT_FALSE(out.valid);
  EXPECT_EQ(out.refusal, gnc::DipoleRefusal::kNoObservability);
  EXPECT_NEAR(out.observability, 0.0, 1.0e-12);
  EXPECT_GT(out.effective_samples, 10.0);
  EXPECT_FALSE(est.hasEstimate());
  EXPECT_EQ(est.estimate().eigen(), Eigen::Vector3d::Zero());
}

TEST(DipoleEstimation, RefusesForLackOfEvidenceOnceTheGeometryIsGood) {
  // Same geometry as the recovery test, but the information gate raised so high
  // that no realistic pass clears it. The refusal must name evidence, not
  // geometry — the two instructions to the ground are different.
  gnc::DipoleEstimatorConfig cfg = referenceConfig();
  cfg.min_information = 1.0e6;
  gnc::DipoleEstimator est(cfg);
  const gnc::DipoleResult out = run(est, Eigen::Vector3d(0.30, -0.15, 0.22), RunOptions{});

  EXPECT_FALSE(out.valid);
  EXPECT_EQ(out.refusal, gnc::DipoleRefusal::kInsufficientInformation);
  EXPECT_GT(out.observability, cfg.min_observability);
  EXPECT_GT(out.information, 0.0);
}

TEST(DipoleEstimation, ObservabilityClimbsAsTheFieldTurns) {
  // What a ground operator watches during a calibration pass. Monotonicity is
  // not claimed (the field magnitude varies, so an individual sample can shift
  // the worst axis); the pass-level climb is.
  gnc::DipoleEstimator est(referenceConfig());
  gnc::DipoleResult out;
  double quarter_orbit = 0.0;
  double full_orbit = 0.0;
  for (double t = 0.0; t < kOrbitPeriodS; t += 100.0) {
    const Vec3B field = leoField(t);
    est.update(Vec3B(Eigen::Vector3d(0.3, -0.15, 0.22).cross(field.eigen())), field, tag(t), out);
    if (t < 0.25 * kOrbitPeriodS) {
      quarter_orbit = out.observability;
    }
    full_orbit = out.observability;
  }
  EXPECT_GT(full_orbit, quarter_orbit);
  EXPECT_GT(full_orbit, referenceConfig().min_observability);
}

// ---------------------------------------------------------------------------
// Noise and the honesty of the sigma
// ---------------------------------------------------------------------------

TEST(DipoleEstimation, ScatterUnderObserverNoiseMatchesThePublishedSigma) {
  // 40 independent one-day passes on an anomaly-class dipole. The claim is not
  // that any single fit is close, but that the published sigma is the right
  // *size* — a sigma that were an order out would make every downstream decision
  // (feed it forward? raise the event?) wrong in a way no single run shows.
  const gnc::DipoleEstimatorConfig cfg = referenceConfig();
  const Eigen::Vector3d truth(0.60, -0.25, 0.35);
  constexpr int kRuns = 40;

  Eigen::Vector3d sum_sq_error = Eigen::Vector3d::Zero();
  Eigen::Vector3d last_sigma = Eigen::Vector3d::Zero();
  int valid_runs = 0;
  for (int r = 0; r < kRuns; ++r) {
    polaris::random::SplitMix64 rng(
        polaris::random::streamSeed(0xD190E5u, static_cast<std::uint64_t>(r)));
    gnc::DipoleEstimator est(cfg);
    RunOptions opt;
    opt.duration_s = 86400.0;
    opt.noise_nm = cfg.torque_sigma_nm;
    const gnc::DipoleResult out = run(est, truth, opt, &rng);
    ASSERT_TRUE(out.valid) << "run " << r << " refusal " << static_cast<int>(out.refusal);
    sum_sq_error += (out.dipole_am2.eigen() - truth).cwiseAbs2();
    last_sigma = out.sigma_am2.eigen();
    ++valid_runs;
  }
  ASSERT_EQ(valid_runs, kRuns);
  const Eigen::Vector3d rms = (sum_sq_error / kRuns).cwiseSqrt();
  for (int i = 0; i < 3; ++i) {
    // The published sigma is conservative by ~sqrt(2) under exponential
    // forgetting (documented), so it must bound the scatter and must not bound
    // it by more than a factor of a few.
    EXPECT_LT(rms[i], 2.0 * last_sigma[i]) << "axis " << i;
    EXPECT_GT(rms[i], 0.15 * last_sigma[i]) << "axis " << i;
  }
}

TEST(DipoleEstimation, AbsorbsANonDipoleTorqueThatNoAmountOfAveragingRemoves) {
  // The observer's estimate is every unmodelled torque, not just the dipole's,
  // and the part of it that correlates with B lands in the fit as a **bias**.
  //
  // The sharp statement of "a covariance cannot see a bias" is not that the bias
  // is larger than sigma — at any one setting that is a matter of tuning — but
  // that the two respond differently to accumulating more evidence: sigma falls
  // as more information accumulates, the bias does not move at all. Ten times
  // the memory is ~sqrt(10) less sigma and the *same* error, which is the whole
  // limitation in one comparison.
  const Eigen::Vector3d truth(0.60, -0.25, 0.35);
  const double signature_nm = truth.norm() * 3.0e-5;

  const auto flyWithAero = [&](double forgetting_s) {
    gnc::DipoleEstimatorConfig cfg = referenceConfig();
    cfg.forgetting_time_s = forgetting_s;
    gnc::DipoleEstimator est(cfg);
    RunOptions opt;
    // Ten memory horizons, so the accumulator is at its steady state either way.
    opt.duration_s = 10.0 * forgetting_s;
    opt.extra_torque_nm = Eigen::Vector3d(0.15, -0.1, 0.05).normalized() * (0.15 * signature_nm);
    return run(est, truth, opt);
  };

  const gnc::DipoleResult shrt = flyWithAero(3.0 * kOrbitPeriodS);
  const gnc::DipoleResult lng = flyWithAero(30.0 * kOrbitPeriodS);
  ASSERT_TRUE(shrt.valid);
  ASSERT_TRUE(lng.valid);

  const double short_error = (shrt.dipole_am2.eigen() - truth).norm();
  const double long_error = (lng.dipole_am2.eigen() - truth).norm();
  const double short_sigma = shrt.sigma_am2.eigen().norm();
  const double long_sigma = lng.sigma_am2.eigen().norm();

  // The bias is there and it does not average down.
  EXPECT_GT(short_error, 0.01) << "the leak this test exists to demonstrate did not happen";
  EXPECT_NEAR(long_error, short_error, 0.25 * short_error);
  // The sigma, meanwhile, falls as sqrt(information) — so it says nothing about
  // the error that is actually left.
  EXPECT_LT(long_sigma, 0.5 * short_sigma);
  std::printf(
      "[ dipole ] non-dipole leak: |error| = %.3e (3 orbits memory) vs %.3e (30), "
      "sigma = %.3e vs %.3e A·m²\n",
      short_error, long_error, short_sigma, long_sigma);
}

// ---------------------------------------------------------------------------
// Forgetting and drift
// ---------------------------------------------------------------------------

TEST(DipoleEstimation, ShortMemoryTracksAPowerStateStepAndLongMemoryLagsIt) {
  // The physical moment moves with the power state. Both directions of the
  // tradeoff, measured from one step and read at the *same* absolute time: the
  // residual weight on pre-step data is exp(-T/T_f), so a memory of three orbits
  // has forgotten it nine orbits later and a memory of thirty has not.
  const Eigen::Vector3d before(0.60, -0.25, 0.35);
  const Eigen::Vector3d after(0.20, 0.30, -0.40);
  const double step_s = 6.0 * kOrbitPeriodS;
  const double settle_s = 9.0 * kOrbitPeriodS;  // 3 x the short memory

  const auto flyStep = [&](double forgetting_s) {
    gnc::DipoleEstimatorConfig cfg = referenceConfig();
    cfg.forgetting_time_s = forgetting_s;
    gnc::DipoleEstimator est(cfg);
    gnc::DipoleResult out;
    for (double t = 0.0; t < step_s + settle_s; t += cfg.min_sample_interval_s) {
      const Vec3B field = leoField(t);
      const Eigen::Vector3d m = (t < step_s) ? before : after;
      est.update(Vec3B(m.cross(field.eigen())), field, tag(t), out);
    }
    return (out.dipole_am2.eigen() - after).norm();
  };

  const double step_size = (after - before).norm();
  const double short_memory_error = flyStep(3.0 * kOrbitPeriodS);
  const double long_memory_error = flyStep(30.0 * kOrbitPeriodS);

  EXPECT_LT(short_memory_error, 0.15 * step_size);
  EXPECT_GT(long_memory_error, 3.0 * short_memory_error);
  std::printf(
      "[ dipole ] power-state step of %.2f A·m², read %.1f orbits later: error %.3e "
      "(3 orbits memory) vs %.3e (30)\n",
      step_size, settle_s / kOrbitPeriodS, short_memory_error, long_memory_error);
}

TEST(DipoleEstimation, TooShortAMemoryStarvesTheFitWhileALongerOneOnTheSameDataSucceeds) {
  // The lower bound on the forgetting time. Two bounds are in play and it is
  // worth being exact about which one binds *here*: the field must turn inside
  // the memory window (it does — one 400 s sample interval already turns this
  // field ~50°, since it turns twice per 5677 s orbit), and the window must hold
  // enough samples for the evidence gate. On this vehicle the second binds, so
  // the refusal names evidence, not geometry. Both are refusals to publish,
  // which is the property that matters, and asserting the one that actually
  // fires is what stops this test from quietly becoming decorative.
  const Eigen::Vector3d truth(0.60, -0.25, 0.35);

  gnc::DipoleEstimatorConfig starved = referenceConfig();
  starved.forgetting_time_s = 0.05 * kOrbitPeriodS;  // shorter than one sample interval
  gnc::DipoleEstimator starved_est(starved);
  const gnc::DipoleResult starved_out = run(starved_est, truth, RunOptions{});
  EXPECT_FALSE(starved_out.valid);
  EXPECT_EQ(starved_out.refusal, gnc::DipoleRefusal::kInsufficientInformation);
  EXPECT_LT(starved_out.effective_samples, 2.0);
  EXPECT_FALSE(starved_est.hasEstimate());

  // Same data, same gates, a memory that spans the pass: it publishes.
  gnc::DipoleEstimator ok(referenceConfig());
  EXPECT_TRUE(run(ok, truth, RunOptions{}).valid);
}

TEST(DipoleEstimation, PrecisionSaturatesAtTheForgettingHorizonRatherThanAtThePassLength) {
  // The consequence of forgetting that decides how a calibration pass is
  // commanded, and the one that is easiest to get wrong by analogy with a
  // growing-memory least squares: the information matrix reaches a steady state
  // at ~T_f/dt effective samples, so **a longer pass buys nothing** and the
  // sigma floor is set by T_f alone. The ground therefore trades drift tracking
  // against precision directly, and "run the pass longer" is not a lever.
  const Eigen::Vector3d truth(0.60, -0.25, 0.35);

  const auto sigmaAfter = [&](double forgetting_s, double duration_s) {
    gnc::DipoleEstimatorConfig cfg = referenceConfig();
    cfg.forgetting_time_s = forgetting_s;
    gnc::DipoleEstimator est(cfg);
    RunOptions opt;
    opt.duration_s = duration_s;
    const gnc::DipoleResult out = run(est, truth, opt);
    EXPECT_TRUE(out.valid);
    return out.sigma_am2.eigen().norm();
  };

  const double t_f = 3.0 * kOrbitPeriodS;
  const double one_horizon = sigmaAfter(t_f, 5.0 * t_f);
  const double four_horizons = sigmaAfter(t_f, 20.0 * t_f);
  EXPECT_NEAR(four_horizons, one_horizon, 0.02 * one_horizon);

  // Ten times the memory is sqrt(10) = 3.16 times less sigma, at the same pass
  // length. That is the lever that exists.
  const double long_memory = sigmaAfter(10.0 * t_f, 50.0 * t_f);
  EXPECT_NEAR(long_memory, one_horizon / std::sqrt(10.0), 0.1 * one_horizon / std::sqrt(10.0));
  std::printf(
      "[ dipole ] sigma floor: %.3e A·m² at T_f = 3 orbits (unchanged over a 4x longer "
      "pass), %.3e at T_f = 30 orbits\n",
      one_horizon, long_memory);
}

// ---------------------------------------------------------------------------
// The bound that stops the estimate making pointing worse
// ---------------------------------------------------------------------------

TEST(DipoleEstimation, RefusesAFitOutsideTheCleanlinessBound) {
  gnc::DipoleEstimatorConfig cfg = referenceConfig();
  cfg.max_dipole_am2 = 0.5;
  gnc::DipoleEstimator est(cfg);
  const Eigen::Vector3d truth(0.60, -0.25, 0.35);  // norm 0.744, past the bound
  const gnc::DipoleResult out = run(est, truth, RunOptions{});

  EXPECT_FALSE(out.valid);
  EXPECT_EQ(out.refusal, gnc::DipoleRefusal::kOutOfBounds);
  // Refused, not clamped: nothing at the bound's magnitude is published, and the
  // caller keeps flying the configured allocation.
  EXPECT_FALSE(est.hasEstimate());
  EXPECT_EQ(est.estimate().eigen(), Eigen::Vector3d::Zero());
  // The gates that come first still passed, so the refusal is about the bound
  // and the telemetry says so.
  EXPECT_GT(out.observability, cfg.min_observability);
  EXPECT_GT(out.information, cfg.min_information);
}

TEST(DipoleEstimation, NothingOutsideTheBoundIsEverPublished) {
  // The property the bound has to have, stated over the whole trajectory rather
  // than at one instant. A dipole that grows past the bound is *tracked* while
  // the forgetting window is a mixture of the old and new values — every one of
  // those intermediate fits is a real fit, inside the bound, and publishing them
  // is correct — and then refused. What must never happen is a published value
  // outside the bound, on any cycle.
  gnc::DipoleEstimatorConfig cfg = referenceConfig();
  cfg.max_dipole_am2 = 0.5;
  cfg.forgetting_time_s = 1.5 * kOrbitPeriodS;
  gnc::DipoleEstimator est(cfg);

  const Eigen::Vector3d small(0.10, -0.05, 0.07);
  RunOptions good;
  good.duration_s = 4.0 * kOrbitPeriodS;
  ASSERT_TRUE(run(est, small, good).valid);
  EXPECT_LT((est.estimate().eigen() - small).norm(), 1.0e-9);

  gnc::DipoleResult out;
  const Eigen::Vector3d huge(2.0, -1.0, 1.5);
  for (double t = good.duration_s; t < good.duration_s + 6.0 * kOrbitPeriodS;
       t += cfg.min_sample_interval_s) {
    const Vec3B field = leoField(t);
    est.update(Vec3B(huge.cross(field.eigen())), field, tag(t), out);
    EXPECT_LE(est.estimate().eigen().norm(), cfg.max_dipole_am2);
    if (out.valid) {
      EXPECT_LE(out.dipole_am2.eigen().norm(), cfg.max_dipole_am2);
    }
  }
  // It ends refused — the true dipole is four times the bound — with the last
  // in-bounds fit still held, because a refusal is not evidence the dipole went
  // away.
  EXPECT_EQ(out.refusal, gnc::DipoleRefusal::kOutOfBounds);
  EXPECT_TRUE(est.hasEstimate());
  EXPECT_GT(est.estimate().eigen().norm(), 0.0);
}

// ---------------------------------------------------------------------------
// Input hygiene
// ---------------------------------------------------------------------------

TEST(DipoleEstimation, RejectsFieldSamplesOutsideTheBand) {
  gnc::DipoleEstimator est(referenceConfig());
  gnc::DipoleResult out;
  // A magnetorquer near-field: hundreds of microtesla, and not a field the
  // vehicle's own dipole ever saw.
  EXPECT_FALSE(est.update(Vec3B(Eigen::Vector3d(1.0e-6, 0.0, 0.0)),
                          Vec3B(Eigen::Vector3d(3.0e-4, 0.0, 0.0)), tag(0.0), out));
  EXPECT_EQ(out.refusal, gnc::DipoleRefusal::kFieldOutOfRange);
  EXPECT_EQ(out.effective_samples, 0.0);

  // A dead sensor.
  EXPECT_FALSE(
      est.update(Vec3B(Eigen::Vector3d::Zero()), Vec3B(Eigen::Vector3d::Zero()), tag(1.0), out));
  EXPECT_EQ(out.refusal, gnc::DipoleRefusal::kFieldOutOfRange);
  EXPECT_EQ(out.effective_samples, 0.0);
}

TEST(DipoleEstimation, RejectsSamplesArrivingInsideTheDecorrelationInterval) {
  // Feeding the estimator at the 10 Hz control rate would be feeding it the same
  // low-passed value 4000 times per accepted sample, and the information matrix
  // would report that as evidence.
  gnc::DipoleEstimator est(referenceConfig());
  const Eigen::Vector3d truth(0.30, -0.15, 0.22);
  gnc::DipoleResult out;
  int accepted = 0;
  for (int k = 0; k < 5000; ++k) {
    const double t = 0.1 * static_cast<double>(k);
    const Vec3B field = leoField(t);
    est.update(Vec3B(truth.cross(field.eigen())), field, tag(t), out);
    if (out.refusal != gnc::DipoleRefusal::kSampleTooSoon) {
      ++accepted;
    }
  }
  // 500 s of wall time at a 400 s minimum interval: the first sample and one more.
  EXPECT_EQ(accepted, 2);
  EXPECT_LE(out.effective_samples, 2.0);
}

TEST(DipoleEstimation, RefusesNonMonotonicTimeWithoutMovingTheAnchor) {
  gnc::DipoleEstimator est(referenceConfig());
  const Eigen::Vector3d truth(0.30, -0.15, 0.22);
  gnc::DipoleResult out;
  const Vec3B f0 = leoField(0.0);
  est.update(Vec3B(truth.cross(f0.eigen())), f0, tag(0.0), out);
  const double after_first = out.effective_samples;

  const Vec3B f1 = leoField(1000.0);
  EXPECT_FALSE(est.update(Vec3B(truth.cross(f1.eigen())), f1, tag(-5.0), out));
  EXPECT_EQ(out.refusal, gnc::DipoleRefusal::kNonMonotonicTime);
  EXPECT_EQ(out.effective_samples, after_first);

  // The anchor did not move backwards, so a normal sample still lands.
  est.update(Vec3B(truth.cross(f1.eigen())), f1, tag(1000.0), out);
  EXPECT_GT(out.effective_samples, after_first);
}

TEST(DipoleEstimation, RefusesNonFiniteInputWithoutTouchingTheAccumulator) {
  gnc::DipoleEstimator est(referenceConfig());
  const Eigen::Vector3d truth(0.30, -0.15, 0.22);
  const gnc::DipoleResult good = run(est, truth, RunOptions{});
  ASSERT_TRUE(good.valid);

  gnc::DipoleResult out;
  const double nan = std::nan("");
  EXPECT_FALSE(est.update(Vec3B(Eigen::Vector3d(nan, 0.0, 0.0)), leoField(1.0e5), tag(1.0e5), out));
  EXPECT_EQ(out.refusal, gnc::DipoleRefusal::kBadInput);
  EXPECT_FALSE(est.update(Vec3B(Eigen::Vector3d::Zero()), Vec3B(Eigen::Vector3d(nan, 0.0, 0.0)),
                          tag(1.1e5), out));
  EXPECT_EQ(out.refusal, gnc::DipoleRefusal::kBadInput);
  // The good estimate survived both.
  EXPECT_TRUE(est.hasEstimate());
  EXPECT_LT((est.estimate().eigen() - truth).norm(), 1.0e-12);
}

TEST(DipoleEstimation, IsInertWithoutAValidConfiguration) {
  gnc::DipoleEstimator est{};
  EXPECT_FALSE(est.isConfigured());
  gnc::DipoleResult out;
  EXPECT_FALSE(est.update(Vec3B(Eigen::Vector3d::Zero()), leoField(0.0), tag(0.0), out));
  EXPECT_EQ(out.refusal, gnc::DipoleRefusal::kUnconfigured);

  gnc::DipoleEstimatorConfig bad = referenceConfig();
  bad.min_field_t = bad.max_field_t;  // empty band
  EXPECT_FALSE(gnc::DipoleEstimator(bad).isConfigured());
  bad = referenceConfig();
  bad.min_observability = 1.5;  // not a ratio
  EXPECT_FALSE(gnc::DipoleEstimator(bad).isConfigured());
  EXPECT_TRUE(gnc::DipoleEstimator(referenceConfig()).isConfigured());
}

TEST(DipoleEstimation, ResetDropsTheAccumulatorAndTheEstimate) {
  gnc::DipoleEstimator est(referenceConfig());
  ASSERT_TRUE(run(est, Eigen::Vector3d(0.30, -0.15, 0.22), RunOptions{}).valid);
  est.reset();
  EXPECT_FALSE(est.hasEstimate());
  EXPECT_EQ(est.estimate().eigen(), Eigen::Vector3d::Zero());
  EXPECT_EQ(est.sigma().eigen(), Eigen::Vector3d::Zero());

  gnc::DipoleResult out;
  const Vec3B f = leoField(0.0);
  est.update(Vec3B(Eigen::Vector3d(0.3, -0.15, 0.22).cross(f.eigen())), f, tag(0.0), out);
  EXPECT_EQ(out.refusal, gnc::DipoleRefusal::kNoObservability);
  EXPECT_EQ(out.effective_samples, 1.0);
}

// ---------------------------------------------------------------------------
// How long a calibration pass must be — the number the ground needs
// ---------------------------------------------------------------------------

/// Sim-seconds of field rotation until every axis is inside `tolerance` of
/// `truth` and stays there for the rest of the pass. Returns a negative value if
/// that never happens within `horizon_s`.
double convergenceTimeS(const gnc::DipoleEstimatorConfig& cfg, const Eigen::Vector3d& truth,
                        double tolerance, double horizon_s, double noise_nm,
                        polaris::random::SplitMix64* rng) {
  gnc::DipoleEstimator est(cfg);
  gnc::DipoleResult out;
  double converged_at = -1.0;
  for (double t = 0.0; t < horizon_s; t += cfg.min_sample_interval_s) {
    const Vec3B field = leoField(t);
    Eigen::Vector3d tau = truth.cross(field.eigen());
    if (rng != nullptr) {
      for (int i = 0; i < 3; ++i) {
        tau[i] += noise_nm * rng->gaussian();
      }
    }
    if (!est.update(Vec3B(tau), field, tag(t), out)) {
      converged_at = -1.0;
      continue;
    }
    const bool inside = ((out.dipole_am2.eigen() - truth).cwiseAbs().array() <=
                         tolerance * truth.cwiseAbs().array())
                            .all();
    if (inside) {
      if (converged_at < 0.0) {
        converged_at = t;
      }
    } else {
      converged_at = -1.0;
    }
  }
  return converged_at;
}

TEST(DipoleEstimation, ReferenceVehicleNoiselessPassIsGeometryLimited) {
  // With no observer noise the only thing left is the geometry gate, so this is
  // the floor on any calibration pass: the field has to turn first, whatever the
  // dipole is. It is reported so the noisy numbers below can be read as
  // "geometry plus evidence" rather than as one opaque figure.
  const double t_conv = convergenceTimeS(referenceConfig(), kReferenceDipole, 0.10,
                                         4.0 * kOrbitPeriodS, 0.0, nullptr);
  ASSERT_GT(t_conv, 0.0);
  EXPECT_LT(t_conv, kOrbitPeriodS)
      << "noiseless convergence took longer than one orbit: " << t_conv << " s";
  std::printf("[ dipole ] noiseless, reference dipole: converged in %.0f s (%.2f orbits)\n", t_conv,
              t_conv / kOrbitPeriodS);
}

TEST(DipoleEstimation, TheReferenceVehiclesAllocationSizedDipoleIsBelowTheObserversNoiseFloor) {
  // The finding this whole module has to be honest about. The reference dipole
  // is 2.7e-3 A·m², i.e. an ~8e-8 N·m signature against a 3.2e-6 N·m observer
  // floor — an SNR of 0.025 per sample. A full day of pass does not get within
  // 10% of the smallest component, and the *published sigma says so*, which is
  // the behaviour that matters: the estimator does not claim a fit it does not
  // have.
  const gnc::DipoleEstimatorConfig cfg = referenceConfig();
  polaris::random::SplitMix64 rng(polaris::random::streamSeed(0xC1EA11u, 3));
  gnc::DipoleEstimator est(cfg);
  RunOptions opt;
  opt.duration_s = 86400.0;
  opt.noise_nm = cfg.torque_sigma_nm;
  const gnc::DipoleResult out = run(est, kReferenceDipole, opt, &rng);

  ASSERT_TRUE(out.valid);
  const double smallest = kReferenceDipole.cwiseAbs().minCoeff();
  EXPECT_GT(out.sigma_am2.eigen().maxCoeff(), smallest)
      << "the published sigma must not claim to resolve a dipole this small";
  std::printf(
      "[ dipole ] reference allocation |m| = %.2e A·m², one-day pass: sigma = "
      "[%.2e %.2e %.2e] A·m², |error| = %.2e A·m²\n",
      kReferenceDipole.norm(), out.sigma_am2.eigen()[0], out.sigma_am2.eigen()[1],
      out.sigma_am2.eigen()[2], (out.dipole_am2.eigen() - kReferenceDipole).norm());
}

TEST(DipoleEstimation, AnAnomalyClassDipoleConvergesInsideOneCalibrationPass) {
  // What tier 3 is actually for on this vehicle, and the number the ground needs.
  // `DisturbanceBudgetNm` = 2.0e-5 N·m is the §9 momentum-anomaly threshold,
  // which at a 3e-5 T field is a dipole of ~0.67 A·m². When that latches, this is
  // what names the offending moment in body axes.
  //
  // A *commanded calibration pass* is not the drift-tracking configuration: the
  // vehicle's power state is held for the duration, so the forgetting time is
  // lengthened to buy precision — the only lever that exists, per
  // PrecisionSaturatesAtTheForgettingHorizon. The reference drift-tracking
  // memory of 3 orbits leaves a 2.8e-2 A·m² sigma floor, which is 10 % of this
  // dipole's smallest component and therefore right at the tolerance; 30 orbits
  // takes it to 9e-3 and converges well inside a day.
  const Eigen::Vector3d anomaly(0.50, -0.30, 0.35);  // |m| = 0.68 A·m²
  gnc::DipoleEstimatorConfig pass = referenceConfig();
  pass.forgetting_time_s = 30.0 * kOrbitPeriodS;

  polaris::random::SplitMix64 rng(polaris::random::streamSeed(0xC1EA11u, 11));
  const double t_conv =
      convergenceTimeS(pass, anomaly, 0.10, 3.0 * 86400.0, pass.torque_sigma_nm, &rng);

  ASSERT_GT(t_conv, 0.0) << "an anomaly-class dipole must be resolvable";
  EXPECT_LT(t_conv, 86400.0) << "convergence took " << t_conv << " s";
  std::printf(
      "[ dipole ] anomaly-class |m| = %.2f A·m², T_f = 30 orbits: 10%% on all axes after "
      "%.0f s (%.1f h, %.1f orbits)\n",
      anomaly.norm(), t_conv, t_conv / 3600.0, t_conv / kOrbitPeriodS);
}

}  // namespace
