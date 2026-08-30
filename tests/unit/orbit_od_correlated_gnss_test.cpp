/// @file
/// @brief The orbit filter against a **correlated** GNSS error, and the `R`
/// repair that makes it honest (§8.3/§6.2; Push 77; REQ-ODP-014).
///
/// Every OD number this project has measured was taken against a receiver whose
/// position error is white. Real single-point GNSS error is not: residual
/// ionosphere, broadcast ephemeris and satellite clock are common to the
/// satellites in view and decorrelate over minutes (Misra & Enge §5 [misra2011];
/// Montenbruck & Gill §7.2 [montenbruck2000]). A filter that averages successive
/// fixes improves on white noise and does **not** improve on this, so it drives
/// its covariance below the error it actually has and then rejects honest fixes
/// on its own NIS gate.
///
/// This file measures that, and measures the repair: inflating `R` by the
/// correlated variance the receiver cannot report. The sim-side model is tested
/// in `tests/unit/sim_sensors_gnss_test.cpp`; the error is regenerated here so
/// the filter's behaviour is measured against a source this file fully controls.

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <Eigen/Core>
#include <Eigen/Geometry>

#include "constants/constants.hpp"
#include "frames/eci_ecef.hpp"
#include "frames/eop.hpp"
#include "gnc/orbit_od.hpp"
#include "math/typed_vector.hpp"
#include "time/leap_seconds.hpp"
#include "time/timescales.hpp"
#include "time/utc.hpp"

namespace {

namespace pc = polaris::constants;
namespace pf = polaris::frames;
namespace pg = polaris::gnc;
namespace pm = polaris::math;
namespace pmf = polaris::math::frames;
namespace pt = polaris::time;

using pg::GnssFix;
using pg::OrbitOd;
using pg::OrbitOdConfig;
using pg::OrbitOdRefusal;
using pg::OrbitOdResult;

constexpr double kAltitudeM = 400'000.0;
constexpr double kInclinationRad = 0.9006;

/// The receiver's error budget, split the way the model splits it. Per-axis
/// sigmas: an OEM7600-class 1.2 m horizontal 2D-RMS white part, and a 1.5 m
/// 2D-RMS correlated part on a 300 s timescale.
constexpr double kWhiteSigmaM = 1.2 / 1.4142135623730951;
constexpr double kCorrSigmaM = 1.5 / 1.4142135623730951;
double g_corr_tau_s = 300.0;  // NOLINT — scan knob
constexpr double kVelSigmaMps = 0.03;

/// Fix cadence and arc: 10 s fixes over two orbits, which is long enough for the
/// covariance to settle and for the correlated error to turn over ~37 times.
double g_fix_period_s = 10.0;  // NOLINT — scan knob, restored by each case
constexpr double kArcS = 11'000.0;

pt::Tai testEpoch() {
  pt::UtcDateTime utc;
  utc.year = 2026;
  utc.month = 1;
  utc.day = 1;
  return pt::taiFromUtc(utc, pt::LeapSecondTable::historical());
}

pt::Tai advance(const pt::Tai& base, double seconds) {
  return base + pt::Duration::fromSecondsF(seconds);
}

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

/// The flown filter configuration, with the `R` repair off unless a case sets it.
OrbitOdConfig baseConfig() {
  OrbitOdConfig cfg;
  cfg.mu_m3_per_s2 = pc::gravity::kGM;
  cfg.reference_radius_m = pc::gravity::kReferenceRadius;
  cfg.zonal_j2 = pc::gravity::kJ2;
  cfg.drag_ballistic_coeff_m2_per_kg = 0.011;
  cfg.drag_ref_density_kg_m3 = 3.725e-12;
  cfg.drag_ref_altitude_m = kAltitudeM;
  cfg.drag_scale_height_m = 58'515.0;
  cfg.accel_psd_m2_per_s3 = 1.8e-7;
  cfg.position_nis_gate = 16.27;  // chi^2_3 at 99.9%
  cfg.velocity_nis_gate = 16.27;
  cfg.max_coast_s = 1.0e9;
  cfg.max_degraded_coast_s = 1.0e9;
  cfg.max_dt_s = 60.0;
  cfg.max_step_s = 1.0;
  cfg.max_fix_latency_s = 0.2;
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

/// A small deterministic Gaussian source, so a case is reproducible and the
/// with/without pair sees the identical error realisation.
class Gauss {
 public:
  explicit Gauss(std::uint64_t seed) : s_(seed * 6364136223846793005ULL + 1442695040888963407ULL) {}

  double operator()() {
    // Box-Muller on a splitmix64 stream; cached second variate.
    if (have_) {
      have_ = false;
      return spare_;
    }
    const double u1 = std::max(1.0e-12, uniform());
    const double u2 = uniform();
    const double r = std::sqrt(-2.0 * std::log(u1));
    const double th = 2.0 * 3.14159265358979323846 * u2;
    spare_ = r * std::sin(th);
    have_ = true;
    return r * std::cos(th);
  }

 private:
  double uniform() {
    s_ += 0x9E3779B97F4A7C15ULL;
    std::uint64_t z = s_;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    z = z ^ (z >> 31);
    return static_cast<double>(z >> 11) * (1.0 / 9007199254740992.0);
  }

  std::uint64_t s_;
  double spare_ = 0.0;
  bool have_ = false;
};

/// What one arc produced. Everything the two cases are compared on.
struct ArcResult {
  std::uint32_t rejected = 0;  ///< NIS-gate refusals of honest fixes
  std::uint32_t accepted = 0;  ///< fixes folded in
  double mean_nees = 0.0;      ///< time-averaged 6-state NEES against truth
  double rms_pos_err_m = 0.0;  ///< RMS |estimate - truth| over the arc
};

/// Fly the filter over one arc against fixes carrying white + correlated error.
///
/// @param corr_sigma_m per-axis 1σ of the correlated component injected into the
///                     *fixes*. The receiver never reports it.
/// @param seed         the error realisation; the same seed gives the same
///                     errors, so a with/without pair differs only in `R`.
ArcResult flyArc(const OrbitOdConfig& cfg, double corr_sigma_m, std::uint64_t seed) {
  const pt::Tai epoch0 = testEpoch();
  const pf::EopValue eop = zeroEop(epoch0);

  Eigen::Vector3d r0;
  Eigen::Vector3d v0;
  circularState(r0, v0);

  // Truth: the filter's own model, unperturbed. The only thing separating the
  // estimate from it is the measurement error, which is what is under study.
  OrbitOd truth(baseConfig());
  EXPECT_EQ(truth.initialize(epoch0, pm::Vec3<pmf::ECI>(r0), pm::Vec3<pmf::ECI>(v0),
                             OrbitOd::Covariance::Identity()),
            OrbitOdRefusal::kNone);

  OrbitOd filter(cfg);
  EXPECT_TRUE(filter.isConfigured());
  OrbitOd::Covariance seed_cov = OrbitOd::Covariance::Identity();
  seed_cov.block<3, 3>(0, 0) *= 100.0;
  EXPECT_EQ(filter.initialize(epoch0, pm::Vec3<pmf::ECI>(r0), pm::Vec3<pmf::ECI>(v0), seed_cov),
            OrbitOdRefusal::kNone);

  Gauss g(seed);
  // Correlated error state, started from its stationary distribution.
  Eigen::Vector3d corr(corr_sigma_m * g(), corr_sigma_m * g(), corr_sigma_m * g());
  const double phi = std::exp(-g_fix_period_s / g_corr_tau_s);
  const double driving = std::sqrt(std::max(0.0, 1.0 - phi * phi));

  ArcResult out;
  double nees_sum = 0.0;
  double err_sq_sum = 0.0;
  int samples = 0;

  for (double t = g_fix_period_s; t <= kArcS; t += g_fix_period_s) {
    const pt::Tai now = advance(epoch0, t);
    while (truth.epoch() < now) {
      const double remaining = (now - truth.epoch()).seconds();
      const pt::Tai next = remaining <= 10.0 ? now : advance(truth.epoch(), 10.0);
      EXPECT_EQ(truth.propagate(next, eop), OrbitOdRefusal::kNone);
    }

    // Advance the correlated component one fix interval.
    for (int i = 0; i < 3; ++i) {
      corr(i) = phi * corr(i) + corr_sigma_m * driving * g();
    }
    const Eigen::Vector3d white(kWhiteSigmaM * g(), kWhiteSigmaM * g(), kWhiteSigmaM * g());
    const Eigen::Vector3d r_meas = truth.position().eigen() + white + corr;
    const Eigen::Vector3d v_meas =
        truth.velocity().eigen() +
        Eigen::Vector3d(kVelSigmaMps * g(), kVelSigmaMps * g(), kVelSigmaMps * g());

    pm::Vec3<pmf::ECEF> r_ecef;
    pm::Vec3<pmf::ECEF> v_ecef;
    EXPECT_TRUE(pf::ecefStateFromEci(now, eop, pm::Vec3<pmf::ECI>(r_meas),
                                     pm::Vec3<pmf::ECI>(v_meas), r_ecef, v_ecef));
    GnssFix fix;
    fix.time_tag = pt::toGps(now);
    fix.position_m = r_ecef;
    fix.velocity_m_s = v_ecef;
    // The receiver reports its **white** sigma only — it cannot see an error
    // common to every satellite it tracks. That is the whole premise.
    fix.position_sigma_h_m = kWhiteSigmaM;
    fix.position_sigma_v_m = kWhiteSigmaM;
    fix.velocity_sigma_m_s = kVelSigmaMps;
    fix.velocity_valid = true;

    OrbitOdResult res;
    filter.ingest(fix, eop, res);

    // Skip the opening transient: the seed covariance has to come down before
    // either the gate or the NEES says anything about steady state.
    if (t < 2000.0) {
      continue;
    }
    double nees = 0.0;
    if (filter.nees(truth.position(), truth.velocity(), nees) && std::isfinite(nees)) {
      nees_sum += nees;
      err_sq_sum += (filter.position().eigen() - truth.position().eigen()).squaredNorm();
      ++samples;
    }
  }

  out.rejected = filter.rejectedCount();
  out.accepted = static_cast<std::uint32_t>(kArcS / g_fix_period_s) - out.rejected;
  out.mean_nees = (samples > 0) ? nees_sum / samples : 0.0;
  out.rms_pos_err_m = (samples > 0) ? std::sqrt(err_sq_sum / samples) : 0.0;
  return out;
}

}  // namespace

/// **White noise flatters the filter, and this measures by how much.**
///
/// Same filter, same seed, same white error; the only difference is whether the
/// fixes also carry a correlated component the receiver does not report. The
/// numbers are recorded rather than merely bounded, because they are the reason
/// the `R` repair below exists.
TEST(OrbitOdCorrelatedGnss, CorrelatedErrorDegradesTheFilterAndItsConsistency) {
  RecordProperty("verifies", "REQ-ODP-014");
  const OrbitOdConfig cfg = baseConfig();  // no R repair

  const ArcResult white_only = flyArc(cfg, /*corr_sigma_m=*/0.0, /*seed=*/12345);
  const ArcResult correlated = flyArc(cfg, kCorrSigmaM, /*seed=*/12345);

  RecordProperty("white_mean_nees", std::to_string(white_only.mean_nees));
  RecordProperty("white_rms_pos_err_m", std::to_string(white_only.rms_pos_err_m));
  RecordProperty("white_rejected", std::to_string(white_only.rejected));
  RecordProperty("corr_mean_nees", std::to_string(correlated.mean_nees));
  RecordProperty("corr_rms_pos_err_m", std::to_string(correlated.rms_pos_err_m));
  RecordProperty("corr_rejected", std::to_string(correlated.rejected));

  // The white case is the baseline every OD number in this project was taken
  // against: consistent, by construction, because the filter's R is the truth.
  EXPECT_LT(white_only.mean_nees, 12.0) << "the white baseline is not consistent — fix that first";

  // The correlated case is worse on both counts, and the covariance is the more
  // important one: a larger error the filter *knows* about is a wider covariance,
  // while a larger error it does not know about is an overconfident filter.
  EXPECT_GT(correlated.rms_pos_err_m, white_only.rms_pos_err_m);
  EXPECT_GT(correlated.mean_nees, 2.0 * white_only.mean_nees)
      << "correlated NEES " << correlated.mean_nees << " against white " << white_only.mean_nees;
}

/// **The `R` repair restores consistency — but only when it is sized for the
/// error's persistence, not for its size.**
///
/// This is the push's substantive finding, and it is the opposite of the obvious
/// implementation. Inflating `R` by the correlated variance itself (`k` = 1) is
/// what the textbook remedy for unmodelled measurement error suggests, and it
/// closes only *half* the gap: NEES 49.6 -> 23.7 against a chi^2_6 mean of 6.
/// The reason is structural. A Kalman filter assumes measurement noise is white,
/// so over the `tau/dt` fixes that share one realisation of the correlated error
/// it drives its covariance down as if it were averaging independent samples,
/// while the error itself does not average away at all. Per-update inflation
/// cannot reproduce time correlation; it can only be made large enough that the
/// over-averaging lands somewhere honest.
///
/// Measured, at the 10 s cadence and 300 s correlation time flown here
/// (`k` in units of the correlated per-axis sigma):
///
///     k     0      1      2      3      5     10
///  NEES  49.6   23.7   10.8   6.46   3.66   2.41
///   RMS  1.815  1.781  1.756  1.763  1.829  2.138  [m]
///
/// `k` = 3 lands on the chi^2_6 mean at no cost in accuracy — the estimate is as
/// good as the best of the row — and past it the filter turns conservative and
/// the estimate degrades as the measurement is progressively thrown away.
///
/// **`k` is a tuning, not a formula, and this is where it is written down.**
/// Sweeping the fix cadence at fixed tau gives k ~ 0.58*sqrt(tau/dt) (k = 4.4,
/// 3.1, 1.9, 1.3 at dt = 5, 10, 30, 60 s), which looks like a law until tau is
/// swept at fixed dt: there k *saturates* near 3.2 for tau >= 300 s rather than
/// continuing to grow (k = 2.2, 3.1, 3.2, 3.2 at tau = 120, 300, 600, 1200 s).
/// The filter's own process noise reopens its covariance on a timescale set by
/// `q_a`, and once tau exceeds that, the correlated error is no longer the
/// binding constraint. A formula fitted to one of those regimes would be wrong
/// in the other, so the config carries an explicit sigma and the tuning is
/// verified by NEES per configuration. What removes the need for the tuning
/// altogether is the raw pseudorange path (§8.3, still owed), where the
/// common-mode terms have their own signature across the satellites in view.
TEST(OrbitOdCorrelatedGnss, InflatingRRestoresConsistencyAgainstACorrelatedReceiver) {
  RecordProperty("verifies", "REQ-ODP-014");

  const ArcResult naive = flyArc(baseConfig(), kCorrSigmaM, /*seed=*/2024);

  // 3x the correlated sigma — the tuned value, for the reason in the comment
  // above. 1x is the obvious choice and is measured here as insufficient.
  OrbitOdConfig naive_inflation = baseConfig();
  naive_inflation.gnss_corr_sigma_h_m = kCorrSigmaM;
  naive_inflation.gnss_corr_sigma_v_m = kCorrSigmaM;
  const ArcResult one_sigma = flyArc(naive_inflation, kCorrSigmaM, /*seed=*/2024);
  RecordProperty("one_sigma_mean_nees", std::to_string(one_sigma.mean_nees));
  EXPECT_GT(one_sigma.mean_nees, 12.0)
      << "inflating by 1x the correlated sigma is enough after all (" << one_sigma.mean_nees
      << ") — re-read this test's comment, the tuning table has moved";

  OrbitOdConfig repaired = baseConfig();
  repaired.gnss_corr_sigma_h_m = 3.0 * kCorrSigmaM;
  repaired.gnss_corr_sigma_v_m = 3.0 * kCorrSigmaM;
  ASSERT_TRUE(repaired.isValid());
  const ArcResult fixed = flyArc(repaired, kCorrSigmaM, /*seed=*/2024);

  RecordProperty("naive_mean_nees", std::to_string(naive.mean_nees));
  RecordProperty("repaired_mean_nees", std::to_string(fixed.mean_nees));
  RecordProperty("naive_rejected", std::to_string(naive.rejected));
  RecordProperty("repaired_rejected", std::to_string(fixed.rejected));
  RecordProperty("naive_rms_pos_err_m", std::to_string(naive.rms_pos_err_m));
  RecordProperty("repaired_rms_pos_err_m", std::to_string(fixed.rms_pos_err_m));

  EXPECT_LT(fixed.mean_nees, naive.mean_nees)
      << "R inflation did not improve consistency: " << fixed.mean_nees << " vs "
      << naive.mean_nees;
  // chi^2_6 has mean 6; a consistent filter sits near it. Both sides are bounded
  // — too small means overconfident, too large means the measurement is being
  // thrown away — so this cannot be satisfied by inflating without limit.
  EXPECT_LT(fixed.mean_nees, 10.0) << "repaired NEES " << fixed.mean_nees << " is still optimistic";
  EXPECT_GT(fixed.mean_nees, 3.0) << "repaired NEES " << fixed.mean_nees
                                  << " is conservative — the fixes are being discarded";
  // And consistency is not bought with accuracy: the estimate is no worse than
  // the uninflated filter's.
  EXPECT_LT(fixed.rms_pos_err_m, 1.05 * naive.rms_pos_err_m);
}

/// Zero inflation is exactly the pre-Push-77 filter.
TEST(OrbitOdCorrelatedGnss, ZeroInflationLeavesTheFilterBitForBit) {
  RecordProperty("verifies", "REQ-ODP-014");
  OrbitOdConfig explicit_zero = baseConfig();
  explicit_zero.gnss_corr_sigma_h_m = 0.0;
  explicit_zero.gnss_corr_sigma_v_m = 0.0;

  const ArcResult a = flyArc(baseConfig(), kCorrSigmaM, /*seed=*/7);
  const ArcResult b = flyArc(explicit_zero, kCorrSigmaM, /*seed=*/7);
  EXPECT_EQ(a.mean_nees, b.mean_nees);
  EXPECT_EQ(a.rms_pos_err_m, b.rms_pos_err_m);
  EXPECT_EQ(a.rejected, b.rejected);
}

/// A negative inflation is refused: that is a covariance being shrunk by config.
TEST(OrbitOdCorrelatedGnss, NegativeInflationIsRefused) {
  RecordProperty("verifies", "REQ-ODP-014");
  OrbitOdConfig cfg = baseConfig();
  cfg.gnss_corr_sigma_h_m = -1.0;
  EXPECT_FALSE(cfg.isValid());
  cfg = baseConfig();
  cfg.gnss_corr_sigma_v_m = -1.0;
  EXPECT_FALSE(cfg.isValid());
}
