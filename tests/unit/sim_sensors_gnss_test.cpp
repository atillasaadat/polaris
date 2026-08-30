/// @file Unit tests for the GNSS receiver truth model (§6.2, §8.3).
///
/// The properties that matter and are easiest to get quietly wrong: the datasheet
/// 2D-RMS → per-axis σ conversion and the horizontal/vertical split in the local
/// geodetic frame, the GPS time tag, sample-period gating, the cold-start and
/// post-outage reacquisition delays, and the fault hooks (outage, spoof that stays
/// *valid*, clock jump). Statistics are checked by Monte Carlo against the spec.
///
/// Fixtures mirror config/hardware/gnss/*.yaml (design doc §19.4 — hardcoded specs
/// are permitted in tests, and only there).

#include <gtest/gtest.h>

#include <cmath>
#include <Eigen/Core>
#include <vector>

#include "constants/constants.hpp"
#include "math/frames.hpp"
#include "math/typed_vector.hpp"
#include "sensors/gnss.hpp"
#include "sensors/gnss_jamming.hpp"
#include "time/timescales.hpp"

namespace {

namespace sensors = polaris::sim::sensors;
namespace pm = polaris::math;
namespace pmf = polaris::math::frames;
namespace pt = polaris::time;

using Vec3E = pm::Vec3<pmf::ECEF>;

constexpr double kRe = polaris::constants::wgs84::kSemiMajorAxis;
constexpr std::uint64_t kSeed = 20260101;
constexpr std::uint64_t kStream = 0x9E3779B97F4A7C15ULL;
const pt::Tai kEpoch = pt::Tai::fromNanosecondsSinceEpoch(1767225637000000000LL);

pt::Tai epochPlus(double seconds) {
  return pt::Tai::fromNanosecondsSinceEpoch(kEpoch.nanosecondsSinceEpoch() +
                                            static_cast<std::int64_t>(seconds * 1.0e9));
}

/// OEM7600-like fixture, mirroring novatel_oem7600.yaml but with the acquisition
/// delays overridable, so a statistics test does not have to wait out a cold start.
sensors::GnssSpec oem7600Spec(double cold_start_s = 34.0, double reacquisition_s = 0.5) {
  return sensors::GnssSpec::fromParams({
      {"horizontal_position_rms_m", 1.2},
      {"velocity_accuracy_m_s_rms", 0.03},
      {"time_accuracy_ns_rms", 5.0},
      {"max_rate_hz", 100.0},
      {"cold_start_s", cold_start_s},
      {"hot_start_s", 20.0},
      {"reacquisition_s", reacquisition_s},
  });
}

/// A satellite at 500 km altitude on the ECEF +X axis. There the local geodetic
/// basis is exactly up=+X, east=+Y, north=+Z, so the vertical error lands on X and
/// the two horizontal errors on Y and Z — which lets the split be measured axis by
/// axis.
Vec3E onXAxis() {
  return Vec3E(kRe + 500.0e3, 0.0, 0.0);
}

sensors::GnssInput inputOnXAxis() {
  sensors::GnssInput in;
  in.position_m = onXAxis();
  in.velocity_m_s = Vec3E(0.0, 7612.0, 0.0);  // ~circular LEO speed, along +Y
  return in;
}

constexpr double kDeg2Rad = 0.017453292519943295;

/// Forward WGS84: place a satellite over a given sub-satellite point.
sensors::GnssInput inputOverGeodetic(double lat_deg, double lon_deg) {
  const double a = kRe;
  const double e2 = polaris::constants::wgs84::kEccentricitySq;
  const double lat = lat_deg * kDeg2Rad;
  const double lon = lon_deg * kDeg2Rad;
  const double alt = 500.0e3;
  const double n = a / std::sqrt(1.0 - e2 * std::sin(lat) * std::sin(lat));
  sensors::GnssInput in;
  in.position_m =
      Vec3E((n + alt) * std::cos(lat) * std::cos(lon), (n + alt) * std::cos(lat) * std::sin(lon),
            (n * (1.0 - e2) + alt) * std::sin(lat));
  in.velocity_m_s = Vec3E(0.0, 0.0, 0.0);
  return in;
}

/// A one-region jamming map: a box over lon∈[32,37], lat∈[44,46.5] (Crimea-ish).
sensors::JammingRegions crimeaBox() {
  sensors::JammingRegions regions;
  const std::string kml =
      "<kml><Document><Placemark><name>Crimea</name><Polygon><outerBoundaryIs><LinearRing>"
      "<coordinates>32,44 37,44 37,46.5 32,46.5 32,44</coordinates>"
      "</LinearRing></outerBoundaryIs></Polygon></Placemark></Document></kml>";
  sensors::JammingRegions::fromKml(kml, regions, nullptr);
  return regions;
}

}  // namespace

TEST(GnssSpec, ConvertsDatasheetFiguresToSi) {
  const auto s = oem7600Spec();
  // 2D-RMS horizontal → per-axis σ = RMS/√2.
  EXPECT_NEAR(s.position_sigma_h_m, 1.2 / std::sqrt(2.0), 1e-12);
  // Vertical omitted → 1.5× the horizontal per-axis σ.
  EXPECT_NEAR(s.position_sigma_v_m, 1.5 * 1.2 / std::sqrt(2.0), 1e-12);
  EXPECT_NEAR(s.velocity_sigma_m_s, 0.03, 1e-12);
  EXPECT_NEAR(s.time_sigma_s, 5.0e-9, 1e-21);
  EXPECT_NEAR(s.sample_period_s, 0.01, 1e-12);  // 1/100 Hz
}

TEST(GnssSpec, ExplicitVerticalOverridesTheDefault) {
  const auto s = sensors::GnssSpec::fromParams({
      {"horizontal_position_rms_m", 2.0},
      {"vertical_position_rms_m", 5.0},
  });
  EXPECT_NEAR(s.position_sigma_h_m, 2.0 / std::sqrt(2.0), 1e-12);
  EXPECT_NEAR(s.position_sigma_v_m, 5.0, 1e-12);  // taken as-is, not derived
}

TEST(Gnss, TimeTagIsGpsTime) {
  // Zero clock error so the tag is exactly TAI−19s with no bias.
  auto s = oem7600Spec(0.0, 0.0);
  s.time_sigma_s = 0.0;
  sensors::Gnss g(s, kSeed, kStream);
  const auto m = g.sample(kEpoch, inputOnXAxis());
  ASSERT_TRUE(m.valid);
  EXPECT_EQ(m.time_tag.nanosecondsSinceEpoch(), pt::toGps(kEpoch).nanosecondsSinceEpoch());
  EXPECT_EQ(m.clock_bias_s, 0.0);
}

TEST(Gnss, ErrorStatisticsMatchTheSpecPerAxis) {
  // Cold start suppressed so every sample is a valid fix. On the +X axis the
  // vertical σ falls on X and the horizontal σ on Y and Z.
  const auto s = oem7600Spec(0.0, 0.0);
  sensors::Gnss g(s, kSeed, kStream);
  const auto in = inputOnXAxis();
  const Eigen::Vector3d truth = in.position_m.eigen();

  const int n = 20000;
  Eigen::Vector3d sum = Eigen::Vector3d::Zero();
  Eigen::Vector3d sumsq = Eigen::Vector3d::Zero();
  double vel_sumsq = 0.0;
  for (int i = 0; i < n; ++i) {
    // Step past the 10 ms sample period so each draw is a fresh fix.
    const auto m = g.sample(epochPlus(0.02 * i), in);
    ASSERT_TRUE(m.fresh);
    const Eigen::Vector3d err = m.position_m.eigen() - truth;
    sum += err;
    sumsq += err.cwiseProduct(err);
    vel_sumsq += (m.velocity_m_s.eigen() - in.velocity_m_s.eigen()).squaredNorm();
  }
  const Eigen::Vector3d mean = sum / n;
  const Eigen::Vector3d rms = (sumsq / n).cwiseSqrt();

  // Zero-mean (white), and each axis matches its own σ within sampling error.
  EXPECT_NEAR(mean.x(), 0.0, 0.05);
  EXPECT_NEAR(rms.x(), s.position_sigma_v_m, 0.05);  // X = vertical
  EXPECT_NEAR(rms.y(), s.position_sigma_h_m, 0.05);  // Y = horizontal
  EXPECT_NEAR(rms.z(), s.position_sigma_h_m, 0.05);  // Z = horizontal

  // 2D horizontal RMS reconstructs the datasheet's 1.2 m headline.
  EXPECT_NEAR(std::sqrt(rms.y() * rms.y() + rms.z() * rms.z()), 1.2, 0.05);
  // Velocity: 3 axes each at 0.03 → 3D RMS = 0.03·√3.
  EXPECT_NEAR(std::sqrt(vel_sumsq / n), 0.03 * std::sqrt(3.0), 0.002);
}

TEST(Gnss, IsBitReproducibleFromSeedAndStream) {
  const auto s = oem7600Spec(0.0, 0.0);
  sensors::Gnss a(s, kSeed, kStream);
  sensors::Gnss b(s, kSeed, kStream);
  const auto in = inputOnXAxis();
  for (int i = 0; i < 10; ++i) {
    const auto ma = a.sample(epochPlus(0.02 * i), in);
    const auto mb = b.sample(epochPlus(0.02 * i), in);
    EXPECT_EQ(ma.position_m.eigen(), mb.position_m.eigen());
    EXPECT_EQ(ma.velocity_m_s.eigen(), mb.velocity_m_s.eigen());
    EXPECT_EQ(ma.clock_bias_s, mb.clock_bias_s);
  }
}

TEST(Gnss, RepeatsTheLastFixWithinTheSamplePeriod) {
  const auto s = oem7600Spec(0.0, 0.0);
  sensors::Gnss g(s, kSeed, kStream);
  const auto in = inputOnXAxis();

  const auto first = g.sample(kEpoch, in);
  ASSERT_TRUE(first.fresh);
  // 5 ms later — inside the 10 ms period — the receiver has no new solution.
  const auto stale = g.sample(epochPlus(0.005), in);
  EXPECT_FALSE(stale.fresh);
  EXPECT_EQ(stale.position_m.eigen(), first.position_m.eigen());
  // Past the period, a genuinely new fix.
  const auto fresh = g.sample(epochPlus(0.02), in);
  EXPECT_TRUE(fresh.fresh);
  EXPECT_NE(fresh.position_m.eigen(), first.position_m.eigen());
}

TEST(Gnss, ColdStartWithholdsFixesUntilFirstFixTime) {
  const auto s = oem7600Spec(34.0, 0.5);
  sensors::Gnss g(s, kSeed, kStream);
  const auto in = inputOnXAxis();

  EXPECT_FALSE(g.sample(kEpoch, in).valid);           // acquiring
  EXPECT_FALSE(g.sample(epochPlus(33.0), in).valid);  // still acquiring
  EXPECT_TRUE(g.sample(epochPlus(34.0), in).valid);   // fix acquired
}

TEST(Gnss, OutageInvalidatesAndReacquisitionDelaysRecovery) {
  const auto s = oem7600Spec(0.0, 0.5);
  sensors::Gnss g(s, kSeed, kStream);
  const auto in = inputOnXAxis();

  EXPECT_TRUE(g.sample(kEpoch, in).valid);

  g.setOutage(true);
  EXPECT_FALSE(g.sample(epochPlus(1.0), in).valid);

  g.setOutage(false);
  // Recovery is not instant: the reacquisition delay must elapse first.
  EXPECT_FALSE(g.sample(epochPlus(2.0), in).valid);
  EXPECT_FALSE(g.sample(epochPlus(2.4), in).valid);
  EXPECT_TRUE(g.sample(epochPlus(2.5), in).valid);
}

TEST(Gnss, SpoofOffsetsThePositionButStaysValid) {
  // No noise, so the offset is exactly recoverable.
  auto s = oem7600Spec(0.0, 0.0);
  s.position_sigma_h_m = 0.0;
  s.position_sigma_v_m = 0.0;
  sensors::Gnss g(s, kSeed, kStream);
  const auto in = inputOnXAxis();

  g.injectPositionOffset(Vec3E(1000.0, 0.0, 0.0));
  const auto m = g.sample(kEpoch, in);
  // The point of a spoof: it looks like a real fix. FDIR must catch it on
  // innovation, not on a flag.
  EXPECT_TRUE(m.valid);
  EXPECT_NEAR(m.position_m.eigen().x(), in.position_m.eigen().x() + 1000.0, 1e-6);
}

TEST(Gnss, NoiseDisabledReportsTruthExactFixes) {
  // A bring-up run flies a perfect GPS: the datasheet σ are still carried, but no
  // error is drawn, so the fix equals truth and the clock bias is zero.
  auto s = oem7600Spec(0.0, 0.0);
  s.noise_enabled = false;
  sensors::Gnss g(s, kSeed, kStream);
  const auto in = inputOnXAxis();
  const auto m = g.sample(kEpoch, in);
  EXPECT_TRUE(m.valid);
  EXPECT_EQ(m.position_m.eigen(), in.position_m.eigen());
  EXPECT_EQ(m.velocity_m_s.eigen(), in.velocity_m_s.eigen());
  EXPECT_EQ(m.clock_bias_s, 0.0);
  // The σ are still reported — an estimator wants them even on a clean run.
  EXPECT_GT(m.position_sigma_h_m, 0.0);
}

TEST(Gnss, GeographicJammingInvalidatesOverARegion) {
  const auto s = oem7600Spec(0.0, 0.0);
  const auto regions = crimeaBox();
  sensors::Gnss g(s, kSeed, kStream);
  g.setJammingRegions(&regions);

  // Over the Atlantic — not jammed, a normal valid fix.
  const auto clear = g.sample(kEpoch, inputOverGeodetic(0.0, -30.0));
  EXPECT_TRUE(clear.valid);
  EXPECT_FALSE(clear.jammed);
  EXPECT_TRUE(clear.jamming_region.empty());

  // Over the region — jammed, invalid, and telemetering which zone.
  const auto jammed = g.sample(epochPlus(1.0), inputOverGeodetic(45.0, 34.5));
  EXPECT_FALSE(jammed.valid);
  EXPECT_TRUE(jammed.jammed);
  EXPECT_EQ(jammed.jamming_region, "Crimea");
}

TEST(Gnss, ReacquisitionDelayAppliesAfterLeavingAJammedZone) {
  const auto s = oem7600Spec(0.0, 0.5);
  const auto regions = crimeaBox();
  sensors::Gnss g(s, kSeed, kStream);
  g.setJammingRegions(&regions);

  EXPECT_FALSE(g.sample(kEpoch, inputOverGeodetic(45.0, 34.5)).valid);  // jammed
  // Just cleared the zone: recovery is not instant — the reacquisition delay runs.
  EXPECT_FALSE(g.sample(epochPlus(1.0), inputOverGeodetic(0.0, -30.0)).valid);
  EXPECT_FALSE(g.sample(epochPlus(1.4), inputOverGeodetic(0.0, -30.0)).valid);
  EXPECT_TRUE(g.sample(epochPlus(1.5), inputOverGeodetic(0.0, -30.0)).valid);
}

TEST(Gnss, ClockJumpShiftsTheTimeTag) {
  auto s = oem7600Spec(0.0, 0.0);
  s.time_sigma_s = 0.0;
  sensors::Gnss g(s, kSeed, kStream);
  const auto in = inputOnXAxis();

  g.injectClockJump(1.0e-6);  // 1 µs
  const auto m = g.sample(kEpoch, in);
  EXPECT_NEAR(m.clock_bias_s, 1.0e-6, 1e-15);
  EXPECT_EQ(m.time_tag.nanosecondsSinceEpoch(), pt::toGps(kEpoch).nanosecondsSinceEpoch() + 1000);
}

// ---------------------------------------------------------------------------
// Fix latency (§6.2, §8.3)
// ---------------------------------------------------------------------------

namespace {

/// The OEM7600 fixture with a delay line. 10 Hz rather than 100 so the arithmetic
/// below is legible: at a 0.05 s latency, exactly one 0.1 s fix is in flight.
sensors::GnssSpec latencySpec(double fix_latency_s, double rate_hz = 10.0) {
  return sensors::GnssSpec::fromParams({
      {"horizontal_position_rms_m", 1.2},
      {"velocity_accuracy_m_s_rms", 0.03},
      {"time_accuracy_ns_rms", 5.0},
      {"max_rate_hz", rate_hz},
      {"fix_latency_s", fix_latency_s},
      {"cold_start_s", 0.0},
      {"reacquisition_s", 0.0},
  });
}

}  // namespace

TEST(GnssSpec, CarriesFixLatencyAndDefaultsItToZero) {
  EXPECT_DOUBLE_EQ(latencySpec(0.05).fix_latency_s, 0.05);
  // Absent from the params, the term is disabled — the same zero-default
  // convention every other key in `fromParams` follows.
  EXPECT_DOUBLE_EQ(oem7600Spec().fix_latency_s, 0.0);
}

/// The delivered fix is tagged at the epoch it was *measured*, not the epoch it
/// arrives — which is the whole point. A receiver stamps when it measured; a
/// model that stamped on delivery would hide the latency from the filter and
/// make it uncorrectable rather than merely present.
TEST(Gnss, LatentFixIsTaggedAtItsMeasurementEpochNotItsDelivery) {
  constexpr double kLatency = 0.05;
  sensors::GnssSpec spec = latencySpec(kLatency);
  spec.noise_enabled = false;  // isolate the timing from the error stack
  sensors::Gnss rx(spec, kSeed, kStream);

  const sensors::GnssInput in = inputOnXAxis();

  // t = 0: the first solution enters the delay line and nothing has cleared the
  // receiver yet, so there is no fix to report.
  const sensors::GnssMeasurement first = rx.sample(kEpoch, in);
  EXPECT_FALSE(first.valid) << "a receiver that has not finished a solution has nothing to report";

  // t = 0.1 s: the t = 0 solution has been in the receiver 0.1 s > 0.05 s, so it
  // is delivered — tagged at t = 0.
  const sensors::GnssMeasurement second = rx.sample(epochPlus(0.1), in);
  EXPECT_TRUE(second.valid);
  const std::int64_t delivered_ns = second.time_tag.nanosecondsSinceEpoch();
  const std::int64_t measured_ns = polaris::time::toGps(kEpoch).nanosecondsSinceEpoch();
  EXPECT_NEAR(static_cast<double>(delivered_ns - measured_ns) / 1.0e9, 0.0, 1.0e-6)
      << "the tag moved with delivery instead of staying at the measurement epoch";

  EXPECT_EQ(rx.pendingDropped(), 0u);
}

/// A latent fix describes where the vehicle *was*. Asserted against a moving
/// truth, because with a stationary one every epoch looks alike and the test
/// would pass on a model that ignored latency entirely.
TEST(Gnss, LatentFixReportsTheEarlierPositionNotTheCurrentOne) {
  constexpr double kLatency = 0.05;
  constexpr double kSpeed = 7612.0;
  sensors::GnssSpec spec = latencySpec(kLatency);
  spec.noise_enabled = false;
  sensors::Gnss rx(spec, kSeed, kStream);

  const auto truthAt = [&](double t_s) {
    sensors::GnssInput in;
    in.position_m = Vec3E(kRe + 500.0e3, kSpeed * t_s, 0.0);
    in.velocity_m_s = Vec3E(0.0, kSpeed, 0.0);
    return in;
  };

  ASSERT_FALSE(rx.sample(kEpoch, truthAt(0.0)).valid);
  const sensors::GnssMeasurement m = rx.sample(epochPlus(0.1), truthAt(0.1));
  ASSERT_TRUE(m.valid);

  // Delivered at t = 0.1 s but measured at t = 0, so it must report y = 0, not
  // the 761.2 m the vehicle has since travelled.
  EXPECT_NEAR(m.position_m.eigen().y(), 0.0, 1.0e-6);
  EXPECT_NEAR((m.position_m.eigen() - truthAt(0.1).position_m.eigen()).norm(), kSpeed * 0.1, 1.0e-3)
      << "the delivered fix is not one latency behind the current truth";
}

/// **Latency jitter moves delivery, never the tag.** With a jittered delay line
/// polled at the fix rate, some fixes come due a poll early and some a poll
/// late — but every delivered fix is still tagged at its own measurement epoch,
/// which is what keeps the onboard correction exact per fix (Push 70; the
/// paper it answers measured 15 ± 7.5 ms bus delays).
TEST(Gnss, LatencyJitterChangesWhenAFixIsDueButNotItsTag) {
  sensors::GnssSpec spec = latencySpec(0.05, 20.0);
  spec.fix_latency_jitter_s = 0.02;
  spec.noise_enabled = false;
  sensors::Gnss rx(spec, kSeed, kStream);
  const sensors::GnssInput in = inputOnXAxis();

  // Poll at 20 Hz for 5 s: with a 50 ± 20 ms latency a 50 ms poll sees each
  // fix either at the next poll or the one after; count how many arrive with
  // each age, and check every tag is a whole number of fix periods behind.
  int age_one = 0;
  int age_two_plus = 0;
  std::int64_t last_tag = 0;
  for (int k = 0; k <= 100; ++k) {
    const double t = 0.05 * k;
    const sensors::GnssMeasurement m = rx.sample(epochPlus(t), in);
    if (!m.valid || m.time_tag.nanosecondsSinceEpoch() == last_tag) {
      continue;
    }
    last_tag = m.time_tag.nanosecondsSinceEpoch();
    const double age_s =
        static_cast<double>(polaris::time::toGps(epochPlus(t)).nanosecondsSinceEpoch() - last_tag) /
        1.0e9;
    // Tags land on the 50 ms sample grid, never on a delivery instant.
    EXPECT_NEAR(std::fmod(age_s + 1.0e-9, 0.05), 0.0, 1.0e-6);
    if (age_s < 0.075) {
      ++age_one;
    } else {
      ++age_two_plus;
    }
  }
  EXPECT_GT(age_one, 10) << "no fix ever came due within one poll";
  EXPECT_GT(age_two_plus, 5) << "no fix was ever late: the jitter did nothing";
  EXPECT_EQ(rx.pendingDropped(), 0u);
}

/// With the term off, the model is bit-identical to the pre-latency one. The
/// delay line must be a feature that switches on, not a behaviour change every
/// existing scenario silently inherits.
TEST(Gnss, ZeroLatencyDeliversTheCurrentSolutionImmediately) {
  sensors::Gnss with_latency(latencySpec(0.0), kSeed, kStream);
  sensors::Gnss without(oem7600Spec(/*cold_start_s=*/0.0), kSeed, kStream);

  const sensors::GnssInput in = inputOnXAxis();
  for (int i = 0; i < 5; ++i) {
    const sensors::GnssMeasurement a = with_latency.sample(epochPlus(0.2 * i), in);
    const sensors::GnssMeasurement b = without.sample(epochPlus(0.2 * i), in);
    ASSERT_TRUE(a.valid) << "i = " << i;
    EXPECT_EQ(a.position_m.eigen(), b.position_m.eigen()) << "i = " << i;
    EXPECT_EQ(a.time_tag.nanosecondsSinceEpoch(), b.time_tag.nanosecondsSinceEpoch())
        << "i = " << i;
  }
}

/// The delay line must survive a long run at the configured rate without
/// overrunning, and must say so if it ever does. A silently-dropping buffer
/// would look like an intermittent receiver.
TEST(Gnss, DelayLineDoesNotOverrunAtTheConfiguredRateAndLatency) {
  // The datasheet corner: 100 Hz fixes against the 0.05 s configured latency,
  // i.e. five solutions in flight at any moment against a 64-deep line.
  sensors::GnssSpec spec = latencySpec(0.05, /*rate_hz=*/100.0);
  spec.noise_enabled = false;
  sensors::Gnss rx(spec, kSeed, kStream);

  const sensors::GnssInput in = inputOnXAxis();
  int valid_fixes = 0;
  std::int64_t previous_tag_ns = 0;
  for (int i = 0; i < 2000; ++i) {  // 20 s at 100 Hz
    const sensors::GnssMeasurement m = rx.sample(epochPlus(0.01 * i), in);
    if (!m.valid) {
      continue;
    }
    valid_fixes += 1;
    const std::int64_t tag_ns = m.time_tag.nanosecondsSinceEpoch();
    if (previous_tag_ns != 0 && m.fresh) {
      // The property the onboard filter actually depends on: tags never go
      // backwards. `OrbitOd::ingest` refuses a non-increasing fix epoch, so a
      // delay line that ever handed over an out-of-order solution would show up
      // in flight as rejected measurements, not as a sim bug.
      EXPECT_GE(tag_ns, previous_tag_ns) << "i = " << i;
    }
    previous_tag_ns = tag_ns;
  }
  EXPECT_EQ(rx.pendingDropped(), 0u) << "the delay line overran at its own datasheet rate";

  // Not 2000: five solutions are in flight at any moment, and a poll that finds
  // two due at once takes the newer and supersedes the older (see
  // GnssSpec::fix_latency_s). With the poll epochs landing on nanosecond-rounded
  // 0.01 s boundaries, that coalescing happens a couple of percent of the time.
  // The bound is on the *delivery rate*, which is what a scenario cares about;
  // pinning the exact count would be pinning double-rounding.
  EXPECT_GT(valid_fixes, 1900) << "the receiver is delivering far fewer fixes than it solves";
}

// ===========================================================================
// Correlated (common-mode) position error — Push 77
// ===========================================================================

namespace {

/// The OEM7600 fixture plus a correlated component: 1.5 m horizontal 2D-RMS
/// decorrelating over 300 s, which is the scale and timescale residual
/// ionosphere / broadcast-ephemeris / satellite-clock error actually has on an
/// unaided single-point receiver.
sensors::GnssSpec correlatedSpec(double total_h_rms_m = 1.2, double tau_s = 300.0,
                                 double fraction = 0.5) {
  return sensors::GnssSpec::fromParams({
      {"horizontal_position_rms_m", total_h_rms_m},
      {"velocity_accuracy_m_s_rms", 0.03},
      {"time_accuracy_ns_rms", 5.0},
      {"max_rate_hz", 100.0},
      {"cold_start_s", 0.0},
      {"hot_start_s", 20.0},
      {"reacquisition_s", 0.0},
      {"correlated_position_fraction", fraction},
      {"correlated_position_tau_s", tau_s},
  });
}

/// Collect the per-axis position error of @p count fixes at @p cadence_s, on the
/// +X-axis fixture where east/north/up land on Y/Z/X.
std::vector<Eigen::Vector3d> collectErrors(sensors::Gnss& gnss, int count, double cadence_s) {
  std::vector<Eigen::Vector3d> out;
  out.reserve(static_cast<std::size_t>(count));
  const Eigen::Vector3d truth = onXAxis().eigen();
  for (int i = 0; i < count; ++i) {
    const sensors::GnssMeasurement m = gnss.sample(epochPlus(i * cadence_s), inputOnXAxis());
    if (m.valid && m.fresh) {
      out.push_back(m.position_m.eigen() - truth);
    }
  }
  return out;
}

}  // namespace

/// The datasheet keys reach the spec, with the same conversions the white term
/// uses — and an entry that omits them stays at zero rather than acquiring a
/// correlated error by default.
TEST(GnssSpec, SplitsTheDatasheetTotalAndDefaultsTheCorrelatedPartOff) {
  const sensors::GnssSpec plain = oem7600Spec();
  EXPECT_EQ(plain.position_corr_sigma_h_m, 0.0);
  EXPECT_EQ(plain.position_corr_sigma_v_m, 0.0);
  EXPECT_EQ(plain.position_corr_tau_s, 0.0);
  // With the term off the reported sigma is the white one, exactly as before.
  EXPECT_EQ(plain.position_reported_sigma_h_m, plain.position_sigma_h_m);
  EXPECT_EQ(plain.position_reported_sigma_v_m, plain.position_sigma_v_m);

  // A half-correlated receiver: same datasheet total, split evenly in variance.
  const sensors::GnssSpec corr = correlatedSpec(1.2, 300.0, 0.5);
  const double total_h = 1.2 / std::sqrt(2.0);
  EXPECT_NEAR(corr.position_sigma_h_m, total_h * std::sqrt(0.5), 1e-12);
  EXPECT_NEAR(corr.position_corr_sigma_h_m, total_h * std::sqrt(0.5), 1e-12);
  // **The total is preserved** — the split changes the spectrum, not the size.
  EXPECT_NEAR(corr.position_reported_sigma_h_m, total_h, 1e-12);
  EXPECT_NEAR(corr.position_reported_sigma_v_m, total_h * 1.5, 1e-12);
  EXPECT_EQ(corr.position_corr_tau_s, 300.0);

  // A fully correlated receiver has no white part left.
  const sensors::GnssSpec all = correlatedSpec(1.2, 300.0, 1.0);
  EXPECT_NEAR(all.position_sigma_h_m, 0.0, 1e-12);
  EXPECT_NEAR(all.position_corr_sigma_h_m, total_h, 1e-12);
  EXPECT_NEAR(all.position_reported_sigma_h_m, total_h, 1e-12);
}

/// Variances add in quadrature, and the process is stationary from the first fix.
///
/// The second half matters as much as the first: seeding from the stationary
/// distribution rather than from zero is what keeps a short scenario from
/// sampling a warm-up transient that a long one has forgotten.
TEST(Gnss, CorrelatedErrorAddsInQuadratureAndIsStationaryFromTheFirstFix) {
  // Averaged over independent seeds, not over one long record. A tau = 300 s
  // process sampled at 1 Hz carries only ~N*dt/tau independent samples, so a
  // single 20000-fix run estimates its variance to ~9 % — measured 16 % high on
  // the vertical axis, which is the estimator's sampling error and not the
  // model's. Eight realisations buy the independence the arc cannot.
  constexpr int kSeeds = 8;
  constexpr int kFixes = 20000;
  // The split preserves the datasheet total, so the *measured* per-axis error is
  // the datasheet sigma however the fraction is set — which is the property
  // being checked, and the reason this push does not quietly make the receiver
  // worse than the part it models.
  const double expect_h = 1.2 / std::sqrt(2.0);
  const double expect_v = 1.5 * expect_h;

  double sum_x = 0.0;
  double sum_y = 0.0;
  double sum_z = 0.0;
  double head_y = 0.0;
  std::size_t n = 0;
  for (int s = 0; s < kSeeds; ++s) {
    sensors::Gnss gnss(correlatedSpec(), kSeed + static_cast<std::uint64_t>(s), kStream);
    const std::vector<Eigen::Vector3d> e = collectErrors(gnss, kFixes, 1.0);
    ASSERT_GT(e.size(), 19000u);
    // east=Y, north=Z, up=X on this fixture.
    for (const Eigen::Vector3d& v : e) {
      sum_x += v.x() * v.x();
      sum_y += v.y() * v.y();
      sum_z += v.z() * v.z();
    }
    n += e.size();
    // Stationary from the start: the opening fixes of each realisation are not
    // systematically quiet, which they would be if the state began at zero.
    for (int i = 0; i < 100; ++i) {
      head_y += e[static_cast<std::size_t>(i)].y() * e[static_cast<std::size_t>(i)].y();
    }
  }
  const double dn = static_cast<double>(n);
  EXPECT_NEAR(std::sqrt(sum_y / dn), expect_h, 0.08 * expect_h) << "east";
  EXPECT_NEAR(std::sqrt(sum_z / dn), expect_h, 0.08 * expect_h) << "north";
  EXPECT_NEAR(std::sqrt(sum_x / dn), expect_v, 0.08 * expect_v) << "up";

  const double head = std::sqrt(head_y / (100.0 * kSeeds));
  EXPECT_GT(head, 0.6 * expect_h) << "opening fixes are quiet (" << head
                                  << ") — the state is warming up from zero";
}

/// The error decorrelates on its configured timescale, not per fix.
///
/// This is the whole reason the term exists: a filter averaging successive fixes
/// improves on white noise and does not improve on this.
TEST(Gnss, CorrelatedErrorDecorrelatesOnItsConfiguredTimescale) {
  // White noise pushed far below the correlated term so the measured
  // autocorrelation is the correlated process's own, not a diluted one.
  const double tau = 300.0;
  const double cadence = 30.0;
  // Fully correlated (fraction 1), so the measured autocorrelation is the
  // process's own rather than one diluted by a white part.
  sensors::Gnss gnss(correlatedSpec(1.2, tau, /*fraction=*/1.0), kSeed, kStream);
  const std::vector<Eigen::Vector3d> e = collectErrors(gnss, 8000, cadence);
  ASSERT_GT(e.size(), 7000u);

  std::vector<double> y;
  for (const Eigen::Vector3d& v : e) {
    y.push_back(v.y());
  }
  double num = 0.0;
  double den = 0.0;
  for (std::size_t i = 0; i + 1 < y.size(); ++i) {
    num += y[i] * y[i + 1];
    den += y[i] * y[i];
  }
  const double rho = num / den;
  const double expect = std::exp(-cadence / tau);  // 0.9048
  EXPECT_NEAR(rho, expect, 0.03) << "lag-1 autocorrelation " << rho << " against exp(-dt/tau) "
                                 << expect;
}

/// The stationary variance does not depend on how often the fix is polled.
///
/// The exact Gauss-Markov discretisation is what buys this; an Euler step valid
/// only for dt << tau would quietly change the error's size when a scenario
/// changed its fix cadence, which is the kind of coupling that makes two
/// campaigns incomparable for a reason nobody can find.
TEST(Gnss, CorrelatedErrorVarianceIsInvariantToFixCadence) {
  double rms[2] = {0.0, 0.0};
  const double cadences[2] = {1.0, 60.0};
  for (int k = 0; k < 2; ++k) {
    sensors::Gnss gnss(correlatedSpec(1.2, 300.0, /*fraction=*/1.0), kSeed, kStream);
    const std::vector<Eigen::Vector3d> e = collectErrors(gnss, 20000, cadences[k]);
    double s = 0.0;
    for (const Eigen::Vector3d& v : e) {
      s += v.y() * v.y();
    }
    rms[k] = std::sqrt(s / static_cast<double>(e.size()));
  }
  const double sc = 1.2 / std::sqrt(2.0);
  EXPECT_NEAR(rms[0], sc, 0.10 * sc) << "1 s cadence";
  EXPECT_NEAR(rms[1], sc, 0.10 * sc) << "60 s cadence";
}

/// **The receiver reports the right size and says nothing about the colour**,
/// and that asymmetry is the point.
///
/// The reported sigma is the datasheet total, so the filter's `R` is not wrong
/// in magnitude — it is wrong in *spectrum*, because nothing in the fix says how
/// much of that error will still be there on the next one. That is the generous
/// reading of what a receiver knows, and the filter is defeated by it anyway
/// (`tests/unit/orbit_od_correlated_gnss_test.cpp`), which is a stronger result
/// than one obtained by also understating the magnitude.
TEST(Gnss, ReportedSigmaIsTheTotalAndCarriesNoHintOfTheCorrelation) {
  sensors::Gnss gnss(correlatedSpec(1.2, 300.0, 0.5), kSeed, kStream);
  const sensors::GnssMeasurement m = gnss.sample(kEpoch, inputOnXAxis());
  ASSERT_TRUE(m.valid);
  const double total_h = 1.2 / std::sqrt(2.0);
  EXPECT_NEAR(m.position_sigma_h_m, total_h, 1e-12);

  // Identical to what a fully white receiver of the same datasheet reports —
  // the fix cannot be told apart on its sigmas alone.
  sensors::Gnss white(correlatedSpec(1.2, 300.0, /*fraction=*/0.0), kSeed, kStream);
  const sensors::GnssMeasurement w = white.sample(kEpoch, inputOnXAxis());
  // To round-off, not bit-exactly: the reported figure is recombined as
  // hypot(sigma*sqrt(1-f), sigma*sqrt(f)), which returns the original to a few
  // ULP rather than identically.
  EXPECT_DOUBLE_EQ(m.position_sigma_h_m, w.position_sigma_h_m);
  EXPECT_DOUBLE_EQ(m.position_sigma_v_m, w.position_sigma_v_m);
}

/// With the term off, every white draw is bit-identical to the old model.
///
/// The correlated samples are drawn *after* the white ones and skipped entirely
/// when disabled, so enabling the term does not renumber the white stream. That
/// is what lets one seed produce a with/without pair that differs only by the
/// thing under study.
TEST(Gnss, DisabledCorrelatedTermLeavesTheWhiteStreamBitIdentical) {
  sensors::Gnss plain(oem7600Spec(/*cold_start_s=*/0.0, /*reacquisition_s=*/0.0), kSeed, kStream);
  sensors::Gnss off(correlatedSpec(1.2, 300.0, /*fraction=*/0.0), kSeed, kStream);

  for (int i = 0; i < 50; ++i) {
    const sensors::GnssMeasurement a = plain.sample(epochPlus(i * 1.0), inputOnXAxis());
    const sensors::GnssMeasurement b = off.sample(epochPlus(i * 1.0), inputOnXAxis());
    ASSERT_EQ(a.position_m.eigen(), b.position_m.eigen()) << "fix " << i;
    ASSERT_EQ(a.velocity_m_s.eigen(), b.velocity_m_s.eigen()) << "fix " << i;
  }
}
