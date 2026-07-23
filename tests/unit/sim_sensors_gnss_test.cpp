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
