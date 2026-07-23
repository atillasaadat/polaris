/// @file Unit tests for scheduled GNSS fault injection (§9.2/§23.1.1).
///
/// Two layers: resolving a declarative schedule to the fault state active at a
/// time (window membership, unit filtering, overlap), and applying that state to
/// a real receiver so entering and leaving a window both take effect.

#include <gtest/gtest.h>

#include <Eigen/Core>
#include <vector>

#include "constants/constants.hpp"
#include "scenario/gnss_faults.hpp"
#include "scenario/sim_config.hpp"
#include "sensors/gnss.hpp"
#include "time/timescales.hpp"

namespace {

namespace scenario = polaris::sim::scenario;
namespace sensors = polaris::sim::sensors;
namespace pm = polaris::math;
namespace pmf = polaris::math::frames;
namespace pt = polaris::time;

using Vec3E = pm::Vec3<pmf::ECEF>;
using Event = scenario::GnssFaultEvent;

constexpr double kRe = polaris::constants::wgs84::kSemiMajorAxis;
const pt::Tai kEpoch = pt::Tai::fromNanosecondsSinceEpoch(1767225637000000000LL);

Event outage(const std::string& unit, double start, double stop) {
  Event e;
  e.unit = unit;
  e.type = Event::Type::kOutage;
  e.start_s = start;
  e.stop_s = stop;
  return e;
}

sensors::GnssSpec cleanSpec() {
  auto s = sensors::GnssSpec::fromParams({{"horizontal_position_rms_m", 1.2},
                                          {"velocity_accuracy_m_s_rms", 0.03},
                                          {"max_rate_hz", 100.0}});
  return s;  // no cold start / reacquisition, so validity tracks faults directly
}

sensors::GnssInput inputOnXAxis() {
  sensors::GnssInput in;
  in.position_m = Vec3E(kRe + 500.0e3, 0.0, 0.0);
  in.velocity_m_s = Vec3E(0.0, 7612.0, 0.0);
  return in;
}

}  // namespace

TEST(GnssFaultSchedule, WindowMembershipIsHalfOpen) {
  const std::vector<Event> events = {outage("gps_a", 100.0, 200.0)};
  EXPECT_FALSE(scenario::gnssFaultsAt(events, "gps_a", 99.0).outage);
  EXPECT_TRUE(scenario::gnssFaultsAt(events, "gps_a", 100.0).outage);  // inclusive start
  EXPECT_TRUE(scenario::gnssFaultsAt(events, "gps_a", 199.0).outage);
  EXPECT_FALSE(scenario::gnssFaultsAt(events, "gps_a", 200.0).outage);  // exclusive stop
}

TEST(GnssFaultSchedule, FiltersByUnitName) {
  const std::vector<Event> events = {outage("gps_a", 100.0, 200.0)};
  EXPECT_TRUE(scenario::gnssFaultsAt(events, "gps_a", 150.0).outage);
  EXPECT_FALSE(scenario::gnssFaultsAt(events, "gps_b", 150.0).outage);
}

TEST(GnssFaultSchedule, SpoofAndClockJumpCarryTheirParameters) {
  Event spoof;
  spoof.unit = "gps_a";
  spoof.type = Event::Type::kSpoof;
  spoof.start_s = 0.0;
  spoof.stop_s = 10.0;
  spoof.spoof_offset_ecef_m = Eigen::Vector3d(1000.0, 0.0, 0.0);
  Event jump;
  jump.unit = "gps_a";
  jump.type = Event::Type::kClockJump;
  jump.start_s = 0.0;
  jump.stop_s = 10.0;
  jump.clock_jump_s = 1.0e-6;

  const auto st = scenario::gnssFaultsAt({spoof, jump}, "gps_a", 5.0);
  EXPECT_TRUE(st.spoof);
  EXPECT_EQ(st.spoof_offset_ecef_m.x(), 1000.0);
  EXPECT_EQ(st.clock_jump_s, 1.0e-6);
  EXPECT_FALSE(st.outage);
}

TEST(GnssFaultSchedule, OverlappingSpoofsSum) {
  Event a;
  a.unit = "gps_a";
  a.type = Event::Type::kSpoof;
  a.start_s = 0.0;
  a.stop_s = 10.0;
  a.spoof_offset_ecef_m = Eigen::Vector3d(100.0, 0.0, 0.0);
  Event b = a;
  b.spoof_offset_ecef_m = Eigen::Vector3d(0.0, 50.0, 0.0);
  const auto st = scenario::gnssFaultsAt({a, b}, "gps_a", 5.0);
  EXPECT_EQ(st.spoof_offset_ecef_m.x(), 100.0);
  EXPECT_EQ(st.spoof_offset_ecef_m.y(), 50.0);
}

TEST(GnssFaultApply, OutageWindowInvalidatesThenRecovers) {
  const std::vector<Event> events = {outage("gps_a", 100.0, 200.0)};
  sensors::Gnss g(cleanSpec(), 20260101, 0x1234);
  const auto in = inputOnXAxis();

  auto sampleAt = [&](double t_s) {
    scenario::applyGnssFaults(g, scenario::gnssFaultsAt(events, "gps_a", t_s));
    return g.sample(pt::Tai::fromNanosecondsSinceEpoch(kEpoch.nanosecondsSinceEpoch() +
                                                       static_cast<std::int64_t>(t_s * 1.0e9)),
                    in);
  };

  EXPECT_TRUE(sampleAt(50.0).valid);    // before the window
  EXPECT_FALSE(sampleAt(150.0).valid);  // inside — outage
  EXPECT_TRUE(sampleAt(250.0).valid);   // after — the schedule cleared it
}

TEST(GnssFaultApply, SpoofWindowOffsetsButStaysValidThenClears) {
  Event spoof;
  spoof.unit = "gps_a";
  spoof.type = Event::Type::kSpoof;
  spoof.start_s = 100.0;
  spoof.stop_s = 200.0;
  spoof.spoof_offset_ecef_m = Eigen::Vector3d(1000.0, 0.0, 0.0);

  auto s = cleanSpec();
  s.noise_enabled = false;  // isolate the spoof offset from measurement noise
  sensors::Gnss g(s, 20260101, 0x1234);
  const auto in = inputOnXAxis();
  const double truth_x = in.position_m.eigen().x();

  auto sampleAt = [&](double t_s) {
    scenario::applyGnssFaults(g, scenario::gnssFaultsAt({spoof}, "gps_a", t_s));
    return g.sample(pt::Tai::fromNanosecondsSinceEpoch(kEpoch.nanosecondsSinceEpoch() +
                                                       static_cast<std::int64_t>(t_s * 1.0e9)),
                    in);
  };

  const auto before = sampleAt(50.0);
  EXPECT_TRUE(before.valid);
  EXPECT_NEAR(before.position_m.eigen().x(), truth_x, 1e-6);

  const auto during = sampleAt(150.0);
  EXPECT_TRUE(during.valid);  // a spoof looks like a real fix
  EXPECT_NEAR(during.position_m.eigen().x(), truth_x + 1000.0, 1e-6);

  const auto after = sampleAt(250.0);
  EXPECT_TRUE(after.valid);
  EXPECT_NEAR(after.position_m.eigen().x(), truth_x, 1e-6);  // offset cleared
}
