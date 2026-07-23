/// @file Unit tests for GNSS geographic jamming (§6.2, §9.2).
///
/// Three layers: the ECEF→geodetic conversion (round-tripped against the forward
/// WGS84 formula), the KML polygon parser, and point-in-region membership. The
/// binding into the GNSS receiver — jamming as a geographically-gated outage with
/// a reacquisition delay on exit — is exercised in sim_sensors_gnss_test.cpp.

#include <gtest/gtest.h>

#include <cmath>
#include <Eigen/Core>
#include <string>

#include "constants/constants.hpp"
#include "math/frames.hpp"
#include "math/typed_vector.hpp"
#include "sensors/gnss_jamming.hpp"

namespace {

namespace sensors = polaris::sim::sensors;
namespace pm = polaris::math;
namespace pmf = polaris::math::frames;

using Vec3E = pm::Vec3<pmf::ECEF>;

constexpr double kDeg2Rad = 0.017453292519943295;

/// Forward WGS84: geodetic lat/lon [deg] + altitude [m] → ECEF, so a test can
/// place a satellite over a known sub-satellite point.
Vec3E ecefFromGeodetic(double lat_deg, double lon_deg, double alt_m) {
  const double a = polaris::constants::wgs84::kSemiMajorAxis;
  const double e2 = polaris::constants::wgs84::kEccentricitySq;
  const double lat = lat_deg * kDeg2Rad;
  const double lon = lon_deg * kDeg2Rad;
  const double n = a / std::sqrt(1.0 - e2 * std::sin(lat) * std::sin(lat));
  return Vec3E((n + alt_m) * std::cos(lat) * std::cos(lon),
               (n + alt_m) * std::cos(lat) * std::sin(lon),
               (n * (1.0 - e2) + alt_m) * std::sin(lat));
}

/// A single-square KML region, lon in [lon0,lon1], lat in [lat0,lat1].
std::string squareKml(const std::string& name, double lon0, double lat0, double lon1, double lat1) {
  auto c = [](double lon, double lat) {
    return std::to_string(lon) + "," + std::to_string(lat) + ",0 ";
  };
  return "<kml><Document><Placemark><name>" + name +
         "</name><Polygon><outerBoundaryIs><LinearRing><coordinates>" + c(lon0, lat0) +
         c(lon1, lat0) + c(lon1, lat1) + c(lon0, lat1) + c(lon0, lat0) +
         "</coordinates></LinearRing></outerBoundaryIs></Polygon></Placemark></Document></kml>";
}

}  // namespace

TEST(GnssGeodetic, RoundTripsTheForwardFormula) {
  for (double lat : {-60.0, -30.0, 0.0, 30.0, 45.0, 60.0}) {
    for (double lon : {-170.0, -90.0, 0.0, 30.0, 90.0, 179.0}) {
      const auto r = ecefFromGeodetic(lat, lon, 500.0e3);
      double got_lat = 0.0;
      double got_lon = 0.0;
      sensors::ecefToGeodeticDeg(r, got_lat, got_lon);
      EXPECT_NEAR(got_lat, lat, 1e-6) << "lat " << lat << " lon " << lon;
      EXPECT_NEAR(got_lon, lon, 1e-6) << "lat " << lat << " lon " << lon;
    }
  }
}

TEST(GnssGeodetic, GeodeticLatitudeExceedsGeocentricAtMidLatitudes) {
  // At 45° geodetic the geocentric latitude is ~0.19° lower — the whole reason to
  // use geodetic lat against a KML, which is authored in geodetic coordinates.
  const auto r = ecefFromGeodetic(45.0, 0.0, 0.0);
  const double geocentric =
      std::atan2(r.eigen().z(), std::hypot(r.eigen().x(), r.eigen().y())) / kDeg2Rad;
  double lat = 0.0;
  double lon = 0.0;
  sensors::ecefToGeodeticDeg(r, lat, lon);
  EXPECT_NEAR(lat, 45.0, 1e-6);
  EXPECT_GT(lat - geocentric, 0.1);
}

TEST(GnssJammingParse, ExtractsNamedPolygon) {
  sensors::JammingRegions regions;
  std::string error;
  ASSERT_TRUE(sensors::JammingRegions::fromKml(squareKml("Crimea", 32.0, 44.0, 37.0, 46.5), regions,
                                               &error))
      << error;
  ASSERT_EQ(regions.size(), 1u);
  EXPECT_EQ(regions.regions()[0].name, "Crimea");
  EXPECT_EQ(regions.regions()[0].ring.size(), 5u);  // closed square
}

TEST(GnssJammingParse, EmptyKmlIsAnError) {
  sensors::JammingRegions regions;
  std::string error;
  EXPECT_FALSE(sensors::JammingRegions::fromKml("<kml></kml>", regions, &error));
  EXPECT_FALSE(error.empty());
}

TEST(GnssJammingParse, ParsesMultiplePlacemarks) {
  std::string two = "<kml><Document>";
  two +=
      "<Placemark><name>A</name><Polygon><outerBoundaryIs><LinearRing><coordinates>"
      "0,0 10,0 10,10 0,10 0,0</coordinates></LinearRing></outerBoundaryIs></Polygon></Placemark>";
  two +=
      "<Placemark><name>B</name><Polygon><outerBoundaryIs><LinearRing><coordinates>"
      "20,20 30,20 30,30 20,30 "
      "20,20</coordinates></LinearRing></outerBoundaryIs></Polygon></Placemark>";
  two += "</Document></kml>";
  sensors::JammingRegions regions;
  ASSERT_TRUE(sensors::JammingRegions::fromKml(two, regions, nullptr));
  ASSERT_EQ(regions.size(), 2u);
  EXPECT_EQ(regions.regions()[0].name, "A");
  EXPECT_EQ(regions.regions()[1].name, "B");
}

TEST(GnssJammingMembership, InsideAndOutsideARegion) {
  sensors::JammingRegions regions;
  ASSERT_TRUE(sensors::JammingRegions::fromKml(squareKml("Crimea", 32.0, 44.0, 37.0, 46.5), regions,
                                               nullptr));

  // A satellite whose sub-satellite point is inside the box is jammed.
  const auto inside = ecefFromGeodetic(45.0, 34.5, 500.0e3);
  const std::string* hit = regions.jammedRegion(inside);
  ASSERT_NE(hit, nullptr);
  EXPECT_EQ(*hit, "Crimea");

  // Points outside — one far away, one just west of the box edge.
  EXPECT_FALSE(regions.jammed(ecefFromGeodetic(0.0, 0.0, 500.0e3)));
  EXPECT_FALSE(regions.jammed(ecefFromGeodetic(45.0, 31.0, 500.0e3)));
}

TEST(GnssJammingLoad, ReadsTheCommittedExampleKml) {
  // The example ships three regions; loading it is the config path a scenario uses.
  sensors::JammingRegions regions;
  std::string error;
  ASSERT_TRUE(sensors::JammingRegions::loadKmlFile(POLARIS_GNSS_JAMMING_KML, regions, &error))
      << error;
  EXPECT_EQ(regions.size(), 3u);
  // A point in the Black Sea / Crimea box is jammed.
  EXPECT_TRUE(regions.jammed(ecefFromGeodetic(45.0, 34.5, 500.0e3)));
}
