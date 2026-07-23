#include "sensors/gnss_jamming.hpp"

#include <cmath>
#include <fstream>
#include <sstream>

#include "constants/constants.hpp"

namespace polaris::sim::sensors {
namespace {

constexpr double kRad2Deg = 57.29577951308232;

/// Ray-casting point-in-polygon on (lon, lat) degrees. Odd crossings = inside.
/// The ring need not be explicitly closed; the edge from last vertex back to the
/// first is included. Points exactly on an edge are not guaranteed either way —
/// acceptable for coarse jamming zones.
bool pointInRing(double lon, double lat, const std::vector<std::array<double, 2>>& ring) {
  bool inside = false;
  const std::size_t n = ring.size();
  for (std::size_t i = 0, j = n - 1; i < n; j = i++) {
    const double xi = ring[i][0];
    const double yi = ring[i][1];
    const double xj = ring[j][0];
    const double yj = ring[j][1];
    const bool straddles = (yi > lat) != (yj > lat);
    if (straddles) {
      const double x_cross = (xj - xi) * (lat - yi) / (yj - yi) + xi;
      if (lon < x_cross) {
        inside = !inside;
      }
    }
  }
  return inside;
}

/// Case-insensitive-free, tolerant tag search: return the text between the first
/// `<tag>` at or after @p from and its `</tag>`, or empty if not found. @p from is
/// advanced past the closing tag.
std::string extractTag(const std::string& s, const std::string& tag, std::size_t& from) {
  const std::string open = "<" + tag;
  const std::string close = "</" + tag + ">";
  const std::size_t o = s.find(open, from);
  if (o == std::string::npos) {
    from = std::string::npos;
    return {};
  }
  // Skip to the end of the open tag (handles attributes: `<coordinates ...>`).
  const std::size_t o_end = s.find('>', o);
  if (o_end == std::string::npos) {
    from = std::string::npos;
    return {};
  }
  const std::size_t c = s.find(close, o_end);
  if (c == std::string::npos) {
    from = std::string::npos;
    return {};
  }
  from = c + close.size();
  return s.substr(o_end + 1, c - o_end - 1);
}

/// Strip a leading `<![CDATA[` / trailing `]]>` and surrounding whitespace.
std::string cleanText(std::string t) {
  const std::size_t cdata = t.find("<![CDATA[");
  if (cdata != std::string::npos) {
    const std::size_t end = t.find("]]>", cdata);
    if (end != std::string::npos) {
      t = t.substr(cdata + 9, end - (cdata + 9));
    }
  }
  const std::size_t a = t.find_first_not_of(" \t\r\n");
  const std::size_t b = t.find_last_not_of(" \t\r\n");
  return (a == std::string::npos) ? std::string{} : t.substr(a, b - a + 1);
}

/// Parse a `<coordinates>` body — whitespace-separated `lon,lat[,alt]` tuples —
/// into a ring. Malformed tuples are skipped.
std::vector<std::array<double, 2>> parseCoordinates(const std::string& body) {
  std::vector<std::array<double, 2>> ring;
  std::istringstream ss(body);
  std::string tuple;
  while (ss >> tuple) {
    double lon = 0.0;
    double lat = 0.0;
    // Replace commas with spaces and read the first two numbers.
    for (char& ch : tuple) {
      if (ch == ',') {
        ch = ' ';
      }
    }
    std::istringstream ts(tuple);
    if (ts >> lon >> lat) {
      ring.push_back({lon, lat});
    }
  }
  return ring;
}

}  // namespace

void ecefToGeodeticDeg(const math::Vec3<math::frames::ECEF>& r_ecef, double& lat_deg,
                       double& lon_deg) {
  const double x = r_ecef.eigen().x();
  const double y = r_ecef.eigen().y();
  const double z = r_ecef.eigen().z();

  constexpr double a = constants::wgs84::kSemiMajorAxis;
  constexpr double b = constants::wgs84::kSemiMinorAxis;
  constexpr double e2 = constants::wgs84::kEccentricitySq;
  const double ep2 = e2 / (1.0 - e2);  // second eccentricity squared

  const double p = std::sqrt(x * x + y * y);
  lon_deg = std::atan2(y, x) * kRad2Deg;

  if (p < 1.0e-9) {  // on the polar axis
    lat_deg = (z >= 0.0 ? 90.0 : -90.0);
    return;
  }
  // Bowring's closed-form geodetic latitude.
  const double theta = std::atan2(z * a, p * b);
  const double st = std::sin(theta);
  const double ct = std::cos(theta);
  const double lat = std::atan2(z + ep2 * b * st * st * st, p - e2 * a * ct * ct * ct);
  lat_deg = lat * kRad2Deg;
}

bool JammingRegions::fromKml(const std::string& kml_text, JammingRegions& out,
                             std::string* error) {
  out.regions_.clear();

  std::size_t pm = 0;
  while (pm != std::string::npos) {
    std::size_t pm_scan = pm;
    const std::string placemark = extractTag(kml_text, "Placemark", pm_scan);
    if (pm_scan == std::string::npos) {
      break;
    }
    const std::size_t placemark_end = pm_scan;

    std::size_t name_scan = 0;
    const std::string name = cleanText(extractTag(placemark, "name", name_scan));

    // Every outer boundary in this placemark becomes a region (a MultiGeometry of
    // several polygons under one name yields several rings with the same name).
    std::size_t ob = 0;
    while (ob != std::string::npos) {
      std::size_t ob_scan = ob;
      const std::string outer = extractTag(placemark, "outerBoundaryIs", ob_scan);
      if (ob_scan == std::string::npos) {
        break;
      }
      std::size_t coord_scan = 0;
      const std::string coords = extractTag(outer, "coordinates", coord_scan);
      auto ring = parseCoordinates(coords);
      if (ring.size() >= 3) {
        out.regions_.push_back({name, std::move(ring)});
      }
      ob = ob_scan;
    }
    pm = placemark_end;
  }

  if (out.regions_.empty()) {
    if (error != nullptr) {
      *error = "KML contained no usable <Polygon> outer boundary";
    }
    return false;
  }
  return true;
}

bool JammingRegions::loadKmlFile(const std::string& path, JammingRegions& out,
                                 std::string* error) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    if (error != nullptr) {
      *error = "cannot open GNSS jamming KML: " + path;
    }
    return false;
  }
  std::ostringstream ss;
  ss << in.rdbuf();
  return fromKml(ss.str(), out, error);
}

const std::string* JammingRegions::jammedRegion(
    const math::Vec3<math::frames::ECEF>& r_ecef) const {
  if (regions_.empty()) {
    return nullptr;
  }
  double lat = 0.0;
  double lon = 0.0;
  ecefToGeodeticDeg(r_ecef, lat, lon);
  for (const JammingRegion& region : regions_) {
    if (pointInRing(lon, lat, region.ring)) {
      return &region.name;
    }
  }
  return nullptr;
}

}  // namespace polaris::sim::sensors
