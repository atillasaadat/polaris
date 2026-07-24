#ifndef POLARIS_SIM_SENSORS_GNSS_JAMMING_HPP
#define POLARIS_SIM_SENSORS_GNSS_JAMMING_HPP

/// @file
/// @brief Geographic GNSS-jamming regions from a KML polygon set (§6.2, §9.2).
///
/// A GNSS receiver overflying a terrestrial jammer loses its fix — a real,
/// geographically-fixed effect (persistent interference over conflict zones is
/// routinely observed *from orbit*, not just on the ground). This models it as a
/// set of lon/lat polygons: when the sub-satellite point falls inside one, the
/// receiver is jammed and its fixes go invalid, with the same reacquisition delay
/// on exit as any other outage. The receiver does not "know" geography — the sim
/// binds position to the RF-environment map, exactly as `shadow_factor` binds
/// position to the eclipse model for the sun sensor.
///
/// **Regions come from a config-provided KML** (design doc §3.7 committed-verbatim
/// spirit — the KML is authored as a scenario input and parsed as-is, not
/// pre-digested). Each `<Placemark>` is one named region; its polygon's outer
/// `<LinearRing>` is the boundary. This is a *jamming map*, not a propagation
/// model: it answers "is this lon/lat inside a jammed area", nothing about signal
/// power or partial degradation — a first-order, binary model, which is what a
/// keep-out-region scenario needs.
///
/// **Limits (documented, not hidden):** inner boundaries (polygon holes) are
/// ignored — a jammed zone with a clear hole is not a realistic requirement — and
/// a polygon crossing the ±180° antimeridian must be split into two, because the
/// point-in-polygon test works in raw lon degrees without wrap handling.
///
/// Sim-side: file I/O, heap, exceptions-free error returns.
///
/// Implements REQ-SIM-005 (scriptable fault injection: geographic GNSS jamming).

#include <array>
#include <cstddef>
#include <string>
#include <vector>

#include "math/frames.hpp"
#include "math/typed_vector.hpp"

namespace polaris::sim::sensors {

/// Convert an ECEF position to WGS84 **geodetic** latitude/longitude [deg] via
/// Bowring's closed-form (sub-mm for terrestrial radii, and the geodetic latitude
/// of a point at altitude is the latitude of its nadir foot — the sub-satellite
/// point). Longitude is `atan2(y, x)`, in (-180, 180].
///
/// With \f$p = \sqrt{x^2+y^2}\f$, semi-axes \f$a, b\f$, eccentricities
/// \f$e^2, e'^2 = e^2/(1-e^2)\f$, and parametric latitude
/// \f$\theta = \operatorname{atan2}(z\,a,\ p\,b)\f$:
/// \f[
///   \lambda = \operatorname{atan2}(y, x), \qquad
///   \varphi = \operatorname{atan2}\!\big(z + e'^2\, b \sin^3\theta,\ \ p - e^2\, a
///   \cos^3\theta\big).
/// \f]
/// On the polar axis (\f$p < 10^{-9}\f$) latitude is \f$\pm 90°\f$.
void ecefToGeodeticDeg(const math::Vec3<math::frames::ECEF>& r_ecef, double& lat_deg,
                       double& lon_deg);

/// One named jamming region: the outer boundary as (lon, lat) degrees, in the
/// order the KML lists them (closed or not — the point-in-polygon test closes it).
struct JammingRegion {
  std::string name;
  std::vector<std::array<double, 2>> ring;  ///< {lon_deg, lat_deg} vertices
};

/// A set of jamming regions with a point-in-any-region test.
///
/// **Point-in-region test.** The sub-satellite point \f$(\lambda, \varphi)\f$
/// (from `ecefToGeodeticDeg`) is tested against each region's ring by ray casting:
/// for ring vertices \f$(x_i, y_i)\f$ (lon, lat), the point is inside when an odd
/// number of edges straddle its latitude and lie to its east —
/// \f[
///   \big[(y_i > \varphi) \neq (y_j > \varphi)\big]
///   \ \land\ \lambda < x_i + (x_j - x_i)\,\frac{\varphi - y_i}{y_j - y_i},
/// \f]
/// counted over all edges \f$(j, i)\f$ with the ring implicitly closed. The test is
/// in raw longitude degrees, so a region must not cross the ±180° antimeridian.
class JammingRegions {
 public:
  JammingRegions() = default;

  /// Parse KML text into regions. Extracts each `<Placemark>`'s name and the
  /// outer-boundary `<coordinates>` of every `<Polygon>` it contains. Returns
  /// false (with @p error) if the text has no usable polygon — an empty jamming
  /// map is almost always a wrong path or malformed file, worth failing on.
  static bool fromKml(const std::string& kml_text, JammingRegions& out,
                      std::string* error = nullptr);

  /// Load and parse a KML file. False if it cannot be opened or `fromKml` fails.
  static bool loadKmlFile(const std::string& path, JammingRegions& out,
                          std::string* error = nullptr);

  /// The name of the first region containing the sub-satellite point of
  /// @p r_ecef, or nullptr if none. A pointer so the caller can telemeter *which*
  /// zone jammed the receiver.
  [[nodiscard]] const std::string* jammedRegion(const math::Vec3<math::frames::ECEF>& r_ecef) const;

  /// Whether @p r_ecef's sub-satellite point is inside any region.
  [[nodiscard]] bool jammed(const math::Vec3<math::frames::ECEF>& r_ecef) const {
    return jammedRegion(r_ecef) != nullptr;
  }

  [[nodiscard]] bool empty() const { return regions_.empty(); }

  [[nodiscard]] std::size_t size() const { return regions_.size(); }

  [[nodiscard]] const std::vector<JammingRegion>& regions() const { return regions_; }

 private:
  std::vector<JammingRegion> regions_;
};

}  // namespace polaris::sim::sensors

#endif  // POLARIS_SIM_SENSORS_GNSS_JAMMING_HPP
