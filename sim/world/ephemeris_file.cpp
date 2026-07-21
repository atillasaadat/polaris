#include "world/ephemeris_file.hpp"

#include <cstdint>
#include <fstream>
#include <sstream>

namespace polaris::sim::world {
namespace {

/// Read `degree + 1` coefficients into @p out. False if the line runs short or
/// carries a non-numeric field.
bool readCoefficients(std::istringstream& in, int degree, double* out) {
  for (int i = 0; i <= degree; ++i) {
    if (!(in >> out[i])) {
      return false;
    }
  }
  return true;
}

bool fail(std::string* error, const std::string& message) {
  if (error != nullptr) {
    *error = message;
  }
  return false;
}

}  // namespace

bool loadEphemerisFile(const std::string& path, EphemerisSet& out, std::string* error) {
  std::ifstream file(path);
  if (!file.good()) {
    return fail(error, "cannot open ephemeris fixture: " + path);
  }

  std::string line;
  int line_number = 0;
  std::size_t loaded = 0;
  while (std::getline(file, line)) {
    ++line_number;
    if (line.empty() || line[0] == '#') {
      continue;
    }

    std::istringstream in(line);
    std::string tag;
    std::string body;
    ephemeris::ChebyshevSegment seg;
    std::int64_t mid_ns = 0;
    if (!(in >> tag >> body >> mid_ns >> seg.radius_seconds >> seg.degree)) {
      return fail(error, "malformed segment header at line " + std::to_string(line_number));
    }
    if (tag != "seg") {
      return fail(error, "unexpected record '" + tag + "' at line " + std::to_string(line_number));
    }
    if (seg.degree < 0 || seg.degree > ephemeris::kMaxChebyshevDegree) {
      return fail(error, "degree " + std::to_string(seg.degree) + " out of range at line " +
                             std::to_string(line_number));
    }
    seg.mid_ns = mid_ns;

    if (!readCoefficients(in, seg.degree, seg.cx) || !readCoefficients(in, seg.degree, seg.cy) ||
        !readCoefficients(in, seg.degree, seg.cz)) {
      return fail(error, "truncated coefficients at line " + std::to_string(line_number));
    }

    SimEphemerisTable* table = nullptr;
    if (body == "sun") {
      table = &out.sun;
    } else if (body == "moon") {
      table = &out.moon;
    } else {
      return fail(error, "unknown body '" + body + "' at line " + std::to_string(line_number));
    }

    // addSegment rejects malformed segments too; a false here past the checks
    // above means the table is full.
    if (!table->addSegment(seg)) {
      return fail(error, "ephemeris capacity exceeded for '" + body + "' at line " +
                             std::to_string(line_number) +
                             " (raise kEphemerisCapacity or "
                             "regenerate with longer intervals)");
    }
    ++loaded;
  }

  if (loaded == 0) {
    return fail(error, "no segments found in " + path);
  }
  return true;
}

BodyPositionFn bodyPositionFn(const SimEphemerisTable& table) {
  return [&table](const time::Tdb& t, math::Vec3<math::frames::ECI>& out) {
    return table.position(t, out);
  };
}

}  // namespace polaris::sim::world
