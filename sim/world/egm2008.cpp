/// @file
/// @brief EGM2008 `.gfc` (ICGEM) coefficient loader. See egm2008.hpp.

#include "world/egm2008.hpp"

#include <algorithm>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace polaris::sim::world {

namespace {

/// Zero-filled triangular table: row n has n+1 columns (m = 0..n) — the shape the
/// `GravityCoeffs` invariant and the recursion require.
std::vector<std::vector<double>> triangular(int nmax) {
  std::vector<std::vector<double>> t(static_cast<std::size_t>(nmax) + 1);
  for (int n = 0; n <= nmax; ++n) {
    t[static_cast<std::size_t>(n)].assign(static_cast<std::size_t>(n) + 1, 0.0);
  }
  return t;
}

/// Fortran `D`/`d` exponents -> `E` so `std::istringstream` parses them. A pure
/// character fix on read, not a rewrite of the committed file (§3.7).
void normalizeExponent(std::string& s) {
  for (char& ch : s) {
    if (ch == 'D' || ch == 'd') {
      ch = 'E';
    }
  }
}

}  // namespace

GravityCoeffs loadEgm2008Gfc(std::istream& in, int max_degree, Egm2008Header* header) {
  if (max_degree < 0) {
    throw std::runtime_error("loadEgm2008Gfc: max_degree must be >= 0");
  }

  Egm2008Header hdr;
  std::string line;

  // --- Header block: scan until end_of_head, capturing GM / radius / max_degree.
  // Extract via the stream (no throwing conversions; a malformed value leaves the
  // field at its benign default rather than raising an inconsistent exception
  // type). The Fortran `D` exponent is normalized on the *value token only* — not
  // the whole line, which would corrupt keywords like `end_of_head`/`max_degree`.
  // A file with no header block at all is fine — we fall through to the data lines.
  while (std::getline(in, line)) {
    std::istringstream ls(line);
    std::string key;
    ls >> key;
    if (key == "earth_gravity_constant" || key == "radius") {
      std::string value;
      ls >> value;
      normalizeExponent(value);
      std::istringstream vs(value);
      (key == "radius" ? vs >> hdr.radius : vs >> hdr.gm);
    } else if (key == "max_degree") {
      ls >> hdr.max_degree;
    } else if (key == "end_of_head") {
      break;
    }
  }
  // Some trimmed/simple .gfc files omit the header block entirely; that's fine —
  // we just won't have GM/radius and truncate to the requested degree.
  const int file_max = hdr.max_degree > 0 ? hdr.max_degree : max_degree;
  const int nmax = std::min(max_degree, file_max);

  GravityCoeffs g;
  g.nmax = nmax;
  g.C = triangular(nmax);
  g.S = triangular(nmax);
  g.C[0][0] = 1.0;  // point-mass monopole, independent of the file

  // --- Data lines: gfc  L  M  Cbar  Sbar  [sigmaC sigmaS].
  int loaded = 0;
  int max_degree_seen = -1;
  while (std::getline(in, line)) {
    normalizeExponent(line);
    std::istringstream ls(line);
    std::string tag;
    ls >> tag;
    // `gfc` = static coefficient; `gfct` = time-variable reference value (its
    // secular/periodic partners are separate records we don't model) — take its
    // base value the same way.
    if (tag != "gfc" && tag != "gfct") {
      continue;
    }
    int degree = -1;
    int order = -1;
    double c = 0.0;
    double s = 0.0;
    if (!(ls >> degree >> order >> c >> s)) {
      continue;  // malformed line — skip rather than abort the whole load
    }
    if (degree < 0 || order < 0 || order > degree || degree > nmax) {
      continue;  // beyond the requested truncation (or degree-order swap) — skip
    }
    const auto un = static_cast<std::size_t>(degree);
    const auto um = static_cast<std::size_t>(order);
    g.C[un][um] = c;
    g.S[un][um] = s;
    max_degree_seen = std::max(max_degree_seen, degree);
    ++loaded;
  }

  if (loaded == 0) {
    throw std::runtime_error("loadEgm2008Gfc: no gfc coefficient lines found");
  }
  // Coverage guard: the file must actually carry coefficients up to the requested
  // degree, not just declare a large max_degree in its header. Otherwise a caller
  // would silently get a lower-fidelity field (high-degree rows all zero) — a
  // deliberate-fidelity hazard (sim/CLAUDE.md). Degrees 0/1 are special (monopole
  // is forced, degree-1 geocenter is zero), so only guard nmax >= 2.
  if (nmax >= 2 && max_degree_seen < nmax) {
    throw std::runtime_error("loadEgm2008Gfc: file covers only degree " +
                             std::to_string(max_degree_seen) + " but degree " +
                             std::to_string(nmax) + " was requested");
  }
  g.C[0][0] = 1.0;  // re-assert: a file listing '0 0' must not override the monopole
  if (header != nullptr) {
    *header = hdr;
  }
  return g;
}

GravityCoeffs loadEgm2008Gfc(const std::string& path, int max_degree, Egm2008Header* header) {
  std::ifstream file(path);
  if (!file.is_open()) {
    throw std::runtime_error("loadEgm2008Gfc: cannot open " + path);
  }
  return loadEgm2008Gfc(file, max_degree, header);
}

}  // namespace polaris::sim::world
