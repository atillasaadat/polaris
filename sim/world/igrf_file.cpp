#include "world/igrf_file.hpp"

#include <cmath>
#include <fstream>
#include <sstream>
#include <vector>

namespace polaris::sim::world {
namespace {

bool fail(std::string* error, const std::string& message) {
  if (error != nullptr) {
    *error = message;
  }
  return false;
}

/// Parse the `g/h n m <epoch>...<sv>` header row into its epoch years.
///
/// The final column is labelled by interval ("2025-30") rather than by a year,
/// so it is not parsed as an epoch — it is the secular-variation column and the
/// caller handles it positionally.
bool parseEpochRow(const std::string& line, std::vector<double>& epochs) {
  std::istringstream in(line);
  std::string tag;
  std::string n_label;
  std::string m_label;
  if (!(in >> tag >> n_label >> m_label) || tag != "g/h") {
    return false;
  }
  std::vector<std::string> fields;
  for (std::string field; in >> field;) {
    fields.push_back(field);
  }
  if (fields.size() < 3) {
    return false;
  }
  // Drop the trailing SV column.
  fields.pop_back();
  epochs.clear();
  for (const std::string& field : fields) {
    std::istringstream value(field);
    double year = 0.0;
    if (!(value >> year) || !value.eof()) {
      return false;
    }
    epochs.push_back(year);
  }
  return epochs.size() >= 2;
}

}  // namespace

bool loadIgrfFile(const std::string& path, double decimal_year, environment::IgrfCoefficients& out,
                  std::string* error) {
  if (!std::isfinite(decimal_year)) {
    return fail(error, "decimal_year is not finite");
  }

  std::ifstream file(path);
  if (!file.good()) {
    return fail(error, "cannot open IGRF coefficient file: " + path);
  }

  std::vector<double> epochs;
  bool have_epochs = false;
  // Index of the base epoch column, and the rate applied per year. Resolved once
  // the epoch row is known; see the header for why one (epoch, rate) pair is
  // enough to reproduce the model exactly.
  std::size_t base_index = 0;
  std::size_t next_index = 0;
  bool use_sv_column = false;

  out = environment::IgrfCoefficients{};
  int max_degree = 0;
  int max_sv_degree = 0;
  int line_number = 0;
  std::size_t rows = 0;
  std::string line;

  while (std::getline(file, line)) {
    ++line_number;
    if (line.empty() || line[0] == '#') {
      continue;
    }
    if (line.compare(0, 3, "c/s") == 0) {  // the "c/s deg ord ..." type row
      continue;
    }
    if (!have_epochs) {
      if (!parseEpochRow(line, epochs)) {
        return fail(error, "unrecognised IGRF epoch header at line " + std::to_string(line_number));
      }
      have_epochs = true;

      const double first = epochs.front();
      const double last = epochs.back();
      if (decimal_year < first) {
        return fail(error, "decimal_year " + std::to_string(decimal_year) +
                               " precedes the first tabulated epoch " + std::to_string(first));
      }
      if (decimal_year >= last) {
        base_index = epochs.size() - 1;
        use_sv_column = true;
      } else {
        base_index = 0;
        while (base_index + 1 < epochs.size() && epochs[base_index + 1] <= decimal_year) {
          ++base_index;
        }
        next_index = base_index + 1;
        if (!(epochs[next_index] > epochs[base_index])) {
          return fail(error, "IGRF epochs are not strictly increasing");
        }
      }
      out.epoch_year = epochs[base_index];
      continue;
    }

    std::istringstream in(line);
    std::string kind;
    int n = 0;
    int m = 0;
    if (!(in >> kind >> n >> m)) {
      return fail(error, "malformed coefficient row at line " + std::to_string(line_number));
    }
    if (kind != "g" && kind != "h") {
      return fail(error, "unexpected coefficient kind '" + kind + "' at line " +
                             std::to_string(line_number));
    }
    if (n < 1 || n > environment::kIgrfMaxDegree || m < 0 || m > n) {
      return fail(error, "degree/order " + std::to_string(n) + "/" + std::to_string(m) +
                             " out of range at line " + std::to_string(line_number));
    }

    std::vector<double> values;
    values.reserve(epochs.size() + 1);
    for (double v = 0.0; in >> v;) {
      values.push_back(v);
    }
    if (values.size() != epochs.size() + 1) {
      return fail(error, "expected " + std::to_string(epochs.size() + 1) + " values but found " +
                             std::to_string(values.size()) + " at line " +
                             std::to_string(line_number));
    }

    const double base = values[base_index];
    const double rate =
        use_sv_column ? values.back()
                      : (values[next_index] - base) / (epochs[next_index] - epochs[base_index]);

    // The SV column is only published to degree 8; a derived rate is exact at
    // every degree the file tabulates.
    const bool has_rate = !use_sv_column || n <= environment::kIgrfMaxSvDegree;

    if (kind == "g") {
      out.g[n][m] = base;
      out.g_sv[n][m] = has_rate ? rate : 0.0;
    } else {
      out.h[n][m] = base;
      out.h_sv[n][m] = has_rate ? rate : 0.0;
    }

    if (n > max_degree) {
      max_degree = n;
    }
    if (has_rate && n > max_sv_degree) {
      max_sv_degree = n;
    }
    ++rows;
  }

  if (!have_epochs || rows == 0) {
    return fail(error, "no IGRF coefficients found in " + path);
  }

  out.degree = max_degree;
  out.sv_degree = max_sv_degree;
  if (!environment::validIgrfCoefficients(out)) {
    return fail(error, "loaded IGRF coefficients failed validation");
  }
  return true;
}

}  // namespace polaris::sim::world
