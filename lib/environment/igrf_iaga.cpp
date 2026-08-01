/// @file
/// @brief IAGA IGRF coefficient-file reader. See `igrf_iaga.hpp` for the contract.
///
/// Flight-safe parsing: `<cstdio>` line reads into a fixed buffer, `strtod`
/// scanning, no `std::string`/`std::vector`, no exceptions. ponytail: cstdio at
/// load/upload only, never steady state.

#include "environment/igrf_iaga.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace polaris::environment {
namespace {

/// Write @p msg into @p reason (if provided) and return false, so callers can
/// `return fail(...)` on every rejection path.
bool fail(char* reason, std::size_t reason_cap, const char* msg) {
  if (reason != nullptr && reason_cap > 0) {
    std::snprintf(reason, reason_cap, "%s", msg);
  }
  return false;
}

/// Same, with one integer detail (a line number) appended.
bool failAt(char* reason, std::size_t reason_cap, const char* msg, int line_number) {
  if (reason != nullptr && reason_cap > 0) {
    std::snprintf(reason, reason_cap, "%s at line %d", msg, line_number);
  }
  return false;
}

/// Advance @p cursor past spaces/tabs and read one whitespace-delimited token
/// into @p token (capacity @p cap). False at end of line or if the token does
/// not fit.
bool nextToken(const char*& cursor, char* token, std::size_t cap) {
  while (*cursor == ' ' || *cursor == '\t') {
    ++cursor;
  }
  if (*cursor == '\0' || *cursor == '\n' || *cursor == '\r') {
    return false;
  }
  std::size_t n = 0;
  while (*cursor != '\0' && *cursor != '\n' && *cursor != '\r' && *cursor != ' ' &&
         *cursor != '\t') {
    if (n + 1 >= cap) {
      return false;  // token longer than the buffer: malformed
    }
    token[n] = *cursor;
    ++n;
    ++cursor;
  }
  token[n] = '\0';
  return true;
}

/// Read the next token as a full double. False if there is no token or it has
/// trailing non-numeric characters (a silently truncated number is worse than a
/// rejected file).
bool nextDouble(const char*& cursor, double& out) {
  char token[64];
  if (!nextToken(cursor, token, sizeof(token))) {
    return false;
  }
  char* end = nullptr;
  const double v = std::strtod(token, &end);
  if (end == token || *end != '\0' || !std::isfinite(v)) {
    return false;
  }
  out = v;
  return true;
}

/// Parse the `g/h n m <epoch>...<sv>` header row into @p epochs.
///
/// The final column is labelled by interval ("2025-30") rather than by a year,
/// so it is not parsed as an epoch — it is the secular-variation column and the
/// caller handles it positionally. Because that label is not a number, the row
/// is read by *counting* fields first and only converting the leading ones.
bool parseEpochRow(const char* line, double* epochs, std::size_t cap, std::size_t& count) {
  const char* cursor = line;
  char token[64];
  // Leading "g/h n m" labels.
  if (!nextToken(cursor, token, sizeof(token)) || std::strcmp(token, "g/h") != 0) {
    return false;
  }
  if (!nextToken(cursor, token, sizeof(token))) {
    return false;
  }
  if (!nextToken(cursor, token, sizeof(token))) {
    return false;
  }

  // Every remaining field except the trailing SV column is an epoch year. Read
  // one ahead so the last field is dropped without a second pass.
  count = 0;
  char pending[64];
  bool have_pending = false;
  while (nextToken(cursor, token, sizeof(token))) {
    if (have_pending) {
      if (count >= cap) {
        return false;  // more epochs than the fixed table holds
      }
      char* end = nullptr;
      const double year = std::strtod(pending, &end);
      if (end == pending || *end != '\0' || !std::isfinite(year)) {
        return false;
      }
      epochs[count] = year;
      ++count;
    }
    std::memcpy(pending, token, std::strlen(token) + 1);  // copy the string, not the buffer
    have_pending = true;
  }
  return have_pending && count >= 2;
}

}  // namespace

bool loadIgrfIaga(const char* path, double decimal_year, IgrfCoefficients& out, char* reason,
                  std::size_t reason_cap) {
  if (path == nullptr) {
    return fail(reason, reason_cap, "null IGRF coefficient path");
  }
  if (!std::isfinite(decimal_year)) {
    return fail(reason, reason_cap, "decimal_year is not finite");
  }

  std::FILE* file = std::fopen(path, "r");
  if (file == nullptr) {
    return fail(reason, reason_cap, "cannot open IGRF coefficient file");
  }

  double epochs[kIgrfMaxEpochs] = {};
  std::size_t epoch_count = 0;
  bool have_epochs = false;
  // Index of the base epoch column, and the rate applied per year. Resolved once
  // the epoch row is known; see the header for why one (epoch, rate) pair
  // reproduces the model exactly.
  std::size_t base_index = 0;
  std::size_t next_index = 0;
  bool use_sv_column = false;

  out = IgrfCoefficients{};
  int max_degree = 0;
  int max_sv_degree = 0;
  int line_number = 0;
  std::size_t rows = 0;
  char line[kIgrfMaxLineLength];

  while (std::fgets(line, static_cast<int>(sizeof(line)), file) != nullptr) {
    ++line_number;
    if (line[0] == '\0' || line[0] == '\n' || line[0] == '\r' || line[0] == '#') {
      continue;
    }
    if (std::strncmp(line, "c/s", 3) == 0) {  // the "c/s deg ord ..." type row
      continue;
    }

    if (!have_epochs) {
      if (!parseEpochRow(line, epochs, kIgrfMaxEpochs, epoch_count)) {
        std::fclose(file);
        return failAt(reason, reason_cap, "unrecognised IGRF epoch header", line_number);
      }
      have_epochs = true;

      if (decimal_year < epochs[0]) {
        std::fclose(file);
        return fail(reason, reason_cap, "decimal_year precedes the first tabulated IGRF epoch");
      }
      if (decimal_year >= epochs[epoch_count - 1]) {
        base_index = epoch_count - 1;
        use_sv_column = true;
        // The SV column is published for the 5-year interval following the last
        // tabulated epoch; past that the model says nothing.
        out.valid_until_year = epochs[base_index] + kIgrfSvIntervalYears;
      } else {
        base_index = 0;
        while (base_index + 1 < epoch_count && epochs[base_index + 1] <= decimal_year) {
          ++base_index;
        }
        next_index = base_index + 1;
        if (!(epochs[next_index] > epochs[base_index])) {
          std::fclose(file);
          return fail(reason, reason_cap, "IGRF epochs are not strictly increasing");
        }
        // Inside the grid the derived rate reproduces the published interpolation
        // exactly up to the next tabulated epoch, and is an extrapolation after.
        out.valid_until_year = epochs[next_index];
      }
      out.epoch_year = epochs[base_index];
      continue;
    }

    const char* cursor = line;
    char kind[8];
    if (!nextToken(cursor, kind, sizeof(kind))) {
      std::fclose(file);
      return failAt(reason, reason_cap, "malformed IGRF coefficient row", line_number);
    }
    if (std::strcmp(kind, "g") != 0 && std::strcmp(kind, "h") != 0) {
      std::fclose(file);
      return failAt(reason, reason_cap, "unexpected IGRF coefficient kind", line_number);
    }
    double n_value = 0.0;
    double m_value = 0.0;
    if (!nextDouble(cursor, n_value) || !nextDouble(cursor, m_value)) {
      std::fclose(file);
      return failAt(reason, reason_cap, "malformed IGRF degree/order", line_number);
    }
    const int n = static_cast<int>(n_value);
    const int m = static_cast<int>(m_value);
    if (n < 1 || n > kIgrfMaxDegree || m < 0 || m > n) {
      std::fclose(file);
      return failAt(reason, reason_cap, "IGRF degree/order out of range", line_number);
    }

    // One value per epoch column plus the SV column, and nothing after them.
    double values[kIgrfMaxEpochs + 1] = {};
    for (std::size_t i = 0; i < epoch_count + 1; ++i) {
      if (!nextDouble(cursor, values[i])) {
        std::fclose(file);
        return failAt(reason, reason_cap, "fewer values than expected on IGRF row", line_number);
      }
    }
    char extra[64];
    if (nextToken(cursor, extra, sizeof(extra))) {
      std::fclose(file);
      return failAt(reason, reason_cap, "more values than expected on IGRF row", line_number);
    }

    const double base = values[base_index];
    const double rate =
        use_sv_column ? values[epoch_count]
                      : (values[next_index] - base) / (epochs[next_index] - epochs[base_index]);

    // The SV column is only published to degree 8; a derived rate is exact at
    // every degree the file tabulates.
    const bool has_rate = !use_sv_column || n <= kIgrfMaxSvDegree;

    if (kind[0] == 'g') {
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
  std::fclose(file);

  if (!have_epochs || rows == 0) {
    return fail(reason, reason_cap, "no IGRF coefficients found in file");
  }

  out.degree = max_degree;
  out.sv_degree = max_sv_degree;
  if (!validIgrfCoefficients(out)) {
    return fail(reason, reason_cap, "loaded IGRF coefficients failed validation");
  }
  return true;
}

}  // namespace polaris::environment
