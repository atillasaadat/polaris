#include "gnc/tle.hpp"

#include <cmath>

#include "time/civil.hpp"
#include "time/utc.hpp"

namespace polaris::gnc {
namespace {

/// The 69 columns the format defines. Anything past them is a driver's
/// annotation (the verification fixture's start/stop/step), never data.
constexpr std::size_t kLineLength = 69;
constexpr std::size_t kChecksumColumn = 68;

constexpr double kMinutesPerDay = 1440.0;
/// The deep-space cut (Vallado 2006 §3): SDP4 runs at periods >= 225 min.
constexpr double kDeepSpacePeriodMin = 225.0;

bool isDigit(char c) {
  return c >= '0' && c <= '9';
}

/// Trim trailing CR/LF/space so a CRLF-terminated file parses like an LF one.
std::string_view rtrim(std::string_view s) {
  while (!s.empty()) {
    const char c = s.back();
    if (c == '\r' || c == '\n' || c == ' ' || c == '\t') {
      s.remove_suffix(1);
    } else {
      break;
    }
  }
  return s;
}

/// Parse a fixed-width signed decimal. Blanks are skipped; an empty or
/// all-blank field is *not* an error for the fields that permit it, so the
/// caller decides by checking @p any_digit.
bool parseDouble(std::string_view field, double& out, bool& any_digit) {
  double sign = 1.0;
  double value = 0.0;
  double frac_scale = 0.0;  // 0 => integer part; else 10^-k for the next digit
  any_digit = false;
  for (const char c : field) {
    if (c == ' ' || c == '\t') {
      continue;
    }
    if (c == '+') {
      continue;
    }
    if (c == '-') {
      // A sign is only meaningful before any digit; a '-' after digits is an
      // exponent marker, which this helper does not handle by design (the
      // assumed-exponent fields have their own parser below).
      if (any_digit) {
        return false;
      }
      sign = -1.0;
      continue;
    }
    if (c == '.') {
      if (frac_scale != 0.0) {
        return false;  // second decimal point
      }
      frac_scale = 0.1;
      continue;
    }
    if (!isDigit(c)) {
      return false;
    }
    const double digit = static_cast<double>(c - '0');
    if (frac_scale == 0.0) {
      value = value * 10.0 + digit;
    } else {
      value += digit * frac_scale;
      frac_scale *= 0.1;
    }
    any_digit = true;
  }
  out = sign * value;
  return true;
}

bool parseInt(std::string_view field, std::int32_t& out, bool& any_digit) {
  double as_double = 0.0;
  if (!parseDouble(field, as_double, any_digit)) {
    return false;
  }
  out = static_cast<std::int32_t>(as_double);
  return true;
}

/// A field with an **assumed decimal point** — `1859667` means 0.1859667.
bool parseAssumedDecimal(std::string_view field, double& out) {
  double mantissa = 0.0;
  bool any_digit = false;
  if (!parseDouble(field, mantissa, any_digit) || !any_digit) {
    return false;
  }
  // The implied point sits before the first column of the field, so the scale
  // is fixed by the field's *width*, not by how many digits happen to be there.
  double scale = 1.0;
  for (std::size_t i = 0; i < field.size(); ++i) {
    scale *= 0.1;
  }
  out = mantissa * scale;
  return true;
}

/// A field with an **assumed exponent** — `28098-4` means 0.28098e-4, and
/// ` 00000-0` means 0. The last two characters are a signed exponent; the rest
/// is a mantissa with an assumed leading decimal point.
bool parseAssumedExponent(std::string_view field, double& out) {
  if (field.size() < 3) {
    return false;
  }
  const std::string_view mantissa_field = field.substr(0, field.size() - 2);
  const std::string_view exponent_field = field.substr(field.size() - 2);

  double mantissa = 0.0;
  bool any_digit = false;
  if (!parseDouble(mantissa_field, mantissa, any_digit)) {
    return false;
  }
  if (!any_digit) {
    // An all-blank mantissa is how a zero drag term is sometimes written.
    out = 0.0;
    return true;
  }
  // Scale by the mantissa field's width, excluding a leading sign column, since
  // the implied point sits before the first *digit* column.
  std::size_t digit_columns = 0;
  for (const char c : mantissa_field) {
    if (isDigit(c)) {
      ++digit_columns;
    }
  }
  double scale = 1.0;
  for (std::size_t i = 0; i < digit_columns; ++i) {
    scale *= 0.1;
  }

  double exponent = 0.0;
  bool exp_digit = false;
  double exp_sign = 1.0;
  for (const char c : exponent_field) {
    if (c == ' ' || c == '+') {
      continue;
    }
    if (c == '-') {
      exp_sign = -1.0;
      continue;
    }
    if (!isDigit(c)) {
      return false;
    }
    exponent = exponent * 10.0 + static_cast<double>(c - '0');
    exp_digit = true;
  }
  if (!exp_digit) {
    exponent = 0.0;
  }
  out = mantissa * scale * std::pow(10.0, exp_sign * exponent);
  return true;
}

}  // namespace

const char* toString(TleStatus status) {
  switch (status) {
    case TleStatus::kOk:
      return "OK";
    case TleStatus::kLineTooShort:
      return "LINE_TOO_SHORT";
    case TleStatus::kBadLineNumber:
      return "BAD_LINE_NUMBER";
    case TleStatus::kSatelliteMismatch:
      return "SATELLITE_MISMATCH";
    case TleStatus::kBadChecksum:
      return "BAD_CHECKSUM";
    case TleStatus::kMalformedField:
      return "MALFORMED_FIELD";
    case TleStatus::kOutOfRange:
      return "OUT_OF_RANGE";
  }
  return "UNKNOWN";
}

int tleChecksum(std::string_view line) {
  int sum = 0;
  const std::size_t n = line.size() < kChecksumColumn ? line.size() : kChecksumColumn;
  for (std::size_t i = 0; i < n; ++i) {
    const char c = line[i];
    if (isDigit(c)) {
      sum += c - '0';
    } else if (c == '-') {
      sum += 1;
    }
  }
  return sum % 10;
}

double TleElements::periodMinutes() const {
  if (!(mean_motion_rev_per_day > 0.0)) {
    return 0.0;
  }
  return kMinutesPerDay / mean_motion_rev_per_day;
}

bool TleElements::isDeepSpace() const {
  const double period = periodMinutes();
  return period > 0.0 && period >= kDeepSpacePeriodMin;
}

bool TleElements::epochTai(const time::LeapSecondTable& leap, time::Tai& out) const {
  if (!(epoch_day >= 1.0) || !(epoch_day < 367.0) || !std::isfinite(epoch_day)) {
    return false;
  }
  // Day-of-year is 1-based: 1.0 is January 1 at 00:00 UTC.
  const double day_of_year = epoch_day;
  const auto whole_day = static_cast<std::int64_t>(std::floor(day_of_year));
  const double fraction_of_day = day_of_year - static_cast<double>(whole_day);

  const std::int64_t days_to_jan1 = time::daysFromCivil(epoch_year, 1, 1);
  const time::CivilDate civil = time::civilFromDays(days_to_jan1 + (whole_day - 1));

  time::UtcDateTime utc;
  utc.year = civil.year;
  utc.month = civil.month;
  utc.day = civil.day;

  // Split the day fraction into h/m/s/ns without ever forming a seconds count
  // larger than a day, so the rounding stays at nanosecond scale.
  const double seconds_of_day = fraction_of_day * 86400.0;
  auto whole_seconds = static_cast<std::int64_t>(std::floor(seconds_of_day));
  double sub_second = seconds_of_day - static_cast<double>(whole_seconds);
  auto nanosecond = static_cast<std::int64_t>(std::llround(sub_second * 1.0e9));
  if (nanosecond >= 1000000000LL) {
    nanosecond -= 1000000000LL;
    whole_seconds += 1;
  }
  if (whole_seconds >= 86400LL) {
    // A fraction that rounds up into the next day: carry the date rather than
    // emitting an hour of 24, which `isValidUtc` would (correctly) refuse.
    whole_seconds -= 86400LL;
    const time::CivilDate next = time::civilFromDays(days_to_jan1 + whole_day);
    utc.year = next.year;
    utc.month = next.month;
    utc.day = next.day;
  }
  utc.hour = static_cast<unsigned>(whole_seconds / 3600);
  utc.minute = static_cast<unsigned>((whole_seconds % 3600) / 60);
  utc.second = static_cast<unsigned>(whole_seconds % 60);
  utc.nanosecond = static_cast<std::int32_t>(nanosecond);

  if (!time::isValidUtc(utc)) {
    return false;
  }
  out = time::taiFromUtc(utc, leap);
  return true;
}

TleStatus parseTle(std::string_view line1, std::string_view line2, TleElements& out,
                   TleChecksumPolicy checksum) {
  const std::string_view l1 = rtrim(line1);
  const std::string_view l2 = rtrim(line2);
  if (l1.size() < kLineLength || l2.size() < kLineLength) {
    return TleStatus::kLineTooShort;
  }
  if (l1[0] != '1' || l1[1] != ' ' || l2[0] != '2' || l2[1] != ' ') {
    return TleStatus::kBadLineNumber;
  }
  if (checksum == TleChecksumPolicy::kVerify) {
    if (!isDigit(l1[kChecksumColumn]) || !isDigit(l2[kChecksumColumn])) {
      return TleStatus::kMalformedField;
    }
    if ((l1[kChecksumColumn] - '0') != tleChecksum(l1) ||
        (l2[kChecksumColumn] - '0') != tleChecksum(l2)) {
      return TleStatus::kBadChecksum;
    }
  }

  TleElements e;
  bool any = false;

  std::int32_t sat1 = 0;
  std::int32_t sat2 = 0;
  if (!parseInt(l1.substr(2, 5), sat1, any) || !any) {
    return TleStatus::kMalformedField;
  }
  if (!parseInt(l2.substr(2, 5), sat2, any) || !any) {
    return TleStatus::kMalformedField;
  }
  if (sat1 != sat2) {
    return TleStatus::kSatelliteMismatch;
  }
  e.satellite_number = sat1;
  e.classification = l1[7];

  std::int32_t two_digit_year = 0;
  if (!parseInt(l1.substr(18, 2), two_digit_year, any) || !any) {
    return TleStatus::kMalformedField;
  }
  // Space-Track's window. See the header: this is the format's cliff, not ours.
  e.epoch_year = (two_digit_year < 57) ? (2000 + two_digit_year) : (1900 + two_digit_year);

  if (!parseDouble(l1.substr(20, 12), e.epoch_day, any) || !any) {
    return TleStatus::kMalformedField;
  }
  if (!parseDouble(l1.substr(33, 10), e.ndot, any)) {
    return TleStatus::kMalformedField;
  }
  if (!parseAssumedExponent(l1.substr(44, 8), e.nddot)) {
    return TleStatus::kMalformedField;
  }
  if (!parseAssumedExponent(l1.substr(53, 8), e.bstar)) {
    return TleStatus::kMalformedField;
  }
  if (!parseInt(l1.substr(64, 4), e.element_number, any)) {
    return TleStatus::kMalformedField;
  }

  if (!parseDouble(l2.substr(8, 8), e.inclination_deg, any) || !any) {
    return TleStatus::kMalformedField;
  }
  if (!parseDouble(l2.substr(17, 8), e.raan_deg, any) || !any) {
    return TleStatus::kMalformedField;
  }
  if (!parseAssumedDecimal(l2.substr(26, 7), e.eccentricity)) {
    return TleStatus::kMalformedField;
  }
  if (!parseDouble(l2.substr(34, 8), e.arg_perigee_deg, any) || !any) {
    return TleStatus::kMalformedField;
  }
  if (!parseDouble(l2.substr(43, 8), e.mean_anomaly_deg, any) || !any) {
    return TleStatus::kMalformedField;
  }
  if (!parseDouble(l2.substr(52, 11), e.mean_motion_rev_per_day, any) || !any) {
    return TleStatus::kMalformedField;
  }
  if (!parseInt(l2.substr(63, 5), e.revolution_number, any)) {
    return TleStatus::kMalformedField;
  }

  // Range gate. The checksum catches a mistyped digit only by luck, so the
  // physically impossible values are refused here: a negative or >=1
  // eccentricity is not an orbit, a non-positive mean motion has no period, and
  // an inclination outside [0, 180] is a transcription error rather than a
  // retrograde orbit (which is expressed as i > 90, not as i < 0).
  if (!(e.eccentricity >= 0.0) || !(e.eccentricity < 1.0)) {
    return TleStatus::kOutOfRange;
  }
  if (!(e.mean_motion_rev_per_day > 0.0)) {
    return TleStatus::kOutOfRange;
  }
  if (!(e.inclination_deg >= 0.0) || !(e.inclination_deg <= 180.0)) {
    return TleStatus::kOutOfRange;
  }
  if (!std::isfinite(e.bstar) || !std::isfinite(e.ndot) || !std::isfinite(e.nddot)) {
    return TleStatus::kOutOfRange;
  }

  out = e;
  return TleStatus::kOk;
}

}  // namespace polaris::gnc
