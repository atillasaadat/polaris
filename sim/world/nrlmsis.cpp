#include "world/nrlmsis.hpp"

#include <cmath>
#include <cstdint>
#include <fstream>

#include "time/civil.hpp"
#include "time/utc.hpp"

#ifndef POLARIS_MSIS_PARM_PATH
#define POLARIS_MSIS_PARM_PATH ""
#endif

extern "C" {
void polaris_msis_init(const char* path, int path_len, int* status);
void polaris_msis_density(double day, double utsec, double alt_km, double lat_deg, double lon_deg,
                          double f107a, double f107, const double* ap, double* density);
}

namespace polaris::sim::world {
namespace {

constexpr double kRadToDeg = 57.295'779'513'082'320'876'8;
constexpr double kSecondsPerDay = 86'400.0;

/// Serializes every call into the model. NRLMSIS holds its evaluation cache in
/// Fortran module `save` storage shared by the whole process, so this must be
/// process-wide: a per-object mutex would leave two instances free to corrupt
/// each other's cached state while each held its own uncontended lock.
std::mutex& msisMutex() {
  static std::mutex m;
  return m;
}

/// The coefficient file currently loaded into the model's global arrays, or
/// empty if none. Guarded by `msisMutex()`.
std::string& loadedParmPath() {
  static std::string path;
  return path;
}

}  // namespace

std::string NrlmsisAtmosphere::defaultParmPath() {
  return std::string(POLARIS_MSIS_PARM_PATH);
}

NrlmsisAtmosphere::NrlmsisAtmosphere(const std::string& parm_path) {
  if (parm_path.empty()) {
    error_ = "NRLMSIS parameter file path is empty (POLARIS_MSIS_PARM_PATH unset)";
    return;
  }
  // msisinit halts the process on an unreadable parameter file, which is not an
  // acceptable failure mode mid-simulation. Check readability here so a bad path
  // degrades to zero density instead of killing the run.
  {
    std::ifstream probe(parm_path, std::ios::binary);
    if (!probe.good()) {
      error_ = "cannot read NRLMSIS parameter file: " + parm_path;
      return;
    }
  }

  const std::lock_guard<std::mutex> lock(msisMutex());

  // msisinit loads into process-global module arrays, so a second parameter set
  // would silently repoint every existing instance. Refuse instead of corrupting
  // the model out from under whoever loaded first.
  std::string& loaded = loadedParmPath();
  if (!loaded.empty()) {
    if (loaded != parm_path) {
      error_ = "NRLMSIS already initialized with a different coefficient file (" + loaded +
               "); only one may be live per process";
      return;
    }
    initialized_ = true;  // same file, already loaded — share it
    return;
  }

  int status = 0;
  polaris_msis_init(parm_path.c_str(), static_cast<int>(parm_path.size()), &status);
  if (status != 0) {
    error_ = "NRLMSIS initialization failed for: " + parm_path;
    return;
  }
  loaded = parm_path;
  initialized_ = true;
}

double NrlmsisAtmosphere::density(const time::Tai& epoch,
                                  const math::Vec3<math::frames::ECI>& r_eci) const {
  // Every missing piece means "no atmosphere" rather than a guessed one, so a
  // half-wired model gives obviously-zero drag instead of plausibly-wrong drag.
  if (!initialized_ || !eci_to_ecef_ || leap_ == nullptr) {
    return 0.0;
  }

  // MSIS wants a geodetic longitude, so unlike the exponential model this must
  // go through the Earth-rotation reduction: longitude is what sets local solar
  // time, and the diurnal bulge is a first-order effect up here.
  math::Quat<math::frames::ECEF, math::frames::ECI> q;
  if (!eci_to_ecef_(epoch, q)) {
    return 0.0;
  }
  const Geodetic g = geodetic(q.core().rotate(r_eci.eigen()));

  // MSIS is a whole-atmosphere model but is not defined below the ground.
  if (!(g.altitude_m >= 0.0)) {
    return 0.0;
  }

  const time::UtcDateTime utc = time::utcFromTai(epoch, *leap_);
  const std::int64_t day_number = time::daysFromCivil(utc.year, utc.month, utc.day);
  const std::int64_t year_start = time::daysFromCivil(utc.year, 1, 1);
  const double day_of_year = static_cast<double>(day_number - year_start) + 1.0;
  // During a positive leap second (second == 60) this reaches 86400, which the
  // fmod below folds forward to 0.0 — the start of the next UT day rather than
  // the tail of this one. A one-second placement error is orders of magnitude
  // below anything the atmosphere resolves.
  const double utsec = static_cast<double>(utc.hour) * 3600.0 +
                       static_cast<double>(utc.minute) * 60.0 + static_cast<double>(utc.second) +
                       static_cast<double>(utc.nanosecond) * 1e-9;

  double rho = 0.0;
  {
    // msiscalc caches its last evaluation in Fortran `save` variables shared by
    // the whole process, so the lock has to be process-wide, not per-object.
    const std::lock_guard<std::mutex> lock(msisMutex());
    polaris_msis_density(day_of_year, std::fmod(utsec, kSecondsPerDay), g.altitude_m * 1e-3,
                         g.latitude_rad * kRadToDeg, g.longitude_rad * kRadToDeg,
                         space_weather_.f107a, space_weather_.f107, space_weather_.ap.data(), &rho);
  }

  // The shim already maps MSIS's missing-value sentinel to zero; this is the
  // belt-and-braces guard that no non-finite value reaches the integrator.
  if (!std::isfinite(rho) || rho < 0.0) {
    return 0.0;
  }
  return rho;
}

}  // namespace polaris::sim::world
