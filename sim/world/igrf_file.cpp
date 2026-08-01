#include "world/igrf_file.hpp"

#include "environment/igrf_iaga.hpp"

namespace polaris::sim::world {

bool loadIgrfFile(const std::string& path, double decimal_year, environment::IgrfCoefficients& out,
                  std::string* error) {
  char reason[environment::kIgrfMaxReasonLength] = {};
  const bool ok =
      environment::loadIgrfIaga(path.c_str(), decimal_year, out, reason, sizeof(reason));
  if (!ok && error != nullptr) {
    *error = std::string(reason) + " (" + path + ")";
  }
  return ok;
}

}  // namespace polaris::sim::world
