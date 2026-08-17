#ifndef POLARIS_GNC_MEASUREMENT_POLICY_HPP
#define POLARIS_GNC_MEASUREMENT_POLICY_HPP

/// @file
/// @brief The three-way measurement editing flag shared by the onboard filters
/// (NASA/TP-2018-219822 §9.1; NESC Technical Bulletin 20-03 item d).
///
/// The NESC best practices ask for "a command-able capability to selectively
/// apply a three-way editing flag to each measurement type ... 'accept',
/// 'inhibit', and 'force'": accept lets the residual-edit (NIS) test decide,
/// inhibit rejects the measurement regardless of the test, force ingests it
/// regardless of the test [carpenter2018 §9.1; dennehy2020 item d]. Both the
/// orbit filter (@ref OrbitOd) and the attitude filter (@ref Mekf) take it per
/// measurement type; the components expose it as U8 parameters that mirror
/// this enum value for value.
///
/// One rule this codebase adds: **force overrides the gate only.** A negative
/// or non-finite NIS is the covariance gone indefinite, not a large residual,
/// and no operator flag makes an update against that meaningful — it stays a
/// refusal.

#include <cstdint>

namespace polaris::gnc {

/// How one measurement type is processed (TP §9.1).
enum class MeasurementMode : std::uint8_t {
  kAccept = 0,  ///< applied if the NIS gate accepts it (the flight default)
  kInhibit,     ///< never applied, whatever the gate says
  kForce,       ///< applied even if the gate would reject it (a numeric fault still refuses)
};

}  // namespace polaris::gnc

#endif  // POLARIS_GNC_MEASUREMENT_POLICY_HPP
