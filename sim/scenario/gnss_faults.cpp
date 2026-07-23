#include "scenario/gnss_faults.hpp"

namespace polaris::sim::scenario {

GnssFaultState gnssFaultsAt(const std::vector<GnssFaultEvent>& events, const std::string& unit,
                            double t_s) {
  GnssFaultState state;
  for (const GnssFaultEvent& ev : events) {
    if (ev.unit != unit || t_s < ev.start_s || t_s >= ev.stop_s) {
      continue;
    }
    switch (ev.type) {
      case GnssFaultEvent::Type::kOutage:
        state.outage = true;
        break;
      case GnssFaultEvent::Type::kSpoof:
        state.spoof = true;
        state.spoof_offset_ecef_m += ev.spoof_offset_ecef_m;
        break;
      case GnssFaultEvent::Type::kClockJump:
        state.clock_jump_s += ev.clock_jump_s;
        break;
    }
  }
  return state;
}

void applyGnssFaults(sensors::Gnss& receiver, const GnssFaultState& state) {
  receiver.setOutage(state.outage);
  // Reconcile both directions: an ended window resolves to a zero offset / jump,
  // which clears the fault rather than leaving the last value latched.
  receiver.injectPositionOffset(
      math::Vec3<math::frames::ECEF>(state.spoof ? state.spoof_offset_ecef_m
                                                 : Eigen::Vector3d::Zero()));
  receiver.injectClockJump(state.clock_jump_s);
}

}  // namespace polaris::sim::scenario
