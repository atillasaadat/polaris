#include "gnc/target_catalog.hpp"

#include <cmath>

#include "frames/teme_eci.hpp"

namespace polaris::gnc {
namespace {

constexpr double kSecondsPerDay = 86400.0;
constexpr double kSecondsPerMinute = 60.0;
constexpr double kKmToM = 1000.0;

}  // namespace

const char* toString(TargetKind kind) {
  switch (kind) {
    case TargetKind::kTle:
      return "TLE";
    case TargetKind::kStateVector:
      return "STATE_VECTOR";
  }
  return "UNKNOWN";
}

const char* toString(TargetStatus status) {
  switch (status) {
    case TargetStatus::kOk:
      return "OK";
    case TargetStatus::kBadSlot:
      return "BAD_SLOT";
    case TargetStatus::kEmpty:
      return "EMPTY";
    case TargetStatus::kPropagationFailed:
      return "PROPAGATION_FAILED";
    case TargetStatus::kBadElements:
      return "BAD_ELEMENTS";
    case TargetStatus::kBadEpoch:
      return "BAD_EPOCH";
    case TargetStatus::kBadOrbit:
      return "BAD_ORBIT";
  }
  return "UNKNOWN";
}

TargetStatus TargetCatalog::loadTle(int index, std::string_view line1, std::string_view line2,
                                    const time::LeapSecondTable& leap, TleChecksumPolicy checksum) {
  if (!validIndex(index)) {
    return TargetStatus::kBadSlot;
  }
  // Everything that can fail happens on locals first; the slot is written only
  // once all of it has succeeded.
  TleElements elements;
  if (parseTle(line1, line2, elements, checksum) != TleStatus::kOk) {
    return TargetStatus::kBadElements;
  }
  time::Tai epoch;
  if (!elements.epochTai(leap, epoch)) {
    return TargetStatus::kBadEpoch;
  }
  Sgp4 propagator;
  if (propagator.initialise(elements, Sgp4OpsMode::kAfspc) != Sgp4Status::kOk) {
    return TargetStatus::kBadOrbit;
  }

  tle_[index].propagator = propagator;
  tle_[index].epoch = epoch;
  tle_[index].occupied = true;
  return TargetStatus::kOk;
}

TargetStatus TargetCatalog::loadStateVector(int index, const StateVectorSlot& slot) {
  if (!validIndex(index)) {
    return TargetStatus::kBadSlot;
  }
  TargetPropagator propagator;
  if (!propagator.setState(slot)) {
    return TargetStatus::kPropagationFailed;
  }
  state_[index].propagator = propagator;
  state_[index].occupied = true;
  return TargetStatus::kOk;
}

TargetStatus TargetCatalog::clear(TargetKind kind, int index) {
  if (!validIndex(index)) {
    return TargetStatus::kBadSlot;
  }
  if (kind == TargetKind::kTle) {
    tle_[index] = TleEntry{};
  } else {
    state_[index] = StateEntry{};
  }
  return TargetStatus::kOk;
}

bool TargetCatalog::isOccupied(TargetKind kind, int index) const {
  if (!validIndex(index)) {
    return false;
  }
  return kind == TargetKind::kTle ? tle_[index].occupied : state_[index].occupied;
}

int TargetCatalog::occupiedCount(TargetKind kind) const {
  int n = 0;
  for (int i = 0; i < kMaxSlots; ++i) {
    if (isOccupied(kind, i)) {
      ++n;
    }
  }
  return n;
}

void TargetCatalog::setForceModel(const TargetForceModel& model) {
  for (auto& entry : state_) {
    entry.propagator.setForceModel(model);
  }
}

TargetStatus TargetCatalog::positionAt(TargetKind kind, int index, const time::Tai& t,
                                       TargetState& out, const frames::EopValue* eop) const {
  if (!validIndex(index)) {
    return TargetStatus::kBadSlot;
  }
  if (!isOccupied(kind, index)) {
    return TargetStatus::kEmpty;
  }

  TargetState result;
  result.kind = kind;

  if (kind == TargetKind::kTle) {
    const TleEntry& entry = tle_[index];
    const double age_s = (t - entry.epoch).seconds();
    Sgp4::PositionKm r_teme;
    Sgp4::VelocityKmS v_teme;
    const Sgp4Status status = entry.propagator.propagate(age_s / kSecondsPerMinute, r_teme, v_teme);
    // kDecayed is deliberately not accepted. SGP4 still returns a position for a
    // decayed object, but it is a position the theory no longer believes in, and
    // pointing an instrument at it would look exactly like tracking.
    if (status != Sgp4Status::kOk) {
      return TargetStatus::kPropagationFailed;
    }
    // SGP4 works in TEME and nothing else here does — the conversion is not
    // optional and Golden Rule 4 makes skipping it a compile error.
    math::Vec3<math::frames::ECI> r_eci;
    math::Vec3<math::frames::ECI> v_eci;
    if (!frames::eciStateFromTeme(t, r_teme, v_teme, r_eci, v_eci)) {
      return TargetStatus::kPropagationFailed;
    }
    result.position_m = r_eci * kKmToM;
    result.velocity_m_s = v_eci * kKmToM;
    result.age_s = age_s;
    result.sigma_m = kTleSigmaAtEpochM + kTleSigmaGrowthMPerDay * std::fabs(age_s) / kSecondsPerDay;
  } else {
    const StateEntry& entry = state_[index];
    math::Vec3<math::frames::ECI> r;
    math::Vec3<math::frames::ECI> v;
    if (entry.propagator.propagate(t, eop, r, v) != PropagationStatus::kOk) {
      return TargetStatus::kPropagationFailed;
    }
    result.position_m = r;
    result.velocity_m_s = v;
    result.age_s = (t - entry.propagator.state().epoch).seconds();
    result.sigma_m = entry.propagator.sigmaAt(t);
  }

  out = result;
  return TargetStatus::kOk;
}

}  // namespace polaris::gnc
