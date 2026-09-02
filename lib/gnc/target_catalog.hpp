#ifndef POLARIS_GNC_TARGET_CATALOG_HPP
#define POLARIS_GNC_TARGET_CATALOG_HPP

/// @file
/// @brief The onboard catalogue of trackable objects (design doc §8.3, §8.4;
/// REQ-ODP-002, REQ-AGN-004).
///
/// Five **TLE** slots and five **state-vector** slots, and one query that
/// answers "where is target *k* at time *t*, in ECI" without the caller knowing
/// which kind it is. That last property is the point of the file: a pointing
/// mode should command *a target*, not a propagator, so that changing how an
/// object's position is known never changes the guidance that consumes it.
///
/// ## Why two kinds and not one
///
/// They carry genuinely different information and neither substitutes for the
/// other:
///
///  - A **TLE** is what a satellite catalogue distributes. It is a set of *mean*
///    elements that are only meaningful under the theory they were fitted with,
///    so it is propagated by `gnc::Sgp4` in TEME and converted through
///    `frames::eciFromTeme` (REQ-CONV-002). Accuracy ~1 km at epoch, degrading
///    by a few km per day. It needs no ground segment: an operator can uplink
///    two lines copied from a public catalogue.
///  - A **state vector** is an osculating Cartesian state from a ground OD
///    solution, an operator upload, or another vehicle's downlink. It is
///    propagated by `gnc::J2Propagator`. It can be far more accurate than a TLE
///    — metres, if the ground solution was good — and it degrades faster,
///    because nothing models the target's drag.
///
/// Collapsing them would mean converting a TLE to a state vector at upload,
/// which throws away the one thing that makes a TLE useful: it stays valid for
/// days without a ground segment.
///
/// ## Fixed capacity, and why five
///
/// `kMaxSlots` is a compile-time bound per kind (§3.6: no heap, fixed-size
/// storage, bounded loops). Five is an operational number rather than a
/// technical one — enough to hold a handful of conjunction or rendezvous
/// candidates and a couple of imaging targets across a pass, and small enough
/// that the whole catalogue fits in telemetry an operator can read in one view.
/// A sixth object replaces a slot by command; there is no eviction policy,
/// because an autonomous eviction is a decision about mission priority the
/// vehicle has no basis to make.
///
/// ## Staleness is reported, never enforced here
///
/// Every answer carries the age of the element set it came from and a 1-sigma
/// position uncertainty. This class refuses nothing on age: how stale is too
/// stale depends on what the target is for — a kilometre of error is nothing
/// for a wide-field camera slew and fatal for a conjunction assessment — so the
/// policy belongs to the consumer, and the *number* belongs here. What it does
/// refuse is a propagation that failed, because a failed propagation has no
/// position at all and the last good one would be indistinguishable from a
/// current one.
///
/// Flight-safe (§3.6): no heap, no exceptions, fixed-size storage, bounded
/// loops. Every mutator is total — it either fully accepts an upload or leaves
/// the slot exactly as it was.

#include <cstdint>

#include "gnc/j2_propagator.hpp"
#include "gnc/sgp4.hpp"
#include "gnc/tle.hpp"
#include "math/frames.hpp"
#include "math/typed_vector.hpp"
#include "time/leap_seconds.hpp"
#include "time/timescales.hpp"

namespace polaris::gnc {

/// Which kind of slot a target lives in.
enum class TargetKind : unsigned char {
  kTle = 0,         ///< mean elements, propagated by SGP4/SDP4 in TEME
  kStateVector = 1  ///< osculating Cartesian state, propagated by J2
};

const char* toString(TargetKind kind);

/// Why a catalogue query produced no position.
enum class TargetStatus : unsigned char {
  kOk = 0,
  kBadSlot,            ///< slot index outside [0, kMaxSlots)
  kEmpty,              ///< nothing has been uploaded to this slot
  kPropagationFailed,  ///< the slot's propagator refused this epoch
  // The three ways an *upload* can fail. They were one value until a SITL row
  // was refused and the event said only "refused": a bad checksum, an
  // impossible date and an element set the theory cannot start from are three
  // different things for an operator to do next, and reporting them
  // identically leaves the only recovery as guessing.
  kBadElements,  ///< the two lines did not parse, or the checksum did not verify
  kBadEpoch,     ///< the epoch fields do not form a real date
  kBadOrbit      ///< parsed, but SGP4 cannot be initialised from it
};

const char* toString(TargetStatus status);

/// One answer from the catalogue.
struct TargetState {
  math::Vec3<math::frames::ECI> position_m;
  math::Vec3<math::frames::ECI> velocity_m_s;
  /// Seconds between the slot's element epoch and the requested time. Signed:
  /// negative means the request is *before* the epoch, which a look-ahead
  /// upload makes routine.
  double age_s = 0.0;
  /// 1-sigma position uncertainty [m], or negative if the slot's propagator
  /// cannot express one.
  double sigma_m = -1.0;
  TargetKind kind = TargetKind::kTle;
};

/// Five TLE slots and five state-vector slots.
class TargetCatalog {
 public:
  static constexpr int kMaxSlots = 5;

  /// The TLE accuracy figure published with every TLE answer [m], grown per day
  /// of age.
  ///
  /// SGP4's own error is ~1 km at epoch and grows by roughly 1-3 km/day for a
  /// typical LEO object. These are the *theory's* numbers, not this
  /// implementation's — Push 80 measured the implementation reproducing its
  /// reference to 1e-7 km, which says nothing about how well the reference
  /// matches reality. Published so a consumer comparing a TLE target against a
  /// state-vector target is comparing like with like.
  static constexpr double kTleSigmaAtEpochM = 1000.0;
  static constexpr double kTleSigmaGrowthMPerDay = 2000.0;

  /// Load a TLE into slot @p index. The element set is parsed and the propagator
  /// initialised before anything is stored, so a rejected upload leaves the
  /// previous target in service — the same double-buffer discipline
  /// `OnboardTables` uses, for the same reason: a half-loaded target is worse
  /// than a stale one.
  ///
  /// @p checksum defaults to verification, since an uplinked TLE with one
  /// flipped character is a plausible orbit somewhere else.
  TargetStatus loadTle(int index, std::string_view line1, std::string_view line2,
                       const time::LeapSecondTable& leap,
                       TleChecksumPolicy checksum = TleChecksumPolicy::kVerify);

  /// Load an osculating state vector into slot @p index. Same all-or-nothing
  /// discipline as @ref loadTle.
  TargetStatus loadStateVector(int index, const StateVectorSlot& slot);

  /// Empty a slot. Emptying an already-empty slot succeeds: clearing is
  /// idempotent so that a recovery sequence need not first ask what is there.
  TargetStatus clear(TargetKind kind, int index);

  bool isOccupied(TargetKind kind, int index) const;

  /// Where target (@p kind, @p index) is at @p t, in ECI.
  ///
  /// The one call a guidance mode makes. Deliberately does **not** take a
  /// staleness limit: see the header.
  TargetStatus positionAt(TargetKind kind, int index, const time::Tai& t, TargetState& out) const;

  /// How many slots of @p kind are occupied.
  int occupiedCount(TargetKind kind) const;

 private:
  struct TleEntry {
    Sgp4 propagator;
    time::Tai epoch;
    bool occupied = false;
  };

  struct StateEntry {
    J2Propagator propagator;
    bool occupied = false;
  };

  static bool validIndex(int index) { return index >= 0 && index < kMaxSlots; }

  TleEntry tle_[kMaxSlots];
  StateEntry state_[kMaxSlots];
};

}  // namespace polaris::gnc

#endif  // POLARIS_GNC_TARGET_CATALOG_HPP
