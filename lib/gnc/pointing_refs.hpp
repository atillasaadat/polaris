#ifndef POLARIS_GNC_POINTING_REFS_HPP
#define POLARIS_GNC_POINTING_REFS_HPP

/// @file
/// @brief The vocabulary of an align/constrain attitude command (design doc
/// §8.4; REQ-AGN-004, REQ-AGN-005).
///
/// An attitude is three degrees of freedom. Pointing something at something is
/// two. So *every* pointing mode — nadir hold, sun-safe, ground-station track,
/// star stare, satellite track — is the same command with different nouns:
///
///     ALIGN     <body vector>  with  <inertial target>     (2 DOF, exact)
///     CONSTRAIN <body vector>  toward <inertial target>     (1 DOF, best-effort)
///
/// This file is the two noun lists. Keeping them enumerated rather than letting
/// each mode hard-code its own geometry is what makes the set *composable*: a
/// new mission mode is a new pair of nouns from lists that already exist, not
/// new flight code — and each combination is validated by one function rather
/// than by whatever each mode remembered to check.
///
/// ## Both lists carry a sign
///
/// `negate` rather than doubled enumerators, because the negative of a direction
/// is as ordinary as the direction: a radiator points **anti**-sun, a camera
/// often sits on **-Z**, and an antenna points **anti**-nadir at a relay. Twelve
/// more enumerators would say the same thing less clearly and would let
/// `BODY_MINUS_X` and `BODY_X` with `negate` both exist, which is one
/// representation too many.
///
/// ## Why body vectors are named, not just numbers
///
/// A star tracker boresight, a sun-sensor normal and a camera axis are already
/// on the vehicle as *mounting* data — `AttitudeEstimator.StBoresightsBody` and
/// its siblings — so a pointing command that took raw components would carry a
/// second, independently-maintained copy of a number the estimator already
/// holds. That is the flight/sim parameter-pair failure class one level in: the
/// two copies disagree silently and the symptom is a pointing error nobody can
/// attribute. Naming the unit means there is one number.
///
/// `CUSTOM_BODY_VEC` exists for what mounting data cannot cover — an antenna, a
/// thruster axis, a payload aperture added after the parameter set was frozen —
/// and is operator-writable at runtime for exactly that reason.
///
/// Flight-safe (§3.6): plain aggregates, fixed-size storage, no heap.

#include <cstdint>

#include "math/frames.hpp"
#include "math/typed_vector.hpp"

namespace polaris::gnc {

/// Upper bound on sensor units of one kind, matching `GncMaxUnits` in
/// `GncPorts.fpp`. The two are cross-checked by a static_assert at the component
/// boundary rather than merely commented — see `flight/PolarisFsw/GncPorts`.
inline constexpr int kMaxSensorUnits = 8;

/// Operator-writable body vectors.
inline constexpr int kMaxCustomBodyVectors = 10;

/// Target slots of each propagated kind, matching `TargetCatalog::kMaxSlots`.
inline constexpr int kMaxTargetSlots = 5;

/// Stored Earth-fixed points — ground stations, imaging sites, calibration
/// targets.
///
/// Thirty rather than the five a propagated target gets, because these are a
/// different kind of thing: a ground station list is *static mission data* that
/// is uplinked once and used for years, where a propagated target is a
/// short-lived operational choice. Thirty covers a real ground-station network
/// with room for imaging sites, and the table is small enough (30 x 3 doubles)
/// to hold onboard without ceremony.
///
/// Populating it is deliberately separate from using it: the auto-selection
/// logic that decides *which* station to point at, and when, is a later push.
/// This is the storage that makes such a thing possible without another upload
/// path.
inline constexpr int kMaxGroundPoints = 30;

// ---------------------------------------------------------------------------
// The body side: what is being pointed
// ---------------------------------------------------------------------------

enum class BodyVectorKind : std::uint8_t {
  kBodyX = 0,        ///< structural +X
  kBodyY = 1,        ///< structural +Y
  kBodyZ = 2,        ///< structural +Z
  kStarTracker = 3,  ///< boresight of star tracker `index`, from mounting data
  kSunSensor = 4,    ///< normal of sun sensor `index`, from mounting data
  kCamera = 5,       ///< boresight of camera `index`, from mounting data
  kCustom = 6,       ///< operator-stored vector `index`
};

const char* toString(BodyVectorKind kind);

/// A named direction in the body frame.
struct BodyVectorRef {
  BodyVectorKind kind = BodyVectorKind::kBodyZ;
  /// Unit number for the sensor kinds, slot number for `kCustom`, ignored for
  /// the structural axes.
  std::uint8_t index = 0;
  /// Take the opposite direction. See the file header on why this is a flag.
  bool negate = false;

  /// Whether two references name the same physical direction.
  ///
  /// Sign is deliberately **not** part of this: `+X` and `-X` are the same axis,
  /// and aligning one while constraining the other is exactly as unsatisfiable
  /// as naming the same one twice. A comparison that ignored the sign question
  /// would let that command through.
  bool sameAxisAs(const BodyVectorRef& o) const {
    return kind == o.kind && (usesIndex() ? index == o.index : true);
  }

  bool usesIndex() const {
    return kind == BodyVectorKind::kStarTracker || kind == BodyVectorKind::kSunSensor ||
           kind == BodyVectorKind::kCamera || kind == BodyVectorKind::kCustom;
  }
};

// ---------------------------------------------------------------------------
// The inertial side: what it is pointed at
// ---------------------------------------------------------------------------

enum class PointingTargetKind : std::uint8_t {
  kSun = 0,        ///< line of sight to the Sun
  kMoon = 1,       ///< line of sight to the Moon
  kNadir = 2,      ///< geocentric nadir, -r̂
  kEcefPoint = 3,  ///< stored Earth-fixed point `index` (ground station, site)
  kStarJ2000 = 4,  ///< an inertial direction from (right ascension, declination)
  kJ2000X = 5,     ///< the inertial frame's own axes — for a fixed inertial hold
  kJ2000Y = 6,
  kJ2000Z = 7,
  kLvlhX = 8,  ///< LVLH axes (x ~ +velocity, y = -orbit normal, z = nadir)
  kLvlhY = 9,
  kLvlhZ = 10,
  kSatTle = 11,    ///< catalogue TLE slot `index`
  kSatState = 12,  ///< catalogue state-vector slot `index`
};

const char* toString(PointingTargetKind kind);

/// A named direction in inertial space.
struct PointingTargetRef {
  PointingTargetKind kind = PointingTargetKind::kNadir;
  /// Slot number: catalogue slot for `kSatTle` / `kSatState`, ground-point slot
  /// for `kEcefPoint`. Ignored otherwise.
  std::uint8_t index = 0;
  bool negate = false;
  /// Parameters for `kStarJ2000`: {right ascension, declination, unused} [rad].
  ///
  /// Carried inline rather than in a table because a star is named by two
  /// numbers that belong to the command that uses them, and there is no
  /// operational reason to accumulate a catalogue of them onboard. Ground points
  /// are the opposite case — a station list is static mission data reused for
  /// years — which is why those live in @ref GroundPointTable instead.
  double params[3] = {0.0, 0.0, 0.0};

  bool sameTargetAs(const PointingTargetRef& o) const {
    if (kind != o.kind) {
      return false;
    }
    return usesIndex() ? index == o.index : true;
  }

  bool usesIndex() const {
    return kind == PointingTargetKind::kSatTle || kind == PointingTargetKind::kSatState ||
           kind == PointingTargetKind::kEcefPoint;
  }

  bool usesParams() const { return kind == PointingTargetKind::kStarJ2000; }

  /// Upper bound on a valid `index` for this kind, or 0 when it takes none.
  int indexBound() const {
    if (kind == PointingTargetKind::kEcefPoint) {
      return kMaxGroundPoints;
    }
    return usesIndex() ? kMaxTargetSlots : 0;
  }
};

// ---------------------------------------------------------------------------
// Mounting data and operator-stored vectors
// ---------------------------------------------------------------------------

/// The body directions a command may name.
///
/// Sensor entries mirror the estimator's mounting parameters; a slot for a unit
/// that is not installed is left invalid rather than zeroed, so naming it is
/// refused instead of silently resolving to a null vector — the same convention
/// `StBoresightsBody` uses, and for the same reason.
class BodyVectorTable {
 public:
  /// Install a sensor boresight. Rejects a non-finite or null-norm vector and an
  /// out-of-range unit, leaving the slot as it was.
  bool setSensor(BodyVectorKind kind, int index, const math::Vec3<math::frames::Body>& v);

  /// Store an operator-supplied custom vector. Same rejection rules.
  bool setCustom(int index, const math::Vec3<math::frames::Body>& v);

  /// Forget a custom vector, so naming it is refused again.
  bool clearCustom(int index);

  /// The unit direction @p ref names, or false if the slot is empty or the
  /// reference is malformed. The returned vector is normalised and sign-applied.
  bool resolve(const BodyVectorRef& ref, math::Vec3<math::frames::Body>& out) const;

  bool isCustomSet(int index) const;

  int customCount() const;

 private:
  struct Slot {
    math::Vec3<math::frames::Body> v;
    bool valid = false;
  };

  static bool validSensorIndex(int i) { return i >= 0 && i < kMaxSensorUnits; }

  static bool validCustomIndex(int i) { return i >= 0 && i < kMaxCustomBodyVectors; }

  Slot star_tracker_[kMaxSensorUnits];
  Slot sun_sensor_[kMaxSensorUnits];
  Slot camera_[kMaxSensorUnits];
  Slot custom_[kMaxCustomBodyVectors];
};

/// Stored Earth-fixed points, addressed by `PointingTargetRef::index`.
///
/// Geodetic rather than Cartesian ECEF, because that is how a ground station is
/// specified, published and checked by a human — storing the Cartesian form
/// would mean the uplinked number and the number in the mission document were
/// different, which is the flight/sim parameter-pair failure class again. The
/// conversion happens once, at resolution.
class GroundPointTable {
 public:
  struct Point {
    double latitude_rad = 0.0;   ///< geodetic latitude, [-pi/2, pi/2]
    double longitude_rad = 0.0;  ///< east longitude
    double height_m = 0.0;       ///< height above the WGS-84 ellipsoid
    bool valid = false;
  };

  /// Store a point. Rejects a non-finite value, a latitude outside +/-90 deg,
  /// an implausible height, or an out-of-range slot, leaving the slot as it was.
  ///
  /// The height bound is deliberately wide (-1 km to +100 km): it exists to
  /// catch a metres-vs-kilometres unit error on upload, which is the mistake
  /// that actually happens, not to police where a site may be.
  bool set(int index, double latitude_rad, double longitude_rad, double height_m);

  /// Forget a point, so naming it is refused again.
  bool clear(int index);

  bool isSet(int index) const;

  /// The stored point, or false when the slot is empty or out of range.
  bool get(int index, Point& out) const;

  int count() const;

 private:
  static bool validIndex(int i) { return i >= 0 && i < kMaxGroundPoints; }

  Point points_[kMaxGroundPoints];
};

}  // namespace polaris::gnc

#endif  // POLARIS_GNC_POINTING_REFS_HPP
