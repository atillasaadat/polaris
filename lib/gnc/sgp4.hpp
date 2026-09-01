#ifndef POLARIS_GNC_SGP4_HPP
#define POLARIS_GNC_SGP4_HPP

/// @file
/// @brief SGP4/SDP4 analytical propagation of a NORAD element set (design doc
/// §8.3, REQ-ODP-003) [vallado2006revisiting], [vallado2013].
///
/// ## What this is, and what it is not
///
/// SGP4 is not a force model and this is not a propagator in the sense that
/// @ref polaris::gnc::OrbitOd is. It is an **analytical theory paired with a
/// specific fitting process**: a TLE's mean elements are whatever values make
/// *this* theory reproduce the observations, so the elements and the propagator
/// are a matched pair and neither is meaningful alone. Feeding TLE elements to a
/// numerical integrator, or SGP4 to elements fitted some other way, produces a
/// confident wrong answer — which is the single most common misuse of the
/// format and the reason the element set carries its own type
/// (@ref polaris::gnc::TleElements) rather than being converted to a state at
/// parse time.
///
/// Consequently the accuracy is what the theory gives: order **1 km at epoch**
/// and degrading by roughly 1–3 km/day, with no covariance. It is used here for
/// **secondary objects** (REQ-ODP-002 — conjunction screening, ground-station
/// and payload geometry against catalogued objects) and for ephemeris interop,
/// never for the vehicle's own navigation, which is the orbit filter's job.
///
/// ## The output frame is TEME, and that is load-bearing
///
/// SGP4 works in True Equator, Mean Equinox, which is **not** J2000/GCRF. The
/// difference is precession, nutation and the equation of the equinoxes — tens
/// of metres within a year of the TLE epoch and order 100 km after decades —
/// close enough that using one for the other yields a plausible answer rather
/// than an obviously broken one. So the result is tagged
/// @ref polaris::math::frames::TEME and reaches the rest of the system only
/// through an explicit conversion (Golden Rule 4 makes the omission a compile
/// error rather than a silent bias).
///
/// ## WGS-72, not WGS-84
///
/// The gravitational constants are **WGS-72**, and this is not a stale default
/// to be modernised: the TLEs are *fitted* with WGS-72, so evaluating them with
/// WGS-84 introduces an inconsistency the theory has no way to absorb. The
/// constants are therefore fixed here rather than taken from
/// `constants/constants.hpp`, which carries the WGS-84 set the rest of the
/// vehicle uses. A future reader deleting this "duplication" would break every
/// TLE propagation by a few kilometres.
///
/// ## Verification
///
/// The acceptance test is the official AIAA 2006-6753 set — 33 cases chosen to
/// exercise every branch (near-Earth, deep-space resonance, Lyddane choice,
/// decaying and negative-perigee orbits), committed verbatim under
/// `tests/golden/SGP4-VER.TLE` with the reference state vectors in
/// `tests/golden/tforverf.out`. For an algorithm defined by its reference
/// implementation, matching that set is not evidence of correctness by
/// analogy — it *is* the definition. The tolerance is a printing-precision
/// floor, not an engineering allowance: the Fortran and MATLAB references agree
/// with each other to 7e-8 km over the same points.
///
/// ## Flight standard
///
/// No heap, no exceptions, fixed-size storage, every failure a named status
/// (§3.6). @ref Sgp4::propagate is `const` and **order-independent**: the
/// deep-space resonance integration restarts from epoch on every call rather
/// than caching its position the way the reference implementation does, so the
/// state returned for a given time never depends on what was asked before it.
/// That costs a bounded number of 720-second steps and buys a propagator that
/// cannot produce two different answers for one epoch.

#include <cstdint>

#include "gnc/tle.hpp"
#include "math/frames.hpp"
#include "math/typed_vector.hpp"

namespace polaris::gnc {

/// Which historical convention to reproduce.
///
/// The two differ in the Greenwich sidereal time at epoch and in how the
/// deep-space resonance integrator picks its steps. They are both "correct" —
/// they reproduce different operational software — and the choice must match
/// whoever fitted the TLE, which for Space-Track's public catalogue is AFSPC.
enum class Sgp4OpsMode : std::uint8_t {
  kAfspc = 0,  ///< AFSPC-compatible ('a'): what Space-Track's own elements assume
  kImproved,   ///< Vallado's improved ('i'): better continuity, different by ~metres
};

/// Why a propagation was refused, mirroring the reference's error codes so a
/// comparison against published behaviour is possible.
enum class Sgp4Status : std::uint8_t {
  kOk = 0,
  kNotInitialised,         ///< propagate() before a successful initialise()
  kBadElements,            ///< the element set itself is not usable
  kMeanElementsDiverged,   ///< eccentricity left [0, 1) during propagation (code 1)
  kMeanMotionNegative,     ///< mean motion went non-positive (code 2)
  kPerturbedEccentricity,  ///< perturbed eccentricity left [0, 1) (code 3)
  kSemiLatusRectum,        ///< semi-latus rectum went negative (code 4)
  kDecayed,                ///< the satellite has decayed below the Earth's surface (code 6)
};

/// Human-readable name of @p status, for events and telemetry. Never null.
const char* toString(Sgp4Status status);

/// Internal element storage. Public only because it is the propagator's state;
/// callers construct it through @ref Sgp4::initialise and never populate it by
/// hand. Field names follow the reference so the implementation reads against
/// the published algorithm rather than against a private renaming.
struct Sgp4Elements {
  // --- Copied and unit-converted from the TLE ------------------------------
  double bstar{0.0};
  double ecco{0.0};
  double argpo{0.0};     ///< [rad]
  double inclo{0.0};     ///< [rad]
  double mo{0.0};        ///< [rad]
  double no_kozai{0.0};  ///< [rad/min], the TLE's mean motion (Kozai)
  double nodeo{0.0};     ///< [rad]

  // --- Derived at initialisation ------------------------------------------
  double no_unkozai{0.0};  ///< [rad/min], Brouwer mean motion
  double a{0.0};
  double alta{0.0};
  double altp{0.0};
  double gsto{0.0};

  double aycof{0.0}, con41{0.0}, cc1{0.0}, cc4{0.0}, cc5{0.0};
  double d2{0.0}, d3{0.0}, d4{0.0}, delmo{0.0};
  double eta{0.0}, argpdot{0.0}, omgcof{0.0}, sinmao{0.0};
  double t2cof{0.0}, t3cof{0.0}, t4cof{0.0}, t5cof{0.0};
  double x1mth2{0.0}, x7thm1{0.0}, mdot{0.0}, nodedot{0.0}, xlcof{0.0}, xmcof{0.0};
  double nodecf{0.0};

  // --- Deep-space (SDP4) ---------------------------------------------------
  double d2201{0.0}, d2211{0.0}, d3210{0.0}, d3222{0.0}, d4410{0.0}, d4422{0.0};
  double d5220{0.0}, d5232{0.0}, d5421{0.0}, d5433{0.0};
  double dedt{0.0}, del1{0.0}, del2{0.0}, del3{0.0}, didt{0.0}, dmdt{0.0};
  double dnodt{0.0}, domdt{0.0};
  double e3{0.0}, ee2{0.0}, peo{0.0}, pgho{0.0}, pho{0.0}, pinco{0.0}, plo{0.0};
  double se2{0.0}, se3{0.0}, sgh2{0.0}, sgh3{0.0}, sgh4{0.0};
  double sh2{0.0}, sh3{0.0}, si2{0.0}, si3{0.0}, sl2{0.0}, sl3{0.0}, sl4{0.0};
  double gsto_deep{0.0};
  double xfact{0.0}, xgh2{0.0}, xgh3{0.0}, xgh4{0.0}, xh2{0.0}, xh3{0.0};
  double xi2{0.0}, xi3{0.0}, xl2{0.0}, xl3{0.0}, xl4{0.0};
  double xlamo{0.0}, zmol{0.0}, zmos{0.0};
  double atime{0.0}, xli{0.0}, xni{0.0};

  // --- Flags ---------------------------------------------------------------
  bool is_deep_space{false};
  bool use_simplified_drag{false};  ///< perigee < 220 km: the reduced-drag branch
  bool resonant{false};             ///< any resonance active
  bool synchronous{false};          ///< 24-hour (1:1) resonance
  bool initialised{false};
  Sgp4OpsMode opsmode{Sgp4OpsMode::kAfspc};
};

/// One initialised element set, propagable to any time from its epoch.
class Sgp4 {
 public:
  /// Position [km] and velocity [km/s] in TEME.
  using PositionKm = math::Vec3<math::frames::TEME>;
  using VelocityKmS = math::Vec3<math::frames::TEME>;

  Sgp4() = default;

  /// Initialise from a parsed element set.
  ///
  /// @p mode must match whoever fitted the TLE; see @ref Sgp4OpsMode. Returns
  /// `kBadElements` for an element set the theory cannot start from, leaving
  /// this object uninitialised rather than half-configured.
  Sgp4Status initialise(const TleElements& tle, Sgp4OpsMode mode = Sgp4OpsMode::kAfspc);

  /// Propagate to @p minutes_since_epoch (may be negative) and write TEME
  /// position and velocity.
  ///
  /// Outputs are written only on `kOk`. A non-`kOk` status is not necessarily
  /// permanent — a decayed orbit reports `kDecayed` for times past decay and
  /// propagates normally before it — so the status belongs to the *epoch*
  /// asked for, not to the object.
  Sgp4Status propagate(double minutes_since_epoch, PositionKm& position_km,
                       VelocityKmS& velocity_km_s) const;

  bool isInitialised() const { return elements_.initialised; }

  bool isDeepSpace() const { return elements_.is_deep_space; }

  /// The initialised elements, for tests and diagnostics.
  const Sgp4Elements& elements() const { return elements_; }

 private:
  Sgp4Elements elements_{};
};

}  // namespace polaris::gnc

#endif  // POLARIS_GNC_SGP4_HPP
