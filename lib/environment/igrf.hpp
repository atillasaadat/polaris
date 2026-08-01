#ifndef POLARIS_ENVIRONMENT_IGRF_HPP
#define POLARIS_ENVIRONMENT_IGRF_HPP

/// @file
/// @brief IGRF-14 main geomagnetic field (design doc §5.2, §6.2; REQ-SIM-002).
///
/// Evaluates the International Geomagnetic Reference Field, 14th generation, as
/// a Schmidt semi-normalised spherical-harmonic expansion of the internal-source
/// scalar potential to degree 13. This lives in `lib/` rather than `sim/world/`
/// because it is genuinely dual-use: the truth sim needs it for magnetometer
/// truth and residual-dipole torque, and the FSW needs the *same* model onboard
/// as the inertial reference for the measured-vs-modelled magnetometer
/// consistency check and for coarse (sun + mag) attitude in Safe mode.
///
/// **Flight path — obeys the flight memory/exception rules** (root `CLAUDE.md`
/// Golden Rule 6): the coefficient set is a fixed-size POD, evaluation allocates
/// nothing, throws nothing, and uses only fixed-size stack scratch. Reading the
/// IAGA coefficient text file is deliberately *not* here — that is ground/sim
/// side (`sim/world/igrf_file.hpp`). Onboard, an `IgrfCoefficients` arrives the
/// same way a Chebyshev ephemeris segment does: as an upload, not as a file.
///
/// The potential (Alken et al. 2021, Eq. 1; Langel 1987 §4) is
///
///   V(r,θ,φ) = a Σ_{n=1..N} (a/r)^{n+1} Σ_{m=0..n}
///                [g_n^m cos(mφ) + h_n^m sin(mφ)] P_n^m(cos θ),
///
/// with **a = 6371.2 km**, the IGRF reference radius — a geomagnetic convention,
/// *not* the WGS84 semi-major axis. Substituting the WGS84 value is a silent
/// ~0.2 % field-magnitude error, so it is defined here rather than pulled from
/// the WGS84 constants block. `P_n^m` are Schmidt semi-normalised associated
/// Legendre functions, the normalisation the published IGRF coefficients assume.
///
/// The field is B = −∇V, in geocentric spherical components:
///
///   B_r     =  Σ (n+1)(a/r)^{n+2} Σ [g cos(mφ) + h sin(mφ)] P_n^m
///   B_θ     = −Σ     (a/r)^{n+2} Σ [g cos(mφ) + h sin(mφ)] dP_n^m/dθ
///   B_φ     =  Σ     (a/r)^{n+2} Σ m[g sin(mφ) − h cos(mφ)] P_n^m / sin θ
///
/// Note B_θ points toward increasing colatitude (i.e. geographic **south**) and
/// B_r is radially **outward**, so the familiar north/east/down triad is
/// (−B_θ, B_φ, −B_r). `field()` returns Cartesian ECEF instead, which is what
/// the sim and the magnetometer model actually consume.
///
/// Coefficients are published in nT and stored that way (the tables are read
/// straight from the IAGA file); every public accessor returns **Tesla**, since
/// SI is the repo-wide rule.
///
/// Time dependence is the official IGRF linear model, g(t) = g(t₀) + (t−t₀)·ġ,
/// evaluated per call. IGRF publishes coefficients on a 5-year grid with a
/// secular-variation set for the final interval; the ground-side loader collapses
/// whichever pair brackets the epoch of interest into one (t₀, ġ) snapshot, so
/// this evaluator only ever handles the linear case. Secular variation is
/// published to degree 8 only — terms above that have ġ = 0 and are simply held
/// constant.
///
/// References:
///  - Alken et al., *International Geomagnetic Reference Field: the thirteenth
///    generation*, Earth Planets Space 73:49, 2021. [alken2021]
///  - Langel, *The Main Field*, in Geomagnetism Vol. 1, Academic Press, 1987
///    (the Legendre recursion used here). [langel1987]
///  - Winch et al., *Geomagnetism and Schmidt quasi-normalization*, Geophys. J.
///    Int. 160, 2005 (normalisation convention). [winch2005]

#include "math/frames.hpp"
#include "math/typed_vector.hpp"

namespace polaris::environment {

/// Maximum spherical-harmonic degree of the IGRF main field.
inline constexpr int kIgrfMaxDegree = 13;

/// Maximum degree at which IGRF publishes secular variation.
inline constexpr int kIgrfMaxSvDegree = 8;

/// IGRF geomagnetic reference radius [m]. A convention of the model, fixed at
/// 6371.2 km; deliberately not the WGS84 semi-major axis (see file header).
inline constexpr double kIgrfReferenceRadius = 6371200.0;

/// Schmidt semi-normalised Gauss coefficients at one base epoch, plus their
/// secular variation. Fixed-size POD so flight can hold one by value.
///
/// Indexing is `[n][m]` with `1 <= n <= degree` and `0 <= m <= n`; every other
/// slot is zero and ignored. `h_n^0` is identically zero by definition (sin(0·φ)
/// vanishes), so row `m = 0` of @ref h is unused.
struct IgrfCoefficients {
  /// Decimal year the @ref g / @ref h tables are valid at.
  double epoch_year{0.0};
  /// Decimal year past which this snapshot's linear model stops being the
  /// published one: the next tabulated epoch when the snapshot came from inside
  /// the IAGA grid, or the last epoch + 5 years when it came from the
  /// secular-variation column. Beyond it the extrapolation is unvalidated and
  /// its error is invisible to a consumer, so onboard users refuse rather than
  /// evaluate (see `flight.AttitudeEstimator`). Zero when unset.
  double valid_until_year{0.0};
  /// Highest degree populated in @ref g / @ref h.
  int degree{0};
  /// Highest degree populated in @ref g_sv / @ref h_sv.
  int sv_degree{0};
  double g[kIgrfMaxDegree + 1][kIgrfMaxDegree + 1]{};     ///< [nT]
  double h[kIgrfMaxDegree + 1][kIgrfMaxDegree + 1]{};     ///< [nT]
  double g_sv[kIgrfMaxDegree + 1][kIgrfMaxDegree + 1]{};  ///< [nT/yr]
  double h_sv[kIgrfMaxDegree + 1][kIgrfMaxDegree + 1]{};  ///< [nT/yr]
};

/// True if @p c is structurally usable: degrees in range and a finite epoch.
/// Does not judge whether the coefficients are physically sensible.
bool validIgrfCoefficients(const IgrfCoefficients& c);

/// IGRF main-field evaluator.
///
/// **Model.** The internal-source scalar potential is a Schmidt semi-normalized
/// spherical-harmonic expansion to degree \f$N=13\f$ (Alken et al. 2021 Eq. 1;
/// Langel 1987) [alken2021; langel1987]:
/// \f[
///   V(r,\theta,\phi) = a\sum_{n=1}^{N}\left(\frac{a}{r}\right)^{\!n+1}
///     \sum_{m=0}^{n}\left[g_n^m\cos m\phi + h_n^m\sin m\phi\right]
///       P_n^m(\cos\theta),
/// \f]
/// with \f$a = 6371.2\f$ km the IGRF reference radius (a geomagnetic convention,
/// not WGS84), \f$\theta\f$ colatitude, \f$\phi\f$ east longitude, and \f$P_n^m\f$
/// the Schmidt semi-normalized associated Legendre functions the published Gauss
/// coefficients assume [winch2005]. The field is \f$\mathbf B = -\nabla V\f$, in
/// geocentric spherical components:
/// \f[
///   B_r = \sum_n (n+1)\left(\tfrac{a}{r}\right)^{\!n+2}
///           \sum_m \left[g_n^m\cos m\phi + h_n^m\sin m\phi\right]P_n^m,
/// \f]
/// \f[
///   B_\theta = -\sum_n \left(\tfrac{a}{r}\right)^{\!n+2}
///           \sum_m \left[g_n^m\cos m\phi + h_n^m\sin m\phi\right]
///             \frac{\partial P_n^m}{\partial\theta},
/// \f]
/// \f[
///   B_\phi = \frac{1}{\sin\theta}\sum_n \left(\tfrac{a}{r}\right)^{\!n+2}
///           \sum_m m\left[g_n^m\sin m\phi - h_n^m\cos m\phi\right]P_n^m,
/// \f]
/// where \f$B_\theta\f$ points toward increasing colatitude (geographic south) and
/// \f$B_r\f$ radially outward. Time dependence is the official linear model,
/// \f$g_n^m(t) = g_n^m(t_0) + (t - t_0)\,\dot g_n^m\f$, with secular variation
/// \f$\dot g_n^m\f$ published (and applied) only to degree 8. `field()` returns
/// the equivalent Cartesian ECEF vector; `fieldSpherical()` returns the
/// components above.
///
/// Holds its coefficient set by value (≈6 kB), so an instance is
/// self-contained and safe to keep as a component member. Copyable, no heap.
class IgrfField {
 public:
  /// Construct from a coefficient set. If @p coefficients fails
  /// @ref validIgrfCoefficients the instance is inert: @ref good() is false and
  /// every evaluation returns false. Constructing cannot fail loudly because
  /// flight has no exceptions (Golden Rule 6).
  explicit IgrfField(const IgrfCoefficients& coefficients);

  /// False if the coefficient set was rejected at construction.
  bool good() const { return good_; }

  const IgrfCoefficients& coefficients() const { return coefficients_; }

  /// Field in geocentric spherical components at radius @p radius_m,
  /// colatitude @p colatitude_rad (0 at the north pole, π at the south) and
  /// east longitude @p longitude_rad, for decimal year @p decimal_year.
  ///
  /// @param radius_m        Geocentric radius [m].
  /// @param colatitude_rad  Colatitude [rad], 0 at the north pole.
  /// @param longitude_rad   East longitude [rad].
  /// @param decimal_year    Epoch, e.g. 2026.5.
  /// @param b_r     Radially outward component [T].
  /// @param b_theta Southward (increasing-colatitude) component [T].
  /// @param b_phi   Eastward component [T].
  /// @return false — leaving the outputs untouched — if the instance is inert,
  ///         any input is non-finite, or @p radius_m is not positive. A
  ///         non-finite field is far more corrosive to a propagator than a
  ///         refusal, so the caller is made to notice (§3.6).
  ///
  /// At the geographic poles sin θ vanishes and the B_φ sum is evaluated in the
  /// limit: the m = 1 terms tend to a finite value and all m ≥ 2 terms vanish,
  /// so the field stays finite there rather than dividing by zero.
  bool fieldSpherical(double radius_m, double colatitude_rad, double longitude_rad,
                      double decimal_year, double& b_r, double& b_theta, double& b_phi) const;

  /// Field as a Cartesian ECEF vector [T] at ECEF position @p r_ecef.
  ///
  /// ECEF in, ECEF out: the harmonic expansion is defined in the Earth-fixed
  /// frame, so no frame reduction happens here. Converting to ECI is the
  /// caller's job and needs EOP — see `sim/world/magnetic_field.hpp`, which does
  /// it through the same `EciToEcef` resolver the gravity field uses.
  ///
  /// @return false, leaving @p out untouched, under the same conditions as
  ///         @ref fieldSpherical.
  bool field(const math::Vec3<math::frames::ECEF>& r_ecef, double decimal_year,
             math::Vec3<math::frames::ECEF>& out) const;

 private:
  IgrfCoefficients coefficients_;
  bool good_;
};

}  // namespace polaris::environment

#endif  // POLARIS_ENVIRONMENT_IGRF_HPP
