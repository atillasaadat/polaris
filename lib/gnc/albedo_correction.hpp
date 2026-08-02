#ifndef POLARIS_GNC_ALBEDO_CORRECTION_HPP
#define POLARIS_GNC_ALBEDO_CORRECTION_HPP

/// @file
/// @brief Earth-albedo correction of a measured sun-sensor direction
/// (design doc §8.1, calibration roadmap item (2); supports REQ-ADET-005/006).
///
/// **Why this exists.** After the hard/soft-iron calibration of
/// `gnc/mag_calibration.hpp` removed the magnetic systematic, the Monte Carlo
/// campaign measured the remaining tail of both the coarse chain and the MEKF to
/// be the **sun** budget alone, and that budget is 2.0° of Earth albedo against
/// 0.4° of analytic-ephemeris error. Systematics do not average down, so that
/// term is the floor of every attitude solution the vehicle can form without a
/// star tracker. This is what removes most of it.
///
/// **Why it is correctable at all.** Reflected Earthshine is not noise. It
/// arrives from the sunlit ground inside the sensor's field of view, so it drags
/// the reported sun vector **toward the Earth** by an angle that is a
/// deterministic function of four things the flight software already holds every
/// cycle: the spacecraft position (GNSS), the Sun direction (the onboard
/// ephemeris the estimator already queries for its reference), the sensor
/// boresight (mounting, a parameter), and the vehicle attitude (the previous
/// cycle's estimate). Anything deterministic in known quantities is a *model*,
/// not a calibration — which is why §8.1 sequences this independently of the
/// commanded magnetometer fit and runs it always-on rather than on command.
///
/// **The model.** The truth-side model this inverts is `sim/sensors/sun_sensor`
/// (§6.2), a *centroid* model: Earthshine is treated as arriving from the centre
/// of the visible Earth disk \f$\hat d\f$ (nadir), with a peak angular pull
/// \f[
///   A = \sigma_a^{\max}\;\Phi\;\eta,
/// \f]
/// where \f$\sigma_a^{\max}\f$ is the part's uncorrected peak error
/// (@ref AlbedoCorrectionConfig::albedo_error_rad, from the datasheet),
/// \f[
///   \Phi = \mathrm{fovCoveredFraction}\big(\text{FOV},\;
///   \angle(\hat b, \hat d),\; \rho_\oplus\big), \qquad
///   \rho_\oplus = \arcsin(R_\oplus / r)
/// \f]
/// is how much of the field the Earth fills, and \f$\eta = \max(0,\hat r\cdot
/// \hat s)\f$ is how sunlit the ground below is (zero on the night side). The
/// applied pull angle is
/// \f[
///   \varphi = A \, \sin\!\big(\angle(\hat s_{\mathrm{meas}}, \hat d)\big),
/// \f]
/// defined on the **measured** separation — which is precisely why the inverse
/// here is a closed form rather than an iteration: flight software only ever
/// holds the measured vector. The correction rotates it *away* from the Earth by
/// \f$\varphi\f$ about \f$\hat a = (\hat s_{\mathrm{meas}} \times \hat d) /
/// \|\cdot\|\f$:
/// \f[
///   \hat s_{\mathrm{corr}} = R(\hat a, -\varphi)\,\hat s_{\mathrm{meas}} .
/// \f]
/// With the truth-side dispersion at zero this recovers the true direction to
/// round-off; with it at its flight value the residual is the dispersion, which
/// is what the vehicle's post-correction `SigmaSunSysRad` is derived from.
///
/// Both endpoints of \f$\psi\f$ are **doubly covered**, which is why neither
/// needs special handling: at \f$\psi = 0\f$ and at \f$\psi = \pi\f$ the Sun and
/// the Earth's centre are collinear, so \f$\sin\psi\f$ makes the pull vanish
/// *and* \f$\hat s \times \hat d\f$ vanishes, leaving the axis undetermined.
/// The axis guard below refuses there rather than forming a 0/0 — a refusal
/// that costs nothing, since the correction it would have applied is zero.
/// \f$\psi = \pi\f$ is not exotic: it is a Sun directly opposite nadir, i.e. the
/// sub-solar point straight below, which a day-side pass flies through.
///
/// **What it does not model, and why that matters.** This is a centroid model,
/// not an Earthshine radiance integral over the visible cap. It does not carry a
/// surface-reflectance map (ocean, cloud and ice differ by more than 5× in
/// albedo), the offset of the *sunlit* centroid toward the sub-solar limb, or
/// the detector's spectral response. Those are the model error, and they are the
/// honest reason the correction is credited with a **fraction** of the albedo
/// term rather than all of it: the vehicle budget keeps
/// `albedo_dispersion_fraction` of the uncorrected value (§19.2). A correction
/// claimed to be exact would be the one way this push could make the vehicle
/// *worse* than not correcting at all.
///
/// **Refusal, never assertion, and never a guess.** @ref albedoCorrection
/// returns false and leaves the measurement untouched whenever the geometry is
/// missing or degenerate — no position fix, a non-finite input, the Sun and the
/// Earth's centre collinear (the pull is then zero anyway), the Earth out of the
/// field, or the night side. The caller must then weight the measurement with
/// the *uncorrected* σ: correcting on stale or invented geometry would inject a
/// bias of the same size as the one being removed, pointed in an arbitrary
/// direction. That is the whole of the graceful-degradation rule, and
/// `AttitudeEstimator` implements it by selecting between two σ parameters.
///
/// **Frames and units.** Every vector is `Vec3<Body>` and dimensionless
/// (directions) except the geocentric radius, in metres. Angles in radians. SI
/// throughout.
///
/// **Sensitivity to the caller's attitude error, and why it is not small.** The
/// caller rotates nadir into body axes with its own attitude estimate, so an
/// attitude error \f$\varepsilon\f$ misplaces the Earth in the sensor's field.
/// The tempting reading — that this perturbs the *magnitude* \f$\varphi\f$ by a
/// relative \f$\varepsilon\f$ and is therefore negligible — is wrong, and by
/// about a factor of six on the reference part. The dominant term is the
/// rotation **axis**: \f$\hat a = (\hat s \times \hat d)/\|\hat s \times \hat
/// d\|\f$ swings by \f$\varepsilon/\sin\psi\f$, and the resulting error in the
/// correction *vector* is
/// \f[
///   \varphi \cdot \frac{\varepsilon}{\sin\psi}
///   \;=\; A\,\Phi\,\eta\,\sin\psi \cdot \frac{\varepsilon}{\sin\psi}
///   \;\approx\; A\,\varepsilon ,
/// \f]
/// set by the **peak** scale \f$A\f$ (12° for the reference part) and *not* by
/// the applied \f$\varphi\f$ — so it is present at full size even in geometries
/// where the correction itself is small, and the \f$\sin\psi\f$ cancellation is
/// what removes the obvious place one would look for it. Measured at **≈0.1° of
/// sun-vector error per degree of attitude error** (0.22°/deg worst case over
/// the swept geometry) in `albedo_correction_test.cpp`
/// (`AttitudeErrorGainIsBoundedByThePeakScale`).
///
/// That is a *caller's* budget item, not a refusal condition here: the
/// correction still improves the measurement wherever the attitude is known
/// better than about \f$1/A\f$ of the albedo term. `AttitudeEstimator` carries
/// it as a per-cycle inflation of the sun systematic, `A·σ_att/2` in quadrature
/// with the static budget, so a freshly acquired 10° solution is weighted
/// honestly rather than at the converged number. At a converged fine-mode
/// solution (~0.6°) the term is ~1 mrad and invisible.
///
/// **Flight path.** Fixed-size Eigen, no heap, no exceptions, no recursion, no
/// unbounded loops, return codes checked, finiteness checks on the output. No F´
/// types and no I/O — the `AttitudeEstimator` component wraps this.
///
/// References:
///  - Bhanderi & Bak, "Modeling Earth Albedo for Satellites in Earth Orbit",
///    AIAA Guidance, Navigation and Control Conference, AIAA 2005-6465, 2005 —
///    the reflectivity-model treatment this simplifies, and the source of the
///    residual magnitudes that justify a fractional, not total, correction.
///    [bhanderi2005]
///  - Wertz (ed.), *Spacecraft Attitude Determination and Control*, Reidel 1978,
///    §6.1 (sun-sensor error sources; albedo as a systematic, not a noise term).
///    [wertz1978]
///  - Markley & Crassidis, *Fundamentals of Spacecraft Attitude Determination
///    and Control*, Springer 2014, §4.1 (sun sensor models). [markley2014]

#include <Eigen/Core>

#include "math/frames.hpp"
#include "math/typed_vector.hpp"

namespace polaris::gnc {

/// The per-unit constants the correction needs: what the part's uncorrected
/// albedo error peaks at, how wide its field is, and where it points.
///
/// All three are **configuration**, not measurement — the first two come from
/// the unit's hardware-library entry and the third from its mounting — so they
/// arrive through the §19.3 parameter path with no defaults, like every other
/// physical value in flight software.
struct AlbedoCorrectionConfig {
  /// Peak angular error from Earthshine [rad]: the value reached with the Earth
  /// filling the field on a fully sunlit day side, 90° from the Sun. The
  /// datasheet figure (`albedo_error_deg` in the sun-sensor catalog entry).
  double albedo_error_rad = 0.0;

  /// The unit's acceptance half-angle [rad], which sets how much Earth can be in
  /// the field at all.
  double half_fov_rad = 0.0;

  /// Sensor boresight in body axes (unit). The mounting quaternion applied to
  /// the sensor's +Z, matching the sim's `SunSensor::boresightBody`.
  math::Vec3<math::frames::Body> boresight_body{};

  /// True when every value is present, finite and physical. Refused rather than
  /// defaulted: a correction on invented geometry is worse than no correction.
  bool isValid() const;
};

/// One cycle's geometry, all in body axes except the radius.
struct AlbedoCorrectionInput {
  /// The measured sun direction to correct (need not be exactly unit).
  math::Vec3<math::frames::Body> sun_meas{};
  /// Nadir — toward the Earth's centre — in body axes (need not be unit). The
  /// caller forms it from its position fix and its attitude estimate.
  math::Vec3<math::frames::Body> nadir_body{};
  /// Geocentric radius of the spacecraft [m], which sets the Earth's apparent
  /// angular radius. Must exceed the Earth's equatorial radius.
  double radius_m = 0.0;
  /// How sunlit the ground below is: \f$\max(0, \hat r \cdot \hat s)\f$ with both
  /// vectors geocentric, i.e. the cosine of the spacecraft's angle from the
  /// sub-solar point. Zero on the night side, where there is no Earthshine.
  double dayside = 0.0;
};

/// Remove the modelled Earth-albedo pull from @p in.sun_meas.
///
/// @param cfg the unit's albedo constants; an invalid config refuses.
/// @param in this cycle's geometry.
/// @param sun_corrected [out] the corrected unit direction, written only on
///        success. Left untouched on refusal so a caller that ignores the return
///        code cannot silently consume a half-corrected vector.
/// @param applied_rad [out] the pull angle removed [rad], always written (0 on
///        refusal) — telemetered, so the ground can see the correction working
///        rather than infer it.
/// @return true when a correction was computed and applied. **false means the
///         caller must weight the measurement with the uncorrected σ**, not that
///         anything is wrong: no position fix, night side, no Earth in the field
///         and eclipse are all normal, frequent conditions.
bool albedoCorrection(const AlbedoCorrectionConfig& cfg, const AlbedoCorrectionInput& in,
                      math::Vec3<math::frames::Body>& sun_corrected, double& applied_rad);

}  // namespace polaris::gnc

#endif  // POLARIS_GNC_ALBEDO_CORRECTION_HPP
