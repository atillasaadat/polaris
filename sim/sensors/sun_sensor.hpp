#ifndef POLARIS_SIM_SENSORS_SUN_SENSOR_HPP
#define POLARIS_SIM_SENSORS_SUN_SENSOR_HPP

/// @file
/// @brief Sun sensor truth model (design doc §6.2).
///
/// **The output is per-diode counts, not a direction.** That is deliberate and it
/// is the whole architectural point: a real sun sensor is a set of photodiodes,
/// each reporting how much light it sees, and reconstructing a sun vector from
/// those numbers is the *flight software's* job (§8.1 fusion). A truth model that
/// handed back a clean unit vector would quietly do the FSW's work for it, hide
/// the geometry where the interesting failures live — a diode in shadow, a diode
/// saturated, a diode seeing the Earth instead of the Sun — and make the coarse
/// estimator look better than it will be in flight.
///
/// Diode response is the **cosine law**: a photodiode's current is proportional
/// to the projected area it presents to the source, so `count = full_scale ·
/// cos(θ)` for incidence angle θ, falling to zero at grazing incidence and
/// outside the acceptance cone. Sensitivity therefore degrades exactly where the
/// cosine flattens — near the diode normal, where dcount/dθ → 0 — which is why
/// real coarse sensors are flown in canted clusters rather than one diode per
/// face, and why the geometry is configurable here.
///
/// Two arrangements cover almost every COTS part, and both come from the same
/// parameters:
///  - **Single diode** (`diode_count: 1`): a coarse sun sensor, one per face.
///    Several of them on different faces are fused by the FSW.
///  - **Canted cluster** (`diode_count: 4`, `diode_cant_deg`): the quadrant or
///    pyramid arrangement of a fine sun sensor, where the *differences* between
///    opposing diodes give a well-conditioned two-axis angle near boresight.
///
/// **Albedo is the dominant error, and it is a bias, not noise.** Earthshine
/// reaching a diode is a first-order effect for a coarse sensor in LEO — the
/// Earth can fill a wide-FOV diode's view and reflect ~30% of the incident
/// sunlight, which is why coarse sun sensors are routinely quoted at several
/// degrees of accuracy despite millivolt-clean electronics. It is modelled here
/// from the §6.1 occlusion fractions: how much of the diode's field of view the
/// Earth fills, scaled by how sunlit the sub-satellite region is. On the
/// vector-output path it enters as a **directed pull toward the Earth** plus a
/// per-unit dispersion, because that is what it physically is and it is
/// therefore correctable onboard (§8.1); see the class docstring for the model
/// and its limits.
///
/// **Eclipse comes in through the caller** (`SunSensorInput::shadow_factor`),
/// the same injected-resolver pattern the rest of the sim uses: the sensor layer
/// stays free of the world models, and the one conical-shadow implementation in
/// `world/eclipse.hpp` remains the only one.
///
/// References:
///  - Markley & Crassidis, *Fundamentals of Spacecraft Attitude Determination
///    and Control*, 2014, §4.1 (sun sensor models, cosine response, albedo).
///    [markley2014]
///
/// Implements REQ-SIM-003 (sensor truth models with full error stacks, shared
/// occlusion) and REQ-SIM-005 (scriptable fault injection).

#include <cstdint>
#include <Eigen/Core>
#include <map>
#include <string>
#include <vector>

#include "math/frames.hpp"
#include "math/typed_vector.hpp"
#include "random/rng.hpp"
#include "sensors/occlusion.hpp"
#include "time/timescales.hpp"

namespace polaris::sim::sensors {

/// What the part puts on the bus.
///
/// The distinction is a real one about where the processing happens, and it
/// decides what the truth model is even allowed to claim. An analogue part hands
/// over photocurrents and the FSW does the reconstruction, so the model must
/// produce counts. A digital part (GomSpace NanoSense FSS, and most modern fine
/// sensors) runs the quadrant maths and its own factory calibration inside a
/// microcontroller and reports a **vector**; its per-unit calibration tables are
/// proprietary and shipped with the hardware. Simulating photocurrents for such a
/// part would mean inventing the one thing the vendor does not publish, and then
/// undoing it with an algorithm that is not theirs. What the datasheet *does*
/// specify is the accuracy of the delivered vector, so that is what is modelled —
/// the same reasoning that has the star tracker modelled at its attitude output.
enum class SunSensorOutput {
  kDiodeCounts,  ///< analogue: per-diode counts, FSW reconstructs the direction
  kSunVector,    ///< digital: the unit reports a sun direction over a data bus
};

/// One sun sensor's diodes and electronics, in SI (angles in radians, signal in
/// ADC counts — the unit a real part actually reports).
struct SunSensorSpec {
  /// Which of the two output contracts this part implements.
  SunSensorOutput output = SunSensorOutput::kDiodeCounts;

  /// Number of photodiodes. 1 is a coarse single-diode sensor; 4 is the usual
  /// quadrant/pyramid fine sensor.
  int diode_count = 1;
  /// Cant of each diode normal away from the sensor boresight [rad]. Zero points
  /// every diode along the boresight (only meaningful for a single diode); a
  /// cluster spreads its diodes at equal azimuths around it.
  double diode_cant_rad = 0.0;
  /// Acceptance half-angle of one diode [rad]. Beyond it the diode sees nothing.
  double half_fov_rad = 0.0;

  /// Counts at normal incidence in full sunlight (the diode's full scale).
  double full_scale_counts = 0.0;
  /// Reading ceiling — the ADC cannot report more than this however bright.
  double saturation_counts = 0.0;
  /// Dark current / electronic offset, in counts, present with no illumination.
  double dark_counts = 0.0;
  /// Per-sample Gaussian noise, 1σ counts.
  double noise_counts = 0.0;
  /// ADC quantization step [counts]; 0 disables.
  double resolution_counts = 0.0;
  /// Per-diode scale-factor error, 1σ fraction. Fixed per unit at construction.
  double scale_factor = 0.0;
  /// Per-diode normal misalignment, 1σ [rad]. Fixed per unit at construction.
  double alignment_sigma = 0.0;

  /// Peak albedo signal as a fraction of full scale, when the Earth fills the
  /// diode's field of view and the sub-satellite region is fully sunlit. ~0.3
  /// corresponds to the Earth's mean bond albedo.
  double albedo_coefficient = 0.0;

  // --- Vector-output accuracy (kSunVector parts) -----------------------------
  //
  // Datasheets quote sun-sensor accuracy as a function of **incidence angle**,
  // because a quadrant part's angular sensitivity falls off as the spot moves to
  // the edge of its aperture: the GomSpace NanoSense FSS is ±0.5° (3σ) inside 45°
  // and ±2.0° out to its 60° half-FOV — a factor of four across one part's field.
  // Quoting a single number would either flatter the sensor at wide angles or
  // slander it near boresight, and coarse-attitude performance depends on which
  // regime the vehicle actually flies in.

  /// Incidence angle out to which the tighter accuracy applies [rad].
  double accuracy_inner_half_angle_rad = 0.0;
  /// 1σ pointing accuracy inside that angle [rad].
  double accuracy_inner_sigma = 0.0;
  /// 1σ pointing accuracy from there to the edge of the field of view [rad].
  double accuracy_outer_sigma = 0.0;
  /// Peak angular error from Earthshine [rad], reached when the Earth fills the
  /// field of view on the day side **and** stands 90° from the Sun. Vendors are
  /// explicit that this dominates: GomSpace quote errors above 10° uncorrected,
  /// against a 0.5° clean-sky figure. The error is *directed* (toward the Earth),
  /// not random — see the class docstring.
  double albedo_error_rad = 0.0;
  /// **Total** 1σ dispersion of the realised Earthshine about the modelled
  /// value, as a **fraction** of it. Drawn once per unit (§6.2): the departure
  /// from a uniform Lambertian sphere is the surface and cloud field under the
  /// orbit, which changes over minutes rather than between samples. It acts on
  /// two independent axes — a scale error along the pull and an out-of-plane
  /// centroid offset — and this figure is their quadrature sum, so it can be
  /// used as the post-correction budget directly with no hidden √2.
  ///
  /// This is the part of the albedo error that survives the onboard correction
  /// of §8.1, so it — not @ref albedo_error_rad — is what the post-correction
  /// sun budget is derived from. Zero means a perfectly modelled Earth, which no
  /// vehicle flies over.
  double albedo_dispersion_fraction = 0.0;

  /// Minimum interval between genuinely new readings [s] — the part's sample
  /// period. Reading faster returns the previous value again rather than fresh
  /// information, which is what a real bus-attached sensor does.
  double sample_period_s = 0.0;

  /// Native output rate [Hz]; informational (the §2.4 buffer does the gating).
  double update_rate_hz = 0.0;
  /// Shadow factor below which the Sun is treated as absent (umbra). A small
  /// non-zero threshold, because a diode reading a sliver of a penumbral Sun
  /// carries no usable direction information.
  double sun_present_threshold = 0.05;

  /// Build a spec from datasheet-native hardware-library params (the keys used
  /// by `config/hardware/sun_sensor/*.yaml`), converting each to SI. Missing keys
  /// default to 0 (that term disabled). See sun_sensor.cpp for the key list.
  static SunSensorSpec fromParams(const std::map<std::string, double>& params);
};

/// Everything one reading needs from the truth state.
struct SunSensorInput {
  /// True Sun direction in **body** axes (unit). The caller rotates the ECI
  /// sun vector through the truth attitude.
  math::Vec3<math::frames::Body> sun_dir_body{};
  /// Nadir direction in **body** axes (unit) — toward the Earth's centre. Needed
  /// for the albedo view factor: how much Earth a given diode is looking at.
  /// Supplied rather than derived because the caller already holds the attitude,
  /// and rotating it here would mean passing the attitude in just to undo it.
  math::Vec3<math::frames::Body> nadir_dir_body{};
  /// Fraction of the solar disk visible — `world::shadowFactor` (§5.2). 1 in
  /// full sunlight, 0 in umbra, fractional across the penumbra. Injected rather
  /// than recomputed so there is one eclipse model in the repo.
  double shadow_factor = 1.0;
  /// Spacecraft / Sun / Moon geometry, for the albedo view factor.
  SkyGeometry sky{};
};

/// One reading.
struct SunSensorMeasurement {
  /// Per-diode counts, in the spec's diode order. Always `diode_count` long.
  /// Empty of meaning for a `kSunVector` part, which reports no photocurrents.
  std::vector<double> counts;

  /// Measured Sun direction in body axes (unit), for `kSunVector` parts. Zero
  /// when the Sun is absent or the part is analogue.
  math::Vec3<math::frames::Body> sun_dir_body{};
  /// **Truth** incidence angle from the sensor boresight [rad]. Carried for
  /// diagnostics and because it is what selects the accuracy regime — a study of
  /// where the sensor is being operated needs it alongside the error.
  double incidence_angle_rad = 0.0;
  /// The 1σ accuracy this reading was drawn with [rad]: the datasheet figure for
  /// this incidence regime, plus the albedo **dispersion**. Exposed so an
  /// estimator can be fed the measurement noise the sensor actually had rather
  /// than a datasheet headline. The deterministic albedo pull is deliberately
  /// **not** in here — it is a bias, reported separately as
  /// @ref albedo_angle_rad, and the §8.1 onboard correction removes it.
  double accuracy_sigma_rad = 0.0;
  /// **Truth diagnostic** (does not cross the §2.3 SITL boundary): the
  /// deterministic Earthshine pull applied to this reading [rad], toward the
  /// Earth's centre in the sensor's field. Zero in eclipse, on the night side,
  /// with no Earth in view, or for an ideal sensor. Carried so a study can
  /// separate the correctable bias from the residual dispersion rather than
  /// infer it.
  double albedo_angle_rad = 0.0;
  /// False when the sample period had not elapsed: this reading is the previous
  /// one repeated, carrying no new information. An estimator that treats
  /// repeated values as independent will grow overconfident, so it is flagged.
  bool fresh = true;
  /// True when the Sun is above the eclipse threshold **and** at least one diode
  /// is illuminated by it. False means the FSW has no sun measurement — which is
  /// a normal, frequent condition, not a fault.
  bool sun_present = false;
  /// Same as @ref sun_present unless a fault is injected; separated so a dropout
  /// is distinguishable from a legitimate eclipse in telemetry.
  bool valid = false;
  /// The shadow factor this reading was taken under, passed through for
  /// diagnostics — an estimator rejecting a penumbral reading wants to know.
  double shadow_factor = 0.0;
  /// Albedo counts included in each diode's reading. Carried explicitly because
  /// albedo is the dominant coarse-sensor error and a study wants to see it
  /// separated from the signal rather than inferred.
  std::vector<double> albedo_counts;
  time::Tai time_tag{};
};

/// A sun sensor. Construct with its spec, its unit→body mounting, and a
/// per-source stream id under the run's master seed. Per-diode scale errors and
/// normal misalignments are realised at construction, so the same
/// {spec, seed, stream_id} always builds the same physical unit.
///
/// **Measurement model — analogue** (`kDiodeCounts`). For diode \f$i\f$ with body
/// normal \f$\hat{n}_i\f$, true Sun direction \f$\hat{s}\f$, shadow factor \f$\gamma\f$,
/// and incidence cosine \f$c_i = \hat{n}_i\cdot\hat{s}\f$, the diode is illuminated
/// when \f$\gamma > \gamma_{\min}\f$, \f$c_i > 0\f$, and \f$c_i \ge \cos(\text{FOV})\f$.
/// Its count is the cosine law plus Earthshine plus electronics:
/// \f[
///   \mathrm{cnt}_i = \operatorname{clamp}\!\Big( Q_{\delta}\big( D_i + A_i + c_{\mathrm{dk}} +
///   \sigma_c\, g_i \big),\ 0,\ \mathrm{cnt}_{\max} \Big),
/// \f]
/// \f[
///   D_i = F\, k_i\, c_i\, \gamma \quad(\text{if illuminated, else }0), \qquad
///   A_i = a\, F\, \Phi_i\, \eta,
/// \f]
/// with full scale \f$F\f$ (`full_scale_counts`), per-unit scale \f$k_i = 1+\varepsilon_i\f$,
/// dark counts \f$c_{\mathrm{dk}}\f$, noise \f$\sigma_c\f$ (`noise_counts`), LSB
/// \f$\delta\f$, ceiling \f$\mathrm{cnt}_{\max}\f$ (`saturation_counts`), and albedo
/// coefficient \f$a\f$. The Earthshine view factor
/// \f$\Phi_i = \mathrm{fovCoveredFraction}(\text{FOV},\ \angle(\hat{n}_i,\hat{d}),\ \rho_\oplus)\f$
/// uses the shared §6.1 overlap with Earth angular radius
/// \f$\rho_\oplus = \arcsin(R_\oplus/\|r_{\mathrm{sat}}\|)\f$ and nadir \f$\hat{d}\f$,
/// scaled by the dayside factor
/// \f$\eta = \max\!\big(0,\ (r_{\mathrm{sat}}\cdot
/// r_{\mathrm{sun}})/(\|r_{\mathrm{sat}}\|\,\|r_{\mathrm{sun}}\|)\big)\f$. A failed diode reads
/// \f$0\f$.
///
/// **Measurement model — digital** (`kSunVector`). The unit reports a direction;
/// the model perturbs the truth by the datasheet accuracy, and **then applies the
/// Earthshine pull as a rotation, not as noise**. The incidence angle
/// \f$\theta = \arccos(\hat{b}\cdot\hat{s})\f$ selects the white regime:
/// \f[
///   \sigma_\theta = \begin{cases}\sigma_{\mathrm{in}} & \theta \le
///   \theta_{\mathrm{in}}\\ \sigma_{\mathrm{out}} & \text{otherwise}\end{cases}.
/// \f]
/// The albedo peak scale is the field-fill fraction times the dayside factor,
/// dispersed by this unit's fixed draw \f$d_0\f$:
/// \f[
///   A = \sigma_a^{\max}\,\Phi\,\eta\,(1 + \tfrac{f}{\sqrt2}\,d_0), \qquad
///   \Phi = \mathrm{fovCoveredFraction}(\text{FOV}, \angle(\hat{b},\hat{d}),
///   \rho_\oplus),
/// \f]
/// with \f$f\f$ = `albedo_dispersion_fraction`. The pull angle \f$\varphi\f$ is
/// defined on the **reported** separation from the Earth's centre \f$\hat{d}\f$,
/// \f$\varphi = A\sin\!\big(\angle(\hat{s}_{\mathrm{meas}},\hat{d})\big)\f$ —
/// solved by fixed point, and the definition that makes the onboard inverse of
/// §8.1 a closed form — and the reading is
/// \f[
///   \hat{s}_{\mathrm{meas}} = \operatorname{normalize}\!\Big(
///   R(\hat{a},\varphi)\,\hat{s} + \tfrac{f}{\sqrt2}\,\varphi\,d_1\,\hat{a}
///   + \sigma_\theta\,(g_1\hat{e}_1 + g_2\hat{e}_2)\Big), \qquad
///   \hat{a} = \frac{\hat{s}\times\hat{d}}{\|\hat{s}\times\hat{d}\|},
/// \f]
/// i.e. a rotation of \f$\varphi\f$ *toward the Earth* in the Sun–Earth plane, an
/// out-of-plane dispersion, and the white draw last. \f$f\f$ is the **total** 1σ
/// dispersion, split \f$1/\sqrt2\f$ to each of the two independent axes. The
/// reported \f$\sigma\f$ is \f$\sqrt{\sigma_\theta^2 + (f\varphi)^2}\f$: the pull
/// itself is a bias a correction removes, so it does not belong in a σ.
///
/// **Why directed and not random.** Reflected Earthshine arrives from the sunlit
/// ground in the sensor's field, so it drags the reported vector toward the
/// Earth — a deterministic function of position, Sun direction and boresight.
/// Modelling it as a random tilt of the same magnitude would make it *provably
/// uncorrectable*, which is a claim about the physics that is simply false, and
/// would have hidden the fact that the largest term in the §8.1 sun budget is
/// one the flight software can compute and subtract. What genuinely cannot be
/// removed is the departure of the real Earth from a uniform Lambertian sphere —
/// ocean, cloud and ice differ by more than 5× in reflectance — and that is what
/// \f$f\f$ carries.
///
/// **Model class and its limits.** This is a *centroid* model, not an Earthshine
/// radiance integral: the reflected light is treated as arriving from the centre
/// of the visible Earth disk, weighted by how much of the field that disk fills
/// and by how sunlit the sub-satellite region is. It captures the magnitude, the
/// direction and the orbital phasing. It does not model the offset of the
/// *sunlit* centroid toward the sub-solar limb, the terrain, or the wavelength
/// dependence of the detector's filter. Those are model error, and in flight
/// they are exactly what the dispersion fraction stands in for — which is why
/// that fraction, not the peak, sets the post-correction budget.
///
/// An **ideal** sensor (`noise_enabled = false`) forces \f$\sigma_\theta = 0\f$,
/// \f$A = 0\f$, and drops \f$A_i\f$, dark and noise on the analogue path, while
/// still drawing to keep the stream aligned; the FOV cut and eclipse threshold
/// \f$\gamma_{\min}\f$ still apply, being geometry not noise.
///
/// Markley & Crassidis Ch. 4 (Sensors and Actuators), sun-sensor model [markley2014].
class SunSensor {
 public:
  /// @param spec The datasheet-derived geometry/accuracy specification.
  /// @param mounting_dcm Unit→body rotation placing the boresight.
  /// @param master_seed The run's master RNG seed (§3.5).
  /// @param stream_id This unit's per-source stream id.
  /// @param noise_enabled false builds an **ideal** sun sensor: exact truth Sun
  ///        direction (vector part) or clean cosine-law counts (analogue part),
  ///        with no per-diode miscalibration, dark current, noise, albedo, or
  ///        quantization. The FOV cut-off and eclipse still apply — geometry, not
  ///        noise (§6.2).
  SunSensor(const SunSensorSpec& spec, const Eigen::Matrix3d& mounting_dcm,
            std::uint64_t master_seed, std::uint64_t stream_id, bool noise_enabled = true);

  /// The spec this unit was built from (rates drive the §2.4 loop).
  const SunSensorSpec& spec() const { return spec_; }

  /// Read every diode at truth time @p epoch.
  SunSensorMeasurement sample(const time::Tai& epoch, const SunSensorInput& input);

  /// Diode normals in body axes, as mounted and miscalibrated. Exposed so the
  /// FSW-side reconstruction (§8.1) can be tested against the truth geometry.
  const std::vector<Eigen::Vector3d>& diodeNormalsBody() const { return normals_body_; }

  // --- Fault injection (§9) --------------------------------------------------

  /// Force one diode to read dark (a failed or disconnected cell) until cleared.
  /// Out-of-range indices are ignored. The remaining diodes keep working, which
  /// is the point: a partial failure degrades the reconstruction rather than
  /// removing the sensor, and that is the case FDIR has to catch.
  void failDiode(int index, bool failed = true);

  /// Force every reading invalid (loss of the sensor) until cleared.
  void setDropout(bool dropped) { fault_dropout_ = dropped; }

  void clearFaults();

 private:
  /// Boresight in body axes — the sensor's +z as mounted. For a canted cluster
  /// this is the axis the diodes are arranged around, not any one diode normal.
  Eigen::Vector3d boresight_body_ = Eigen::Vector3d::UnitZ();

  SunSensorSpec spec_;
  random::SplitMix64 rng_;
  bool noise_enabled_ = true;
  std::vector<Eigen::Vector3d> normals_body_;
  std::vector<double> diode_scale_;  ///< realised per-diode scale factor (1 + ε)

  /// This unit's realised albedo dispersion, drawn once at construction: a
  /// standard normal on the Earthshine *scale* and one on its out-of-plane
  /// centroid offset, both multiplied by
  /// @ref SunSensorSpec::albedo_dispersion_fraction when applied. Zero for an
  /// ideal sensor.
  double albedo_scale_dispersion_ = 0.0;
  double albedo_cross_dispersion_ = 0.0;
  std::vector<bool> diode_failed_;
  bool fault_dropout_ = false;

  // Sample-period gating: the last genuinely new reading, and when it was taken.
  SunSensorMeasurement last_measurement_{};
  time::Tai last_sample_time_{};
  bool has_sampled_ = false;
};

}  // namespace polaris::sim::sensors

#endif  // POLARIS_SIM_SENSORS_SUN_SENSOR_HPP
