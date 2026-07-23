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
/// **Albedo is the dominant error, not the noise.** Earthshine reaching a diode
/// is a first-order effect for a coarse sensor in LEO — the Earth can fill a
/// wide-FOV diode's view and reflect ~30% of the incident sunlight, which is why
/// coarse sun sensors are routinely quoted at several degrees of accuracy despite
/// millivolt-clean electronics. It is modelled here from the §6.1 occlusion
/// fractions: how much of the diode's field of view the Earth fills, scaled by
/// how sunlit the sub-satellite region is. This is a **first-order model** — a
/// rigorous treatment needs a surface-reflectance map (ocean, cloud, ice differ
/// by more than 5×) and a view-factor integral over the visible cap. It captures
/// the magnitude and the orbital phasing, not the terrain.
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
  /// Peak *additional* angular error from Earthshine [rad], reached when the
  /// Earth fills the field of view on the day side. Vendors are explicit that
  /// this dominates: GomSpace quote errors above 10° uncorrected, against a
  /// 0.5° clean-sky figure.
  double albedo_error_rad = 0.0;

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
  /// The 1σ accuracy this reading was drawn with [rad], albedo included. Exposed
  /// so an estimator can be fed the measurement noise the sensor actually had
  /// rather than a datasheet headline, and so a study can see the degradation.
  double accuracy_sigma_rad = 0.0;
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
class SunSensor {
 public:
  SunSensor(const SunSensorSpec& spec, const Eigen::Matrix3d& mounting_dcm,
            std::uint64_t master_seed, std::uint64_t stream_id);

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
  std::vector<Eigen::Vector3d> normals_body_;
  std::vector<double> diode_scale_;  ///< realised per-diode scale factor (1 + ε)
  std::vector<bool> diode_failed_;
  bool fault_dropout_ = false;

  // Sample-period gating: the last genuinely new reading, and when it was taken.
  SunSensorMeasurement last_measurement_{};
  time::Tai last_sample_time_{};
  bool has_sampled_ = false;
};

}  // namespace polaris::sim::sensors

#endif  // POLARIS_SIM_SENSORS_SUN_SENSOR_HPP
