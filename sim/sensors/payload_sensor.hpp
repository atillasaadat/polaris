#ifndef POLARIS_SIM_SENSORS_PAYLOAD_SENSOR_HPP
#define POLARIS_SIM_SENSORS_PAYLOAD_SENSOR_HPP

/// @file
/// @brief Generic payload-sensor geometry model — anything with a boresight
/// (design doc §6.3).
///
/// An imager, a laser terminal, a radiometer, a rangefinder: physically they
/// have almost nothing in common, but every one of them is *pointed*, and the
/// pointing questions are identical. Where is the boresight? Is the Earth, Sun
/// or Moon in the way? How far off the Sun is the observation? How far off
/// nadir? Those questions do not depend on whether the detector counts photons
/// or measures a return time, so they are answered once, here, and a
/// radiometric or error model is layered on later per instrument rather than
/// re-deriving the geometry each time.
///
/// **+Z in the sensor frame is always the boresight.** Not a configurable
/// direction, and deliberately so. A per-unit `boresight_sensor` vector on top
/// of a mounting rotation gives two ways to express the same orientation and
/// therefore two places for it to be wrong, with the failure — a payload
/// pointing somewhere other than the analysis assumed — being silent. The
/// mounting rotation carries sensor +Z into body axes and is the *only* place a
/// payload's orientation is expressed. Sensor +X and +Y complete the
/// right-handed frame and are the two field-of-view axes for a non-circular
/// field (below).
///
/// **Field-of-view shapes.** Three, because the instruments genuinely differ and
/// the difference is not cosmetic — a rectangular imager sees a corner target a
/// circular model of the same across-track width rejects:
///
///  - **Conic** — one half-angle about +Z. A rangefinder, a laser terminal, a
///    radiometer with a circular beam.
///  - **Square** — equal half-angles about the sensor X and Y axes; a square
///    detector.
///  - **Rectangular** — independent half-angles; the common pushbroom or framing
///    imager, where the across-track and along-track fields differ.
///
/// The shape is **inferred from which parameters the catalog entry sets** rather
/// than named by a separate key: `half_fov_deg` alone is conic, `half_fov_x_deg`
/// with `half_fov_y_deg` is rectangular (or square when the two are equal). One
/// fact stated once — a `fov_shape: conic` beside two per-axis half-angles would
/// be a contradiction the schema could not resolve, and the param map is
/// `double`-valued anyway.
///
/// **Pixel counts** (`pixels_x`, `pixels_y`) give the instantaneous field of
/// view — the angular size of one pixel, which is what turns a pointing error
/// into a smear in pixels. `1 x 1` is the correct and expected configuration for
/// a single-element instrument such as a laser or a rangefinder; it is not a
/// degenerate case.
///
/// **Occlusion is the shared §6.1 model, not a private copy.** The evaluator is
/// the one the star tracker uses, atmosphere height and FOV-coverage fractions
/// included, so a payload and a tracker on the same vehicle cannot disagree
/// about whether the Earth is in the way. It takes a *circular* half-angle, so a
/// square or rectangular field is passed as the **equal-solid-angle equivalent
/// cone** (`equivalentHalfFovRad`) — the fractions are then right in the mean and
/// wrong in the corners, which is the same order of approximation as the planar
/// two-circle lens formula the fractions already use. The keep-out verdict is
/// unaffected: it is a boresight-to-limb angle and never touches the FOV shape.
/// Use `inFieldOfView` for an exact per-direction shape test.
///
/// **This is the simple model.** Geometry and the pointing products, nothing
/// more: no radiometry, no detector noise, no MTF, no smear, no jitter response,
/// no image formation. Those are per-instrument and get built when an instrument
/// needs them; the geometry underneath will not change when they do. There is
/// consequently **no RNG stream and no `noise_enabled` switch** — geometry is not
/// noise, so the §6.2 master switch has nothing to turn off here, exactly as
/// occlusion and the field-of-view cut-off stay live for every other sensor.
///
/// **What it deliberately does not do:** produce a measurement for the FSW. A
/// payload's data path is not the ADCS sensor path, and there is no flight-side
/// payload component in this push — the products below stay sim-side, alongside
/// the truth-derived occlusion fractions the star tracker already keeps there.
///
/// References:
///  - Wertz, Everett & Puschell, *Space Mission Engineering: The New SMAD*,
///    2011, §9.3 (payload field of view, IFOV, pointing geometry). [wertz2011]
///
/// Implements REQ-SIM-003 (sensor truth models, shared occlusion) and
/// REQ-SIM-005 (scriptable fault injection); the cross-boresight knowledge
/// requirement it exists to make measurable is REQ-PAY-001.

#include <cstdint>
#include <Eigen/Core>
#include <map>
#include <string>

#include "math/frames.hpp"
#include "math/quaternion.hpp"
#include "math/typed_vector.hpp"
#include "sensors/occlusion.hpp"
#include "time/timescales.hpp"

namespace polaris::sim::sensors {

/// The three field-of-view shapes. Inferred from the configured half-angles (see
/// the file header), never named directly in the catalog.
enum class FovShape {
  kConic,        ///< one half-angle about the boresight
  kSquare,       ///< equal half-angles about sensor X and Y
  kRectangular,  ///< independent half-angles about sensor X and Y
};

/// One payload sensor's geometry, in SI. Boresight is sensor +Z, always.
struct PayloadSensorSpec {
  FovShape shape = FovShape::kConic;
  /// Conic: the cone half-angle. Square/rectangular: the half-angle in the
  /// sensor **X** direction, i.e. measured in the X–Z plane [rad].
  double half_fov_x_rad = 0.0;
  /// Half-angle in the sensor **Y** direction (Y–Z plane) [rad]. Equal to
  /// `half_fov_x_rad` for a square field; unused for a conic one.
  double half_fov_y_rad = 0.0;
  /// Detector format. `1 x 1` is a single-element instrument (laser,
  /// rangefinder), not a misconfiguration.
  std::int32_t pixels_x = 1;
  std::int32_t pixels_y = 1;
  /// Bright-body exclusion, as absolute boresight-to-limb angles (§6.1). Zero
  /// disables a constraint — a laser terminal may have no Sun keep-out at all.
  KeepOutSpec keep_out;
  /// Native product rate [Hz]; 0 means "sampled when the loop samples", the same
  /// convention the other discrete sensors use.
  double update_rate_hz = 0.0;

  /// Half-angle [rad] of the circular field with the **same solid angle** as this
  /// one — what the §6.1 occlusion evaluator is handed, since it works on a cone.
  /// Identity for a conic field. For a rectangular pyramid of half-angles
  /// \f$(h_x, h_y)\f$ the solid angle is
  /// \f$\Omega = 4\arcsin(\sin h_x \sin h_y)\f$, and the equivalent cone follows
  /// from \f$\Omega = 2\pi(1 - \cos h_e)\f$.
  double equivalentHalfFovRad() const;

  /// Instantaneous field of view [rad]: the angular size of one pixel along the
  /// sensor X and Y directions, \f$2h/n\f$. For a conic field both are the full
  /// cone divided by the respective pixel count.
  double ifovXRad() const;
  double ifovYRad() const;

  /// Build a spec from the catalog's params (`config/hardware/payload_sensor/`),
  /// converting to SI. Keys: `half_fov_deg` (conic) **or** `half_fov_x_deg` +
  /// `half_fov_y_deg` (square/rectangular); `pixels_x`, `pixels_y`;
  /// `update_rate_hz`; `sun_exclusion_deg`, `earth_exclusion_deg`,
  /// `moon_exclusion_deg`. Missing keys default to 0 (that term disabled).
  static PayloadSensorSpec fromParams(const std::map<std::string, double>& params);
};

/// Everything one geometry evaluation needs from the truth state.
struct PayloadSensorInput {
  /// True Body←ECI attitude.
  math::Quat<math::frames::Body, math::frames::ECI> attitude{};
  /// Spacecraft / Sun / Moon geometry for the occlusion check.
  SkyGeometry sky{};
};

/// One payload sensor's pointing products at an instant.
struct PayloadSensorSample {
  /// Boresight direction in ECI (unit), as mounted and after any injected
  /// misalignment. The quantity a cross-boresight knowledge error is measured
  /// against (REQ-PAY-001).
  math::Vec3<math::frames::ECI> boresight_eci{};
  /// Shared §6.1 line-of-sight verdict, FOV coverage fractions, and the
  /// boresight's **Sun and nadir angles** (`occlusion.sun_angle_rad`,
  /// `occlusion.nadir_angle_rad`) — the same fields, from the same evaluator,
  /// the star tracker reports.
  OcclusionState occlusion{};
  /// False when a keep-out is violated or the unit is failed. Carried alongside
  /// the geometry rather than replacing it: a consumer wants to know *why* an
  /// observation was refused, and the occluder says so.
  bool valid{true};
  time::Tai time_tag{};
};

/// A payload sensor. Construct with its spec and its unit→body mounting.
///
/// Deterministic and stateless apart from the injected faults: there is no
/// random stream to seed (see the file header on why geometry carries no noise).
class PayloadSensor {
 public:
  /// @param spec         Field of view, format, keep-out.
  /// @param mounting_dcm Unit→body rotation. Its third column is the boresight in
  ///                     body axes, since the boresight is sensor +Z.
  PayloadSensor(const PayloadSensorSpec& spec, const Eigen::Matrix3d& mounting_dcm);

  const PayloadSensorSpec& spec() const { return spec_; }

  /// The boresight in body axes, as mounted and including any injected
  /// misalignment.
  const Eigen::Vector3d& boresightBody() const { return boresight_body_; }

  /// Evaluate the pointing geometry at truth time @p epoch.
  PayloadSensorSample sample(const time::Tai& epoch, const PayloadSensorInput& input) const;

  /// Whether @p direction_sensor (in **sensor** axes, need not be unit) falls
  /// inside the field of view — the exact shape test, where the sample's
  /// occlusion fractions use the equivalent cone. A direction in the rear
  /// hemisphere (non-positive Z) is always outside.
  bool inFieldOfView(const Eigen::Vector3d& direction_sensor) const;

  // --- Fault injection (§9) --------------------------------------------------

  /// Persistent boresight misalignment [rad], as a small-angle rotation vector in
  /// **body** axes, until cleared (replaces, not accumulates). Thermal distortion
  /// of an optical bench, or a deployment that did not latch where it should:
  /// the instrument keeps reporting, it is simply not looking where the pointing
  /// solution believes. That is the failure a payload FDIR check has to catch
  /// from the data, since nothing in the geometry flags it.
  void injectBoresightMisalignment(const math::Vec3<math::frames::Body>& delta_rad);

  /// Fail the unit until cleared: samples come back invalid with the geometry
  /// still filled in (a closed shutter, a dead detector, a lost link).
  void setDropout(bool dropped) { fault_dropout_ = dropped; }

  void clearFaults();

 private:
  /// Recompute `boresight_body_` from the mounting and the current fault.
  void updateBoresight();

  PayloadSensorSpec spec_;
  Eigen::Matrix3d mounting_dcm_;
  Eigen::Vector3d boresight_body_;
  Eigen::Vector3d fault_misalignment_ = Eigen::Vector3d::Zero();
  bool fault_dropout_ = false;
};

}  // namespace polaris::sim::sensors

#endif  // POLARIS_SIM_SENSORS_PAYLOAD_SENSOR_HPP
