#include "scenario/sim_runner.hpp"

#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>

#include "frames/eci_ecef.hpp"
#include "frames/eop.hpp"
#include "world/drag.hpp"
#include "world/egm2008.hpp"
#include "world/eop_file.hpp"
#include "world/ephemeris_file.hpp"
#include "world/gravity_field.hpp"
#include "world/gravity_gradient.hpp"
#include "world/igrf_file.hpp"
#include "world/magnetic_field.hpp"
#include "world/space_weather_file.hpp"
#include "world/srp.hpp"
#include "world/third_body.hpp"

#ifdef POLARIS_HAS_NRLMSIS
#include "world/nrlmsis.hpp"
#endif

namespace polaris::sim::scenario {
namespace {

/// EOP records held per run. Bulletin A is daily and loading is windowed to the
/// scenario span plus a margin, so this covers well over a year of propagation
/// while keeping the table ~16 kB rather than the ~600 kB the full ~20 000-entry
/// product would need.
constexpr std::size_t kEopCapacity = 512;

/// Upper bound on trajectory samples per run — a guard against a config whose
/// duration/output-step ratio would hang the process, not a physical limit.
/// Ten million samples is ~1.4 GB of TruthState, far past anything intended.
constexpr std::size_t kMaxSamples = 10'000'000;

using SimEopTable = frames::EopTable<kEopCapacity>;

bool fail(std::string* error, const std::string& message) {
  if (error != nullptr) {
    *error = message;
  }
  return false;
}

}  // namespace

SimRunner::SimRunner() = default;
SimRunner::~SimRunner() = default;

DataPaths DataPaths::under(const std::string& root) {
  return DataPaths{root + "/finals.all.iau2000.txt", root + "/de440_bodies.cheb",
                   root + "/igrf14coeffs.txt", root + "/EGM2008_to200.gfc", root + "/SW-All.csv"};
}

/// Everything the resolvers capture. Address-stable for the runner's lifetime.
struct SimRunner::Impl {
  time::LeapSecondTable leap = time::LeapSecondTable::historical();
  std::unique_ptr<SimEopTable> eop;
  std::unique_ptr<world::EphemerisSet> ephemeris;
  std::unique_ptr<world::EarthMagneticField> magnetic;

  std::unique_ptr<dynamics::TwoBodyGravity> two_body;
  std::unique_ptr<world::SphericalHarmonicGravity> gravity;
  std::unique_ptr<world::ThirdBodyGravity> third_body;
  std::unique_ptr<world::SolarRadiationPressure> srp;
  std::unique_ptr<world::AtmosphericDrag> drag;
  std::unique_ptr<world::ResidualDipoleTorque> dipole;
  std::unique_ptr<world::GravityGradientTorque> gravity_gradient;
#ifdef POLARIS_HAS_NRLMSIS
  std::unique_ptr<world::NrlmsisAtmosphere> nrlmsis;
  std::unique_ptr<world::SpaceWeatherTable> space_weather;
#endif

  /// ECI->ECEF resolver over the loaded EOP window. Shared by the gravity field,
  /// the magnetic field, and NRLMSIS — one wiring, three consumers.
  world::EciToEcefFn eciToEcef() {
    return [this](const time::Tai& t, math::Quat<math::frames::ECEF, math::frames::ECI>& q) {
      return eop != nullptr && frames::ecefFromEci(t, *eop, leap, q);
    };
  }
};

bool SimRunner::build(const SimConfig& config, const DataPaths& paths, std::string* error,
                      const dynamics::ForceTorqueModel* extra_model) {
  config_ = config;
  // Invalidate the previous build FIRST. `body_` holds a pointer into the old
  // composite, and the steps below can fail partway (a missing data product);
  // leaving `body_` set across that would leave ready() reporting true while it
  // pointed at a composite this line is about to free — a use-after-free on the
  // next run(). A failed rebuild must leave the runner unusable, not stale.
  body_.reset();
  impl_ = std::make_unique<Impl>();
  composite_ = std::make_unique<dynamics::CompositeForceModel>();
  Impl& d = *impl_;

  const EnvironmentConfig& env = config.environment;
  const SpacecraftConfig& sc = config.spacecraft;

  // Anything that needs the Earth's orientation needs EOP: the ECEF-frame
  // gravity tesserals, NRLMSIS's longitude, and IGRF. Loading is windowed to the
  // scenario span.
  const bool needs_eop = env.gravity_degree > 0 || env.magnetic_field == MagneticModel::kIgrf ||
                         env.atmosphere == AtmosphereModel::kNrlmsis;
  if (needs_eop) {
    const time::Tai start = config.initial_state.epoch;
    const time::Tai end = start + time::Duration::fromSecondsF(config.propagation.duration_s);
    d.eop = std::make_unique<SimEopTable>();
    if (!world::loadEopFile(paths.eop, d.leap, start, end, *d.eop, error)) {
      return false;
    }
  }

  // Sun and Moon positions feed third-body gravity, SRP, and eclipse alike;
  // planetary perturbers ride the same fixture.
  const bool needs_ephemeris = env.sun_third_body || env.moon_third_body || env.srp_enabled ||
                               !env.planet_third_bodies.empty();
  if (needs_ephemeris) {
    d.ephemeris = std::make_unique<world::EphemerisSet>();
    if (!world::loadEphemerisFile(paths.ephemeris, *d.ephemeris, error)) {
      return false;
    }
  }

  // --- Gravity -------------------------------------------------------------
  // Degree 0 *is* the point-mass term (Cbar_00 = 1 supplies GM/r), so it is
  // served by TwoBodyGravity rather than by loading a 2 MB coefficient file to
  // read one number. A negative degree means no gravity at all — free drift,
  // which is the analytic conservation baseline the integration tests need and
  // not something a real scenario asks for.
  if (env.gravity_degree == 0) {
    d.two_body = std::make_unique<dynamics::TwoBodyGravity>();
    composite_->add(d.two_body.get());
  } else if (env.gravity_degree > 0) {
    world::GravityCoeffs coeffs;
    world::Egm2008Header header;
    try {
      coeffs = world::loadEgm2008Gfc(paths.gravity, env.gravity_degree, &header);
    } catch (const std::exception& e) {
      // The .gfc loader signals a missing file or insufficient coverage by
      // exception; the runner's interface is exceptions-free, so it converts.
      return fail(error, std::string("gravity coefficients: ") + e.what());
    }
    const int order = env.gravity_order < 0 ? env.gravity_degree : env.gravity_order;
    // The model's OWN GM and reference radius, not WGS84's. The coefficients are
    // scaled to the pair the model was solved with — EGM2008 uses
    // GM = 3.986004415e14 and Re = 6378136.3 m, both slightly different from the
    // WGS84 constants the constructor defaults to. Substituting WGS84 rescales
    // every harmonic term by (Re_wgs84/Re_model)^n and shifts the central term,
    // which is a systematic model error that conserves energy perfectly and so
    // survives every self-consistency check. Cross-validation against GMAT is
    // what surfaced it.
    d.gravity = std::make_unique<world::SphericalHarmonicGravity>(
        std::move(coeffs), env.gravity_degree, order, header.gm, header.radius);
    d.gravity->setEciToEcef(d.eciToEcef());
    composite_->add(d.gravity.get());
  }

  // --- Third body ----------------------------------------------------------
  if (env.sun_third_body || env.moon_third_body || !env.planet_third_bodies.empty()) {
    d.third_body = std::make_unique<world::ThirdBodyGravity>();
    if (env.sun_third_body) {
      d.third_body->addBody(constants::bodies::kSunGM, world::bodyPositionFn(d.ephemeris->sun));
    }
    if (env.moon_third_body) {
      d.third_body->addBody(constants::bodies::kMoonGM, world::bodyPositionFn(d.ephemeris->moon));
    }
    // DE440 system GMs, parallel to world::kPlanetNames (Mercury → Neptune).
    static constexpr std::array<double, 7> kPlanetGMs = {
        constants::bodies::kMercuryGM, constants::bodies::kVenusGM,  constants::bodies::kMarsGM,
        constants::bodies::kJupiterGM, constants::bodies::kSaturnGM, constants::bodies::kUranusGM,
        constants::bodies::kNeptuneGM};
    for (const std::string& name : env.planet_third_bodies) {
      const world::SimEphemerisTable* table = d.ephemeris->find(name);
      // The config layer already validated the name; what can still go wrong is
      // an old Sun/Moon-only fixture, which loads fine but has no segments for
      // the planet. An empty table would silently skip the perturbation on
      // every epoch, so refuse loudly instead.
      if (table == nullptr || table->size() == 0) {
        return fail(error, "third body '" + name +
                               "' is not in the ephemeris fixture — regenerate it with "
                               "tools/ephem (de440_bodies.cheb carries the planets)");
      }
      for (std::size_t i = 0; i < world::kPlanetNames.size(); ++i) {
        if (name == world::kPlanetNames[i]) {
          d.third_body->addBody(kPlanetGMs[i], world::bodyPositionFn(*table));
          break;
        }
      }
    }
    composite_->add(d.third_body.get());
  }

  // --- Solar radiation pressure -------------------------------------------
  if (env.srp_enabled) {
    d.srp = std::make_unique<world::SolarRadiationPressure>(
        sc.srp_area_m2, sc.mass_kg, sc.srp_cr, world::bodyPositionFn(d.ephemeris->sun));
    // The lever arm is what turns the force into a torque, so leaving it zero is
    // exactly how the torque is switched off — no separate gate needed.
    if (env.srp_torque_enabled) {
      d.srp->setCenterOfPressureOffset(sc.cp_offset_srp_m);
    }
    d.srp->setEclipseEnabled(env.eclipse_enabled);
    composite_->add(d.srp.get());
  }

  // --- Drag ----------------------------------------------------------------
  if (env.drag_enabled) {
    world::DensityFn density;
    if (env.atmosphere == AtmosphereModel::kNrlmsis) {
#ifdef POLARIS_HAS_NRLMSIS
      d.nrlmsis = std::make_unique<world::NrlmsisAtmosphere>();
      if (!d.nrlmsis->good()) {
        return fail(error, "NRLMSIS init failed: " + d.nrlmsis->error());
      }
      d.nrlmsis->setEciToEcef(d.eciToEcef());
      d.nrlmsis->setLeapSeconds(&d.leap);

      // Drive F10.7/Ap from the committed CelesTrak record over the scenario
      // span, so drag tracks the real solar cycle rather than a fixed snapshot.
      const time::Tai start = config.initial_state.epoch;
      const time::Tai end = start + time::Duration::fromSecondsF(config.propagation.duration_s);
      d.space_weather = std::make_unique<world::SpaceWeatherTable>();
      if (!d.space_weather->load(paths.space_weather, d.leap, start, end, error)) {
        return false;
      }
      d.nrlmsis->setSpaceWeatherSource(
          [tbl = d.space_weather.get()](const time::Tai& t, world::SpaceWeather& sw) {
            double f107 = 0.0;
            double f107a = 0.0;
            double ap_daily = 0.0;
            if (!tbl->at(t, f107, f107a, ap_daily)) {
              return false;
            }
            sw.f107 = f107;
            sw.f107a = f107a;
            sw.ap.fill(ap_daily);  // only ap[0] is read; storm-time switches off
            return true;
          });
      density = d.nrlmsis->densityFn();
#else
      // Refuse rather than quietly substituting the coarse model: a run that
      // asked for NRLMSIS and silently got the exponential fit would report drag
      // numbers that answer a different question.
      return fail(error,
                  "scenario requests the NRLMSIS atmosphere but this build has it disabled "
                  "(configure with -DPOLARIS_BUILD_NRLMSIS=ON and a Fortran compiler)");
#endif
    } else {
      density = world::exponentialAtmosphere;
    }
    d.drag = std::make_unique<world::AtmosphericDrag>(sc.drag_area_m2, sc.mass_kg, sc.drag_cd,
                                                      std::move(density));
    if (env.aero_torque_enabled) {
      d.drag->setCenterOfPressureOffset(sc.cp_offset_aero_m);
    }
    composite_->add(d.drag.get());
  }

  // --- Magnetic residual dipole -------------------------------------------
  if (env.magnetic_field == MagneticModel::kIgrf) {
    environment::IgrfCoefficients coefficients;
    // IGRF is parameterised by decimal year; the loader wants the epoch of
    // interest, and the scenario start is what the run is centred on.
    double year = 0.0;
    if (!world::decimalYear(config.initial_state.epoch, d.leap, year)) {
      return fail(error, "cannot resolve the scenario epoch to a decimal year");
    }
    if (!world::loadIgrfFile(paths.igrf, year, coefficients, error)) {
      return false;
    }
    d.magnetic = std::make_unique<world::EarthMagneticField>(coefficients);
    d.magnetic->setEciToEcef(d.eciToEcef());
    d.magnetic->setLeapSeconds(&d.leap);

    if (env.residual_dipole_torque_enabled) {
      d.dipole = std::make_unique<world::ResidualDipoleTorque>(sc.residual_dipole_am2,
                                                               d.magnetic->fieldFn());
      composite_->add(d.dipole.get());
    }
  }

  // --- Gravity gradient ----------------------------------------------------
  // Uses the gravity model's own GM where there is one, so the gradient and the
  // central force are consistent (EGM2008's GM differs from WGS84's in the 8th
  // digit — small, but there is no reason to introduce the discrepancy). The
  // inertia is referenced from the stored config, which outlives the composite.
  //
  // A negative gravity_degree is free drift — no gravity at all — so there is no
  // GM to take a gradient of, and a scenario that switched gravity off would be
  // surprised to find a gravity torque on the body. The couple is therefore
  // gated on gravity being modelled; an MC run that wants it can ask for
  // gravity_degree: 0 (point mass), which is the term the gradient uses anyway.
  if (env.gravity_gradient_torque_enabled && env.gravity_degree >= 0) {
    const double mu = d.gravity != nullptr ? d.gravity->mu() : constants::wgs84::kGM;
    d.gravity_gradient =
        std::make_unique<world::GravityGradientTorque>(config_.spacecraft.inertia_kgm2, mu);
    composite_->add(d.gravity_gradient.get());
  }

  // The closed loop's actuator-feedback channel (or any test-supplied extra
  // source). Composed last so the budget tests' model counts stay stable when
  // no extra model is passed.
  composite_->add(extra_model);

  if (composite_->size() == 0) {
    // Not an error — a free-drift scenario is a legitimate baseline, and the
    // conservation tests depend on it — but the plant still needs a model.
    composite_->add(nullptr);
  }

  body_ = std::make_unique<dynamics::RigidBody6Dof>(sc.inertia_kgm2, *composite_);
  return true;
}

world::MagneticFieldFn SimRunner::magneticFieldFn() const {
  if (impl_ == nullptr || impl_->magnetic == nullptr) {
    return {};
  }
  return impl_->magnetic->fieldFn();
}

world::BodyPositionFn SimRunner::sunPositionFn() const {
  if (impl_ == nullptr || impl_->ephemeris == nullptr) {
    return {};
  }
  return world::bodyPositionFn(impl_->ephemeris->sun);
}

world::BodyPositionFn SimRunner::moonPositionFn() const {
  if (impl_ == nullptr || impl_->ephemeris == nullptr) {
    return {};
  }
  return world::bodyPositionFn(impl_->ephemeris->moon);
}

math::Vec3<math::frames::ECI> SimRunner::nonGravAcceleration(const state::TruthState& s) const {
  Eigen::Vector3d a = Eigen::Vector3d::Zero();
  if (impl_ != nullptr) {
    if (impl_->srp != nullptr) {
      a += impl_->srp->acceleration(s).eigen();
    }
    if (impl_->drag != nullptr) {
      a += impl_->drag->acceleration(s).eigen();
    }
  }
  return math::Vec3<math::frames::ECI>(a);
}

const world::SphericalHarmonicGravity* SimRunner::gravityField() const {
  return impl_ == nullptr ? nullptr : impl_->gravity.get();
}

std::size_t SimRunner::modelCount() const {
  return composite_ == nullptr ? 0 : composite_->size();
}

bool SimRunner::run(std::vector<TrajectorySample>& out, std::string* error) const {
  out.clear();
  if (!ready()) {
    return fail(error, "SimRunner::run called before a successful build()");
  }

  const PropagationConfig& prop = config_.propagation;
  dynamics::StepControl control;
  control.abs_tol = prop.abs_tol;
  control.rel_tol = prop.rel_tol;
  control.max_step = prop.max_step_s;

  state::TruthState s = config_.initial_state;
  out.push_back({0.0, s});

  const double raw_steps = std::floor(prop.duration_s / prop.output_step_s);
  // A hand-edited config (main.cpp takes any --config path, not just a compiled
  // one) can ask for duration/step ratios that would hang the process and grow
  // the sample vector without bound. Refuse rather than appear to work.
  if (raw_steps > static_cast<double>(kMaxSamples)) {
    return fail(error, "propagation would produce " + std::to_string(raw_steps) +
                           " samples, above the " + std::to_string(kMaxSamples) +
                           " limit; raise output_step_s or shorten duration_s");
  }
  const auto steps = static_cast<std::size_t>(raw_steps);
  double t = 0.0;
  for (std::size_t i = 0; i < steps; ++i) {
    s = body_->propagate(s, prop.output_step_s, control);
    // Computed from the index rather than accumulated, so a long run cannot
    // drift the reported sample times away from the epochs actually integrated.
    t = static_cast<double>(i + 1) * prop.output_step_s;
    out.push_back({t, s});
  }
  // A duration that is not a whole number of output steps gets a final short
  // step, so the last sample is at the requested end time rather than short of
  // it.
  const double remainder = prop.duration_s - t;
  if (remainder > 1.0e-9) {
    s = body_->propagate(s, remainder, control);
    out.push_back({prop.duration_s, s});
  }
  return true;
}

bool SimRunner::runAt(const std::vector<double>& times_s, std::vector<TrajectorySample>& out,
                      std::string* error) const {
  out.clear();
  if (!ready()) {
    return fail(error, "SimRunner::runAt called before a successful build()");
  }

  const PropagationConfig& prop = config_.propagation;
  dynamics::StepControl control;
  control.abs_tol = prop.abs_tol;
  control.rel_tol = prop.rel_tol;
  control.max_step = prop.max_step_s;

  state::TruthState s = config_.initial_state;
  double previous = 0.0;
  bool first = true;
  out.reserve(times_s.size());
  for (const double t : times_s) {
    // The first sample may be at t=0 (the epoch); every later one must strictly
    // advance so each gap-step propagate() moves forward. `>= previous` silently
    // accepted duplicate times, which the message claimed it rejected.
    const bool ordered = first ? (t >= 0.0) : (t > previous);
    if (!std::isfinite(t) || !ordered) {
      return fail(error, "runAt times must be finite, non-negative, and strictly increasing");
    }
    first = false;
    // Advance by the gap rather than restarting from the epoch each time, so the
    // integrator sees one continuous trajectory.
    s = body_->propagate(s, t - previous, control);
    previous = t;
    out.push_back({t, s});
  }
  return true;
}

bool writeTrajectoryCsv(const std::string& path, const SimConfig& config,
                        const std::vector<TrajectorySample>& samples, std::string* error) {
  std::ofstream file(path);
  if (!file.good()) {
    return fail(error, "cannot write trajectory: " + path);
  }

  file << "# Polaris truth trajectory\n"
       << "# scenario:    " << config.scenario_name << "\n"
       << "# spacecraft:  " << config.spacecraft.name << "\n"
       << "# config_hash: " << config.config_hash << "\n"
       << "# Columns: seconds since epoch; ECI position [m]; ECI velocity [m/s];\n"
       << "# attitude quaternion Body<-ECI scalar-first; body rate [rad/s].\n"
       << "t_s,x_m,y_m,z_m,vx_ms,vy_ms,vz_ms,q0,q1,q2,q3,wx_rads,wy_rads,wz_rads\n";

  // 17 significant digits so a written trajectory round-trips exactly through
  // IEEE-754 — a regression fixture that loses digits on write compares against
  // a number the sim never produced.
  file << std::setprecision(17);
  for (const TrajectorySample& sample : samples) {
    const Eigen::Vector3d r = sample.state.position.eigen();
    const Eigen::Vector3d v = sample.state.velocity.eigen();
    const Eigen::Vector4d q = sample.state.attitude.core().coeffs();
    const Eigen::Vector3d w = sample.state.body_rate.eigen();
    file << sample.t_s << ',' << r.x() << ',' << r.y() << ',' << r.z() << ',' << v.x() << ','
         << v.y() << ',' << v.z() << ',' << q[0] << ',' << q[1] << ',' << q[2] << ',' << q[3] << ','
         << w.x() << ',' << w.y() << ',' << w.z() << '\n';
  }
  if (!file.good()) {
    return fail(error, "failed while writing trajectory: " + path);
  }
  return true;
}

}  // namespace polaris::sim::scenario
