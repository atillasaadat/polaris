/// @file
/// @brief The 7-day orbit-determination Monte Carlo campaign (design doc §8.3,
/// §9.2, §13, §23.2).
///
/// What the §8.3 filter's unit tests cannot answer. They run 200 s arcs against
/// a truth perturbed by exactly the `Q_d` the filter budgets, which is the right
/// way to ask whether the covariance is self-consistent — and it is deliberately
/// a *closed* question: the truth model and the filter model are the same model.
/// This campaign asks the open ones. Over a week of real dynamics, with a truth
/// stack the filter does not have (32x32 geopotential, the full piecewise
/// atmosphere, Sun and Moon third bodies, SRP), does the estimate stay bounded,
/// does the covariance still mean what it says, and what happens when the
/// receiver misbehaves in each of the ways §9.2 says it can?
///
/// **Why this is a driver and not a test.** Same reason as `detumble_mc.cpp`: it
/// is a measurement instrument. One 7-day run is minutes of wall clock, the
/// campaign is hours, and it makes no pass/fail claim of its own — the verdict is
/// produced downstream by `analysis/od`, which reads the JSONL this writes. It is
/// therefore **not registered with ctest** (see `tests/mc/CMakeLists.txt`). The
/// claims that *are* gated live in `tests/unit/orbit_od_test.cpp`.
///
/// **Why in-process, when the detumble campaign forks the flight binary.**
/// Detumble's input is the voted, interlock-gated field an F´ component
/// publishes, so reconstructing it sim-side would transcribe the flight path.
/// The orbit filter has no such chain: `gnc::OrbitOd` *is* the flight code, it
/// takes a `GnssFix` struct and an EOP table and nothing else, and its F´
/// component seam does not exist yet (§8.3, "still owed"). Running it in-process
/// against the receiver model is therefore the real article, not a stand-in —
/// and when the seam lands, this driver should move behind it for the same
/// reason detumble already is.
///
/// **Faults are applied to the receiver, never to the filter.** The filter has
/// to discover a fault from the data it is handed, exactly as it would in
/// flight. `orbit_od_scenarios.hpp` carries the timelines and the argument for
/// each one.
///
/// **Each cycle propagates to *now* and only then ingests.** That is the order
/// the FSW runs in, and it is load-bearing rather than cosmetic. Ingesting first
/// leaves the filter's epoch at the fix's own, so every later fix looks *forward*
/// and the latent-fix correction is never reached — a campaign in that order
/// cannot exercise the branch it most needs to. It is also the quantity that
/// matters: "where am I now" is what pointing, pass planning and maneuver
/// targeting all ask, and it is not the same as "where was I at the last fix".
///
/// **The default cadence is 10 s, and that is a conservative choice.** The
/// reference receiver runs to 100 Hz and the flown case is ~1 Hz; 10 s is well
/// inside the 300 s coast horizon, so every policy under test still behaves the
/// same way, but the filter gets two orders less information than it will fly
/// with. The campaign therefore *over*-states the error, which is the right
/// direction for a bound. It is also what makes a 7-day arc affordable: the
/// truth sim is re-entered once per cycle, and at 1 Hz that would be 604800
/// re-entries per run.
///
/// **One scenario opts out of that cadence, and must.** A 10 s cycle cannot
/// resolve the receiver's 50 ms fix latency — the delay line delivers the newest
/// solution at least one latency old, so at 10 s the delivered fix is a whole
/// *poll* old and the datasheet value would model a 10 s delay rather than a
/// 50 ms one. `latency_fast` therefore runs 50 Hz over ten minutes with the real
/// latency armed, which is the only place in the campaign where the latent-fix
/// branch fires. `Scenario::cycle_period_s` and friends carry the per-scenario
/// override.
///
/// Usage (see `analysis/od/README.md` for the full recipe):
/// @code
///   ./polaris_orbit_od_mc --runs 1 --duration-s 3600        # smoke
///   ./polaris_orbit_od_mc --first-run 0 --runs 4 --duration-s 604800
///       --out build-artifacts/orbit-od-mc/shard-0.jsonl
/// (one line; split here because a trailing backslash in a `//` comment trips
///  -Wcomment and this file is compiled with -Werror)
/// @endcode

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <Eigen/Core>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "constants/constants.hpp"
#include "frames/eci_ecef.hpp"
#include "frames/eop.hpp"
#include "gnc/orbit_od.hpp"
#include "math/frames.hpp"
#include "math/typed_vector.hpp"
#include "orbit_od_scenarios.hpp"
#include "random/rng.hpp"
#include "scenario/sim_config.hpp"
#include "scenario/sim_runner.hpp"
#include "sensors/gnss.hpp"
#include "sensors/gnss_jamming.hpp"
#include "time/leap_seconds.hpp"
#include "time/timescales.hpp"
#include "world/eop_file.hpp"

namespace {

namespace pc = polaris::constants;
namespace pf = polaris::frames;
namespace pg = polaris::gnc;
namespace pm = polaris::math;
namespace pmf = polaris::math::frames;
namespace ps = polaris::sim::scenario;
namespace psen = polaris::sim::sensors;
namespace pt = polaris::time;
namespace pw = polaris::sim::world;
namespace mco = polaris::mc::od;

// ---------------------------------------------------------------------------
// Campaign configuration
// ---------------------------------------------------------------------------

/// Reference orbit: the ISS-like 400 km, 51.6 deg case the filter's own tests and
/// the GMAT fixtures use, so the campaign is flown on the same trajectory every
/// other §8.3 measurement was taken on.
constexpr double kAltitudeM = 400.0e3;
constexpr double kInclinationRad = 51.6 * 3.14159265358979323846 / 180.0;

/// Truth gravity degree/order. Above the onboard model's 8x8 by enough that the
/// difference is a real truncation and not two implementations of one field —
/// the mistake §8.3 records having made once already.
constexpr int kTruthGravityDegree = 32;

/// Fix cadence [s]. See the file header for why this is 10 s and not 1 s.
constexpr double kFixPeriodS = 10.0;

/// Filter tuning, from `tests/unit/orbit_od_test.cpp` — the same values the
/// gated tests use, so a campaign result and a test result are about one filter.
constexpr double kCoastHorizonS = 300.0;
constexpr double kTruncationAtHorizonM = 1.8;
constexpr double kAccelPsd = 3.0 * kTruncationAtHorizonM * kTruncationAtHorizonM /
                             (kCoastHorizonS * kCoastHorizonS * kCoastHorizonS);
constexpr double kChi2_3_999 = 16.266;
constexpr double kMaxFixLatencyS = 0.2;
constexpr double kBallisticCoeff = 2.2 * 0.06 / 12.0;

/// Initial-state dispersion, 1σ. A campaign that seeded every run on the same
/// state would measure one trajectory many times: the geopotential residual and
/// the drag both depend on where in the orbit the arc starts, so the spread
/// across runs is part of what is being measured.
constexpr double kInitialPosSigmaM = 5.0e3;
constexpr double kInitialVelSigmaMps = 5.0;

/// Receiver spec keys, mirroring `config/hardware/gnss/novatel_oem7600.yaml`.
///
/// **Latency is the one value not taken from the catalog**, because it is only
/// meaningful relative to the poll cadence and the cadence is per-scenario. The
/// catalog carries `fix_latency_s: 0.05`; the delay-line model delivers the
/// newest solution *at least* one latency old, so a scenario that polls slower
/// than the latency gets a fix a whole poll old rather than 50 ms old. Flying
/// the datasheet value at the long arcs' 10 s cadence therefore models a 10 s
/// latency — measured, a constant 76.7 km of along-track offset that the filter
/// tracks perfectly, because it is consistent and simply not the trajectory the
/// record compares against. So the long arcs pass zero and `latency_fast` polls
/// at 50 Hz with the real value. See `GnssSpec::fix_latency_s`.
psen::GnssSpec receiverSpec(double fix_latency_s) {
  return psen::GnssSpec::fromParams({
      {"horizontal_position_rms_m", 1.2},
      {"velocity_accuracy_m_s_rms", 0.03},
      {"time_accuracy_ns_rms", 5.0},
      {"max_rate_hz", 100.0},
      {"fix_latency_s", fix_latency_s},
      {"cold_start_s", 34.0},
      {"hot_start_s", 20.0},
      {"reacquisition_s", 0.5},
  });
}

pg::OrbitOdConfig filterConfig() {
  pg::OrbitOdConfig cfg;
  cfg.mu_m3_per_s2 = pc::gravity::kGM;
  cfg.reference_radius_m = pc::gravity::kReferenceRadius;
  cfg.geopotential_degree = pg::kGeopotentialMaxDegree;
  cfg.geopotential_order = pg::kGeopotentialMaxDegree;
  cfg.drag_ballistic_coeff_m2_per_kg = kBallisticCoeff;
  cfg.drag_ref_density_kg_m3 = 3.725e-12;  // Vallado Table 8-4, 400 km band
  cfg.drag_ref_altitude_m = 400.0e3;
  cfg.drag_scale_height_m = 58'515.0;
  cfg.accel_psd_m2_per_s3 = kAccelPsd;
  cfg.position_nis_gate = kChi2_3_999;
  cfg.velocity_nis_gate = kChi2_3_999;
  cfg.max_coast_s = kCoastHorizonS;
  cfg.max_dt_s = 60.0;
  cfg.max_step_s = 10.0;
  cfg.max_fix_latency_s = kMaxFixLatencyS;
  cfg.min_radius_m = 6.4e6;
  cfg.max_radius_m = 5.0e7;
  return cfg;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

pt::Tai campaignEpoch() {
  return pt::Tai::fromNanosecondsSinceEpoch(1767225637000000000LL);  // 2026-01-01 TAI
}

pt::Tai advance(const pt::Tai& t, double seconds) {
  return t + pt::Duration::fromSecondsF(seconds);
}

/// Circular orbit at @p altitude and @p inclination, with the ascending node at
/// the +X axis.
void circularState(double altitude_m, double inclination_rad, double mu, Eigen::Vector3d& r,
                   Eigen::Vector3d& v) {
  const double radius = pc::gravity::kReferenceRadius + altitude_m;
  const double speed = std::sqrt(mu / radius);
  r = Eigen::Vector3d(radius, 0.0, 0.0);
  v = Eigen::Vector3d(0.0, speed * std::cos(inclination_rad), speed * std::sin(inclination_rad));
}

ps::SimConfig truthConfig(const pt::Tai& epoch, const Eigen::Vector3d& r0,
                          const Eigen::Vector3d& v0, double duration_s) {
  ps::SimConfig c;
  c.scenario_name = "orbit-od-mc";
  c.spacecraft.name = "od-reference";
  c.spacecraft.mass_kg = 12.0;
  c.spacecraft.inertia_kgm2 = Eigen::Matrix3d::Identity() * 0.2;
  c.spacecraft.drag_area_m2 = 0.06;
  c.spacecraft.drag_cd = 2.2;
  c.spacecraft.srp_area_m2 = 0.06;
  c.spacecraft.srp_cr = 1.3;

  c.environment.gravity_degree = kTruthGravityDegree;
  c.environment.gravity_order = kTruthGravityDegree;
  c.environment.drag_enabled = true;
  c.environment.srp_enabled = true;
  c.environment.sun_third_body = true;
  c.environment.moon_third_body = true;
  // The campaign never reads the field, and IGRF is half the environment's cost
  // (tests/benchmark/sim_env_bench: 28x real time with it, 56x without).
  c.environment.magnetic_field = ps::MagneticModel::kNone;

  c.initial_state.epoch = epoch;
  c.initial_state.position = pm::Vec3<pmf::ECI>(r0);
  c.initial_state.velocity = pm::Vec3<pmf::ECI>(v0);
  c.initial_state.attitude = pm::Quat<pmf::Body, pmf::ECI>::Identity();
  c.initial_state.body_rate = pm::Vec3<pmf::Body>(Eigen::Vector3d::Zero());

  c.propagation.abs_tol = 1.0e-11;
  c.propagation.rel_tol = 1.0e-11;
  c.propagation.max_step_s = 60.0;
  c.propagation.duration_s = duration_s;
  // Only a hint: the arc is sampled through `runAt` at the cycle epochs.
  c.propagation.output_step_s = kFixPeriodS;
  return c;
}

/// One sample's worth of record. Written as one JSONL object per accepted
/// sample; `analysis/od` is the only consumer and the schema is documented in
/// `analysis/od/records.py`.
struct Record {
  double t_s{0.0};
  double pos_err_m{0.0};
  double vel_err_mps{0.0};
  double pos_sigma_m{0.0};  ///< sqrt(trace) of the position covariance block
  double vel_sigma_mps{0.0};
  double nees{0.0};
  double nis{0.0};
  bool nis_valid{false};
  bool solution_valid{false};
  bool fix_valid{false};
  bool fix_accepted{false};
  double age_s{0.0};
  std::uint32_t rejected_total{0};
  int refusal{0};
  const char* regime{"nominal"};
};

void writeRecord(std::ostream& out, int run, const std::string& scenario, const Record& r) {
  out << "{\"run\":" << run << ",\"scenario\":\"" << scenario << "\""
      << ",\"t_s\":" << r.t_s << ",\"regime\":\"" << r.regime << "\""
      << ",\"pos_err_m\":" << r.pos_err_m << ",\"vel_err_mps\":" << r.vel_err_mps
      << ",\"pos_sigma_m\":" << r.pos_sigma_m << ",\"vel_sigma_mps\":" << r.vel_sigma_mps
      << ",\"nees\":" << r.nees;
  if (r.nis_valid) {
    out << ",\"nis\":" << r.nis;
  }
  out << ",\"solution_valid\":" << (r.solution_valid ? 1 : 0)
      << ",\"fix_valid\":" << (r.fix_valid ? 1 : 0)
      << ",\"fix_accepted\":" << (r.fix_accepted ? 1 : 0) << ",\"age_s\":" << r.age_s
      << ",\"rejected_total\":" << r.rejected_total << ",\"refusal\":" << r.refusal << "}\n";
}

}  // namespace

// ---------------------------------------------------------------------------
// One run
// ---------------------------------------------------------------------------

namespace {

/// Apply @p scenario's armed events to @p rx at @p t_s, and report which regime
/// the sample belongs to. Returns the sigma-inflation factor the caller applies
/// to the reported fix (1.0 when no degradation is armed).
double applyFaults(const mco::Scenario& scenario, double t_s, psen::Gnss& rx,
                   const psen::JammingRegions* jamming, const Eigen::Vector3d& truth_ecef,
                   const char*& regime) {
  bool outage = false;
  bool jam = false;
  double sigma_scale = 1.0;
  Eigen::Vector3d spoof = Eigen::Vector3d::Zero();
  double clock = 0.0;
  regime = "nominal";

  for (const mco::FaultEvent& e : scenario.events) {
    if (!mco::active(e, t_s)) {
      continue;
    }
    regime = mco::kindName(e.kind);
    switch (e.kind) {
      case mco::FaultKind::kOutage:
        outage = true;
        break;
      case mco::FaultKind::kJam:
        jam = true;
        break;
      case mco::FaultKind::kSpoof: {
        // Ramp along the local vertical: a radial spoof is the direction a
        // range-domain attack moves the solution, and it is the direction the
        // filter's own dynamics constrains least over a short arc.
        const double frac = (e.ramp_s > 0.0) ? std::min(1.0, (t_s - e.start_s) / e.ramp_s) : 1.0;
        spoof += (e.magnitude * frac) * truth_ecef.normalized();
        break;
      }
      case mco::FaultKind::kClockJump:
        clock += e.magnitude;
        break;
      case mco::FaultKind::kRadiusJump: {
        // Push the reported position out to GEO radius along its own direction:
        // an implausible *radius*, which is what the §9.1 band checks.
        const double r_now = truth_ecef.norm();
        spoof += (e.magnitude - r_now) * truth_ecef.normalized();
        break;
      }
      case mco::FaultKind::kSigmaDegrade:
        sigma_scale *= e.magnitude;
        break;
    }
  }

  rx.setOutage(outage);
  rx.setJammingRegions(jam ? jamming : nullptr);
  rx.injectPositionOffset(pm::Vec3<pmf::ECEF>(spoof));
  rx.injectClockJump(clock);
  return sigma_scale;
}

struct RunResult {
  std::size_t samples{0};
  double worst_pos_err_m{0.0};
};

RunResult flyOne(int run, const mco::Scenario& scenario, double duration_s,
                 const pf::EopTable<24576>& eop_table, const pt::LeapSecondTable& leap,
                 const psen::JammingRegions* jamming, std::ostream& out, std::string* error) {
  RunResult result;
  // The scenario owns its cadence and its arc length; the campaign flags are the
  // default and the ceiling, not an override. A fast-cadence scenario capped at
  // ten minutes must stay ten minutes when `--duration-s` asks for seven days.
  const double cycle_s = scenario.cycle_period_s > 0.0 ? scenario.cycle_period_s : kFixPeriodS;
  if (scenario.max_duration_s > 0.0) {
    duration_s = std::min(duration_s, scenario.max_duration_s);
  }
  const std::uint64_t seed = 0x0D0DULL * 1000003ULL + static_cast<std::uint64_t>(run);
  polaris::random::SplitMix64 rng = polaris::random::streamRng(seed, 1);

  const pg::OrbitOdConfig cfg = filterConfig();
  Eigen::Vector3d r0;
  Eigen::Vector3d v0;
  circularState(kAltitudeM, kInclinationRad, cfg.mu_m3_per_s2, r0, v0);
  // Disperse the initial state. Run 0 is undispersed on purpose, so the campaign
  // always contains one reproducible nominal trajectory to read the others
  // against.
  if (run != 0) {
    for (int i = 0; i < 3; ++i) {
      r0(i) += kInitialPosSigmaM * rng.gaussian();
      v0(i) += kInitialVelSigmaMps * rng.gaussian();
    }
  }

  const pt::Tai epoch0 = campaignEpoch();
  ps::SimRunner runner;
  if (!runner.build(truthConfig(epoch0, r0, v0, duration_s),
                    ps::DataPaths::under(POLARIS_GOLDEN_DIR), error)) {
    return result;
  }

  psen::Gnss rx(receiverSpec(scenario.fix_latency_s), seed, 0x9E3779B97F4A7C15ULL);
  pg::OrbitOd filter(cfg);

  std::vector<double> times;
  for (double t = cycle_s; t <= duration_s + 1.0e-9; t += cycle_s) {
    times.push_back(t);
  }
  std::vector<ps::TrajectorySample> truth;
  if (!runner.runAt(times, truth, error)) {
    return result;
  }

  for (std::size_t i = 0; i < truth.size(); ++i) {
    const double t_s = times[i];
    const pt::Tai now = advance(epoch0, t_s);
    const Eigen::Vector3d r_true = truth[i].state.position.eigen();
    const Eigen::Vector3d v_true = truth[i].state.velocity.eigen();

    pf::EopValue eop;
    if (!eop_table.lookup(now, leap, eop)) {
      if (error != nullptr) {
        *error = "EOP table does not cover the campaign arc";
      }
      return result;
    }

    pm::Vec3<pmf::ECEF> r_ecef;
    pm::Vec3<pmf::ECEF> v_ecef;
    if (!pf::ecefStateFromEci(now, eop, pm::Vec3<pmf::ECI>(r_true), pm::Vec3<pmf::ECI>(v_true),
                              r_ecef, v_ecef)) {
      continue;
    }

    const char* regime = "nominal";
    const double sigma_scale = applyFaults(scenario, t_s, rx, jamming, r_ecef.eigen(), regime);

    psen::GnssInput in;
    in.position_m = r_ecef;
    in.velocity_m_s = v_ecef;
    const psen::GnssMeasurement m = rx.sample(now, in);

    Record rec;
    rec.t_s = t_s;
    rec.regime = regime;
    rec.fix_valid = m.valid;

    // --- The GNC cycle, in flight order --------------------------------------
    // Propagate to *now* first, unconditionally, then fold in whatever the
    // receiver has. This is the order the FSW runs in and it is not cosmetic:
    // it is what puts the filter's epoch ahead of an arriving fix, so a latent
    // fix is genuinely latent and the correction for it is exercised rather than
    // bypassed. Ingesting first would leave the filter sitting at the fix's own
    // epoch, where every subsequent fix looks forward and the latent branch is
    // dead code. It is also the quantity that matters: "where am I *now*" is
    // what pointing, pass planning and maneuver targeting all ask.
    const pg::OrbitOdRefusal pr = filter.propagate(now, eop_table, leap);
    rec.refusal = static_cast<int>(pr);

    if (m.valid) {
      pg::GnssFix fix;
      fix.time_tag = m.time_tag;
      fix.position_m = m.position_m;
      fix.velocity_m_s = m.velocity_m_s;
      // The receiver's *reported* sigmas, degraded when the scenario says so.
      // The filter must read them per fix; a campaign that always handed over
      // nominal sigmas would never check that it does.
      fix.position_sigma_h_m = m.position_sigma_h_m * sigma_scale;
      fix.position_sigma_v_m = m.position_sigma_v_m * sigma_scale;
      fix.velocity_sigma_m_s = m.velocity_sigma_m_s * sigma_scale;
      fix.velocity_valid = true;

      pg::OrbitOdResult res;
      rec.fix_accepted = filter.ingest(fix, eop, res);
      rec.refusal = static_cast<int>(res.refusal);
      if (res.position.accepted) {
        rec.nis = res.position.nis;
        rec.nis_valid = true;
      }
    }

    rec.solution_valid = filter.solutionValid();
    rec.age_s = filter.ageSeconds();
    rec.rejected_total = filter.rejectedCount();

    if (filter.isInitialised()) {
      rec.pos_err_m = (filter.position().eigen() - r_true).norm();
      rec.vel_err_mps = (filter.velocity().eigen() - v_true).norm();
      const pg::OrbitOd::Covariance& p = filter.covariance();
      rec.pos_sigma_m = std::sqrt(p.block<3, 3>(0, 0).trace());
      rec.vel_sigma_mps = std::sqrt(p.block<3, 3>(3, 3).trace());
      double nees = 0.0;
      if (filter.nees(pm::Vec3<pmf::ECI>(r_true), pm::Vec3<pmf::ECI>(v_true), nees)) {
        rec.nees = nees;
      }
      if (rec.solution_valid) {
        result.worst_pos_err_m = std::max(result.worst_pos_err_m, rec.pos_err_m);
      }
    }

    writeRecord(out, run, scenario.name, rec);
    result.samples += 1;
  }
  return result;
}

}  // namespace

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
  int first_run = 0;
  int runs = 1;
  double duration_s = 7.0 * 86400.0;
  std::string out_path;
  std::string only_scenario;

  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    const auto next = [&](double fallback) -> double {
      return (i + 1 < argc) ? std::atof(argv[++i]) : fallback;
    };
    if (a == "--first-run") {
      first_run = static_cast<int>(next(0));
    } else if (a == "--runs") {
      runs = static_cast<int>(next(1));
    } else if (a == "--duration-s") {
      duration_s = next(duration_s);
    } else if (a == "--out" && i + 1 < argc) {
      out_path = argv[++i];
    } else if (a == "--scenario" && i + 1 < argc) {
      only_scenario = argv[++i];
    } else if (a == "--help") {
      std::printf(
          "usage: polaris_orbit_od_mc [--first-run N] [--runs N] [--duration-s S]\n"
          "                           [--scenario NAME] [--out PATH]\n");
      return 0;
    }
  }

  std::ofstream file;
  if (!out_path.empty()) {
    file.open(out_path);
    if (!file) {
      std::fprintf(stderr, "orbit_od_mc: cannot open %s\n", out_path.c_str());
      return 1;
    }
  }
  std::ostream& out = out_path.empty() ? std::cout : file;

  // Data layers, loaded once for the whole shard: they are read-only and the
  // EOP table alone is a multi-megabyte parse.
  std::vector<pw::FinalsRow> rows;
  std::string error;
  if (!pw::parseFinals(std::string(POLARIS_GOLDEN_DIR) + "/finals.all.iau2000.txt", 1.0, 0.0, 0.0,
                       rows, &error)) {
    std::fprintf(stderr, "orbit_od_mc: EOP parse failed: %s\n", error.c_str());
    return 1;
  }
  auto eop_table = std::make_unique<pf::EopTable<24576>>();
  for (const pw::FinalsRow& r : rows) {
    if (!eop_table->addEntry({r.mjd_utc, r.dut1_s, r.xp_arcsec, r.yp_arcsec})) {
      break;  // table full; the campaign arc is far inside what fits
    }
  }
  const pt::LeapSecondTable leap = pt::LeapSecondTable::historical();

  psen::JammingRegions jamming;
  const std::string jam_path =
      std::string(POLARIS_CONFIG_DIR) + "/scenarios/jamming/eastern_europe.kml";
  const bool have_jamming = psen::JammingRegions::loadKmlFile(jam_path, jamming, nullptr);

  const std::vector<mco::Scenario> all = mco::scenarios();
  for (int run = first_run; run < first_run + runs; ++run) {
    for (const mco::Scenario& s : all) {
      if (!only_scenario.empty() && s.name != only_scenario) {
        continue;
      }
      if (s.name == "jamming" && !have_jamming) {
        std::fprintf(stderr, "orbit_od_mc: skipping 'jamming' — %s not readable\n",
                     jam_path.c_str());
        continue;
      }
      std::string run_error;
      const RunResult r = flyOne(run, s, duration_s, *eop_table, leap, &jamming, out, &run_error);
      if (!run_error.empty()) {
        std::fprintf(stderr, "orbit_od_mc: run %d scenario %s failed: %s\n", run, s.name.c_str(),
                     run_error.c_str());
        return 1;
      }
      std::fprintf(stderr, "run %d  %-16s  %zu samples  worst |dr| = %.2f m\n", run, s.name.c_str(),
                   r.samples, r.worst_pos_err_m);
      out.flush();
    }
  }
  return 0;
}
