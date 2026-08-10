/// @file Cross-validation of the **onboard** orbit propagator against GMAT
/// (design doc §8.3; REQ-VV-002).
///
/// `gmat_propagation_golden_test.cpp` compares the *truth sim's* propagator
/// against the same fixture. This file does the other half: the deliberately
/// coarse force model the flight software flies (`gnc::OrbitOd`, two-body + J2 +
/// exponential drag) against the same GMAT reference states, and it splits the
/// question into the two different things a coarse model has to answer.
///
/// **1. At matched fidelity, agreement is a correctness question.** The
/// `two_body` and `zonal_j2` fixture cases are exactly the force models the
/// onboard propagator implements — a point mass, and a point mass plus the
/// degree-2 zonal about the true pole. There is no model difference left to hide
/// behind, so the two propagators must agree to the same bands the truth sim is
/// held to, and a disagreement is a defect in `orbit_od.cpp`. This is the check
/// that would have caught an ECI-evaluated (pole-misaligned) J2, a J2 sign, a
/// wrong reference radius, or a mis-scaled coefficient — none of which any
/// conservation property can see, because a wrong-but-smooth field is still
/// perfectly conservative.
///
/// **2. Against full fidelity, divergence is a characterisation, not a
/// failure.** Run the same propagator against the 8×8-geopotential LEO cases and
/// the gap is the model truncation the design deliberately bought (§8.3), which
/// is the design input for the filter's process noise `q_a`. It is asserted here
/// only as an upper bound — a live gate, so a force-model regression fails CI
/// rather than silently invalidating the tuning — and the *number* is reported
/// through `RecordProperty`. The tighter, denser measurement that actually sizes
/// `q_a` (at the coast horizon, against the truth sim, which GMAT validates and
/// which can be sampled anywhere) lives in `tests/unit/orbit_od_test.cpp`; the
/// fixture's LEO cases are sampled only every 600–700 s, well past the horizon.
///
/// Both use the same committed fixture and the same committed IERS EOP file the
/// truth-sim comparison does, so the pole the onboard J2 is referenced to is the
/// pole the reference states were generated under.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <Eigen/Core>
#include <fstream>
#include <memory>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

#include "constants/constants.hpp"
#include "frames/eop.hpp"
#include "gnc/orbit_od.hpp"
#include "math/frames.hpp"
#include "math/typed_vector.hpp"
#include "time/leap_seconds.hpp"
#include "time/timescales.hpp"
#include "time/utc.hpp"
#include "world/eop_file.hpp"

namespace {

namespace pc = polaris::constants;
namespace pf = polaris::frames;
namespace pg = polaris::gnc;
namespace pm = polaris::math;
namespace pt = polaris::time;
using nlohmann::json;

/// Above the committed finals.all row count (1973→~2027 daily ≈ 20k).
/// Heap-allocated in the test; the flight table is sized to its upload window.
constexpr std::size_t kEopCapacity = 20480;

/// Propagation chunk [s] and RK4 sub-step [s] for this harness. The sub-step is
/// what sets the integrator's own truncation, and it must sit far below the
/// model residual being measured or the two are indistinguishable. Verified by
/// refinement rather than by the error formula: quartering it to 0.5 s moves the
/// worst `zonal_j2` residual by 0.3 mm and the `two_body` one by 0.03 mm, so at
/// h = 1 s the integrator contributes a few percent of the smallest residual
/// reported here and none of the conclusions. The chunk is bounded by
/// `kMaxSubsteps · max_step_s`, which `OrbitOdConfig::isValid` enforces.
constexpr double kChunkS = 60.0;
constexpr double kSubStepS = 1.0;

Eigen::Vector3d vec3(const json& node) {
  return Eigen::Vector3d(node[0].get<double>(), node[1].get<double>(), node[2].get<double>());
}

/// The onboard force-model configuration under test — the **consistent set**,
/// taken whole from the shared registry rather than assembled from whichever
/// constant was nearest.
///
/// A geopotential model is one fit of GM, the reference radius and the harmonics
/// together, so when J2 is on all three come from `constants::gravity`; with J2
/// off the vehicle is on a pure point mass and WGS84's GM is the right one. That
/// split is not a convenience — it mirrors what the reference tool does, and the
/// ablations that established it are worth recording because each one moved a
/// residual by more than the band it sits in:
///
///  - pairing `kJ2` with `wgs84::kSemiMajorAxis` (0.7 m larger than the radius
///    the coefficient was solved with) cost **9.1 cm → 7.0 cm** on `zonal_j2`;
///  - using `wgs84::kGM` on `zonal_j2` cost **7.0 cm → 1.0 cm**, because GMAT
///    propagates that case on the EGM96 potential file's own GM and not on the
///    `Earth.Mu = 398600.4418` its script sets;
///  - the same GM swap applied to the file-less `two_body` case *degrades* it
///    from 3.5 mm to 6.1 cm, which is what proves the split is the fixture's
///    behaviour and not a fudge that happened to help one case.
///
/// Transcribing any of these values here would be the copy that drifts.
pg::OrbitOdConfig onboardConfig(bool j2_enabled) {
  pg::OrbitOdConfig cfg;
  cfg.mu_m3_per_s2 = j2_enabled ? pc::gravity::kGM : pc::wgs84::kGM;
  cfg.zonal_j2 = j2_enabled ? pc::gravity::kJ2 : 0.0;
  cfg.reference_radius_m = pc::gravity::kReferenceRadius;
  // The fixture cases all fly `fm.Drag = None`, so drag is off here too — this
  // is a *matched fidelity* comparison, and leaving a force on that the
  // reference does not model would be measuring our own drag model against zero.
  cfg.drag_ballistic_coeff_m2_per_kg = 0.0;
  // Filter-side tuning is irrelevant to a propagation-only harness but must pass
  // isValid(). The coast horizon is set out of the way deliberately: this test
  // exercises the propagator, not the validity policy, and a horizon that
  // expired mid-arc would drop the solution being measured.
  cfg.accel_psd_m2_per_s3 = 1.0e-6;
  cfg.position_nis_gate = 16.27;
  cfg.velocity_nis_gate = 16.27;
  cfg.max_coast_s = 1.0e9;
  cfg.max_dt_s = kChunkS;
  cfg.max_step_s = kSubStepS;
  cfg.min_radius_m = 6.4e6;
  cfg.max_radius_m = 5.0e7;
  return cfg;
}

/// The config the filter actually flies: the compiled-in 8x8 EGM2008 field.
///
/// Separate from `onboardConfig` because that one exists to reproduce a
/// *reference tool's* fidelity — GMAT running two-body or J2-only — and matching
/// a reference is the whole point of the cases that use it. This one is for the
/// truncation characterisation, where the question is how far the flown model
/// drifts from full fidelity, so it has to be the flown model.
pg::OrbitOdConfig onboardHarmonicConfig() {
  pg::OrbitOdConfig cfg = onboardConfig(/*j2_enabled=*/true);
  cfg.zonal_j2 = 0.0;  // the harmonic field contains the degree-2 zonal
  cfg.geopotential_degree = pg::kGeopotentialMaxDegree;
  cfg.geopotential_order = pg::kGeopotentialMaxDegree;
  return cfg;
}

/// Parse a fixture `epoch_utc` through the same path the sim uses, so a fixture
/// epoch cannot be interpreted two ways.
pt::Tai epochOf(const json& c, const pt::LeapSecondTable& leap) {
  const std::string text = c.at("epoch_utc").get<std::string>();
  int year = 0;
  unsigned month = 0;
  unsigned day = 0;
  unsigned hour = 0;
  unsigned minute = 0;
  double second = 0.0;
  EXPECT_EQ(std::sscanf(text.c_str(), "%4d-%2u-%2uT%2u:%2u:%lfZ", &year, &month, &day, &hour,
                        &minute, &second),
            6)
      << text;
  pt::UtcDateTime utc;
  utc.year = year;
  utc.month = month;
  utc.day = day;
  utc.hour = hour;
  utc.minute = minute;
  utc.second = static_cast<unsigned>(second);
  return pt::taiFromUtc(utc, leap);
}

pt::Tai advance(const pt::Tai& base, double seconds) {
  return base + pt::Duration::fromSecondsF(seconds);
}

/// Load the committed IERS finals file into an EOP table, through the shared
/// parser (`sim/world/eop_file`) rather than a transcription of its column spec.
std::unique_ptr<pf::EopTable<kEopCapacity>> loadEop() {
  std::vector<polaris::sim::world::FinalsRow> rows;
  std::string error;
  const std::string path = std::string(GOLDEN_DIR) + "/finals.all.iau2000.txt";
  if (!polaris::sim::world::parseFinals(path, 1.0, 0.0, 0.0, rows, &error)) {
    ADD_FAILURE() << error;
    return nullptr;
  }
  auto table = std::make_unique<pf::EopTable<kEopCapacity>>();
  for (const polaris::sim::world::FinalsRow& r : rows) {
    if (!table->addEntry({r.mjd_utc, r.dut1_s, r.xp_arcsec, r.yp_arcsec})) {
      ADD_FAILURE() << "EOP row rejected at mjd " << r.mjd_utc;
      return nullptr;
    }
  }
  return table;
}

/// Seed covariance for the propagation harness. Any positive-definite matrix
/// works — nothing here reads the covariance — but it must be a real covariance,
/// because `initialize` refuses an indefinite one on purpose.
pg::OrbitOd::Covariance seedCovariance() {
  pg::OrbitOd::Covariance cov = pg::OrbitOd::Covariance::Zero();
  cov.block<3, 3>(pg::OrbitOd::kPosition, pg::OrbitOd::kPosition) =
      Eigen::Matrix3d::Identity() * 4.0;
  cov.block<3, 3>(pg::OrbitOd::kVelocity, pg::OrbitOd::kVelocity) =
      Eigen::Matrix3d::Identity() * 1.0e-4;
  return cov;
}

/// Propagate @p filter from its current epoch to `epoch0 + t_s`, in chunks the
/// config accepts. Returns false on any refusal.
bool walkTo(pg::OrbitOd& filter, const pt::Tai& epoch0, double t_s,
            const pf::EopTable<kEopCapacity>& eop, const pt::LeapSecondTable& leap) {
  const pt::Tai target = advance(epoch0, t_s);
  while (filter.epoch() < target) {
    const double remaining = (target - filter.epoch()).seconds();
    const pt::Tai next = (remaining <= kChunkS) ? target : advance(filter.epoch(), kChunkS);
    const pg::OrbitOdRefusal r = filter.propagate(next, eop, leap);
    if (r != pg::OrbitOdRefusal::kNone) {
      ADD_FAILURE() << "propagate refused with code " << static_cast<int>(r);
      return false;
    }
  }
  return true;
}

const json& caseNamed(const json& cases, const char* name) {
  for (const json& c : cases) {
    if (c.at("name").get<std::string>() == name) {
      return c;
    }
  }
  ADD_FAILURE() << "fixture has no case '" << name << "'";
  return cases[0];
}

json loadFixture() {
  std::ifstream file(std::string(GOLDEN_DIR) + "/gmat_propagation.json");
  EXPECT_TRUE(file.good()) << "cannot open gmat_propagation.json";
  json fixture;
  file >> fixture;
  return fixture;
}

}  // namespace

/// The onboard propagator against GMAT on the two cases it models exactly.
///
/// Bands are the fixture's own — the same numbers the truth sim is held to. That
/// is the point: at matched fidelity the onboard model is not a coarser model,
/// it is the *same* model, and it has to land in the same place.
TEST(OrbitOdGolden, OnboardPropagatorMatchesGmatAtMatchedFidelity) {
  RecordProperty("verifies", "REQ-VV-002");

  const json fixture = loadFixture();
  const json& cases = fixture.at("cases");
  const pt::LeapSecondTable leap = pt::LeapSecondTable::historical();
  const auto eop = loadEop();
  ASSERT_NE(eop, nullptr);

  for (const char* name : {"two_body", "zonal_j2"}) {
    SCOPED_TRACE(std::string("case: ") + name);
    const json& c = caseNamed(cases, name);
    const bool j2 = std::string(name) == "zonal_j2";

    const pt::Tai epoch0 = epochOf(c, leap);
    pg::OrbitOd filter(onboardConfig(j2));
    ASSERT_TRUE(filter.isConfigured());
    ASSERT_EQ(filter.initialize(
                  epoch0, pm::Vec3<pm::frames::ECI>(vec3(c.at("initial_state").at("position_m"))),
                  pm::Vec3<pm::frames::ECI>(vec3(c.at("initial_state").at("velocity_m_s"))),
                  seedCovariance()),
              pg::OrbitOdRefusal::kNone);

    const double position_tolerance = c.at("position_tolerance_m").get<double>();
    const double velocity_tolerance = c.at("velocity_tolerance_m_s").get<double>();
    double worst_position = 0.0;
    double worst_velocity = 0.0;

    for (const json& sample : c.at("samples")) {
      const double t_s = sample.at("t_s").get<double>();
      ASSERT_TRUE(walkTo(filter, epoch0, t_s, *eop, leap));

      const double dr = (filter.position().eigen() - vec3(sample.at("position_m"))).norm();
      const double dv = (filter.velocity().eigen() - vec3(sample.at("velocity_m_s"))).norm();
      worst_position = std::max(worst_position, dr);
      worst_velocity = std::max(worst_velocity, dv);
      EXPECT_LT(dr, position_tolerance) << "t=" << t_s << " s";
      EXPECT_LT(dv, velocity_tolerance) << "t=" << t_s << " s";
    }

    RecordProperty(std::string(name) + "_worst_position_m", std::to_string(worst_position));
    RecordProperty(std::string(name) + "_worst_velocity_m_s", std::to_string(worst_velocity));

    // A band far looser than the achieved error has stopped testing anything.
    // Same 100x guard, and the same coupling, as the truth-sim comparison: if a
    // fixture band is ever widened, re-check this together with it.
    EXPECT_GT(worst_position * 100.0, position_tolerance)
        << "case '" << name << "' achieves " << worst_position << " m against a "
        << position_tolerance << " m band — the tolerance is too loose to be meaningful";
  }
}

/// The onboard harmonic field against GMAT's, at matched degree — and against
/// the closed-form J2 model it replaced, on the same arcs.
///
/// **This test used to measure a truncation and no longer can**, and saying so
/// is the point. The fixture's LEO cases fly `gravity_degree: 8`, which was
/// comfortably above a closed-form-J2 onboard model: the divergence was 14.3 m
/// (iss_leo, 699 s) and 12.9 m (sso_leo, 600 s), and it was asserted as an upper
/// bound. Now that the filter itself carries 8x8, the same comparison is
/// **matched fidelity**, and a bound on it would be a bound on nothing.
///
/// What it measures instead is stronger. GMAT runs EGM96 to degree 8 and Polaris
/// runs EGM2008 to degree 8, by completely different code, and they agree to
/// ~1e-2 m over 600-700 s — which is the EGM96/EGM2008 coefficient separation
/// itself, not a propagator difference. That is an independent-tool validation
/// of the flight evaluator's recursion, its de-normalized coefficient table and
/// its ECEF frame handling, all at once, and it is precisely the check that the
/// unit-level comparison against `sim/world/gravity_field` cannot supply, since
/// a defect shared between the sim and the flight side would pass that one.
///
/// The J2-only model is still flown here, over the identical arcs, so the
/// improvement stays anchored to an independent tool rather than to our own
/// truth sim. The remaining *truncation* characterisation — the one that sizes
/// `q_a`, against a genuinely higher-fidelity 32x32 reference — lives in
/// `tests/unit/orbit_od_test.cpp`, which is where the density of samples the
/// coast horizon needs is available.
TEST(OrbitOdGolden, HarmonicFieldMatchesGmatAtMatchedDegreeAndBeatsTheJ2Model) {
  RecordProperty("verifies", "REQ-VV-002");
  RecordProperty("verifies", "REQ-ODP-005");

  const json fixture = loadFixture();
  const json& cases = fixture.at("cases");
  const pt::LeapSecondTable leap = pt::LeapSecondTable::historical();
  const auto eop = loadEop();
  ASSERT_NE(eop, nullptr);

  // Measured at 0.012 m (iss_leo, 699 s) and 0.004 m (sso_leo, 600 s). The bound
  // carries ~8x over the worst of them — tight enough that any real regression
  // (a lost degree, a pole or frame error, a wrong reference radius, a
  // de-normalization mistake) is orders above it, loose enough that it is not
  // tracking the epoch dependence of the EGM96/EGM2008 separation. The measured
  // numbers are in design doc §8.3.
  constexpr double kMaxMatchedFidelityM = 0.1;

  // The closed-form J2 model, on the same arcs against the same reference:
  // 14.3 m and 12.9 m when it flew. Asserted as a *floor* rather than a band —
  // the claim is that the harmonic field beats it by a wide margin, and pinning
  // the old model's error tightly would mean maintaining a model no longer flown.
  constexpr double kMinJ2ModelM = 5.0;

  for (const char* name : {"iss_leo", "sso_leo"}) {
    SCOPED_TRACE(std::string("case: ") + name);
    const json& c = caseNamed(cases, name);
    const pt::Tai epoch0 = epochOf(c, leap);
    const pm::Vec3<pm::frames::ECI> r0(vec3(c.at("initial_state").at("position_m")));
    const pm::Vec3<pm::frames::ECI> v0(vec3(c.at("initial_state").at("velocity_m_s")));

    // The first sample past t = 0 is the shortest arc the fixture offers, i.e.
    // the closest thing it has to a coast horizon.
    const json& first = c.at("samples")[1];
    const double t_s = first.at("t_s").get<double>();
    ASSERT_GT(t_s, 0.0);

    const auto coastTo = [&](const pg::OrbitOdConfig& model) -> double {
      pg::OrbitOd filter(model);
      EXPECT_EQ(filter.initialize(epoch0, r0, v0, seedCovariance()), pg::OrbitOdRefusal::kNone);
      EXPECT_TRUE(walkTo(filter, epoch0, t_s, *eop, leap));
      return (filter.position().eigen() - vec3(first.at("position_m"))).norm();
    };

    const std::string suffix = "_m_at_" + std::to_string(static_cast<int>(t_s)) + "s";
    const double harmonic = coastTo(onboardHarmonicConfig());
    const double j2_only = coastTo(onboardConfig(/*j2_enabled=*/true));
    RecordProperty(std::string(name) + "_harmonic" + suffix, std::to_string(harmonic));
    RecordProperty(std::string(name) + "_j2_only" + suffix, std::to_string(j2_only));

    EXPECT_GT(harmonic, 0.0) << name
                             << ": exact agreement would mean the two propagators are not "
                                "actually independent";
    EXPECT_LT(harmonic, kMaxMatchedFidelityM)
        << name << ": the onboard 8x8 field diverged from GMAT's by " << harmonic << " m over "
        << t_s << " s at matched degree — that is a defect in the field, not a truncation";
    EXPECT_GT(j2_only, kMinJ2ModelM)
        << name << ": the closed-form J2 model only diverged " << j2_only
        << " m, so this fixture no longer discriminates between the two gravity models";
    EXPECT_LT(harmonic, 0.1 * j2_only)
        << name << ": the 8x8 field (" << harmonic << " m) does not beat the J2 model (" << j2_only
        << " m) by the margin the flight-side coefficient table was added for";
  }
}
