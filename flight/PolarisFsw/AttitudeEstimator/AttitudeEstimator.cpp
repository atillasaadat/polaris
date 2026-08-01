// ======================================================================
// \title  AttitudeEstimator.cpp
// \brief  Coarse attitude estimator component (§8.1, §10; REQ-ADET-002,
//         REQ-ADET-003, REQ-ADET-004)
// ======================================================================

#include "flight/PolarisFsw/AttitudeEstimator/AttitudeEstimator.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>

#include "environment/igrf_iaga.hpp"
#include "frames/eci_ecef.hpp"
#include "frames/eop.hpp"
#include "Fw/Log/LogString.hpp"
#include "time/utc.hpp"

namespace flight {

namespace {

constexpr I64 kNsPerSecond = 1000000000LL;
constexpr I64 kNsPerMicrosecond = 1000LL;

namespace pm = polaris::math;
using ECEF = pm::frames::ECEF;
using ECI = pm::frames::ECI;
using Body = pm::frames::Body;

//! Number of tuning parameters read from ParameterDb.
constexpr FwSizeType kParamCount = 12;

//! Read a `Vec3F64` telemetry/port array member into a raw 3-vector.
Eigen::Vector3d toEigen(const Vec3F64& v) {
  return Eigen::Vector3d(v[0], v[1], v[2]);
}

//! Pack a raw 3-vector into the port/telemetry array type.
Vec3F64 toVec3F64(const Eigen::Vector3d& v) {
  Vec3F64 out;
  out[0] = v.x();
  out[1] = v.y();
  out[2] = v.z();
  return out;
}

//! True if @p timeTagNs is within @p maxAgeS of @p nowTaiNs. A measurement from
//! the future is stale too — a time tag ahead of the master clock is a fault,
//! not freshness (§9.1).
bool fresh(I64 nowTaiNs, I64 timeTagNs, F64 maxAgeS) {
  if (!(maxAgeS > 0.0)) {
    return false;
  }
  const I64 delta = nowTaiNs - timeTagNs;
  const I64 limit = static_cast<I64>(maxAgeS * static_cast<F64>(kNsPerSecond));
  return delta >= -limit && delta <= limit;
}

//! Map the lib estimation mode onto the telemetry/port enum.
EstimationMode::T toEstimationMode(polaris::state::EstimationMode m) {
  switch (m) {
    case polaris::state::EstimationMode::Coarse:
      return EstimationMode::COARSE;
    case polaris::state::EstimationMode::Fine:
      return EstimationMode::FINE;
    default:
      return EstimationMode::INVALID;
  }
}

//! The worse of two source grades — a reference chain is only as good as its
//! weakest link, so a coarse EOP degrades a precise ephemeris.
TableGrade::T worseGrade(TableGrade::T a, TableGrade::T b) {
  return (static_cast<U8>(a) <= static_cast<U8>(b)) ? a : b;
}

}  // namespace

// ----------------------------------------------------------------------
// Component construction and destruction
// ----------------------------------------------------------------------

AttitudeEstimator ::AttitudeEstimator(const char* const compName)
    : AttitudeEstimatorComponentBase(compName),
      // Both start inert: a default-constructed config fails isValid() and a
      // default coefficient set fails validIgrfCoefficients(), so nothing runs
      // on invented values before configuration arrives.
      estimator_(polaris::gnc::CoarseAttitudeConfig{}),
      igrf_(polaris::environment::IgrfCoefficients{}),
      leap_(polaris::time::LeapSecondTable::historical()) {}

AttitudeEstimator ::~AttitudeEstimator() {}

// ----------------------------------------------------------------------
// Configuration
// ----------------------------------------------------------------------

bool AttitudeEstimator ::configureIgrf(const char* igrfPath, double decimalYear) {
  FW_ASSERT(igrfPath != nullptr);
  std::strncpy(this->igrf_path_, igrfPath, kMaxPathLength - 1);
  this->igrf_path_[kMaxPathLength - 1] = '\0';

  if (!std::isfinite(decimalYear)) {
    const Fw::LogStringArg reason("mission epoch for the IGRF snapshot is not a finite year");
    this->log_WARNING_HI_IgrfLoadFailed(reason);
    return false;
  }

  polaris::environment::IgrfCoefficients coefficients;
  char why[polaris::environment::kIgrfMaxReasonLength] = {};
  if (!polaris::environment::loadIgrfIaga(this->igrf_path_, decimalYear, coefficients, why,
                                          sizeof(why))) {
    const Fw::LogStringArg reason(why);
    this->log_WARNING_HI_IgrfLoadFailed(reason);
    return false;
  }

  this->igrf_ = polaris::environment::IgrfField(coefficients);
  if (!this->igrf_.good()) {
    const Fw::LogStringArg reason("IGRF evaluator rejected the loaded coefficients");
    this->log_WARNING_HI_IgrfLoadFailed(reason);
    return false;
  }
  this->igrf_epoch_year_ = coefficients.epoch_year;
  this->igrf_valid_until_year_ = coefficients.valid_until_year;
  this->igrf_stale_flagged_ = false;
  this->log_ACTIVITY_HI_IgrfLoaded(coefficients.epoch_year, coefficients.valid_until_year,
                                   static_cast<U32>(coefficients.degree));
  return true;
}

bool AttitudeEstimator ::refreshConfig() {
  // Each get is its own statement: `valid` is an out-parameter, so reading it in
  // the same expression that calls the getter would be unsequenced.
  Fw::ParamValid valids[kParamCount];
  F64 values[kParamCount];
  values[0] = this->paramGet_SigmaSunWhiteRad(valids[0]);
  values[1] = this->paramGet_SigmaSunSysRad(valids[1]);
  values[2] = this->paramGet_SigmaMagWhiteRad(valids[2]);
  values[3] = this->paramGet_SigmaMagSysRad(valids[3]);
  values[4] = this->paramGet_GyroArw(valids[4]);
  values[5] = this->paramGet_MinSinAngle(valids[5]);
  values[6] = this->paramGet_TriadGain(valids[6]);
  values[7] = this->paramGet_MaxCoastSec(valids[7]);
  values[8] = this->paramGet_MaxDtSec(valids[8]);
  values[9] = this->paramGet_MaxMeasAgeSec(valids[9]);
  values[10] = this->paramGet_MinPositionRadiusM(valids[10]);
  values[11] = this->paramGet_MaxPositionRadiusM(valids[11]);

  static const char* const kNames[kParamCount] = {
      "SigmaSunWhiteRad", "SigmaSunSysRad", "SigmaMagWhiteRad",   "SigmaMagSysRad",
      "GyroArw",          "MinSinAngle",    "TriadGain",          "MaxCoastSec",
      "MaxDtSec",         "MaxMeasAgeSec",  "MinPositionRadiusM", "MaxPositionRadiusM"};

  for (FwSizeType i = 0; i < kParamCount; ++i) {
    if (valids[i] != Fw::ParamValid::VALID || !std::isfinite(values[i])) {
      char detail[80];
      (void)std::snprintf(detail, sizeof(detail), "%s missing or not finite in ParameterDb",
                          kNames[i]);
      this->failConfig(detail);
      return false;
    }
  }

  polaris::gnc::CoarseAttitudeConfig cfg;
  cfg.sigma_sun_white_rad = values[0];
  cfg.sigma_sun_sys_rad = values[1];
  cfg.sigma_mag_white_rad = values[2];
  cfg.sigma_mag_sys_rad = values[3];
  cfg.gyro_arw = values[4];
  cfg.min_sin_angle = values[5];
  cfg.triad_gain = values[6];
  cfg.max_coast_s = values[7];
  cfg.max_dt_s = values[8];

  if (!cfg.isValid()) {
    this->failConfig("tuning values are out of range (see CoarseAttitudeConfig::isValid)");
    return false;
  }
  // A staleness window wider than the coast horizon would keep handing the
  // estimator measurements it has already declared itself too old to trust.
  if (!(values[9] > 0.0) || values[9] > cfg.max_coast_s) {
    this->failConfig("MaxMeasAgeSec must be positive and <= MaxCoastSec");
    return false;
  }
  if (!(values[10] > 0.0) || !(values[11] > values[10])) {
    this->failConfig("position radius gate must satisfy 0 < Min < Max");
    return false;
  }

  // Rebuilding drops any solution in flight, which is the honest behaviour: the
  // covariance the old solution carries was computed under the old budget.
  this->noteAttitudeLost();  // the rebuild drops the solution; say so
  this->estimator_ = polaris::gnc::CoarseAttitudeEstimator(cfg);
  this->max_meas_age_s_ = values[9];
  this->min_position_radius_m_ = values[10];
  this->max_position_radius_m_ = values[11];
  this->config_invalid_flagged_ = false;
  return true;
}

void AttitudeEstimator ::failConfig(const char* detail) {
  // Going inert loses the solution as surely as a coast expiry does, and a
  // consumer watching the validity flag deserves the same edge either way.
  this->noteAttitudeLost();
  // Edge-gated: one alert on entering the unconfigured state, not one per cycle.
  if (!this->config_invalid_flagged_) {
    const Fw::LogStringArg arg(detail);
    this->log_WARNING_HI_ConfigInvalid(arg);
    this->config_invalid_flagged_ = true;
  }
  this->estimator_ = polaris::gnc::CoarseAttitudeEstimator(polaris::gnc::CoarseAttitudeConfig{});
  this->max_meas_age_s_ = 0.0;
}

void AttitudeEstimator ::noteAttitudeLost() {
  if (!this->attitude_valid_) {
    return;
  }
  this->log_WARNING_HI_AttitudeLost(this->last_age_s_);
  this->attitude_valid_ = false;
}

void AttitudeEstimator ::parameterUpdated(FwPrmIdType id) {
  // Any change re-reads the whole set: the values are validated together
  // (CoarseAttitudeConfig::isValid), so one at a time means nothing.
  this->params_dirty_ = true;
}

// ----------------------------------------------------------------------
// Measurement latching
// ----------------------------------------------------------------------

void AttitudeEstimator ::imuIn_handler(FwIndexType portNum, const ImuMeas& meas) {
  this->imu_[portNum] = meas;
}

void AttitudeEstimator ::sunSensorIn_handler(FwIndexType portNum, const SunSensorMeas& meas) {
  this->sun_[portNum] = meas;
}

void AttitudeEstimator ::magnetometerIn_handler(FwIndexType portNum, const MagnetometerMeas& meas) {
  this->mag_[portNum] = meas;
}

void AttitudeEstimator ::gnssIn_handler(FwIndexType portNum, const GnssMeas& meas) {
  this->gnss_[portNum] = meas;
}

void AttitudeEstimator ::starTrackerIn_handler(FwIndexType portNum, const StarTrackerMeas& meas) {
  // TODO(fine mode): the MEKF push fuses this. Counted here so the interface is
  // exercised and a tracker delivering solutions is visible in telemetry; the
  // coarse solution stays tracker-independent on purpose (§10).
  if (meas.get_valid()) {
    ++this->star_tracker_count_;
  }
}

const ImuMeas* AttitudeEstimator ::selectImu(I64 nowTaiNs) const {
  for (FwIndexType i = 0; i < NUM_IMUIN_INPUT_PORTS; ++i) {
    const ImuMeas& m = this->imu_[i];
    if (m.get_valid() && m.get_intervalSec() > 0.0 &&
        fresh(nowTaiNs, m.get_timeTagNs(), this->max_meas_age_s_)) {
      return &this->imu_[i];
    }
  }
  return nullptr;
}

const SunSensorMeas* AttitudeEstimator ::selectSunSensor(I64 nowTaiNs) const {
  for (FwIndexType i = 0; i < NUM_SUNSENSORIN_INPUT_PORTS; ++i) {
    const SunSensorMeas& m = this->sun_[i];
    // No sun in view is a normal condition (eclipse, or the unit facing away),
    // not a fault — it simply excludes this unit from the pair this cycle.
    if (m.get_valid() && m.get_sunPresent() &&
        fresh(nowTaiNs, m.get_timeTagNs(), this->max_meas_age_s_)) {
      return &this->sun_[i];
    }
  }
  return nullptr;
}

const MagnetometerMeas* AttitudeEstimator ::selectMagnetometer(I64 nowTaiNs) const {
  for (FwIndexType i = 0; i < NUM_MAGNETOMETERIN_INPUT_PORTS; ++i) {
    const MagnetometerMeas& m = this->mag_[i];
    if (m.get_valid() && fresh(nowTaiNs, m.get_timeTagNs(), this->max_meas_age_s_)) {
      return &this->mag_[i];
    }
  }
  return nullptr;
}

const GnssMeas* AttitudeEstimator ::selectGnss(I64 nowTaiNs) const {
  for (FwIndexType i = 0; i < NUM_GNSSIN_INPUT_PORTS; ++i) {
    const GnssMeas& m = this->gnss_[i];
    if (!m.get_valid()) {
      continue;
    }
    // §9.1 range gate. A GNSS fix is wire data from outside the FSW, and a
    // non-finite or absurd position does not fail loudly downstream — it
    // poisons *both* references (a NaN radius through the field model, a NaN
    // sun direction through r_eci) while every validity flag still reads true.
    // So it is checked here, with the other per-source gates.
    const Eigen::Vector3d r = toEigen(m.get_posEcefM());
    if (!r.allFinite()) {
      continue;
    }
    const double radius = r.norm();
    if (radius < this->min_position_radius_m_ || radius > this->max_position_radius_m_) {
      continue;
    }
    // The receiver stamps GPS time; TAI = GPS + 19 s on ingest (§3.2).
    const polaris::time::Tai tag =
        polaris::time::toTai(polaris::time::Gps::fromNanosecondsSinceEpoch(m.get_timeTagGpsNs()));
    if (fresh(nowTaiNs, tag.nanosecondsSinceEpoch(), this->max_meas_age_s_)) {
      return &this->gnss_[i];
    }
  }
  return nullptr;
}

// ----------------------------------------------------------------------
// Estimation cycle
// ----------------------------------------------------------------------

void AttitudeEstimator ::run_handler(FwIndexType portNum, U32 context) {
  const I64 nowNs = this->currentTaiNs();

  if (this->params_dirty_) {
    this->params_dirty_ = false;
    (void)this->refreshConfig();  // emits ConfigInvalid and leaves us inert
  }
  if (!this->estimator_.isConfigured()) {
    // No tuning, no estimate. Telemetering INVALID every cycle is what tells the
    // operator (and FDIR) that the vehicle has no attitude solution.
    this->state_ = polaris::state::EstimatedState{};
    this->state_.epoch = polaris::time::Tai::fromNanosecondsSinceEpoch(nowNs);
    this->tlmWrite_EstMode(EstimationMode::INVALID);
    return;
  }

  // A stuck or backwards clock is refused here rather than inside the estimator,
  // so the refusal is counted and telemetered instead of looking like a quiet
  // no-op cycle.
  if (this->have_epoch_ && nowNs <= this->last_epoch_tai_ns_) {
    ++this->cycles_refused_;
    this->tlmWrite_CyclesRefused(this->cycles_refused_);
    return;
  }
  this->last_epoch_tai_ns_ = nowNs;
  this->have_epoch_ = true;

  const polaris::time::Tai epoch = polaris::time::Tai::fromNanosecondsSinceEpoch(nowNs);
  polaris::gnc::CoarseAttitudeInput in;
  in.epoch = epoch;

  // --- Gyro -----------------------------------------------------------------
  const ImuMeas* const imu = this->selectImu(nowNs);
  if (imu != nullptr) {
    const Eigen::Vector3d rate = toEigen(imu->get_deltaAngleRad()) / imu->get_intervalSec();
    in.gyro = pm::Vec3<Body>(rate);
    in.gyro_valid = in.gyro.isFinite();
  }
  // gyro_bias stays zero: the coarse mode does not estimate bias (the MEKF
  // does), and substituting an unestimated bias would be inventing one.

  // --- Position and the ECEF->ECI rotation ----------------------------------
  // Both inertial references need the rotation; the magnetic one also needs the
  // position, which today comes straight off GNSS. There is no onboard orbit
  // propagator yet (§8.3), so a GNSS outage costs the magnetic pair — flagged
  // by PositionUnavailable rather than papered over with a guessed position.
  pm::Vec3<ECEF> r_ecef;
  const GnssMeas* const gnss = this->selectGnss(nowNs);
  const bool have_position = gnss != nullptr;
  if (have_position) {
    r_ecef = pm::Vec3<ECEF>(toEigen(gnss->get_posEcefM()));
  }

  TableGrade::T eop_grade = TableGrade::UNAVAILABLE;
  pm::Quat<ECI, ECEF> q_eci_ecef;
  bool have_rotation = false;
  if (this->isConnected_getEopAt_OutputPort(0)) {
    EopSample sample;
    if (this->getEopAt_out(0, nowNs, sample)) {
      const polaris::frames::EopValue eop{sample.get_ut1MinusTai(), sample.get_xpArcsec(),
                                          sample.get_ypArcsec()};
      have_rotation = polaris::frames::eciFromEcef(epoch, eop, q_eci_ecef);
      eop_grade = sample.get_grade();
    }
  }

  // --- Sun reference --------------------------------------------------------
  pm::Vec3<ECI> sun_ref;
  bool have_sun_ref = false;
  TableGrade::T ephem_grade = TableGrade::UNAVAILABLE;
  if (this->isConnected_getBodyPosition_OutputPort(0)) {
    PosEciMeters sunPos;
    if (this->getBodyPosition_out(0, OnboardBody::SUN, nowNs, sunPos)) {
      pm::Vec3<ECI> to_sun(sunPos.get_x(), sunPos.get_y(), sunPos.get_z());
      if (have_position && have_rotation) {
        // Geocentric to spacecraft-centric. Worth ~4e-5 rad at LEO — far below
        // the coarse budget, but it costs one subtraction and the rotation is
        // already in hand for the field.
        to_sun = to_sun - q_eci_ecef.rotate(r_ecef);
      }
      have_sun_ref = to_sun.normalized(sun_ref);
      ephem_grade = sunPos.get_grade();
    }
  }
  // A query that could not answer leaves its domain UNAVAILABLE, so the reported
  // grade never claims PRECISE on the strength of the *other* domain alone.
  const TableGrade::T grade = worseGrade(eop_grade, ephem_grade);

  // --- Magnetic reference ---------------------------------------------------
  pm::Vec3<ECI> mag_ref;
  bool have_mag_ref = false;
  if (have_position && have_rotation && this->igrf_.good()) {
    double year = 0.0;
    pm::Vec3<ECEF> b_ecef;
    if (polaris::time::decimalYear(epoch, this->leap_, year)) {
      // Refuse a field the loaded snapshot cannot honestly model. The horizon is
      // the one the IAGA file itself implies — the next tabulated epoch, or five
      // years past the last — not a fixed gap from the base epoch, which would
      // expire a snapshot taken near the end of a grid interval almost
      // immediately after launch.
      if (year > this->igrf_valid_until_year_) {
        if (!this->igrf_stale_flagged_) {
          this->log_WARNING_HI_MagneticReferenceStale(year, this->igrf_valid_until_year_);
          this->igrf_stale_flagged_ = true;
        }
      } else {
        this->igrf_stale_flagged_ = false;  // re-arm on the epoch check, not on a good eval
        if (this->igrf_.field(r_ecef, year, b_ecef)) {
          // B is a vector field: it rotates with the same rotation, no translation.
          mag_ref = q_eci_ecef.rotate(b_ecef);
          have_mag_ref = mag_ref.isFinite();
        }
      }
    }
  }

  // Edge-gated position alert: no position means no field model, hence no TRIAD.
  if (!have_position && !this->position_unavailable_flagged_) {
    this->log_WARNING_LO_PositionUnavailable();
    this->position_unavailable_flagged_ = true;
  } else if (have_position) {
    this->position_unavailable_flagged_ = false;
  }
  this->noteReferenceGrade(TableDomain::EPHEMERIS, ephem_grade, this->last_ephem_grade_,
                           this->have_ephem_grade_);
  this->noteReferenceGrade(TableDomain::EOP, eop_grade, this->last_eop_grade_,
                           this->have_eop_grade_);

  // --- Measurement pairs ----------------------------------------------------
  const SunSensorMeas* const sun = this->selectSunSensor(nowNs);
  if (sun != nullptr && have_sun_ref) {
    in.sun_body = pm::Vec3<Body>(toEigen(sun->get_dirBody()));
    in.sun_ref = sun_ref;
    in.sun_valid = in.sun_body.isFinite();
  }
  const MagnetometerMeas* const magnetometer = this->selectMagnetometer(nowNs);
  if (magnetometer != nullptr && have_mag_ref) {
    in.mag_body = pm::Vec3<Body>(toEigen(magnetometer->get_fieldTesla()));
    in.mag_ref = mag_ref;
    in.mag_valid = in.mag_body.isFinite();
  }

  // --- One estimation cycle -------------------------------------------------
  polaris::gnc::CoarseAttitudeOutput out;
  const bool valid_attitude = this->estimator_.update(in, out);

  // A refused cycle returns a default-constructed product. With a valid finite
  // gyro a cycle that actually ran always publishes a rate, so this identifies
  // the internal-refusal case (non-finite result) without guessing; a refusal
  // while the gyro is invalid is indistinguishable from a normal no-solution
  // cycle and is not counted.
  if (!valid_attitude && in.gyro_valid && !out.rate_valid) {
    ++this->cycles_refused_;
  }
  if (in.sun_valid && in.mag_valid) {
    if (out.triad_applied) {
      ++this->triad_accepted_;
    } else {
      ++this->triad_rejected_;
    }
  }

  // Acquisition/loss edges (REQ-ADET-004: mode transitions surfaced to FDIR).
  this->last_age_s_ = out.age_s;
  if (out.attitude_valid) {
    if (!this->attitude_valid_) {
      this->log_ACTIVITY_HI_AttitudeAcquired(out.age_s, out.covariance.trace());
    }
    this->attitude_valid_ = true;
  } else {
    this->noteAttitudeLost();
  }

  this->publish(out, epoch);

  this->tlmWrite_GyroValid(in.gyro_valid);
  this->tlmWrite_SunValid(in.sun_valid);
  this->tlmWrite_MagValid(in.mag_valid);
  this->tlmWrite_PositionValid(have_position);
  this->tlmWrite_TriadAccepted(this->triad_accepted_);
  this->tlmWrite_TriadRejected(this->triad_rejected_);
  this->tlmWrite_CyclesRefused(this->cycles_refused_);
  this->tlmWrite_StarTrackerCount(this->star_tracker_count_);
  this->tlmWrite_RefGrade(grade);
}

void AttitudeEstimator ::publish(const polaris::gnc::CoarseAttitudeOutput& out,
                                 const polaris::time::Tai& epoch) {
  // The canonical §8.0 mapping lives in lib/gnc, so the component cannot invent
  // its own idea of what "valid" or "Coarse" means.
  polaris::gnc::writeToEstimatedState(out, epoch, this->state_);

  const polaris::math::Quaternion& q = this->state_.attitude.core();
  QuatF64 quat;
  quat[0] = q.w();
  quat[1] = q.x();
  quat[2] = q.y();
  quat[3] = q.z();
  const Vec3F64 rate = toVec3F64(this->state_.body_rate.eigen());
  const Vec3F64 cov_diag = toVec3F64(Eigen::Vector3d(out.covariance.diagonal()));

  AttitudeEstimate estimate;
  estimate.set_epochTaiNs(this->state_.epoch.nanosecondsSinceEpoch());
  estimate.set_qBodyEci(quat);
  estimate.set_bodyRateRadps(rate);
  estimate.set_attCovDiagRad2(cov_diag);
  estimate.set_ageSec(out.age_s);
  estimate.set_mode(toEstimationMode(this->state_.mode));
  estimate.set_attitudeValid(this->state_.valid.attitude);
  estimate.set_rateValid(this->state_.valid.body_rate);
  if (this->isConnected_estimateOut_OutputPort(0)) {
    this->estimateOut_out(0, estimate);
  }

  this->tlmWrite_EstMode(toEstimationMode(this->state_.mode));
  this->tlmWrite_AttQuat(quat);
  this->tlmWrite_BodyRate(rate);
  // A refused or coasting-expired cycle has no covariance to report; zero would
  // draw as a perfect solution on a strip chart, so say "no value" instead.
  this->tlmWrite_AttCovTrace(out.attitude_valid ? out.covariance.trace()
                                                : std::numeric_limits<double>::quiet_NaN());
  this->tlmWrite_SolutionAge(out.age_s);
}

void AttitudeEstimator ::noteReferenceGrade(TableDomain::T domain, TableGrade::T grade,
                                            TableGrade::T& last, bool& have) {
  if (!have) {
    last = grade;
    have = true;
    // A first cycle already on the coarse fallback is worth one alert: the
    // operator has no earlier transition to have seen it from.
    if (grade != TableGrade::PRECISE) {
      this->log_WARNING_HI_ReferenceDegraded(domain, grade);
    }
    return;
  }
  if (grade == last) {
    return;
  }
  if (static_cast<U8>(grade) < static_cast<U8>(last)) {
    this->log_WARNING_HI_ReferenceDegraded(domain, grade);
  } else {
    this->log_ACTIVITY_HI_ReferenceRecovered(domain, grade);
  }
  last = grade;
}

// ----------------------------------------------------------------------
// Command handler implementations
// ----------------------------------------------------------------------

void AttitudeEstimator ::RESET_ESTIMATOR_cmdHandler(FwOpcodeType opCode, U32 cmdSeq) {
  this->estimator_.reset();
  this->attitude_valid_ = false;
  this->last_age_s_ = 0.0;
  this->have_epoch_ = false;
  // A reset means "tell me everything again": every edge-gated alert is re-armed,
  // not just the configuration one, so a still-broken vehicle re-reports each
  // fault rather than staying quiet because it already mentioned it once.
  this->config_invalid_flagged_ = false;
  this->position_unavailable_flagged_ = false;
  this->igrf_stale_flagged_ = false;
  this->have_ephem_grade_ = false;
  this->have_eop_grade_ = false;
  // Re-read the tuning too: a reset is also the recovery path after a config fix.
  this->params_dirty_ = true;
  this->log_ACTIVITY_HI_EstimatorReset();
  this->cmdResponse_out(opCode, cmdSeq, Fw::CmdResponse::OK);
}

// ----------------------------------------------------------------------
// Helpers
// ----------------------------------------------------------------------

I64 AttitudeEstimator ::currentTaiNs() {
  const Fw::Time now = this->getTime();
  return static_cast<I64>(now.getSeconds()) * kNsPerSecond +
         static_cast<I64>(now.getUSeconds()) * kNsPerMicrosecond;
}

}  // namespace flight
