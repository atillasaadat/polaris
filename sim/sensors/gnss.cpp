#include "sensors/gnss.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

namespace polaris::sim::sensors {
namespace {

/// Look a key up, defaulting to 0 (term disabled) when absent — the shared
/// convention across the sensor `fromParams` (sun_sensor.cpp, magnetometer.cpp).
double get(const std::map<std::string, double>& p, const std::string& key) {
  const auto it = p.find(key);
  return it == p.end() ? 0.0 : it->second;
}

/// √2 — a 2D horizontal DRMS is √(σ_e²+σ_n²) = σ·√2 for equal per-axis σ.
constexpr double kSqrt2 = 1.4142135623730951;
/// Vertical is poorer than horizontal for the same geometry; the usual VDOP/HDOP
/// ratio when a datasheet quotes only the horizontal figure.
constexpr double kVerticalToHorizontal = 1.5;
constexpr double kNsPerSecond = 1.0e9;

/// Local geodetic east/north/up basis at ECEF position @p r, using geocentric up
/// (r̂). Geocentric and geodetic up differ by <0.2° for an oblate Earth — well
/// inside the point of splitting horizontal from vertical error at all. At the
/// poles east is degenerate; fall back to the X axis there.
void enuBasis(const Eigen::Vector3d& r, Eigen::Vector3d& east, Eigen::Vector3d& north,
              Eigen::Vector3d& up) {
  up = r.normalized();
  east = Eigen::Vector3d::UnitZ().cross(up);
  if (east.norm() < 1.0e-9) {
    east = Eigen::Vector3d::UnitX();
  }
  east.normalize();
  north = up.cross(east);
}

}  // namespace

GnssSpec GnssSpec::fromParams(const std::map<std::string, double>& p) {
  GnssSpec s;

  const double h_rms = get(p, "horizontal_position_rms_m");
  s.position_sigma_h_m = (h_rms > 0.0) ? h_rms / kSqrt2 : 0.0;
  const double v_rms = get(p, "vertical_position_rms_m");
  s.position_sigma_v_m = (v_rms > 0.0) ? v_rms : s.position_sigma_h_m * kVerticalToHorizontal;

  s.velocity_sigma_m_s = get(p, "velocity_accuracy_m_s_rms");
  s.time_sigma_s = get(p, "time_accuracy_ns_rms") * 1.0e-9;
  // Correlated (common-mode) position error, Push 77.
  //
  // Expressed as the **fraction of the datasheet variance** that is correlated
  // rather than as an absolute figure, so the total accuracy stays exactly what
  // the datasheet quotes and only its *spectrum* changes:
  //
  //   sigma_white = sigma_total * sqrt(1 - f),   sigma_corr = sigma_total * sqrt(f)
  //
  // That is the honest way to add this to an existing entry. Taking a correlated
  // RMS on top of the datasheet would model a receiver worse than the one the
  // datasheet describes, and would confound "the error is bigger" with "the
  // error is correlated" — and only the second is what the onboard filter is
  // unprepared for. The receiver still *reports* the full datasheet sigma; it
  // cannot see which part of its own error is common-mode.
  //
  // An absent or zero fraction leaves the term off and the model exactly as it
  // was: an entry that has not been characterised does not silently acquire one.
  //
  // Refused rather than clamped outside [0, 1), and that asymmetry was the
  // defect: the flight filter refuses the same value at `configure()`
  // (`OrbitOdConfig::isValid`), so a sim that quietly clamped disagreed with it
  // on exactly the inputs a typo produces. The vehicle then flew a receiver with
  // no independent noise at all while the filter refused to configure, which
  // surfaces three layers away as "no orbit solution, no magnetic reference, no
  // attitude" — the silent-cascade class this model must not re-enter. At f = 1
  // the white part of R is identically zero and S is singular in the limit,
  // which is why the bound is half-open on the right.
  const double f = get(p, "correlated_position_fraction");
  if (!(f >= 0.0) || !(f < 1.0)) {
    throw std::invalid_argument(
        "GnssSpec::fromParams: correlated_position_fraction = " + std::to_string(f) +
        " is outside [0, 1). It is the fraction of the datasheet variance that is "
        "common-mode; at 1 the receiver has no independent noise at all. The flight "
        "filter refuses the same value, so clamping here would hide the disagreement "
        "rather than fix it.");
  }
  if (f > 0.0) {
    const double frac = f;
    s.position_corr_sigma_h_m = s.position_sigma_h_m * std::sqrt(frac);
    s.position_corr_sigma_v_m = s.position_sigma_v_m * std::sqrt(frac);
    s.position_sigma_h_m *= std::sqrt(1.0 - frac);
    s.position_sigma_v_m *= std::sqrt(1.0 - frac);
  }
  s.position_corr_tau_s = get(p, "correlated_position_tau_s");
  // A declared correlated fraction with no timescale is the same divergence in
  // the other variable: `advanceCorrelatedError` returns zero for a non-positive
  // tau, so the sim would fly a white receiver while the entry claims a
  // correlated one, and the flight filter (which demands tau >= 10*max_step_s)
  // would refuse to configure at all.
  if (f > 0.0 && !(s.position_corr_tau_s > 0.0)) {
    throw std::invalid_argument(
        "GnssSpec::fromParams: correlated_position_fraction = " + std::to_string(f) +
        " declares a common-mode error, but correlated_position_tau_s = " +
        std::to_string(s.position_corr_tau_s) +
        " is not positive. A correlated error needs the timescale it decorrelates over.");
  }
  // What the receiver reports: the datasheet total, i.e. both parts recombined.
  // Identical to the white sigmas when the term is off.
  s.position_reported_sigma_h_m = std::hypot(s.position_sigma_h_m, s.position_corr_sigma_h_m);
  s.position_reported_sigma_v_m = std::hypot(s.position_sigma_v_m, s.position_corr_sigma_v_m);

  s.max_rate_hz = get(p, "max_rate_hz");
  s.sample_period_s = (s.max_rate_hz > 0.0) ? 1.0 / s.max_rate_hz : 0.0;

  s.fix_latency_s = get(p, "fix_latency_s");
  s.fix_latency_jitter_s = get(p, "fix_latency_jitter_s");

  s.cold_start_s = get(p, "cold_start_s");
  s.hot_start_s = get(p, "hot_start_s");
  s.reacquisition_s = get(p, "reacquisition_s");
  return s;
}

/// Advance the correlated position-error state to @p gps_ns and return it as
/// east/north/up components [m] (Push 77).
///
/// Exact first-order Gauss-Markov discretisation rather than an Euler step:
/// `e <- e*phi + sigma*sqrt(1 - phi^2)*g` with `phi = exp(-dt/tau)`, which is
/// stationary for *any* step size. That matters here because the step between
/// fixes is whatever the caller's polling and the receiver's sample period make
/// it — an approximation valid only for `dt << tau` would quietly change the
/// error's variance when a scenario changed its fix cadence.
///
/// The first fix draws from the stationary distribution directly, so a scenario
/// does not sample a warm-up transient that a longer one would have forgotten.
Eigen::Vector3d Gnss::advanceCorrelatedError(std::int64_t gps_ns) {
  const double sigma_h = spec_.position_corr_sigma_h_m;
  const double sigma_v = spec_.position_corr_sigma_v_m;
  if (!(sigma_h > 0.0) && !(sigma_v > 0.0)) {
    return Eigen::Vector3d::Zero();
  }
  if (!(spec_.position_corr_tau_s > 0.0)) {
    return Eigen::Vector3d::Zero();
  }
  const Eigen::Vector3d sigma(sigma_h, sigma_h, sigma_v);

  if (!corr_seeded_) {
    for (int i = 0; i < 3; ++i) {
      corr_enu_(i) = sigma(i) * rng_.gaussian();
    }
    corr_seeded_ = true;
    corr_last_gps_ns_ = gps_ns;
    return corr_enu_;
  }

  const double dt = static_cast<double>(gps_ns - corr_last_gps_ns_) / kNsPerSecond;
  // A non-advancing or backward tag (a clock jump, a replayed epoch) leaves the
  // state where it is rather than driving exp() with a negative argument: the
  // error is a property of the sky, not of how the time tag was written.
  if (!(dt > 0.0)) {
    return corr_enu_;
  }
  const double phi = std::exp(-dt / spec_.position_corr_tau_s);
  const double driving = std::sqrt(std::max(0.0, 1.0 - phi * phi));
  for (int i = 0; i < 3; ++i) {
    corr_enu_(i) = phi * corr_enu_(i) + sigma(i) * driving * rng_.gaussian();
  }
  corr_last_gps_ns_ = gps_ns;
  return corr_enu_;
}

GnssMeasurement Gnss::sample(const time::Tai& epoch, const GnssInput& input) {
  const time::Gps gps = time::toGps(epoch);
  const std::int64_t gps_ns = gps.nanosecondsSinceEpoch();

  // Sample-period gating: within one period of the last genuine fix, the receiver
  // has not produced a new solution — return the previous one, flagged stale.
  if (has_sampled_ && spec_.sample_period_s > 0.0) {
    const double elapsed =
        static_cast<double>(gps_ns - last_fix_time_.nanosecondsSinceEpoch()) / kNsPerSecond;
    if (elapsed >= 0.0 && elapsed < spec_.sample_period_s) {
      GnssMeasurement repeated = last_fix_;
      repeated.fresh = false;
      return repeated;
    }
  }

  // Geographic jamming is a loss of fix gated by the sub-satellite point, so it
  // folds into the same "outage" signal as a commanded dropout: same
  // invalidation, same reacquisition delay on exit.
  const std::string* jam_region =
      (jamming_ != nullptr) ? jamming_->jammedRegion(input.position_m) : nullptr;
  const bool outage = fault_outage_ || jam_region != nullptr;

  // Acquisition / reacquisition timing. A cold start withholds fixes for the
  // time-to-first-fix; the falling edge of an outage arms the reacquisition delay.
  if (!has_sampled_) {
    valid_from_gps_ns_ = gps_ns + static_cast<std::int64_t>(spec_.cold_start_s * kNsPerSecond);
  } else if (prev_outage_ && !outage) {
    valid_from_gps_ns_ = gps_ns + static_cast<std::int64_t>(spec_.reacquisition_s * kNsPerSecond);
  }
  prev_outage_ = outage;

  GnssMeasurement m;
  m.position_sigma_h_m = spec_.position_reported_sigma_h_m;
  m.position_sigma_v_m = spec_.position_reported_sigma_v_m;
  m.velocity_sigma_m_s = spec_.velocity_sigma_m_s;
  m.time_sigma_s = spec_.time_sigma_s;

  // Position error, split horizontal vs vertical in the local geodetic frame.
  // With noise disabled the fix is truth-exact (the spoof offset still applies —
  // it is a fault, not measurement noise).
  Eigen::Vector3d pos_err = Eigen::Vector3d::Zero();
  Eigen::Vector3d vel_err = Eigen::Vector3d::Zero();
  double clock_noise = 0.0;
  if (spec_.noise_enabled) {
    Eigen::Vector3d east;
    Eigen::Vector3d north;
    Eigen::Vector3d up;
    enuBasis(input.position_m.eigen(), east, north, up);
    pos_err = spec_.position_sigma_h_m * rng_.gaussian() * east +
              spec_.position_sigma_h_m * rng_.gaussian() * north +
              spec_.position_sigma_v_m * rng_.gaussian() * up;
    // Correlated (common-mode) component, advanced to this fix and added in the
    // same local basis (Push 77). Drawn *after* the white terms so that enabling
    // it does not renumber the white draws — a scenario's white noise stays
    // bit-identical whether or not the correlated term is on, which is what
    // makes the two comparable on one seed.
    const Eigen::Vector3d corr = advanceCorrelatedError(gps_ns);
    pos_err += corr.x() * east + corr.y() * north + corr.z() * up;
    vel_err = Eigen::Vector3d(spec_.velocity_sigma_m_s * rng_.gaussian(),
                              spec_.velocity_sigma_m_s * rng_.gaussian(),
                              spec_.velocity_sigma_m_s * rng_.gaussian());
    clock_noise = spec_.time_sigma_s * rng_.gaussian();
  }
  m.position_m =
      math::Vec3<math::frames::ECEF>(input.position_m.eigen() + pos_err + fault_pos_offset_);
  m.velocity_m_s = math::Vec3<math::frames::ECEF>(input.velocity_m_s.eigen() + vel_err);

  // Receiver-clock bias on the time tag (noise + any injected jump).
  m.clock_bias_s = clock_noise + fault_clock_jump_s_;
  m.time_tag = time::Gps::fromNanosecondsSinceEpoch(
      gps_ns + static_cast<std::int64_t>(std::llround(m.clock_bias_s * kNsPerSecond)));

  m.fresh = true;
  m.valid = !outage && gps_ns >= valid_from_gps_ns_;
  m.jammed = jam_region != nullptr;
  if (jam_region != nullptr) {
    m.jamming_region = *jam_region;
  }

  has_sampled_ = true;
  last_fix_time_ = gps;

  if (!(spec_.fix_latency_s > 0.0)) {
    last_fix_ = m;
    return m;
  }

  // --- Fix latency: the solution just computed is not the one delivered ------
  // `m` describes the vehicle at `gps_ns` and enters the delay line; what leaves
  // it is the newest solution that has been in there at least `fix_latency_s`.
  // The tag stays at the *measurement* epoch — a receiver stamps when it
  // measured, not when it finished talking — which is precisely what makes the
  // latency correctable downstream instead of merely wrong.
  if (pending_count_ == kMaxPending) {
    for (std::size_t i = 1; i < kMaxPending; ++i) {
      pending_[i - 1] = pending_[i];
      pending_epoch_ns_[i - 1] = pending_epoch_ns_[i];
    }
    pending_count_ -= 1;
    pending_dropped_ += 1;
  }
  // Each fix carries its own delivery epoch: the latency plus a per-fix jitter
  // draw, clamped at zero. The tag `m` carries stays the measurement epoch.
  double latency_s = spec_.fix_latency_s;
  if (spec_.fix_latency_jitter_s > 0.0) {
    latency_s = std::max(0.0, latency_s + spec_.fix_latency_jitter_s * rng_.gaussian());
  }
  pending_[pending_count_] = m;
  pending_epoch_ns_[pending_count_] =
      gps_ns + static_cast<std::int64_t>(std::llround(latency_s * kNsPerSecond));
  pending_count_ += 1;

  std::size_t ready = 0;
  while (ready < pending_count_ && pending_epoch_ns_[ready] <= gps_ns) {
    ready += 1;
  }
  if (ready == 0) {
    // Nothing has cleared the receiver yet. Report the sigmas and flags of a
    // receiver with no solution rather than a stale or truth-exact position:
    // this is the same "no fix available" state a cold start is in, and a
    // consumer must treat it the same way.
    GnssMeasurement none;
    none.position_sigma_h_m = spec_.position_reported_sigma_h_m;
    none.position_sigma_v_m = spec_.position_reported_sigma_v_m;
    none.velocity_sigma_m_s = spec_.velocity_sigma_m_s;
    none.time_sigma_s = spec_.time_sigma_s;
    none.time_tag = gps;
    none.fresh = true;
    none.valid = false;
    last_fix_ = none;
    return none;
  }

  const GnssMeasurement delivered = pending_[ready - 1];
  const std::size_t remaining = pending_count_ - ready;
  for (std::size_t i = 0; i < remaining; ++i) {
    pending_[i] = pending_[ready + i];
    pending_epoch_ns_[i] = pending_epoch_ns_[ready + i];
  }
  pending_count_ = remaining;
  // `last_fix_` is what the receiver last *reported*, not what it last computed:
  // the sample-period repeat path above replays it, and replaying a solution
  // still sitting in the delay line would deliver it early.
  last_fix_ = delivered;
  return delivered;
}

}  // namespace polaris::sim::sensors
