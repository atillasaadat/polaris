/// @file
/// @brief Coarse SS+MAG+IMU attitude estimator implementation (design doc
/// §8.1; REQ-ADET-002). See coarse_attitude.hpp for conventions and references.

#include "gnc/coarse_attitude.hpp"

#include <cmath>
#include <Eigen/Core>

#include "gnc/triad.hpp"
#include "math/quaternion.hpp"

namespace polaris::gnc {

namespace {

/// Rotation angle below which an eigenaxis is not extractable in double
/// precision; the rotation is then a no-op anyway [rad]. Used both for the
/// propagation increment and for the blend step.
constexpr double kMinRotationAngleRad = 1.0e-12;

/// Force exact symmetry after an update. The algebra is symmetric, but the
/// products that build it are not bitwise symmetric, and an asymmetric
/// covariance breaks the Cholesky factorisations downstream consumers do.
void symmetrise(Eigen::Matrix3d& p) {
  p = 0.5 * (p + p.transpose().eval());
}

}  // namespace

bool CoarseAttitudeConfig::isValid() const {
  const bool finite = std::isfinite(sigma_sun_white_rad) && std::isfinite(sigma_sun_sys_rad) &&
                      std::isfinite(sigma_mag_white_rad) && std::isfinite(sigma_mag_sys_rad) &&
                      std::isfinite(gyro_arw) && std::isfinite(min_sin_angle) &&
                      std::isfinite(triad_gain) && std::isfinite(max_coast_s) &&
                      std::isfinite(max_dt_s);
  return finite && sigma_sun_white_rad > 0.0 && sigma_mag_white_rad > 0.0 &&
         sigma_sun_sys_rad >= 0.0 && sigma_mag_sys_rad >= 0.0 && gyro_arw >= 0.0 &&
         min_sin_angle > 0.0 && min_sin_angle < 1.0 && triad_gain > 0.0 && triad_gain <= 1.0 &&
         max_coast_s > 0.0 && max_dt_s > 0.0;
}

CoarseAttitudeEstimator::CoarseAttitudeEstimator(const CoarseAttitudeConfig& config)
    : cfg_(config), configured_(config.isValid()) {}

void CoarseAttitudeEstimator::reset() {
  attitude_ = math::Quaternion::Identity();
  cov_random_.setZero();
  cov_systematic_.setZero();
  last_epoch_ = time::Tai{};
  age_s_ = 0.0;
  initialised_ = false;
  have_epoch_ = false;
}

bool CoarseAttitudeEstimator::propagate(const Eigen::Vector3d& rate, double dt_s) {
  // Quaternion kinematics q̇ = ½·Ω(ω)·q integrated in closed form over the step
  // at constant ω (Markley & Crassidis §3.1): the body-frame increment is a
  // rotation of |ω|·dt about ω̂, left-multiplying the Body ← ECI attitude.
  Eigen::Matrix3d phi = Eigen::Matrix3d::Identity();
  const double rate_norm = rate.norm();
  const double angle = rate_norm * dt_s;
  if (angle > kMinRotationAngleRad) {
    const math::Quaternion dq = math::Quaternion::FromAxisAngle(rate / rate_norm, angle);
    math::Quaternion propagated = dq * attitude_;
    if (!propagated.normalize()) {
      return false;  // degenerate quaternion: the solution is gone, say so
    }
    attitude_ = propagated.canonical();
    phi = dq.toRotationMatrix();
  }

  // Both covariance parts rotate with the body frame; only the reducible part
  // grows, by the gyro angle random walk over the step (Markley & Crassidis
  // §7.1). Gyro *bias* uncertainty is not modelled here — the coarse mode does
  // not estimate bias; it consumes whatever bias the caller supplies.
  cov_random_ = phi * cov_random_ * phi.transpose();
  cov_random_ += (cfg_.gyro_arw * cfg_.gyro_arw * dt_s) * Eigen::Matrix3d::Identity();
  cov_systematic_ = phi * cov_systematic_ * phi.transpose();
  symmetrise(cov_random_);
  symmetrise(cov_systematic_);
  return true;
}

bool CoarseAttitudeEstimator::update(const CoarseAttitudeInput& in, CoarseAttitudeOutput& out) {
  out = CoarseAttitudeOutput{};
  if (!configured_) {
    return false;
  }

  // --- Body rate ----------------------------------------------------------
  Eigen::Vector3d omega = Eigen::Vector3d::Zero();
  bool rate_ok = false;
  if (in.gyro_valid && in.gyro.isFinite() && in.gyro_bias.isFinite()) {
    omega = in.gyro.eigen() - in.gyro_bias.eigen();
    rate_ok = omega.allFinite();
  }

  // --- Time step ----------------------------------------------------------
  // Strictly increasing epochs only. A backwards clock is obvious; a *stuck*
  // one is the dangerous case, because re-running the update at the same epoch
  // would fold the same measurement in twice and shrink the covariance on
  // information already used. Both are refused without destroying the solution.
  double dt_s = 0.0;
  if (have_epoch_) {
    dt_s = (in.epoch - last_epoch_).seconds();
    if (dt_s <= 0.0) {
      return false;
    }
    age_s_ += dt_s;
  }
  last_epoch_ = in.epoch;
  have_epoch_ = true;

  // --- Gyro propagation ---------------------------------------------------
  // A step longer than `max_dt_s` is a dropout, not a slow cycle: extrapolating
  // a stale rate across it would inject an unbounded error, so the solution is
  // held and the coast timeout is left to invalidate it.
  if (initialised_ && dt_s > 0.0) {
    if (rate_ok && dt_s <= cfg_.max_dt_s) {
      if (!propagate(omega, dt_s)) {
        reset();
        out = CoarseAttitudeOutput{};
        return false;
      }
    } else {
      // Held attitude. The random-walk term is a *lower bound* on the true
      // growth without a gyro; `max_coast_s` is the real guard here.
      cov_random_ += (cfg_.gyro_arw * cfg_.gyro_arw * dt_s) * Eigen::Matrix3d::Identity();
    }
  }

  // Past the coast horizon the solution is declared lost rather than left to
  // drift: it stops being published, and the next TRIAD re-acquires whole
  // instead of blending against an attitude that no longer means anything. The
  // covariance is *not* cleared — it keeps telling the truth about the drift.
  if (initialised_ && age_s_ > cfg_.max_coast_s) {
    initialised_ = false;
  }

  // --- TRIAD update -------------------------------------------------------
  if (in.sun_valid && in.mag_valid) {
    // TRIAD is solved on the *white* sigmas alone, so its covariance is the
    // reducible part; the systematic part is evaluated on the same geometry
    // (the covariance is linear in the variances, so the two sum to the full
    // one-shot uncertainty) and kept aside as a floor.
    TriadInput ti{};
    ti.primary.body = in.sun_body;  // sun leads: TRIAD fits the primary exactly
    ti.primary.reference = in.sun_ref;
    ti.primary.sigma_rad = cfg_.sigma_sun_white_rad;
    ti.secondary.body = in.mag_body;
    ti.secondary.reference = in.mag_ref;
    ti.secondary.sigma_rad = cfg_.sigma_mag_white_rad;
    ti.min_sin_angle = cfg_.min_sin_angle;

    TriadSolution ts{};
    Eigen::Matrix3d cov_sys;
    if (triad(ti, ts) && triadCovariance(in.sun_body, in.mag_body, cfg_.sigma_sun_sys_rad,
                                         cfg_.sigma_mag_sys_rad, cfg_.min_sin_angle, cov_sys)) {
      bool applied = true;
      if (!initialised_) {
        // Cold start / re-acquisition: the propagated attitude carries no
        // information, so the gain is bypassed and TRIAD is taken whole.
        attitude_ = ts.attitude.core();
        cov_random_ = ts.covariance;
        initialised_ = true;
      } else {
        // Fixed-gain complementary blend along the eigenaxis of the
        // propagated-to-TRIAD error rotation δq = q_triad ⊗ q_prop⁻¹ (a
        // body-frame rotation, by the JPL homomorphism A(a⊗b) = A(a)A(b)).
        // Taking the canonical representative picks the short way round.
        const math::Quaternion dq = (ts.attitude.core() * attitude_.inverse()).canonical();
        const double vec_norm = dq.vec().norm();
        const double error_angle = 2.0 * std::atan2(vec_norm, dq.scalar());
        if (error_angle > kMinRotationAngleRad) {
          const math::Quaternion step =
              math::Quaternion::FromAxisAngle(dq.vec() / vec_norm, cfg_.triad_gain * error_angle);
          math::Quaternion blended = step * attitude_;
          if (blended.normalize()) {
            attitude_ = blended.canonical();
          } else {
            // The attitude could not be moved, so the covariance must not be
            // shrunk either — state and uncertainty stay consistent.
            applied = false;
          }
        }
        if (applied) {
          // Covariance of the same fixed-gain blend of two independent
          // estimates: δθ⁺ = (1−k)·δθ⁻ + k·δθ_triad. Only the reducible part is
          // blended; the systematic floor below is re-applied whole.
          const double k = cfg_.triad_gain;
          cov_random_ = (1.0 - k) * (1.0 - k) * cov_random_ + (k * k) * ts.covariance;
        }
      }
      if (applied) {
        cov_systematic_ = cov_sys;
        symmetrise(cov_random_);
        symmetrise(cov_systematic_);
        age_s_ = 0.0;
        out.triad_applied = true;
      }
    }
  }

  // --- Output -------------------------------------------------------------
  // Nothing is published until it is known finite: a NaN reaching a consumer
  // that gates on `attitude_valid` alone would be worse than no answer.
  const Eigen::Matrix3d published = cov_random_ + cov_systematic_;
  if (!attitude_.isFinite() || !published.allFinite()) {
    reset();  // a NaN can never recover on its own; drop back to cold start
    out = CoarseAttitudeOutput{};
    return false;
  }

  out.body_rate = math::Vec3<math::frames::Body>(omega);
  out.rate_valid = rate_ok;
  out.age_s = age_s_;
  out.attitude = math::Quat<math::frames::Body, math::frames::ECI>(attitude_);
  out.covariance = published;
  out.attitude_valid = initialised_ && age_s_ <= cfg_.max_coast_s;
  return out.attitude_valid;
}

void writeToEstimatedState(const CoarseAttitudeOutput& out, const time::Tai& epoch,
                           state::EstimatedState& state) {
  state.epoch = epoch;
  state.attitude = out.attitude;
  state.body_rate = out.body_rate;
  state.valid.attitude = out.attitude_valid;
  state.valid.body_rate = out.rate_valid;
  state.valid.covariance = out.attitude_valid;
  if (out.attitude_valid) {
    state.covariance.block<3, 3>(state::ErrorState::kAttitude, state::ErrorState::kAttitude) =
        out.covariance;
  }
  state.mode = out.attitude_valid ? state::EstimationMode::Coarse : state::EstimationMode::Invalid;
}

}  // namespace polaris::gnc
