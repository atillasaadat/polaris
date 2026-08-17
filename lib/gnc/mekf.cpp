/// @file
/// @brief Fine-mode attitude MEKF implementation (design doc §8.1;
/// REQ-ADET-001, REQ-ADET-004). See mekf.hpp for the error-state convention,
/// the Φ/Qd derivation, and the references.

#include "gnc/mekf.hpp"

#include <cmath>
#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/Geometry>

#include "math/quaternion.hpp"

namespace polaris::gnc {

namespace {

/// Rotation angle below which an eigenaxis is not extractable in double
/// precision; the rotation is then a no-op anyway [rad]. Same threshold the
/// coarse estimator uses, for the same reason.
constexpr double kMinRotationAngleRad = 1.0e-12;

/// Cross-product (skew) matrix `[v×]`, so `[v×]u = v × u`.
Eigen::Matrix3d skew(const Eigen::Vector3d& v) {
  Eigen::Matrix3d m;
  m << 0.0, -v.z(), v.y(), v.z(), 0.0, -v.x(), -v.y(), v.x(), 0.0;
  return m;
}

/// Normalise @p v into @p out; false if it is too short or not finite.
bool unit(const Eigen::Vector3d& v, Eigen::Vector3d& out) {
  if (!v.allFinite()) {
    return false;
  }
  const double n = v.norm();
  if (!(n > 1.0e-12)) {
    return false;
  }
  out = v / n;
  return true;
}

/// Force exact symmetry. The algebra is symmetric but the products that build
/// it are not bitwise symmetric, and an asymmetric covariance breaks the
/// Cholesky factorisations downstream consumers do.
void symmetrise(Mekf::Covariance& p) {
  p = 0.5 * (p + p.transpose().eval());
}

/// Rotation vector of @p q (axis × angle) [rad], taken the short way round.
Eigen::Vector3d rotationVector(const math::Quaternion& q) {
  const math::Quaternion c = q.canonical();
  const double vec_norm = c.vec().norm();
  if (!(vec_norm > kMinRotationAngleRad)) {
    return Eigen::Vector3d::Zero();
  }
  return (2.0 * std::atan2(vec_norm, c.scalar()) / vec_norm) * c.vec();
}

}  // namespace

bool MekfConfig::isValid() const {
  const bool finite = std::isfinite(arw_rad_per_sqrt_s) &&
                      std::isfinite(rrw_rad_per_s_per_sqrt_s) && std::isfinite(nis_gate) &&
                      std::isfinite(attitude_nis_gate) && std::isfinite(max_coast_s) &&
                      std::isfinite(max_dt_s);
  return finite && arw_rad_per_sqrt_s > 0.0 && rrw_rad_per_s_per_sqrt_s >= 0.0 && nis_gate > 0.0 &&
         attitude_nis_gate > 0.0 && max_coast_s > 0.0 && max_dt_s > 0.0;
}

Mekf::Mekf(const MekfConfig& config) : cfg_(config), configured_(config.isValid()) {}

void Mekf::dropSolution() {
  attitude_ = math::Quaternion::Identity();
  bias_.setZero();
  rate_.setZero();
  p_.setZero();
  last_epoch_ = time::Tai{};
  age_s_ = 0.0;
  initialised_ = false;
  rate_valid_ = false;
}

bool Mekf::retune(const MekfConfig& config) {
  // TB 20-03 item (g) / TP §9.3: the tuning changes, the navigation data does
  // not. A bad upload leaves everything, the old configuration included.
  if (!config.isValid()) {
    return false;
  }
  cfg_ = config;
  configured_ = true;
  return true;
}

bool Mekf::reinitializeCovariance(double sigma_att_rad, double sigma_bias_rad_s) {
  // TB 20-03 item (f) / TP §9.2: the covariance is re-opened, the state kept.
  if (!configured_ || !initialised_) {
    return false;
  }
  if (!std::isfinite(sigma_att_rad) || !std::isfinite(sigma_bias_rad_s) || !(sigma_att_rad > 0.0) ||
      !(sigma_bias_rad_s > 0.0)) {
    return false;
  }
  p_.setZero();
  p_.block<3, 3>(kAttitude, kAttitude) =
      sigma_att_rad * sigma_att_rad * Eigen::Matrix3d::Identity();
  p_.block<3, 3>(kGyroBias, kGyroBias) =
      sigma_bias_rad_s * sigma_bias_rad_s * Eigen::Matrix3d::Identity();
  return true;
}

bool Mekf::covarianceHealthy() const {
  // TP Ch. 7: definiteness, asked for explicitly since P is not factorised.
  if (!initialised_) {
    return true;
  }
  if (!p_.allFinite()) {
    return false;
  }
  const Eigen::LDLT<Covariance> ldlt(p_);
  return ldlt.info() == Eigen::Success && ldlt.isPositive();
}

void Mekf::reset() {
  dropSolution();
  // Only a *commanded* reset clears the count. An internal fault drops the
  // solution through dropSolution() and leaves it standing, because a filter
  // that just diverged is precisely when FDIR needs to see how many
  // measurements it had been rejecting on the way there.
  rejected_ = 0;
  forced_ = 0;
}

bool Mekf::initialize(const time::Tai& epoch,
                      const math::Quat<math::frames::Body, math::frames::ECI>& attitude,
                      const Eigen::Matrix3d& attitude_cov,
                      const math::Vec3<math::frames::Body>& gyro_bias,
                      const Eigen::Matrix3d& bias_cov) {
  if (!configured_) {
    return false;
  }
  math::Quaternion q = attitude.core();
  if (!q.isFinite() || !q.normalize()) {
    return false;
  }
  if (!attitude_cov.allFinite() || !bias_cov.allFinite() || !gyro_bias.isFinite()) {
    return false;
  }
  // The seed covariance is a trust boundary: an indefinite P makes S indefinite,
  // which makes the NIS gate meaningless (a negative NIS would sail through a
  // "too large?" test). Cholesky is the cheap, fixed-size, stack-only check that
  // it is a covariance at all.
  if (attitude_cov.llt().info() != Eigen::Success || bias_cov.llt().info() != Eigen::Success) {
    return false;
  }

  attitude_ = q.canonical();
  bias_ = gyro_bias.eigen();
  rate_.setZero();
  rate_valid_ = false;
  // Cross-covariance between the seed attitude and the seed bias is zero: the
  // single-frame initializer knows nothing about the gyro, and claiming a
  // correlation it did not measure would be the filter inventing information.
  p_.setZero();
  p_.block<3, 3>(kAttitude, kAttitude) = attitude_cov;
  p_.block<3, 3>(kGyroBias, kGyroBias) = bias_cov;
  symmetrise(p_);
  last_epoch_ = epoch;
  age_s_ = 0.0;
  initialised_ = true;
  return true;
}

bool Mekf::propagate(const time::Tai& epoch, const math::Vec3<math::frames::Body>& gyro,
                     bool gyro_valid) {
  if (!configured_ || !initialised_) {
    return false;
  }

  // Strictly increasing epochs only. Backwards is obvious; a *stuck* clock is
  // the dangerous one, because a zero-length step followed by an update would
  // fold the same measurement in twice. Both are refused without destroying the
  // solution.
  const double dt_s = (epoch - last_epoch_).seconds();
  if (!(dt_s > 0.0)) {
    return false;
  }
  last_epoch_ = epoch;
  age_s_ += dt_s;

  const bool usable_gyro = gyro_valid && gyro.isFinite() && dt_s <= cfg_.max_dt_s;
  rate_valid_ = usable_gyro;
  rate_ = usable_gyro ? Eigen::Vector3d(gyro.eigen() - bias_) : Eigen::Vector3d::Zero();

  // --- State transition ----------------------------------------------------
  // Φ = [[Φ11, Φ12], [0, I]] with Φ11 = exp(−[ω̂×]dt) and
  // Φ12 = −∫₀^dt exp(−[ω̂×]s) ds, both in closed form at constant ω̂. Without a
  // usable gyro the attitude is held (Φ11 = I) and **Φ12 is zero**, not −dt·I:
  // the bias never entered this step's propagation, so ∂δθ̇/∂δb = 0 and a
  // −dt·P_bb correlation would be invented out of nothing — at a 20 s dropout
  // that is a −20·I block, and the first measurement afterwards would drag the
  // bias to explain attitude error the bias never caused. Only the process
  // noise is added, which is a lower bound on the true growth; `max_coast_s` is
  // the real guard.
  Eigen::Matrix3d phi11 = Eigen::Matrix3d::Identity();
  Eigen::Matrix3d phi12 = Eigen::Matrix3d::Zero();
  if (usable_gyro) {
    const double rate_norm = rate_.norm();
    const double angle = rate_norm * dt_s;
    phi12 = -dt_s * Eigen::Matrix3d::Identity();
    if (angle > kMinRotationAngleRad) {
      const Eigen::Vector3d axis = rate_ / rate_norm;
      const math::Quaternion dq = math::Quaternion::FromAxisAngle(axis, angle);
      math::Quaternion propagated = dq * attitude_;
      if (!propagated.normalize()) {
        dropSolution();
        return false;
      }
      attitude_ = propagated.canonical();
      // A(dq) is exp(−[ω̂×]dt) for the passive convention (§3.3), so the
      // transition block is the increment's own attitude matrix — no second
      // exponential, and no chance of the two disagreeing.
      phi11 = dq.toRotationMatrix();

      // ∫₀^dt exp(−[n̂×]ωs) ds
      //   = I·dt − [(1−cos ωdt)/ω]·[n̂×] + [dt − (sin ωdt)/ω]·[n̂×]².
      const Eigen::Matrix3d n_skew = skew(axis);
      phi12 =
          -(dt_s * Eigen::Matrix3d::Identity() - ((1.0 - std::cos(angle)) / rate_norm) * n_skew +
            (dt_s - std::sin(angle) / rate_norm) * (n_skew * n_skew));
    }
  }

  Covariance phi = Covariance::Identity();
  phi.block<3, 3>(kAttitude, kAttitude) = phi11;
  phi.block<3, 3>(kAttitude, kGyroBias) = phi12;

  // --- Discrete process noise (Markley & Crassidis Eq. 6.93 form) ----------
  // The off-diagonal −½σ_u²dt² blocks are the point: they encode that an
  // attitude error and a bias error of the right sign are the same error seen
  // twice, which is what makes the bias observable in a few tens of seconds
  // rather than never.
  const double sigma_v_sq = cfg_.arw_rad_per_sqrt_s * cfg_.arw_rad_per_sqrt_s;
  const double sigma_u_sq = cfg_.rrw_rad_per_s_per_sqrt_s * cfg_.rrw_rad_per_s_per_sqrt_s;
  const Eigen::Matrix3d id = Eigen::Matrix3d::Identity();
  Covariance q_d = Covariance::Zero();
  q_d.block<3, 3>(kAttitude, kAttitude) =
      (sigma_v_sq * dt_s + sigma_u_sq * dt_s * dt_s * dt_s / 3.0) * id;
  const Eigen::Matrix3d q_cross = (-0.5 * sigma_u_sq * dt_s * dt_s) * id;
  q_d.block<3, 3>(kAttitude, kGyroBias) = q_cross;
  q_d.block<3, 3>(kGyroBias, kAttitude) = q_cross;
  q_d.block<3, 3>(kGyroBias, kGyroBias) = (sigma_u_sq * dt_s) * id;

  p_ = phi * p_ * phi.transpose() + q_d;
  symmetrise(p_);

  if (!attitude_.isFinite() || !p_.allFinite()) {
    dropSolution();  // a NaN can never recover on its own; drop back to cold start
    return false;
  }
  return true;
}

bool Mekf::update(const math::Vec3<math::frames::Body>& body_meas,
                  const math::Vec3<math::frames::ECI>& reference, double sigma_rad, MekfUpdate& out,
                  bool force) {
  out = MekfUpdate{};
  if (!configured_ || !initialised_) {
    return false;
  }
  if (!(sigma_rad > 0.0) || !std::isfinite(sigma_rad)) {
    return false;
  }
  Eigen::Vector3d b_meas;
  Eigen::Vector3d r_ref;
  if (!unit(body_meas.eigen(), b_meas) || !unit(reference.eigen(), r_ref)) {
    return false;
  }

  // --- Innovation ----------------------------------------------------------
  const Eigen::Vector3d b_pred = attitude_.rotate(r_ref);
  const Eigen::Vector3d y = b_meas - b_pred;
  // H = [[b̂_pred ×] 0₃]: to first order the predicted direction moves by
  // −[δθ×]b̂_pred, so the innovation is +[b̂_pred×]δθ under the sign convention
  // of mekf.hpp (q_true = δq ⊗ q̂).
  Eigen::Matrix<double, 3, kDim> h = Eigen::Matrix<double, 3, kDim>::Zero();
  h.block<3, 3>(0, kAttitude) = skew(b_pred);

  const double r_var = sigma_rad * sigma_rad;
  const Eigen::Matrix3d s = h * p_ * h.transpose() + r_var * Eigen::Matrix3d::Identity();
  const Eigen::Matrix3d s_inv = s.inverse();
  if (!s_inv.allFinite() || !y.allFinite()) {
    return false;
  }

  const double nis = y.dot(s_inv * y);
  if (!std::isfinite(nis)) {
    return false;
  }
  out.innovation = y;
  out.innovation_cov = s;
  out.nis = nis;

  // --- Divergence guard ----------------------------------------------------
  // An outlier folded in at full gain drags the reference attitude off and the
  // filter never comes back, so it is refused here. One rejection is normal
  // (a χ² tail, a glint, a momentary occlusion); a *stream* of them is what
  // FDIR watches @ref rejectedCount for.
  //
  // Written as an accept range rather than `nis > gate`: if P ever went
  // indefinite, S would follow and the NIS could come back **negative**, which
  // a one-sided "too large?" test waves straight through. Silent acceptance is
  // the one failure mode a divergence guard must not have.
  //
  // `force` (TP §9.1's editing flag) overrides the *gate* only: a negative NIS
  // is the covariance gone indefinite, and no operator flag makes an update
  // against that meaningful.
  if (!(nis >= 0.0)) {
    ++rejected_;
    return false;
  }
  if (nis > cfg_.nis_gate) {
    if (!force) {
      ++rejected_;
      return false;
    }
    ++forced_;
    out.forced = true;
  }

  // --- Gain, Joseph-form covariance, multiplicative reset ------------------
  const Eigen::Matrix<double, kDim, 3> k_gain = p_ * h.transpose() * s_inv;
  const Eigen::Matrix<double, kDim, 1> dx = k_gain * y;
  if (!k_gain.allFinite() || !dx.allFinite()) {
    return false;
  }

  // Joseph form rather than (I−KH)P: it stays symmetric positive-definite under
  // round-off and under a gain that is not exactly optimal, which is the case
  // the moment R is inflated for a systematic budget (see mekf.hpp).
  const Covariance ikh = Covariance::Identity() - k_gain * h;
  p_ = ikh * p_ * ikh.transpose() + r_var * (k_gain * k_gain.transpose());
  symmetrise(p_);

  // Reference-attitude reset: q̂⁺ = δq̂ ⊗ q̂, exact axis-angle rather than the
  // small-angle [1, δθ/2], so a large correction (re-acquisition after a coast)
  // stays a proper rotation. The error state is zero by construction afterwards
  // — it is never stored.
  const Eigen::Vector3d dtheta = dx.head<3>();
  const double dtheta_norm = dtheta.norm();
  if (dtheta_norm > kMinRotationAngleRad) {
    const math::Quaternion dq = math::Quaternion::FromAxisAngle(dtheta / dtheta_norm, dtheta_norm);
    math::Quaternion corrected = dq * attitude_;
    if (!corrected.normalize()) {
      dropSolution();
      out = MekfUpdate{};
      return false;
    }
    attitude_ = corrected.canonical();
  }
  bias_ += dx.tail<3>();

  if (!attitude_.isFinite() || !bias_.allFinite() || !p_.allFinite()) {
    dropSolution();
    out = MekfUpdate{};
    return false;
  }

  age_s_ = 0.0;
  out.accepted = true;
  return true;
}

bool Mekf::updateAttitude(const math::Quat<math::frames::Body, math::frames::ECI>& measured,
                          const Eigen::Matrix3d& noise_cov, MekfUpdate& out, bool force) {
  out = MekfUpdate{};
  if (!configured_ || !initialised_) {
    return false;
  }
  math::Quaternion q_meas = measured.core();
  if (!q_meas.isFinite() || !q_meas.normalize()) {
    return false;
  }
  if (!noise_cov.allFinite()) {
    return false;
  }
  // R is a trust boundary for the same reason the seed covariance is: an
  // indefinite R makes S indefinite, and the NIS could then come back negative
  // and sail through a one-sided gate. Symmetrised first because a caller
  // building `A D Aᵀ` gets a matrix symmetric in exact arithmetic and not
  // bitwise, then required positive-definite by a successful Cholesky.
  const Eigen::Matrix3d r_cov = 0.5 * (noise_cov + noise_cov.transpose());
  const Eigen::LLT<Eigen::Matrix3d> r_llt(r_cov);
  if (r_llt.info() != Eigen::Success) {
    return false;
  }

  // --- Innovation ----------------------------------------------------------
  // z = δθ of q_meas = δq ⊗ q̂, the *exact* rotation vector taken the short way
  // round rather than 2·vec(δq). At acquisition — a tracker returning after a
  // coast — the two differ by enough to matter, and the exact form is the same
  // reduction `nees` uses, so the innovation is in the filter's own error
  // coordinates by construction rather than by a linearisation that happens to
  // agree for small angles.
  const Eigen::Vector3d z = rotationVector(q_meas * attitude_.inverse());
  // H = [I₃ 0₃]: the measurement *is* the attitude, so it sees the attitude
  // error directly and the gyro bias not at all.
  Eigen::Matrix<double, 3, kDim> h = Eigen::Matrix<double, 3, kDim>::Zero();
  h.block<3, 3>(0, kAttitude) = Eigen::Matrix3d::Identity();

  const Eigen::Matrix3d s = p_.block<3, 3>(kAttitude, kAttitude) + r_cov;
  const Eigen::Matrix3d s_inv = s.inverse();
  if (!s_inv.allFinite() || !z.allFinite()) {
    return false;
  }

  const double nis = z.dot(s_inv * z);
  if (!std::isfinite(nis)) {
    return false;
  }
  out.innovation = z;
  out.innovation_cov = s;
  out.nis = nis;

  // --- Divergence guard, on **3** degrees of freedom -----------------------
  // Unlike a vector update, none of the three components is degenerate: a
  // tracker constrains the rotation about its boresight too, just far more
  // loosely. So this gate is χ²₃ and carries its own configured threshold.
  // Same accept-range form, for the same reason (a negative NIS must not pass).
  if (!(nis >= 0.0)) {
    ++rejected_;
    return false;
  }
  if (nis > cfg_.attitude_nis_gate) {
    if (!force) {  // TP §9.1 force flag, gate only — see update()
      ++rejected_;
      return false;
    }
    ++forced_;
    out.forced = true;
  }

  // --- Gain, Joseph-form covariance, multiplicative reset ------------------
  const Eigen::Matrix<double, kDim, 3> k_gain = p_ * h.transpose() * s_inv;
  const Eigen::Matrix<double, kDim, 1> dx = k_gain * z;
  if (!k_gain.allFinite() || !dx.allFinite()) {
    return false;
  }

  const Covariance ikh = Covariance::Identity() - k_gain * h;
  p_ = ikh * p_ * ikh.transpose() + k_gain * r_cov * k_gain.transpose();
  symmetrise(p_);

  const Eigen::Vector3d dtheta = dx.head<3>();
  const double dtheta_norm = dtheta.norm();
  if (dtheta_norm > kMinRotationAngleRad) {
    const math::Quaternion dq = math::Quaternion::FromAxisAngle(dtheta / dtheta_norm, dtheta_norm);
    math::Quaternion corrected = dq * attitude_;
    if (!corrected.normalize()) {
      dropSolution();
      out = MekfUpdate{};
      return false;
    }
    attitude_ = corrected.canonical();
  }
  bias_ += dx.tail<3>();

  if (!attitude_.isFinite() || !bias_.allFinite() || !p_.allFinite()) {
    dropSolution();
    out = MekfUpdate{};
    return false;
  }

  age_s_ = 0.0;
  out.accepted = true;
  return true;
}

bool Mekf::nees(const math::Quat<math::frames::Body, math::frames::ECI>& attitude_true,
                const math::Vec3<math::frames::Body>& bias_true, double& out) const {
  if (!initialised_ || !attitude_true.core().isFinite() || !bias_true.isFinite()) {
    return false;
  }
  // δθ from q_true = δq ⊗ q̂ (mekf.hpp convention), δb = b_true − b̂.
  Eigen::Matrix<double, kDim, 1> e;
  e.head<3>() = rotationVector(attitude_true.core() * attitude_.inverse());
  e.tail<3>() = bias_true.eigen() - bias_;

  const Covariance p_inv = p_.inverse();
  if (!p_inv.allFinite() || !e.allFinite()) {
    return false;
  }
  const double value = e.dot(p_inv * e);
  if (!std::isfinite(value)) {
    return false;
  }
  out = value;
  return true;
}

void writeToEstimatedState(const Mekf& filter, const time::Tai& epoch,
                           state::EstimatedState& state) {
  const bool valid = filter.attitudeValid();
  state.epoch = epoch;
  state.attitude = filter.attitude();
  state.body_rate = filter.bodyRate();
  state.gyro_bias = filter.gyroBias();
  state.valid.attitude = valid;
  state.valid.body_rate = filter.rateValid();
  state.valid.gyro_bias = valid;
  state.valid.covariance = valid;
  if (valid) {
    const Mekf::Covariance& p = filter.covariance();
    state.covariance.block<3, 3>(state::ErrorState::kAttitude, state::ErrorState::kAttitude) =
        p.block<3, 3>(Mekf::kAttitude, Mekf::kAttitude);
    state.covariance.block<3, 3>(state::ErrorState::kGyroBias, state::ErrorState::kGyroBias) =
        p.block<3, 3>(Mekf::kGyroBias, Mekf::kGyroBias);
    state.covariance.block<3, 3>(state::ErrorState::kAttitude, state::ErrorState::kGyroBias) =
        p.block<3, 3>(Mekf::kAttitude, Mekf::kGyroBias);
    state.covariance.block<3, 3>(state::ErrorState::kGyroBias, state::ErrorState::kAttitude) =
        p.block<3, 3>(Mekf::kGyroBias, Mekf::kAttitude);
  }
  state.mode = valid ? state::EstimationMode::Fine : state::EstimationMode::Invalid;
}

}  // namespace polaris::gnc
