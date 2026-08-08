#include "gnc/dipole_estimation.hpp"

#include <cmath>
#include <Eigen/Eigenvalues>

namespace polaris::gnc {
namespace {

constexpr double kNsPerSecond = 1.0e9;

bool refuse(DipoleRefusal reason, DipoleResult& out) {
  out.dipole_am2 = math::Vec3<math::frames::Body>(Eigen::Vector3d::Zero());
  out.sigma_am2 = math::Vec3<math::frames::Body>(Eigen::Vector3d::Zero());
  out.valid = false;
  out.refusal = reason;
  return false;
}

}  // namespace

bool DipoleEstimatorConfig::isValid() const {
  return std::isfinite(nominal_field_t) && nominal_field_t > 0.0 && std::isfinite(min_field_t) &&
         min_field_t > 0.0 && std::isfinite(max_field_t) && max_field_t > min_field_t &&
         std::isfinite(forgetting_time_s) && forgetting_time_s > 0.0 &&
         std::isfinite(min_sample_interval_s) && min_sample_interval_s > 0.0 &&
         std::isfinite(min_observability) && min_observability > 0.0 && min_observability <= 1.0 &&
         std::isfinite(min_information) && min_information > 0.0 &&
         std::isfinite(torque_sigma_nm) && torque_sigma_nm > 0.0 && std::isfinite(max_dipole_am2) &&
         max_dipole_am2 > 0.0;
}

DipoleEstimator::DipoleEstimator(const DipoleEstimatorConfig& config)
    : config_(config), configured_(config.isValid()) {}

void DipoleEstimator::reset() {
  have_previous_ = false;
  have_estimate_ = false;
  information_.setZero();
  rhs_.setZero();
  effective_samples_ = 0.0;
  previous_time_ns_ = 0;
  estimate_ = math::Vec3<math::frames::Body>(Eigen::Vector3d::Zero());
  sigma_ = math::Vec3<math::frames::Body>(Eigen::Vector3d::Zero());
}

bool DipoleEstimator::update(const math::Vec3<math::frames::Body>& observer_torque_nm,
                             const math::Vec3<math::frames::Body>& field_tesla,
                             std::int64_t time_tag_tai_ns, DipoleResult& out) {
  out = DipoleResult{};
  if (!configured_) {
    return refuse(DipoleRefusal::kUnconfigured, out);
  }
  const Eigen::Vector3d tau = observer_torque_nm.eigen();
  const Eigen::Vector3d field = field_tesla.eigen();
  if (!tau.allFinite() || !field.allFinite()) {
    return refuse(DipoleRefusal::kBadInput, out);
  }

  // Report the accumulator's state on every path below, so a refusal still tells
  // the ground how far the pass has got.
  const auto report = [this](DipoleResult& r) { r.effective_samples = effective_samples_; };

  const double field_norm = field.norm();
  if (field_norm < config_.min_field_t || field_norm > config_.max_field_t) {
    refuse(DipoleRefusal::kFieldOutOfRange, out);
    report(out);
    return false;
  }

  double weight = 1.0;
  if (have_previous_) {
    if (time_tag_tai_ns <= previous_time_ns_) {
      // A stuck or backwards clock cannot produce a forgetting weight, and the
      // sample carries no dateable evidence. Refuse without re-anchoring: an
      // out-of-order tag must not be allowed to set the anchor forward.
      refuse(DipoleRefusal::kNonMonotonicTime, out);
      report(out);
      return false;
    }
    const double dt_s = static_cast<double>(time_tag_tai_ns - previous_time_ns_) / kNsPerSecond;
    if (dt_s < config_.min_sample_interval_s) {
      // Correlated with the previous sample: accumulating it would inflate the
      // information matrix without adding evidence.
      refuse(DipoleRefusal::kSampleTooSoon, out);
      report(out);
      return false;
    }
    weight = std::exp(-dt_s / config_.forgetting_time_s);
  }

  // tau = m x B = -[B x] m, so H = -[B x] and H^T H = |B|^2 I - B B^T, while
  // H^T tau = [B x] tau = B x tau. Both preconditioned by nominal_field_t: the
  // information matrix becomes dimensionless (and reads as an effective sample
  // count), and the right-hand side lands directly in A*m^2.
  const Eigen::Vector3d b = field / config_.nominal_field_t;
  const Eigen::Matrix3d h_t_h = b.squaredNorm() * Eigen::Matrix3d::Identity() - b * b.transpose();
  const Eigen::Vector3d h_t_y = b.cross(tau) / config_.nominal_field_t;

  const Eigen::Matrix3d next_information = weight * information_ + h_t_h;
  const Eigen::Vector3d next_rhs = weight * rhs_ + h_t_y;
  if (!next_information.allFinite() || !next_rhs.allFinite()) {
    // Leave the accumulator untouched rather than poisoning a window of good
    // evidence with one unusable sample.
    refuse(DipoleRefusal::kBadInput, out);
    report(out);
    return false;
  }
  information_ = next_information;
  rhs_ = next_rhs;
  effective_samples_ = weight * effective_samples_ + 1.0;
  previous_time_ns_ = time_tag_tai_ns;
  have_previous_ = true;

  const bool solved = solveInto(out);
  report(out);
  if (solved) {
    estimate_ = out.dipole_am2;
    sigma_ = out.sigma_am2;
    have_estimate_ = true;
  }
  return solved;
}

bool DipoleEstimator::solveInto(DipoleResult& out) const {
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eig;
  eig.computeDirect(information_);
  if (eig.info() != Eigen::Success) {
    return refuse(DipoleRefusal::kNumerical, out);
  }
  // computeDirect returns eigenvalues in increasing order.
  const double lambda_min = eig.eigenvalues()(0);
  const double lambda_max = eig.eigenvalues()(2);
  if (!std::isfinite(lambda_min) || !std::isfinite(lambda_max) || lambda_max <= 0.0) {
    return refuse(DipoleRefusal::kNoObservability, out);
  }
  out.observability = (lambda_min > 0.0) ? lambda_min / lambda_max : 0.0;
  out.information = (lambda_min > 0.0) ? lambda_min : 0.0;

  // Geometry first, then evidence: an estimate published from a rank-deficient
  // information matrix invents its third component out of the null space, so
  // that gate has to close before any question of how much noise is left.
  if (out.observability < config_.min_observability) {
    return refuse(DipoleRefusal::kNoObservability, out);
  }
  if (out.information < config_.min_information) {
    return refuse(DipoleRefusal::kInsufficientInformation, out);
  }

  const Eigen::LLT<Eigen::Matrix3d> llt(information_);
  if (llt.info() != Eigen::Success) {
    return refuse(DipoleRefusal::kNumerical, out);
  }
  const Eigen::Vector3d dipole = llt.solve(rhs_);
  const Eigen::Matrix3d covariance = llt.solve(Eigen::Matrix3d::Identity()) *
                                     (config_.torque_sigma_nm / config_.nominal_field_t) *
                                     (config_.torque_sigma_nm / config_.nominal_field_t);
  if (!dipole.allFinite() || !covariance.allFinite()) {
    return refuse(DipoleRefusal::kNumerical, out);
  }
  const Eigen::Vector3d variance = covariance.diagonal();
  if ((variance.array() < 0.0).any()) {
    return refuse(DipoleRefusal::kNumerical, out);
  }
  const Eigen::Vector3d sigma = variance.cwiseSqrt();
  if (!sigma.allFinite()) {
    return refuse(DipoleRefusal::kNumerical, out);
  }

  // The configured bound is a cleanliness budget established on the ground. A
  // fit outside it is refused, not clamped: clamping would publish a direction
  // the data never supported at a magnitude the policy chose, straight into the
  // tier-1 feedforward.
  if (dipole.norm() > config_.max_dipole_am2) {
    out.dipole_am2 = math::Vec3<math::frames::Body>(Eigen::Vector3d::Zero());
    out.sigma_am2 = math::Vec3<math::frames::Body>(Eigen::Vector3d::Zero());
    out.valid = false;
    out.refusal = DipoleRefusal::kOutOfBounds;
    return false;
  }

  out.dipole_am2 = math::Vec3<math::frames::Body>(dipole);
  out.sigma_am2 = math::Vec3<math::frames::Body>(sigma);
  out.valid = true;
  out.refusal = DipoleRefusal::kNone;
  return true;
}

}  // namespace polaris::gnc
