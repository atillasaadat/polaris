/// @file
/// @brief Attitude-independent magnetometer calibration implementation (design
/// doc §8.1). See mag_calibration.hpp for the model, conventions, and
/// references.

#include "gnc/mag_calibration.hpp"

#include <cmath>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>

namespace polaris::gnc {

namespace {

using ParamVector = Eigen::Matrix<double, MagCalibrationAccumulator::kParameters, 1>;
using ParamMatrix = Eigen::Matrix<double, MagCalibrationAccumulator::kParameters,
                                  MagCalibrationAccumulator::kParameters>;

/// Relative floor on the eigenvalues of the fitted quadric `A`. Below this the
/// ellipsoid is degenerate in one axis and `A⁻¹` — which recovers the hard iron
/// — is not usable. A structural numeric guard, not a tuning value: a physical
/// soft-iron matrix is within a few percent of the identity, so any fitted `A`
/// anywhere near this floor is noise.
constexpr double kMinQuadricEigenRatio = 1.0e-6;

/// The design row `h(x)` of the linear model, ordered to match `θ`.
ParamVector designRow(const Eigen::Vector3d& x) {
  ParamVector h;
  h << x.x() * x.x(), x.y() * x.y(), x.z() * x.z(), 2.0 * x.x() * x.y(), 2.0 * x.x() * x.z(),
      2.0 * x.y() * x.z(), -2.0 * x.x(), -2.0 * x.y(), -2.0 * x.z(), 1.0;
  return h;
}

/// The symmetric `A` block of a parameter vector.
Eigen::Matrix3d quadricOf(const ParamVector& theta) {
  Eigen::Matrix3d a;
  a << theta(0), theta(3), theta(4),  //
      theta(3), theta(1), theta(5),   //
      theta(4), theta(5), theta(2);
  return a;
}

/// Residual sum of squares of the linear model at @p theta, in closed form from
/// the accumulated moments: `Σ(hᵀθ − f²)² = θᵀNθ − 2θᵀg + Σf⁴`. Cancellation can
/// push this a hair below zero on a perfect fit, so it is clamped.
double sumSquaredResiduals(const ParamMatrix& normal, const ParamVector& rhs, double sum_f4,
                           const ParamVector& theta) {
  const double ssr = theta.dot(normal * theta) - 2.0 * theta.dot(rhs) + sum_f4;
  return (ssr > 0.0) ? ssr : 0.0;
}

}  // namespace

bool MagCalibrationConfig::isValid() const {
  const bool finite = std::isfinite(nominal_field_t) && std::isfinite(min_field_t) &&
                      std::isfinite(max_field_t) && std::isfinite(min_coverage) &&
                      std::isfinite(max_condition) && std::isfinite(min_residual_improvement);
  return finite && nominal_field_t > 0.0 && min_field_t > 0.0 && max_field_t > min_field_t &&
         min_samples >= 2 * MagCalibrationAccumulator::kParameters && min_coverage > 0.0 &&
         min_coverage <= 1.0 && max_condition > 1.0 && min_residual_improvement >= 1.0;
}

MagCalibrationAccumulator::MagCalibrationAccumulator(const MagCalibrationConfig& config)
    : cfg_(config), configured_(config.isValid()) {}

void MagCalibrationAccumulator::reset() {
  normal_.setZero();
  rhs_.setZero();
  directions_.setZero();
  sum_f2_squared_ = 0.0;
  sum_field_t_ = 0.0;
  count_ = 0;
}

bool MagCalibrationAccumulator::addSample(const math::Vec3<math::frames::Body>& m_raw,
                                          double igrf_magnitude_t) {
  if (!configured_) {
    return false;
  }
  const Eigen::Vector3d m = m_raw.eigen();
  if (!m.allFinite() || !std::isfinite(igrf_magnitude_t)) {
    return false;
  }
  const double m_norm = m.norm();
  if (m_norm < cfg_.min_field_t || m_norm > cfg_.max_field_t ||
      igrf_magnitude_t < cfg_.min_field_t || igrf_magnitude_t > cfg_.max_field_t) {
    return false;
  }

  const Eigen::Vector3d x = m / cfg_.nominal_field_t;
  const double f2 =
      (igrf_magnitude_t / cfg_.nominal_field_t) * (igrf_magnitude_t / cfg_.nominal_field_t);
  const ParamVector h = designRow(x);

  normal_ += h * h.transpose();
  rhs_ += h * f2;
  const Eigen::Vector3d direction = m / m_norm;
  directions_ += direction * direction.transpose();
  sum_f2_squared_ += f2 * f2;
  sum_field_t_ += igrf_magnitude_t;
  ++count_;
  return true;
}

double MagCalibrationAccumulator::coverage() const {
  if (count_ <= 0) {
    return 0.0;
  }
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(directions_ / static_cast<double>(count_));
  if (solver.info() != Eigen::Success) {
    return 0.0;
  }
  return 3.0 * solver.eigenvalues()(0);
}

bool MagCalibrationAccumulator::solve(MagCalibrationResult& out) const {
  MagCalibrationRefusal ignored = MagCalibrationRefusal::None;
  return solve(out, ignored);
}

bool MagCalibrationAccumulator::solve(MagCalibrationResult& out, MagCalibrationRefusal& why) const {
  out = MagCalibrationResult{};
  why = MagCalibrationRefusal::None;
  if (!configured_) {
    why = MagCalibrationRefusal::NotConfigured;
    return false;
  }
  if (count_ < cfg_.min_samples) {
    why = MagCalibrationRefusal::Samples;
    return false;
  }
  const double n = static_cast<double>(count_);

  // Coverage first: it is the cheapest gate and the one whose failure has a
  // physical explanation the ground can act on ("tumble further"), where a
  // condition-number failure does not.
  const double coverage = MagCalibrationAccumulator::coverage();
  if (!(coverage >= cfg_.min_coverage)) {
    why = MagCalibrationRefusal::Coverage;
    return false;
  }

  // The normal equations. Streaming forces this form — the design matrix is
  // never held — so the effective condition is squared relative to a QR of the
  // rows; `max_condition` is set on the normal matrix accordingly. The same
  // decomposition gates the conditioning and solves the system.
  Eigen::SelfAdjointEigenSolver<ParamMatrix> solver(normal_);
  if (solver.info() != Eigen::Success) {
    why = MagCalibrationRefusal::Numerical;
    return false;
  }
  const ParamVector& lambda = solver.eigenvalues();
  if (!(lambda(kParameters - 1) > 0.0) || !(lambda(0) > 0.0)) {
    // A non-positive eigenvalue is a normal matrix that is singular to working
    // precision — the same unobservability the condition gate catches, past the
    // point where the ratio is even computable.
    why = MagCalibrationRefusal::Condition;
    return false;
  }
  const double condition = lambda(kParameters - 1) / lambda(0);
  if (!(condition <= cfg_.max_condition)) {
    why = MagCalibrationRefusal::Condition;
    return false;
  }
  const ParamVector theta =
      solver.eigenvectors() * (solver.eigenvectors().transpose() * rhs_).cwiseQuotient(lambda);
  if (!theta.allFinite()) {
    why = MagCalibrationRefusal::Numerical;
    return false;
  }

  // Recover the ellipsoid: `A` must be positive definite for `A^{1/2}` and
  // `A⁻¹` to exist. A non-PD fit is noise or coverage, never a field.
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> quadric(quadricOf(theta));
  if (quadric.info() != Eigen::Success) {
    why = MagCalibrationRefusal::Numerical;
    return false;
  }
  const Eigen::Vector3d& a_eig = quadric.eigenvalues();
  if (!(a_eig(0) > kMinQuadricEigenRatio * a_eig(2)) || !(a_eig(0) > 0.0)) {
    why = MagCalibrationRefusal::Numerical;
    return false;
  }
  const Eigen::Matrix3d& u = quadric.eigenvectors();
  const Eigen::Matrix3d soft_iron_inverse =
      u * a_eig.cwiseSqrt().asDiagonal() * u.transpose();  // A^{1/2}, symmetric PD
  const Eigen::Vector3d v = theta.segment<3>(6);
  const Eigen::Vector3d beta =
      u * (u.transpose() * v).cwiseQuotient(a_eig);  // A⁻¹v, non-dimensional centre

  // Residuals. The fit carries `c` as a free parameter, so the *applied*
  // correction's magnitude residual differs from the linear model's by the
  // constant `κ = c − βᵀAβ` — nonzero only through noise. Since the constant
  // column makes the linear residuals zero-mean, the applied residual's mean
  // square is the model's plus `κ²`; folding it in is what keeps this number
  // honest about the correction actually shipped.
  const double kappa = theta(kParameters - 1) - beta.dot(quadricOf(theta) * beta);
  const double ssr_fit = sumSquaredResiduals(normal_, rhs_, sum_f2_squared_, theta);
  const double mean_square = ssr_fit / n + kappa * kappa;

  ParamVector uncalibrated = ParamVector::Zero();
  uncalibrated(0) = 1.0;
  uncalibrated(1) = 1.0;
  uncalibrated(2) = 1.0;
  const double ssr_raw = sumSquaredResiduals(normal_, rhs_, sum_f2_squared_, uncalibrated);

  // `|B|² − F² ≈ 2F(|B| − F)`, so the squared-field residual divides by twice
  // the mean field to become a field-magnitude residual [T].
  const double mean_field_t = sum_field_t_ / n;
  const double scale = cfg_.nominal_field_t * cfg_.nominal_field_t / (2.0 * mean_field_t);
  const double residual_rms_t = scale * std::sqrt(mean_square);
  const double uncalibrated_rms_t = scale * std::sqrt(ssr_raw / n);
  // The `> 0` is not redundant with the ratio: on data that is already perfect
  // both residuals are zero, and `0 >= factor * 0` would pass the gate and ship
  // a "calibration" fitted to nothing.
  if (!(uncalibrated_rms_t > 0.0) ||
      !(uncalibrated_rms_t >= cfg_.min_residual_improvement * residual_rms_t)) {
    why = MagCalibrationRefusal::NoImprovement;
    return false;
  }

  const Eigen::Vector3d hard_iron = cfg_.nominal_field_t * beta;
  if (!hard_iron.allFinite() || !soft_iron_inverse.allFinite() || !std::isfinite(residual_rms_t) ||
      !std::isfinite(uncalibrated_rms_t) || !std::isfinite(mean_field_t)) {
    why = MagCalibrationRefusal::Numerical;
    return false;
  }

  out.hard_iron_offset = math::Vec3<math::frames::Body>(hard_iron);
  out.soft_iron_inverse = soft_iron_inverse;
  out.residual_rms_t = residual_rms_t;
  out.uncalibrated_residual_rms_t = uncalibrated_rms_t;
  out.residual_angle_rad = residual_rms_t / mean_field_t;
  out.mean_field_t = mean_field_t;
  out.coverage = coverage;
  out.condition = condition;
  out.sample_count = count_;
  out.valid = true;
  return true;
}

math::Vec3<math::frames::Body> applyMagCalibration(const MagCalibrationResult& cal,
                                                   const math::Vec3<math::frames::Body>& m_raw) {
  if (!cal.valid) {
    return m_raw;
  }
  return math::Vec3<math::frames::Body>(cal.soft_iron_inverse *
                                        (m_raw.eigen() - cal.hard_iron_offset.eigen()));
}

}  // namespace polaris::gnc
