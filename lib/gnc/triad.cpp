/// @file
/// @brief TRIAD implementation (design doc §8.1; REQ-ADET-003). See triad.hpp
/// for conventions, assumptions, and references.

#include "gnc/triad.hpp"

#include <cmath>
#include <Eigen/Core>
#include <Eigen/Geometry>

#include "math/quaternion.hpp"

namespace polaris::gnc {

namespace {

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

/// Orthonormal TRIAD basis from two unit directions: columns
/// `[û₁, (û₁×û₂)/|·|, û₁×((û₁×û₂)/|·|)]` (Markley & Crassidis Eq. 5.3).
/// Returns false if the pair is closer to parallel than @p min_sin.
bool triadBasis(const Eigen::Vector3d& u1, const Eigen::Vector3d& u2, double min_sin,
                Eigen::Matrix3d& basis) {
  const Eigen::Vector3d cross = u1.cross(u2);
  const double sin_theta = cross.norm();
  if (!(sin_theta >= min_sin)) {
    return false;
  }
  basis.col(0) = u1;
  basis.col(1) = cross / sin_theta;
  basis.col(2) = u1.cross(basis.col(1));
  return true;
}

}  // namespace

bool triadCovariance(const math::Vec3<math::frames::Body>& primary_body,
                     const math::Vec3<math::frames::Body>& secondary_body, double sigma_primary_rad,
                     double sigma_secondary_rad, double min_sin_angle, Eigen::Matrix3d& out) {
  if (!std::isfinite(sigma_primary_rad) || !std::isfinite(sigma_secondary_rad) ||
      sigma_primary_rad < 0.0 || sigma_secondary_rad < 0.0 || !(min_sin_angle > 0.0)) {
    return false;
  }

  Eigen::Vector3d b1;
  Eigen::Vector3d b2;
  if (!unit(primary_body.eigen(), b1) || !unit(secondary_body.eigen(), b2)) {
    return false;
  }

  // Shuster covariance of the TRIAD solution (Shuster & Oh 1981 [shuster1981];
  // Markley & Crassidis §5.2 [markley2014]). With the QUEST measurement model
  // E[δbᵢ δbᵢᵀ] = σᵢ²(I − b̂ᵢb̂ᵢᵀ) and the body-frame error δθ defined by
  // A_est = (I − [δθ×])·A_true, TRIAD fits the primary exactly, so
  //   δθ_⊥ = −b̂₁ × δb̂₁                          (the two axes off b̂₁)
  //   δθ·b̂₁ = [cos θ (n̂·δb̂₁) − (n̂·δb̂₂)] / sin θ  (roll about b̂₁)
  // with n̂ = (b̂₁×b̂₂)/sin θ. Taking expectations gives, in terms of the
  // orthonormal body triad {b̂₁, m̂, n̂} where m̂ = (b̂₂ − cos θ·b̂₁)/sin θ,
  //   P = σ₁²(I − b̂₁b̂₁ᵀ)
  //     + [(σ₁² cos²θ + σ₂²)/sin²θ]·b̂₁b̂₁ᵀ
  //     + (σ₁² cos θ / sin θ)·(b̂₁m̂ᵀ + m̂b̂₁ᵀ).
  // Trace 2σ₁² + (σ₁²cos²θ + σ₂²)/sin²θ reduces to the familiar 2σ₁² + σ₂² for
  // orthogonal observations, and the 1/sin²θ growth is the roll about the
  // primary going unobservable as the pair aligns.
  const double cos_theta = b1.dot(b2);
  const double sin_theta = b1.cross(b2).norm();
  if (!(sin_theta >= min_sin_angle)) {
    return false;
  }
  const Eigen::Vector3d m_hat = (b2 - cos_theta * b1) / sin_theta;

  const double s1_sq = sigma_primary_rad * sigma_primary_rad;
  const double s2_sq = sigma_secondary_rad * sigma_secondary_rad;
  const Eigen::Matrix3d p1 = b1 * b1.transpose();
  const double roll_var = (s1_sq * cos_theta * cos_theta + s2_sq) / (sin_theta * sin_theta);
  const double coupling = s1_sq * cos_theta / sin_theta;

  const Eigen::Matrix3d cov = s1_sq * (Eigen::Matrix3d::Identity() - p1) + roll_var * p1 +
                              coupling * (b1 * m_hat.transpose() + m_hat * b1.transpose());
  if (!cov.allFinite()) {
    return false;
  }
  out = cov;
  return true;
}

bool triad(const TriadInput& in, TriadSolution& out) {
  out = TriadSolution{};

  if (!(in.primary.sigma_rad > 0.0) || !(in.secondary.sigma_rad > 0.0) ||
      !(in.min_sin_angle > 0.0)) {
    return false;
  }

  Eigen::Vector3d b1;
  Eigen::Vector3d b2;
  Eigen::Vector3d v1;
  Eigen::Vector3d v2;
  if (!unit(in.primary.body.eigen(), b1) || !unit(in.secondary.body.eigen(), b2) ||
      !unit(in.primary.reference.eigen(), v1) || !unit(in.secondary.reference.eigen(), v2)) {
    return false;
  }

  // Geometry is gated in both frames: a body pair that passes while the
  // reference pair does not means the measurements disagree with the models by
  // more than the geometry itself, which is not a solution worth publishing.
  Eigen::Matrix3d m_body;
  Eigen::Matrix3d m_ref;
  if (!triadBasis(b1, b2, in.min_sin_angle, m_body) ||
      !triadBasis(v1, v2, in.min_sin_angle, m_ref)) {
    return false;
  }

  // A maps ECI -> Body: it carries the reference triad onto the body triad,
  // A·m_ref = m_body, and both are orthonormal, so A = m_body·m_refᵀ.
  const Eigen::Matrix3d a_bi = m_body * m_ref.transpose();
  math::Quaternion q = math::Quaternion::FromRotationMatrix(a_bi);
  if (!q.normalize()) {
    return false;
  }
  q = q.canonical();

  Eigen::Matrix3d cov;
  if (!triadCovariance(in.primary.body, in.secondary.body, in.primary.sigma_rad,
                       in.secondary.sigma_rad, in.min_sin_angle, cov) ||
      !q.isFinite()) {
    return false;
  }

  out.attitude = math::Quat<math::frames::Body, math::frames::ECI>(q);
  out.covariance = cov;
  out.separation_rad = std::atan2(b1.cross(b2).norm(), b1.dot(b2));
  out.valid = true;
  return true;
}

}  // namespace polaris::gnc
