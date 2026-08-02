/// @file
/// @brief Implementation of the JPL scalar-first quaternion core.
/// See quaternion.hpp for conventions and references.

#include "math/quaternion.hpp"

#include <cmath>

namespace polaris::math {

Quaternion Quaternion::FromAxisAngle(const Eigen::Vector3d& unit_axis, double angle_rad) {
  const double half = 0.5 * angle_rad;
  const double s = std::sin(half);
  return Quaternion(std::cos(half), s * unit_axis.x(), s * unit_axis.y(), s * unit_axis.z());
}

bool Quaternion::isUnit(double tol) const {
  return std::abs(q_.norm() - 1.0) <= tol;
}

bool Quaternion::normalize(double eps) {
  const double n = q_.norm();
  if (n < eps) {
    return false;
  }
  q_ /= n;
  return true;
}

Quaternion Quaternion::conjugate() const {
  return Quaternion(q_[0], -q_[1], -q_[2], -q_[3]);
}

Quaternion Quaternion::canonical() const {
  if (q_[0] < 0.0) {
    return Quaternion(Eigen::Vector4d(-q_));
  }
  return *this;
}

Quaternion Quaternion::operator*(const Quaternion& rhs) const {
  // JPL-convention product (scalar-first): A(a*b) = A(a) A(b).
  const double a0 = q_[0];
  const Eigen::Vector3d av = q_.tail<3>();
  const double b0 = rhs.q_[0];
  const Eigen::Vector3d bv = rhs.q_.tail<3>();

  const double prod_scalar = a0 * b0 - av.dot(bv);
  const Eigen::Vector3d prod_vec = a0 * bv + b0 * av - av.cross(bv);
  return Quaternion(prod_scalar, prod_vec.x(), prod_vec.y(), prod_vec.z());
}

Eigen::Matrix3d Quaternion::toRotationMatrix() const {
  // Passive attitude matrix A(q) = (2 q0^2 - 1) I - 2 q0 [qv x] + 2 qv qv^T.
  const double q0 = q_[0];
  const Eigen::Vector3d qv = q_.tail<3>();

  Eigen::Matrix3d skew;
  skew << 0.0, -qv.z(), qv.y(),  //
      qv.z(), 0.0, -qv.x(),      //
      -qv.y(), qv.x(), 0.0;

  return (2.0 * q0 * q0 - 1.0) * Eigen::Matrix3d::Identity() - 2.0 * q0 * skew +
         2.0 * (qv * qv.transpose());
}

Eigen::Vector3d Quaternion::rotate(const Eigen::Vector3d& v) const {
  return toRotationMatrix() * v;
}

double Quaternion::angularDistance(const Quaternion& other) const {
  // Error quaternion δq = q⁻¹ ⊗ other, whose scalar part is the four-vector dot
  // product and whose vector part has norm sin(θ/2). Taking the angle from the
  // atan2 of the two parts — rather than acos of the scalar alone — keeps the
  // small-angle result accurate: near identity the scalar is 1 - θ²/8 and loses
  // half its significant digits to cancellation, while the vector part is O(θ)
  // and loses none. The absolute value on the scalar maps δq and -δq (the same
  // rotation) to the same angle, keeping the result on the θ ≤ π branch.
  const Quaternion e = inverse() * other;
  return 2.0 * std::atan2(e.vec().norm(), std::abs(e.scalar()));
}

Quaternion Quaternion::FromRotationMatrix(const Eigen::Matrix3d& dcm) {
  // Shepperd's method: pivot on the largest of {q0^2, q1^2, q2^2, q3^2} for
  // numerical stability, then recover the rest from off-diagonal relations.
  const double a00 = dcm(0, 0);
  const double a11 = dcm(1, 1);
  const double a22 = dcm(2, 2);

  const double t0 = a00 + a11 + a22;   // 4 q0^2 - 1
  const double t1 = a00 - a11 - a22;   // 4 q1^2 - 1
  const double t2 = -a00 + a11 - a22;  // 4 q2^2 - 1
  const double t3 = -a00 - a11 + a22;  // 4 q3^2 - 1

  double q0 = 1.0;
  double q1 = 0.0;
  double q2 = 0.0;
  double q3 = 0.0;

  if (t0 >= t1 && t0 >= t2 && t0 >= t3) {
    q0 = 0.5 * std::sqrt(1.0 + t0);
    const double inv = 0.25 / q0;
    q1 = (dcm(1, 2) - dcm(2, 1)) * inv;
    q2 = (dcm(2, 0) - dcm(0, 2)) * inv;
    q3 = (dcm(0, 1) - dcm(1, 0)) * inv;
  } else if (t1 >= t2 && t1 >= t3) {
    q1 = 0.5 * std::sqrt(1.0 + t1);
    const double inv = 0.25 / q1;
    q0 = (dcm(1, 2) - dcm(2, 1)) * inv;
    q2 = (dcm(0, 1) + dcm(1, 0)) * inv;
    q3 = (dcm(2, 0) + dcm(0, 2)) * inv;
  } else if (t2 >= t3) {
    q2 = 0.5 * std::sqrt(1.0 + t2);
    const double inv = 0.25 / q2;
    q0 = (dcm(2, 0) - dcm(0, 2)) * inv;
    q1 = (dcm(0, 1) + dcm(1, 0)) * inv;
    q3 = (dcm(1, 2) + dcm(2, 1)) * inv;
  } else {
    q3 = 0.5 * std::sqrt(1.0 + t3);
    const double inv = 0.25 / q3;
    q0 = (dcm(0, 1) - dcm(1, 0)) * inv;
    q1 = (dcm(2, 0) + dcm(0, 2)) * inv;
    q2 = (dcm(1, 2) + dcm(2, 1)) * inv;
  }

  // Defensively renormalize so the result is a unit quaternion even if @p dcm
  // is slightly non-orthonormal; the pivot guarantees norm >= 0.5 so this never
  // fails. Callers needing strict input validation should check orthonormality
  // upstream.
  Quaternion q(q0, q1, q2, q3);
  q.normalize();
  return q.canonical();
}

}  // namespace polaris::math
