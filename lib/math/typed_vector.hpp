#ifndef POLARIS_MATH_TYPED_VECTOR_HPP
#define POLARIS_MATH_TYPED_VECTOR_HPP

/// @file
/// @brief Boundary-tagged 3-vector (design doc §3.1, Golden Rule 4).
///
/// `Vec3<Frame>` is a thin, fixed-size wrapper over `Eigen::Vector3d` carrying a
/// compile-time frame tag. Arithmetic is frame-preserving: adding a `Vec3<ECI>`
/// to a `Vec3<ECEF>` does not compile. Frame *changes* never happen here — they
/// route exclusively through the transform library (`Quat<>` / frames lib).
/// Inside hot kernels, drop to `.eigen()` and re-tag on the way out.

#include <Eigen/Core>
#include <Eigen/Geometry>  // Vector3d::cross

#include "math/frames.hpp"

namespace polaris::math {

/// Frame-tagged 3-vector over fixed-size Eigen (no heap; flight-safe).
/// @tparam Frame one of the tags in `polaris::math::frames`.
template <class Frame>
class Vec3 {
 public:
  using Scalar = double;

  /// Zero vector.
  Vec3() = default;

  /// Construct from components [SI units of the quantity].
  Vec3(double x, double y, double z) : v_(x, y, z) {}

  /// Re-tag a raw Eigen vector as belonging to @c Frame (kernel -> boundary).
  explicit Vec3(const Eigen::Vector3d& v) : v_(v) {}

  /// Zero vector (named constructor).
  static Vec3 Zero() { return Vec3(Eigen::Vector3d::Zero()); }

  /// @name Raw access (boundary <-> hot kernel)
  /// @{
  const Eigen::Vector3d& eigen() const { return v_; }

  Eigen::Vector3d& eigen() { return v_; }

  /// @}

  double x() const { return v_.x(); }

  double y() const { return v_.y(); }

  double z() const { return v_.z(); }

  double operator[](Eigen::Index i) const { return v_[i]; }

  /// @name Frame-preserving arithmetic (same @c Frame only)
  /// @{
  Vec3 operator+(const Vec3& o) const { return Vec3(v_ + o.v_); }

  Vec3 operator-(const Vec3& o) const { return Vec3(v_ - o.v_); }

  Vec3 operator-() const { return Vec3(Eigen::Vector3d(-v_)); }

  Vec3 operator*(double s) const { return Vec3(v_ * s); }

  Vec3 operator/(double s) const { return Vec3(v_ / s); }

  /// @}

  double dot(const Vec3& o) const { return v_.dot(o.v_); }

  /// Cross product; result remains in the same frame.
  Vec3 cross(const Vec3& o) const { return Vec3(v_.cross(o.v_)); }

  double norm() const { return v_.norm(); }

  double squaredNorm() const { return v_.squaredNorm(); }

  /// True if all components are finite (no NaN/Inf). Boundary guard (§3.6).
  bool isFinite() const { return v_.allFinite(); }

  /// Unit vector via @p out; returns false (and leaves @p out unchanged) if the
  /// vector is shorter than @p eps and cannot be safely normalized.
  bool normalized(Vec3& out, double eps = 1e-12) const {
    const double n = v_.norm();
    if (n < eps) {
      return false;
    }
    out = Vec3(v_ / n);
    return true;
  }

  static constexpr const char* frame_name() { return Frame::kName; }

 private:
  Eigen::Vector3d v_{Eigen::Vector3d::Zero()};
};

/// Left scalar multiply.
template <class Frame>
inline Vec3<Frame> operator*(double s, const Vec3<Frame>& v) {
  return v * s;
}

}  // namespace polaris::math

#endif  // POLARIS_MATH_TYPED_VECTOR_HPP
