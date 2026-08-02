#ifndef POLARIS_MATH_QUATERNION_HPP
#define POLARIS_MATH_QUATERNION_HPP

/// @file
/// @brief The single quaternion library (design doc §3.3, Golden Rule 2).
///
/// Convention: **JPL, scalar-first** `q = [q0, q1, q2, q3]` with `q0` the
/// scalar part; canonical form `q0 >= 0`. The quaternion product follows the
/// JPL convention, for which the attitude (direction-cosine) matrix composes in
/// the same order as the quaternions: `A(a * b) = A(a) A(b)`.
///
/// The attitude matrix `A(q)` is a passive coordinate transformation: it maps
/// the coordinates of a fixed vector from the reference frame to the rotated
/// frame, `v_rot = A(q) v_ref` (matches Vallado's ROT sequence). For a rotation
/// about +z by angle θ this gives the standard `ROT3(θ)`.
///
/// No heap, no exceptions, fixed-size Eigen only (flight-safe). Operations that
/// can fail (normalize) return a status; constructors assume valid input as
/// documented.
///
/// References:
///  - Markley & Crassidis, *Fundamentals of Spacecraft Attitude Determination
///    and Control*, 2014, §2.9 (quaternions, attitude matrix). [markley2014]
///  - Trawny & Roumeliotis, "Indirect Kalman Filter for 3D Attitude
///    Estimation," Univ. of Minnesota MARS Lab TR 2005-002 (JPL convention,
///    quaternion product, attitude matrix). [trawny2005]
///  - Shepperd, "Quaternion from Rotation Matrix," J. Guidance & Control,
///    1(3):223-224, 1978 (numerically stable extraction). [shepperd1978]

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "math/frames.hpp"
#include "math/typed_vector.hpp"

namespace polaris::math {

/// Untagged quaternion core (JPL scalar-first). Used inside hot kernels; the
/// boundary type is `Quat<To, From>` below.
///
/// Writing \f$\bar q = [q_0,\ \mathbf{q}_v]\f$ with scalar \f$q_0\f$ and vector
/// part \f$\mathbf{q}_v = [q_1,q_2,q_3]^\top\f$, the **JPL product** and **passive
/// attitude matrix** are
/// \f[
///   \bar a \otimes \bar b = \begin{bmatrix}
///     a_0 b_0 - \mathbf{a}_v\!\cdot\mathbf{b}_v \\[2pt]
///     a_0\,\mathbf{b}_v + b_0\,\mathbf{a}_v - \mathbf{a}_v \times \mathbf{b}_v
///   \end{bmatrix}, \qquad
///   A(\bar q) = (2q_0^2 - 1)\,\mathbf{I} - 2 q_0\,[\mathbf{q}_v\times]
///     + 2\,\mathbf{q}_v \mathbf{q}_v^\top,
/// \f]
/// where \f$[\mathbf{q}_v\times]\f$ is the cross-product (skew) matrix. Both are
/// the JPL forms of Trawny & Roumeliotis [trawny2005] — the product is their
/// Eq. (9) and the attitude matrix their Eq. (78), written here scalar-first
/// (\f$q_0\f$ leading) where that report is scalar-last (\f$q_4\f$ trailing). The
/// negative sign on \f$\mathbf{a}_v\times\mathbf{b}_v\f$ is the JPL convention;
/// it makes \f$A\f$ a homomorphism,
/// \f$A(\bar a \otimes \bar b) = A(\bar a)\,A(\bar b)\f$, and \f$A\f$ maps a
/// vector's reference-frame coordinates to the rotated frame,
/// \f$\mathbf{v}_\mathrm{rot} = A(\bar q)\,\mathbf{v}_\mathrm{ref}\f$
/// (Markley & Crassidis §2.9 [markley2014]).
class Quaternion {
 public:
  /// Identity rotation `[1, 0, 0, 0]`.
  Quaternion() = default;

  /// Construct from scalar-first components `[q0, q1, q2, q3]`.
  Quaternion(double q0, double q1, double q2, double q3) : q_(q0, q1, q2, q3) {}

  /// Construct from a scalar-first coefficient vector `[q0, q1, q2, q3]`.
  explicit Quaternion(const Eigen::Vector4d& coeffs_q0_first) : q_(coeffs_q0_first) {}

  /// Identity rotation.
  static Quaternion Identity() { return Quaternion(1.0, 0.0, 0.0, 0.0); }

  /// Quaternion for a rotation of @p angle_rad about @p unit_axis (assumed unit
  /// norm). Produces the passive transform `A(q)` about that axis:
  /// \f$\bar q = [\cos(\theta/2),\ \sin(\theta/2)\,\hat{\mathbf{n}}]\f$.
  static Quaternion FromAxisAngle(const Eigen::Vector3d& unit_axis, double angle_rad);

  /// Extract a quaternion from a proper rotation matrix `A` (Shepperd's
  /// numerically stable method [shepperd1978]). The extraction pivots on the
  /// largest of \f$\{4q_0^2, 4q_1^2, 4q_2^2, 4q_3^2\} - 1\f$, formed from the trace
  /// and diagonal of \f$A\f$, then recovers the remaining components from
  /// off-diagonal differences/sums divided by \f$4\times\f$ the pivot; the result
  /// is renormalised and made canonical. @p dcm must be orthonormal with det +1.
  static Quaternion FromRotationMatrix(const Eigen::Matrix3d& dcm);

  /// @name Accessors (scalar-first)
  /// @{
  double w() const { return q_[0]; }  ///< scalar part q0

  double x() const { return q_[1]; }

  double y() const { return q_[2]; }

  double z() const { return q_[3]; }

  double scalar() const { return q_[0]; }

  Eigen::Vector3d vec() const { return q_.tail<3>(); }

  /// Coefficients in scalar-first order `[q0, q1, q2, q3]`.
  const Eigen::Vector4d& coeffs() const { return q_; }

  /// @}

  double norm() const { return q_.norm(); }

  bool isUnit(double tol = 1e-9) const;

  /// True if all four components are finite (no NaN/Inf). Boundary guard for
  /// estimator/control outputs (design doc §3.6).
  bool isFinite() const { return q_.allFinite(); }

  /// Normalize in place. Returns false and leaves the value unchanged if the
  /// norm is below @p eps (degenerate; cannot safely normalize).
  bool normalize(double eps = 1e-12);

  /// Conjugate `[q0, -q1, -q2, -q3]` (inverse rotation for a unit quaternion).
  Quaternion conjugate() const;

  /// Inverse rotation (conjugate; assumes unit norm).
  Quaternion inverse() const { return conjugate(); }

  /// Canonical representative with `q0 >= 0` (q and -q are the same rotation;
  /// `q0 == 0` is left unchanged, as the sign is then ambiguous).
  Quaternion canonical() const;

  /// JPL-convention product: `A(a * b) = A(a) A(b)`.
  Quaternion operator*(const Quaternion& rhs) const;

  /// Passive attitude matrix `A(q)`: `v_rot = A(q) v_ref`.
  Eigen::Matrix3d toRotationMatrix() const;

  /// Rotate raw vector coordinates from the reference to the rotated frame.
  Eigen::Vector3d rotate(const Eigen::Vector3d& v) const;

  /// Smallest rotation angle [rad] between this and @p other (in [0, π]), from the
  /// error quaternion \f$\delta\bar q = \bar q_1^{-1} \otimes \bar q_2\f$ as
  /// \f$\theta = 2\,\mathrm{atan2}\big(\|\delta\mathbf q_v\|,\ |\delta q_0|\big)\f$.
  /// The absolute value maps \f$\delta\bar q\f$ and \f$-\delta\bar q\f$ (the same
  /// rotation) to the same angle, i.e. the \f$\theta \le \pi\f$ branch. The
  /// \c atan2 form is used rather than \f$2\arccos(\delta q_0)\f$ because the
  /// scalar part alone is stationary at \f$\theta = 0\f$ and so loses half its
  /// significant digits there — see the house rule in \c lib/README.md.
  double angularDistance(const Quaternion& other) const;

 private:
  /// Scalar-first `[q0, q1, q2, q3]`; identity is `[1, 0, 0, 0]`.
  Eigen::Vector4d q_{1.0, 0.0, 0.0, 0.0};
};

/// Boundary-tagged rotation. `Quat<To, From>` transforms vector coordinates
/// from frame @c From to frame @c To: `v_To = q.rotate(v_From)`. Mirrors the
/// design-doc notation `Quat<Body, ECI>` (Body ← ECI).
///
/// The frame tags are checked at compile time — a rotation only applies to a
/// vector in its @c From frame, and rotations compose only where the middle
/// frames match:
///
/// @code
/// Quat<Body, ECI> q_bi = attitude;      // Body ← ECI
/// Vec3<ECI>  r_eci = ...;
/// Vec3<Body> r_body = q_bi.rotate(r_eci);   // OK: input is ECI
/// // q_bi.rotate(r_body);                   // compile error: r_body is not ECI
///
/// Quat<ECI, ECEF> q_ie = ...;
/// Quat<Body, ECEF> q_be = q_bi * q_ie;      // OK: ECI cancels
/// // auto bad = q_ie * q_bi;                // compile error: ECEF ≠ Body
/// @endcode
///
/// @tparam To   destination frame tag
/// @tparam From source frame tag
template <class To, class From>
class Quat {
 public:
  /// Identity (To and From coincident).
  Quat() = default;

  /// Wrap a core quaternion as the rotation From -> To.
  explicit Quat(const Quaternion& q) : q_(q) {}

  static Quat Identity() { return Quat(Quaternion::Identity()); }

  const Quaternion& core() const { return q_; }

  Quaternion& core() { return q_; }

  /// Rotate vector coordinates From -> To.
  Vec3<To> rotate(const Vec3<From>& v) const { return Vec3<To>(q_.rotate(v.eigen())); }

  Vec3<To> operator*(const Vec3<From>& v) const { return rotate(v); }

  /// Inverse orientation, To -> From.
  Quat<From, To> inverse() const { return Quat<From, To>(q_.inverse()); }

  bool normalize(double eps = 1e-12) { return q_.normalize(eps); }

  Quat canonical() const { return Quat(q_.canonical()); }

 private:
  Quaternion q_{};
};

/// Compose rotations: `(To <- Mid) * (Mid <- From) = (To <- From)`.
template <class To, class Mid, class From>
inline Quat<To, From> operator*(const Quat<To, Mid>& a, const Quat<Mid, From>& b) {
  return Quat<To, From>(a.core() * b.core());
}

}  // namespace polaris::math

#endif  // POLARIS_MATH_QUATERNION_HPP
