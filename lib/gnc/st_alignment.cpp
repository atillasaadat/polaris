/// @file
/// @brief Inter-star-tracker alignment calibration (design doc §8.2;
/// REQ-ADET-013). See st_alignment.hpp for the king-tracker argument, the
/// maximum-eigenvalue average [markley2007] and every refusal it carries.

#include "gnc/st_alignment.hpp"

#include <cmath>
#include <Eigen/Eigenvalues>

namespace polaris::gnc {

namespace {

namespace pm = polaris::math;
using Body = pm::frames::Body;
using ECI = pm::frames::ECI;

/// Column vector [q0,q1,q2,q3] of @p q, in the JPL scalar-first layout the whole
/// repository uses (§3.3). The outer product below is sign-invariant, so no
/// canonicalisation is needed — and none is done, deliberately: forcing q0 ≥ 0
/// before accumulating would be a no-op for the estimate and a chance to get the
/// hemisphere wrong on a sample near q0 = 0.
Eigen::Vector4d asVector(const pm::Quaternion& q) {
  return Eigen::Vector4d(q.scalar(), q.vec().x(), q.vec().y(), q.vec().z());
}

pm::Quaternion fromVector(const Eigen::Vector4d& v) {
  return pm::Quaternion(v[0], v[1], v[2], v[3]);
}

}  // namespace

bool StAlignmentConfig::isValid() const {
  return min_samples >= 2 && std::isfinite(max_residual_rad) && max_residual_rad > 0.0 &&
         std::isfinite(min_eigen_gap) && min_eigen_gap > 0.0 && min_eigen_gap < 1.0;
}

StAlignmentAccumulator::StAlignmentAccumulator(const StAlignmentConfig& config) {
  if (config.isValid()) {
    config_ = config;
    configured_ = true;
  }
}

void StAlignmentAccumulator::reset() {
  moment_.setZero();
  samples_ = 0;
}

bool StAlignmentAccumulator::addSample(const pm::Quat<Body, ECI>& king,
                                       const pm::Quat<Body, ECI>& second) {
  if (!configured_) {
    return false;
  }
  pm::Quaternion q_king = king.core();
  pm::Quaternion q_second = second.core();
  if (!q_king.isFinite() || !q_second.isFinite() || !q_king.normalize() || !q_second.normalize()) {
    return false;
  }

  // q_rel = q_2 ⊗ q_k⁻¹, the Body ← Body residual misalignment. Constant by
  // hypothesis; the eigen-gap gate in fit() is what tests that hypothesis.
  pm::Quaternion q_rel = q_second * q_king.inverse();
  if (!q_rel.isFinite() || !q_rel.normalize()) {
    return false;
  }

  const Eigen::Vector4d v = asVector(q_rel);
  moment_.noalias() += v * v.transpose();
  if (!moment_.allFinite()) {
    // Cannot happen from unit quaternions, and checked because an accumulator that
    // has gone non-finite would otherwise produce a plausible-looking refusal
    // rather than a diagnosable one.
    moment_.setZero();
    samples_ = 0;
    return false;
  }
  ++samples_;
  return true;
}

StAlignmentRejection StAlignmentAccumulator::fit(StAlignmentResult& out) const {
  out = StAlignmentResult{};
  if (!configured_) {
    return StAlignmentRejection::kConfig;
  }
  if (samples_ < config_.min_samples) {
    return StAlignmentRejection::kSamples;
  }

  Eigen::SelfAdjointEigenSolver<Eigen::Matrix4d> solver;
  // Symmetrised first: the algebra is symmetric but the accumulated products are
  // not bitwise symmetric, and the solver is entitled to read one triangle.
  solver.compute(0.5 * (moment_ + moment_.transpose()));
  if (solver.info() != Eigen::Success) {
    return StAlignmentRejection::kNumerical;
  }

  // Eigen orders eigenvalues ascending, so the largest is last and the runner-up
  // one before it.
  const Eigen::Vector4d eigenvalues = solver.eigenvalues();
  const double n = static_cast<double>(samples_);
  const double lambda_max = eigenvalues[3];
  const double lambda_second = eigenvalues[2];
  if (!std::isfinite(lambda_max) || !std::isfinite(lambda_second)) {
    return StAlignmentRejection::kNumerical;
  }

  const double eigen_gap = (lambda_max - lambda_second) / n;
  // λ_max = Σ cos²(θ_i/2), so this is 2·sqrt(⟨sin²(θ/2)⟩) exactly. Clamped at zero
  // against round-off only: λ_max ≤ N by construction, so a negative argument is
  // never physical.
  const double mean_cos_sq = lambda_max / n;
  const double residual = 2.0 * std::sqrt(std::fmax(0.0, 1.0 - mean_cos_sq));
  if (!std::isfinite(eigen_gap) || !std::isfinite(residual)) {
    return StAlignmentRejection::kNumerical;
  }

  // Order matters for the reported reason, not for the verdict: a set of pairs
  // that do not share a rotation also has a large residual, and "these pairs do
  // not describe one rotation" is the more actionable report than "the dispersion
  // is large".
  if (eigen_gap < config_.min_eigen_gap) {
    return StAlignmentRejection::kDegenerate;
  }
  if (residual > config_.max_residual_rad) {
    return StAlignmentRejection::kDispersion;
  }

  pm::Quaternion average = fromVector(solver.eigenvectors().col(3));
  if (!average.isFinite() || !average.normalize()) {
    return StAlignmentRejection::kNumerical;
  }
  average = average.canonical();

  // Published as the *inverse*: the caller applies it, and applying a correction
  // should read as "put this reading in the king's frame" rather than as an
  // inversion at every call site.
  pm::Quaternion correction = average.inverse();
  if (!correction.isFinite() || !correction.normalize()) {
    return StAlignmentRejection::kNumerical;
  }

  out.correction = pm::Quat<Body, Body>(correction.canonical());
  out.residual_angle_rad = residual;
  out.eigen_gap = eigen_gap;
  // 2·atan2(‖v‖, |q₀|) rather than 2·acos(q₀) (§3.3, lib/README.md): the total
  // rotation the correction removes, well-conditioned at the small angles a
  // healthy mounting produces.
  const pm::Quaternion canonical_average = average;
  out.misalignment_angle_rad =
      2.0 * std::atan2(canonical_average.vec().norm(), std::fabs(canonical_average.scalar()));
  out.samples = samples_;
  out.valid = true;
  if (!std::isfinite(out.misalignment_angle_rad)) {
    out = StAlignmentResult{};
    return StAlignmentRejection::kNumerical;
  }
  return StAlignmentRejection::kNone;
}

pm::Quat<Body, ECI> applyStAlignment(const StAlignmentResult& alignment,
                                     const pm::Quat<Body, ECI>& measured) {
  if (!alignment.valid) {
    return measured;
  }
  pm::Quaternion corrected = alignment.correction.core() * measured.core();
  if (!corrected.isFinite() || !corrected.normalize()) {
    // A correction that cannot be applied is not applied: returning the raw
    // reading is strictly better than returning something non-finite behind a
    // validity flag the caller has already set.
    return measured;
  }
  return pm::Quat<Body, ECI>(corrected.canonical());
}

}  // namespace polaris::gnc
