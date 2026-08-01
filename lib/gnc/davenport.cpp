/// @file
/// @brief Davenport q-method implementation (design doc §8.1; REQ-ADET-003).
/// See davenport.hpp for conventions, assumptions, and references.

#include "gnc/davenport.hpp"

#include <cmath>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>

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

/// Fisher information of an observation set, `M = Σ wᵢ(I − ûᵢûᵢᵀ)`, together
/// with the observability ratio `λ_min/λ_max` that gates it. `M` is symmetric
/// positive semi-definite by construction, so the ratio is in `[0, 1]` and
/// vanishes exactly when the directions are collinear.
double informationRatio(const Eigen::Matrix3d& m) {
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver;
  solver.computeDirect(m, Eigen::EigenvaluesOnly);
  if (solver.info() != Eigen::Success) {
    return 0.0;
  }
  const double lambda_min = solver.eigenvalues()(0);
  const double lambda_max = solver.eigenvalues()(2);
  if (!(lambda_max > 0.0) || !std::isfinite(lambda_min)) {
    return 0.0;
  }
  return lambda_min / lambda_max;
}

}  // namespace

bool davenport(const DavenportInput& in, DavenportSolution& out) {
  out = DavenportSolution{};

  // The gate is a ratio of eigenvalues, so it is only meaningful in (0, 1): a
  // value above 1 is unreachable and would silently refuse every solve forever.
  if (in.count < 2 || in.count > DavenportInput::kMaxObservations ||
      !(in.min_observability > 0.0) || !(in.min_observability < 1.0)) {
    return false;
  }

  // --- Weighted attitude profile ------------------------------------------
  // One bounded pass builds everything: the profile matrix B, the total weight
  // (for the Wahba loss), and the two Fisher information matrices whose
  // conditioning decides whether the set is solvable at all.
  Eigen::Matrix3d b_profile = Eigen::Matrix3d::Zero();
  Eigen::Matrix3d info_body = Eigen::Matrix3d::Zero();
  Eigen::Matrix3d info_ref = Eigen::Matrix3d::Zero();
  Eigen::Vector3d z = Eigen::Vector3d::Zero();
  double weight_sum = 0.0;

  for (int i = 0; i < in.count; ++i) {
    const VectorObservation& obs = in.observations[i];
    if (!(obs.sigma_rad > 0.0) || !std::isfinite(obs.sigma_rad)) {
      return false;
    }
    Eigen::Vector3d b_hat;
    Eigen::Vector3d r_hat;
    if (!unit(obs.body.eigen(), b_hat) || !unit(obs.reference.eigen(), r_hat)) {
      return false;
    }

    // Maximum-likelihood weight: w = 1/σ². This is the choice that makes the
    // optimum efficient and its covariance the inverse Fisher information — a
    // different weighting would still solve Wahba, but the covariance below
    // would then be a claim the estimator has not earned.
    const double w = 1.0 / (obs.sigma_rad * obs.sigma_rad);
    if (!std::isfinite(w)) {
      return false;
    }
    weight_sum += w;
    b_profile += w * b_hat * r_hat.transpose();
    z += w * b_hat.cross(r_hat);

    const Eigen::Matrix3d id = Eigen::Matrix3d::Identity();
    info_body += w * (id - b_hat * b_hat.transpose());
    info_ref += w * (id - r_hat * r_hat.transpose());
  }

  // --- Observability gate --------------------------------------------------
  // Gated in **both** frames, for the same reason TRIAD is (§8.1): a body set
  // that spans three axes while the reference set does not means the
  // measurements disagree with the models by more than the geometry, which is
  // not a solution worth publishing.
  const double ratio_body = informationRatio(info_body);
  const double ratio_ref = informationRatio(info_ref);
  out.observability = ratio_body;
  if (!(ratio_body >= in.min_observability) || !(ratio_ref >= in.min_observability)) {
    return false;
  }

  // --- K matrix and its dominant eigenvector -------------------------------
  // K written scalar-first to match the JPL layout (davenport.hpp): the top-left
  // scalar is σ = tr B, the first row/column carry z, and the 3×3 block is
  // S − σI. The gain function is exactly q̄ᵀKq̄, so the maximiser over unit
  // quaternions is the eigenvector of the largest eigenvalue.
  const double sigma = b_profile.trace();
  const Eigen::Matrix3d s = b_profile + b_profile.transpose();
  Eigen::Matrix4d k = Eigen::Matrix4d::Zero();
  k(0, 0) = sigma;
  k.block<1, 3>(0, 1) = z.transpose();
  k.block<3, 1>(1, 0) = z;
  k.block<3, 3>(1, 1) = s - sigma * Eigen::Matrix3d::Identity();
  if (!k.allFinite()) {
    return false;
  }

  Eigen::SelfAdjointEigenSolver<Eigen::Matrix4d> solver;
  solver.compute(k);
  if (solver.info() != Eigen::Success) {
    return false;  // iteration budget exhausted; refuse rather than guess
  }
  // Eigen sorts eigenvalues in increasing order, so the optimum is the last.
  const double lambda_max = solver.eigenvalues()(3);
  math::Quaternion q(Eigen::Vector4d(solver.eigenvectors().col(3)));
  if (!q.normalize()) {
    return false;
  }
  q = q.canonical();

  // --- Covariance ----------------------------------------------------------
  // Inverse Fisher information on the body observations. The observability gate
  // above bounds the condition number of `info_body`, so the inverse is taken
  // on a matrix already known to be well away from singular — but the result is
  // still checked, because "well conditioned" is not "finite".
  const Eigen::Matrix3d cov = info_body.inverse();
  if (!cov.allFinite() || !q.isFinite() || !std::isfinite(lambda_max)) {
    return false;
  }

  out.attitude = math::Quat<math::frames::Body, math::frames::ECI>(q);
  out.covariance = 0.5 * (cov + cov.transpose());
  // Wahba loss at the optimum. Clamped at zero: it is non-negative in exact
  // arithmetic, and a round-off-sized negative would read as a fault signature.
  out.loss = std::fmax(weight_sum - lambda_max, 0.0);
  out.valid = true;
  return true;
}

}  // namespace polaris::gnc
