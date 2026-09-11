#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <cmath>
#include <vector>

#include "fused_icp/se3.hpp"

/// The residual algebra of "Point-to-Point + Point-to-Plane Residual/Jacobian 상세 유도" as plain
/// Eigen functions. Everything here is independent of Ceres, so the cost function, the demo and the
/// tests all call the same formulas. Equation numbers refer to that note.
///
///   e   = R p + t - q                                   (3)  geometric error of one pair
///   G   = de/ddelta_xi = [-R[p]x  R]  in R^{3x6}         (20) geometric Jacobian, right perturbation
///   Omega = alpha I + beta n n^T                         (33) information matrix of the fused cost
///   L   = sqrt(alpha) I + (sqrt(alpha+beta) - sqrt(alpha)) n n^T,  L^T L = Omega   (41)
///   r_f = L e                                            (42) fused 3-D residual
///   J_f = L G                                            (44) its Jacobian
///
/// For the same pair 2C = alpha |e|^2 + beta (n^T e)^2 = |r_f|^2, so one 3-D residual carries both
/// terms; the tangent-plane directions are weighted by alpha, the normal direction by alpha + beta.
namespace fused_icp
{
using Matrix36 = Eigen::Matrix<double, 3, 6>;
using Matrix16 = Eigen::Matrix<double, 1, 6>;
using Matrix66 = Eigen::Matrix<double, 6, 6>;
using Vector6 = Eigen::Matrix<double, 6, 1>;

/// The two non-negative weights of the fused cost. alpha = 0 is pure point-to-plane, beta = 0 is
/// pure point-to-point. alpha > 0 makes Omega positive definite.
struct FusedWeights
{
    double alpha = 1.0;  ///< point-to-point weight, e.g. 1 / sigma_pt^2
    double beta = 1.0;  ///< point-to-plane weight, e.g. 1 / sigma_pl^2
};

/**
 * @brief Geometric error e = R p + t - q, note (3)
 *
 * @param pose source -> target transform T = (R, t)
 * @param source_point p, in the source frame
 * @param target_point q, in the target frame
 * @return e
 */
inline Eigen::Vector3d geometric_error(const Eigen::Isometry3d& pose, const Eigen::Vector3d& source_point, const Eigen::Vector3d& target_point)
{
    return pose * source_point - target_point;
}

/**
 * @brief Geometric Jacobian G = de / ddelta_xi at delta_xi = 0 for T+ = T Exp(delta_xi), note (20)
 *
 * e+ = R (I + [delta_phi]x) p + R delta_rho + t - q = e - R [p]x delta_phi + R delta_rho, hence
 * G = [-R [p]x   R]. The rotation block is the source point in the *source* frame, the translation
 * block carries R because delta_rho is expressed in the source frame.
 *
 * @param rotation R
 * @param source_point p, in the source frame
 * @return G in R^{3x6}, columns [delta_phi | delta_rho]
 */
inline Matrix36 geometric_jacobian(const Eigen::Matrix3d& rotation, const Eigen::Vector3d& source_point)
{
    Matrix36 G;
    G.leftCols<3>() = -rotation * se3::skew(source_point);
    G.rightCols<3>() = rotation;
    return G;
}

/**
 * @brief Information matrix Omega = alpha I + beta n n^T, note (33)
 *
 * @param normal n, unit target normal
 * @param weights alpha, beta
 * @return Omega
 */
inline Eigen::Matrix3d information_matrix(const Eigen::Vector3d& normal, const FusedWeights& weights)
{
    return weights.alpha * Eigen::Matrix3d::Identity() + weights.beta * normal * normal.transpose();
}

/**
 * @brief Symmetric square root L of Omega, note (41): L = sqrt(alpha) I + c n n^T with L^T L = Omega
 *
 * c = sqrt(alpha + beta) - sqrt(alpha) is the only root of c^2 + 2 sqrt(alpha) c - beta = 0 that
 * keeps L positive semidefinite, note (40).
 *
 * @param normal n, unit target normal
 * @param weights alpha, beta
 * @return L
 */
inline Eigen::Matrix3d fusion_whitener(const Eigen::Vector3d& normal, const FusedWeights& weights)
{
    const double sqrt_alpha = std::sqrt(weights.alpha);
    const double c = std::sqrt(weights.alpha + weights.beta) - sqrt_alpha;
    return sqrt_alpha * Eigen::Matrix3d::Identity() + c * normal * normal.transpose();
}

/**
 * @brief Fused residual r_f = L e = sqrt(alpha) e + c n (n^T e), note (42)
 *
 * @param error e
 * @param normal n, unit target normal
 * @param weights alpha, beta
 * @return r_f in R^3
 */
inline Eigen::Vector3d fused_residual(const Eigen::Vector3d& error, const Eigen::Vector3d& normal, const FusedWeights& weights)
{
    const double sqrt_alpha = std::sqrt(weights.alpha);
    const double c = std::sqrt(weights.alpha + weights.beta) - sqrt_alpha;
    return sqrt_alpha * error + c * normal * normal.dot(error);
}

/**
 * @brief Fused Jacobian J_f = L G = sqrt(alpha) G + c n (n^T G), note (44)
 *
 * L is held fixed (n does not move with the pose inside one linearization), so the chain rule is
 * just the left multiplication by L.
 *
 * @param G geometric Jacobian from geometric_jacobian()
 * @param normal n, unit target normal
 * @param weights alpha, beta
 * @return J_f in R^{3x6}
 */
inline Matrix36 fused_jacobian(const Matrix36& G, const Eigen::Vector3d& normal, const FusedWeights& weights)
{
    const double sqrt_alpha = std::sqrt(weights.alpha);
    const double c = std::sqrt(weights.alpha + weights.beta) - sqrt_alpha;
    const Matrix16 a = normal.transpose() * G;  // the point-to-plane row, n^T G, note (27)
    return sqrt_alpha * G + c * normal * a;
}

/**
 * @brief Huber weight of the IRLS approximation, note (64), on s = e^T Omega e = |r_f|^2
 *
 * @param squared_norm s
 * @param huber_delta delta_H; 0 or negative disables robustification (w = 1)
 * @return w(s)
 */
inline double huber_weight(double squared_norm, double huber_delta)
{
    if (huber_delta <= 0.0 || squared_norm <= huber_delta * huber_delta)
    {
        return 1.0;
    }
    return huber_delta / std::sqrt(squared_norm);
}

/// One correspondence in the frame the increment acts on: p in the source frame, q and n in the
/// target frame.
struct Correspondence
{
    Eigen::Vector3d source_point{Eigen::Vector3d::Zero()};  ///< p
    Eigen::Vector3d target_point{Eigen::Vector3d::Zero()};  ///< q
    Eigen::Vector3d target_normal{Eigen::Vector3d::Zero()};  ///< n, unit
};

/// The Gauss-Newton normal equations H delta_xi = b of note (49)/(50).
struct NormalEquations
{
    Matrix66 H = Matrix66::Zero();  ///< sum_i w_i G_i^T Omega_i G_i = sum_i w_i J_f,i^T J_f,i
    Vector6 b = Vector6::Zero();  ///< -sum_i w_i G_i^T Omega_i e_i = -sum_i w_i J_f,i^T r_f,i
    double cost = 0.0;  ///< 1/2 sum_i rho(|r_f,i|^2) with the Huber rho of note (64)
};

/**
 * @brief Direct accumulation of H and b without forming L, note (50) with the IRLS weight of (64)
 *
 * This is the C++ loop of the note, kept as a reference implementation. The Ceres path of
 * IcpFusedPointPlane assembles exactly the same H and b from FusedPointPlaneCostFunction
 * (r_f, J_f), which test_icp checks numerically. Ceres' Huber corrector additionally applies the
 * rho'' curvature term, so beyond the IRLS approximation the two differ only when the loss is active.
 *
 * @param pose current T
 * @param correspondences pairs in the source / target frames
 * @param weights alpha, beta
 * @param huber_delta delta_H, 0 disables the loss
 * @return H, b and the robust cost at T
 */
inline NormalEquations accumulate_normal_equations(const Eigen::Isometry3d& pose, const std::vector<Correspondence>& correspondences,
                                                   const FusedWeights& weights, double huber_delta)
{
    NormalEquations equations;
    const Eigen::Matrix3d& R = pose.linear();
    for (const Correspondence& pair : correspondences)
    {
        const Eigen::Vector3d e = geometric_error(pose, pair.source_point, pair.target_point);
        const Matrix36 G = geometric_jacobian(R, pair.source_point);
        const Matrix16 a = pair.target_normal.transpose() * G;  // n^T G
        const double d = pair.target_normal.dot(e);  // n^T e

        const double s = weights.alpha * e.squaredNorm() + weights.beta * d * d;  // e^T Omega e
        const double w = huber_weight(s, huber_delta);

        equations.H += w * (weights.alpha * G.transpose() * G + weights.beta * a.transpose() * a);
        equations.b -= w * (weights.alpha * G.transpose() * e + weights.beta * a.transpose() * d);

        // rho(s) of note (64), halved like every cost in the note.
        const double rho = (huber_delta <= 0.0 || s <= huber_delta * huber_delta) ? s : 2.0 * huber_delta * std::sqrt(s) - huber_delta * huber_delta;
        equations.cost += 0.5 * rho;
    }
    return equations;
}

}  // namespace fused_icp
