#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <cmath>

/// Minimal SO(3) / SE(3) Lie group helpers for the right perturbation T+ = T Exp(delta_xi).
///
/// The derivation note fixes the convention once and the whole package sticks to it:
///   delta_xi = [delta_phi; delta_rho] in R^6, rotation first, then translation,
///   T+ = T Exp(delta_xi^),   i.e.  R+ = R exp([delta_phi]x),   t+ = t + R V(delta_phi) delta_rho.
/// delta_rho is a small translation expressed in the current source/body frame, which is why the
/// translation block of the geometric Jacobian carries R. The finite retraction keeps the exact
/// SE(3) exponential (with V) so that the numeric Jacobian check can use the same map.
namespace fused_icp
{
namespace se3
{
/**
 * @brief Hat operator: [a]x b = a x b (note (12) of the derivation)
 *
 * @param a vector in R^3
 * @return the skew-symmetric matrix [a]x
 */
inline Eigen::Matrix3d skew(const Eigen::Vector3d& a)
{
    Eigen::Matrix3d m;
    // clang-format off
    m <<    0.0, -a.z(),  a.y(),
          a.z(),    0.0, -a.x(),
         -a.y(),  a.x(),    0.0;
    // clang-format on
    return m;
}

/**
 * @brief SO(3) exponential, exp([phi]x) = I + sin(t)/t [phi]x + (1 - cos t)/t^2 [phi]x^2
 *
 * @param phi rotation vector, axis times angle [rad]
 * @return the rotation matrix
 */
inline Eigen::Matrix3d exp_so3(const Eigen::Vector3d& phi)
{
    const double theta_sq = phi.squaredNorm();
    const Eigen::Matrix3d K = skew(phi);
    if (theta_sq < 1e-16)
    {
        // Second order Taylor expansion; it keeps the result orthonormal to O(theta^3).
        return Eigen::Matrix3d::Identity() + K + 0.5 * K * K;
    }
    const double theta = std::sqrt(theta_sq);
    return Eigen::Matrix3d::Identity() + (std::sin(theta) / theta) * K + ((1.0 - std::cos(theta)) / theta_sq) * K * K;
}

/**
 * @brief SO(3) logarithm, inverse of exp_so3(), through Eigen::AngleAxis for stability
 *
 * @param R rotation matrix
 * @return rotation vector [rad]
 */
inline Eigen::Vector3d log_so3(const Eigen::Matrix3d& R)
{
    const Eigen::AngleAxisd aa(R);
    return aa.angle() * aa.axis();
}

/**
 * @brief The V matrix of the SE(3) exponential, note (22): V = I + (1 - cos t)/t^2 K + (t - sin t)/t^3 K^2
 *
 * V delta_rho = delta_rho + O(|delta_xi|^2), so V does not enter the first order Jacobian, only the
 * finite retraction.
 *
 * @param phi rotation vector [rad]
 * @return V(phi)
 */
inline Eigen::Matrix3d left_jacobian_so3(const Eigen::Vector3d& phi)
{
    const double theta_sq = phi.squaredNorm();
    const Eigen::Matrix3d K = skew(phi);
    if (theta_sq < 1e-16)
    {
        return Eigen::Matrix3d::Identity() + 0.5 * K + (1.0 / 6.0) * K * K;
    }
    const double theta = std::sqrt(theta_sq);
    return Eigen::Matrix3d::Identity() + ((1.0 - std::cos(theta)) / theta_sq) * K + ((theta - std::sin(theta)) / (theta_sq * theta)) * K * K;
}

/**
 * @brief Inverse of left_jacobian_so3()
 *
 * @param phi rotation vector [rad]
 * @return V(phi)^-1
 */
inline Eigen::Matrix3d left_jacobian_so3_inverse(const Eigen::Vector3d& phi)
{
    const double theta_sq = phi.squaredNorm();
    const Eigen::Matrix3d K = skew(phi);
    if (theta_sq < 1e-16)
    {
        return Eigen::Matrix3d::Identity() - 0.5 * K + (1.0 / 12.0) * K * K;
    }
    const double theta = std::sqrt(theta_sq);
    const double half = 0.5 * theta;
    // 1/t^2 - (1 + cos t) / (2 t sin t) = 1/t^2 - cot(t/2) / (2 t)
    const double coefficient = 1.0 / theta_sq - std::cos(half) / (2.0 * theta * std::sin(half));
    return Eigen::Matrix3d::Identity() - 0.5 * K + coefficient * K * K;
}

/**
 * @brief SE(3) exponential of a tangent vector, note (21): Exp([delta_phi; delta_rho])
 *
 * @param xi [delta_phi; delta_rho], rotation first
 * @return the rigid transform (R, t) = (exp([phi]x), V(phi) rho)
 */
inline Eigen::Isometry3d exp_se3(const Eigen::Matrix<double, 6, 1>& xi)
{
    const Eigen::Vector3d phi = xi.head<3>();
    const Eigen::Vector3d rho = xi.tail<3>();
    Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
    T.linear() = exp_so3(phi);
    T.translation() = left_jacobian_so3(phi) * rho;
    return T;
}

/**
 * @brief SE(3) logarithm, inverse of exp_se3()
 *
 * @param T rigid transform
 * @return [phi; rho], rotation first
 */
inline Eigen::Matrix<double, 6, 1> log_se3(const Eigen::Isometry3d& T)
{
    Eigen::Matrix<double, 6, 1> xi;
    const Eigen::Vector3d phi = log_so3(T.linear());
    xi.head<3>() = phi;
    xi.tail<3>() = left_jacobian_so3_inverse(phi) * T.translation();
    return xi;
}

/**
 * @brief Right perturbation of the derivation, note (1): T+ = T Exp(delta_xi)
 *
 * @param T current pose
 * @param delta_xi tangent increment, rotation first
 * @return the retracted pose
 */
inline Eigen::Isometry3d retract(const Eigen::Isometry3d& T, const Eigen::Matrix<double, 6, 1>& delta_xi)
{
    return T * exp_se3(delta_xi);
}

}  // namespace se3
}  // namespace fused_icp
