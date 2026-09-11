#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <cmath>

/// Minimal SO(3) Lie algebra helpers (Hat / Exp / Log).
///
/// The rotation group is a 3-D manifold, not a vector space, so an additive update R + dR is
/// invalid. States live on the manifold and updates are tangent-space (R^3) vectors mapped back
/// through Exp.
namespace imu_preint
{
namespace so3
{
/**
 * @brief Hat operator: [phi]x b = phi x b
 *
 * @param v phi in R^3
 * @return [phi]x in so(3), a skew-symmetric matrix
 */
inline Eigen::Matrix3d Hat(const Eigen::Vector3d& v)
{
    Eigen::Matrix3d m;
    // clang-format off
  m <<     0.0, -v.z(),  v.y(),
         v.z(),    0.0, -v.x(),
        -v.y(),  v.x(),    0.0;
    // clang-format on
    return m;
}

/**
 * @brief Exponential map, Exp(phi) = I + sin(t) K + (1 - cos t) K^2 with t = |phi|, K = [phi/t]x
 *
 * @param phi rotation vector, axis times angle [rad]
 * @return Exp(phi) in SO(3)
 */
inline Eigen::Matrix3d Exp(const Eigen::Vector3d& phi)
{
    const double theta = phi.norm();
    if (theta < 1e-10)
    {
        // phi/theta is 0/0 as theta -> 0, so fall back to the first-order Exp(phi) ~ I + [phi]x.
        return Eigen::Matrix3d::Identity() + Hat(phi);
    }
    const Eigen::Matrix3d K = Hat(phi / theta);
    return Eigen::Matrix3d::Identity() + std::sin(theta) * K + (1.0 - std::cos(theta)) * K * K;
}

/**
 * @brief Inverse of Exp(), computed through Eigen::AngleAxis for numerical stability
 *
 * @param R rotation matrix
 * @return rotation vector phi [rad]
 */
inline Eigen::Vector3d Log(const Eigen::Matrix3d& R)
{
    const Eigen::AngleAxisd aa(R);
    return aa.angle() * aa.axis();
}

inline Eigen::Matrix3d RightJacobian(const Eigen::Vector3d& phi)
{
    const double t2 = phi.squaredNorm();
    const Eigen::Matrix3d h = Hat(phi);
    if (t2 < 1e-8)
        return Eigen::Matrix3d::Identity() - (0.5 - t2 / 24.0) * h + (1.0 / 6.0 - t2 / 120.0) * h * h;
    const double t = std::sqrt(t2);
    return Eigen::Matrix3d::Identity() - (1.0 - std::cos(t)) / t2 * h + (t - std::sin(t)) / (t2 * t) * h * h;
}

/// d Log(R Exp(e)) / de at e = 0, with phi = Log(R).
inline Eigen::Matrix3d RightJacobianInverse(const Eigen::Vector3d& phi)
{
    const double t2 = phi.squaredNorm();
    const Eigen::Matrix3d h = Hat(phi);
    double coefficient;
    if (t2 < 1e-8)
    {
        coefficient = 1.0 / 12.0 + t2 / 720.0 + t2 * t2 / 30240.0;
    }
    else
    {
        const double t = std::sqrt(t2);
        // Half-angle form remains stable near pi (the principal Log branch boundary).
        coefficient = (1.0 - 0.5 * t / std::tan(0.5 * t)) / t2;
    }
    return Eigen::Matrix3d::Identity() + 0.5 * h + coefficient * h * h;
}

}  // namespace so3
}  // namespace imu_preint
