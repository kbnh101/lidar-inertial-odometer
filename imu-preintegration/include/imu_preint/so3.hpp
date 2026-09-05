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

}  // namespace so3
}  // namespace imu_preint
