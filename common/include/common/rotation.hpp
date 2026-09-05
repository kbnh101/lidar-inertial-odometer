#pragma once

#include <Eigen/Core>

namespace common
{
/**
 * @brief Z-Y-X Euler angles to a rotation matrix
 *
 * This is the rotation convention used throughout the project.
 *   theta = (alpha, beta, gamma), roll about x / pitch about y / yaw about z
 *   R(theta) = Rz(gamma) Ry(beta) Rx(alpha)
 *
 * @param alpha roll [rad]
 * @param beta  pitch [rad]
 * @param gamma yaw [rad]
 * @return R(theta) in SO(3)
 */
Eigen::Matrix3d euler_zyx_to_rotation(double alpha, double beta, double gamma);

/**
 * @brief Vector argument overload of the function above
 *
 * @param euler_abg (alpha, beta, gamma)
 * @return R(theta)
 */
Eigen::Matrix3d euler_zyx_to_rotation(const Eigen::Vector3d& euler_abg);

/**
 * @brief Derivative of R with respect to alpha
 *
 * @param alpha roll [rad]
 * @param beta  pitch [rad]
 * @param gamma yaw [rad]
 * @return dR/dalpha
 */
Eigen::Matrix3d rotation_derivative_alpha(double alpha, double beta, double gamma);

/**
 * @brief Derivative of R with respect to beta
 *
 * @param alpha roll [rad]
 * @param beta  pitch [rad]
 * @param gamma yaw [rad]
 * @return dR/dbeta
 */
Eigen::Matrix3d rotation_derivative_beta(double alpha, double beta, double gamma);

/**
 * @brief Derivative of R with respect to gamma
 *
 * @param alpha roll [rad]
 * @param beta  pitch [rad]
 * @param gamma yaw [rad]
 * @return dR/dgamma
 */
Eigen::Matrix3d rotation_derivative_gamma(double alpha, double beta, double gamma);

/**
 * @brief Inverse of euler_zyx_to_rotation(), used only to print the final pose for a human
 *
 * @param rotation R(theta)
 * @return (alpha, beta, gamma) [rad]
 */
Eigen::Vector3d rotation_to_euler_zyx(const Eigen::Matrix3d& rotation);

}  // namespace common
