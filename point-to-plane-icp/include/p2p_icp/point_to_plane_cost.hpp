#pragma once

#include <ceres/sized_cost_function.h>

#include <Eigen/Core>

#include "p2p_icp/rotation.hpp"

namespace p2p_icp
{
/**
 * @brief Point-to-plane residual
 *
 *   parameter block  xi = [tx, ty, tz, alpha, beta, gamma]^T in R^6
 *   residual         r_n = n_y^T (R(theta) x_n + t - y_n)
 *   Jacobian (1x6)   dr/dt = n_y^T,  dr/dtheta_i = n_y^T (dR/dtheta_i) x_n
 *
 * The residual is a 1-D scalar, hence the `ceres::SizedCostFunction<1, 6>` base.
 *
 * Two points worth noting.
 * - Only the *target* normal n_y enters the residual. Point-to-plane is the distance to the target
 *   tangent plane, so the source normal n_x never appears; n_x is used solely by the normal
 *   agreement test of the correspondence search.
 * - xi is an ordinary R^6 vector updated additively, so no `ceres::Manifold` is attached. Instead
 *   `IcpPointToPlane::do_icp()` re-solves xi from 0 every outer iteration, keeping the Euler angles
 *   near 0 and away from the beta = +-pi/2 gimbal lock.
 */
class PointToPlaneCostFunction : public ceres::SizedCostFunction<1, 6>
{
public:
    /**
     * @brief Constructs the cost function for one correspondence pair
     *
     * @param source_point  x_n, the source point in the frame the parameter block transforms
     * @param target_point  y_n, the corresponding target point
     * @param target_normal n_y, assumed to be a unit vector (load_point_cloud() normalizes on load)
     */
    PointToPlaneCostFunction(const Eigen::Vector3d& source_point, const Eigen::Vector3d& target_point, const Eigen::Vector3d& target_normal)
        : source_point_(source_point), target_point_(target_point), target_normal_(target_normal)
    {
    }

    /**
     * @brief Computes the residual and the analytic Jacobian
     *
     * Ceres assembles the Gauss-Newton normal equations (sum J^T J) dxi = -sum J^T r internally, so
     * only residuals[0] and jacobians[0][0..5] have to be filled in here.
     *
     * @param parameters parameters[0] = xi = [tx, ty, tz, alpha, beta, gamma]
     * @param residuals  output: r_n, a single value
     * @param jacobians  output: J_n in R^{1x6}; nullptr means only the residual was requested
     * @return always true, since the computation cannot fail
     */
    bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const override
    {
        const Eigen::Map<const Eigen::Vector3d> translation(parameters[0]);
        const double alpha = parameters[0][3];
        const double beta = parameters[0][4];
        const double gamma = parameters[0][5];

        // R(theta) = Rz(gamma) Ry(beta) Rx(alpha)
        const Eigen::Matrix3d rotation = euler_zyx_to_rotation(alpha, beta, gamma);

        // e_n = R(theta) x_n + t - y_n
        const Eigen::Vector3d error = rotation * source_point_ + translation - target_point_;

        // r_n = n_y^T e_n, which is exactly the signed point-to-plane distance.
        residuals[0] = target_normal_.dot(error);

        if (jacobians != nullptr && jacobians[0] != nullptr)
        {
            double* jacobian = jacobians[0];

            // dr/dt = n_y^T I_3 = n_y^T, so the normal components go in directly.
            jacobian[0] = target_normal_.x();
            jacobian[1] = target_normal_.y();
            jacobian[2] = target_normal_.z();

            // dr/dtheta_i = n_y^T (dR/dtheta_i) x_n, with the three derivatives in rotation.cpp.
            jacobian[3] = target_normal_.dot(rotation_derivative_alpha(alpha, beta, gamma) * source_point_);
            jacobian[4] = target_normal_.dot(rotation_derivative_beta(alpha, beta, gamma) * source_point_);
            jacobian[5] = target_normal_.dot(rotation_derivative_gamma(alpha, beta, gamma) * source_point_);
        }
        return true;
    }

    /**
     * @brief The source point of this correspondence
     *
     * @return x_n
     */
    const Eigen::Vector3d& source_point() const
    {
        return source_point_;
    }

    /**
     * @brief The target point of this correspondence
     *
     * @return y_n
     */
    const Eigen::Vector3d& target_point() const
    {
        return target_point_;
    }

    /**
     * @brief The target normal of this correspondence
     *
     * @return n_y, a unit vector
     */
    const Eigen::Vector3d& target_normal() const
    {
        return target_normal_;
    }

private:
    Eigen::Vector3d source_point_;  // x_n
    Eigen::Vector3d target_point_;  // y_n
    Eigen::Vector3d target_normal_;  // n_y, a unit vector
};

}  // namespace p2p_icp
