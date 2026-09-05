#pragma once

#include <ceres/sized_cost_function.h>

#include <Eigen/Core>

#include "common/rotation.hpp"

namespace p2pt_icp
{
/**
 * @brief Point-to-point residual
 *
 *   parameter block  xi = [tx, ty, tz, alpha, beta, gamma]^T in R^6
 *   residual         r_n = R(theta) x_n + t - y_n                     in R^3
 *   Jacobian (3x6)   dr/dt = I_3,  dr/dtheta_i = (dR/dtheta_i) x_n
 *
 * Unlike point-to-plane, the residual is the full 3-D error vector rather than its projection onto
 * the target normal, so the base is `ceres::SizedCostFunction<3, 6>` and the Jacobian is 3x6.
 * Normals play no part here at all -- neither in the residual nor in the correspondence search.
 *
 * The rotation convention is shared with the point-to-plane project: R(theta) = Rz(g) Ry(b) Rx(a),
 * and `common::rotation_derivative_alpha/beta/gamma()` already provide the three derivatives.
 *
 * Note there is also a closed-form solution for this cost (Kabsch / Umeyama via SVD), so Ceres is
 * not strictly required; it is used here to mirror the point-to-plane project.
 */
class PointToPointCostFunction : public ceres::SizedCostFunction<3, 6>
{
public:
    /**
     * @brief Constructs the cost function for one correspondence pair
     *
     * @param source_point x_n, the source point in the frame the parameter block transforms
     * @param target_point y_n, the corresponding target point
     */
    PointToPointCostFunction(const Eigen::Vector3d& source_point, const Eigen::Vector3d& target_point)
        : source_point_(source_point), target_point_(target_point)
    {
    }

    /**
     * @brief Computes the residual and the analytic Jacobian
     *
     * @param parameters parameters[0] = xi = [tx, ty, tz, alpha, beta, gamma]
     * @param residuals  output: r_n, three values
     * @param jacobians  output: J_n in R^{3x6}, row major; nullptr means only the residual is wanted
     * @return true on success
     */
    bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const override
    {
        const Eigen::Map<const Eigen::Vector3d> translation(parameters[0]);
        const double alpha = parameters[0][3];
        const double beta = parameters[0][4];
        const double gamma = parameters[0][5];

        const Eigen::Matrix3d rotation = common::euler_zyx_to_rotation(alpha, beta, gamma);

        // r_n = R x_n + t - y_n, the full 3-D error vector.
        const Eigen::Vector3d error = rotation * source_point_ + translation - target_point_;
        residuals[0] = error.x();
        residuals[1] = error.y();
        residuals[2] = error.z();

        if (jacobians != nullptr && jacobians[0] != nullptr)
        {
            // There is one parameter block, so there is one jacobian block: jacobians[0], holding
            // 3x6 = 18 doubles row major. The outer index selects the parameter block, never the
            // residual row -- row r lives at jacobians[0][6 * r] within this one block.
            Eigen::Map<Eigen::Matrix<double, 3, 6, Eigen::RowMajor>> jacobian(jacobians[0]);

            // dr/dt = I_3. setIdentity() also writes the off-diagonal zeros, which matters because
            // Ceres hands over an uninitialized buffer.
            jacobian.leftCols<3>().setIdentity();

            // dr/dtheta_i = (dR/dtheta_i) x_n, one column per angle.
            jacobian.col(3) = common::rotation_derivative_alpha(alpha, beta, gamma) * source_point_;
            jacobian.col(4) = common::rotation_derivative_beta(alpha, beta, gamma) * source_point_;
            jacobian.col(5) = common::rotation_derivative_gamma(alpha, beta, gamma) * source_point_;
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

private:
    Eigen::Vector3d source_point_;  // x_n
    Eigen::Vector3d target_point_;  // y_n
};

}  // namespace p2pt_icp
