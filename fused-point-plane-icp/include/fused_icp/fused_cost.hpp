#pragma once

#include <ceres/manifold.h>
#include <ceres/sized_cost_function.h>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <array>

#include "fused_icp/fused_residual.hpp"
#include "fused_icp/se3.hpp"

namespace fused_icp
{
/// Ambient size of the pose parameter block: [qw, qx, qy, qz, tx, ty, tz].
constexpr int kPoseAmbientSize = 7;
/// Tangent size: delta_xi = [delta_phi; delta_rho].
constexpr int kPoseTangentSize = 6;

using PoseParameters = std::array<double, kPoseAmbientSize>;

/**
 * @brief Packs a pose into the parameter block layout [qw, qx, qy, qz, tx, ty, tz]
 *
 * The quaternion is stored w first, explicitly, so that the layout never depends on
 * Eigen::Quaterniond::coeffs() (which is x, y, z, w).
 *
 * @param pose T = (R, t)
 * @return the 7 parameters
 */
inline PoseParameters pose_to_parameters(const Eigen::Isometry3d& pose)
{
    const Eigen::Quaterniond q(pose.linear());
    return PoseParameters{q.w(), q.x(), q.y(), q.z(), pose.translation().x(), pose.translation().y(), pose.translation().z()};
}

/**
 * @brief Inverse of pose_to_parameters()
 *
 * @param parameters [qw, qx, qy, qz, tx, ty, tz]
 * @return T = (R, t), with the quaternion normalized on the way
 */
inline Eigen::Isometry3d parameters_to_pose(const double* parameters)
{
    Eigen::Quaterniond q(parameters[0], parameters[1], parameters[2], parameters[3]);
    q.normalize();
    Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
    pose.linear() = q.toRotationMatrix();
    pose.translation() = Eigen::Vector3d(parameters[4], parameters[5], parameters[6]);
    return pose;
}

/**
 * @brief SE(3) manifold with the right perturbation of the note, T+ = T Exp(delta_xi)
 *
 * Plus() is the exact SE(3) exponential of se3::retract() (rotation first, then translation, with
 * the V matrix of note (22)), so the pose block moves along the same curve the note differentiates.
 *
 * Jacobian convention (read together with FusedPointPlaneCostFunction):
 * Ceres forms the tangent-space Jacobian as J_ambient (m x 7) * PlusJacobian (7 x 6), note (59).
 * The built-in QuaternionManifold's PlusJacobian would rescale a tangent Jacobian dropped into
 * the ambient array (note, page 11). This manifold instead defines PlusJacobian = [I_6; 0] and the
 * cost functions of this package return [J_tangent | 0] as their ambient Jacobian, so the product
 * is exactly J_tangent = L G of note (44). The pair is consistent for the solver; it is *not* the
 * derivative of Plus() in the ambient coordinates, therefore ceres::GradientChecker (which
 * differentiates in the ambient space) does not apply. test_icp verifies J_f by central differences
 * through the same retraction instead, note (56).
 */
class Se3RightManifold : public ceres::Manifold
{
public:
    int AmbientSize() const override
    {
        return kPoseAmbientSize;
    }

    int TangentSize() const override
    {
        return kPoseTangentSize;
    }

    /**
     * @brief x_plus_delta = pack(unpack(x) * Exp(delta)), note (1)/(21)
     */
    bool Plus(const double* x, const double* delta, double* x_plus_delta) const override
    {
        const Eigen::Map<const Vector6> delta_xi(delta);
        const Eigen::Isometry3d retracted = se3::retract(parameters_to_pose(x), delta_xi);
        const PoseParameters packed = pose_to_parameters(retracted);
        for (int i = 0; i < kPoseAmbientSize; ++i)
        {
            x_plus_delta[i] = packed[i];
        }
        return true;
    }

    /**
     * @brief [I_6; 0], row-major 7 x 6 (see the class comment)
     */
    bool PlusJacobian(const double* /*x*/, double* jacobian) const override
    {
        Eigen::Map<Eigen::Matrix<double, kPoseAmbientSize, kPoseTangentSize, Eigen::RowMajor>> J(jacobian);
        J.setZero();
        J.topRows<kPoseTangentSize>().setIdentity();
        return true;
    }

    /**
     * @brief y_minus_x = Log(unpack(x)^-1 unpack(y)), the inverse of Plus()
     */
    bool Minus(const double* y, const double* x, double* y_minus_x) const override
    {
        const Eigen::Isometry3d relative = parameters_to_pose(x).inverse() * parameters_to_pose(y);
        Eigen::Map<Vector6> delta(y_minus_x);
        delta = se3::log_se3(relative);
        return true;
    }

    /**
     * @brief [I_6 | 0], row-major 6 x 7, the left inverse of PlusJacobian()
     */
    bool MinusJacobian(const double* /*x*/, double* jacobian) const override
    {
        Eigen::Map<Eigen::Matrix<double, kPoseTangentSize, kPoseAmbientSize, Eigen::RowMajor>> J(jacobian);
        J.setZero();
        J.leftCols<kPoseTangentSize>().setIdentity();
        return true;
    }
};

/**
 * @brief Fused point-to-point + point-to-plane residual of one correspondence
 *
 *   parameter block  [qw, qx, qy, qz, tx, ty, tz], on Se3RightManifold
 *   residual         r_f = L e,  e = R p + t - q,  L = sqrt(alpha) I + (sqrt(alpha+beta) - sqrt(alpha)) n n^T   (42)
 *   Jacobian (3x6)   J_f = L G,  G = [-R [p]x   R]                                                      (44)
 *
 * |r_f|^2 = alpha |e|^2 + beta (n^T e)^2, so this single 3-D block replaces a 3-D point-to-point
 * block plus a 1-D point-to-plane block on the same pair and yields the same Gauss-Newton normal
 * equations, note (46). A ceres::LossFunction attached to this block is the *joint* loss
 * rho(alpha |e|^2 + beta (n^T e)^2) of note (61), not two separate losses.
 *
 * The Jacobian is analytic (no AutoDiff). Its ambient 3 x 7 array holds J_f in the first six
 * columns and zeros in the seventh -- the convention documented on Se3RightManifold.
 *
 * Only the target normal enters; the source normal is used solely by the correspondence rejection.
 */
class FusedPointPlaneCostFunction : public ceres::SizedCostFunction<3, kPoseAmbientSize>
{
public:
    /**
     * @brief Constructs the cost function for one correspondence pair
     *
     * @param source_point  p, in the frame the parameter block transforms
     * @param target_point  q, the corresponding target point
     * @param target_normal n, a unit vector (common::load_point_cloud() normalizes on load)
     * @param weights       alpha and beta; both are fixed inside the solve
     */
    FusedPointPlaneCostFunction(const Eigen::Vector3d& source_point, const Eigen::Vector3d& target_point, const Eigen::Vector3d& target_normal,
                                const FusedWeights& weights)
        : source_point_(source_point), target_point_(target_point), target_normal_(target_normal), weights_(weights)
    {
    }

    /**
     * @brief Computes r_f and, when asked, J_f
     *
     * @param parameters parameters[0] = the pose block
     * @param residuals  output: r_f in R^3
     * @param jacobians  output: jacobians[0] is 3 x 7 row-major = [J_f | 0]; nullptr means residual only
     * @return always true
     */
    bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const override
    {
        const Eigen::Isometry3d pose = parameters_to_pose(parameters[0]);

        // e = R p + t - q, then r_f = L e -- the derivation from (3) to (42).
        const Eigen::Vector3d error = geometric_error(pose, source_point_, target_point_);
        Eigen::Map<Eigen::Vector3d> r(residuals);
        r = fused_residual(error, target_normal_, weights_);

        if (jacobians != nullptr && jacobians[0] != nullptr)
        {
            // J_f = L G with G = [-R[p]x  R], (20) and (44), placed in the tangent columns.
            const Matrix36 G = geometric_jacobian(pose.linear(), source_point_);
            Eigen::Map<Eigen::Matrix<double, 3, kPoseAmbientSize, Eigen::RowMajor>> J(jacobians[0]);
            J.leftCols<kPoseTangentSize>() = fused_jacobian(G, target_normal_, weights_);
            J.col(kPoseTangentSize).setZero();
        }
        return true;
    }

    const Eigen::Vector3d& source_point() const
    {
        return source_point_;
    }

    const Eigen::Vector3d& target_point() const
    {
        return target_point_;
    }

    const Eigen::Vector3d& target_normal() const
    {
        return target_normal_;
    }

    const FusedWeights& weights() const
    {
        return weights_;
    }

private:
    Eigen::Vector3d source_point_;  // p
    Eigen::Vector3d target_point_;  // q
    Eigen::Vector3d target_normal_;  // n, unit
    FusedWeights weights_;  // alpha, beta
};

}  // namespace fused_icp
