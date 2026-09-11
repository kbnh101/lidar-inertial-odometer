#pragma once

#include <ceres/sized_cost_function.h>
#include <Eigen/Cholesky>
#include <stdexcept>
#include "imu_preint/types.hpp"
#include "imu_preint/so3.hpp"

namespace p2p_icp
{
namespace inertial
{
using imu_preint::Matrix15d;
using imu_preint::Vector15d;
using RowMatrix15d = Eigen::Matrix<double, 15, 15, Eigen::RowMajor>;
namespace so3 = imu_preint::so3;

inline Matrix15d SqrtInformation(const Matrix15d& covariance)
{
    if (!covariance.allFinite() || !covariance.isApprox(covariance.transpose(), 1e-8))
    {
        throw std::invalid_argument("State covariance must be finite and symmetric");
    }
    Eigen::LLT<Matrix15d> llt(covariance);
    if (llt.info() != Eigen::Success)
    {
        throw std::invalid_argument("State covariance must be positive definite");
    }
    // C = L L^T, so whitening is L^-1 (not L^-T).
    return llt.matrixL().solve(Matrix15d::Identity());
}

/// Ceres adds to x; physical rotation is base.R * Exp(x.rotation).
inline imu_preint::NavState Retract(const imu_preint::NavState& base, const double* x)
{
    auto result = base;
    result.rotation = base.rotation * so3::Exp(Eigen::Map<const Eigen::Vector3d>(x));
    result.position += Eigen::Map<const Eigen::Vector3d>(x + 3);
    result.velocity += Eigen::Map<const Eigen::Vector3d>(x + 6);
    result.gyro_bias += Eigen::Map<const Eigen::Vector3d>(x + 9);
    result.accel_bias += Eigen::Map<const Eigen::Vector3d>(x + 12);
    return result;
}

class PriorCost : public ceres::SizedCostFunction<15, 15>
{
public:
    explicit PriorCost(const Matrix15d& sqrt_information) : sqrt_information_(sqrt_information)
    {
    }

    bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const override
    {
        Eigen::Map<Vector15d> residual(residuals);
        residual = sqrt_information_ * Eigen::Map<const Vector15d>(parameters[0]);
        if (jacobians != nullptr && jacobians[0] != nullptr)
        {
            Eigen::Map<RowMatrix15d> jacobian(jacobians[0]);
            jacobian = sqrt_information_;
        }
        return true;
    }

private:
    Matrix15d sqrt_information_;
};

/// Residual order: rotation, position, velocity, bg_j-bg_i, ba_j-ba_i.
/// All Jacobians differentiate additive Exp coordinates, including nonzero rotation increments
/// and nonzero preintegration bias corrections. Whitening applies to residuals AND Jacobians.
class ImuCost : public ceres::SizedCostFunction<15, 15, 15>
{
public:
    ImuCost(const imu_preint::NavState& base_i, const imu_preint::NavState& base_j, const imu_preint::PreintegratedDelta& delta, const Eigen::Vector3d& gravity,
            const Matrix15d& sqrt_information)
        : base_i_(base_i), base_j_(base_j), delta_(delta), gravity_(gravity), sqrt_information_(sqrt_information)
    {
    }

    bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const override
    {
        const auto i = Retract(base_i_, parameters[0]);
        const auto j = Retract(base_j_, parameters[1]);
        Eigen::Matrix<double, 6, 1> db;
        db << i.gyro_bias - delta_.gyro_bias, i.accel_bias - delta_.accel_bias;
        const Eigen::Matrix<double, 9, 1> correction = delta_.bias_jacobian * db;
        const Eigen::Matrix3d corrected_rotation = delta_.rotation * so3::Exp(correction.head<3>());
        const Eigen::Matrix3d rit = i.rotation.transpose();
        const double dt = delta_.dt;
        const Eigen::Vector3d local_position = rit * (j.position - i.position - i.velocity * dt - 0.5 * gravity_ * dt * dt);
        const Eigen::Vector3d local_velocity = rit * (j.velocity - i.velocity - gravity_ * dt);
        Vector15d r;
        r.head<3>() = so3::Log(corrected_rotation.transpose() * rit * j.rotation);
        r.segment<3>(3) = local_position - delta_.position - correction.segment<3>(3);
        r.segment<3>(6) = local_velocity - delta_.velocity - correction.segment<3>(6);
        r.segment<3>(9) = j.gyro_bias - i.gyro_bias;
        r.segment<3>(12) = j.accel_bias - i.accel_bias;
        Eigen::Map<Vector15d> residual(residuals);
        residual = sqrt_information_ * r;

        if (jacobians == nullptr || (jacobians[0] == nullptr && jacobians[1] == nullptr))
        {
            return true;
        }
        if (jacobians[0] != nullptr)
        {
            const Eigen::Matrix3d jr_i = so3::RightJacobian(Eigen::Map<const Eigen::Vector3d>(parameters[0]));
            // Log(Exp(e) E) differentiates with J_l^-1(Log E) = J_r^-1(-Log E).
            const Eigen::Matrix3d jl_inverse = so3::RightJacobianInverse(-r.head<3>());
            Matrix15d ji = Matrix15d::Zero();
            ji.block<3, 3>(0, 0) = -jl_inverse * corrected_rotation.transpose() * jr_i;
            ji.block<3, 6>(0, 9) = -jl_inverse * so3::RightJacobian(correction.head<3>()) * delta_.bias_jacobian.topRows<3>();
            ji.block<3, 3>(3, 0) = so3::Hat(local_position) * jr_i;
            ji.block<3, 3>(3, 3) = -rit;
            ji.block<3, 3>(3, 6) = -rit * dt;
            ji.block<3, 6>(3, 9) = -delta_.bias_jacobian.middleRows<3>(3);
            ji.block<3, 3>(6, 0) = so3::Hat(local_velocity) * jr_i;
            ji.block<3, 3>(6, 6) = -rit;
            ji.block<3, 6>(6, 9) = -delta_.bias_jacobian.bottomRows<3>();
            ji.bottomRightCorner<6, 6>() = -Eigen::Matrix<double, 6, 6>::Identity();
            Eigen::Map<RowMatrix15d> jacobian(jacobians[0]);
            jacobian.noalias() = sqrt_information_ * ji;
        }
        if (jacobians[1] != nullptr)
        {
            Matrix15d jj = Matrix15d::Zero();
            jj.block<3, 3>(0, 0) = so3::RightJacobianInverse(r.head<3>()) * so3::RightJacobian(Eigen::Map<const Eigen::Vector3d>(parameters[1]));
            jj.block<3, 3>(3, 3) = rit;
            jj.block<3, 3>(6, 6) = rit;
            jj.bottomRightCorner<6, 6>() = Eigen::Matrix<double, 6, 6>::Identity();
            Eigen::Map<RowMatrix15d> jacobian(jacobians[1]);
            jacobian.noalias() = sqrt_information_ * jj;
        }
        return true;
    }

private:
    imu_preint::NavState base_i_, base_j_;
    imu_preint::PreintegratedDelta delta_;
    Eigen::Vector3d gravity_;
    Matrix15d sqrt_information_;
};

class PlaneCost : public ceres::SizedCostFunction<1, 15>
{
public:
    PlaneCost(const imu_preint::NavState& base, const Eigen::Vector3d& source_imu, const Eigen::Vector3d& target, const Eigen::Vector3d& normal, double sigma)
        : base_rotation_(base.rotation), base_position_(base.position), source_imu_(source_imu), target_(target), normal_(normal / sigma)
    {
    }

    bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const override
    {
        const Eigen::Map<const Eigen::Vector3d> phi(parameters[0]);
        const Eigen::Matrix3d rotation = base_rotation_ * so3::Exp(phi);
        const Eigen::Vector3d position = base_position_ + Eigen::Map<const Eigen::Vector3d>(parameters[0] + 3);
        residuals[0] = normal_.dot(rotation * source_imu_ + position - target_);
        if (jacobians != nullptr && jacobians[0] != nullptr)
        {
            Eigen::Map<Eigen::Matrix<double, 1, 15>> jacobian(jacobians[0]);
            jacobian.setZero();  // fixed deskewed point: no direct velocity or bias dependence
            jacobian.head<3>() = -normal_.transpose() * rotation * so3::Hat(source_imu_) * so3::RightJacobian(phi);
            jacobian.segment<3>(3) = normal_.transpose();
        }
        return true;
    }

private:
    Eigen::Matrix3d base_rotation_;
    Eigen::Vector3d base_position_, source_imu_, target_, normal_;
};
}  // namespace inertial
}  // namespace p2p_icp
