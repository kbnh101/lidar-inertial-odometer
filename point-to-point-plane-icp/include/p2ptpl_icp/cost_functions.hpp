#pragma once

#include <ceres/problem.h>
#include <ceres/sized_cost_function.h>
#include <cmath>
#include <stdexcept>
#include "p2ptpl_icp/shared_error.hpp"

namespace p2ptpl_icp
{
// These costs must be used with SharedErrorEvaluation registered on the Problem.
// Evaluate only reads the cache, so parallel Ceres evaluation cannot race on e.
class PointCostFunction : public ceres::SizedCostFunction<3, 6>
{
public:
    PointCostFunction(std::shared_ptr<const SharedError> error, double weight) : error_(std::move(error)), scale_(std::sqrt(weight))
    {
    }
    bool Evaluate(double const* const*, double* residuals, double** jacobians) const override
    {
        if (!error_->error_ready)
            return false;
        Eigen::Map<Eigen::Vector3d> residual(residuals);
        residual = scale_ * error_->error;
        if (jacobians && jacobians[0])
        {
            if (!error_->jacobian_ready)
                return false;
            Eigen::Map<ErrorJacobian> jacobian(jacobians[0]);
            jacobian = scale_ * error_->jacobian;
        }
        return residual.allFinite();
    }

private:
    std::shared_ptr<const SharedError> error_;
    double scale_;
};

class PlaneCostFunction : public ceres::SizedCostFunction<1, 6>
{
public:
    PlaneCostFunction(std::shared_ptr<const SharedError> error, double weight) : error_(std::move(error)), scale_(std::sqrt(weight))
    {
    }
    bool Evaluate(double const* const*, double* residuals, double** jacobians) const override
    {
        if (!error_->error_ready)
            return false;
        residuals[0] = scale_ * error_->normal.dot(error_->error);
        if (jacobians && jacobians[0])
        {
            if (!error_->jacobian_ready)
                return false;
            Eigen::Map<Eigen::Matrix<double, 1, 6, Eigen::RowMajor>> jacobian(jacobians[0]);
            jacobian = scale_ * error_->normal.transpose() * error_->jacobian;
        }
        return std::isfinite(residuals[0]);
    }

private:
    std::shared_ptr<const SharedError> error_;
    double scale_;
};

inline void AddCorrespondenceResiduals(ceres::Problem& problem, const std::shared_ptr<const SharedError>& error, double* xi, ceres::LossFunction* loss,
                                       double point_weight, double plane_weight)
{
    if (!std::isfinite(point_weight) || !std::isfinite(plane_weight) || point_weight <= 0 || plane_weight <= 0)
        throw std::invalid_argument("both hybrid residual weights must be finite and positive");
    // Two distinct blocks, sharing both the correspondence error and pose block.
    problem.AddResidualBlock(new PointCostFunction(error, point_weight), loss, xi);
    problem.AddResidualBlock(new PlaneCostFunction(error, plane_weight), loss, xi);
}
}  // namespace p2ptpl_icp
