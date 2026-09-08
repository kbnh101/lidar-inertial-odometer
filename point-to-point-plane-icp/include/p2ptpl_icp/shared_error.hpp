#pragma once

#include <ceres/evaluation_callback.h>
#include <Eigen/Core>
#include <memory>
#include <vector>
#include "common/rotation.hpp"

namespace p2ptpl_icp
{
using ErrorJacobian = Eigen::Matrix<double, 3, 6, Eigen::RowMajor>;

// One correspondence owns ONE e = R*x + t - y. Both cost blocks hold this same
// object. Only SharedErrorEvaluation writes it, before Ceres evaluates any blocks.
struct SharedError
{
    Eigen::Vector3d source;
    Eigen::Vector3d target;
    Eigen::Vector3d normal;
    Eigen::Vector3d error = Eigen::Vector3d::Zero();
    ErrorJacobian jacobian = ErrorJacobian::Zero();
    bool error_ready = false;
    bool jacobian_ready = false;
};

class SharedErrorEvaluation : public ceres::EvaluationCallback
{
public:
    explicit SharedErrorEvaluation(const double* xi) : xi_(xi)
    {
    }

    std::shared_ptr<const SharedError> Add(const Eigen::Vector3d& source, const Eigen::Vector3d& target, const Eigen::Vector3d& normal)
    {
        auto pair = std::make_shared<SharedError>();
        pair->source = source;
        pair->target = target;
        pair->normal = normal;
        pairs_.push_back(pair);
        return pair;
    }

    void PrepareForEvaluation(bool evaluate_jacobians, bool new_evaluation_point) override
    {
        const auto rotation = common::euler_zyx_to_rotation(xi_[3], xi_[4], xi_[5]);
        const Eigen::Map<const Eigen::Vector3d> translation(xi_);
        Eigen::Matrix3d da, db, dg;
        if (evaluate_jacobians)
        {
            da = common::rotation_derivative_alpha(xi_[3], xi_[4], xi_[5]);
            db = common::rotation_derivative_beta(xi_[3], xi_[4], xi_[5]);
            dg = common::rotation_derivative_gamma(xi_[3], xi_[4], xi_[5]);
        }
        for (const auto& pair : pairs_)
        {
            if (new_evaluation_point || !pair->error_ready)
            {
                pair->error = rotation * pair->source + translation - pair->target;
                pair->error_ready = true;
                pair->jacobian_ready = false;
            }
            if (evaluate_jacobians && !pair->jacobian_ready)
            {
                pair->jacobian.leftCols<3>().setIdentity();
                pair->jacobian.col(3) = da * pair->source;
                pair->jacobian.col(4) = db * pair->source;
                pair->jacobian.col(5) = dg * pair->source;
                pair->jacobian_ready = true;
            }
        }
    }

private:
    const double* xi_;
    std::vector<std::shared_ptr<SharedError>> pairs_;
};
}  // namespace p2ptpl_icp
