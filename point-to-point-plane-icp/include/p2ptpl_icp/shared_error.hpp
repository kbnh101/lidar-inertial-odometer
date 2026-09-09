#pragma once

#include <ceres/evaluation_callback.h>
#include <Eigen/Core>
#include <memory>
#include <chrono>
#include <vector>
#include "common/rotation.hpp"
#include "p2ptpl_icp/timing.hpp"

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
    explicit SharedErrorEvaluation(const double* xi, IcpTiming* timing = nullptr) : xi_(xi), timing_(timing)
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
        const auto begin = timing_ ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
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
        if (timing_)
        {
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
            if (evaluate_jacobians)
            {
                timing_->callback_with_jacobian_ms += ms;
                ++timing_->callback_jacobian_calls;
            }
            else
            {
                timing_->callback_residual_only_ms += ms;
                ++timing_->callback_residual_calls;
            }
        }
    }

private:
    const double* xi_;
    IcpTiming* timing_;
    std::vector<std::shared_ptr<SharedError>> pairs_;
};
}  // namespace p2ptpl_icp
