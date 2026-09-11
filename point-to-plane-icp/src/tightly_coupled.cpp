#include "p2p_icp/icp_point_to_plane.hpp"
#include "p2p_icp/inertial_cost.hpp"
#include "imu_preint/so3.hpp"

#include <ceres/ceres.h>
#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>

namespace p2p_icp
{
TightlyCoupledResult IcpPointToPlane::do_tightly_icp(
    const NavStatePrior& previous, const imu_preint::ImuPreintegrator& preintegration,
    const Eigen::Vector3d& gravity, const Eigen::Isometry3d& T_imu_lidar,
    const TightlyCoupledOptions& tight_options)
{
    const auto& opt = tight_options;
    const auto valid_rotation = [](const Eigen::Matrix3d& r) {
        return r.allFinite() && (r.transpose() * r).isApprox(Eigen::Matrix3d::Identity(), 1e-6) && std::abs(r.determinant() - 1.0) < 1e-6;
    };
    if (!(preintegration.delta_t() > 0.0) || !std::isfinite(previous.state.timestamp) ||
        !gravity.allFinite() || !T_imu_lidar.matrix().allFinite() || !valid_rotation(T_imu_lidar.rotation()) ||
        !valid_rotation(previous.state.rotation) || !previous.state.position.allFinite() || !previous.state.velocity.allFinite() ||
        !previous.state.gyro_bias.allFinite() || !previous.state.accel_bias.allFinite())
        throw std::invalid_argument("Tightly coupled ICP requires finite states, valid rotations and a positive IMU interval");
    for (double value : {opt.lidar_sigma, opt.max_translation_correction, opt.max_rotation_correction,
                         opt.gyro_reintegration_threshold, opt.accel_reintegration_threshold})
        if (!std::isfinite(value) || value <= 0.0)
            throw std::invalid_argument("Tightly coupled ICP scales and thresholds must be positive and finite");
    if (opt.min_correspondences < 1 || options_.max_iterations < 1 || options_.max_solver_iterations < 1)
        throw std::invalid_argument("Tightly coupled ICP requires positive iteration and correspondence limits");

    const auto prior_information = inertial::SqrtInformation(previous.covariance);
    auto integrated = preintegration;
    integrated.reintegrate(previous.state.gyro_bias, previous.state.accel_bias);
    const auto predicted = integrated.predict(previous.state, gravity);
    const auto predicted_pose = predicted.isometry() * T_imu_lidar;
    TightlyCoupledResult result;
    result.posterior.state = predicted;
    result.previous_state = previous.state;
    result.icp.transform = predicted_pose;
    result.icp.final_error = std::numeric_limits<double>::infinity();

    // Each retry starts from the same prior: rejected planes must never shrink the covariance.
    const auto imu_only = [&]() {
        auto fallback_options = opt;
        fallback_options.use_lidar = false;
        return do_tightly_icp(previous, preintegration, gravity, T_imu_lidar, fallback_options);
    };
    bool use_lidar = opt.use_lidar && !source_.empty() && !target_.empty();
    std::array<double, 15> xi{}, xj{};
    const int iterations = use_lidar ? options_.max_iterations : 1;
    for (int iteration = 0; iteration < iterations; ++iteration)
    {
        const auto before_pose = result.icp.transform;
        const auto correspondences = use_lidar ? find_correspondences(before_pose) : std::vector<Correspondence>{};
        if (use_lidar && static_cast<int>(correspondences.size()) < opt.min_correspondences)
            return imu_only();
        const auto current_i = inertial::Retract(previous.state, xi.data());
        const auto bias_delta = integrated.delta();
        if ((current_i.gyro_bias - bias_delta.gyro_bias).norm() > opt.gyro_reintegration_threshold ||
            (current_i.accel_bias - bias_delta.accel_bias).norm() > opt.accel_reintegration_threshold)
            integrated.reintegrate(current_i.gyro_bias, current_i.accel_bias);

        auto delta = integrated.delta();
        // A single Euler sample has rank-deficient position/velocity noise. This tiny integration
        // floor allows short intervals without treating zero-variance directions as measurements.
        delta.covariance.diagonal().array() += 1e-12;
        ceres::Problem problem;
        problem.AddResidualBlock(new inertial::PriorCost{prior_information}, nullptr, xi.data());
        problem.AddResidualBlock(new inertial::ImuCost{previous.state, predicted, delta, gravity, inertial::SqrtInformation(delta.covariance)},
            nullptr, xi.data(), xj.data());
        ceres::LossFunction* loss = use_lidar && options_.huber_delta > 0.0
            ? new ceres::HuberLoss(options_.huber_delta / opt.lidar_sigma) : nullptr;
        for (const auto& pair : correspondences)
        {
            const auto& target = target_[pair.target_index];
            problem.AddResidualBlock(new inertial::PlaneCost{predicted, T_imu_lidar * source_[pair.source_index].point,
                                                            target.point, target.normal, opt.lidar_sigma}, loss, xj.data());
        }
        ceres::Solver::Options solver;
        solver.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;
        solver.max_num_iterations = options_.max_solver_iterations;
        solver.function_tolerance = 1e-8;
        solver.gradient_tolerance = 1e-10;
        solver.parameter_tolerance = 1e-9;
        solver.logging_type = ceres::SILENT;
        ceres::Solver::Summary summary;
        ceres::Solve(solver, &problem, &summary);
        if (!summary.IsSolutionUsable())
            return use_lidar ? imu_only() : result;

        result.previous_state = inertial::Retract(previous.state, xi.data());
        result.posterior.state = inertial::Retract(predicted, xj.data());
        result.icp.transform = result.posterior.state.isometry() * T_imu_lidar;
        const auto deviation = predicted_pose.inverse() * result.icp.transform;
        if (use_lidar && (deviation.translation().norm() > opt.max_translation_correction ||
                         Eigen::AngleAxisd(deviation.rotation()).angle() > opt.max_rotation_correction))
            return imu_only();

        const auto change = before_pose.inverse() * result.icp.transform;
        IcpIterationLog log;
        log.iteration = iteration;
        log.correspondences = static_cast<int>(correspondences.size());
        log.delta_translation = change.translation().norm();
        log.delta_rotation = Eigen::AngleAxisd(change.rotation()).angle();
        evaluate_error(before_pose, correspondences, &log.error_before, nullptr, nullptr);
        evaluate_error(result.icp.transform, correspondences, &log.error_after, nullptr, nullptr);
        result.icp.history.push_back(log);
        result.icp.iterations = iteration + 1;
        result.icp.converged = !use_lidar ||
            (log.delta_translation < options_.translation_tolerance && log.delta_rotation < options_.rotation_tolerance);
        if (!result.icp.converged && iteration + 1 < iterations)
            continue;

        // The current block of (J^T J)^-1 marginalizes xi, retaining cross-state/bias information.
        ceres::Covariance::Options covariance_options;
        covariance_options.algorithm_type = ceres::DENSE_SVD;
        covariance_options.min_reciprocal_condition_number = 1e-18;
        ceres::Covariance covariance(covariance_options);
        const std::vector<std::pair<const double*, const double*>> blocks{{xj.data(), xj.data()}};
        Eigen::Matrix<double, 15, 15, Eigen::RowMajor> raw_covariance;
        if (!covariance.Compute(blocks, &problem) ||
            !covariance.GetCovarianceBlock(xj.data(), xj.data(), raw_covariance.data()))
            return use_lidar ? imu_only() : result;
        // Ceres differentiated additive Exp coordinates at xj; carry a right tangent at the new R.
        imu_preint::Matrix15d transport = imu_preint::Matrix15d::Identity();
        transport.topLeftCorner<3, 3>() = imu_preint::so3::RightJacobian(Eigen::Map<const Eigen::Vector3d>(xj.data()));
        result.posterior.covariance = transport * raw_covariance * transport.transpose();
        result.posterior.covariance = (0.5 * (result.posterior.covariance + result.posterior.covariance.transpose())).eval();
        if (!result.posterior.covariance.allFinite() || Eigen::LLT<imu_preint::Matrix15d>(result.posterior.covariance).info() != Eigen::Success)
            return use_lidar ? imu_only() : result;
        if (use_lidar)
        {
            const auto final_pairs = find_correspondences(result.icp.transform);
            if (static_cast<int>(final_pairs.size()) < opt.min_correspondences)
                return imu_only();
            result.icp.correspondences = static_cast<int>(final_pairs.size());
            evaluate_error(result.icp.transform, final_pairs, &result.icp.final_error,
                           &result.icp.final_mean_abs_error, &result.icp.final_max_abs_error);
        }
        result.lidar_accepted = use_lidar;
        result.usable = true;
        return result;
    }
    return result;
}
}  // namespace p2p_icp
