#include "p2p_icp/icp_point_to_plane.hpp"

#include <ceres/ceres.h>

#include <array>
#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>

#include "p2p_icp/point_to_plane_cost.hpp"
#include "p2p_icp/rotation.hpp"

namespace p2p_icp
{
void IcpPointToPlane::set_source(const PointCloud& source)
{
    source_ = source;
}

void IcpPointToPlane::set_target(const PointCloud& target)
{
    target_ = target;
    target_tree_.build(target_);  // every correspondence search goes through this kd-tree
}

std::vector<IcpPointToPlane::Correspondence> IcpPointToPlane::find_correspondences(const Eigen::Isometry3d& pose) const
{
    std::vector<Correspondence> correspondences;
    correspondences.reserve(source_.size());

    const double max_squared_distance = options_.max_correspondence_distance * options_.max_correspondence_distance;

    for (std::size_t i = 0; i < source_.size(); ++i)
    {
        // Move by the current pose, then find the nearest point; a better pose gives better pairs.
        const Eigen::Vector3d transformed = pose * source_[i].point;

        int target_index = -1;
        double squared_distance = 0.0;
        if (!target_tree_.nearest(transformed, &target_index, &squared_distance))
        {
            continue;
        }
        // Rejection 1, distance: a pair this far apart is probably a wrong correspondence.
        if (squared_distance > max_squared_distance)
        {
            continue;
        }

        const Eigen::Vector3d& target_normal = target_[target_index].normal;
        // A target point without a normal defines no tangent plane and is therefore unusable.
        if (target_normal.squaredNorm() <= 0.0)
        {
            continue;
        }

        // Rejection 2, normal agreement -- the only place the source normal n_x is used.
        if (options_.min_normal_dot > -1.0 && source_[i].normal.squaredNorm() > 0.0)
        {
            const Eigen::Vector3d rotated_source_normal = pose.linear() * source_[i].normal;
            if (rotated_source_normal.dot(target_normal) < options_.min_normal_dot)
            {
                continue;
            }
        }

        correspondences.push_back(Correspondence{static_cast<int>(i), target_index});
    }
    return correspondences;
}

void IcpPointToPlane::evaluate_error(const Eigen::Isometry3d& pose, const std::vector<Correspondence>& correspondences, double* rms, double* mean_abs,
                                     double* max_abs) const
{
    double sum_squared = 0.0;
    double sum_abs = 0.0;
    double max_value = 0.0;
    for (const Correspondence& correspondence : correspondences)
    {
        const Eigen::Vector3d transformed = pose * source_[correspondence.source_index].point;
        // The same residual as in the cost function: r_n = n_y . (R x_n + t - y_n)
        const double residual = target_[correspondence.target_index].normal.dot(transformed - target_[correspondence.target_index].point);
        sum_squared += residual * residual;
        sum_abs += std::abs(residual);
        max_value = std::max(max_value, std::abs(residual));
    }
    const double count = static_cast<double>(correspondences.size());
    if (rms != nullptr)
    {
        *rms = correspondences.empty() ? 0.0 : std::sqrt(sum_squared / count);
    }
    if (mean_abs != nullptr)
    {
        *mean_abs = correspondences.empty() ? 0.0 : sum_abs / count;
    }
    if (max_abs != nullptr)
    {
        *max_abs = max_value;
    }
}

IcpResult IcpPointToPlane::do_icp(const Eigen::Isometry3d& initial_guess)
{
    if (source_.empty())
    {
        throw std::runtime_error("IcpPointToPlane::do_icp: source cloud is empty");
    }
    if (target_.empty())
    {
        throw std::runtime_error("IcpPointToPlane::do_icp: target cloud is empty");
    }

    IcpResult result;
    result.transform = initial_guess;

    double previous_error = std::numeric_limits<double>::infinity();

    for (int iteration = 0; iteration < options_.max_iterations; ++iteration)
    {
        // --- 1. search the correspondences with the current estimate -----------
        const std::vector<Correspondence> correspondences = find_correspondences(result.transform);
        if (static_cast<int>(correspondences.size()) < 6)
        {
            // With 6 unknowns and fewer than 6 constraints the problem is not solvable.
            if (options_.verbose)
            {
                std::printf("[icp] iteration %2d: only %zu correspondences, stopping\n", iteration, correspondences.size());
            }
            break;
        }

        IcpIterationLog log;
        log.iteration = iteration;
        log.correspondences = static_cast<int>(correspondences.size());
        evaluate_error(result.transform, correspondences, &log.error_before, nullptr, nullptr);

        // --- 2. solve for the increment xi = [t, alpha, beta, gamma] -----------
        // xi is an *increment*, not the accumulated absolute pose, and restarts from 0 each
        // iteration, so the Euler angles stay near 0 and never reach the beta = +-pi/2 gimbal lock.
        std::array<double, 6> xi{};
        xi.fill(0.0);

        // The rotation pivot is absent from the derivation but matters numerically. It shifts x and
        // y by the same constant, so the residual and the Jacobian are unchanged; it only
        // decorrelates the rotation columns of J from the translation columns.
        Eigen::Vector3d pivot = Eigen::Vector3d::Zero();
        if (options_.pivot_increment_at_centroid)
        {
            for (const Correspondence& correspondence : correspondences)
            {
                pivot += result.transform * source_[correspondence.source_index].point;
            }
            pivot /= static_cast<double>(correspondences.size());
        }

        ceres::Problem problem;
        problem.AddParameterBlock(xi.data(), 6);
        ceres::LossFunction* loss = options_.huber_delta > 0.0 ? new ceres::HuberLoss(options_.huber_delta) : nullptr;

        for (const Correspondence& correspondence : correspondences)
        {
            // Pre-transform by the current pose. The cost function then sees x in a partially
            // aligned frame, which is what makes xi = 0 mean "the current state".
            const Eigen::Vector3d source_point = result.transform * source_[correspondence.source_index].point - pivot;
            const Eigen::Vector3d target_point = target_[correspondence.target_index].point - pivot;
            problem.AddResidualBlock(
                    new PointToPlaneCostFunction(source_point, target_point, target_[correspondence.target_index].normal),
                    loss, xi.data());
        }

        ceres::Solver::Options solver_options;
        solver_options.linear_solver_type = ceres::DENSE_QR;
        solver_options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
        solver_options.max_num_iterations = options_.max_solver_iterations;
        solver_options.minimizer_progress_to_stdout = false;
        solver_options.logging_type = ceres::SILENT;
        solver_options.function_tolerance = 1e-16;
        solver_options.gradient_tolerance = 1e-18;
        solver_options.parameter_tolerance = 1e-16;

        ceres::Solver::Summary summary;
        ceres::Solve(solver_options, &problem, &summary);

        // --- 3. compose the increment onto the accumulated pose ----------------
        // The increment acts as p -> R(p - pivot) + pivot + t, so the world frame translation is
        // (pivot - R*pivot + t).
        Eigen::Isometry3d increment = Eigen::Isometry3d::Identity();
        increment.linear() = euler_zyx_to_rotation(xi[3], xi[4], xi[5]);
        increment.translation() = pivot - increment.linear() * pivot + Eigen::Vector3d(xi[0], xi[1], xi[2]);
        result.transform = increment * result.transform;

        log.delta_translation = increment.translation().norm();
        log.delta_rotation = Eigen::AngleAxisd(increment.linear()).angle();
        evaluate_error(result.transform, correspondences, &log.error_after, nullptr, nullptr);
        result.history.push_back(log);
        result.iterations = iteration + 1;
        result.correspondences = log.correspondences;

        if (options_.verbose)
        {
            std::printf("[icp] iter %2d | corr %4d | rms %.12e -> %.12e | |dt| %.3e | |dR| %.3e rad\n", log.iteration, log.correspondences, log.error_before,
                        log.error_after, log.delta_translation, log.delta_rotation);
        }

        // --- 4. convergence test ------------------------------------------------
        // Ceres' termination_type only reports convergence for a fixed correspondence set. New
        // correspondences change the objective itself, so the outer convergence is judged here.
        const bool small_step = log.delta_translation < options_.translation_tolerance && log.delta_rotation < options_.rotation_tolerance;
        const bool error_stalled = std::abs(previous_error - log.error_after) < options_.error_tolerance;
        previous_error = log.error_after;
        if (small_step || error_stalled)
        {
            result.converged = true;
            break;
        }
    }

    // --- 5. re-pair the correspondences and report the final error --------------
    const std::vector<Correspondence> final_correspondences = find_correspondences(result.transform);
    result.correspondences = static_cast<int>(final_correspondences.size());
    if (final_correspondences.empty())
    {
        // Returning 0.0 here would read as a perfect fit; this is a failure, so report inf.
        const double infinity = std::numeric_limits<double>::infinity();
        result.final_error = infinity;
        result.final_mean_abs_error = infinity;
        result.final_max_abs_error = infinity;
        result.converged = false;
        return result;
    }
    evaluate_error(result.transform, final_correspondences, &result.final_error, &result.final_mean_abs_error, &result.final_max_abs_error);
    return result;
}

IcpResult IcpPointToPlane::do_icp(const PointCloud& source, const PointCloud& target)
{
    return do_icp(source, target, Eigen::Isometry3d::Identity());
}

IcpResult IcpPointToPlane::do_icp(const PointCloud& source, const PointCloud& target, const Eigen::Isometry3d& initial_guess)
{
    set_source(source);
    set_target(target);
    return do_icp(initial_guess);
}

}  // namespace p2p_icp
