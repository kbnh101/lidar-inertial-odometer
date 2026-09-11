#include "fused_icp/icp_fused_point_plane.hpp"

#include <ceres/ceres.h>

#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>

#include "fused_icp/fused_cost.hpp"

namespace fused_icp
{
void IcpFusedPointPlane::set_source(const common::PointCloud& source)
{
    source_ = source;
}

void IcpFusedPointPlane::set_target(const common::PointCloud& target)
{
    target_ = target;
    target_tree_.build(target_);  // every correspondence search goes through this kd-tree
}

std::vector<IcpFusedPointPlane::IndexPair> IcpFusedPointPlane::find_correspondences(const Eigen::Isometry3d& pose) const
{
    std::vector<IndexPair> pairs;
    pairs.reserve(source_.size());

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
        // The derivation assumes |n| = 1 for every pair; a target point without a normal defines no
        // Omega = alpha I + beta n n^T and is left out, exactly as in point-to-plane.
        if (target_normal.squaredNorm() <= 0.0)
        {
            continue;
        }

        // Rejection 2, normal agreement -- the only place the source normal is used.
        if (options_.min_normal_dot > -1.0 && source_[i].normal.squaredNorm() > 0.0)
        {
            const Eigen::Vector3d rotated_source_normal = pose.linear() * source_[i].normal;
            if (rotated_source_normal.dot(target_normal) < options_.min_normal_dot)
            {
                continue;
            }
        }

        pairs.push_back(IndexPair{static_cast<int>(i), target_index});
    }
    return pairs;
}

void IcpFusedPointPlane::evaluate_error(const Eigen::Isometry3d& pose, const std::vector<IndexPair>& pairs, double* rms, double* mean_abs, double* max_abs,
                                        double* point_rms, double* plane_rms) const
{
    const FusedWeights weights{options_.point_weight, options_.plane_weight};
    // |r_f|^2 = alpha |e|^2 + beta (n^T e)^2; dividing by alpha + beta brings it back to metres.
    const double normalizer = weights.alpha + weights.beta;

    double sum_squared = 0.0;
    double sum_abs = 0.0;
    double max_value = 0.0;
    double sum_point_squared = 0.0;
    double sum_plane_squared = 0.0;
    for (const IndexPair& pair : pairs)
    {
        const Eigen::Vector3d e = geometric_error(pose, source_[pair.source_index].point, target_[pair.target_index].point);
        const double d = target_[pair.target_index].normal.dot(e);
        const double fused_squared = normalizer > 0.0 ? (weights.alpha * e.squaredNorm() + weights.beta * d * d) / normalizer : 0.0;
        const double fused = std::sqrt(fused_squared);

        sum_squared += fused_squared;
        sum_abs += fused;
        max_value = std::max(max_value, fused);
        sum_point_squared += e.squaredNorm();
        sum_plane_squared += d * d;
    }
    const double count = static_cast<double>(pairs.size());
    if (rms != nullptr)
    {
        *rms = pairs.empty() ? 0.0 : std::sqrt(sum_squared / count);
    }
    if (mean_abs != nullptr)
    {
        *mean_abs = pairs.empty() ? 0.0 : sum_abs / count;
    }
    if (max_abs != nullptr)
    {
        *max_abs = max_value;
    }
    if (point_rms != nullptr)
    {
        *point_rms = pairs.empty() ? 0.0 : std::sqrt(sum_point_squared / count);
    }
    if (plane_rms != nullptr)
    {
        *plane_rms = pairs.empty() ? 0.0 : std::sqrt(sum_plane_squared / count);
    }
}

IcpResult IcpFusedPointPlane::do_icp(const Eigen::Isometry3d& initial_guess)
{
    if (source_.empty())
    {
        throw std::runtime_error("IcpFusedPointPlane::do_icp: source cloud is empty");
    }
    if (target_.empty())
    {
        throw std::runtime_error("IcpFusedPointPlane::do_icp: target cloud is empty");
    }
    if (options_.point_weight < 0.0 || options_.plane_weight < 0.0 || options_.point_weight + options_.plane_weight <= 0.0)
    {
        throw std::runtime_error("IcpFusedPointPlane::do_icp: point_weight and plane_weight must be >= 0 and not both 0");
    }

    const FusedWeights weights{options_.point_weight, options_.plane_weight};

    IcpResult result;
    result.transform = initial_guess;

    double previous_error = std::numeric_limits<double>::infinity();

    for (int iteration = 0; iteration < options_.max_iterations; ++iteration)
    {
        // --- 1. search the correspondences with the current estimate -----------
        const std::vector<IndexPair> pairs = find_correspondences(result.transform);
        if (static_cast<int>(pairs.size()) < 6)
        {
            // With 6 unknowns and fewer than 6 constraints the problem is not solvable.
            if (options_.verbose)
            {
                std::printf("[icp] iteration %2d: only %zu correspondences, stopping\n", iteration, pairs.size());
            }
            break;
        }

        IcpIterationLog log;
        log.iteration = iteration;
        log.correspondences = static_cast<int>(pairs.size());
        evaluate_error(result.transform, pairs, &log.error_before, nullptr, nullptr, nullptr, nullptr);

        // --- 2. pivot: re-parametrize the source around its centroid -----------
        // p' = p - c and T' = T S with S = [I c; 0 1] leave x = T p = T' p' unchanged, so the
        // residual and the Jacobian keep the form of the note with p replaced by p'. The only effect
        // is on the conditioning of H: its rotation/translation coupling sum_i [p'_i]x vanishes.
        Eigen::Vector3d pivot = Eigen::Vector3d::Zero();
        if (options_.pivot_increment_at_centroid)
        {
            for (const IndexPair& pair : pairs)
            {
                pivot += source_[pair.source_index].point;
            }
            pivot /= static_cast<double>(pairs.size());
        }
        Eigen::Isometry3d shift = Eigen::Isometry3d::Identity();
        shift.translation() = pivot;

        // --- 3. solve the pose on the SE(3) right-perturbation manifold --------
        // The parameter block is the full pose T' (not an increment restarted at 0), so every Ceres
        // iteration relinearizes at its current pose and the analytic J_f of note (44) is exact
        // at every iterate.
        PoseParameters pose_parameters = pose_to_parameters(result.transform * shift);

        ceres::Problem::Options problem_options;
        problem_options.manifold_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
        ceres::Problem problem(problem_options);
        Se3RightManifold manifold;
        problem.AddParameterBlock(pose_parameters.data(), kPoseAmbientSize, &manifold);

        // One loss on the 3-D block is the joint robust cost rho(alpha |e|^2 + beta (n^T e)^2).
        ceres::LossFunction* loss = options_.huber_delta > 0.0 ? new ceres::HuberLoss(options_.huber_delta) : nullptr;

        for (const IndexPair& pair : pairs)
        {
            const Eigen::Vector3d source_point = source_[pair.source_index].point - pivot;
            problem.AddResidualBlock(
                    new FusedPointPlaneCostFunction(source_point, target_[pair.target_index].point, target_[pair.target_index].normal, weights), loss,
                    pose_parameters.data());
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
        log.solver_iterations = static_cast<int>(summary.iterations.size()) - 1;

        // --- 4. undo the pivot and measure the step ------------------------------
        const Eigen::Isometry3d previous = result.transform;
        result.transform = parameters_to_pose(pose_parameters.data()) * shift.inverse();

        const Eigen::Isometry3d step = previous.inverse() * result.transform;
        log.delta_translation = step.translation().norm();
        log.delta_rotation = Eigen::AngleAxisd(step.linear()).angle();
        evaluate_error(result.transform, pairs, &log.error_after, nullptr, nullptr, nullptr, nullptr);
        result.history.push_back(log);
        result.iterations = iteration + 1;
        result.correspondences = log.correspondences;

        if (options_.verbose)
        {
            std::printf("[icp] iter %2d | corr %4d | rms %.12e -> %.12e | |dt| %.3e | |dR| %.3e rad | lm %d\n", log.iteration, log.correspondences,
                        log.error_before, log.error_after, log.delta_translation, log.delta_rotation, log.solver_iterations);
        }

        // --- 5. convergence test ------------------------------------------------
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

    // --- 6. re-pair the correspondences and report the final error --------------
    const std::vector<IndexPair> final_pairs = find_correspondences(result.transform);
    result.correspondences = static_cast<int>(final_pairs.size());
    if (final_pairs.empty())
    {
        // Returning 0.0 here would read as a perfect fit; this is a failure, so report inf.
        const double infinity = std::numeric_limits<double>::infinity();
        result.final_error = infinity;
        result.final_mean_abs_error = infinity;
        result.final_max_abs_error = infinity;
        result.final_point_error = infinity;
        result.final_plane_error = infinity;
        result.converged = false;
        return result;
    }
    evaluate_error(result.transform, final_pairs, &result.final_error, &result.final_mean_abs_error, &result.final_max_abs_error, &result.final_point_error,
                   &result.final_plane_error);
    return result;
}

IcpResult IcpFusedPointPlane::do_icp(const common::PointCloud& source, const common::PointCloud& target)
{
    return do_icp(source, target, Eigen::Isometry3d::Identity());
}

IcpResult IcpFusedPointPlane::do_icp(const common::PointCloud& source, const common::PointCloud& target, const Eigen::Isometry3d& initial_guess)
{
    set_source(source);
    set_target(target);
    return do_icp(initial_guess);
}

}  // namespace fused_icp
