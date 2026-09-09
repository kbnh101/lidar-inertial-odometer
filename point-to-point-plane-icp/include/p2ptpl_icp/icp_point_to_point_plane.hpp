#pragma once

#include <Eigen/Geometry>
#include <iomanip>
#include <memory>
#include <ostream>
#include <vector>

#include "common/kdtree.hpp"
#include "common/point_cloud.hpp"
#include "p2ptpl_icp/cuda_error.hpp"
#include "p2ptpl_icp/timing.hpp"

namespace ceres
{
class Context;
}

namespace p2ptpl_icp
{
// Reports linked Ceres build support, not whether a GPU is present at runtime.
bool CeresCudaSolverAvailable() noexcept;

/// ICP runtime parameters.
struct IcpOptions
{
    int max_iterations = 50;  ///< outer iteration limit (one correspondence search plus one solve)
    int max_solver_iterations = 20;  ///< Ceres iteration limit per subproblem

    double max_correspondence_distance = 1.0;  ///< pairs farther apart than this are rejected [m]
    /// Rejects a correspondence when the rotated source normal disagrees with the target normal.
    /// This is the *only* place the source normal n_x is used; it never enters the residual.
    /// -1 disables the test.
    double min_normal_dot = 0.0;

    double translation_tolerance = 1e-12;  ///< converged once |dt| falls below this [m]
    double rotation_tolerance = 1e-12;  ///< converged once |dR| falls below this [rad]
    double error_tolerance = 1e-14;  ///< converged once the RMS error stops improving by this much

    /// Pivots the incremental rotation at the source cloud centroid instead of the origin.
    /// The residual and the Jacobian do not change at all (x and y just shift by the same constant);
    /// it only decorrelates the rotation columns of J from the translation columns.
    /// On the provided data cond(J^T J) drops from 7.3e5 to 9.2.
    bool pivot_increment_at_centroid = true;

    /// Huber loss applied to the residual. 0 disables it, so noise-free data converges to exactly 0.
    /// Real LiDAR scans contain outliers, so keep it above 0 there.
    double huber_delta = 0.0;

    double point_weight = 1.0;  ///< positive coefficient of ||e||^2
    double plane_weight = 1.0;  ///< positive coefficient of (n^T e)^2

    bool use_cuda = CeresCudaSolverAvailable();  ///< Ceres CUDA DENSE_QR only; residuals/Jacobians always use CPU. False selects Eigen QR.

    bool verbose = false;  ///< prints the per-iteration progress to stdout
    bool collect_timing = false;  ///< Module profiling; demos enable this, ordinary LIO stays uninstrumented
};

/// Per-iteration diagnostics.
struct IcpIterationLog
{
    int iteration = 0;
    int correspondences = 0;
    double error_before = 0.0;  ///< RMS error before the solve [m]
    double error_after = 0.0;  ///< RMS error after the update [m]
    double delta_translation = 0.0;  ///< |dt| of this iteration [m]
    double delta_rotation = 0.0;  ///< |dR| of this iteration [rad]
    bool used_cuda_solver = false;  ///< Actual dense backend reported by Ceres for this iteration
    IcpTiming timing;
};

/// Final ICP result -- the source to target relative pose plus diagnostics.
struct IcpResult
{
    IcpTiming timing;  ///< Accumulated across iterations, including final correspondence/diagnostic passes
    Eigen::Isometry3d transform = Eigen::Isometry3d::Identity();  ///< source -> target: y ~ transform * x
    double final_error = 0.0;  ///< RMS |r_n| over the final correspondences [m]
    double final_mean_abs_error = 0.0;  ///< mean |r_n| [m]
    double final_max_abs_error = 0.0;  ///< max |r_n| [m]
    int iterations = 0;
    int correspondences = 0;
    bool converged = false;
    std::vector<IcpIterationLog> history;
};

/// Per-outer-iteration module timing, so a slow registration can be traced to the iteration and the
/// module that caused it. Columns are the disjoint stages of IcpTiming plus two Ceres detail
/// columns (lin.solv, cb_*) that run inside `solve` and must not be added to it.
inline void PrintIterationTiming(std::ostream& out, const std::vector<IcpIterationLog>& history, const char* title = "Per-iteration module timing")
{
    if (history.empty())
    {
        return;
    }
    const auto flags = out.flags();
    const auto precision = out.precision();
    out << '\n' << title << " [ms]\n"
        << std::right << std::setw(5) << "iter" << std::setw(8) << "corr" << std::setw(6) << "QR" << std::setw(10) << "search" << std::setw(9) << "pivot"
        << std::setw(9) << "setup" << std::setw(10) << "blocks" << std::setw(10) << "solve" << std::setw(10) << "lin.solv" << std::setw(9) << "cb_res"
        << std::setw(9) << "cb_jac" << std::setw(9) << "cleanup" << std::setw(8) << "pose" << std::setw(9) << "diag" << std::setw(10) << "total" << '\n';
    for (const IcpIterationLog& iteration : history)
    {
        const IcpTiming& timing = iteration.timing;
        out << std::right << std::setw(5) << iteration.iteration << std::setw(8) << iteration.correspondences << std::setw(6)
            << (iteration.used_cuda_solver ? "cuda" : "cpu") << std::fixed << std::setprecision(3) << std::setw(10) << timing.correspondence_ms << std::setw(9)
            << timing.pivot_ms << std::setw(9) << timing.problem_setup_ms << std::setw(10) << timing.residual_block_ms << std::setw(10)
            << timing.ceres_solve_ms << std::setw(10) << timing.ceres_linear_solver_ms << std::setw(9) << timing.callback_residual_only_ms << std::setw(9)
            << timing.callback_with_jacobian_ms << std::setw(9) << timing.problem_cleanup_ms << std::setw(8) << timing.pose_update_ms << std::setw(9)
            << timing.diagnostic_ms << std::setw(10) << timing.total_ms << '\n';
    }
    out.flags(flags);
    out.precision(precision);
}

/**
 * @brief Combined point-to-point and point-to-plane ICP
 *
 * Runs on the analytic Jacobian of PointCostFunction and PlaneCostFunction.
 *
 * Every outer iteration relinearizes: the source is transformed by the current pose, the
 * correspondences are searched again, and Ceres solves for an *increment* xi starting from 0.
 * Restarting each subproblem from 0 matters, because the Euler angles then stay near 0 and never
 * reach the beta = +-pi/2 gimbal lock.
 */
class IcpPointToPointPlane
{
public:
    IcpPointToPointPlane() = default;
    explicit IcpPointToPointPlane(const IcpOptions& options) : options_(options)
    {
    }

    /**
     * @brief Replaces the options
     *
     * @param options new options
     */
    void set_options(const IcpOptions& options)
    {
        options_ = options;
    }

    /**
     * @brief Current options, read only
     *
     * @return the options in use
     */
    const IcpOptions& options() const
    {
        return options_;
    }

    /**
     * @brief Accessor for editing the options in place
     *
     * @return the options in use, mutable
     */
    IcpOptions& mutable_options()
    {
        return options_;
    }

    /**
     * @brief Sets the source cloud
     *
     * @param source points with normals
     */
    void set_source(const common::PointCloud& source);

    /**
     * @brief Sets the target cloud, also building the kd-tree used for the correspondence search
     *
     * @param target points with normals
     */
    void set_target(const common::PointCloud& target);

    /**
     * @brief Runs ICP over the two clouds configured above
     *
     * @param initial_guess initial pose; the LIO passes the IMU preintegration prediction here
     * @return the final relative pose plus the error and convergence information
     */
    IcpResult do_icp(const Eigen::Isometry3d& initial_guess = Eigen::Isometry3d::Identity());

    /**
     * @brief Convenience overload setting both clouds and running, with an identity initial guess
     *
     * @param source source cloud
     * @param target target cloud
     * @return the final relative pose plus the error and convergence information
     */
    IcpResult do_icp(const common::PointCloud& source, const common::PointCloud& target);

    /**
     * @brief Convenience overload setting both clouds and running
     *
     * @param source source cloud
     * @param target target cloud
     * @param initial_guess initial pose
     * @return the final relative pose plus the error and convergence information
     */
    IcpResult do_icp(const common::PointCloud& source, const common::PointCloud& target, const Eigen::Isometry3d& initial_guess);

    /**
     * @brief The configured source cloud
     *
     * @return the source cloud
     */
    const common::PointCloud& source() const
    {
        return source_;
    }

    /**
     * @brief The configured target cloud
     *
     * @return the target cloud
     */
    const common::PointCloud& target() const
    {
        return target_;
    }

private:
    /// Index pair of one source point and its corresponding target point.
    struct Correspondence
    {
        int source_index = -1;
        int target_index = -1;
    };

    /**
     * @brief Finds the nearest target point for every source point transformed by @p pose
     *
     * Both the distance rejection and the normal agreement rejection are applied.
     *
     * @param pose the current pose estimate
     * @return the surviving correspondence pairs
     */
    std::vector<Correspondence> find_correspondences(const Eigen::Isometry3d& pose) const;

    /**
     * @brief Computes the RMS / mean / max of |r_n| over a given correspondence set
     *
     * @param pose the current pose estimate
     * @param correspondences the correspondence pairs
     * @param rms      output, nullptr allowed [m]
     * @param mean_abs output, nullptr allowed [m]
     * @param max_abs  output, nullptr allowed [m]
     */
    void evaluate_error(const Eigen::Isometry3d& pose, const std::vector<Correspondence>& correspondences, double* rms, double* mean_abs,
                        double* max_abs) const;

    IcpOptions options_;
    std::shared_ptr<ceres::Context> ceres_context_;  ///< Lazily initialized for CUDA solves; reused across scans
    common::PointCloud source_;
    common::PointCloud target_;
    common::KdTree3d target_tree_;
};

}  // namespace p2ptpl_icp
