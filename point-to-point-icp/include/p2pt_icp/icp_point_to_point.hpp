#pragma once

#include <Eigen/Geometry>
#include <vector>

#include "common/kdtree.hpp"
#include "common/point_cloud.hpp"

namespace p2pt_icp
{
/// ICP runtime parameters.
struct IcpOptions
{
    int max_iterations = 50;  ///< outer iteration limit (one correspondence search plus one solve)
    int max_solver_iterations = 20;  ///< Ceres iteration limit per subproblem

    double max_correspondence_distance = 1.0;  ///< pairs farther apart than this are rejected [m]

    double translation_tolerance = 1e-12;  ///< converged once |dt| falls below this [m]
    double rotation_tolerance = 1e-12;  ///< converged once |dR| falls below this [rad]
    double error_tolerance = 1e-14;  ///< converged once the RMS error stops improving by this much

    /// Pivots the incremental rotation at the source cloud centroid instead of the origin.
    /// The residual and the Jacobian do not change, it only decorrelates the rotation columns of J
    /// from the translation columns.
    bool pivot_increment_at_centroid = true;

    /// Huber loss applied to the residual. 0 disables it, so noise-free data converges to exactly 0.
    double huber_delta = 0.0;

    bool verbose = false;  ///< prints the per-iteration progress to stdout
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
};

/// Final ICP result -- the source to target relative pose plus diagnostics.
struct IcpResult
{
    Eigen::Isometry3d transform = Eigen::Isometry3d::Identity();  ///< source -> target: y ~ transform * x
    double final_error = 0.0;  ///< RMS |r_n| over the final correspondences [m]
    double final_mean_abs_error = 0.0;  ///< mean |r_n| [m]
    double final_max_abs_error = 0.0;  ///< max |r_n| [m]
    int iterations = 0;
    int correspondences = 0;
    bool converged = false;
    std::vector<IcpIterationLog> history;
};

/**
 * @brief Point-to-point ICP
 *
 * Minimizes the squared Euclidean distance between corresponding points, |R x_n + t - y_n|^2, over
 * the analytic Jacobian of PointToPointCostFunction.
 *
 * Every outer iteration relinearizes: transform the source by the current pose, search the
 * correspondences again, and solve for an increment xi starting from 0. Restarting each subproblem
 * from 0 keeps the Euler angles near 0, away from the beta = +-pi/2 gimbal lock.
 */
class IcpPointToPoint
{
public:
    IcpPointToPoint() = default;
    explicit IcpPointToPoint(const IcpOptions& options) : options_(options)
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
     * @param source the source points
     */
    void set_source(const common::PointCloud& source);

    /**
     * @brief Sets the target cloud, also building the kd-tree used for the correspondence search
     *
     * @param target the target points
     */
    void set_target(const common::PointCloud& target);

    /**
     * @brief Runs ICP over the two clouds configured above
     *
     * @param initial_guess initial pose
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
    common::PointCloud source_;
    common::PointCloud target_;
    common::KdTree3d target_tree_;
};

}  // namespace p2pt_icp
