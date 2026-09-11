#pragma once

#include <Eigen/Geometry>
#include <vector>

#include "common/kdtree.hpp"
#include "common/point_cloud.hpp"
#include "fused_icp/fused_residual.hpp"

namespace fused_icp
{
/// ICP runtime parameters.
struct IcpOptions
{
    int max_iterations = 50;  ///< outer iteration limit (one correspondence search plus one solve)
    int max_solver_iterations = 20;  ///< Ceres (Levenberg-Marquardt) iteration limit per subproblem

    /// Point-to-point weight alpha and point-to-plane weight beta of the fused cost
    /// 2C = alpha |e|^2 + beta (n^T e)^2. Tangent-plane errors are weighted by alpha, errors along
    /// the normal by alpha + beta. alpha = 0 is pure point-to-plane, beta = 0 pure point-to-point.
    /// With 1/sigma^2 weights the residual is in units of standard deviations; with beta = 1 and
    /// alpha in [0, 1] it stays in metres along the normal.
    double point_weight = 1.0;  ///< alpha >= 0
    double plane_weight = 1.0;  ///< beta >= 0

    double max_correspondence_distance = 1.0;  ///< pairs farther apart than this are rejected [m]
    /// Rejects a correspondence when the rotated source normal disagrees with the target normal.
    /// This is the *only* place the source normal is used; it never enters the residual.
    /// -1 disables the test.
    double min_normal_dot = 0.0;

    double translation_tolerance = 1e-12;  ///< converged once |dt| falls below this [m]
    double rotation_tolerance = 1e-12;  ///< converged once |dR| falls below this [rad]
    double error_tolerance = 1e-14;  ///< converged once the RMS error stops improving by this much

    /// Re-parametrizes the source around its centroid c before the solve: p' = p - c and
    /// T' = T [I c; 0 1], so the right perturbation pivots at the centroid instead of the sensor
    /// origin. The residual, G and J_f keep their form (only p is replaced by p'); it just removes
    /// the sum_i [p_i]x coupling between the rotation and translation blocks of H.
    bool pivot_increment_at_centroid = true;

    /// Huber loss applied to |r_f|^2, i.e. the joint loss rho(alpha |e|^2 + beta (n^T e)^2) of the
    /// note (61). 0 disables it, so noise-free data converges to exactly 0. Real LiDAR scans
    /// contain outliers, so keep it above 0 there.
    double huber_delta = 0.0;

    bool verbose = false;  ///< prints the per-iteration progress to stdout
};

/// Per-iteration diagnostics.
struct IcpIterationLog
{
    int iteration = 0;
    int correspondences = 0;
    double error_before = 0.0;  ///< RMS fused distance before the solve [m]
    double error_after = 0.0;  ///< RMS fused distance after the update [m]
    double delta_translation = 0.0;  ///< |dt| of this iteration [m]
    double delta_rotation = 0.0;  ///< |dR| of this iteration [rad]
    int solver_iterations = 0;  ///< Ceres iterations spent on the subproblem
};

/// Final ICP result -- the source to target relative pose plus diagnostics.
///
/// "fused distance" is |r_f| / sqrt(alpha + beta) = sqrt((alpha |e_tan|^2 + (alpha+beta) |e_n|^2) / (alpha+beta)),
/// in metres. It reduces to |n^T e| for alpha = 0 and to |e| for beta = 0.
struct IcpResult
{
    Eigen::Isometry3d transform = Eigen::Isometry3d::Identity();  ///< source -> target: q ~ transform * p
    double final_error = 0.0;  ///< RMS fused distance over the final correspondences [m]
    double final_mean_abs_error = 0.0;  ///< mean fused distance [m]
    double final_max_abs_error = 0.0;  ///< max fused distance [m]
    double final_point_error = 0.0;  ///< RMS |e|, the point-to-point part alone [m]
    double final_plane_error = 0.0;  ///< RMS |n^T e|, the point-to-plane part alone [m]
    int iterations = 0;
    int correspondences = 0;
    bool converged = false;
    std::vector<IcpIterationLog> history;
};

/**
 * @brief Fused point-to-point + point-to-plane ICP
 *
 * Every correspondence contributes one 3-D residual r_f = L e (FusedPointPlaneCostFunction) whose
 * squared norm is alpha |e|^2 + beta (n^T e)^2; the analytic Jacobian J_f = L G is used, no AutoDiff.
 *
 * Every outer iteration relinearizes: the source is transformed by the current pose, the
 * correspondences are searched again, and Ceres solves the pose on Se3RightManifold, i.e. with the
 * right perturbation T+ = T Exp(delta_xi) of the derivation note. Because the pose block lives on
 * the manifold there is no Euler angle parametrization and no gimbal lock to avoid.
 */
class IcpFusedPointPlane
{
public:
    IcpFusedPointPlane() = default;
    explicit IcpFusedPointPlane(const IcpOptions& options) : options_(options)
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
    struct IndexPair
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
    std::vector<IndexPair> find_correspondences(const Eigen::Isometry3d& pose) const;

    /**
     * @brief Computes the RMS / mean / max fused distance and the two component RMS values
     *
     * @param pose the current pose estimate
     * @param pairs the correspondence pairs
     * @param rms       output, nullptr allowed [m]
     * @param mean_abs  output, nullptr allowed [m]
     * @param max_abs   output, nullptr allowed [m]
     * @param point_rms output, RMS |e|, nullptr allowed [m]
     * @param plane_rms output, RMS |n^T e|, nullptr allowed [m]
     */
    void evaluate_error(const Eigen::Isometry3d& pose, const std::vector<IndexPair>& pairs, double* rms, double* mean_abs, double* max_abs, double* point_rms,
                        double* plane_rms) const;

    IcpOptions options_;
    common::PointCloud source_;
    common::PointCloud target_;
    common::KdTree3d target_tree_;
};

}  // namespace fused_icp
