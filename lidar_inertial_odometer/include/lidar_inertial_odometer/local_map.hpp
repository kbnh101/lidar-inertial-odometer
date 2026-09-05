#pragma once

#include <Eigen/Geometry>
#include <deque>

#include "lidar_inertial_odometer/util.hpp"
#include "common/point_cloud.hpp"

/**
 * @brief Sliding-window local map that serves as the target of the scan-to-map ICP
 *
 * Each keyframe holds a world frame cloud, and the merged cloud handed to
 * `IcpPointToPlane::set_target()` is rebuilt whenever a keyframe is added or evicted.
 *
 * A window is kept instead of a global map for two reasons: it holds the kd-tree size constant, and
 * it stops old observations, whose drift has accumulated, from pulling on the current registration.
 *
 * @see LocalMapOptions (util.hpp)
 */
class LocalMap
{
public:
    LocalMap() = default;
    explicit LocalMap(const LocalMapOptions& options) : options_(options)
    {
    }

    /**
     * @brief Replaces the options
     *
     * @param options new options
     */
    void set_options(const LocalMapOptions& options)
    {
        options_ = options;
    }

    /**
     * @brief Current options
     *
     * @return the options in use
     */
    const LocalMapOptions& options() const
    {
        return options_;
    }

    /**
     * @brief Adds one world frame cloud as a keyframe
     *
     * @param cloud_world planar features with normals, in the world frame
     * @param viewpoint   centre of the crop, normally the current IMU position
     */
    void AddKeyframe(const common::PointCloud& cloud_world, const Eigen::Vector3d& viewpoint);

    /**
     * @brief The merged and downsampled map, usable as the ICP target as is
     *
     * @return the map cloud
     */
    const common::PointCloud& cloud() const
    {
        return merged_;
    }

    /**
     * @brief Whether the map is empty
     *
     * @return true when it holds no points
     */
    bool empty() const
    {
        return merged_.empty();
    }

    /**
     * @brief Number of keyframes inside the window
     *
     * @return the keyframe count
     */
    int num_keyframes() const
    {
        return static_cast<int>(keyframes_.size());
    }

    /**
     * @brief Empties the map
     */
    void Clear();

private:
    /**
     * @brief Rebuilds the map from the keyframes in the window, by cropping and voxel downsampling
     *
     * @param viewpoint centre of the crop
     */
    void Rebuild(const Eigen::Vector3d& viewpoint);

    LocalMapOptions options_;
    std::deque<common::PointCloud> keyframes_;
    common::PointCloud merged_;
};
