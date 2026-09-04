#pragma once

#include <Eigen/Core>
#include <string>
#include <vector>

namespace p2p_icp
{
/// One point together with its surface normal.
///
/// The point-to-plane residual uses the *target* normal only. The source side carries a `normal`
/// too because the correspondence rejection (the normal agreement test) needs it.
struct PointNormal
{
    Eigen::Vector3d point{Eigen::Vector3d::Zero()};
    Eigen::Vector3d normal{Eigen::Vector3d::Zero()};
};

/// A cloud is just a vector of PointNormal; no external point cloud library such as PCL is used.
using PointCloud = std::vector<PointNormal>;

/**
 * @brief Loads a cloud from a text file in `x y z nx ny nz` format
 *
 * Blank lines and lines starting with '#' are skipped.
 *
 * Normals are normalized right after loading, because the point-to-plane distance formula assumes
 * |n_y| = 1. A zero-length normal is left at zero and gets filtered out by the correspondence search.
 *
 * @param path file path
 * @return the loaded cloud
 * @throws std::runtime_error when the file cannot be opened or a line is malformed
 */
PointCloud load_point_cloud(const std::string& path);

/**
 * @brief Applies a rigid transform to a whole cloud, rotating the normals only
 *
 * @param cloud input cloud
 * @param rotation R
 * @param translation t
 * @return the transformed cloud
 */
PointCloud transform_point_cloud(const PointCloud& cloud, const Eigen::Matrix3d& rotation, const Eigen::Vector3d& translation);

}  // namespace p2p_icp
