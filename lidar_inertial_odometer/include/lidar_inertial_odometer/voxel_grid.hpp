#pragma once

#include <Eigen/Core>
#include <cmath>
#include <cstdint>
#include <unordered_map>
#include <vector>

#include "common/point_cloud.hpp"

/// A 3-D voxel index, used as the key of the spatial hash.
struct VoxelKey
{
    std::int64_t x = 0;
    std::int64_t y = 0;
    std::int64_t z = 0;

    bool operator==(const VoxelKey& other) const
    {
        return x == other.x && y == other.y && z == other.z;
    }
};

struct VoxelKeyHash
{
    std::size_t operator()(const VoxelKey& key) const
    {
        // The standard spatial hash: mix the coordinates with large primes.
        const std::size_t h1 = static_cast<std::size_t>(key.x) * 73856093u;
        const std::size_t h2 = static_cast<std::size_t>(key.y) * 19349663u;
        const std::size_t h3 = static_cast<std::size_t>(key.z) * 83492791u;
        return h1 ^ h2 ^ h3;
    }
};

inline VoxelKey MakeVoxelKey(const Eigen::Vector3d& point, double voxel_size)
{
    VoxelKey key;
    key.x = static_cast<std::int64_t>(std::floor(point.x() / voxel_size));
    key.y = static_cast<std::int64_t>(std::floor(point.y() / voxel_size));
    key.z = static_cast<std::int64_t>(std::floor(point.z() / voxel_size));
    return key;
}

/// Accumulates points with normals into a single voxel.
/// The Eigen members must be explicitly zero initialized: Eigen's default constructor leaves them
/// unset, so a += onto an element created by unordered_map::operator[] would start from garbage.
struct NormalAccumulator
{
    Eigen::Vector3d point = Eigen::Vector3d::Zero();
    Eigen::Vector3d normal = Eigen::Vector3d::Zero();
    int count = 0;
};

/// Accumulates points without normals.
struct PointAccumulator
{
    Eigen::Vector3d sum = Eigen::Vector3d::Zero();
    int count = 0;
};

typedef std::unordered_map<VoxelKey, NormalAccumulator, VoxelKeyHash> NormalVoxelMap;
typedef std::unordered_map<VoxelKey, PointAccumulator, VoxelKeyHash> PointVoxelMap;

/**
 * @brief Keeps one centroid point per voxel, averaging then renormalizing the normals
 *
 * @param cloud      input cloud, with normals
 * @param voxel_size voxel edge length [m]; the cloud is returned unchanged when not positive
 * @return the downsampled cloud
 */
inline common::PointCloud VoxelDownsample(const common::PointCloud& cloud, double voxel_size)
{
    if (voxel_size <= 0.0 || cloud.empty())
    {
        return cloud;
    }

    NormalVoxelMap voxels;
    voxels.reserve(cloud.size());

    for (const common::PointNormal& item : cloud)
    {
        NormalAccumulator& accumulator = voxels[MakeVoxelKey(item.point, voxel_size)];
        accumulator.point += item.point;
        // Normals on one plane can still point opposite ways, so align the sign before adding them
        // (adding them directly would cancel them out to zero).
        if (accumulator.count > 0 && accumulator.normal.dot(item.normal) < 0.0)
        {
            accumulator.normal -= item.normal;
        }
        else
        {
            accumulator.normal += item.normal;
        }
        ++accumulator.count;
    }

    common::PointCloud downsampled;
    downsampled.reserve(voxels.size());
    for (NormalVoxelMap::const_iterator it = voxels.begin(); it != voxels.end(); ++it)
    {
        const NormalAccumulator& accumulator = it->second;

        common::PointNormal item;
        item.point = accumulator.point / static_cast<double>(accumulator.count);
        if (accumulator.normal.squaredNorm() > 1e-12)
        {
            item.normal = accumulator.normal.normalized();
        }
        downsampled.push_back(item);
    }
    return downsampled;
}

/**
 * @brief Overload for a point set without normals
 *
 * @param points     input points
 * @param voxel_size voxel edge length [m]; the points are returned unchanged when not positive
 * @return the downsampled points
 */
inline std::vector<Eigen::Vector3d> VoxelDownsample(const std::vector<Eigen::Vector3d>& points, double voxel_size)
{
    if (voxel_size <= 0.0 || points.empty())
    {
        return points;
    }
    PointVoxelMap voxels;
    voxels.reserve(points.size());

    for (const Eigen::Vector3d& point : points)
    {
        PointAccumulator& accumulator = voxels[MakeVoxelKey(point, voxel_size)];
        accumulator.sum += point;
        ++accumulator.count;
    }

    std::vector<Eigen::Vector3d> downsampled;
    downsampled.reserve(voxels.size());

    for (PointVoxelMap::const_iterator it = voxels.begin(); it != voxels.end(); ++it)
    {
        const PointAccumulator& accumulator = it->second;
        downsampled.push_back(accumulator.sum / static_cast<double>(accumulator.count));
    }

    return downsampled;
}
