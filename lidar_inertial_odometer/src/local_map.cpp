#include "lidar_inertial_odometer/local_map.hpp"

#include "lidar_inertial_odometer/voxel_grid.hpp"

void LocalMap::Clear()
{
    keyframes_.clear();
    merged_.clear();
}

void LocalMap::AddKeyframe(const common::PointCloud& cloud_world, const Eigen::Vector3d& viewpoint)
{
    if (cloud_world.empty())
    {
        return;
    }
    keyframes_.push_back(cloud_world);
    // Once the window overflows, evict the oldest keyframe first.
    while (static_cast<int>(keyframes_.size()) > options_.max_keyframes)
    {
        keyframes_.pop_front();
    }
    Rebuild(viewpoint);
}

void LocalMap::Rebuild(const Eigen::Vector3d& viewpoint)
{
    common::PointCloud accumulated;
    std::size_t total = 0;
    for (const common::PointCloud& keyframe : keyframes_)
    {
        total += keyframe.size();
    }
    accumulated.reserve(total);

    const double crop_squared = options_.crop_radius * options_.crop_radius;
    for (const common::PointCloud& keyframe : keyframes_)
    {
        for (const common::PointNormal& item : keyframe)
        {
            if (options_.crop_radius > 0.0 && (item.point - viewpoint).squaredNorm() > crop_squared)
            {
                continue;
            }
            accumulated.push_back(item);
        }
    }

    // The window's points, cropped and voxel downsampled, become the ICP target of the next frame.
    merged_ = VoxelDownsample(accumulated, options_.voxel_size);
}
