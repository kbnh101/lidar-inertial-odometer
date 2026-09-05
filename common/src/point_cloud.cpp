#include "common/point_cloud.hpp"

#include <fstream>
#include <sstream>
#include <stdexcept>

namespace common
{
PointCloud load_point_cloud(const std::string& path)
{
    std::ifstream file(path);
    if (!file.is_open())
    {
        throw std::runtime_error("load_point_cloud: cannot open file '" + path + "'");
    }

    PointCloud cloud;
    std::string line;
    std::size_t line_number = 0;
    while (std::getline(file, line))
    {
        ++line_number;
        // Skip comments (#) and blank lines.
        const std::size_t first = line.find_first_not_of(" \t\r\n");
        if (first == std::string::npos || line[first] == '#')
        {
            continue;
        }

        std::istringstream stream(line);
        PointNormal entry;
        if (!(stream >> entry.point.x() >> entry.point.y() >> entry.point.z() >> entry.normal.x() >> entry.normal.y() >> entry.normal.z()))
        {
            throw std::runtime_error("load_point_cloud: malformed line " + std::to_string(line_number) + " in '" + path + "' (expected: x y z nx ny nz)");
        }

        // Normalize here to guarantee the |n_y| = 1 the point-to-plane distance formula assumes; a
        // zero-length normal is left at zero so the ICP correspondence search filters it out.
        const double norm = entry.normal.norm();
        if (norm > 0.0)
        {
            entry.normal /= norm;
        }
        else
        {
            entry.normal.setZero();
        }
        cloud.push_back(entry);
    }

    if (cloud.empty())
    {
        throw std::runtime_error("load_point_cloud: '" + path + "' contains no points");
    }
    return cloud;
}

PointCloud transform_point_cloud(const PointCloud& cloud, const Eigen::Matrix3d& rotation, const Eigen::Vector3d& translation)
{
    PointCloud out;
    out.reserve(cloud.size());
    for (const PointNormal& entry : cloud)
    {
        PointNormal transformed;
        transformed.point = rotation * entry.point + translation;
        transformed.normal = rotation * entry.normal;  // a normal is a direction, so it is only rotated
        out.push_back(transformed);
    }
    return out;
}

}  // namespace common
