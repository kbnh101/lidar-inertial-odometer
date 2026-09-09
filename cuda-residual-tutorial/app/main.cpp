#include "cuda_residual_tutorial/residual.hpp"
#include "common/kdtree.hpp"
#include "common/point_cloud.hpp"
#include "common/rotation.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace tutorial = cuda_residual_tutorial;
namespace fs = std::filesystem;

namespace
{
fs::path default_data_dir()
{
    // Installed layout: <prefix>/lib/cuda_residual_tutorial/cuda_residual_demo.
    std::error_code error;
    const auto executable = fs::read_symlink("/proc/self/exe", error);
    if (!error)
    {
        const auto installed = executable.parent_path().parent_path().parent_path() / "share/cuda_residual_tutorial/data";
        if (fs::exists(installed / "source_point.txt") && fs::exists(installed / "target_point.txt"))
        {
            return installed;
        }
    }
    return TUTORIAL_DATA_DIR;
}

double parse_number(const std::string& value)
{
    std::size_t end = 0;
    const double number = std::stod(value, &end);
    if (end != value.size() || !std::isfinite(number))
    {
        throw std::invalid_argument("expected a finite number: " + value);
    }
    return number;
}

void validate_cloud(const common::PointCloud& cloud)
{
    for (const auto& entry : cloud)
    {
        if (!entry.point.allFinite() || !entry.normal.allFinite())
        {
            throw std::runtime_error("cloud contains non-finite coordinates or normals");
        }
    }
}
}  // namespace

int main(int argc, char** argv)
{
    try
    {
        const auto data = default_data_dir();
        std::string source_path = (data / "source_point.txt").string();
        std::string target_path = (data / "target_point.txt").string();
        std::string output_path;
        double xi[6] = {};  // tx ty tz roll pitch yaw, angles in radians.
        bool verify = false;
        for (int arg = 1; arg < argc; ++arg)
        {
            const std::string option = argv[arg];
            auto next = [&]() -> std::string
            {
                if (++arg >= argc)
                {
                    throw std::invalid_argument("missing value for " + option);
                }
                return argv[arg];
            };
            if (option == "--source")
            {
                source_path = next();
            }
            else if (option == "--target")
            {
                target_path = next();
            }
            else if (option == "--output")
            {
                output_path = next();
            }
            else if (option == "--pose")
            {
                for (double& value : xi)
                {
                    value = parse_number(next());
                }
            }
            else if (option == "--verify")
            {
                verify = true;
            }
            else if (option == "--help")
            {
                std::cout << "Usage: cuda_residual_demo [--source FILE] [--target FILE]\n"
                             "       [--pose tx ty tz roll pitch yaw] [--output FILE.csv] [--verify]\n"
                             "Defaults: repository/installed sample data, identity pose; angles in radians.\n"
                             "CPU: load + nearest neighbors. GPU: point (3D) and plane (1D) residuals.\n";
                return 0;
            }
            else
            {
                throw std::invalid_argument("unknown option: " + option);
            }
        }

        // Step 1 (CPU): shared loader reads x y z nx ny nz and normalizes normals.
        const auto source = common::load_point_cloud(source_path);
        const auto target = common::load_point_cloud(target_path);
        validate_cloud(source);
        validate_cloud(target);
        const Eigen::Matrix3d rotation = common::euler_zyx_to_rotation(xi[3], xi[4], xi[5]);
        const Eigen::Vector3d translation(xi[0], xi[1], xi[2]);
        tutorial::Pose pose;
        for (int row = 0; row < 3; ++row)
        {
            pose.translation[row] = translation[row];
            for (int col = 0; col < 3; ++col)
            {
                pose.rotation[3 * row + col] = rotation(row, col);
            }
        }

        // Step 2 (CPU): query R*x+t, but upload ORIGINAL x so CUDA applies the pose once.
        common::KdTree3d tree;
        tree.build(target);
        std::vector<tutorial::Pair> pairs;
        std::vector<std::pair<std::size_t, std::size_t>> indices;
        pairs.reserve(source.size());
        indices.reserve(source.size());
        std::size_t skipped = 0;
        for (std::size_t i = 0; i < source.size(); ++i)
        {
            const Eigen::Vector3d query = rotation * source[i].point + translation;
            int nearest = -1;
            double distance_squared = 0;
            if (!query.allFinite() || !tree.nearest(query, &nearest, &distance_squared) || !std::isfinite(distance_squared))
            {
                throw std::runtime_error("nearest-neighbor query failed or overflowed");
            }
            const auto& match = target.at(static_cast<std::size_t>(nearest));
            if (match.normal.squaredNorm() == 0)
            {
                ++skipped;  // Plane residual needs a valid target normal.
                continue;
            }
            tutorial::Pair pair{};
            for (int axis = 0; axis < 3; ++axis)
            {
                pair.source[axis] = source[i].point[axis];
                pair.target[axis] = match.point[axis];
                pair.normal[axis] = match.normal[axis];
            }
            pairs.push_back(pair);
            indices.emplace_back(i, static_cast<std::size_t>(nearest));
        }
        if (pairs.empty())
        {
            throw std::runtime_error("no valid pairs (nearest targets have zero normals)");
        }

        // Step 3 (GPU): a single batch; residuals[i] belongs to pairs[i].
        const auto residuals = tutorial::evaluate_cuda(pairs, pose);

        // Step 4 (CPU): display / optional reference check. This is not a solver.
        double point_squared_sum = 0;
        double plane_squared_sum = 0;
        double max_difference = 0;
        for (std::size_t i = 0; i < residuals.size(); ++i)
        {
            const auto& residual = residuals[i];
            const Eigen::Vector3d gpu(residual.point[0], residual.point[1], residual.point[2]);
            if (!gpu.allFinite() || !std::isfinite(residual.plane))
            {
                throw std::runtime_error("CUDA produced a non-finite residual");
            }
            point_squared_sum += gpu.squaredNorm();
            plane_squared_sum += residual.plane * residual.plane;
            if (verify)
            {
                const auto [s, t] = indices[i];
                const Eigen::Vector3d reference = rotation * source[s].point + translation - target[t].point;
                const double plane_reference = target[t].normal.dot(reference);
                for (int axis = 0; axis < 4; ++axis)
                {
                    const double expected = axis < 3 ? reference[axis] : plane_reference;
                    const double actual = axis < 3 ? gpu[axis] : residual.plane;
                    const double difference = std::abs(expected - actual);
                    max_difference = std::max(max_difference, difference);
                    if (difference > 1e-10 * (1 + std::abs(expected)))
                    {
                        throw std::runtime_error("CPU/GPU residual mismatch at pair " + std::to_string(i));
                    }
                }
            }
        }
        std::cout << std::setprecision(12) << "source=" << source.size() << " target=" << target.size() << " pairs=" << pairs.size()
                  << " skipped_zero_normal=" << skipped << '\n'
                  << "Point RMSE (sqrt(mean(||r_point||^2))): " << std::sqrt(point_squared_sum / pairs.size()) << '\n'
                  << "Plane RMSE (sqrt(mean(r_plane^2))): " << std::sqrt(plane_squared_sum / pairs.size()) << '\n';
        if (verify)
        {
            std::cout << "CPU/GPU check passed; max absolute difference=" << max_difference << '\n';
        }
        std::cout << "source_index,target_index,rx,ry,rz,r_plane (first 5 pairs)\n";
        auto write_row = [&](std::ostream& out, std::size_t i)
        {
            const auto& r = residuals[i];
            out << indices[i].first << ',' << indices[i].second << ',' << r.point[0] << ',' << r.point[1] << ',' << r.point[2] << ',' << r.plane << '\n';
        };
        for (std::size_t i = 0; i < std::min<std::size_t>(5, pairs.size()); ++i)
        {
            write_row(std::cout, i);
        }
        if (!output_path.empty())
        {
            // Avoid accidentally replacing either input cloud, including via symlinks.
            const auto output = fs::weakly_canonical(output_path);
            if (output == fs::canonical(source_path) || output == fs::canonical(target_path) ||
                (fs::exists(output) && (fs::equivalent(output, source_path) || fs::equivalent(output, target_path))))
            {
                throw std::invalid_argument("output must differ from input files");
            }
            std::ofstream csv;
            csv.exceptions(std::ios::failbit | std::ios::badbit);
            csv.open(output_path);
            csv << std::setprecision(17) << "source_index,target_index,rx,ry,rz,r_plane\n";
            for (std::size_t i = 0; i < pairs.size(); ++i)
            {
                write_row(csv, i);
            }
            csv.close();
            std::cout << "Saved " << pairs.size() << " residuals to " << output_path << '\n';
        }
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "cuda_residual_demo: " << error.what() << '\n';
        return 1;
    }
}
