// Demo: estimates the relative pose between source_point.txt and
// target_point.txt, printing the per-iteration error and the final pose.

#include <Eigen/Geometry>
#include <cstdio>
#include <iostream>
#include <string>

#include "common/point_cloud.hpp"
#include "common/rotation.hpp"
#include "p2pt_icp/icp_point_to_point.hpp"

namespace
{
const double kRadToDeg = 57.295779513082320876798154814105;

void print_pose(const Eigen::Isometry3d& pose)
{
    const Eigen::Vector3d translation = pose.translation();
    const Eigen::Vector3d euler = common::rotation_to_euler_zyx(pose.linear());

    std::printf("  t = [% .12f, % .12f, % .12f]  [m]\n", translation.x(), translation.y(), translation.z());
    std::printf("  euler ZYX (alpha, beta, gamma) = [% .12f, % .12f, % .12f]  [rad]\n", euler.x(), euler.y(), euler.z());
    std::printf("                                 = [% .8f, % .8f, % .8f]  [deg]\n", euler.x() * kRadToDeg, euler.y() * kRadToDeg, euler.z() * kRadToDeg);
    std::printf("  R =\n");
    for (int row = 0; row < 3; ++row)
    {
        std::printf("      [% .12f  % .12f  % .12f]\n", pose.linear()(row, 0), pose.linear()(row, 1), pose.linear()(row, 2));
    }
}

}  // namespace

int main(int argc, char** argv)
{
    std::string data_dir = P2PT_ICP_DATA_DIR;
    if (argc > 1)
    {
        data_dir = argv[1];
    }
    const std::string source_path = data_dir + "/source_point.txt";
    const std::string target_path = data_dir + "/target_point.txt";

    try
    {
        // The loader is shared with the point-to-plane project; the normal columns are ignored here.
        const common::PointCloud source_points = common::load_point_cloud(source_path);
        const common::PointCloud target_points = common::load_point_cloud(target_path);
        std::printf("loaded %zu source points from %s\n", source_points.size(), source_path.c_str());
        std::printf("loaded %zu target points from %s\n\n", target_points.size(), target_path.c_str());

        p2pt_icp::IcpOptions options;
        options.verbose = false;  // the loop below prints the table itself
        p2pt_icp::IcpPointToPoint icp(options);

        const p2pt_icp::IcpResult result = icp.do_icp(source_points, target_points);

        std::printf("%-5s %-8s %-24s %-24s %-12s %-12s\n", "iter", "corr", "rms error before [m]", "rms error after [m]", "|dt| [m]", "|dR| [rad]");
        std::printf("%s\n", std::string(92, '-').c_str());
        for (const p2pt_icp::IcpIterationLog& log : result.history)
        {
            std::printf("%-5d %-8d %-24.15e %-24.15e %-12.4e %-12.4e\n", log.iteration, log.correspondences, log.error_before, log.error_after,
                        log.delta_translation, log.delta_rotation);
        }

        std::printf("\nconverged            : %s after %d iteration(s)\n", result.converged ? "yes" : "no", result.iterations);
        std::printf("final correspondences: %d / %zu source points\n", result.correspondences, source_points.size());
        std::printf("final error (rms)    : %.15e [m]\n", result.final_error);
        std::printf("final error (mean)   : %.15e [m]\n", result.final_mean_abs_error);
        std::printf("final error (max)    : %.15e [m]\n", result.final_max_abs_error);
        std::printf("\nfinal relative pose (source -> target):\n");
        print_pose(result.transform);
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "error: " << error.what() << "\n";
        std::cerr << "usage: " << argv[0] << " [data_directory]\n";
        return 1;
    }
}
