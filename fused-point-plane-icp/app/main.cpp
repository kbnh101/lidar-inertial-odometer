// Demo: estimates the relative pose between source_point.txt and target_point.txt with the fused
// point-to-point + point-to-plane residual, printing the per-iteration error and the final pose.
//
//   icp_demo [data_directory] [alpha] [beta]

#include <Eigen/Geometry>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>

#include "common/point_cloud.hpp"
#include "common/rotation.hpp"
#include "fused_icp/icp_fused_point_plane.hpp"

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
    std::string data_dir = FUSED_ICP_DATA_DIR;
    if (argc > 1)
    {
        data_dir = argv[1];
    }
    const std::string source_path = data_dir + "/source_point.txt";
    const std::string target_path = data_dir + "/target_point.txt";

    try
    {
        const common::PointCloud source_points = common::load_point_cloud(source_path);
        const common::PointCloud target_points = common::load_point_cloud(target_path);
        std::printf("loaded %zu source points from %s\n", source_points.size(), source_path.c_str());
        std::printf("loaded %zu target points from %s\n\n", target_points.size(), target_path.c_str());

        fused_icp::IcpOptions options;
        options.point_weight = argc > 2 ? std::atof(argv[2]) : 1.0;
        options.plane_weight = argc > 3 ? std::atof(argv[3]) : 1.0;
        options.verbose = false;  // the loop below prints the table itself
        std::printf("weights: alpha (point) = %g, beta (plane) = %g\n\n", options.point_weight, options.plane_weight);
        fused_icp::IcpFusedPointPlane icp(options);

        const fused_icp::IcpResult result = icp.do_icp(source_points, target_points);

        std::printf("%-5s %-8s %-24s %-24s %-12s %-12s %-4s\n", "iter", "corr", "rms error before [m]", "rms error after [m]", "|dt| [m]", "|dR| [rad]", "lm");
        std::printf("%s\n", std::string(96, '-').c_str());
        for (const fused_icp::IcpIterationLog& log : result.history)
        {
            std::printf("%-5d %-8d %-24.15e %-24.15e %-12.4e %-12.4e %-4d\n", log.iteration, log.correspondences, log.error_before, log.error_after,
                        log.delta_translation, log.delta_rotation, log.solver_iterations);
        }

        std::printf("\nconverged            : %s after %d iteration(s)\n", result.converged ? "yes" : "no", result.iterations);
        std::printf("final correspondences: %d / %zu source points\n", result.correspondences, source_points.size());
        std::printf("final error (rms)    : %.15e [m]  (fused distance)\n", result.final_error);
        std::printf("final error (mean)   : %.15e [m]\n", result.final_mean_abs_error);
        std::printf("final error (max)    : %.15e [m]\n", result.final_max_abs_error);
        std::printf("final point rms |e|  : %.15e [m]\n", result.final_point_error);
        std::printf("final plane rms |n.e|: %.15e [m]\n", result.final_plane_error);
        std::printf("\nfinal relative pose (source -> target):\n");
        print_pose(result.transform);
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "error: " << error.what() << "\n";
        std::cerr << "usage: " << argv[0] << " [data_directory] [alpha] [beta]\n";
        return 1;
    }
}
