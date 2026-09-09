// Module timing profile of do_icp(): which module dominates, how each one scales with the point
// count, and what the Ceres dense QR backend changes. The assertions only keep the workload
// honest (the registrations must converge); the numbers are printed, never asserted on, because
// they depend on the machine and the build type. Build Release before quoting any of them.
#include <gtest/gtest.h>

#include <random>
#include <string>
#include <vector>

#include "common/rotation.hpp"
#include "icp_timing_report.hpp"

namespace
{
constexpr int kRepeat = 3;  ///< first run warms the allocator/caches, the rest are steady state

/// A small known transform: ICP converges in a few outer iterations, so the tables show the
/// steady-state cost of a module instead of a slow-convergence artefact.
Eigen::Isometry3d Truth()
{
    Eigen::Isometry3d truth = Eigen::Isometry3d::Identity();
    truth.linear() = common::euler_zyx_to_rotation(.01, -.015, .02);
    truth.translation() = Eigen::Vector3d(.04, -.03, .06);
    return truth;
}

common::PointCloud MakeCloud(int count)
{
    std::mt19937 random(7);
    std::uniform_real_distribution<double> value(-20, 20);
    common::PointCloud cloud;
    cloud.reserve(count);
    for (int i = 0; i < count; ++i)
    {
        const Eigen::Vector3d point(value(random), value(random), value(random));
        cloud.push_back({point, point.normalized()});
    }
    return cloud;
}

/// Registers @p source onto @p target @p kRepeat times from a fresh matcher and reports the modules.
std::vector<p2ptpl_icp::IcpTiming> Profile(const std::string& label, const p2ptpl_icp::IcpOptions& options, const common::PointCloud& source,
                                           const common::PointCloud& target)
{
    std::vector<p2ptpl_icp::IcpTiming> samples;
    p2ptpl_icp::IcpResult result;
    for (int run = 0; run < kRepeat; ++run)
    {
        p2ptpl_icp::IcpPointToPointPlane icp(options);  // fresh target kd-tree and Ceres problem every run
        result = icp.do_icp(source, target);
        EXPECT_TRUE(result.converged) << label << " run " << run;
        samples.push_back(result.timing);
    }
    icp_test::ReportTimingSamples(label, samples);
    p2ptpl_icp::PrintIterationTiming(std::cout, result.history, ("  " + label + ", last run per outer iteration").c_str());
    return samples;
}
}  // namespace

TEST(IcpModuleTiming, ScalesWithPointCount)
{
    for (const int count : {500, 2000, 8000})
    {
        const auto source = MakeCloud(count);
        const auto target = common::transform_point_cloud(source, Truth().linear(), Truth().translation());
        auto options = icp_test::TimedOptions();
        options.max_iterations = 10;
        options.use_cuda = false;  // CPU QR keeps this sweep comparable across machines
        Profile(std::to_string(count) + " points, CPU QR", options, source, target);
    }
}

TEST(IcpModuleTiming, ComparesCeresDenseQrBackends)
{
    if (!p2ptpl_icp::CeresCudaSolverAvailable())
    {
        GTEST_SKIP() << "Ceres was built without CUDA support; only the CPU QR profile is available";
    }
    const auto source = MakeCloud(8000);
    const auto target = common::transform_point_cloud(source, Truth().linear(), Truth().translation());
    auto options = icp_test::TimedOptions();
    options.max_iterations = 10;
    // Only ceres_solve_ms / ceres_linear_solver_ms may differ: residuals and Jacobians run on the
    // CPU in both configurations, so every other module is a control.
    options.use_cuda = false;
    const auto cpu = Profile("8000 points, Ceres DENSE_QR on CPU (Eigen)", options, source, target);
    options.use_cuda = true;
    const auto cuda = Profile("8000 points, Ceres DENSE_QR on CUDA", options, source, target);
    std::cout << "\nCeres linear solver mean [ms]: cpu=" << p2ptpl_icp::TimingStatisticsOf(cpu, &p2ptpl_icp::IcpTiming::ceres_linear_solver_ms).mean
              << " cuda=" << p2ptpl_icp::TimingStatisticsOf(cuda, &p2ptpl_icp::IcpTiming::ceres_linear_solver_ms).mean
              << "; whole registration mean [ms]: cpu=" << p2ptpl_icp::TimingStatisticsOf(cpu, &p2ptpl_icp::IcpTiming::total_ms).mean
              << " cuda=" << p2ptpl_icp::TimingStatisticsOf(cuda, &p2ptpl_icp::IcpTiming::total_ms).mean << '\n';
}
