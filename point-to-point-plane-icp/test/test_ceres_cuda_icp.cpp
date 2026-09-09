#include <gtest/gtest.h>
#include <random>
#include "icp_timing_report.hpp"
#include "p2ptpl_icp/icp_point_to_point_plane.hpp"
#include "common/rotation.hpp"

using namespace p2ptpl_icp;

TEST(CeresCudaIcp, MatchesCpuRegistrationWithNoiseAndOutliers)
{
    if (!CeresCudaSolverAvailable())
        GTEST_SKIP() << "Ceres was built without CUDA support";
    std::mt19937 random(42);
    std::uniform_real_distribution<double> value(-4, 4);
    std::normal_distribution<double> noise(0, .002);
    common::PointCloud source;
    for (int i = 0; i < 513; ++i)
    {
        Eigen::Vector3d point(value(random), value(random), value(random));
        source.push_back({point, point.normalized()});
    }
    Eigen::Isometry3d truth = Eigen::Isometry3d::Identity();
    truth.linear() = common::euler_zyx_to_rotation(.01, -.015, .02);
    truth.translation() = Eigen::Vector3d(.04, -.03, .06);
    auto target = common::transform_point_cloud(source, truth.linear(), truth.translation());
    for (std::size_t i = 0; i < target.size(); ++i)
    {
        target[i].point += Eigen::Vector3d(noise(random), noise(random), noise(random));
        if (i % 19 == 0)
            target[i].point += Eigen::Vector3d(.1, -.2, .1);
    }
    IcpOptions options = icp_test::TimedOptions();
    options.huber_delta = .03;
    options.point_weight = 2.0;
    options.plane_weight = .7;
    options.use_cuda = false;
    const auto cpu = IcpPointToPointPlane(options).do_icp(source, target);
    options.use_cuda = true;
    IcpPointToPointPlane gpu_icp(options);
    const auto gpu = gpu_icp.do_icp(source, target);
    ASSERT_TRUE(cpu.converged);
    ASSERT_TRUE(gpu.converged);
    ASSERT_FALSE(gpu.history.empty());
    for (const auto& iteration : cpu.history)
        EXPECT_FALSE(iteration.used_cuda_solver);
    for (const auto& iteration : gpu.history)
        EXPECT_TRUE(iteration.used_cuda_solver);
    EXPECT_EQ(cpu.correspondences, gpu.correspondences);
    EXPECT_LT((cpu.transform.matrix() - gpu.transform.matrix()).norm(), 1e-7);
    EXPECT_NEAR(cpu.final_error, gpu.final_error, 1e-9);
    EXPECT_LT((gpu.transform.translation() - truth.translation()).norm(), .005);
    // Same 513-point workload on both backends: only the ceres_* rows may differ, because the
    // residuals and Jacobians are evaluated on the CPU either way.
    icp_test::ReportTiming("513 points, Ceres DENSE_QR on CPU (Eigen)", cpu);
    icp_test::ReportTiming("513 points, Ceres DENSE_QR on CUDA", gpu);
    // Reuse the Ceres CUDA context on the next scan, then switch back to CPU.
    const auto repeated = gpu_icp.do_icp();
    ASSERT_TRUE(repeated.converged);
    EXPECT_LT((gpu.transform.matrix() - repeated.transform.matrix()).norm(), 1e-7);
    gpu_icp.mutable_options().use_cuda = false;
    const auto switched = gpu_icp.do_icp();
    ASSERT_TRUE(switched.converged);
    for (const auto& iteration : switched.history)
        EXPECT_FALSE(iteration.used_cuda_solver);
    EXPECT_LT((cpu.transform.matrix() - switched.transform.matrix()).norm(), 1e-7);
}

