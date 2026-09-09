#include <gtest/gtest.h>
#include <ceres/ceres.h>
#include <array>
#include <random>
#include "icp_timing_report.hpp"
#include "p2ptpl_icp/cost_functions.hpp"
#include "p2ptpl_icp/icp_point_to_point_plane.hpp"

using namespace p2ptpl_icp;

TEST(HybridResidual, SharesErrorAndMatchesAnalyticJacobiansAtNonzeroPose)
{
    std::array<double, 6> xi{.2, -.1, .3, .13, -.21, .34};
    SharedErrorEvaluation cache(xi.data());
    auto error = cache.Add({1, 2, 3}, {-.4, .2, 1}, Eigen::Vector3d(1, 2, -1).normalized());
    PointCostFunction point(error, 1);
    PlaneCostFunction plane(error, 1);
    cache.PrepareForEvaluation(true, true);
    const double* parameters[] = {xi.data()};
    double point_residual[3], plane_residual[1], jp[18], jn[6];
    double* point_jacobians[] = {jp};
    double* plane_jacobians[] = {jn};
    ASSERT_TRUE(point.Evaluate(parameters, point_residual, point_jacobians));
    ASSERT_TRUE(plane.Evaluate(parameters, plane_residual, plane_jacobians));
    EXPECT_LT((Eigen::Map<Eigen::Vector3d>(point_residual) - error->error).norm(), 1e-14);
    EXPECT_NEAR(plane_residual[0], error->normal.dot(Eigen::Map<Eigen::Vector3d>(point_residual)), 1e-14);
    const Eigen::Vector3d expected =
            common::euler_zyx_to_rotation(xi[3], xi[4], xi[5]) * error->source + Eigen::Map<Eigen::Vector3d>(xi.data()) - error->target;
    EXPECT_LT((error->error - expected).norm(), 1e-14);
    for (int col = 0; col < 6; ++col)
    {
        const double original = xi[col];
        double plus[3], minus[3], plane_plus[1], plane_minus[1];
        xi[col] = original + 1e-6;
        cache.PrepareForEvaluation(false, true);
        ASSERT_TRUE(point.Evaluate(parameters, plus, nullptr));
        ASSERT_TRUE(plane.Evaluate(parameters, plane_plus, nullptr));
        xi[col] = original - 1e-6;
        cache.PrepareForEvaluation(false, true);
        ASSERT_TRUE(point.Evaluate(parameters, minus, nullptr));
        ASSERT_TRUE(plane.Evaluate(parameters, plane_minus, nullptr));
        xi[col] = original;
        for (int row = 0; row < 3; ++row)
            EXPECT_NEAR(jp[6 * row + col], (plus[row] - minus[row]) / 2e-6, 1e-8);
        EXPECT_NEAR(jn[col], (plane_plus[0] - plane_minus[0]) / 2e-6, 1e-8);
    }
    cache.PrepareForEvaluation(false, true);
    EXPECT_FALSE(error->jacobian_ready);
    cache.PrepareForEvaluation(true, false);  // same-point residual -> Jacobian upgrade
    ASSERT_TRUE(plane.Evaluate(parameters, plane_residual, plane_jacobians));
    EXPECT_TRUE(error->jacobian_ready);
    double* skipped[] = {nullptr};
    EXPECT_TRUE(point.Evaluate(parameters, point_residual, skipped));
}

TEST(HybridResidual, CeresUsesTwoBlocksPerPairAndRefreshesCache)
{
    std::array<double, 6> xi{};
    SharedErrorEvaluation cache(xi.data());
    ceres::Problem::Options options;
    options.evaluation_callback = &cache;
    ceres::Problem problem(options);
    auto error = cache.Add({1, 2, 3}, {0, 0, 0}, Eigen::Vector3d::UnitZ());
    AddCorrespondenceResiduals(problem, error, xi.data(), nullptr, 2.0, 3.0);
    EXPECT_EQ(problem.NumResidualBlocks(), 2);
    EXPECT_EQ(problem.NumResiduals(), 4);
    EXPECT_EQ(problem.NumParameterBlocks(), 1);
    ceres::Problem::EvaluateOptions evaluate;
    evaluate.num_threads = 4;
    double cost;
    ASSERT_TRUE(problem.Evaluate(evaluate, &cost, nullptr, nullptr, nullptr));
    EXPECT_NEAR(cost, .5 * (2 * 14 + 3 * 9), 1e-12);
    xi[2] = 1;
    ASSERT_TRUE(problem.Evaluate(evaluate, &cost, nullptr, nullptr, nullptr));
    EXPECT_NEAR(error->error.z(), 4, 1e-12);
    EXPECT_NEAR(cost, .5 * (2 * 21 + 3 * 16), 1e-12);
}

common::PointCloud MakeCloud()
{
    std::mt19937 generator(42);
    std::uniform_real_distribution<double> distribution(-4, 4);
    common::PointCloud cloud;
    for (int i = 0; i < 150; ++i)
    {
        Eigen::Vector3d point(distribution(generator), distribution(generator), distribution(generator));
        cloud.push_back({point, point.normalized()});
    }
    return cloud;
}

TEST(HybridIcp, RecoversKnownTransformFromIdentityWithRobustLoss)
{
    auto source = MakeCloud();
    Eigen::Isometry3d truth = Eigen::Isometry3d::Identity();
    truth.linear() = common::euler_zyx_to_rotation(.01, -.015, .02);
    truth.translation() = Eigen::Vector3d(.04, -.03, .06);
    auto target = common::transform_point_cloud(source, truth.linear(), truth.translation());
    IcpOptions options = icp_test::TimedOptions();
    options.huber_delta = .2;
    IcpPointToPointPlane icp(options);
    const auto result = icp.do_icp(source, target);
    EXPECT_TRUE(result.converged);
    EXPECT_EQ(result.correspondences, static_cast<int>(source.size()));
    EXPECT_LT((result.transform.matrix() - truth.matrix()).norm(), 1e-8);
    EXPECT_LT(result.final_error, 1e-8);
    icp_test::ReportTiming("150 random points, Huber loss", result);
}

TEST(HybridIcp, PointTermConstrainsTangentialMotionOnOnePlane)
{
    common::PointCloud source;
    for (int x = 0; x < 8; ++x)
        for (int y = 0; y < 7; ++y)
            source.push_back({Eigen::Vector3d(x, y, 0), Eigen::Vector3d::UnitZ()});
    auto target = source;
    for (auto& point : target)
        point.point += Eigen::Vector3d(.1, -.1, .03);
    IcpPointToPointPlane icp(icp_test::TimedOptions());
    const auto result = icp.do_icp(source, target);
    EXPECT_TRUE(result.converged);
    EXPECT_LT((result.transform.translation() - Eigen::Vector3d(.1, -.1, .03)).norm(), 1e-8);
    icp_test::ReportTiming("56 points on one plane", result);
}

TEST(HybridIcp, InvalidNormalsAndNoCorrespondencesAreFailures)
{
    auto source = MakeCloud();
    auto target = source;
    for (auto& point : target)
        point.normal.setConstant(NAN);
    IcpPointToPointPlane icp;
    auto result = icp.do_icp(source, target);
    EXPECT_FALSE(result.converged);
    EXPECT_EQ(result.correspondences, 0);
    EXPECT_TRUE(std::isinf(result.final_error));
    target = source;
    for (auto& point : target)
        point.point.x() += 100;
    result = icp.do_icp(source, target);
    EXPECT_FALSE(result.converged);
    EXPECT_TRUE(std::isinf(result.final_error));
}

TEST(HybridIcp, RejectsInvalidWeightsAndEmptyClouds)
{
    IcpPointToPointPlane icp;
    EXPECT_THROW(icp.do_icp(), std::runtime_error);
    icp.mutable_options().point_weight = -1;
    EXPECT_THROW(icp.do_icp(MakeCloud(), MakeCloud()), std::invalid_argument);
    icp.mutable_options().point_weight = 1;
    icp.mutable_options().plane_weight = NAN;
    EXPECT_THROW(icp.do_icp(MakeCloud(), MakeCloud()), std::invalid_argument);
}

TEST(HybridIcp, CeresBuildControlsDefaultSolverAndRejectsUnavailableCuda)
{
    IcpPointToPointPlane icp;
    EXPECT_EQ(icp.options().use_cuda, CeresCudaSolverAvailable());
    if (CeresCudaSolverAvailable())
        return;  // Actual CUDA solves are covered by test_ceres_cuda_icp, even without custom kernels.
    EXPECT_FALSE(icp.options().use_cuda);
    icp.mutable_options().use_cuda = true;
    EXPECT_THROW(icp.do_icp(MakeCloud(), MakeCloud()), std::runtime_error);
}
