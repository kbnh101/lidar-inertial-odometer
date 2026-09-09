#include <gtest/gtest.h>
#include <ceres/ceres.h>
#include <array>
#include <random>
#include "p2ptpl_icp/cost_functions.hpp"
#include "p2ptpl_icp/cuda_error.hpp"
#include "p2ptpl_icp/icp_point_to_point_plane.hpp"

using namespace p2ptpl_icp;

TEST(CudaResidual, MatchesCpuAcrossPosesBatchBoundariesAndBufferReuse)
{
    ASSERT_TRUE(CudaEvaluationCompiled());
    CudaErrorEvaluation gpu(2.3, .7);
    std::mt19937 random(73);
    std::uniform_real_distribution<double> value(-3, 3);
    for (int count : {0, 1, 127, 128, 129, 513, 3})
    {
        std::array<double, 6> xi{};
        for (double& x : xi)
            x = value(random);
        gpu.Reset(xi.data());
        SharedErrorEvaluation cpu(xi.data());
        std::vector<std::shared_ptr<const SharedError>> pairs;
        for (int i = 0; i < count; ++i)
        {
            Eigen::Vector3d source(value(random), value(random), value(random));
            Eigen::Vector3d target(value(random), value(random), value(random));
            Eigen::Vector3d normal(value(random), value(random), value(random));
            normal.normalize();
            pairs.push_back(cpu.Add(source, target, normal));
            ASSERT_EQ(gpu.Add(source, target, normal), static_cast<std::size_t>(i));
        }
        // Repeated same-point evaluation, residual -> Jacobian upgrade, and new
        // residual-only points exercise the callback's cache invalidation contract.
        for (const auto request : {std::array<bool, 2>{false, true}, {true, false}, {true, false}, {false, true}, {true, false}})
        {
            if (request[1])
                xi[4] += .07;
            gpu.PrepareForEvaluation(request[0], request[1]);
            cpu.PrepareForEvaluation(request[0], request[1]);
            ASSERT_TRUE(gpu.last_error().empty()) << gpu.last_error();
            for (int i = 0; i < count; ++i)
            {
                double r_cpu[4], r_gpu[4], j_cpu[24], j_gpu[24];
                double* jp[] = {request[0] ? j_cpu : nullptr};
                double* jn[] = {request[0] ? j_cpu + 18 : nullptr};
                ASSERT_TRUE(PointCostFunction(pairs[i], 2.3).Evaluate(nullptr, r_cpu, jp));
                ASSERT_TRUE(PlaneCostFunction(pairs[i], .7).Evaluate(nullptr, r_cpu + 3, jn));
                ASSERT_TRUE(gpu.Copy(i, false, r_gpu, request[0] ? j_gpu : nullptr));
                ASSERT_TRUE(gpu.Copy(i, true, r_gpu + 3, request[0] ? j_gpu + 18 : nullptr));
                for (int k = 0; k < 4; ++k)
                    EXPECT_NEAR(r_cpu[k], r_gpu[k], 5e-13) << "count=" << count << " pair=" << i;
                if (request[0])
                    for (int k = 0; k < 24; ++k)
                        EXPECT_NEAR(j_cpu[k], j_gpu[k], 5e-13) << "count=" << count << " pair=" << i << " entry=" << k;
                else
                    EXPECT_FALSE(gpu.Copy(i, false, r_gpu, j_gpu));
            }
        }
    }
}

TEST(CudaResidual, MatchesCentralDifferencesAndRejectsStaleCache)
{
    std::array<double, 6> xi{.2, -.1, .3, .13, -.21, .34};
    CudaErrorEvaluation gpu(1.7, .4);
    gpu.Reset(xi.data());
    gpu.Add({1, 2, 3}, {-.4, .2, 1}, Eigen::Vector3d(1, 2, -1).normalized());
    CudaCostFunction<false> point(gpu, 0);
    CudaCostFunction<true> plane(gpu, 0);
    double r[4], jacobian[24];
    double* jp[] = {jacobian};
    double* jn[] = {jacobian + 18};
    EXPECT_FALSE(point.Evaluate(nullptr, r, nullptr));
    gpu.PrepareForEvaluation(true, true);
    ASSERT_TRUE(point.Evaluate(nullptr, r, jp));
    ASSERT_TRUE(plane.Evaluate(nullptr, r + 3, jn));
    double* skipped[] = {nullptr};
    EXPECT_TRUE(point.Evaluate(nullptr, r, skipped));
    EXPECT_TRUE(plane.Evaluate(nullptr, r + 3, skipped));
    for (int col = 0; col < 6; ++col)
    {
        const double original = xi[col];
        double plus[4], minus[4];
        xi[col] = original + 1e-6;
        gpu.PrepareForEvaluation(false, true);
        ASSERT_TRUE(gpu.Copy(0, false, plus, nullptr));
        ASSERT_TRUE(gpu.Copy(0, true, plus + 3, nullptr));
        xi[col] = original - 1e-6;
        gpu.PrepareForEvaluation(false, true);
        ASSERT_TRUE(gpu.Copy(0, false, minus, nullptr));
        ASSERT_TRUE(gpu.Copy(0, true, minus + 3, nullptr));
        xi[col] = original;
        for (int row = 0; row < 4; ++row)
            EXPECT_NEAR(jacobian[6 * row + col], (plus[row] - minus[row]) / 2e-6, 1e-8);
    }
    gpu.Reset(xi.data());
    EXPECT_FALSE(point.Evaluate(nullptr, r, nullptr));
    gpu.Add({4, 5, 6}, {0, 0, 0}, Eigen::Vector3d::UnitX());
    gpu.PrepareForEvaluation(true, false);
    ASSERT_TRUE(point.Evaluate(nullptr, r, jp));
    gpu.Add({7, 8, 9}, {0, 0, 0}, Eigen::Vector3d::UnitY());
    EXPECT_FALSE(point.Evaluate(nullptr, r, nullptr));
    gpu.PrepareForEvaluation(true, false);
    ASSERT_TRUE(gpu.Copy(1, false, r, jacobian));
}

TEST(CudaResidual, CeresPreservesSeparateRobustLossesAndParallelEvaluation)
{
    std::array<double, 6> xi{.2, -.1, .3, .13, -.21, .34};
    SharedErrorEvaluation cpu(xi.data());
    CudaErrorEvaluation gpu(2.0, 3.0);
    gpu.Reset(xi.data());
    ceres::Problem::Options cpu_options, gpu_options;
    cpu_options.evaluation_callback = &cpu;
    gpu_options.evaluation_callback = &gpu;
    ceres::Problem cpu_problem(cpu_options), gpu_problem(gpu_options);
    auto* cpu_loss = new ceres::HuberLoss(.2);
    auto* gpu_loss = new ceres::HuberLoss(.2);
    for (int i = 0; i < 259; ++i)
    {
        Eigen::Vector3d source(.03 * i, 2, -1), target(0, .02 * i, .4);
        const auto normal = Eigen::Vector3d(1, 2, -1).normalized().eval();
        AddCorrespondenceResiduals(cpu_problem, cpu.Add(source, target, normal), xi.data(), cpu_loss, 2.0, 3.0);
        AddCudaCorrespondenceResiduals(gpu_problem, gpu, gpu.Add(source, target, normal), xi.data(), gpu_loss);
    }
    EXPECT_EQ(gpu_problem.NumResidualBlocks(), 2 * 259);
    EXPECT_EQ(gpu_problem.NumResiduals(), 4 * 259);
    EXPECT_EQ(gpu_problem.NumParameterBlocks(), 1);
    for (bool robust : {false, true})
    {
        ceres::Problem::EvaluateOptions options;
        options.num_threads = 4;
        options.apply_loss_function = robust;
        xi[0] += .1;
        double cpu_cost, gpu_cost;
        std::vector<double> cpu_r, gpu_r, cpu_g, gpu_g;
        ceres::CRSMatrix cpu_j, gpu_j;
        ASSERT_TRUE(cpu_problem.Evaluate(options, &cpu_cost, &cpu_r, &cpu_g, &cpu_j));
        ASSERT_TRUE(gpu_problem.Evaluate(options, &gpu_cost, &gpu_r, &gpu_g, &gpu_j));
        EXPECT_NEAR(cpu_cost, gpu_cost, 1e-9);
        ASSERT_EQ(cpu_r.size(), gpu_r.size());
        ASSERT_EQ(cpu_g.size(), gpu_g.size());
        ASSERT_EQ(cpu_j.values.size(), gpu_j.values.size());
        EXPECT_EQ(cpu_j.rows, gpu_j.rows);
        EXPECT_EQ(cpu_j.cols, gpu_j.cols);
        for (std::size_t i = 0; i < cpu_r.size(); ++i)
            EXPECT_NEAR(cpu_r[i], gpu_r[i], 1e-12);
        for (std::size_t i = 0; i < cpu_g.size(); ++i)
            EXPECT_NEAR(cpu_g[i], gpu_g[i], 1e-9);
        for (std::size_t i = 0; i < cpu_j.values.size(); ++i)
            EXPECT_NEAR(cpu_j.values[i], gpu_j.values[i], 1e-12);
    }
}

TEST(CudaIcp, MatchesCpuRegistrationWithNoiseAndOutliers)
{
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
    IcpOptions options;
    options.huber_delta = .03;
    options.point_weight = 2.0;
    options.plane_weight = .7;
    options.use_cuda = false;
    const auto cpu = IcpPointToPointPlane(options).do_icp(source, target);
    options.use_cuda = true;
    const auto gpu = IcpPointToPointPlane(options).do_icp(source, target);
    ASSERT_TRUE(cpu.converged);
    ASSERT_TRUE(gpu.converged);
    EXPECT_EQ(cpu.correspondences, gpu.correspondences);
    EXPECT_LT((cpu.transform.matrix() - gpu.transform.matrix()).norm(), 1e-7);
    EXPECT_NEAR(cpu.final_error, gpu.final_error, 1e-9);
    EXPECT_LT((gpu.transform.translation() - truth.translation()).norm(), .005);
}

TEST(CudaResidual, ReportsInitializationMisuseAndRejectsInvalidWeights)
{
    EXPECT_THROW(CudaErrorEvaluation(-1, 1), std::invalid_argument);
    EXPECT_THROW(CudaErrorEvaluation(1, NAN), std::invalid_argument);
    CudaErrorEvaluation gpu(1, 1);
    gpu.Add({1, 2, 3}, {0, 0, 0}, Eigen::Vector3d::UnitZ());
    gpu.PrepareForEvaluation(true, true);
    EXPECT_FALSE(gpu.last_error().empty());
    double r[3];
    EXPECT_FALSE(gpu.Copy(0, false, r, nullptr));
    std::array<double, 6> xi{};
    gpu.Reset(xi.data());
    gpu.Add({1, 2, 3}, {0, 0, 0}, Eigen::Vector3d::UnitZ());
    gpu.PrepareForEvaluation(true, true);
    EXPECT_TRUE(gpu.last_error().empty());
    EXPECT_TRUE(gpu.Copy(0, false, r, nullptr));
}
