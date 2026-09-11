#include <gtest/gtest.h>
#include <Eigen/Eigenvalues>
#include <array>
#include <limits>
#include <random>
#include "p2p_icp/icp_point_to_plane.hpp"
#include "p2p_icp/inertial_cost.hpp"
#include "imu_preint/so3.hpp"

namespace
{
using namespace imu_preint;
using namespace p2p_icp;
const Eigen::Vector3d gravity(0, 0, -9.80665);

// Central differences are a test oracle only; production factors return analytic Jacobians.
void CheckFactorJacobian(ceres::CostFunction& cost, std::vector<std::array<double, 15>>& values)
{
    const int rows = cost.num_residuals();
    std::vector<const double*> parameters;
    std::vector<Eigen::Matrix<double, Eigen::Dynamic, 15, Eigen::RowMajor>> jacobians;
    for (auto& value : values)
    {
        parameters.push_back(value.data());
        jacobians.emplace_back(rows, 15);
    }
    std::vector<double*> pointers;
    for (auto& jacobian : jacobians)
    {
        jacobian.setConstant(std::numeric_limits<double>::quiet_NaN());
        pointers.push_back(jacobian.data());
    }
    Eigen::VectorXd residual(rows), residual_only(rows);
    ASSERT_TRUE(cost.Evaluate(parameters.data(), residual.data(), pointers.data()));
    ASSERT_TRUE(cost.Evaluate(parameters.data(), residual_only.data(), nullptr));
    EXPECT_TRUE(residual.isApprox(residual_only, 1e-12));
    std::vector<double*> absent(values.size(), nullptr);
    ASSERT_TRUE(cost.Evaluate(parameters.data(), residual_only.data(), absent.data()));
    for (std::size_t block = 0; block < values.size(); ++block)
    {
        ASSERT_TRUE(jacobians[block].allFinite());
        Eigen::Matrix<double, Eigen::Dynamic, 15, Eigen::RowMajor> partial(rows, 15);
        absent[block] = partial.data();
        ASSERT_TRUE(cost.Evaluate(parameters.data(), residual_only.data(), absent.data()));
        EXPECT_TRUE(partial.isApprox(jacobians[block], 1e-12));
        absent[block] = nullptr;
        for (int column = 0; column < 15; ++column)
        {
            const double saved = values[block][column], eps = 1e-6;
            Eigen::VectorXd plus(rows), minus(rows);
            values[block][column] = saved + eps;
            ASSERT_TRUE(cost.Evaluate(parameters.data(), plus.data(), nullptr));
            values[block][column] = saved - eps;
            ASSERT_TRUE(cost.Evaluate(parameters.data(), minus.data(), nullptr));
            values[block][column] = saved;
            const Eigen::VectorXd analytic = jacobians[block].col(column);
            EXPECT_LT(((plus - minus) / (2 * eps) - analytic).norm() / std::max(1.0, analytic.norm()), 2e-6) << "block " << block << " column " << column;
        }
    }
}

ImuPreintegrator Interval(const Eigen::Vector3d& bg = Eigen::Vector3d::Zero(), const Eigen::Vector3d& ba = Eigen::Vector3d::Zero())
{
    ImuPreintegrator result;
    result.reset(bg, ba);
    for (int k = 0; k < 20; ++k)
        result.integrate(Eigen::Vector3d(0.1 + k * 0.01, -0.2, 0.3), Eigen::Vector3d(0.4, -0.1, 9.8 + k * 0.01), 0.005);
    return result;
}

common::PointCloud Planes()
{
    common::PointCloud cloud;
    for (int axis = 0; axis < 3; ++axis)
        for (int u = -3; u <= 3; ++u)
            for (int v = -3; v <= 3; ++v)
            {
                Eigen::Vector3d p = Eigen::Vector3d::Zero();
                p[axis] = 4.0;
                p[(axis + 1) % 3] = 0.7 * u;
                p[(axis + 2) % 3] = 0.7 * v;
                cloud.push_back({p, Eigen::Vector3d::Unit(axis)});
            }
    return cloud;
}

TEST(PreintegrationBias, AnalyticJacobianMatchesReintegration)
{
    const Eigen::Vector3d bg(0.02, -0.01, 0.03), ba(0.1, 0.2, -0.05);
    const auto integrated = Interval(bg, ba);
    const auto delta = integrated.delta();
    const double eps = 1e-6;
    for (int column = 0; column < 6; ++column)
    {
        auto bgp = bg, bgm = bg, bap = ba, bam = ba;
        (column < 3 ? bgp[column] : bap[column - 3]) += eps;
        (column < 3 ? bgm[column] : bam[column - 3]) -= eps;
        const auto plus = Interval(bgp, bap).delta(), minus = Interval(bgm, bam).delta();
        Eigen::Matrix<double, 9, 1> numeric;
        numeric << (so3::Log(delta.rotation.transpose() * plus.rotation) - so3::Log(delta.rotation.transpose() * minus.rotation)) / (2 * eps),
                (plus.position - minus.position) / (2 * eps), (plus.velocity - minus.velocity) / (2 * eps);
        EXPECT_LT((numeric - delta.bias_jacobian.col(column)).norm(), 1e-8) << column;
    }
    auto rebuilt = integrated;
    rebuilt.reintegrate(bg + Eigen::Vector3d::Constant(0.001), ba + Eigen::Vector3d::Constant(0.002));
    const auto corrected = integrated.corrected_delta(bg + Eigen::Vector3d::Constant(0.001), ba + Eigen::Vector3d::Constant(0.002));
    EXPECT_LT((rebuilt.delta().position - corrected.position).norm(), 1e-7);
    EXPECT_EQ(rebuilt.num_samples(), integrated.num_samples());
    EXPECT_LT((rebuilt.delta().velocity - corrected.velocity).norm(), 1e-7);
    EXPECT_GT(Eigen::SelfAdjointEigenSolver<Matrix15d>(delta.covariance).eigenvalues().minCoeff(), 0.0);
    const auto w = inertial::SqrtInformation(delta.covariance);
    EXPECT_LT((w * delta.covariance * w.transpose() - Matrix15d::Identity()).norm(), 1e-8);
}

TEST(InertialFactor, ZeroResidualAndBothStateJacobians)
{
    auto integrated = Interval();
    NavState i;
    i.rotation = so3::Exp(Eigen::Vector3d(0.4, -0.2, 0.3));
    i.position = Eigen::Vector3d(4, -2, 1);
    i.velocity = Eigen::Vector3d(1, 2, -0.3);
    auto j = integrated.predict(i, gravity);
    inertial::ImuCost cost{i, j, integrated.delta(), gravity, inertial::SqrtInformation(integrated.delta().covariance)};
    std::array<double, 15> xi{}, xj{};
    const double* parameters[] = {xi.data(), xj.data()};
    Vector15d residual;
    ASSERT_TRUE(cost.Evaluate(parameters, residual.data(), nullptr));
    EXPECT_LT(residual.norm(), 1e-8);
    std::vector<std::array<double, 15>> zeros(2);
    CheckFactorJacobian(cost, zeros);
    for (int k = 0; k < 15; ++k)
    {
        xi[k] = 0.01 * std::sin(k + 1);
        xj[k] = 0.02 * std::cos(k + 1);
    }
    Eigen::Matrix<double, 15, 15, Eigen::RowMajor> ji, jj;
    double* jacobians[] = {ji.data(), jj.data()};
    ASSERT_TRUE(cost.Evaluate(parameters, residual.data(), jacobians));
    for (int block = 0; block < 2; ++block)
        for (int column = 0; column < 15; ++column)
        {
            auto& x = block == 0 ? xi : xj;
            const double saved = x[column], eps = 1e-6;
            Vector15d plus, minus;
            x[column] = saved + eps;
            cost.Evaluate(parameters, plus.data(), nullptr);
            x[column] = saved - eps;
            cost.Evaluate(parameters, minus.data(), nullptr);
            x[column] = saved;
            const Vector15d analytic = (block == 0 ? ji : jj).col(column);
            EXPECT_LT(((plus - minus) / (2 * eps) - analytic).norm() / std::max(1.0, analytic.norm()), 1e-6) << block << ":" << column;
        }
}

TEST(InertialFactor, NonzeroBiasCorrectionsAndLargeRotations)
{
    std::mt19937 random(913);
    std::uniform_real_distribution<double> uniform(-1.0, 1.0);
    const auto integrated = Interval(Eigen::Vector3d(0.03, -0.02, 0.01), Eigen::Vector3d(0.1, 0.05, -0.08));
    for (int trial = 0; trial < 20; ++trial)
    {
        SCOPED_TRACE(trial);
        NavState i;
        i.rotation = so3::Exp(Eigen::Vector3d(uniform(random), uniform(random), uniform(random)));
        i.position = Eigen::Vector3d(1.0, -3.0, 2.0);
        i.velocity = Eigen::Vector3d(2.0, 0.3, -0.5);
        auto j = integrated.predict(i, gravity);
        std::vector<std::array<double, 15>> values(2);
        for (auto& value : values)
        {
            for (double& item : value)
            {
                item = 0.7 * uniform(random);
            }
        }
        inertial::ImuCost cost{i, j, integrated.delta(), gravity, inertial::SqrtInformation(integrated.delta().covariance)};
        CheckFactorJacobian(cost, values);
    }
}

TEST(InertialFactor, RotationLogNearPi)
{
    const auto integrated = Interval();
    const NavState i;
    auto j = integrated.predict(i, gravity);
    j.rotation *= so3::Exp((M_PI - 0.001) * Eigen::Vector3d(1.0, -2.0, 3.0).normalized());
    inertial::ImuCost cost{i, j, integrated.delta(), gravity, inertial::SqrtInformation(integrated.delta().covariance)};
    std::vector<std::array<double, 15>> values(2);
    CheckFactorJacobian(cost, values);
}

TEST(AnalyticFactors, PlaneJacobianAndZeroVelocityBiasColumns)
{
    NavState base;
    base.rotation = so3::Exp(Eigen::Vector3d(0.3, -0.2, 0.5));
    base.position = Eigen::Vector3d(2.0, -5.0, 1.0);
    Eigen::Isometry3d extrinsic = Eigen::Isometry3d::Identity();
    extrinsic.linear() = so3::Exp(Eigen::Vector3d(-0.2, 0.1, 0.3));
    extrinsic.translation() = Eigen::Vector3d(0.8, -0.4, 0.5);
    const Eigen::Vector3d source = extrinsic * Eigen::Vector3d(12.0, -3.0, 5.0);
    inertial::PlaneCost cost{base, source, Eigen::Vector3d(9.0, -2.0, 6.0), Eigen::Vector3d(1.0, 2.0, -1.0).normalized(), 0.07};
    for (double scale : {0.0, 1e-10, 0.001, 0.5, 2.8})
    {
        SCOPED_TRACE(scale);
        std::vector<std::array<double, 15>> values(1);
        for (int k = 0; k < 15; ++k)
        {
            values[0][k] = scale * std::sin(k + 1);
        }
        CheckFactorJacobian(cost, values);
        const double* parameter[] = {values[0].data()};
        Eigen::Matrix<double, 1, 15> jacobian;
        double* pointers[] = {jacobian.data()};
        double residual;
        ASSERT_TRUE(cost.Evaluate(parameter, &residual, pointers));
        EXPECT_TRUE(jacobian.tail<9>().isZero(0.0));
    }
}

TEST(AnalyticFactors, DenseWhitenedPrior)
{
    inertial::PriorCost cost{inertial::SqrtInformation(Interval().delta().covariance)};
    std::vector<std::array<double, 15>> values(1);
    for (int k = 0; k < 15; ++k)
    {
        values[0][k] = 0.03 * std::cos(k + 1);
    }
    CheckFactorJacobian(cost, values);
}

TEST(AnalyticFactors, RightJacobianInverse)
{
    const Eigen::Vector3d axis = Eigen::Vector3d(0.2, -0.7, 0.5).normalized();
    for (double angle : {0.0, 1e-10, 1e-5, 0.1, 1.0, M_PI - 1e-5})
    {
        const Eigen::Vector3d phi = angle * axis;
        EXPECT_LT((so3::RightJacobian(phi) * so3::RightJacobianInverse(phi) - Eigen::Matrix3d::Identity()).norm(), 1e-12);
    }
}

TEST(TightlyCoupled, ExtrinsicPoseAndVelocity)
{
    IcpPointToPlane icp;
    icp.mutable_options().max_iterations = 8;
    icp.mutable_options().translation_tolerance = 1e-7;
    icp.mutable_options().rotation_tolerance = 1e-7;
    const auto map = Planes();
    icp.set_target(map);
    NavStatePrior prior;
    prior.state.rotation = so3::Exp(Eigen::Vector3d(0.1, -0.05, 0.1));
    prior.state.velocity = Eigen::Vector3d(1.0, 0.2, 0.1);
    const auto integrated = Interval();
    const auto truth = integrated.predict(prior.state, gravity);
    Eigen::Isometry3d extrinsic = Eigen::Isometry3d::Identity();
    extrinsic.linear() = so3::Exp(Eigen::Vector3d(0.1, 0.2, -0.1));
    extrinsic.translation() = Eigen::Vector3d(0.8, 0.3, -0.5);
    const Eigen::Isometry3d pose = truth.isometry() * extrinsic;
    common::PointCloud source;
    for (const auto& point : map)
        source.push_back({pose.inverse() * point.point, pose.rotation().transpose() * point.normal});
    icp.set_source(source);
    prior.state.velocity += Eigen::Vector3d(0.4, -0.3, 0.2);
    const auto result = icp.do_tightly_icp(prior, integrated, gravity, extrinsic);
    ASSERT_TRUE(result.usable);
    ASSERT_TRUE(result.lidar_accepted);
    EXPECT_LT((result.posterior.state.position - truth.position).norm(), 0.005);
    EXPECT_LT((result.posterior.state.velocity - truth.velocity).norm(), 0.05);
    EXPECT_LT((result.icp.transform.matrix() - pose.matrix()).norm(), 0.01);
    EXPECT_LT(result.posterior.covariance(6, 6), prior.covariance(6, 6));
}

TEST(TightlyCoupled, LearnsGyroAndAccelBiasAcrossFrames)
{
    IcpPointToPlane icp;
    icp.set_target(Planes());
    icp.set_source(Planes());
    icp.mutable_options().max_iterations = 5;
    icp.mutable_options().translation_tolerance = 1e-6;
    icp.mutable_options().rotation_tolerance = 1e-6;
    const Eigen::Vector3d bg(0.01, -0.015, 0.02), ba(0.1, -0.08, 0.06);
    NavStatePrior prior;
    // Known initial attitude and rest velocity isolate the bias convergence in this test.
    prior.covariance.topLeftCorner<9, 9>() = Eigen::Matrix<double, 9, 9>::Identity() * 1e-6;
    for (int frame = 0; frame < 80; ++frame)
    {
        ImuPreintegrator integrated;
        integrated.reset(prior.state.gyro_bias, prior.state.accel_bias);
        for (int k = 0; k < 10; ++k)
            integrated.integrate(bg, -gravity + ba, 0.01);
        auto result = icp.do_tightly_icp(prior, integrated, gravity);
        ASSERT_TRUE(result.usable) << frame;
        ASSERT_TRUE(result.lidar_accepted) << frame;
        prior = result.posterior;
    }
    EXPECT_LT((prior.state.gyro_bias - bg).norm(), 0.002);
    EXPECT_LT((prior.state.accel_bias - ba).norm(), 0.03);
    EXPECT_LT(prior.state.velocity.norm(), 0.02);
    EXPECT_LT(prior.state.position.norm(), 0.02);
}

TEST(TightlyCoupled, MissingAndRejectedLidarCarryOnlyImuInformation)
{
    IcpPointToPlane icp;
    NavStatePrior prior;
    const auto integrated = Interval();
    const auto only_imu = icp.do_tightly_icp(prior, integrated, gravity);
    ASSERT_TRUE(only_imu.usable);
    EXPECT_FALSE(only_imu.lidar_accepted);
    EXPECT_LT((only_imu.posterior.state.position - integrated.predict(prior.state, gravity).position).norm(), 1e-8);
    EXPECT_GT(only_imu.posterior.covariance(9, 9), prior.covariance(9, 9));
    EXPECT_GT(only_imu.posterior.covariance(3, 3), prior.covariance(3, 3));
    icp.set_target(Planes());
    auto moved = Planes();
    for (auto& point : moved)
        point.point += Eigen::Vector3d(0.2, 0.1, 0.1);
    icp.set_source(moved);
    TightlyCoupledOptions options;
    options.max_translation_correction = 0.001;
    const auto rejected = icp.do_tightly_icp(prior, integrated, gravity, Eigen::Isometry3d::Identity(), options);
    ASSERT_TRUE(rejected.usable);
    EXPECT_FALSE(rejected.lidar_accepted);
    EXPECT_LT((rejected.posterior.covariance - only_imu.posterior.covariance).norm(), 1e-8);
    EXPECT_LT((rejected.posterior.state.position - only_imu.posterior.state.position).norm(), 1e-8);
}

TEST(TightlyCoupled, InvalidInputsAndShortInterval)
{
    IcpPointToPlane icp;
    NavStatePrior prior;
    EXPECT_THROW(icp.do_tightly_icp(prior, ImuPreintegrator{}, gravity), std::invalid_argument);
    ImuPreintegrator single;
    single.integrate(Eigen::Vector3d::Zero(), -gravity, 0.01);
    EXPECT_TRUE(icp.do_tightly_icp(prior, single, gravity).usable);
    prior.covariance(0, 0) = -1;
    EXPECT_THROW(icp.do_tightly_icp(prior, single, gravity), std::invalid_argument);
    EXPECT_THROW(single.integrate(Eigen::Vector3d::Zero(), gravity, std::numeric_limits<double>::quiet_NaN()), std::invalid_argument);
}

TEST(TightlyCoupled, SinglePlanePreservesUnobservedTranslation)
{
    IcpPointToPlane icp;
    common::PointCloud plane;
    for (const auto& point : Planes())
        if (point.normal.z() > 0.5)
            plane.push_back(point);
    icp.set_source(plane);
    icp.set_target(plane);
    NavStatePrior prior;
    prior.state.velocity = Eigen::Vector3d(0.5, -0.3, 0.2);
    ImuPreintegrator integrated;
    for (int k = 0; k < 10; ++k)
        integrated.integrate(Eigen::Vector3d::Zero(), -gravity, 0.01);
    TightlyCoupledOptions options;
    options.use_lidar = false;
    const auto prediction = icp.do_tightly_icp(prior, integrated, gravity, Eigen::Isometry3d::Identity(), options);
    options.use_lidar = true;
    const auto result = icp.do_tightly_icp(prior, integrated, gravity, Eigen::Isometry3d::Identity(), options);
    ASSERT_TRUE(result.usable);
    ASSERT_TRUE(result.lidar_accepted);
    EXPECT_NEAR(result.posterior.state.position.x(), prediction.posterior.state.position.x(), 1e-5);
    EXPECT_NEAR(result.posterior.state.position.y(), prediction.posterior.state.position.y(), 1e-5);
    EXPECT_GT(result.posterior.covariance(3, 3), 0.95 * prediction.posterior.covariance(3, 3));
    EXPECT_LT(result.posterior.covariance(5, 5), 0.1 * prediction.posterior.covariance(5, 5));
}

TEST(PreintegrationBias, ContinuousNoiseScalesWithTimeNotSampleCount)
{
    auto integrate = [](double dt, int steps)
    {
        ImuPreintegrator result;
        for (int k = 0; k < steps; ++k)
            result.integrate(Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), dt);
        return result.delta();
    };
    const auto fast = integrate(0.005, 20), slow = integrate(0.01, 10), twice = integrate(0.01, 20);
    // Bias random-walk variance is density^2 * elapsed time, independent of sample rate.
    EXPECT_NEAR(fast.covariance(9, 9), slow.covariance(9, 9), 1e-15);
    EXPECT_NEAR(twice.covariance(12, 12), 2.0 * slow.covariance(12, 12), 1e-15);
    EXPECT_NEAR(fast.covariance(6, 6) / slow.covariance(6, 6), 1.0, 1e-4);
}
}  // namespace
