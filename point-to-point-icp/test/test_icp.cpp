// Loads the provided source_point.txt / target_point.txt, estimates the relative pose between the
// source and the target, and checks that the final error converges to 0 within numerical tolerance.
//
// The no-prediction overload uses centroid translation initialization.

#include <gtest/gtest.h>

#include <Eigen/Geometry>
#include <string>

#include "common/point_cloud.hpp"
#include "p2pt_icp/icp_point_to_point.hpp"
#include "p2pt_icp/point_to_point_cost.hpp"

namespace
{
std::string data_path(const std::string& file)
{
    return std::string(P2PT_ICP_DATA_DIR) + "/" + file;
}

constexpr double kEpsilon = 1e-6;

}  // namespace

TEST(IcpPointToPoint, RecoversGroundtruthTransformFromFiles)
{
    const common::PointCloud source_points = common::load_point_cloud(data_path("source_point.txt"));
    const common::PointCloud target_points = common::load_point_cloud(data_path("target_point.txt"));

    p2pt_icp::IcpPointToPoint icp;
    const p2pt_icp::IcpResult result = icp.do_icp(source_points, target_points);

    EXPECT_NEAR(result.transform.translation().x(), -0.2, kEpsilon);
    EXPECT_NEAR(result.transform.translation().y(), -0.2, kEpsilon);
    EXPECT_NEAR(result.transform.translation().z(), 0.0, kEpsilon);
    EXPECT_NEAR(result.final_error, 0.0, kEpsilon);
}

TEST(IcpPointToPoint, RecoversAKnownTransformWithCentroidInitialization)
{
    // Same cloud, displaced by a known pose: ICP must recover exactly that pose.
    const common::PointCloud source_points = common::load_point_cloud(data_path("source_point.txt"));

    Eigen::Isometry3d truth = Eigen::Isometry3d::Identity();
    truth.linear() = Eigen::AngleAxisd(0.05, Eigen::Vector3d::UnitZ()).toRotationMatrix();
    truth.translation() = Eigen::Vector3d(0.15, -0.1, 0.05);

    common::PointCloud target_points;
    target_points.reserve(source_points.size());
    for (const common::PointNormal& item : source_points)
    {
        common::PointNormal moved;
        moved.point = truth * item.point;
        moved.normal = truth.linear() * item.normal;
        target_points.push_back(moved);
    }

    p2pt_icp::IcpOptions options;
    options.max_correspondence_distance = 2.0;
    p2pt_icp::IcpPointToPoint icp(options);
    const p2pt_icp::IcpResult result = icp.do_icp(source_points, target_points);

    EXPECT_TRUE(result.converged);
    EXPECT_NEAR((result.transform.translation() - truth.translation()).norm(), 0.0, kEpsilon);
    EXPECT_NEAR(Eigen::AngleAxisd(result.transform.linear().transpose() * truth.linear()).angle(), 0.0, kEpsilon);
    EXPECT_NEAR(result.final_error, 0.0, kEpsilon);
}

TEST(IcpPointToPoint, ThrowsOnEmptyClouds)
{
    const common::PointCloud source_points = common::load_point_cloud(data_path("source_point.txt"));

    p2pt_icp::IcpPointToPoint icp;
    EXPECT_THROW(icp.do_icp(), std::runtime_error);

    icp.set_source(source_points);
    EXPECT_THROW(icp.do_icp(), std::runtime_error);
}

TEST(PointToPointMinimum, RecoversThreeNonCollinearPairsWithoutNormals)
{
    common::PointCloud source = {{Eigen::Vector3d(0, 0, 0), Eigen::Vector3d::Zero()},
                                 {Eigen::Vector3d(2, 0, 0), Eigen::Vector3d::Zero()},
                                 {Eigen::Vector3d(0, 3, 1), Eigen::Vector3d::Zero()}};
    auto target = source;
    const Eigen::Vector3d translation(0.1, -0.2, 0.05);
    for (auto& point : target)
        point.point += translation;
    p2pt_icp::IcpPointToPoint icp;
    const auto result = icp.do_icp(source, target);
    EXPECT_TRUE(result.converged);
    EXPECT_EQ(result.correspondences, 3);
    EXPECT_LT((result.transform.translation() - translation).norm(), 1e-8);
    EXPECT_LT(result.final_error, 1e-8);
}

TEST(PointToPointCost, AnalyticJacobianMatchesCentralDifference)
{
    p2pt_icp::PointToPointCostFunction cost({1, 2, 3}, {-.2, .3, 1});
    double xi[6] = {.1, -.2, .3, .15, -.2, .1};
    const double* parameters[] = {xi};
    double residual[3], jacobian[18];
    double* jacobians[] = {jacobian};
    ASSERT_TRUE(cost.Evaluate(parameters, residual, jacobians));
    for (int col = 0; col < 6; ++col)
    {
        const double original = xi[col];
        double plus[3], minus[3];
        xi[col] = original + 1e-6;
        cost.Evaluate(parameters, plus, nullptr);
        xi[col] = original - 1e-6;
        cost.Evaluate(parameters, minus, nullptr);
        xi[col] = original;
        for (int row = 0; row < 3; ++row)
            EXPECT_NEAR(jacobian[row * 6 + col], (plus[row] - minus[row]) / 2e-6, 1e-8);
    }
}
