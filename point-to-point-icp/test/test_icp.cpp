// Loads the provided source_point.txt / target_point.txt, estimates the relative pose between the
// source and the target, and checks that the final error converges to 0 within numerical tolerance.
//
// These fail until IcpPointToPoint::do_icp() and PointToPointCostFunction::Evaluate() are written.

#include <gtest/gtest.h>

#include <Eigen/Geometry>
#include <string>

#include "common/point_cloud.hpp"
#include "p2pt_icp/icp_point_to_point.hpp"

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

TEST(IcpPointToPoint, RecoversAKnownTransformFromAnIdentityStart)
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
