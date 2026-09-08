// Loads the provided source_point.txt / target_point.txt, estimates the relative pose between the
// source and the target, and checks that the final error converges to 0 within numerical tolerance.

#include <gtest/gtest.h>

#include <string>

#include "p2p_icp/icp_point_to_plane.hpp"
#include "common/point_cloud.hpp"

namespace
{
std::string data_path(const std::string& file)
{
    return std::string(P2P_ICP_DATA_DIR) + "/" + file;
}

}  // namespace

TEST(IcpPointToPlane, RecoversGroundtruthTransformFromFiles)
{
    const common::PointCloud source_points = common::load_point_cloud(data_path("source_point.txt"));
    const common::PointCloud target_points = common::load_point_cloud(data_path("target_point.txt"));

    p2p_icp::IcpPointToPlane icp;
    const p2p_icp::IcpResult result = icp.do_icp(source_points, target_points);

    constexpr double kEpsilon = 1e-6;
    EXPECT_NEAR(result.transform.translation().x(), -0.2, kEpsilon);
    EXPECT_NEAR(result.transform.translation().y(), -0.2, kEpsilon);
    EXPECT_NEAR(result.transform.translation().z(), 0.0, kEpsilon);
    EXPECT_NEAR(result.final_error, 0.0, kEpsilon);
}

TEST(PointToPlaneInput, NormalizesNormalsAndRejectsMissingTargetPlanes)
{
    common::PointCloud cloud;
    for (int i = 0; i < 12; ++i)
        cloud.push_back({Eigen::Vector3d(i, i % 3, i % 2), Eigen::Vector3d(0, 0, 7)});
    p2p_icp::IcpPointToPlane icp;
    icp.set_source(cloud);
    icp.set_target(cloud);
    EXPECT_NEAR(icp.target()[0].normal.norm(), 1.0, 1e-12);
    EXPECT_NEAR(icp.source()[0].normal.norm(), 1.0, 1e-12);
    EXPECT_TRUE(icp.do_icp().converged);
    for (auto& point : cloud)
        point.normal.setConstant(std::numeric_limits<double>::quiet_NaN());
    icp.set_target(cloud);
    const auto result = icp.do_icp();
    EXPECT_EQ(result.correspondences, 0);
    EXPECT_FALSE(result.converged);
    EXPECT_TRUE(std::isinf(result.final_error));
}
