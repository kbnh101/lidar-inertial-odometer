// Loads the provided source_point.txt / target_point.txt, estimates the relative pose between the
// source and the target, and checks that the final error converges to 0 within numerical tolerance.

#include <gtest/gtest.h>

#include <string>

#include "p2p_icp/icp_point_to_plane.hpp"
#include "p2p_icp/point_cloud.hpp"

namespace
{
std::string data_path(const std::string& file)
{
    return std::string(P2P_ICP_DATA_DIR) + "/" + file;
}

}  // namespace

TEST(IcpPointToPlane, RecoversGroundtruthTransformFromFiles)
{
    const p2p_icp::PointCloud source_points = p2p_icp::load_point_cloud(data_path("source_point.txt"));
    const p2p_icp::PointCloud target_points = p2p_icp::load_point_cloud(data_path("target_point.txt"));

    p2p_icp::IcpPointToPlane icp;
    const p2p_icp::IcpResult result = icp.do_icp(source_points, target_points);

    constexpr double kEpsilon = 1e-6;
    EXPECT_NEAR(result.transform.translation().x(), -0.2, kEpsilon);
    EXPECT_NEAR(result.transform.translation().y(), -0.2, kEpsilon);
    EXPECT_NEAR(result.transform.translation().z(), 0.0, kEpsilon);
    EXPECT_NEAR(result.final_error, 0.0, kEpsilon);
}
