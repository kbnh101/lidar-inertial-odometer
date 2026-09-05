// Checks the geometry the ICP variants and the odometry share: the cloud loader, the two kd-trees
// and the Euler angle helpers. Each is verified against an independent reference -- brute force for
// the trees, central differences for the rotation derivatives.

#include <gtest/gtest.h>

#include <Eigen/Geometry>
#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include "common/kdtree.hpp"
#include "common/kdtree_radius.hpp"
#include "common/point_cloud.hpp"
#include "common/rotation.hpp"

namespace
{
std::string data_path(const std::string& file)
{
    return std::string(COMMON_DATA_DIR) + "/" + file;
}

/// Deterministic point set, so a failure is always reproducible.
std::vector<Eigen::Vector3d> MakeRandomPoints(int count, unsigned seed)
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> coordinate(-10.0, 10.0);
    std::vector<Eigen::Vector3d> points;
    points.reserve(count);
    for (int i = 0; i < count; ++i)
    {
        points.push_back(Eigen::Vector3d(coordinate(rng), coordinate(rng), coordinate(rng)));
    }
    return points;
}

/// The answer the kd-tree has to reproduce.
int BruteForceNearest(const std::vector<Eigen::Vector3d>& points, const Eigen::Vector3d& query)
{
    int best = -1;
    double best_squared = std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < points.size(); ++i)
    {
        const double squared = (points[i] - query).squaredNorm();
        if (squared < best_squared)
        {
            best_squared = squared;
            best = static_cast<int>(i);
        }
    }
    return best;
}

/// The answer the radius tree has to reproduce.
std::vector<int> BruteForceRadius(const std::vector<Eigen::Vector3d>& points, const Eigen::Vector3d& query, double radius)
{
    std::vector<int> found;
    for (std::size_t i = 0; i < points.size(); ++i)
    {
        if ((points[i] - query).squaredNorm() <= radius * radius)
        {
            found.push_back(static_cast<int>(i));
        }
    }
    return found;
}

// ---------------------------------------------------------------------
// kd-tree
// ---------------------------------------------------------------------
TEST(KdTree3d, MatchesBruteForceOnRandomQueries)
{
    const std::vector<Eigen::Vector3d> points = MakeRandomPoints(500, 12345u);
    common::KdTree3d tree;
    tree.build(points);
    ASSERT_EQ(tree.size(), points.size());

    const std::vector<Eigen::Vector3d> queries = MakeRandomPoints(200, 999u);
    for (const Eigen::Vector3d& query : queries)
    {
        int index = -1;
        double squared_distance = 0.0;
        ASSERT_TRUE(tree.nearest(query, &index, &squared_distance));

        const int expected = BruteForceNearest(points, query);
        // Compare the distance, not the index: ties would be a false failure.
        EXPECT_NEAR(squared_distance, (points[expected] - query).squaredNorm(), 1e-12);
    }
}

TEST(KdTree3d, FindsAStoredPointExactly)
{
    const std::vector<Eigen::Vector3d> points = MakeRandomPoints(64, 7u);
    common::KdTree3d tree;
    tree.build(points);

    for (std::size_t i = 0; i < points.size(); ++i)
    {
        int index = -1;
        double squared_distance = -1.0;
        ASSERT_TRUE(tree.nearest(points[i], &index, &squared_distance));
        EXPECT_DOUBLE_EQ(squared_distance, 0.0);
        EXPECT_LT((points[index] - points[i]).norm(), 1e-15);
    }
}

TEST(KdTree3d, EmptyTreeReportsNoNeighbour)
{
    common::KdTree3d tree;
    EXPECT_TRUE(tree.empty());

    int index = -1;
    double squared_distance = 0.0;
    EXPECT_FALSE(tree.nearest(Eigen::Vector3d::Zero(), &index, &squared_distance));

    tree.build(std::vector<Eigen::Vector3d>{});
    EXPECT_TRUE(tree.empty());
    EXPECT_FALSE(tree.nearest(Eigen::Vector3d::Zero(), nullptr, nullptr));
}

TEST(KdTree3d, BuildsFromACloudWithNormals)
{
    const common::PointCloud cloud = common::load_point_cloud(data_path("source_point.txt"));
    common::KdTree3d tree;
    tree.build(cloud);

    ASSERT_EQ(tree.size(), cloud.size());
    int index = -1;
    double squared_distance = -1.0;
    ASSERT_TRUE(tree.nearest(cloud.front().point, &index, &squared_distance));
    EXPECT_DOUBLE_EQ(squared_distance, 0.0);
}

// ---------------------------------------------------------------------
// kd-tree, radius search
// ---------------------------------------------------------------------
TEST(KdTreeRadius, MatchesBruteForceOnRandomQueries)
{
    const std::vector<Eigen::Vector3d> points = MakeRandomPoints(500, 2024u);
    common::KdTreeRadius tree;
    tree.build(points);
    ASSERT_EQ(tree.size(), points.size());

    const std::vector<Eigen::Vector3d> queries = MakeRandomPoints(100, 4048u);
    for (const Eigen::Vector3d& query : queries)
    {
        for (const double radius : {1.0, 3.0, 8.0})
        {
            std::vector<int> found;
            const int count = tree.radius_search(query, radius * radius, 0, &found);

            std::vector<int> expected = BruteForceRadius(points, query, radius);
            ASSERT_EQ(static_cast<std::size_t>(count), expected.size());
            ASSERT_EQ(found.size(), expected.size());

            // The search does not order its output, so compare as sets.
            std::sort(found.begin(), found.end());
            EXPECT_EQ(found, expected);
        }
    }
}

TEST(KdTreeRadius, MaxCountCapsTheResultAndKeepsNearPoints)
{
    const std::vector<Eigen::Vector3d> points = MakeRandomPoints(400, 77u);
    common::KdTreeRadius tree;
    tree.build(points);

    const Eigen::Vector3d query(0.0, 0.0, 0.0);
    const double radius = 8.0;
    const std::size_t uncapped = BruteForceRadius(points, query, radius).size();
    ASSERT_GT(uncapped, 12u);

    std::vector<int> found;
    const int count = tree.radius_search(query, radius * radius, 12, &found);
    EXPECT_EQ(count, 12);
    EXPECT_EQ(found.size(), 12u);

    // Every returned point must still lie inside the radius.
    for (const int index : found)
    {
        EXPECT_LE((points[index] - query).norm(), radius + 1e-12);
    }
}

TEST(KdTreeRadius, ReturnsNothingWhenTheRadiusExcludesEverything)
{
    const std::vector<Eigen::Vector3d> points = MakeRandomPoints(100, 5u);
    common::KdTreeRadius tree;
    tree.build(points);

    std::vector<int> found;
    // A query far outside the point cloud's extent, with a tiny radius.
    EXPECT_EQ(tree.radius_search(Eigen::Vector3d(1000.0, 1000.0, 1000.0), 1e-6, 0, &found), 0);
    EXPECT_TRUE(found.empty());

    // A non-positive radius is rejected outright.
    EXPECT_EQ(tree.radius_search(points.front(), 0.0, 0, &found), 0);
    EXPECT_TRUE(found.empty());
}

TEST(KdTreeRadius, EmptyTreeFindsNothingAndClearsCleanly)
{
    common::KdTreeRadius tree;
    EXPECT_TRUE(tree.empty());

    std::vector<int> found;
    EXPECT_EQ(tree.radius_search(Eigen::Vector3d::Zero(), 1.0, 0, &found), 0);

    tree.build(MakeRandomPoints(32, 1u));
    EXPECT_FALSE(tree.empty());
    EXPECT_EQ(tree.points().size(), 32u);

    tree.clear();
    EXPECT_TRUE(tree.empty());
    EXPECT_EQ(tree.radius_search(Eigen::Vector3d::Zero(), 1.0, 0, &found), 0);
}

// ---------------------------------------------------------------------
// point cloud
// ---------------------------------------------------------------------
TEST(PointCloud, LoadsTheSampleCloudsAndNormalizesNormals)
{
    const common::PointCloud source = common::load_point_cloud(data_path("source_point.txt"));
    const common::PointCloud target = common::load_point_cloud(data_path("target_point.txt"));

    EXPECT_FALSE(source.empty());
    EXPECT_EQ(source.size(), target.size());
    for (const common::PointNormal& item : source)
    {
        // A normal is either a unit vector or exactly zero; nothing in between.
        const double norm = item.normal.norm();
        EXPECT_TRUE(norm < 1e-12 || std::abs(norm - 1.0) < 1e-12);
    }
}

TEST(PointCloud, ThrowsOnAMissingFile)
{
    EXPECT_THROW(common::load_point_cloud(data_path("no_such_cloud.txt")), std::runtime_error);
}

TEST(PointCloud, TransformMovesPointsAndOnlyRotatesNormals)
{
    common::PointCloud cloud(1);
    cloud[0].point = Eigen::Vector3d(1.0, 2.0, 3.0);
    cloud[0].normal = Eigen::Vector3d::UnitX();

    const Eigen::Matrix3d rotation = Eigen::AngleAxisd(M_PI / 2.0, Eigen::Vector3d::UnitZ()).toRotationMatrix();
    const Eigen::Vector3d translation(10.0, -5.0, 0.5);
    const common::PointCloud moved = common::transform_point_cloud(cloud, rotation, translation);

    ASSERT_EQ(moved.size(), 1u);
    EXPECT_NEAR((moved[0].point - (rotation * cloud[0].point + translation)).norm(), 0.0, 1e-12);
    // The normal must pick up the rotation but not the translation.
    EXPECT_NEAR((moved[0].normal - rotation * cloud[0].normal).norm(), 0.0, 1e-12);
    EXPECT_NEAR(moved[0].normal.norm(), 1.0, 1e-12);
}

// ---------------------------------------------------------------------
// rotation
// ---------------------------------------------------------------------
TEST(Rotation, EulerZyxIsAValidRotationMatchingItsFactors)
{
    const double alpha = 0.31, beta = -0.42, gamma = 0.77;
    const Eigen::Matrix3d R = common::euler_zyx_to_rotation(alpha, beta, gamma);

    EXPECT_LT((R.transpose() * R - Eigen::Matrix3d::Identity()).norm(), 1e-12);
    EXPECT_NEAR(R.determinant(), 1.0, 1e-12);

    // The convention is R = Rz(gamma) Ry(beta) Rx(alpha).
    const Eigen::Matrix3d expected = Eigen::AngleAxisd(gamma, Eigen::Vector3d::UnitZ()).toRotationMatrix() *
                                     Eigen::AngleAxisd(beta, Eigen::Vector3d::UnitY()).toRotationMatrix() *
                                     Eigen::AngleAxisd(alpha, Eigen::Vector3d::UnitX()).toRotationMatrix();
    EXPECT_LT((R - expected).norm(), 1e-12);

    const Eigen::Vector3d euler = common::rotation_to_euler_zyx(R);
    EXPECT_NEAR(euler.x(), alpha, 1e-12);
    EXPECT_NEAR(euler.y(), beta, 1e-12);
    EXPECT_NEAR(euler.z(), gamma, 1e-12);
}

TEST(Rotation, VectorOverloadAgreesWithTheScalarOne)
{
    const Eigen::Vector3d euler(0.1, -0.2, 0.3);
    EXPECT_LT((common::euler_zyx_to_rotation(euler) - common::euler_zyx_to_rotation(euler.x(), euler.y(), euler.z())).norm(), 1e-15);
}

TEST(Rotation, DerivativesMatchCentralDifferences)
{
    // The analytic derivatives are what both ICP Jacobians are built from, so they are checked
    // against a numerical derivative of euler_zyx_to_rotation() itself.
    const double alpha = 0.23, beta = -0.51, gamma = 0.94;
    const double h = 1e-6;

    const Eigen::Matrix3d numeric_alpha =
            (common::euler_zyx_to_rotation(alpha + h, beta, gamma) - common::euler_zyx_to_rotation(alpha - h, beta, gamma)) / (2.0 * h);
    const Eigen::Matrix3d numeric_beta =
            (common::euler_zyx_to_rotation(alpha, beta + h, gamma) - common::euler_zyx_to_rotation(alpha, beta - h, gamma)) / (2.0 * h);
    const Eigen::Matrix3d numeric_gamma =
            (common::euler_zyx_to_rotation(alpha, beta, gamma + h) - common::euler_zyx_to_rotation(alpha, beta, gamma - h)) / (2.0 * h);

    EXPECT_LT((common::rotation_derivative_alpha(alpha, beta, gamma) - numeric_alpha).norm(), 1e-8);
    EXPECT_LT((common::rotation_derivative_beta(alpha, beta, gamma) - numeric_beta).norm(), 1e-8);
    EXPECT_LT((common::rotation_derivative_gamma(alpha, beta, gamma) - numeric_gamma).norm(), 1e-8);
}

TEST(Rotation, RoundTripsThroughEulerAwayFromGimbalLock)
{
    std::mt19937 rng(4242u);
    std::uniform_real_distribution<double> angle(-1.0, 1.0);
    for (int i = 0; i < 50; ++i)
    {
        const Eigen::Vector3d euler(angle(rng), angle(rng), angle(rng));
        const Eigen::Matrix3d R = common::euler_zyx_to_rotation(euler);
        EXPECT_LT((common::rotation_to_euler_zyx(R) - euler).norm(), 1e-10);
    }
}

}  // namespace
