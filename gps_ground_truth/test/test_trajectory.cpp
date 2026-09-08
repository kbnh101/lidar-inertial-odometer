#include <gtest/gtest.h>
#include "gps_ground_truth/trajectory.hpp"
using gps_ground_truth::Trajectory;

TEST(GpsTrajectory, WaitsForStartAndInterpolatesOrigin)
{
    Trajectory trajectory;
    ASSERT_TRUE(trajectory.AddFix({10, 49, 8, 100}));
    EXPECT_TRUE(trajectory.Drain().empty());
    trajectory.SetStart(11);
    EXPECT_TRUE(trajectory.Drain().empty());
    ASSERT_TRUE(trajectory.AddFix({12, 49, 8.00002, 102}));
    auto samples = trajectory.Drain();
    ASSERT_EQ(samples.size(), 1u);
    EXPECT_DOUBLE_EQ(samples[0].timestamp, 12);
    EXPECT_NEAR(samples[0].position.x(), 0.7303216, 1e-5);
    EXPECT_NEAR(samples[0].position.y(), 0, 1e-9);
    EXPECT_NEAR(samples[0].position.z(), 1, 1e-9);
    EXPECT_TRUE(trajectory.Drain().empty());
    trajectory.SetStart(12);  // repeated start must not reset the origin
    ASSERT_TRUE(trajectory.AddFix({13, 49, 8.00003, 103}));
    EXPECT_NEAR(trajectory.Drain()[0].position.z(), 2, 1e-9);
}
TEST(GpsTrajectory, FirstFixAndLateStartDelivery)
{
    Trajectory trajectory;
    trajectory.AddFix({10, 49, 8, 100});
    trajectory.AddFix({11, 49.00001, 8, 101});
    trajectory.SetStart(10);
    const auto samples = trajectory.Drain();
    ASSERT_EQ(samples.size(), 2u);
    EXPECT_LT(samples[0].position.norm(), 1e-9);
    EXPECT_GT(samples[1].position.y(), 1.0);
}
TEST(GpsTrajectory, RejectsInvalidAndOutOfOrderFixes)
{
    Trajectory trajectory;
    EXPECT_FALSE(trajectory.AddFix({10, 90, 8, 100}));
    EXPECT_FALSE(trajectory.AddFix({10, 49, 181, 100}));
    EXPECT_FALSE(trajectory.AddFix({10, 49, 8, NAN}));
    EXPECT_TRUE(trajectory.AddFix({10, 49, 8, 100}));
    EXPECT_FALSE(trajectory.AddFix({10, 49, 8, 100}));
    EXPECT_FALSE(trajectory.AddFix({9, 49, 8, 100}));
}
TEST(GpsTrajectory, BufferOverflowCannotSilentlyShiftOrigin)
{
    Trajectory trajectory(2);
    trajectory.AddFix({10, 49, 8, 100});
    trajectory.AddFix({11, 49, 8, 100});
    trajectory.AddFix({12, 49, 8, 100});
    trajectory.SetStart(10);
    EXPECT_THROW(trajectory.Drain(), std::runtime_error);
}
