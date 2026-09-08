#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/nav_sat_fix.hpp>
#include <std_msgs/msg/header.hpp>
#include <nav_msgs/msg/path.hpp>
#include <fstream>
#include <iomanip>
#include "gps_ground_truth/trajectory.hpp"

class GpsGroundTruthNode : public rclcpp::Node
{
public:
    GpsGroundTruthNode() : Node("gps_ground_truth"), trajectory_(declare_parameter<int>("max_buffer_fixes", 100000))
    {
        frame_ = declare_parameter<std::string>("odom_frame", "odom");
        const auto mode = declare_parameter<std::string>("origin_mode", "lio_start");
        if (mode != "lio_start" && mode != "first_fix")
            throw std::invalid_argument("origin_mode must be lio_start or first_fix");
        first_fix_ = mode == "first_fix";
        const auto output = declare_parameter<std::string>("trajectory_csv", "/tmp/lio_gt.txt");
        if (!output.empty())
        {
            stream_.open(output);
            if (!stream_)
                throw std::runtime_error("cannot open GPS trajectory: " + output);
            stream_ << std::fixed << std::setprecision(9) << "# TUM: timestamp tx ty tz qx qy qz qw (GPS position; orientation is a placeholder)\n";
        }
        publisher_ = create_publisher<nav_msgs::msg::Path>("~/path", rclcpp::QoS(1).transient_local());
        start_subscriber_ = create_subscription<std_msgs::msg::Header>(
                declare_parameter<std::string>("trajectory_start_topic", "/lidar_inertial_odometer/trajectory_start"), rclcpp::QoS(1).transient_local(),
                [this](std_msgs::msg::Header::ConstSharedPtr start)
                {
                    if (!first_fix_ && !trajectory_.has_start())
                    {
                        frame_ = start->frame_id;
                        trajectory_.SetStart(rclcpp::Time(start->stamp).seconds());
                        Publish();
                    }
                });
        gps_subscriber_ = create_subscription<sensor_msgs::msg::NavSatFix>(declare_parameter<std::string>("gps_topic", "/gps/fix"),
                                                                           rclcpp::SensorDataQoS().keep_last(200),
                                                                           [this](sensor_msgs::msg::NavSatFix::ConstSharedPtr msg)
                                                                           {
                                                                               const double stamp = rclcpp::Time(msg->header.stamp).seconds();
                                                                               if (msg->status.status < sensor_msgs::msg::NavSatStatus::STATUS_FIX ||
                                                                                   !trajectory_.AddFix({stamp, msg->latitude, msg->longitude, msg->altitude}))
                                                                                   return;
                                                                               if (first_fix_)
                                                                                   trajectory_.SetStart(stamp);
                                                                               Publish();
                                                                           });
        RCLCPP_INFO(get_logger(), "GPS GT ready (origin_mode=%s)", mode.c_str());
    }

private:
    void Publish()
    {
        const auto samples = trajectory_.Drain();
        for (const auto& sample : samples)
        {
            geometry_msgs::msg::PoseStamped pose;
            pose.header.stamp = rclcpp::Time(static_cast<int64_t>(std::llround(sample.timestamp * 1e9)), RCL_ROS_TIME);
            pose.header.frame_id = frame_;
            pose.pose.position.x = sample.position.x();
            pose.pose.position.y = sample.position.y();
            pose.pose.position.z = sample.position.z();
            pose.pose.orientation.w = 1.0;
            path_.header = pose.header;
            path_.poses.push_back(pose);
            if (stream_)
                stream_ << sample.timestamp << ' ' << sample.position.x() << ' ' << sample.position.y() << ' ' << sample.position.z() << " 0 0 0 1\n";
        }
        if (!samples.empty())
        {
            publisher_->publish(path_);
            if (stream_)
                stream_.flush();
        }
    }
    gps_ground_truth::Trajectory trajectory_;
    bool first_fix_ = false;
    std::string frame_;
    std::ofstream stream_;
    nav_msgs::msg::Path path_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr publisher_;
    rclcpp::Subscription<sensor_msgs::msg::NavSatFix>::SharedPtr gps_subscriber_;
    rclcpp::Subscription<std_msgs::msg::Header>::SharedPtr start_subscriber_;
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<GpsGroundTruthNode>());
    rclcpp::shutdown();
    return 0;
}
