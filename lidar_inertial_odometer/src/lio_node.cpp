// ROS1 (Noetic) interface node.
//
// A thin shell that uses ROS purely as the rosbag I/O interface. It is the only file including a ROS
// header; the algorithms (preintegration, features/normals, ICP, local map) all live in the lio_core
// library, which has zero ROS dependency.
//
// This file does four things only:
//   1. convert ROS messages into ImuSample / RawLidarPoint inside the callbacks,
//   2. call into LioOdometer,
//   3. publish the results as Odometry / Path / PointCloud2 (submap, features, scan),
//   4. broadcast the TFs (odom -> base, base -> lidar).
//
// The topic names match the provided bag (2011_09_30_drive_0028.bag, LIO-SAM layout); see
// config/kitti.yaml.

#include <ros/ros.h>
#include <tf2_ros/static_transform_broadcaster.h>
#include <tf2_ros/transform_broadcaster.h>

#include <Eigen/Geometry>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <memory>
#include <string>
#include <vector>

#include "geometry_msgs/TransformStamped.h"
#include "lidar_inertial_odometer/lio_odometer.hpp"
#include "nav_msgs/Odometry.h"
#include "nav_msgs/Path.h"
#include "sensor_msgs/Imu.h"
#include "sensor_msgs/NavSatFix.h"
#include "sensor_msgs/PointCloud2.h"
#include "sensor_msgs/point_cloud2_iterator.h"

namespace
{
const double kEarthRadius = 6378137.0;  ///< the same value as the KITTI devkit

struct GpsFix
{
    double timestamp = 0.0;
    double latitude = 0.0;
    double longitude = 0.0;
    double altitude = 0.0;
};

/**
 * @brief The KITTI devkit's latlonToMercator, an ENU approximation around the reference fix
 *
 * @param latitude  latitude [deg]
 * @param longitude longitude [deg]
 * @param altitude  altitude [m]
 * @param reference_latitude latitude the scale is taken at [deg]
 * @return local ENU coordinates [m]
 */
Eigen::Vector3d LatLonToLocal(double latitude, double longitude, double altitude, double reference_latitude)
{
    const double scale = std::cos(reference_latitude * M_PI / 180.0);
    const double x = scale * longitude * M_PI * kEarthRadius / 180.0;
    const double y = scale * kEarthRadius * std::log(std::tan((90.0 + latitude) * M_PI / 360.0));
    return Eigen::Vector3d(x, y, altitude);
}

/**
 * @brief Assembles an Isometry3d from the rosparam R (9 row-major entries) and t (3 entries)
 *
 * @param rotation_row_major the 9 rotation matrix entries, row major
 * @param translation the 3 translation entries
 * @return the assembled rigid transform
 */
Eigen::Isometry3d MakeIsometry(const std::vector<double>& rotation_row_major, const std::vector<double>& translation)
{
    Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
    if (rotation_row_major.size() == 9)
    {
        Eigen::Matrix3d rotation;
        for (int row = 0; row < 3; ++row)
        {
            for (int column = 0; column < 3; ++column)
            {
                rotation(row, column) = rotation_row_major[3 * row + column];
            }
        }
        // The calib file values are slightly off orthogonal, so renormalize them.
        pose.linear() = Eigen::Quaterniond(rotation).normalized().toRotationMatrix();
    }
    if (translation.size() == 3)
    {
        pose.translation() = Eigen::Vector3d(translation[0], translation[1], translation[2]);
    }
    return pose;
}

geometry_msgs::Quaternion ToMsg(const Eigen::Matrix3d& rotation)
{
    const Eigen::Quaterniond q(rotation);
    geometry_msgs::Quaternion msg;
    msg.x = q.x();
    msg.y = q.y();
    msg.z = q.z();
    msg.w = q.w();
    return msg;
}

/**
 * @brief Reads one field out of the raw PointCloud2 bytes according to its datatype
 *
 * The datatype of the ring / time fields varies from bag to bag, so the value is read directly from
 * the offset and datatype rather than through an iterator.
 *
 * @param data     start address of that field
 * @param datatype the sensor_msgs::PointField datatype value
 * @return the value converted to double
 */
double ReadField(const std::uint8_t* data, std::uint8_t datatype)
{
    switch (datatype)
    {
        case sensor_msgs::PointField::INT8:
        {
            std::int8_t value;
            std::memcpy(&value, data, sizeof(value));
            return value;
        }
        case sensor_msgs::PointField::UINT8:
            return *data;
        case sensor_msgs::PointField::INT16:
        {
            std::int16_t value;
            std::memcpy(&value, data, sizeof(value));
            return value;
        }
        case sensor_msgs::PointField::UINT16:
        {
            std::uint16_t value;
            std::memcpy(&value, data, sizeof(value));
            return value;
        }
        case sensor_msgs::PointField::INT32:
        {
            std::int32_t value;
            std::memcpy(&value, data, sizeof(value));
            return value;
        }
        case sensor_msgs::PointField::UINT32:
        {
            std::uint32_t value;
            std::memcpy(&value, data, sizeof(value));
            return value;
        }
        case sensor_msgs::PointField::FLOAT32:
        {
            float value;
            std::memcpy(&value, data, sizeof(value));
            return value;
        }
        case sensor_msgs::PointField::FLOAT64:
        {
            double value;
            std::memcpy(&value, data, sizeof(value));
            return value;
        }
        default:
            return 0.0;
    }
}

}  // namespace

class LioNode
{
public:
    LioNode() : private_nh_("~")
    {
        ReadTopicsAndFrames();
        ConfigureOdometer();

        lidar_subscriber_ = node_handle_.subscribe(lidar_topic_, 10, &LioNode::OnLidar, this);
        imu_subscriber_ = node_handle_.subscribe(imu_topic_, 5000, &LioNode::OnImu, this);
        if (!gps_topic_.empty())
        {
            gps_subscriber_ = node_handle_.subscribe(gps_topic_, 200, &LioNode::OnGps, this);
        }

        odometry_publisher_ = private_nh_.advertise<nav_msgs::Odometry>("odometry", 10);
        path_publisher_ = private_nh_.advertise<nav_msgs::Path>("path", 2, true);
        gt_path_publisher_ = private_nh_.advertise<nav_msgs::Path>("gt_path", 2, true);
        feature_publisher_ = private_nh_.advertise<sensor_msgs::PointCloud2>("features", 2);
        submap_publisher_ = private_nh_.advertise<sensor_msgs::PointCloud2>("submap", 1, true);
        scan_publisher_ = private_nh_.advertise<sensor_msgs::PointCloud2>("scan", 2);

        OpenTrajectoryFiles();

        const FeatureExtractorOptions& feature_options = odometer_.feature_extractor().options();
        int selected = 0;
        for (int ring = 0; ring < feature_options.num_channels; ++ring)
        {
            if (odometer_.feature_extractor().IsRingSelected(ring))
            {
                ++selected;
            }
        }
        ROS_INFO("lidar_inertial_odometer ready");
        ROS_INFO("  topics : lidar='%s' imu='%s' gps='%s'", lidar_topic_.c_str(), imu_topic_.c_str(), gps_topic_.c_str());
        ROS_INFO("  frames : %s -> %s -> %s", odom_frame_.c_str(), base_frame_.c_str(), lidar_frame_.c_str());
        ROS_INFO("  ring   : selection=%s band=[%d,%d] -> %d/%d ch (dropped %d above + %d below)", ToString(feature_options.ring_selection).c_str(),
                 feature_options.ring_min, feature_options.ring_max, selected, feature_options.num_channels,
                 feature_options.num_channels - 1 - feature_options.ring_max, feature_options.ring_min);
        ROS_INFO("  normal : %s", ToString(feature_options.normal_method).c_str());
    }

    ~LioNode()
    {
        ROS_INFO("processed %zu lidar frames, %zu gt fixes", odometer_.trajectory().size(), gt_path_.poses.size());
    }

private:
    // ------------------------------------------------------------------
    // Parameter loading
    // ------------------------------------------------------------------
    template <typename T>
    T Param(const std::string& name, const T& fallback) const
    {
        T value;
        private_nh_.param<T>(name, value, fallback);
        return value;
    }

    std::vector<double> VectorParam(const std::string& name, const std::vector<double>& fallback) const
    {
        std::vector<double> value = fallback;
        private_nh_.getParam(name, value);
        return value;
    }

    void ReadTopicsAndFrames()
    {
        lidar_topic_ = Param<std::string>("lidar_topic", "/points_raw");
        imu_topic_ = Param<std::string>("imu_topic", "/imu_raw");
        gps_topic_ = Param<std::string>("gps_topic", "/gps/fix");

        odom_frame_ = Param<std::string>("odom_frame", "odom");
        base_frame_ = Param<std::string>("base_frame", "imu_link");
        lidar_frame_ = Param<std::string>("lidar_frame", "velodyne");

        trajectory_path_ = Param<std::string>("trajectory_csv", "");
        gt_trajectory_path_ = Param<std::string>("gt_trajectory_csv", "");

        publish_feature_cloud_ = Param<bool>("publish_feature_cloud", true);
        publish_submap_ = Param<bool>("publish_submap", true);
        publish_scan_cloud_ = Param<bool>("publish_scan_cloud", true);
        publish_tf_ = Param<bool>("publish_tf", true);
        publish_static_tf_ = Param<bool>("publish_static_tf", true);
    }

    void ConfigureOdometer()
    {
        LioOptions options;
        const double default_rotation_values[9] = {
             9.999976e-01,  7.553071e-04, -2.035826e-03,
            -7.854027e-04,  9.998898e-01, -1.482298e-02,
             2.024406e-03,  1.482454e-02,  9.998881e-01};
        const double default_translation_values[3] = {-8.086759e-01, 3.195559e-01, -7.997231e-01};

        const std::vector<double> default_rotation(default_rotation_values, default_rotation_values + 9);
        const std::vector<double> default_translation(default_translation_values, default_translation_values + 3);

        const std::vector<double> rotation = VectorParam("calib_imu_to_velo_R", default_rotation);
        const std::vector<double> translation = VectorParam("calib_imu_to_velo_T", default_translation);
        options.T_imu_lidar = MakeIsometry(rotation, translation).inverse();

        options.gravity = Eigen::Vector3d(0.0, 0.0, -Param<double>("gravity_magnitude", 9.80665));

        options.init_imu_samples = Param<int>("init_imu_samples", 60);
        use_imu_orientation_for_yaw_ = Param<bool>("use_imu_orientation_for_yaw", true);

        options.keyframe_translation = Param<double>("keyframe_translation", 1.0);
        options.keyframe_rotation = Param<double>("keyframe_rotation", 0.18);

        options.max_icp_translation_deviation = Param<double>("max_icp_translation_deviation", 1.5);
        options.max_icp_rotation_deviation = Param<double>("max_icp_rotation_deviation", 0.35);
        options.min_icp_correspondences = Param<int>("min_icp_correspondences", 50);

        options.velocity_correction_gain = Param<double>("velocity_correction_gain", 0.9);

        options.enable_deskew = Param<bool>("enable_deskew", true);
        options.scan_period = Param<double>("scan_period", 0.1);
        options.keep_deskewed_scan = publish_scan_cloud_;
        options.verbose = Param<bool>("verbose", true);

        odometer_.set_options(options);

        // --- preprocessing (ring based 64 -> 32ch) and feature/normal parameters ---
        FeatureExtractorOptions feature_options;
        const std::string ring_selection = Param<std::string>("ring_selection", "all");
        try
        {
            feature_options.ring_selection = ParseRingSelection(ring_selection);
        }
        catch (const std::exception& error)
        {
            ROS_WARN("%s -- falling back to 'all'", error.what());
            feature_options.ring_selection = RingSelection::kAll;
        }
        feature_options.num_channels = Param<int>("num_channels", 64);
        feature_options.ring_min = Param<int>("ring_min", 16);
        feature_options.ring_max = Param<int>("ring_max", 47);
        feature_options.ring_stride = Param<int>("ring_stride", 2);
        feature_options.vertical_fov_min_deg = Param<double>("vertical_fov_min_deg", -24.8);
        feature_options.vertical_fov_max_deg = Param<double>("vertical_fov_max_deg", 2.0);
        feature_options.min_range = Param<double>("min_range", 3.0);
        feature_options.max_range = Param<double>("max_range", 80.0);
        feature_options.min_z = Param<double>("min_z", -3.0);
        feature_options.scan_period = options.scan_period;

        feature_options.curvature_window = Param<int>("curvature_window", 5);
        feature_options.num_sectors = Param<int>("num_sectors", 6);
        feature_options.max_planar_per_sector = Param<int>("max_planar_per_sector", 40);
        feature_options.max_planar_curvature = Param<double>("max_planar_curvature", 0.1);

        const std::string method = Param<std::string>("normal_method", "neighborhood_pca");
        try
        {
            feature_options.normal_method = ParseNormalMethod(method);
        }
        catch (const std::exception& error)
        {
            ROS_WARN("%s -- falling back to 'neighborhood_pca'", error.what());
            feature_options.normal_method = NormalMethod::kNeighborhoodPca;
        }
        feature_options.normal_min_neighbors = Param<int>("normal_min_neighbors", 5);
        feature_options.normal_max_neighbors = Param<int>("normal_max_neighbors", 12);
        feature_options.normal_search_radius = Param<double>("normal_search_radius", 1.0);
        feature_options.normal_search_radius_scale = Param<double>("normal_search_radius_scale", 0.06);
        feature_options.max_plane_rms = Param<double>("max_plane_rms", 0.06);
        feature_options.max_eigenvalue_ratio = Param<double>("max_eigenvalue_ratio", 0.12);
        feature_options.min_planarity = Param<double>("min_planarity", 0.0);
        feature_options.voxel_size = Param<double>("feature_voxel_size", 0.4);
        odometer_.feature_extractor().set_options(feature_options);

        // --- local map (= submap, ICP target) ----------------------------------
        LocalMapOptions map_options;
        map_options.max_keyframes = Param<int>("map_max_keyframes", 25);
        map_options.voxel_size = Param<double>("map_voxel_size", 0.4);
        map_options.crop_radius = Param<double>("map_crop_radius", 80.0);
        odometer_.local_map().set_options(map_options);

        p2p_icp::IcpOptions icp_options = odometer_.icp().options();
        icp_options.max_iterations = Param<int>("icp_max_iterations", 12);
        icp_options.max_solver_iterations = Param<int>("icp_solver_iterations", 6);
        icp_options.max_correspondence_distance = Param<double>("icp_max_correspondence_distance", 1.5);
        icp_options.min_normal_dot = Param<double>("icp_min_normal_dot", 0.5);
        icp_options.translation_tolerance = Param<double>("icp_translation_tolerance", 1e-4);
        icp_options.rotation_tolerance = Param<double>("icp_rotation_tolerance", 1e-5);
        icp_options.error_tolerance = Param<double>("icp_error_tolerance", 1e-6);
        icp_options.huber_delta = Param<double>("icp_huber_delta", 0.2);
        icp_options.verbose = Param<bool>("icp_verbose", false);
        odometer_.icp().set_options(icp_options);

        T_base_lidar_ = options.T_imu_lidar;
    }

    void OnImu(const sensor_msgs::Imu::ConstPtr& msg)
    {
        PublishStaticTransformOnce(msg->header.stamp);

        ImuSample sample;
        sample.timestamp = msg->header.stamp.toSec();
        sample.angular_velocity = Eigen::Vector3d(msg->angular_velocity.x, msg->angular_velocity.y, msg->angular_velocity.z);
        sample.linear_acceleration = Eigen::Vector3d(msg->linear_acceleration.x, msg->linear_acceleration.y, msg->linear_acceleration.z);

        if (use_imu_orientation_for_yaw_ && !yaw_initialized_ && !odometer_.initialized())
        {
            const Eigen::Quaterniond orientation(msg->orientation.w,
                                                 msg->orientation.x,
                                                 msg->orientation.y,
                                                 msg->orientation.z);
            if (std::abs(orientation.norm() - 1.0) < 1e-3)
            {
                const Eigen::Vector3d forward = orientation * Eigen::Vector3d::UnitX();
                odometer_.set_initial_yaw(std::atan2(forward.y(), forward.x()));
                yaw_initialized_ = true;
            }
        }

        odometer_.AddImu(sample);
        PublishResults();
    }

    void OnLidar(const sensor_msgs::PointCloud2::ConstPtr& msg)
    {
        PublishStaticTransformOnce(msg->header.stamp);

        std::vector<RawLidarPoint> points = ConvertPointCloud(*msg);
        if (points.empty())
        {
            return;
        }
        raw_point_count_ = points.size();
        odometer_.AddLidarScan(msg->header.stamp.toSec(), points);
        PublishResults();
    }

    // ------------------------------------------------------------------
    // Ground truth (GPS) trajectory, for overlaying against the estimate.
    //
    // The estimated trajectory starts at the IMU position *at the moment the odometer finished
    // initializing*, so the GPS origin must be that exact moment too. Using the first fix as the
    // origin would offset the two trajectories by however far the vehicle moved before
    // initialization (this bag starts mid-drive).
    //
    // Fixes are buffered until that time is known; once it is, the position there is linearly
    // interpolated, fixed as the origin, and the whole buffer is emitted at once.
    // ------------------------------------------------------------------
    void OnGps(const sensor_msgs::NavSatFix::ConstPtr& msg)
    {
        if (!std::isfinite(msg->latitude) || !std::isfinite(msg->longitude))
        {
            return;
        }
        GpsFix fix;
        fix.timestamp = msg->header.stamp.toSec();
        fix.latitude = msg->latitude;
        fix.longitude = msg->longitude;
        fix.altitude = msg->altitude;
        gps_buffer_.push_back(fix);
        FlushGpsBuffer();  // a no-op until the odometer has initialized
    }

    /**
     * @brief Linearly interpolates the two fixes bracketing @p timestamp
     *
     * @param timestamp time to interpolate at [s]
     * @param out output: the interpolated fix
     * @return false while the buffer does not cover that time yet
     */
    bool InterpolateFix(double timestamp, GpsFix* out) const
    {
        if (gps_buffer_.empty())
        {
            return false;
        }
        if (timestamp <= gps_buffer_.front().timestamp)
        {
            *out = gps_buffer_.front();
            return true;
        }
        if (timestamp > gps_buffer_.back().timestamp)
        {
            return false;  // wait for more fixes to arrive
        }
        for (std::size_t i = 1; i < gps_buffer_.size(); ++i)
        {
            const GpsFix& a = gps_buffer_[i - 1];
            const GpsFix& b = gps_buffer_[i];
            if (timestamp <= b.timestamp)
            {
                const double span = b.timestamp - a.timestamp;
                double t = 0.0;
                if (span > 1e-9)
                {
                    t = (timestamp - a.timestamp) / span;
                }
                out->timestamp = timestamp;
                out->latitude = a.latitude + t * (b.latitude - a.latitude);
                out->longitude = a.longitude + t * (b.longitude - a.longitude);
                out->altitude = a.altitude + t * (b.altitude - a.altitude);
                return true;
            }
        }
        return false;
    }

    void FlushGpsBuffer()
    {
        if (!lio_started_)
        {
            return;  // not initialized yet, so keep buffering
        }
        if (!gps_origin_valid_)
        {
            GpsFix at_start;
            if (!InterpolateFix(lio_start_time_, &at_start))
            {
                return;
            }
            gps_reference_latitude_ = at_start.latitude;
            gps_origin_ = LatLonToLocal(at_start.latitude, at_start.longitude, at_start.altitude, gps_reference_latitude_);
            gps_origin_valid_ = true;
        }
        for (const GpsFix& fix : gps_buffer_)
        {
            if (fix.timestamp < lio_start_time_)
            {
                continue;
            }
            EmitGroundTruth(fix);
        }
        gps_buffer_.clear();
    }

    void EmitGroundTruth(const GpsFix& fix)
    {
        const Eigen::Vector3d local = LatLonToLocal(fix.latitude, fix.longitude, fix.altitude, gps_reference_latitude_) - gps_origin_;

        geometry_msgs::PoseStamped pose;
        pose.header.stamp = ros::Time(fix.timestamp);
        pose.header.frame_id = odom_frame_;
        pose.pose.position.x = local.x();
        pose.pose.position.y = local.y();
        pose.pose.position.z = local.z();
        pose.pose.orientation.w = 1.0;
        gt_path_.header = pose.header;
        gt_path_.poses.push_back(pose);
        gt_path_publisher_.publish(gt_path_);

        if (gt_stream_.is_open())
        {
            gt_stream_ << fix.timestamp << ' ' << local.x() << ' ' << local.y() << ' ' << local.z() << " 0 0 0 1\n";
            gt_stream_.flush();
        }
    }

    /**
     * @brief Converts a PointCloud2 into an array of RawLidarPoint (the ROS/core boundary)
     *
     * @param msg the input message
     * @return points in the lidar frame, or an empty array when there is no x field
     */
    std::vector<RawLidarPoint> ConvertPointCloud(const sensor_msgs::PointCloud2& msg) const
    {
        std::vector<RawLidarPoint> points;
        bool has_x = false;
        const sensor_msgs::PointField* ring_info = nullptr;
        const sensor_msgs::PointField* time_info = nullptr;
        for (const sensor_msgs::PointField& field : msg.fields)
        {
            if (field.name == "x")
            {
                has_x = true;
            }
            else if (field.name == "ring" || field.name == "channel")
            {
                ring_info = &field;
            }
            else if (field.name == "t" || field.name == "time" || field.name == "timestamp" || field.name == "time_offset")
            {
                time_info = &field;
            }
        }
        if (!has_x)
        {
            return points;
        }

        points.reserve(static_cast<std::size_t>(msg.width) * msg.height);
        sensor_msgs::PointCloud2ConstIterator<float> iter_x(msg, "x");
        sensor_msgs::PointCloud2ConstIterator<float> iter_y(msg, "y");
        sensor_msgs::PointCloud2ConstIterator<float> iter_z(msg, "z");

        // The ring / time datatypes vary by bag, so they are read from the raw bytes when present.
        for (std::size_t i = 0; iter_x != iter_x.end(); ++iter_x, ++iter_y, ++iter_z, ++i)
        {
            RawLidarPoint point;
            point.position = Eigen::Vector3d(*iter_x, *iter_y, *iter_z);
            if (!point.position.allFinite())
            {
                continue;
            }
            const std::uint8_t* base = msg.data.data() + i * msg.point_step;
            if (ring_info != nullptr)
            {
                point.ring = static_cast<int>(ReadField(base + ring_info->offset, ring_info->datatype));
            }
            if (time_info != nullptr)
            {
                point.rel_time = ReadField(base + time_info->offset, time_info->datatype);
            }
            points.push_back(point);
        }
        return points;
    }

    // ------------------------------------------------------------------
    // publish
    // ------------------------------------------------------------------
    /**
     * @brief Drains the finished frames and emits them as Odometry / Path / PointCloud2 / TF
     */
    void PublishResults()
    {
        const std::vector<LioFrameResult> results = odometer_.PopResults();
        for (const LioFrameResult& result : results)
        {
            if (!result.valid)
            {
                continue;
            }
            const ros::Time stamp(result.timestamp);

            // Start of the estimated trajectory; the GPS origin is pinned to this.
            if (!lio_started_)
            {
                lio_started_ = true;
                lio_start_time_ = result.timestamp;
                ROS_INFO("odometer initialized (t=%.3f); trajectory and GT are logged from here", lio_start_time_);
                FlushGpsBuffer();
            }

            nav_msgs::Odometry odometry;
            odometry.header.stamp = stamp;
            odometry.header.frame_id = odom_frame_;
            odometry.child_frame_id = base_frame_;
            odometry.pose.pose.position.x = result.state.position.x();
            odometry.pose.pose.position.y = result.state.position.y();
            odometry.pose.pose.position.z = result.state.position.z();
            odometry.pose.pose.orientation = ToMsg(result.state.rotation);
            // Per the nav_msgs/Odometry convention the velocity is expressed in child_frame_id (body).
            const Eigen::Vector3d body_velocity = result.state.rotation.transpose() * result.state.velocity;
            odometry.twist.twist.linear.x = body_velocity.x();
            odometry.twist.twist.linear.y = body_velocity.y();
            odometry.twist.twist.linear.z = body_velocity.z();
            odometry_publisher_.publish(odometry);

            // odom -> base_frame is a dynamic TF that changes every frame, while base_frame ->
            // lidar_frame is a calib value sent once from PublishStaticTransformOnce().
            if (publish_tf_)
            {
                geometry_msgs::TransformStamped transform;
                transform.header = odometry.header;
                transform.child_frame_id = base_frame_;
                transform.transform.translation.x = result.state.position.x();
                transform.transform.translation.y = result.state.position.y();
                transform.transform.translation.z = result.state.position.z();
                transform.transform.rotation = odometry.pose.pose.orientation;
                tf_broadcaster_.sendTransform(transform);
            }

            geometry_msgs::PoseStamped pose;
            pose.header = odometry.header;
            pose.pose = odometry.pose.pose;
            path_.header = odometry.header;
            path_.poses.push_back(pose);
            path_publisher_.publish(path_);

            if (publish_feature_cloud_ && feature_publisher_.getNumSubscribers() > 0)
            {
                feature_publisher_.publish(ToPointCloud2(result.feature_cloud_world, stamp, odom_frame_));
            }
            // The submap only changes at keyframes; publishing every frame would serialize hundreds
            // of thousands of points each time, so it is published on keyframes only and latched.
            if (publish_submap_ && result.is_keyframe && !odometer_.local_map().empty())
            {
                submap_publisher_.publish(ToPointCloud2(odometer_.local_map().cloud(), stamp, odom_frame_));
            }
            if (publish_scan_cloud_ && !result.scan_lidar.empty() && scan_publisher_.getNumSubscribers() > 0)
            {
                // ~/scan is the *preprocessed* scan, not the raw one: the ring band plus the range
                // and height filters are already applied. Use /points_raw for the raw scan.
                const FeatureExtractorOptions& fo = odometer_.feature_extractor().options();
                ROS_INFO_ONCE(
                        "~/scan is preprocessed, not raw: %zu -> %zu points "
                        "(ring band [%d,%d], min_range %.1f, max_range %.1f, min_z %.1f). "
                        "Use /points_raw for the raw scan.",
                        raw_point_count_, result.scan_lidar.size(), fo.ring_min, fo.ring_max, fo.min_range, fo.max_range, fo.min_z);
                scan_publisher_.publish(ToPointCloud2(result.scan_lidar, stamp, lidar_frame_));
            }

            if (trajectory_stream_.is_open())
            {
                const Eigen::Quaterniond q(result.state.rotation);
                trajectory_stream_ << result.timestamp << ' ' << result.state.position.x() << ' ' << result.state.position.y() << ' '
                                   << result.state.position.z() << ' ' << q.x() << ' ' << q.y() << ' ' << q.z() << ' ' << q.w() << '\n';
                trajectory_stream_.flush();
            }
        }
    }

    /**
     * @brief Broadcasts the static TF base_frame -> lidar_frame (the calib T_imu_lidar) exactly once
     *
     * It is stamped with the first bag message's time, because under use_sim_time there is no /clock
     * yet when the node is constructed and ros::Time::now() would be 0.
     *
     * @param stamp the time to use, taken from the first message's header.stamp
     */
    void PublishStaticTransformOnce(const ros::Time& stamp)
    {
        if (static_tf_sent_ || !publish_static_tf_)
        {
            return;
        }
        geometry_msgs::TransformStamped transform;
        transform.header.stamp = stamp;
        transform.header.frame_id = base_frame_;
        transform.child_frame_id = lidar_frame_;
        transform.transform.translation.x = T_base_lidar_.translation().x();
        transform.transform.translation.y = T_base_lidar_.translation().y();
        transform.transform.translation.z = T_base_lidar_.translation().z();
        transform.transform.rotation = ToMsg(T_base_lidar_.linear());
        static_tf_broadcaster_.sendTransform(transform);
        static_tf_sent_ = true;

        ROS_INFO("static TF %s -> %s : t=[%.4f %.4f %.4f]", base_frame_.c_str(), lidar_frame_.c_str(), T_base_lidar_.translation().x(),
                 T_base_lidar_.translation().y(), T_base_lidar_.translation().z());
    }

    /**
     * @brief Converts a cloud with normals (features / submap) into a PointCloud2
     *
     * @param cloud    the cloud to convert
     * @param stamp    message time
     * @param frame_id frame name
     * @return the PointCloud2 message (x, y, z, normal_x, normal_y, normal_z)
     */
    sensor_msgs::PointCloud2 ToPointCloud2(const common::PointCloud& cloud, const ros::Time& stamp, const std::string& frame_id) const
    {
        sensor_msgs::PointCloud2 msg;
        msg.header.stamp = stamp;
        msg.header.frame_id = frame_id;
        msg.height = 1;
        msg.width = static_cast<std::uint32_t>(cloud.size());
        msg.is_dense = true;
        msg.is_bigendian = false;

        sensor_msgs::PointCloud2Modifier modifier(msg);
        modifier.setPointCloud2Fields(6, "x", 1, sensor_msgs::PointField::FLOAT32, "y", 1, sensor_msgs::PointField::FLOAT32, "z", 1,
                                      sensor_msgs::PointField::FLOAT32, "normal_x", 1, sensor_msgs::PointField::FLOAT32, "normal_y", 1,
                                      sensor_msgs::PointField::FLOAT32, "normal_z", 1, sensor_msgs::PointField::FLOAT32);
        modifier.resize(cloud.size());

        sensor_msgs::PointCloud2Iterator<float> iter_x(msg, "x");
        sensor_msgs::PointCloud2Iterator<float> iter_y(msg, "y");
        sensor_msgs::PointCloud2Iterator<float> iter_z(msg, "z");
        sensor_msgs::PointCloud2Iterator<float> iter_nx(msg, "normal_x");
        sensor_msgs::PointCloud2Iterator<float> iter_ny(msg, "normal_y");
        sensor_msgs::PointCloud2Iterator<float> iter_nz(msg, "normal_z");
        for (const common::PointNormal& item : cloud)
        {
            *iter_x = static_cast<float>(item.point.x());
            *iter_y = static_cast<float>(item.point.y());
            *iter_z = static_cast<float>(item.point.z());
            *iter_nx = static_cast<float>(item.normal.x());
            *iter_ny = static_cast<float>(item.normal.y());
            *iter_nz = static_cast<float>(item.normal.z());
            ++iter_x, ++iter_y, ++iter_z, ++iter_nx, ++iter_ny, ++iter_nz;
        }
        return msg;
    }

    /**
     * @brief Converts a preprocessed scan (32ch selection plus deskew) into a PointCloud2
     *
     * `ring` rides along as a float field so rviz can colour by channel
     * ("Color Transformer: Intensity, Channel Name: ring").
     *
     * @param points   the points to convert
     * @param stamp    message time
     * @param frame_id frame name
     * @return the PointCloud2 message (x, y, z, ring)
     */
    sensor_msgs::PointCloud2 ToPointCloud2(const std::vector<RawLidarPoint>& points, const ros::Time& stamp, const std::string& frame_id) const
    {
        sensor_msgs::PointCloud2 msg;
        msg.header.stamp = stamp;
        msg.header.frame_id = frame_id;
        msg.height = 1;
        msg.width = static_cast<std::uint32_t>(points.size());
        msg.is_dense = true;
        msg.is_bigendian = false;

        sensor_msgs::PointCloud2Modifier modifier(msg);
        modifier.setPointCloud2Fields(4, "x", 1, sensor_msgs::PointField::FLOAT32, "y", 1, sensor_msgs::PointField::FLOAT32, "z", 1,
                                      sensor_msgs::PointField::FLOAT32, "ring", 1, sensor_msgs::PointField::FLOAT32);
        modifier.resize(points.size());

        sensor_msgs::PointCloud2Iterator<float> iter_x(msg, "x");
        sensor_msgs::PointCloud2Iterator<float> iter_y(msg, "y");
        sensor_msgs::PointCloud2Iterator<float> iter_z(msg, "z");
        sensor_msgs::PointCloud2Iterator<float> iter_ring(msg, "ring");
        for (const RawLidarPoint& item : points)
        {
            *iter_x = static_cast<float>(item.position.x());
            *iter_y = static_cast<float>(item.position.y());
            *iter_z = static_cast<float>(item.position.z());
            *iter_ring = static_cast<float>(item.ring);
            ++iter_x, ++iter_y, ++iter_z, ++iter_ring;
        }
        return msg;
    }

    /**
     * @brief Opens the TUM format files the estimated and GT trajectories are logged to
     */
    void OpenTrajectoryFiles()
    {
        if (!trajectory_path_.empty())
        {
            trajectory_stream_.open(trajectory_path_.c_str(), std::ios::out | std::ios::trunc);
            if (!trajectory_stream_)
            {
                ROS_WARN("cannot open trajectory_csv '%s'", trajectory_path_.c_str());
            }
            else
            {
                trajectory_stream_ << std::fixed << std::setprecision(9);
                trajectory_stream_ << "# TUM format: timestamp tx ty tz qx qy qz qw\n";
            }
        }
        if (!gt_trajectory_path_.empty())
        {
            gt_stream_.open(gt_trajectory_path_.c_str(), std::ios::out | std::ios::trunc);
            if (!gt_stream_)
            {
                ROS_WARN("cannot open gt_trajectory_csv '%s'", gt_trajectory_path_.c_str());
            }
            else
            {
                gt_stream_ << std::fixed << std::setprecision(9);
                gt_stream_ << "# TUM format: timestamp tx ty tz qx qy qz qw (GPS, local ENU)\n";
            }
        }
    }

    ros::NodeHandle node_handle_;
    ros::NodeHandle private_nh_;

    LioOdometer odometer_;

    std::string lidar_topic_;
    std::string imu_topic_;
    std::string gps_topic_;
    std::string odom_frame_;
    std::string base_frame_;
    std::string lidar_frame_;
    std::string trajectory_path_;
    std::string gt_trajectory_path_;
    bool publish_feature_cloud_ = true;
    bool publish_submap_ = true;
    bool publish_scan_cloud_ = true;
    bool publish_tf_ = true;
    bool publish_static_tf_ = true;
    bool static_tf_sent_ = false;
    std::size_t raw_point_count_ = 0;
    bool use_imu_orientation_for_yaw_ = true;
    bool yaw_initialized_ = false;

    Eigen::Isometry3d T_base_lidar_ = Eigen::Isometry3d::Identity();

    std::vector<GpsFix> gps_buffer_;
    bool lio_started_ = false;
    double lio_start_time_ = 0.0;

    bool gps_origin_valid_ = false;
    double gps_reference_latitude_ = 0.0;
    Eigen::Vector3d gps_origin_ = Eigen::Vector3d::Zero();

    nav_msgs::Path path_;
    nav_msgs::Path gt_path_;
    std::ofstream trajectory_stream_;
    std::ofstream gt_stream_;

    ros::Subscriber lidar_subscriber_;
    ros::Subscriber imu_subscriber_;
    ros::Subscriber gps_subscriber_;
    ros::Publisher odometry_publisher_;
    ros::Publisher path_publisher_;
    ros::Publisher gt_path_publisher_;
    ros::Publisher feature_publisher_;
    ros::Publisher submap_publisher_;
    ros::Publisher scan_publisher_;
    tf2_ros::TransformBroadcaster tf_broadcaster_;
    tf2_ros::StaticTransformBroadcaster static_tf_broadcaster_;
};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "lidar_inertial_odometer");
    LioNode node;
    ros::spin();
    return 0;
}
