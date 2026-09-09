// Offline scan-to-scan matching benchmark over a ROS 2 bag.
//
// Reads a PointCloud2 topic and an Imu topic straight out of the bag (no ROS graph, no playback),
// turns every scan into planar points with normals through FeatureExtractor, and registers each
// scan against the previous one with the hybrid point-to-point / point-to-plane ICP. The gyro
// supplies the rotation of the initial guess, the previous frame its translation.
//
// Every stage is timed on its own, because only the ICP part is comparable with the txt demo in
// point-to-point-plane-icp/app/main.cpp; feature extraction dominates on a real 128-channel scan.

#include <rclcpp/serialization.hpp>
#include <rclcpp/serialized_message.hpp>
#include <rosbag2_cpp/reader.hpp>
#include <rosbag2_storage/storage_filter.hpp>
#include <rosbag2_storage/storage_options.hpp>

#include "sensor_msgs/msg/imu.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "sensor_msgs/msg/point_field.hpp"

#include <Eigen/Geometry>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/kdtree.hpp"
#include "common/rotation.hpp"
#include "imu_preint/so3.hpp"
#include "lidar_inertial_odometer/feature_extractor.hpp"
#include "lidar_inertial_odometer/util.hpp"
#include "p2ptpl_icp/icp_point_to_point_plane.hpp"

namespace
{
using Clock = std::chrono::steady_clock;

const double kRadToDeg = 57.295779513082320876798154814105;

double Milliseconds(Clock::time_point begin, Clock::time_point end)
{
    return std::chrono::duration<double, std::milli>(end - begin).count();
}

/// Everything the benchmark reads off the command line.
struct Arguments
{
    std::string bag;  ///< bag directory (the one holding metadata.yaml)
    std::string storage = "sqlite3";
    std::string cloud_topic = "/center_lidar_points";
    std::string imu_topic = "/center_lidar_imu";
    double start_offset = 0.0;  ///< skip this many seconds of bag before the first frame
    int frames = 50;  ///< number of scan pairs to register
    int imu_clock_samples = 200;  ///< pre-roll used to measure the IMU clock offset
    /// header: use the Imu header stamp; bag: use the record time; auto: header plus the measured
    /// offset when the two clocks differ by more than a second (a lidar-internal IMU counts uptime).
    std::string imu_time = "auto";
    bool use_imu_rotation = true;  ///< gyro rotation as the initial guess
    bool constant_velocity = true;  ///< previous translation as the initial guess
    bool deskew = false;  ///< gyro + constant-velocity motion compensation inside one scan
    bool estimate_gyro_bias = false;  ///< take the pre-roll mean as the gyro bias (sensor must be still)
    Eigen::Vector3d gyro_bias = Eigen::Vector3d::Zero();  ///< explicit gyro bias, lidar frame [rad/s]
    bool per_frame = true;  ///< print the per-frame table
    std::string csv;  ///< optional per-frame csv
    std::string trajectory;  ///< optional TUM trajectory of the accumulated scan-to-scan poses
    /// R_lidar_imu: rotates an IMU-frame vector into the lidar frame. Identity for an IMU built
    /// into the lidar, which is what --imu-rpy overrides.
    Eigen::Matrix3d R_lidar_imu = Eigen::Matrix3d::Identity();
    FeatureExtractorOptions features;
    p2ptpl_icp::IcpOptions icp;
};

/// Defaults for the Hesai 128-channel scans of this bag plus the tuned ICP settings of
/// config/kitti.yaml. Ring 0 is the *top* channel here (+15.2 deg) and ring 127 the bottom (-24.7).
Arguments MakeDefaults()
{
    Arguments args;
    args.icp.collect_timing = true;

    args.features.ring_selection = RingSelection::kStride;
    args.features.num_channels = 128;
    args.features.ring_min = 0;
    args.features.ring_max = 127;
    args.features.ring_stride = 2;
    args.features.vertical_fov_min_deg = -25.0;
    args.features.vertical_fov_max_deg = 15.0;
    args.features.min_range = 3.0;
    args.features.max_range = 100.0;
    args.features.min_z = -3.0;
    args.features.scan_period = 0.1;
    args.features.normal_method = NormalMethod::kLoamCurvature;
    args.features.voxel_size = 0.4;

    args.icp.max_iterations = 12;
    args.icp.max_solver_iterations = 6;
    args.icp.max_correspondence_distance = 1.5;
    args.icp.min_normal_dot = 0.5;
    args.icp.translation_tolerance = 1e-4;
    args.icp.rotation_tolerance = 1e-5;
    args.icp.error_tolerance = 1e-6;
    args.icp.huber_delta = 0.2;
    return args;
}

void Usage(const char* program)
{
    const Arguments defaults = MakeDefaults();
    std::cout << "Usage: " << program << " --bag DIR [options]\n"
              << "  --bag DIR               bag directory holding metadata.yaml (required)\n"
              << "  --storage ID            storage plugin (default: " << defaults.storage << ")\n"
              << "  --cloud-topic NAME      PointCloud2 topic (default: " << defaults.cloud_topic << ")\n"
              << "  --imu-topic NAME        Imu topic, '' disables the IMU (default: " << defaults.imu_topic << ")\n"
              << "  --start SEC             skip this many seconds of bag first (default: 0)\n"
              << "  --frames N              scan pairs to register (default: " << defaults.frames << ")\n"
              << "  --imu-time MODE         auto | header | bag (default: auto)\n"
              << "  --imu-rpy R P Y         R_lidar_imu as ZYX Euler angles [rad] (default: identity)\n"
              << "  --no-imu-rotation       start ICP from identity instead of the gyro rotation\n"
              << "  --no-constant-velocity  start ICP from zero translation instead of the last one\n"
              << "  --deskew                gyro + constant-velocity motion compensation per scan\n"
              << "  --estimate-gyro-bias    subtract the pre-roll mean gyro (the sensor must be still there)\n"
              << "  --gyro-bias X Y Z       explicit gyro bias in the lidar frame [deg/s]\n"
              << "  --quiet                 summary only, no per-frame table\n"
              << "  --csv PATH              per-frame results as csv\n"
              << "  --trajectory PATH       accumulated scan-to-scan trajectory, TUM format\n"
              << "  --ring-min N / --ring-max N / --ring-stride N / --ring-all\n"
              << "  --min-range M / --max-range M / --min-z M\n"
              << "  --normal-method NAME    loam_curvature | neighborhood_pca (default: loam_curvature)\n"
              << "  --voxel M               feature support voxel [m] (default: 0.4)\n"
              << "  --max-planar-per-sector N (default: 40)\n"
              << "  --max-distance M        ICP correspondence distance (default: 1.5)\n"
              << "  --max-iterations N      ICP outer iterations (default: 12)\n"
              << "  --min-normal-dot D      normal agreement threshold, -1 disables (default: 0.5)\n"
              << "  --huber D               Huber delta, 0 disables (default: 0.2)\n"
              << "  --point-weight W / --plane-weight W (default: 1 / 1)\n"
              << "  --backend cpu|cuda (Ceres DENSE_QR only; residual/Jacobian: CPU; default: cuda when Ceres supports it)\n"
              << "  --help\n"
              << "Exit codes: 0 every frame converged, 1 input/error, 2 at least one frame did not.\n";
}

double Number(const std::string& text)
{
    std::size_t used = 0;
    const double value = std::stod(text, &used);
    if (used != text.size() || !std::isfinite(value))
    {
        throw std::invalid_argument("expected a finite number: " + text);
    }
    return value;
}

int Count(const std::string& text, int minimum)
{
    const double value = Number(text);
    if (value < minimum || value > std::numeric_limits<int>::max() || std::floor(value) != value)
    {
        throw std::invalid_argument("expected an integer >= " + std::to_string(minimum) + ": " + text);
    }
    return static_cast<int>(value);
}

Arguments ParseArguments(int argc, char** argv)
{
    Arguments args = MakeDefaults();
    for (int i = 1; i < argc; ++i)
    {
        const std::string option = argv[i];
        auto next = [&]() -> std::string
        {
            if (++i >= argc)
            {
                throw std::invalid_argument("missing value for " + option);
            }
            return argv[i];
        };
        if (option == "--bag")
        {
            args.bag = next();
        }
        else if (option == "--storage")
        {
            args.storage = next();
        }
        else if (option == "--cloud-topic")
        {
            args.cloud_topic = next();
        }
        else if (option == "--imu-topic")
        {
            args.imu_topic = next();
        }
        else if (option == "--start")
        {
            args.start_offset = Number(next());
        }
        else if (option == "--frames")
        {
            args.frames = Count(next(), 1);
        }
        else if (option == "--imu-time")
        {
            args.imu_time = next();
            if (args.imu_time != "auto" && args.imu_time != "header" && args.imu_time != "bag")
            {
                throw std::invalid_argument("--imu-time expects auto, header or bag");
            }
        }
        else if (option == "--imu-rpy")
        {
            const double roll = Number(next());
            const double pitch = Number(next());
            const double yaw = Number(next());
            args.R_lidar_imu = common::euler_zyx_to_rotation(roll, pitch, yaw);
        }
        else if (option == "--no-imu-rotation")
        {
            args.use_imu_rotation = false;
        }
        else if (option == "--no-constant-velocity")
        {
            args.constant_velocity = false;
        }
        else if (option == "--deskew")
        {
            args.deskew = true;
        }
        else if (option == "--estimate-gyro-bias")
        {
            args.estimate_gyro_bias = true;
        }
        else if (option == "--gyro-bias")
        {
            const double x = Number(next());
            const double y = Number(next());
            const double z = Number(next());
            args.gyro_bias = Eigen::Vector3d(x, y, z) / kRadToDeg;
        }
        else if (option == "--quiet")
        {
            args.per_frame = false;
        }
        else if (option == "--csv")
        {
            args.csv = next();
        }
        else if (option == "--trajectory")
        {
            args.trajectory = next();
        }
        else if (option == "--ring-min")
        {
            args.features.ring_min = Count(next(), 0);
        }
        else if (option == "--ring-max")
        {
            args.features.ring_max = Count(next(), 0);
        }
        else if (option == "--ring-stride")
        {
            args.features.ring_stride = Count(next(), 1);
            args.features.ring_selection = RingSelection::kStride;
        }
        else if (option == "--ring-all")
        {
            args.features.ring_selection = RingSelection::kAll;
        }
        else if (option == "--min-range")
        {
            args.features.min_range = Number(next());
        }
        else if (option == "--max-range")
        {
            args.features.max_range = Number(next());
        }
        else if (option == "--min-z")
        {
            args.features.min_z = Number(next());
        }
        else if (option == "--normal-method")
        {
            args.features.normal_method = ParseNormalMethod(next());
        }
        else if (option == "--voxel")
        {
            args.features.voxel_size = Number(next());
        }
        else if (option == "--max-planar-per-sector")
        {
            args.features.max_planar_per_sector = Count(next(), 1);
        }
        else if (option == "--max-distance")
        {
            args.icp.max_correspondence_distance = Number(next());
        }
        else if (option == "--max-iterations")
        {
            args.icp.max_iterations = Count(next(), 1);
        }
        else if (option == "--min-normal-dot")
        {
            args.icp.min_normal_dot = Number(next());
        }
        else if (option == "--huber")
        {
            args.icp.huber_delta = Number(next());
        }
        else if (option == "--backend")
        {
            const auto backend = next();
            if (backend != "cpu" && backend != "cuda")
                throw std::invalid_argument("backend must be cpu or cuda");
            args.icp.use_cuda = backend == "cuda";
        }
        else if (option == "--point-weight")
        {
            args.icp.point_weight = Number(next());
        }
        else if (option == "--plane-weight")
        {
            args.icp.plane_weight = Number(next());
        }
        else
        {
            throw std::invalid_argument("unknown option: " + option);
        }
    }
    if (args.bag.empty())
    {
        throw std::invalid_argument("--bag is required");
    }
    if (args.icp.point_weight <= 0.0 || args.icp.plane_weight <= 0.0 || args.icp.max_correspondence_distance <= 0.0)
    {
        throw std::invalid_argument("weights and --max-distance must be positive");
    }
    if (args.features.ring_min > args.features.ring_max || args.features.min_range >= args.features.max_range)
    {
        throw std::invalid_argument("empty ring band or range band");
    }
    return args;
}

/**
 * @brief Reads one PointCloud2 field out of the raw bytes according to its datatype
 *
 * The ring / timestamp datatypes differ from sensor to sensor, so they are read by offset instead
 * of through a typed iterator.
 *
 * @param data     address of the field inside the point
 * @param datatype sensor_msgs::msg::PointField datatype value
 * @return the value as a double
 */
double ReadField(const std::uint8_t* data, std::uint8_t datatype)
{
    switch (datatype)
    {
        case sensor_msgs::msg::PointField::INT8:
        {
            std::int8_t value;
            std::memcpy(&value, data, sizeof(value));
            return value;
        }
        case sensor_msgs::msg::PointField::UINT8:
            return *data;
        case sensor_msgs::msg::PointField::INT16:
        {
            std::int16_t value;
            std::memcpy(&value, data, sizeof(value));
            return value;
        }
        case sensor_msgs::msg::PointField::UINT16:
        {
            std::uint16_t value;
            std::memcpy(&value, data, sizeof(value));
            return value;
        }
        case sensor_msgs::msg::PointField::INT32:
        {
            std::int32_t value;
            std::memcpy(&value, data, sizeof(value));
            return value;
        }
        case sensor_msgs::msg::PointField::UINT32:
        {
            std::uint32_t value;
            std::memcpy(&value, data, sizeof(value));
            return value;
        }
        case sensor_msgs::msg::PointField::FLOAT32:
        {
            float value;
            std::memcpy(&value, data, sizeof(value));
            return value;
        }
        case sensor_msgs::msg::PointField::FLOAT64:
        {
            double value;
            std::memcpy(&value, data, sizeof(value));
            return value;
        }
        default:
            return 0.0;
    }
}

/**
 * @brief Converts a PointCloud2 into RawLidarPoint, dropping the no-return points
 *
 * A per-point time field may be relative to the scan start or an absolute epoch stamp (this bag
 * publishes the latter, constant over the whole scan, which is useless for deskewing). Anything
 * that does not land inside one scan period is handed to FeatureExtractor as -1, which recovers
 * rel_time from the azimuth instead.
 *
 * @param msg         the message
 * @param stamp       header stamp of the message [s], used to relativize an absolute time field
 * @param scan_period one revolution [s]
 * @return the points in the lidar frame
 */
std::vector<RawLidarPoint> ConvertCloud(const sensor_msgs::msg::PointCloud2& msg, double stamp, double scan_period)
{
    std::vector<RawLidarPoint> points;
    const sensor_msgs::msg::PointField* x_field = nullptr;
    const sensor_msgs::msg::PointField* y_field = nullptr;
    const sensor_msgs::msg::PointField* z_field = nullptr;
    const sensor_msgs::msg::PointField* ring_field = nullptr;
    const sensor_msgs::msg::PointField* time_field = nullptr;
    for (const sensor_msgs::msg::PointField& field : msg.fields)
    {
        if (field.name == "x")
        {
            x_field = &field;
        }
        else if (field.name == "y")
        {
            y_field = &field;
        }
        else if (field.name == "z")
        {
            z_field = &field;
        }
        else if (field.name == "ring" || field.name == "channel")
        {
            ring_field = &field;
        }
        else if (field.name == "t" || field.name == "time" || field.name == "timestamp" || field.name == "time_offset")
        {
            time_field = &field;
        }
    }
    if (x_field == nullptr || y_field == nullptr || z_field == nullptr)
    {
        return points;
    }

    const std::size_t count = static_cast<std::size_t>(msg.width) * msg.height;
    points.reserve(count / 2);
    for (std::size_t i = 0; i < count; ++i)
    {
        const std::uint8_t* base = msg.data.data() + i * msg.point_step;
        RawLidarPoint point;
        point.position = Eigen::Vector3d(ReadField(base + x_field->offset, x_field->datatype), ReadField(base + y_field->offset, y_field->datatype),
                                         ReadField(base + z_field->offset, z_field->datatype));
        // A no-return point is written as an exact zero; it carries no geometry and would poison
        // the azimuth ordering used to recover rel_time.
        if (!point.position.allFinite() || point.position.squaredNorm() < 1e-12)
        {
            continue;
        }
        if (ring_field != nullptr)
        {
            point.ring = static_cast<int>(ReadField(base + ring_field->offset, ring_field->datatype));
        }
        if (time_field != nullptr)
        {
            double time = ReadField(base + time_field->offset, time_field->datatype);
            if (std::abs(time) > 1e5)
            {
                time -= stamp;  // absolute epoch stamp
            }
            point.rel_time = (std::isfinite(time) && time >= -scan_period && time <= scan_period) ? time : -1.0;
        }
        points.push_back(point);
    }
    return points;
}

/**
 * @brief Integrates the gyro over (begin, end] into a relative rotation
 *
 * Trapezoidal rates over consecutive samples, clipped to the interval, composed on SO(3).
 *
 * @param samples IMU samples in time order, already rotated into the lidar frame
 * @param begin   interval start [s]
 * @param end     interval end [s]
 * @param bias    gyro bias to subtract [rad/s]
 * @param used    output: number of sample intervals that contributed, nullptr allowed
 * @return the relative rotation of the interval; identity when nothing covers it
 */
Eigen::Matrix3d IntegrateGyro(const std::vector<ImuSample>& samples, double begin, double end, const Eigen::Vector3d& bias, int* used)
{
    Eigen::Matrix3d rotation = Eigen::Matrix3d::Identity();
    int intervals = 0;
    for (std::size_t i = 1; i < samples.size(); ++i)
    {
        const double from = std::max(samples[i - 1].timestamp, begin);
        const double to = std::min(samples[i].timestamp, end);
        const double dt = to - from;
        if (dt <= 0.0)
        {
            continue;
        }
        const Eigen::Vector3d rate = 0.5 * (samples[i - 1].angular_velocity + samples[i].angular_velocity) - bias;
        rotation = rotation * imu_preint::so3::Exp(rate * dt);
        ++intervals;
    }
    if (used != nullptr)
    {
        *used = intervals;
    }
    return rotation;
}

/**
 * @brief Motion compensation inside one scan, gyro rotation plus constant velocity
 *
 * Moves every point from its own capture time to the start of the scan, so that both clouds handed
 * to ICP are snapshots of a single instant.
 *
 * @param points      preprocessed points, rel_time filled in
 * @param delta       sensor rotation over one full scan period
 * @param velocity    sensor velocity in the scan-start frame [m/s]
 * @param scan_period one revolution [s]
 */
void Deskew(std::vector<RawLidarPoint>* points, const Eigen::Matrix3d& delta, const Eigen::Vector3d& velocity, double scan_period)
{
    if (scan_period <= 0.0)
    {
        return;
    }
    const Eigen::Vector3d axis_angle = imu_preint::so3::Log(delta);
    for (RawLidarPoint& point : *points)
    {
        const double ratio = std::min(std::max(point.rel_time, 0.0), scan_period) / scan_period;
        point.position = imu_preint::so3::Exp(axis_angle * ratio) * point.position + velocity * (ratio * scan_period);
    }
}

/**
 * @brief Point and plane residual RMS at a given pose, on the same gated correspondences as ICP
 *
 * The solver reports one combined RMS; splitting it shows where the error sits. On scan-to-scan
 * data the point term cannot go below roughly half the point spacing, because the two scans never
 * sample the same physical spot, while the plane term can.
 *
 * @param source    source cloud
 * @param target    target cloud
 * @param pose      source -> target transform to evaluate
 * @param options   the ICP options, for the same distance / normal gating
 * @param point_rms output: RMS of ||e|| [m]
 * @param plane_rms output: RMS of n^T e [m]
 */
void EvaluateResiduals(const common::PointCloud& source, const common::PointCloud& target, const Eigen::Isometry3d& pose, const p2ptpl_icp::IcpOptions& options,
                       double* point_rms, double* plane_rms)
{
    common::KdTree3d tree;
    tree.build(target);
    double point_squared = 0.0;
    double plane_squared = 0.0;
    std::size_t count = 0;
    for (const common::PointNormal& item : source)
    {
        const Eigen::Vector3d transformed = pose * item.point;
        int index = -1;
        double squared_distance = 0.0;
        if (!tree.nearest(transformed, &index, &squared_distance) ||
            squared_distance > options.max_correspondence_distance * options.max_correspondence_distance)
        {
            continue;
        }
        const common::PointNormal& match = target[index];
        if (match.normal.squaredNorm() <= 0.0 ||
            (options.min_normal_dot > -1.0 && item.normal.squaredNorm() > 0.0 && (pose.linear() * item.normal).dot(match.normal) < options.min_normal_dot))
        {
            continue;
        }
        const Eigen::Vector3d error = transformed - match.point;
        point_squared += error.squaredNorm();
        const double plane_error = match.normal.dot(error);
        plane_squared += plane_error * plane_error;
        ++count;
    }
    const double infinity = std::numeric_limits<double>::infinity();
    *point_rms = count ? std::sqrt(point_squared / count) : infinity;
    *plane_rms = count ? std::sqrt(plane_squared / count) : infinity;
}

/// One registered scan pair.
struct FrameRecord
{
    p2ptpl_icp::IcpTiming icp_timing;  ///< module breakdown of do_icp() for this pair
    std::vector<p2ptpl_icp::IcpIterationLog> icp_iterations;  ///< per-outer-iteration timing of the same call
    double gyro_ms = 0, preprocess_ms = 0, deskew_ms = 0, extraction_ms = 0;
    double timestamp = 0.0;
    double dt = 0.0;
    int source_features = 0;
    int target_features = 0;
    int correspondences = 0;
    int iterations = 0;
    bool converged = false;
    double rms = 0.0;
    double max_abs = 0.0;
    double point_rms = 0.0;
    double plane_rms = 0.0;
    Eigen::Isometry3d relative = Eigen::Isometry3d::Identity();
    double gyro_angle = 0.0;  ///< rotation angle the gyro predicted for this interval [rad]
    double imu_icp_angle = 0.0;  ///< angle between the gyro prediction and the ICP rotation [rad]
    double convert_ms = 0.0;
    double feature_ms = 0.0;
    double setup_ms = 0.0;
    double icp_ms = 0.0;
    double total_ms = 0.0;
};

void PrintTiming(const char* label, std::vector<double> samples)
{
    if (samples.empty())
    {
        return;
    }
    const double mean = std::accumulate(samples.begin(), samples.end(), 0.0) / samples.size();
    std::sort(samples.begin(), samples.end());
    const std::size_t middle = samples.size() / 2;
    const double median = samples.size() % 2 ? samples[middle] : (samples[middle - 1] + samples[middle]) * 0.5;
    const std::size_t p95 = static_cast<std::size_t>(std::ceil(0.95 * samples.size())) - 1;
    std::cout << std::left << std::setw(24) << label << std::right << std::fixed << std::setprecision(3) << " mean=" << std::setw(9) << mean
              << "  median=" << std::setw(9) << median << "  p95=" << std::setw(9) << samples[p95] << "  min=" << std::setw(9) << samples.front()
              << "  max=" << std::setw(9) << samples.back() << " ms\n";
}

/// Mean of a per-frame quantity, 0 for an empty run.
double Mean(const std::vector<double>& values)
{
    if (values.empty())
    {
        return 0.0;
    }
    return std::accumulate(values.begin(), values.end(), 0.0) / values.size();
}

void WriteCsv(const std::string& path, const std::vector<FrameRecord>& records)
{
    std::ofstream file(path);
    if (!file)
    {
        throw std::runtime_error("cannot write '" + path + "'");
    }
    file << "frame,timestamp,dt,source_features,target_features,correspondences,iterations,converged,rms,point_rms,plane_rms,max_abs,dx,dy,dz,"
         << "translation,speed,rotation_deg,gyro_deg,imu_icp_deg,convert_ms,feature_ms,setup_ms,icp_ms,total_ms";
    p2ptpl_icp::WriteTimingCsvHeader(file, "icp_");  // module breakdown of the icp_ms column
    file << '\n';
    file << std::fixed << std::setprecision(9);
    for (std::size_t i = 0; i < records.size(); ++i)
    {
        const FrameRecord& record = records[i];
        const Eigen::Vector3d translation = record.relative.translation();
        const double angle = Eigen::AngleAxisd(record.relative.linear()).angle();
        file << i << ',' << record.timestamp << ',' << record.dt << ',' << record.source_features << ',' << record.target_features << ','
             << record.correspondences << ',' << record.iterations << ',' << (record.converged ? 1 : 0) << ',' << record.rms << ',' << record.point_rms << ','
             << record.plane_rms << ',' << record.max_abs << ',' << translation.x() << ',' << translation.y() << ',' << translation.z() << ','
             << translation.norm() << ',' << (record.dt > 0.0 ? translation.norm() / record.dt : 0.0) << ',' << angle * kRadToDeg << ','
             << record.gyro_angle * kRadToDeg << ',' << record.imu_icp_angle * kRadToDeg << ',' << record.convert_ms << ',' << record.feature_ms << ','
             << record.setup_ms << ',' << record.icp_ms << ',' << record.total_ms;
        p2ptpl_icp::WriteTimingCsvValues(file, record.icp_timing);
        file << '\n';
    }
}

void WriteTrajectory(const std::string& path, const std::vector<FrameRecord>& records, double first_timestamp)
{
    std::ofstream file(path);
    if (!file)
    {
        throw std::runtime_error("cannot write '" + path + "'");
    }
    file << "# TUM format: timestamp tx ty tz qx qy qz qw (scan-to-scan accumulation, no loop closure)\n";
    file << std::fixed << std::setprecision(9);
    Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
    double timestamp = first_timestamp;
    for (std::size_t i = 0; i <= records.size(); ++i)
    {
        const Eigen::Quaterniond orientation(pose.linear());
        file << timestamp << ' ' << pose.translation().x() << ' ' << pose.translation().y() << ' ' << pose.translation().z() << ' ' << orientation.x() << ' '
             << orientation.y() << ' ' << orientation.z() << ' ' << orientation.w() << '\n';
        if (i == records.size())
        {
            break;
        }
        pose = pose * records[i].relative;
        timestamp = records[i].timestamp;
    }
}
}  // namespace

int main(int argc, char** argv)
{
    try
    {
        for (int i = 1; i < argc; ++i)
        {
            if (std::string(argv[i]) == "--help")
            {
                Usage(argv[0]);
                return 0;
            }
        }
        const Arguments args = ParseArguments(argc, argv);

        rosbag2_storage::StorageOptions storage_options;
        storage_options.uri = args.bag;
        storage_options.storage_id = args.storage;
        rosbag2_cpp::ConverterOptions converter_options;
        converter_options.input_serialization_format = "cdr";
        converter_options.output_serialization_format = "cdr";

        rosbag2_cpp::Reader reader;
        reader.open(storage_options, converter_options);
        rosbag2_storage::StorageFilter filter;
        filter.topics.push_back(args.cloud_topic);
        if (!args.imu_topic.empty())
        {
            filter.topics.push_back(args.imu_topic);
        }
        reader.set_filter(filter);

        const std::int64_t bag_start_ns = reader.get_metadata().starting_time.time_since_epoch().count();
        if (args.start_offset > 0.0)
        {
            reader.seek(bag_start_ns + static_cast<std::int64_t>(args.start_offset * 1e9));
        }

        std::cout << "bag           : " << args.bag << '\n'
                  << "topics        : " << args.cloud_topic << (args.imu_topic.empty() ? " (no IMU)" : " + " + args.imu_topic) << '\n'
                  << "frames        : " << args.frames << ", starting " << args.start_offset << " s into the bag\n"
                  << "features      : " << ToString(args.features.normal_method) << ", rings [" << args.features.ring_min << ", " << args.features.ring_max
                  << "] " << ToString(args.features.ring_selection) << (args.features.ring_selection == RingSelection::kStride ? "/" : "")
                  << (args.features.ring_selection == RingSelection::kStride ? std::to_string(args.features.ring_stride) : "") << ", range ["
                  << args.features.min_range << ", " << args.features.max_range << "] m, voxel " << args.features.voxel_size << " m\n"
                  << "icp           : max_distance " << args.icp.max_correspondence_distance << " m, max_iterations " << args.icp.max_iterations
                  << ", min_normal_dot " << args.icp.min_normal_dot << ", huber " << args.icp.huber_delta << ", weights " << args.icp.point_weight << '/'
                  << args.icp.plane_weight << '\n'
                  << "initial guess : " << (args.use_imu_rotation ? "gyro rotation" : "identity rotation") << " + "
                  << (args.constant_velocity ? "constant velocity" : "zero translation") << (args.deskew ? ", deskew on" : ", deskew off") << '\n'
                  << "build         : " << LIO_BUILD_TYPE << " (use Release for timing)\n"
                  << "backend       : residual/Jacobian: cpu; Ceres DENSE_QR: "
                  << (args.icp.use_cuda ? "CUDA" : "CPU (Eigen)") << '\n'
                  << std::flush;

        rclcpp::Serialization<sensor_msgs::msg::PointCloud2> cloud_serialization;
        rclcpp::Serialization<sensor_msgs::msg::Imu> imu_serialization;
        FeatureExtractor extractor(args.features);
        p2ptpl_icp::IcpPointToPointPlane icp(args.icp);

        // --- IMU clock: a lidar-internal IMU stamps its own uptime, so the header clock and the
        // recording clock can be a full epoch apart. The median of (record time - header stamp)
        // over the pre-roll removes the recording jitter and keeps the sensor-side timing.
        std::vector<double> clock_offsets;
        Eigen::Vector3d gyro_sum = Eigen::Vector3d::Zero();
        Eigen::Vector3d gyro_bias = args.gyro_bias;
        double imu_clock_offset = 0.0;
        bool imu_clock_ready = args.imu_time != "auto";
        const bool imu_use_bag_time = args.imu_time == "bag";

        std::vector<ImuSample> imu_samples;
        std::vector<FrameRecord> records;
        records.reserve(args.frames);

        common::PointCloud previous_features;
        double previous_stamp = 0.0;
        double first_stamp = 0.0;
        Eigen::Vector3d previous_translation = Eigen::Vector3d::Zero();
        double previous_dt = args.features.scan_period;
        int skipped_scans = 0;

        if (args.per_frame)
        {
            std::cout << '\n'
                      << std::left << std::setw(6) << "frame" << std::setw(9) << "dt[s]" << std::setw(8) << "src" << std::setw(8) << "tgt" << std::setw(8)
                      << "corr" << std::setw(5) << "it" << std::setw(10) << "rms[m]" << std::setw(10) << "pt[m]" << std::setw(10) << "pl[m]" << std::setw(10)
                      << "|t|[m]" << std::setw(10) << "v[m/s]" << std::setw(10) << "gyro[deg]" << std::setw(10) << "icp[deg]" << std::setw(10) << "diff[deg]"
                      << std::setw(10) << "feat[ms]" << std::setw(10) << "setup[ms]" << std::setw(10) << "icp[ms]" << std::setw(10) << "total[ms]" << '\n'
                      << std::string(164, '-') << '\n';
        }

        while (reader.has_next() && static_cast<int>(records.size()) < args.frames)
        {
            const std::shared_ptr<rosbag2_storage::SerializedBagMessage> message = reader.read_next();
            rclcpp::SerializedMessage serialized(*message->serialized_data);

            if (message->topic_name == args.imu_topic)
            {
                sensor_msgs::msg::Imu imu;
                imu_serialization.deserialize_message(&serialized, &imu);
                const double header_stamp = imu.header.stamp.sec + imu.header.stamp.nanosec * 1e-9;
                const double bag_stamp = static_cast<double>(message->time_stamp) * 1e-9;
                if (!imu_clock_ready)
                {
                    clock_offsets.push_back(bag_stamp - header_stamp);
                    gyro_sum += Eigen::Vector3d(imu.angular_velocity.x, imu.angular_velocity.y, imu.angular_velocity.z);
                    if (static_cast<int>(clock_offsets.size()) >= args.imu_clock_samples)
                    {
                        std::sort(clock_offsets.begin(), clock_offsets.end());
                        const double median = clock_offsets[clock_offsets.size() / 2];
                        // Under a second apart means the two clocks are the same one; keep the
                        // header stamp as is rather than folding the recording latency into it.
                        imu_clock_offset = std::abs(median) > 1.0 ? median : 0.0;
                        imu_clock_ready = true;
                        const Eigen::Vector3d mean_rate = args.R_lidar_imu * (gyro_sum / static_cast<double>(clock_offsets.size()));
                        if (args.estimate_gyro_bias)
                        {
                            gyro_bias = mean_rate;
                        }
                        std::cout << "imu clock     : " << clock_offsets.size() << " samples, median (record - header) = " << std::fixed << std::setprecision(6)
                                  << median << " s -> offset " << imu_clock_offset << " s\n"
                                  << "gyro pre-roll : mean rate [" << mean_rate.x() * kRadToDeg << ", " << mean_rate.y() * kRadToDeg << ", "
                                  << mean_rate.z() * kRadToDeg << "] deg/s" << (args.estimate_gyro_bias ? " -> subtracted as bias\n" : "\n")
                                  << "gyro bias     : [" << gyro_bias.x() * kRadToDeg << ", " << gyro_bias.y() * kRadToDeg << ", " << gyro_bias.z() * kRadToDeg
                                  << "] deg/s\n"
                                  << std::flush;
                    }
                }
                ImuSample sample;
                sample.timestamp = imu_use_bag_time ? bag_stamp : header_stamp + imu_clock_offset;
                sample.angular_velocity = args.R_lidar_imu * Eigen::Vector3d(imu.angular_velocity.x, imu.angular_velocity.y, imu.angular_velocity.z);
                sample.linear_acceleration =
                        args.R_lidar_imu * Eigen::Vector3d(imu.linear_acceleration.x, imu.linear_acceleration.y, imu.linear_acceleration.z);
                imu_samples.push_back(sample);
                continue;
            }
            if (message->topic_name != args.cloud_topic)
            {
                continue;
            }
            // Hold the scans back until the IMU clock is known, otherwise the first initial guesses
            // would silently use a wrong timeline.
            if (!imu_clock_ready)
            {
                continue;
            }

            sensor_msgs::msg::PointCloud2 cloud;
            cloud_serialization.deserialize_message(&serialized, &cloud);
            const double stamp = cloud.header.stamp.sec + cloud.header.stamp.nanosec * 1e-9;

            const auto frame_begin = Clock::now();
            std::vector<RawLidarPoint> raw = ConvertCloud(cloud, stamp, args.features.scan_period);
            const auto convert_end = Clock::now();

            const double dt = previous_features.empty() ? previous_dt : stamp - previous_stamp;
            int gyro_intervals = 0;
            Eigen::Matrix3d gyro_rotation = Eigen::Matrix3d::Identity();
            if (args.use_imu_rotation && !args.imu_topic.empty() && dt > 0.0)
            {
                gyro_rotation = IntegrateGyro(imu_samples, stamp - dt, stamp, gyro_bias, &gyro_intervals);
            }

            std::vector<RawLidarPoint> preprocessed = extractor.Preprocess(raw);
            if (args.deskew)
            {
                // The scan runs forward from its stamp, so compensate with the rotation of the
                // *next* period; the gyro of the previous one is the best available estimate.
                const Eigen::Vector3d velocity = previous_dt > 0.0 ? Eigen::Vector3d(previous_translation / previous_dt) : Eigen::Vector3d::Zero();
                Deskew(&preprocessed, gyro_rotation, velocity, args.features.scan_period);
            }
            const FeatureCloud features = extractor.Extract(preprocessed);
            const auto feature_end = Clock::now();

            if (static_cast<int>(features.planar.size()) < 10)
            {
                ++skipped_scans;
                std::cerr << "warning: scan at " << std::fixed << std::setprecision(6) << stamp << " yielded only " << features.planar.size()
                          << " features, skipped\n";
                continue;
            }
            if (previous_features.empty())
            {
                previous_features = features.planar;
                previous_stamp = stamp;
                first_stamp = stamp;
                continue;
            }

            // source = current scan, target = previous scan, so the result is T_{k-1,k}: the sensor
            // motion between the two scans, which is also what the initial guess predicts.
            Eigen::Isometry3d guess = Eigen::Isometry3d::Identity();
            guess.linear() = gyro_rotation;
            if (args.constant_velocity && previous_dt > 0.0)
            {
                guess.translation() = previous_translation * (dt / previous_dt);
            }

            icp.set_source(features.planar);
            icp.set_target(previous_features);
            const auto setup_end = Clock::now();
            const p2ptpl_icp::IcpResult result = icp.do_icp(guess);
            const auto icp_end = Clock::now();

            // Outside the timed section on purpose: this is reporting, not part of registration.
            double point_rms = 0.0;
            double plane_rms = 0.0;
            EvaluateResiduals(features.planar, previous_features, result.transform, args.icp, &point_rms, &plane_rms);

            FrameRecord record;
            record.timestamp = stamp;
            record.dt = dt;
            record.source_features = static_cast<int>(features.planar.size());
            record.target_features = static_cast<int>(previous_features.size());
            record.correspondences = result.correspondences;
            record.iterations = result.iterations;
            record.converged = result.converged && result.transform.matrix().allFinite();
            record.rms = result.final_error;
            record.max_abs = result.final_max_abs_error;
            record.point_rms = point_rms;
            record.plane_rms = plane_rms;
            record.relative = result.transform;
            record.gyro_angle = Eigen::AngleAxisd(gyro_rotation).angle();
            record.imu_icp_angle = Eigen::AngleAxisd(gyro_rotation.transpose() * result.transform.linear()).angle();
            record.convert_ms = Milliseconds(frame_begin, convert_end);
            record.feature_ms = Milliseconds(convert_end, feature_end);
            record.setup_ms = Milliseconds(feature_end, setup_end);
            record.icp_ms = Milliseconds(setup_end, icp_end);
            record.icp_timing = result.timing;  // module breakdown inside do_icp(), enabled by collect_timing
            record.icp_iterations = result.history;
            record.total_ms = Milliseconds(frame_begin, icp_end);
            records.push_back(record);

            if (args.per_frame)
            {
                const double translation = record.relative.translation().norm();
                std::cout << std::left << std::setw(6) << records.size() << std::fixed << std::setprecision(4) << std::setw(9) << record.dt << std::setw(8)
                          << record.source_features << std::setw(8) << record.target_features << std::setw(8) << record.correspondences << std::setw(5)
                          << record.iterations << std::setprecision(4) << std::setw(10) << record.rms << std::setw(10) << record.point_rms << std::setw(10)
                          << record.plane_rms << std::setw(10) << translation << std::setw(10) << (record.dt > 0.0 ? translation / record.dt : 0.0)
                          << std::setprecision(3) << std::setw(10) << record.gyro_angle * kRadToDeg << std::setw(10)
                          << Eigen::AngleAxisd(record.relative.linear()).angle() * kRadToDeg << std::setw(10) << record.imu_icp_angle * kRadToDeg
                          << std::setw(10) << record.feature_ms << std::setw(10) << record.setup_ms << std::setw(10) << record.icp_ms << std::setw(10)
                          << record.total_ms << (record.converged ? "" : "  NOT CONVERGED") << '\n';
            }

            previous_features = features.planar;
            previous_stamp = stamp;
            previous_translation = result.transform.translation();
            previous_dt = dt;
            // Only the samples of the next interval are still needed.
            const std::size_t keep = std::distance(imu_samples.begin(), std::lower_bound(imu_samples.begin(), imu_samples.end(), stamp,
                                                                                         [](const ImuSample& sample, double value)
                                                                                         {
                                                                                             return sample.timestamp < value;
                                                                                         }));
            if (keep > 1)
            {
                imu_samples.erase(imu_samples.begin(), imu_samples.begin() + static_cast<long>(keep - 1));
            }
        }

        if (records.empty())
        {
            throw std::runtime_error("no scan pair was registered -- check --cloud-topic / --imu-topic / --start");
        }

        std::vector<double> convert, feature, setup, icp_times, total, matching;
        std::vector<double> rms, point_rms_values, plane_rms_values, correspondences, ratio, speeds, imu_angles, gyro_angles;
        std::vector<p2ptpl_icp::IcpTiming> icp_modules;
        int converged = 0;
        for (const FrameRecord& record : records)
        {
            icp_modules.push_back(record.icp_timing);
            convert.push_back(record.convert_ms);
            feature.push_back(record.feature_ms);
            setup.push_back(record.setup_ms);
            icp_times.push_back(record.icp_ms);
            matching.push_back(record.setup_ms + record.icp_ms);
            total.push_back(record.total_ms);
            rms.push_back(record.rms);
            point_rms_values.push_back(record.point_rms);
            plane_rms_values.push_back(record.plane_rms);
            correspondences.push_back(record.correspondences);
            ratio.push_back(record.source_features > 0 ? static_cast<double>(record.correspondences) / record.source_features : 0.0);
            speeds.push_back(record.dt > 0.0 ? record.relative.translation().norm() / record.dt : 0.0);
            imu_angles.push_back(record.imu_icp_angle * kRadToDeg);
            gyro_angles.push_back(record.gyro_angle * kRadToDeg);
            converged += record.converged ? 1 : 0;
        }

        std::cout << "\nTiming over " << records.size() << " registrations (steady_clock wall time)\n";
        PrintTiming("PointCloud2 -> points", convert);
        PrintTiming("Features + normals", feature);
        PrintTiming("ICP setup + kd-tree", setup);
        PrintTiming("ICP search + Ceres", icp_times);
        PrintTiming("Matching (setup+icp)", matching);
        PrintTiming("Per scan, end to end", total);
        std::cout << "Bag reading, deserialization and printing are outside these numbers.\n";
        // Second level: where the "ICP search + Ceres" row above went, module by module.
        p2ptpl_icp::PrintIcpTiming(std::cout, icp_modules, "Inside do_icp(): module timing");
        if (!records.empty())
        {
            p2ptpl_icp::PrintIterationTiming(std::cout, records.back().icp_iterations, "Last scan pair: per-outer-iteration module timing");
        }

        const double mean_total = Mean(total);
        std::cout << std::fixed << std::setprecision(3) << "Scan period          : " << args.features.scan_period * 1e3 << " ms -> real-time factor "
                  << (args.features.scan_period > 0.0 ? mean_total / (args.features.scan_period * 1e3) : 0.0) << " (mean end to end)\n"
                  << "\nQuality\n"
                  << "Converged            : " << converged << '/' << records.size()
                  << (skipped_scans > 0 ? "  (" + std::to_string(skipped_scans) + " scans skipped)" : "") << '\n'
                  << "Correspondences      : mean " << Mean(correspondences) << " (" << 100.0 * Mean(ratio) << " % of the source features)\n"
                  << "Hybrid RMS [m]       : mean " << Mean(rms) << ", max " << *std::max_element(rms.begin(), rms.end()) << '\n'
                  << "Point RMS ||e|| [m]  : mean " << Mean(point_rms_values) << ", max " << *std::max_element(point_rms_values.begin(), point_rms_values.end())
                  << "   (floored by the scan sampling, not by the alignment)\n"
                  << "Plane RMS n^T e [m]  : mean " << Mean(plane_rms_values) << ", max " << *std::max_element(plane_rms_values.begin(), plane_rms_values.end())
                  << '\n'
                  << "Speed [m/s]          : mean " << Mean(speeds) << ", max " << *std::max_element(speeds.begin(), speeds.end()) << '\n'
                  << "Gyro rotation [deg]  : mean " << Mean(gyro_angles) << " per scan pair\n"
                  << "Gyro vs ICP [deg]    : mean " << Mean(imu_angles) << ", max " << *std::max_element(imu_angles.begin(), imu_angles.end()) << '\n';

        if (!args.csv.empty())
        {
            WriteCsv(args.csv, records);
            std::cout << "per-frame csv        : " << args.csv << '\n';
        }
        if (!args.trajectory.empty())
        {
            WriteTrajectory(args.trajectory, records, first_stamp);
            std::cout << "trajectory (TUM)     : " << args.trajectory << '\n';
        }
        std::cout << "Scan-to-scan matching has no ground truth here: convergence and a small RMS do not prove the\n"
                     "alignment is right, they only say the solver stopped on a consistent correspondence set.\n";
        return converged == static_cast<int>(records.size()) ? 0 : 2;
    }
    catch (const std::exception& error)
    {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
