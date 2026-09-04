// Self-contained checks for lio_core, no KITTI rosbag needed:
//   1. ImuPreintegrator matches its own definition and predict() matches a
//      direct integration of the kinematics,
//   2. FeatureExtractor recovers correct normals on synthetic planes (both methods),
//   3. LioOdometer recovers the trajectory from a synthetic IMU + LiDAR stream.
//
// Written without GoogleTest; a failure exits non-zero.

#include <Eigen/Geometry>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <random>
#include <set>
#include <string>
#include <vector>

#include "lidar_inertial_odometer/feature_extractor.hpp"
#include "lidar_inertial_odometer/imu_preintegrator.hpp"
#include "lidar_inertial_odometer/lio_odometer.hpp"
#include "lidar_inertial_odometer/so3.hpp"

namespace
{
int g_failures = 0;

void Check(bool condition, const std::string& what)
{
    if (!condition)
    {
        std::printf("  [FAIL] %s\n", what.c_str());
        ++g_failures;
    }
    else
    {
        std::printf("  [ ok ] %s\n", what.c_str());
    }
}

void CheckNear(double value, double expected, double tolerance, const std::string& what)
{
    const bool ok = std::abs(value - expected) <= tolerance;
    if (!ok)
    {
        std::printf("  [FAIL] %s (got %.9g, expected %.9g, tol %.3g)\n", what.c_str(), value, expected, tolerance);
        ++g_failures;
    }
    else
    {
        std::printf("  [ ok ] %s (%.6g)\n", what.c_str(), value);
    }
}

/// Ground-truth motion: constant turn rate with a constant body acceleration.
struct GroundTruthMotion
{
    Eigen::Vector3d gravity = Eigen::Vector3d(0.0, 0.0, -9.80665);
    Eigen::Vector3d body_rate = Eigen::Vector3d(0.02, -0.03, 0.15);  ///< [rad/s]
    Eigen::Vector3d body_accel = Eigen::Vector3d(0.4, 0.15, -0.05);  ///< [m/s^2], body

    /// Builds the true state by integrating the kinematics directly.
    NavState Integrate(NavState state, double dt, int steps, std::vector<ImuSample>* samples) const
    {
        for (int k = 0; k < steps; ++k)
        {
            const Eigen::Vector3d accel_world = state.rotation * body_accel;
            if (samples != nullptr)
            {
                ImuSample sample;
                sample.timestamp = state.timestamp;
                // Noise-free measurement of the truth.
                sample.angular_velocity = body_rate;
                // The accelerometer measures the specific force f = R^T (a_world - g).
                sample.linear_acceleration = body_accel - state.rotation.transpose() * gravity;
                samples->push_back(sample);
            }
            state.position += state.velocity * dt + 0.5 * accel_world * dt * dt;
            state.velocity += accel_world * dt;
            state.rotation = state.rotation * so3::Exp(body_rate * dt);
            state.timestamp += dt;
        }
        if (samples != nullptr)
        {
            ImuSample last;
            last.timestamp = state.timestamp;
            last.angular_velocity = body_rate;
            last.linear_acceleration = body_accel - state.rotation.transpose() * gravity;
            samples->push_back(last);
        }
        return state;
    }
};

// ---------------------------------------------------------------------
// 1. IMU preintegration
// ---------------------------------------------------------------------
void TestPreintegration()
{
    std::printf("\n[1] ImuPreintegrator -- recursion, definition, predict()\n");

    GroundTruthMotion motion;
    const double dt = 0.005;
    const int steps = 200;  // 1.0 s

    NavState state_i;
    state_i.rotation = Eigen::AngleAxisd(0.3, Eigen::Vector3d(0.2, -0.5, 0.84).normalized()).toRotationMatrix();
    state_i.velocity = Eigen::Vector3d(3.0, -1.0, 0.2);
    state_i.position = Eigen::Vector3d(10.0, -5.0, 1.0);

    std::vector<ImuSample> samples;
    const NavState state_j = motion.Integrate(state_i, dt, steps, &samples);

    // Preintegration -> predict must reproduce the true state.
    ImuPreintegrator preintegrator;
    preintegrator.reset();
    for (int k = 0; k < steps; ++k)
    {
        preintegrator.integrate(samples[k].angular_velocity, samples[k].linear_acceleration, dt);
    }

    const NavState predicted = preintegrator.predict(state_i, motion.gravity);
    CheckNear((predicted.position - state_j.position).norm(), 0.0, 1e-9, "predict() position matches the directly integrated truth");
    CheckNear((predicted.velocity - state_j.velocity).norm(), 0.0, 1e-9, "predict() velocity matches the truth");
    CheckNear(so3::Log(predicted.rotation.transpose() * state_j.rotation).norm(), 0.0, 1e-9, "predict() attitude matches the truth");

    // Check the Delta definitions against the recursively integrated result.
    const PreintegratedDelta delta = preintegrator.delta();
    const double dt_ij = delta.dt;
    const Eigen::Matrix3d Ri_transpose = state_i.rotation.transpose();
    const Eigen::Matrix3d delta_R_definition = Ri_transpose * state_j.rotation;
    const Eigen::Vector3d delta_v_definition = Ri_transpose * (state_j.velocity - state_i.velocity - motion.gravity * dt_ij);
    const Eigen::Vector3d delta_p_definition =
            Ri_transpose * (state_j.position - state_i.position - state_i.velocity * dt_ij - 0.5 * motion.gravity * dt_ij * dt_ij);

    CheckNear(so3::Log(delta.rotation.transpose() * delta_R_definition).norm(), 0.0, 1e-9, "dR_ij = R_i^T R_j              (definition == recursion)");
    CheckNear((delta.velocity - delta_v_definition).norm(), 0.0, 1e-9, "dv_ij = R_i^T(v_j-v_i-g dt)    (definition == recursion)");
    CheckNear((delta.position - delta_p_definition).norm(), 0.0, 1e-9, "dp_ij = R_i^T(p_j-p_i-v_i dt-..) (definition == recursion)");

    // --- delta_at(): intermediate query used by deskewing ------------------
    const PreintegratedDelta half = preintegrator.delta_at(0.5 * dt_ij);
    ImuPreintegrator half_integrator;
    half_integrator.reset();
    for (int k = 0; k < steps / 2; ++k)
    {
        half_integrator.integrate(samples[k].angular_velocity, samples[k].linear_acceleration, dt);
    }
    CheckNear((half.position - half_integrator.delta().position).norm(), 0.0, 1e-9, "delta_at(t) matches an integration up to t (deskew)");
}

// ---------------------------------------------------------------------
// 2. Feature extraction / normals
// ---------------------------------------------------------------------
/// A bounded planar patch.
struct BoundedPlane
{
    Eigen::Vector3d point;  ///< centre
    Eigen::Vector3d normal;
    Eigen::Vector3d axis_u;  ///< in-plane axis 1
    Eigen::Vector3d axis_v;  ///< in-plane axis 2
    double half_u = 1.0;
    double half_v = 1.0;
};

/// Synthetic scene: a road, two building walls, and cross faces every 12 m.
///
/// Without the cross faces the x direction is unobservable: the normals of two
/// walls plus a floor span only {y, z}, so the point-to-plane residual never
/// penalises motion along x (the tunnel / highway case). Real KITTI drives get
/// this DoF from intersections and building steps.
std::vector<BoundedPlane> MakeScene()
{
    std::vector<BoundedPlane> planes;
    const Eigen::Vector3d x_axis = Eigen::Vector3d::UnitX();
    const Eigen::Vector3d y_axis = Eigen::Vector3d::UnitY();
    const Eigen::Vector3d z_axis = Eigen::Vector3d::UnitZ();

    // road
    planes.push_back({Eigen::Vector3d(40.0, 0.0, -1.7), z_axis, x_axis, y_axis, 200.0, 200.0});
    // building walls on both sides (z: -1.7 .. 6.3)
    for (const double side : {9.0, -9.0})
    {
        planes.push_back({Eigen::Vector3d(40.0, side, 2.3), y_axis, x_axis, z_axis, 200.0, 4.0});
    }
    // cross faces every 12 m (|y| < 3 is left open for the road)
    for (int index = -2; index <= 10; ++index)
    {
        const double x = 12.0 * index;
        for (const double side : {6.0, -6.0})
        {
            planes.push_back({Eigen::Vector3d(x, side, 2.3), x_axis, y_axis, z_axis, 3.0, 4.0});
        }
    }
    return planes;
}

/// How MakeSyntheticScan asks for the sensor pose at each point's timestamp:
/// a fixed pose for a static scan, an integrated one while driving.
class SensorPoseProvider
{
public:
    virtual ~SensorPoseProvider()
    {
    }

    /// @param tau time relative to the scan start [s]
    /// @return the world <- lidar pose at that time
    virtual Eigen::Isometry3d PoseAt(double tau) const = 0;
};

/// Sensor at rest -- no motion distortion.
class StaticSensorPose : public SensorPoseProvider
{
public:
    Eigen::Isometry3d PoseAt(double) const
    {
        return Eigen::Isometry3d::Identity();
    }
};

/// Sensor in motion: integrates the motion tau seconds past the scan start.
class MovingSensorPose : public SensorPoseProvider
{
public:
    MovingSensorPose(const GroundTruthMotion& motion, const NavState& scan_start, const Eigen::Isometry3d& T_imu_lidar, double imu_dt)
    {
        motion_ = motion;
        scan_start_ = scan_start;
        T_imu_lidar_ = T_imu_lidar;
        imu_dt_ = imu_dt;
    }

    Eigen::Isometry3d PoseAt(double tau) const
    {
        const int steps = std::max(1, static_cast<int>(std::lround(tau / imu_dt_)));
        const NavState body = motion_.Integrate(scan_start_, tau / steps, steps, nullptr);
        return body.isometry() * T_imu_lidar_;
    }

private:
    GroundTruthMotion motion_;
    NavState scan_start_;
    Eigen::Isometry3d T_imu_lidar_;
    double imu_dt_;
};

/// Scans the synthetic scene with a 64-channel LiDAR.
///
/// The sensor moves while the azimuth sweeps, exactly like a real Velodyne, so
/// each point is intersected at its own pose -- the scan carries real motion
/// distortion for deskewing to undo.
std::vector<RawLidarPoint> MakeSyntheticScan(const SensorPoseProvider& sensor_pose, unsigned seed)
{
    std::mt19937 generator(seed);
    std::normal_distribution<double> noise(0.0, 0.005);

    static const std::vector<BoundedPlane> planes = MakeScene();

    std::vector<RawLidarPoint> scan;
    const int num_channels = 64;
    const int num_azimuth = 900;
    const double scan_period = 0.1;

    // Precompute the sensor pose per azimuth step (all 64 channels of one step
    // fire together and share a pose).
    std::vector<Eigen::Isometry3d> poses(num_azimuth);
    for (int step = 0; step < num_azimuth; ++step)
    {
        poses[step] = sensor_pose.PoseAt(scan_period * step / num_azimuth);
    }

    for (int ring = 0; ring < num_channels; ++ring)
    {
        const double elevation = (-24.8 + (2.0 - (-24.8)) * ring / (num_channels - 1)) * M_PI / 180.0;
        for (int step = 0; step < num_azimuth; ++step)
        {
            const double azimuth = -2.0 * M_PI * step / num_azimuth;  // clockwise
            const Eigen::Vector3d direction_sensor(std::cos(elevation) * std::cos(azimuth), std::cos(elevation) * std::sin(azimuth), std::sin(elevation));
            const Eigen::Isometry3d& world_from_sensor = poses[step];
            const Eigen::Vector3d origin = world_from_sensor.translation();
            const Eigen::Vector3d direction = world_from_sensor.linear() * direction_sensor;

            // Intersect with the nearest planar patch.
            double best_range = std::numeric_limits<double>::infinity();
            for (const BoundedPlane& plane : planes)
            {
                const double denominator = direction.dot(plane.normal);
                if (std::abs(denominator) < 1e-6)
                {
                    continue;
                }
                const double range = (plane.point - origin).dot(plane.normal) / denominator;
                if (range <= 0.5 || range >= best_range)
                {
                    continue;
                }
                const Eigen::Vector3d offset = origin + range * direction - plane.point;
                if (std::abs(offset.dot(plane.axis_u)) > plane.half_u || std::abs(offset.dot(plane.axis_v)) > plane.half_v)
                {
                    continue;  // passes outside the patch
                }
                best_range = range;
            }
            if (!std::isfinite(best_range) || best_range > 70.0)
            {
                continue;
            }

            RawLidarPoint point;
            point.position = direction_sensor * (best_range + noise(generator));
            point.ring = ring;
            point.rel_time = scan_period * step / num_azimuth;
            scan.push_back(point);
        }
    }
    return scan;
}

void TestFeatureExtraction()
{
    std::printf("\n[2] FeatureExtractor -- planar candidates + PCA normals\n");

    // Static scan -- no motion distortion.
    const StaticSensorPose static_pose;
    const std::vector<RawLidarPoint> scan = MakeSyntheticScan(static_pose, 7);
    std::printf("       synthetic scan: %zu points (64 ch)\n", scan.size());

    for (const NormalMethod method : {NormalMethod::kLoamCurvature, NormalMethod::kNeighborhoodPca})
    {
        FeatureExtractorOptions options;
        options.normal_method = method;
        FeatureExtractor extractor(options);

        const std::vector<RawLidarPoint> preprocessed = extractor.Preprocess(scan);
        const FeatureCloud features = extractor.Extract(preprocessed);

        // 64 -> 32 ch: the default band [16, 47] must drop rings 0..15 and 48..63.
        std::set<int> rings;
        for (const RawLidarPoint& point : preprocessed)
        {
            rings.insert(point.ring);
        }
        bool only_middle = true;
        for (std::set<int>::const_iterator it = rings.begin(); it != rings.end(); ++it)
        {
            const int ring = *it;

            if (ring < 16 || ring > 47)
            {
                only_middle = false;
                break;
            }
        }
        std::printf("     -- %s: preprocessed %zu pts (%zu rings [%d..%d]), candidates %d, normals %zu\n", ToString(method).c_str(), preprocessed.size(),
                    rings.size(), *rings.begin(), *rings.rbegin(), features.num_candidates, features.planar.size());
        Check(rings.size() == 32, ToString(method) + ": 64 -> 32 ch (half the channels)");
        Check(only_middle, ToString(method) + ": only the middle 32 ch (16..47) survive");
        Check(features.planar.size() > 200, ToString(method) + ": enough planar features extracted");

        // Scene normals are axis-aligned, so every estimate must be too.
        int aligned = 0;
        for (const p2p_icp::PointNormal& item : features.planar)
        {
            const double best = item.normal.cwiseAbs().maxCoeff();
            if (best > 0.99)
            {
                ++aligned;
            }
        }
        const double ratio = static_cast<double>(aligned) / features.planar.size();
        CheckNear(ratio, 1.0, 0.05, ToString(method) + ": fraction of normals aligned with the true plane normal");

        // Sign consistency: every normal must face the sensor.
        int facing = 0;
        for (const p2p_icp::PointNormal& item : features.planar)
        {
            if (item.normal.dot(-item.point) > 0.0)
            {
                ++facing;
            }
        }
        Check(facing == static_cast<int>(features.planar.size()), ToString(method) + ": every normal faces the sensor");
    }

    // Full band (0..63) with stride 2 -> a uniform half.
    {
        FeatureExtractorOptions options;
        options.ring_selection = RingSelection::kStride;
        options.ring_min = 0;
        options.ring_max = 63;
        options.ring_stride = 2;
        std::set<int> rings;
        int odd = 0;
        for (const RawLidarPoint& point : FeatureExtractor(options).Preprocess(scan))
        {
            rings.insert(point.ring);
            odd += (point.ring % 2 != 0) ? 1 : 0;
        }
        std::printf("     -- stride (band 0..63): %zu rings, %d odd rings\n", rings.size(), odd);
        Check(rings.size() == 32 && odd == 0, "ring_selection=stride: 32 even rings only");
    }
    // Opening the band to the full range keeps all 64 channels.
    {
        FeatureExtractorOptions options;
        options.ring_min = 0;
        options.ring_max = 63;
        std::set<int> rings;
        for (const RawLidarPoint& point : FeatureExtractor(options).Preprocess(scan))
        {
            rings.insert(point.ring);
        }
        std::printf("     -- all (band 0..63): %zu rings\n", rings.size());
        Check(rings.size() == 64, "band 0..63: all 64 channels");
    }
    // A narrower band must keep exactly that band.
    {
        FeatureExtractorOptions options;
        options.ring_min = 20;
        options.ring_max = 39;
        std::set<int> rings;
        for (const RawLidarPoint& point : FeatureExtractor(options).Preprocess(scan))
        {
            rings.insert(point.ring);
        }
        std::printf("     -- band 20..39: %zu rings [%d..%d]\n", rings.size(), *rings.begin(), *rings.rbegin());
        Check(rings.size() == 20 && *rings.begin() == 20 && *rings.rbegin() == 39, "band [20,39]: 20 middle channels only");
    }
}

// ---------------------------------------------------------------------
// 2b. Deskew in isolation
// ---------------------------------------------------------------------
/// The true world coordinate of every point is known, so the deskewed point is
/// compared against the truth expressed in the scan-start lidar frame.
/// (Two scans cannot be paired by index: as the sensor moves, some rays hit a
/// patch and others miss, which breaks the correspondence.)
void TestDeskew()
{
    std::printf("\n[2b] Deskew -- undoing motion during the scan\n");

    GroundTruthMotion motion;
    motion.body_rate = Eigen::Vector3d(0.0, 0.0, 0.3);  // a clear turn
    motion.body_accel = Eigen::Vector3d::Zero();

    NavState start;
    start.velocity = Eigen::Vector3d(8.0, 0.0, 0.0);  // 0.8 m over 0.1 s
    const double imu_dt = 0.005;
    const double scan_period = 0.1;

    Eigen::Isometry3d T_il = Eigen::Isometry3d::Identity();
    T_il.translation() = Eigen::Vector3d(0.0, 0.0, 0.3);
    const Eigen::Isometry3d T_li = T_il.inverse();

    const MovingSensorPose moving_pose(motion, start, T_il, imu_dt);
    const std::vector<RawLidarPoint> distorted = MakeSyntheticScan(moving_pose, 11);
    const Eigen::Isometry3d start_pose = start.isometry() * T_il;
    const Eigen::Isometry3d start_pose_inverse = start_pose.inverse();

    // Integrate the IMU over the scan interval.
    std::vector<ImuSample> samples;
    motion.Integrate(start, imu_dt, 40, &samples);
    ImuPreintegrator scan_integrator;
    scan_integrator.reset();
    for (std::size_t i = 0; i + 1 < samples.size(); ++i)
    {
        if (samples[i].timestamp >= scan_period)
        {
            break;
        }
        const double dt = std::min(samples[i + 1].timestamp, scan_period) - samples[i].timestamp;
        scan_integrator.integrate(samples[i].angular_velocity, samples[i].linear_acceleration, dt);
    }

    const Eigen::Matrix3d Ri_transpose = start.rotation.transpose();

    double before = 0.0;
    double after = 0.0;
    for (const RawLidarPoint& point : distorted)
    {
        const double tau = point.rel_time;
        // Truth: lift to world at the measurement pose, then into the scan-start frame.
        const Eigen::Vector3d expected = start_pose_inverse * (moving_pose.PoseAt(tau) * point.position);

        // Same expression as LioOdometer::Deskew().
        const PreintegratedDelta delta = scan_integrator.delta_at(tau);
        const Eigen::Vector3d corrected =
                T_li * (delta.rotation * (T_il * point.position) + delta.position + Ri_transpose * (start.velocity * tau + 0.5 * motion.gravity * tau * tau));

        before = std::max(before, (point.position - expected).norm());
        after = std::max(after, (corrected - expected).norm());
    }
    std::printf("       %zu pts | max distortion %.3f m -> %.5f m after deskew\n", distorted.size(), before, after);
    Check(before > 0.3, "the synthetic scan carries clear motion distortion");
    Check(after < 0.02, "residual distortion after deskew < 2 cm");
}

// ---------------------------------------------------------------------
// 3. Full pipeline
// ---------------------------------------------------------------------
/// Runs the pipeline once and reports the error metrics and frame count.
struct PipelineOutcome
{
    std::size_t frames = 0;
    /// Absolute trajectory error, aligned on the first frame. Odometry has no
    /// loop closure, so an early error stays as an offset to the end.
    double max_position_error = 0.0;
    double max_rotation_error = 0.0;
    /// Worst frame-to-frame relative pose error, the per-frame registration
    /// quality. The first kWarmupFrames are excluded: the very first frame has an
    /// empty local map, so ICP cannot run and the velocity is unobservable. It
    /// takes a few frames to converge from 0, and the deskew term R_i^T v_i tau
    /// is wrong meanwhile -- a startup transient, not steady-state performance.
    double max_relative_position_error = 0.0;
    double max_relative_rotation_error = 0.0;
    /// Final position error over distance travelled, after the warm-up [%].
    double drift_percent = 0.0;
    double travelled = 0.0;
};

const std::size_t kWarmupFrames = 5;

PipelineOutcome RunPipeline(NormalMethod method, bool enable_deskew)
{
    GroundTruthMotion motion;
    // Pure turn + forward motion, to stay inside the synthetic scene.
    motion.body_rate = Eigen::Vector3d(0.0, 0.0, 0.05);
    motion.body_accel = Eigen::Vector3d(0.0, 0.0, 0.0);

    LioOptions options;
    options.T_imu_lidar = Eigen::Isometry3d::Identity();
    options.T_imu_lidar.translation() = Eigen::Vector3d(0.0, 0.0, 0.3);
    options.gravity = motion.gravity;
    options.init_imu_samples = 20;
    options.keyframe_translation = 0.5;
    options.enable_deskew = enable_deskew;
    // LIO_VERBOSE=1 turns on the per-frame diagnostics.
    options.verbose = (std::getenv("LIO_VERBOSE") != nullptr);

    LioOdometer odometer(options);
    FeatureExtractorOptions feature_options;
    feature_options.normal_method = method;
    feature_options.min_range = 1.0;
    odometer.feature_extractor().set_options(feature_options);

    p2p_icp::IcpOptions icp_options = odometer.icp().options();
    icp_options.max_iterations = 15;
    icp_options.max_correspondence_distance = 1.5;
    icp_options.min_normal_dot = 0.5;
    icp_options.translation_tolerance = 1e-5;
    icp_options.rotation_tolerance = 1e-6;
    icp_options.error_tolerance = 1e-8;
    odometer.icp().set_options(icp_options);

    // Build the true trajectory and feed IMU at 200 Hz, LiDAR at 10 Hz.
    NavState truth;
    truth.timestamp = 0.0;
    truth.velocity = Eigen::Vector3d(4.0, 0.0, 0.0);  // forward along body x
    const double imu_dt = 0.005;

    std::vector<NavState> truth_at_scan;

    // Feed the initialization IMU first, without LiDAR.
    {
        std::vector<ImuSample> samples;
        truth = motion.Integrate(truth, imu_dt, 40, &samples);
        for (std::size_t i = 0; i + 1 < samples.size(); ++i)
        {
            odometer.AddImu(samples[i]);
        }
    }

    const int num_scans = 40;
    for (int scan_index = 0; scan_index < num_scans; ++scan_index)
    {
        const double scan_time = truth.timestamp;
        truth_at_scan.push_back(truth);

        // The vehicle really moves during the scan; if deskewing does not undo
        // that, no amount of ICP accuracy removes the resulting error floor.
        const NavState scan_start = truth;
        const MovingSensorPose scan_pose(motion, scan_start, options.T_imu_lidar, imu_dt);
        odometer.AddLidarScan(scan_time, MakeSyntheticScan(scan_pose, 100 + scan_index));

        // A queued scan only runs once the IMU covers its sweep plus slack.
        std::vector<ImuSample> samples;
        truth = motion.Integrate(truth, imu_dt, 22, &samples);
        for (std::size_t i = 0; i + 1 < samples.size(); ++i)
        {
            odometer.AddImu(samples[i]);
        }
        odometer.PopResults();
    }

    const std::vector<NavState>& trajectory = odometer.trajectory();
    PipelineOutcome outcome;
    outcome.frames = trajectory.size();
    if (trajectory.empty())
    {
        return outcome;
    }

    // Compare relative trajectories with the first frame as origin (yaw is unobservable).
    const std::size_t offset = truth_at_scan.size() - trajectory.size();
    const Eigen::Isometry3d truth_origin = truth_at_scan[offset].isometry();
    const Eigen::Isometry3d estimate_origin = trajectory.front().isometry();

    for (std::size_t i = 0; i < trajectory.size(); ++i)
    {
        const Eigen::Isometry3d truth_relative = truth_origin.inverse() * truth_at_scan[offset + i].isometry();
        const Eigen::Isometry3d estimate_relative = estimate_origin.inverse() * trajectory[i].isometry();
        const Eigen::Isometry3d error = truth_relative.inverse() * estimate_relative;
        outcome.max_position_error = std::max(outcome.max_position_error, error.translation().norm());
        outcome.max_rotation_error = std::max(outcome.max_rotation_error, Eigen::AngleAxisd(error.linear()).angle());

        // Relative pose error (RPE), excluding the startup transient.
        if (i > kWarmupFrames)
        {
            const Eigen::Isometry3d truth_step = truth_at_scan[offset + i - 1].isometry().inverse() * truth_at_scan[offset + i].isometry();
            const Eigen::Isometry3d estimate_step = trajectory[i - 1].isometry().inverse() * trajectory[i].isometry();
            const Eigen::Isometry3d step_error = truth_step.inverse() * estimate_step;
            outcome.max_relative_position_error = std::max(outcome.max_relative_position_error, step_error.translation().norm());
            outcome.max_relative_rotation_error = std::max(outcome.max_relative_rotation_error, Eigen::AngleAxisd(step_error.linear()).angle());
        }
    }

    // Drift: re-origin after the warm-up and take the final position error over
    // the path length of that segment.
    if (trajectory.size() > kWarmupFrames + 1)
    {
        const Eigen::Isometry3d truth_base = truth_at_scan[offset + kWarmupFrames].isometry();
        const Eigen::Isometry3d estimate_base = trajectory[kWarmupFrames].isometry();
        const Eigen::Isometry3d final_error =
                (truth_base.inverse() * truth_at_scan.back().isometry()).inverse() * (estimate_base.inverse() * trajectory.back().isometry());
        for (std::size_t i = offset + kWarmupFrames + 1; i < truth_at_scan.size(); ++i)
        {
            outcome.travelled += (truth_at_scan[i].position - truth_at_scan[i - 1].position).norm();
        }
        outcome.drift_percent = outcome.travelled > 0.0 ? 100.0 * final_error.translation().norm() / outcome.travelled : 0.0;
    }
    return outcome;
}

void TestOdometryPipeline()
{
    std::printf("\n[3] LioOdometer -- IMU preintegration -> initial_guess -> ICP\n");

    const int expected_frames = 38;  // initialization consumes up to 2 of the 40 scans

    for (const NormalMethod method : {NormalMethod::kLoamCurvature, NormalMethod::kNeighborhoodPca})
    {
        const PipelineOutcome outcome = RunPipeline(method, true);
        std::printf(
                "     -- %-16s: frames %2zu | travelled %.1f m | ATE max %.3f m | drift %.2f %% | "
                "RPE %.4f m / %.3f deg\n",
                ToString(method).c_str(), outcome.frames, outcome.travelled, outcome.max_position_error, outcome.drift_percent,
                outcome.max_relative_position_error, outcome.max_relative_rotation_error * 180.0 / M_PI);
        Check(outcome.frames >= static_cast<std::size_t>(expected_frames), ToString(method) + ": every scan was processed");
        Check(outcome.travelled > 5.0, ToString(method) + ": the synthetic trajectory moves far enough");
        Check(outcome.max_relative_position_error < 0.03, ToString(method) + ": frame-to-frame position error < 3 cm");
        Check(outcome.drift_percent < 0.5, ToString(method) + ": drift < 0.5 % of distance travelled");
        Check(outcome.max_rotation_error < 0.03, ToString(method) + ": attitude error < 1.7 deg");
    }

    // Effect of deskewing. Here the velocity and turn rate are constant, so the
    // distortion pattern repeats every scan and largely cancels even without
    // correction; real driving keeps changing, so it does not. The accuracy of
    // the deskew expression itself is checked point-wise in [2b].
    const PipelineOutcome with_deskew = RunPipeline(NormalMethod::kNeighborhoodPca, true);
    const PipelineOutcome without_deskew = RunPipeline(NormalMethod::kNeighborhoodPca, false);
    std::printf("     -- deskew on drift %.2f %% (RPE %.4f m) vs off %.2f %% (RPE %.4f m)\n", with_deskew.drift_percent,
                with_deskew.max_relative_position_error, without_deskew.drift_percent, without_deskew.max_relative_position_error);
    Check(with_deskew.drift_percent < 0.5, "deskew on: drift < 0.5 % of distance travelled");
}

}  // namespace

int main()
{
    std::printf("=== lidar_inertial_odometer core test ===\n");
    TestPreintegration();
    TestFeatureExtraction();
    TestDeskew();
    TestOdometryPipeline();

    std::printf("\n=== %s (%d failure%s) ===\n", g_failures == 0 ? "PASSED" : "FAILED", g_failures, g_failures == 1 ? "" : "s");
    return g_failures == 0 ? 0 : 1;
}
