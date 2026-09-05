#include "lidar_inertial_odometer/lio_odometer.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace
{
/**
 * @brief Restores orthogonality lost to numerical drift, by re-projecting through a quaternion
 *
 * @param rotation a nearly orthogonal 3x3 matrix
 * @return an exactly orthogonal rotation matrix
 */
Eigen::Matrix3d Orthonormalize(const Eigen::Matrix3d& rotation)
{
    return Eigen::Quaterniond(rotation).normalized().toRotationMatrix();
}

/**
 * @brief Clips a scalar to [low, high]
 *
 * @param value input value
 * @param low   lower bound
 * @param high  upper bound
 * @return the clipped value
 */
double Clamp(double value, double low, double high)
{
    if (value < low)
    {
        return low;
    }
    if (value > high)
    {
        return high;
    }
    return value;
}

/**
 * @brief Applies a rigid transform to a cloud, rotating the normals only
 *
 * @param cloud input cloud
 * @param pose  transform to apply
 * @return the transformed cloud
 */
common::PointCloud TransformCloud(const common::PointCloud& cloud, const Eigen::Isometry3d& pose)
{
    common::PointCloud transformed;
    transformed.reserve(cloud.size());
    for (const common::PointNormal& item : cloud)
    {
        common::PointNormal moved;
        moved.point = pose * item.point;
        moved.normal = pose.linear() * item.normal;  // a normal is a direction, so it is only rotated
        transformed.push_back(moved);
    }
    return transformed;
}

}  // namespace

LioOdometer::LioOdometer(const LioOptions& options) : options_(options)
{
}

void LioOdometer::AddImu(const ImuSample& sample)
{
    if (!imu_queue_.empty() && sample.timestamp <= imu_queue_.back().timestamp)
    {
        return;  // drop samples that go back in time
    }
    imu_queue_.push_back(sample);
    // This sample may be the one that completes the sweep of the scan at the head of the queue.
    ProcessReadyScans();
}

void LioOdometer::AddLidarScan(double timestamp, const std::vector<RawLidarPoint>& points)
{
    if (points.empty())
    {
        return;
    }
    scan_queue_.push_back(QueuedScan{timestamp, points});
    // Bound the queue so a stalled IMU stream cannot grow it without limit.
    while (scan_queue_.size() > 50)
    {
        scan_queue_.pop_front();
    }
    ProcessReadyScans();
}

std::vector<LioFrameResult> LioOdometer::PopResults()
{
    std::vector<LioFrameResult> popped;
    popped.swap(results_);
    return popped;
}

bool LioOdometer::ImuCovers(double t) const
{
    return !imu_queue_.empty() && imu_queue_.back().timestamp >= t;
}

void LioOdometer::TrimImuQueue(double keep_from)
{
    // Keep one sample just before keep_from so interpolation still works.
    // Everything older has already been integrated and will never be read again.
    while (imu_queue_.size() > 2 && imu_queue_[1].timestamp < keep_from)
    {
        imu_queue_.pop_front();
    }
}

void LioOdometer::ProcessReadyScans()
{
    while (!scan_queue_.empty())
    {
        const QueuedScan& scan = scan_queue_.front();

        // A scan is ready only once the IMU queue reaches the end of its sweep, since deskewing
        // needs the motion across the whole sweep. That IMU arrives around the time of the next
        // scan, which is the one frame of latency seen when replaying a rosbag.
        // The queue is time ordered, so if the head is not ready, nothing behind it is either.
        const double sweep_end = scan.timestamp + options_.scan_period;
        if (!ImuCovers(sweep_end))
        {
            break;
        }

        if (!initialized_)
        {
            if (!Initialize(scan.timestamp))
            {
                // Not enough IMU for gravity alignment yet, so this scan is unusable.
                scan_queue_.pop_front();
                continue;
            }
        }

        ProcessScan(scan);
        scan_queue_.pop_front();
        TrimImuQueue(state_.timestamp);
    }
}

bool LioOdometer::Initialize(double timestamp)
{
    // Collect the IMU samples older than timestamp.
    std::vector<const ImuSample*> samples;
    for (const ImuSample& sample : imu_queue_)
    {
        if (sample.timestamp > timestamp)
        {
            break;
        }
        samples.push_back(&sample);
    }
    if (static_cast<int>(samples.size()) < options_.init_imu_samples)
    {
        return false;
    }
    // Use only the most recent N samples.
    const std::size_t count = static_cast<std::size_t>(options_.init_imu_samples);
    samples.erase(samples.begin(), samples.end() - count);

    Eigen::Vector3d mean_accel = Eigen::Vector3d::Zero();
    for (const ImuSample* sample : samples)
    {
        mean_accel += sample->linear_acceleration;
    }
    mean_accel /= static_cast<double>(count);

    if (mean_accel.norm() < 1.0)
    {
        return false;  // not even gravity is being measured
    }

    // --- gravity alignment ---------------------------------------------------
    // At rest the accelerometer measures the specific force
    //   f = R_wi^T (a_w - g_w) = -R_wi^T g_w
    // that is, "up" in the body frame. The initial attitude is the *minimal rotation* taking that
    // direction to world +z. This fixes roll and pitch only; yaw is unobservable and stays 0 unless
    // it is supplied externally.
    const Eigen::Quaterniond q = Eigen::Quaterniond::FromTwoVectors(mean_accel.normalized(), Eigen::Vector3d::UnitZ());
    Eigen::Matrix3d initial_rotation = q.toRotationMatrix();
    if (options_.use_external_initial_yaw)
    {
        // Gravity only pins roll and pitch. Fixing yaw from an external heading (KITTI OXTS) makes
        // the world frame ENU, matching the frame of the GPS ground truth.
        initial_rotation = Eigen::AngleAxisd(options_.initial_yaw, Eigen::Vector3d::UnitZ()).toRotationMatrix() * initial_rotation;
    }

    state_ = imu_preint::NavState{};
    state_.timestamp = timestamp;
    state_.rotation = Orthonormalize(initial_rotation);
    state_.position.setZero();
    state_.velocity.setZero();

    initialized_ = true;
    velocity_initialized_ = false;
    has_keyframe_ = false;
    trajectory_.clear();

    if (options_.verbose)
    {
        std::printf("[lio] initialized at t=%.6f | |a|=%.3f m/s^2\n", timestamp, mean_accel.norm());
    }
    return true;
}

void LioOdometer::IntegrateInterval(imu_preint::ImuPreintegrator* integrator, double t0, double t1) const
{
    if (t1 <= t0 || imu_queue_.size() < 2)
    {
        return;
    }
    for (std::size_t i = 0; i + 1 < imu_queue_.size(); ++i)
    {
        const ImuSample& first = imu_queue_[i];
        const ImuSample& second = imu_queue_[i + 1];
        if (second.timestamp <= t0)
        {
            continue;
        }
        if (first.timestamp >= t1)
        {
            break;
        }

        // Use only the part overlapping [t0, t1]; a keyframe boundary can fall between IMU samples.
        const double begin = std::max(first.timestamp, t0);
        const double end = std::min(second.timestamp, t1);
        const double dt = end - begin;
        if (dt <= 1e-9)
        {
            continue;
        }

        // Interpolate the measurement at the midpoint of the clipped span (more accurate than Euler).
        const double span = second.timestamp - first.timestamp;
        double ratio = 0.0;
        if (span > 1e-9)
        {
            ratio = (0.5 * (begin + end) - first.timestamp) / span;
            ratio = Clamp(ratio, 0.0, 1.0);
        }
        const Eigen::Vector3d gyro = first.angular_velocity + ratio * (second.angular_velocity - first.angular_velocity);
        const Eigen::Vector3d accel = first.linear_acceleration + ratio * (second.linear_acceleration - first.linear_acceleration);

        integrator->integrate(gyro, accel, dt);
    }
}

std::vector<RawLidarPoint> LioOdometer::Deskew(const std::vector<RawLidarPoint>& points, const imu_preint::NavState& state_at_scan_start,
                                               const imu_preint::ImuPreintegrator& scan_integrator) const
{
    // Move every point back into the lidar frame at the scan start time i:
    //
    //   p_ref = T_li * [ dR(tau)*(T_il*p_L) + dp(tau) + R_i^T*(v_i*tau + 0.5*g*tau^2) ]
    //
    // The last term is the key one. dp is by definition free of gravity and initial velocity, so
    // recovering the displacement the vehicle really travelled means adding both back in.
    const Eigen::Isometry3d& T_il = options_.T_imu_lidar;
    const Eigen::Isometry3d T_li = T_il.inverse();
    const Eigen::Matrix3d Ri_transpose = state_at_scan_start.rotation.transpose();
    const Eigen::Vector3d& velocity = state_at_scan_start.velocity;
    const Eigen::Vector3d& gravity = options_.gravity;

    std::vector<RawLidarPoint> deskewed;
    deskewed.reserve(points.size());
    for (const RawLidarPoint& point : points)
    {
        const double tau = Clamp(point.rel_time, 0.0, options_.scan_period);
        const imu_preint::PreintegratedDelta delta = scan_integrator.delta_at(tau);

        const Eigen::Vector3d point_imu = T_il * point.position;
        const Eigen::Vector3d point_ref = delta.rotation * point_imu + delta.position + Ri_transpose * (velocity * tau + 0.5 * gravity * tau * tau);

        RawLidarPoint moved = point;
        moved.position = T_li * point_ref;
        deskewed.push_back(moved);
    }
    return deskewed;
}

void LioOdometer::ProcessScan(const QueuedScan& scan)
{
    LioFrameResult result;
    result.timestamp = scan.timestamp;

    // --- 1. IMU preintegration -> predicted pose (predict()) ---------------------
    // The prediction serves two purposes: the ICP initial_guess and the deskew reference state.
    preintegrator_.reset();
    IntegrateInterval(&preintegrator_, state_.timestamp, scan.timestamp);
    const imu_preint::NavState predicted = preintegrator_.predict(state_, options_.gravity);
    result.imu_prediction = predicted.isometry();

    // --- 2. integrate the scan interval separately, for deskewing ----------------
    // preintegrator_ above covers [last state, scan start]; this one covers [scan start, scan end].
    std::vector<RawLidarPoint> preprocessed = feature_extractor_.Preprocess(scan.points);
    std::vector<RawLidarPoint> corrected;
    if (options_.enable_deskew)
    {
        imu_preint::ImuPreintegrator scan_integrator;
        IntegrateInterval(&scan_integrator, scan.timestamp, scan.timestamp + options_.scan_period);
        corrected = Deskew(preprocessed, predicted, scan_integrator);
    }
    else
    {
        corrected = std::move(preprocessed);
    }
    if (options_.keep_deskewed_scan)
    {
        result.scan_lidar = corrected;
    }

    // --- 3. planar features + normal estimation ----------------------------------
    // KITTI clouds carry no normals, yet point-to-plane ICP absolutely needs target normals.
    // The normals built here ride into the map when this scan becomes a keyframe, and then serve as
    // the target of the next frame.
    const FeatureCloud features = feature_extractor_.Extract(corrected);
    result.num_features = static_cast<int>(features.planar.size());
    result.num_map_points = static_cast<int>(local_map_.cloud().size());

    // --- 4. scan-to-map point-to-plane ICP ---------------------------------------
    // source = this scan's planar features (lidar frame), target = local map (world frame),
    // so the transform ICP returns is exactly the world <- lidar pose.
    const Eigen::Isometry3d lidar_pose_prediction = predicted.isometry() * options_.T_imu_lidar;
    Eigen::Isometry3d lidar_pose = lidar_pose_prediction;
    bool accepted = false;

    if (!local_map_.empty() && static_cast<int>(features.planar.size()) >= options_.min_icp_correspondences)
    {
        icp_.set_source(features.planar);
        if (icp_target_version_ != map_version_)
        {
            icp_.set_target(local_map_.cloud());  // rebuild the kd-tree only when the map changed
            icp_target_version_ = map_version_;
        }

        const p2p_icp::IcpResult icp_result = icp_.do_icp(lidar_pose_prediction);
        result.icp_iterations = icp_result.iterations;
        result.icp_correspondences = icp_result.correspondences;
        result.icp_error = icp_result.final_error;

        // Divergence gate: an ICP result too far from the IMU prediction is dropped for the prediction.
        const Eigen::Isometry3d deviation = lidar_pose_prediction.inverse() * icp_result.transform;
        const double translation_deviation = deviation.translation().norm();
        const double rotation_deviation = Eigen::AngleAxisd(deviation.linear()).angle();

        accepted = std::isfinite(icp_result.final_error) && icp_result.correspondences >= options_.min_icp_correspondences &&
                   translation_deviation <= options_.max_icp_translation_deviation && rotation_deviation <= options_.max_icp_rotation_deviation;
        if (accepted)
        {
            lidar_pose = icp_result.transform;
        }
    }
    result.icp_accepted = accepted;

    // --- 5. state update ----------------------------------------------------------
    imu_preint::NavState updated;
    updated.timestamp = scan.timestamp;
    const Eigen::Isometry3d body_pose = lidar_pose * options_.T_imu_lidar.inverse();
    updated.rotation = Orthonormalize(body_pose.linear());
    updated.position = body_pose.translation();
    updated.velocity = predicted.velocity;

    const double dt = preintegrator_.delta_t();
    const imu_preint::PreintegratedDelta delta = preintegrator_.delta();

    if (accepted && dt > 1e-6)
    {
        // --- 5a. ICP -> IMU feedback: velocity correction ----------------------
        // Position residual:
        //   r_dp = R_i^T (p_j - p_i - v_i dt - 0.5 g dt^2) - dp_ij
        const Eigen::Vector3d position_residual =
                state_.rotation.transpose() * (updated.position - state_.position - state_.velocity * dt - 0.5 * options_.gravity * dt * dt) - delta.position;

        // Since d(r_dp)/d(v_i) = -R_i^T dt, the naive update would be dv_i = R_i*r_dp / dt.
        //
        // Deskewing adds a second feedback path, though. It removes R_i^T*v*tau per point, so a
        // velocity error of dv shifts the whole cloud by dv*(T/2) on average (tau is spread evenly
        // over [0, T]), and the displacement ICP reports shrinks by the same amount. The residual's
        // lever arm is therefore (dt + T/2), not dt:
        //     r_dp ~ -dv * (dt + T/2)
        // Using dt as the lever inflates the effective gain by 1 + T/(2*dt) -- 1.5x on KITTI, where
        // dt = T = 0.1 s -- and the velocity then oscillates from frame to frame.
        double lever = dt;
        if (options_.enable_deskew)
        {
            lever += 0.5 * options_.scan_period;
        }
        // The first correction is taken whole, with unit gain. Initialization sets the velocity to 0
        // without knowing whether the vehicle is moving, so a low gain would take several frames to
        // converge and the deskew term would stay wrong throughout.
        double gain = 1.0;
        if (velocity_initialized_)
        {
            gain = options_.velocity_correction_gain;
        }
        velocity_initialized_ = true;
        const Eigen::Vector3d velocity_i = state_.velocity + gain * (state_.rotation * position_residual) / lever;

        // Propagate v_j again from the corrected v_i, following the dv definition.
        updated.velocity = velocity_i + options_.gravity * dt + state_.rotation * delta.velocity;
    }

    state_ = updated;
    trajectory_.push_back(updated);

    // --- 6. keyframe decision and local map update --------------------------------
    // A scan enters the map only after moving far enough from the last keyframe, and the ICP
    // kd-tree is rebuilt only when the map changes.
    const Eigen::Isometry3d current_pose = updated.isometry();
    bool is_keyframe = !has_keyframe_;
    if (has_keyframe_)
    {
        const Eigen::Isometry3d relative = last_keyframe_pose_.inverse() * current_pose;
        is_keyframe =
                relative.translation().norm() > options_.keyframe_translation || Eigen::AngleAxisd(relative.linear()).angle() > options_.keyframe_rotation;
    }

    const Eigen::Isometry3d world_from_lidar = current_pose * options_.T_imu_lidar;
    result.feature_cloud_world = TransformCloud(features.planar, world_from_lidar);

    if (is_keyframe && !features.planar.empty())
    {
        local_map_.AddKeyframe(result.feature_cloud_world, updated.position);
        last_keyframe_pose_ = current_pose;
        has_keyframe_ = true;
        ++map_version_;
    }

    result.valid = true;
    result.state = updated;
    result.lidar_pose = world_from_lidar;
    result.is_keyframe = is_keyframe;

    if (options_.verbose)
    {
        const char* icp_state = "REJ";
        if (accepted)
        {
            icp_state = "ok ";
        }

        const char* keyframe_mark = "";
        if (is_keyframe)
        {
            keyframe_mark = " [KF]";
        }

        std::printf(
                "[lio] t=%.3f | feat %4d/%4d | map %6d | icp %s iter %2d corr %5d rms %.4f | "
                "p=(%8.2f %8.2f %6.2f) v=%.2f m/s%s\n",
                scan.timestamp, result.num_features, features.num_candidates, result.num_map_points, icp_state, result.icp_iterations,
                result.icp_correspondences, result.icp_error, updated.position.x(), updated.position.y(), updated.position.z(),
                updated.velocity.norm(), keyframe_mark);
    }

    results_.push_back(std::move(result));
}
