#pragma once

#include <Eigen/Geometry>
#include <cstdint>
#include <deque>
#include <vector>

#include "lidar_inertial_odometer/feature_extractor.hpp"
#include "lidar_inertial_odometer/imu_preintegrator.hpp"
#include "lidar_inertial_odometer/local_map.hpp"
#include "lidar_inertial_odometer/util.hpp"
#include "p2p_icp/icp_point_to_plane.hpp"

/**
 * @brief LiDAR-Inertial Odometry core
 *
 *   IMU 100 Hz --> ImuPreintegrator --> predicted pose T_pred
 *                                          |
 *                                     initial_guess
 *                                          v
 *   LiDAR 10 Hz --> deskew --> FeatureExtractor --> IcpPointToPlane(local map)
 *                                          |
 *                                       T_icp --> state update
 *
 * There is zero ROS dependency here. The node only calls AddImu() / AddLidarScan() from its
 * callbacks and drains the results with PopResults().
 *
 * Both inputs are queues. Every sample and scan handed in is appended to its queue, and each call
 * then processes whatever has become ready. A scan is ready once the IMU queue reaches the end of
 * that scan's sweep, because deskewing needs the motion across the whole sweep; that IMU arrives
 * around the time of the next scan, which is the one frame of latency seen when replaying a rosbag.
 *
 * @see LioOptions / LioFrameResult (util.hpp)
 */
class LioOdometer
{
public:
    LioOdometer() = default;
    explicit LioOdometer(const LioOptions& options);

    /**
     * @brief Replaces the options
     *
     * @param options new options
     */
    void set_options(const LioOptions& options)
    {
        options_ = options;
    }

    /**
     * @brief Current options
     *
     * @return the options in use
     */
    const LioOptions& options() const
    {
        return options_;
    }

    /**
     * @brief Access to the internal FeatureExtractor, for configuring it
     *
     * @return the internal FeatureExtractor
     */
    FeatureExtractor& feature_extractor()
    {
        return feature_extractor_;
    }

    /**
     * @brief Access to the internal ICP instance, for configuring it
     *
     * @return the internal ICP instance
     */
    p2p_icp::IcpPointToPlane& icp()
    {
        return icp_;
    }

    /**
     * @brief Access to the internal LocalMap, for configuring it
     *
     * @return the internal LocalMap
     */
    LocalMap& local_map()
    {
        return local_map_;
    }

    /**
     * @brief Called from the IMU callback: queues one sample, then processes any scan it made ready
     *
     * Samples are assumed to arrive in time order.
     *
     * @param sample one IMU measurement
     */
    void AddImu(const ImuSample& sample);

    /**
     * @brief Called from the LiDAR callback: queues one scan, then processes any scan already ready
     *
     * @param timestamp scan start time [s]
     * @param points    raw points in the lidar frame
     */
    void AddLidarScan(double timestamp, const std::vector<RawLidarPoint>& points);

    /**
     * @brief Drains the finished frames, removing them from the internal queue
     *
     * @return the completed frame results
     */
    std::vector<LioFrameResult> PopResults();

    /**
     * @brief Current estimated state
     *
     * @return the current state (world <- imu)
     */
    const NavState& state() const
    {
        return state_;
    }

    /**
     * @brief Whether initialization (gravity alignment) has finished
     *
     * @return true once it has
     */
    bool initialized() const
    {
        return initialized_;
    }

    /**
     * @brief Sets the external initial yaw and enables its use
     *
     * @param yaw initial yaw [rad], about the world z axis
     */
    void set_initial_yaw(double yaw)
    {
        options_.use_external_initial_yaw = true;
        options_.initial_yaw = yaw;
    }

    /**
     * @brief Estimated trajectory (world frame IMU states), for comparison against ground truth
     *
     * @return the per-frame state sequence
     */
    const std::vector<NavState>& trajectory() const
    {
        return trajectory_;
    }

private:
    /// One scan waiting in the queue.
    struct QueuedScan
    {
        double timestamp = 0.0;
        std::vector<RawLidarPoint> points;
    };

    /**
     * @brief Processes scans from the front of the queue for as long as the IMU covers their sweep
     */
    void ProcessReadyScans();

    /**
     * @brief Full per-scan pipeline (predict -> deskew -> features -> ICP -> update -> keyframe)
     *
     * @param scan the scan to process
     */
    void ProcessScan(const QueuedScan& scan);

    /**
     * @brief Drops IMU samples the odometer has already consumed, keeping one for interpolation
     *
     * @param keep_from keep samples from this time onwards [s]
     */
    void TrimImuQueue(double keep_from);

    /**
     * @brief Runs gravity alignment and map initialization on the first scan
     *
     * @param timestamp time of the first scan [s]
     * @return false when there are not enough IMU samples yet
     */
    bool Initialize(double timestamp);

    /**
     * @brief Integrates the IMU over [t0, t1] into @p integrator
     *
     * dt is clipped at the interval bounds and the measurement is linearly interpolated at the
     * midpoint of the clipped span (midpoint rule).
     *
     * @param integrator target integrator
     * @param t0 interval start [s]
     * @param t1 interval end [s]
     */
    void IntegrateInterval(ImuPreintegrator* integrator, double t0, double t1) const;

    /**
     * @brief Whether the IMU queue is filled up to time @p t
     *
     * @param t time to check [s]
     * @return true when it is covered
     */
    bool ImuCovers(double t) const;

    /**
     * @brief Corrects the motion distortion caused by the per-point capture times
     *
     * Every point is moved back into the lidar frame at the scan *start* time.
     *
     * @param points preprocessed scan
     * @param state_at_scan_start state (R_i, v_i) at the scan start
     * @param scan_integrator preintegrator covering the scan interval alone
     * @return the deskewed points
     */
    std::vector<RawLidarPoint> Deskew(const std::vector<RawLidarPoint>& points, const NavState& state_at_scan_start,
                                      const ImuPreintegrator& scan_integrator) const;

    LioOptions options_;
    FeatureExtractor feature_extractor_;
    p2p_icp::IcpPointToPlane icp_;  ///< the point-to-plane-icp implementation, reused as is
    LocalMap local_map_;
    ImuPreintegrator preintegrator_;

    /// IMU samples not yet consumed, oldest first.
    std::deque<ImuSample> imu_queue_;
    /// Scans waiting for the IMU to reach the end of their sweep, oldest first.
    std::deque<QueuedScan> scan_queue_;
    std::vector<LioFrameResult> results_;

    bool initialized_ = false;
    /// The initial velocity is 0, so the first ICP correction is taken whole, with unit gain.
    bool velocity_initialized_ = false;
    NavState state_;

    Eigen::Isometry3d last_keyframe_pose_ = Eigen::Isometry3d::Identity();
    bool has_keyframe_ = false;
    /// The local map only changes at keyframes; the ICP kd-tree is rebuilt only when this version
    /// differs from icp_target_version_.
    std::uint64_t map_version_ = 0;
    std::uint64_t icp_target_version_ = 0;

    std::vector<NavState> trajectory_;
};
