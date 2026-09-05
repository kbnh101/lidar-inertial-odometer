#pragma once

// Shared data types for the whole package.
//
// Class headers hold classes only; the plain-data enum classes / structs and their conversion
// helpers all live here, so opening a header immediately shows where a class starts and ends.
//
//   1. Sensor input                        ImuSample / RawLidarPoint
//   2. Feature extraction                  NormalMethod / RingSelection / FeatureExtractorOptions / FeatureCloud
//   3. Local map                           LocalMapOptions
//   4. Odometry                            LioOptions / LioFrameResult
//
// The navigation state and the preintegrated measurement live in the imu-preintegration package,
// as imu_preint::NavState and imu_preint::PreintegratedDelta.

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <stdexcept>
#include <string>
#include <vector>

#include "imu_preint/imu_preintegrator.hpp"
#include "common/point_cloud.hpp"

// ===========================================================================
// 1. Sensor input
// ===========================================================================

/// One IMU measurement; the node converts sensor_msgs/Imu into this type.
struct ImuSample
{
    double timestamp = 0.0;  ///< [s]
    Eigen::Vector3d angular_velocity = Eigen::Vector3d::Zero();  ///< w [rad/s], IMU frame
    Eigen::Vector3d linear_acceleration = Eigen::Vector3d::Zero();  ///< a [m/s^2], IMU frame
};

/// One LiDAR point; the node converts PointCloud2 into this type.
struct RawLidarPoint
{
    Eigen::Vector3d position = Eigen::Vector3d::Zero();  ///< lidar frame
    int ring = -1;  ///< channel index, -1 means "recover it from the vertical angle"
    double rel_time = -1.0;  ///< capture time relative to the scan start [s], -1 means "recover it from the azimuth"
};

// ===========================================================================
// 2. Feature extraction -- planar point selection and normal estimation
// ===========================================================================

/// How planar points are selected and their normals estimated.
/// KITTI clouds carry no normals, yet point-to-plane ICP needs one per target point.
enum class NormalMethod
{
    /// (A) LOAM: pick planar candidates by scan-line curvature and fit a normal by neighbourhood
    ///     PCA on those candidates only. Few candidates make it fast and it rejects edges along a
    ///     scan line well, but curvature only sees 1-D smoothness, so wires or pole sides can pass.
    kLoamCurvature,

    /// (B) Neighbourhood covariance PCA (default): no curvature pre-selection, every point is
    ///     judged directly by the k-NN covariance eigenvalues (l0 <= l1 <= l2) -- accepted when the
    ///     plane residual sqrt(l0) is small and l0/l1 is small (the normal is unique). It measures
    ///     the local 3-D shape directly and so rejects linear or scattered structure reliably, but
    ///     needs a k-NN query per point and is slower than (A).
    kNeighborhoodPca,
};

/// How a 64-channel KITTI scan is reduced to roughly 32 channels.
/// Either way, channels outside [ring_min, ring_max] are dropped first.
///
/// The ring index of this bag (`/points_raw`) increases from bottom to top:
/// ring 2 -> -23.6 deg, ring 33 -> -10.8 deg, ring 63 -> +2.6 deg.
enum class RingSelection
{
    /// Use every channel inside [ring_min, ring_max]; the band width sets the channel count.
    kAll,
    /// Inside the same band, use only channels with (ring - ring_min) % ring_stride == 0.
    /// A stride of 2 keeps the band and halves the vertical resolution uniformly.
    kStride,
};

/**
 * @brief Converts a string parameter into a NormalMethod
 *
 * @param name "loam_curvature"/"loam"/"A" or "neighborhood_pca"/"pca"/"B"
 * @return the matching NormalMethod
 * @throws std::runtime_error on an unknown name
 */
inline NormalMethod ParseNormalMethod(const std::string& name)
{
    if (name == "loam_curvature" || name == "loam" || name == "A")
    {
        return NormalMethod::kLoamCurvature;
    }
    if (name == "neighborhood_pca" || name == "pca" || name == "B")
    {
        return NormalMethod::kNeighborhoodPca;
    }
    throw std::runtime_error("unknown normal_method '" + name + "' (expected 'loam_curvature' or 'neighborhood_pca')");
}

/**
 * @brief Converts a NormalMethod into a string for logging
 *
 * @param method value to convert
 * @return string representation
 */
inline std::string ToString(NormalMethod method)
{
    if (method == NormalMethod::kLoamCurvature)
    {
        return "loam_curvature";
    }
    return "neighborhood_pca";
}

/**
 * @brief Converts a string parameter into a RingSelection
 *
 * @param name "all"/"none" or "stride"
 * @return the matching RingSelection
 * @throws std::runtime_error on an unknown name
 */
inline RingSelection ParseRingSelection(const std::string& name)
{
    if (name == "stride")
    {
        return RingSelection::kStride;
    }
    if (name == "all" || name == "none")
    {
        return RingSelection::kAll;
    }
    throw std::runtime_error("unknown ring_selection '" + name + "' (expected 'all' or 'stride')");
}

/**
 * @brief Converts a RingSelection into a string for logging
 *
 * @param selection value to convert
 * @return string representation
 */
inline std::string ToString(RingSelection selection)
{
    if (selection == RingSelection::kStride)
    {
        return "stride";
    }
    return "all";
}

/// FeatureExtractor parameters.
struct FeatureExtractorOptions
{
    // --- preprocessing: channel selection (64 -> 32ch) -----------------------
    RingSelection ring_selection = RingSelection::kAll;
    int num_channels = 64;  ///< input channel count (KITTI HDL-64E)
    int ring_min = 16;  ///< lowest channel kept, inclusive
    int ring_max = 47;  ///< highest channel kept, inclusive
    int ring_stride = 2;  ///< used by kStride only
    double vertical_fov_min_deg = -24.8;  ///< HDL-64E lower bound
    double vertical_fov_max_deg = 2.0;  ///< HDL-64E upper bound
    double min_range = 3.0;  ///< drops ego-vehicle returns [m]
    double max_range = 80.0;  ///< drops far-range noise [m]
    double min_z = -3.0;  ///< floor in the lidar frame [m]
    double scan_period = 0.1;  ///< one Velodyne revolution [s]

    // --- (A) LOAM curvature candidate selection ------------------------------
    int curvature_window = 5;  ///< neighbours on one side, as in the LOAM paper
    int num_sectors = 6;  ///< splits a ring to spread the candidates evenly
    int max_planar_per_sector = 40;
    double max_planar_curvature = 0.1;  ///< above this a point is not treated as planar

    // --- PCA normal estimation, shared by both methods -----------------------
    NormalMethod normal_method = NormalMethod::kNeighborhoodPca;
    /// No normal is produced below this many neighbours inside the radius (a plane fit needs 3).
    int normal_min_neighbors = 5;
    /// Stops the search once this many neighbours are collected, capping the cost in the dense
    /// region near the sensor. Not positive means no limit.
    int normal_max_neighbors = 12;
    /// Upper bound on the k-NN radius [m]. Point spacing grows with range (about 0.015*range at
    /// 32ch), so a fixed radius finds no neighbours far away.
    /// The effective radius is max(normal_search_radius, normal_search_radius_scale * range).
    double normal_search_radius = 1.0;
    double normal_search_radius_scale = 0.06;
    double max_plane_rms = 0.06;  ///< upper bound on sqrt(l0) [m] -- is it really a plane
    double max_eigenvalue_ratio = 0.12;  ///< upper bound on l0/l1 -- is the normal direction unique
    /// Lower bound on (l1-l0)/l2, method (B) only. Disabled by default: it assumes isotropic
    /// sampling, but LiDAR patches are dense along a ring and sparse across rings, so l2 >> l1 even
    /// on a healthy plane. Uniqueness of the normal is already covered by l0/l1.
    double min_planarity = 0.0;

    // --- output downsampling -------------------------------------------------
    double voxel_size = 0.4;  ///< support cloud downsample [m], 0 disables it
};

/// Feature extraction result for one scan.
struct FeatureCloud
{
    common::PointCloud planar;  ///< planar points with normals, used as the ICP source/target
    int num_input = 0;  ///< points surviving preprocessing
    int num_candidates = 0;  ///< planar candidates before the normal test
};

// ===========================================================================
// 3. Local map
// ===========================================================================

/// LocalMap parameters.
struct LocalMapOptions
{
    int max_keyframes = 25;  ///< sliding window size
    double voxel_size = 0.4;  ///< map downsample [m], 0 disables it
    double crop_radius = 80.0;  ///< drops points farther than this from the viewpoint [m], 0 disables it
};

// ===========================================================================
// 4. Odometry
// ===========================================================================

/// LioOdometer runtime parameters, filled in by the node from rosparam.
struct LioOptions
{
    // --- frames and constants -------------------------------------------------
    /// T_imu_lidar: moves a point from the lidar frame into the imu (body) frame.
    /// KITTI's calib_imu_to_velo.txt gives T_velo_imu, so this is its inverse.
    Eigen::Isometry3d T_imu_lidar = Eigen::Isometry3d::Identity();
    /// Gravity in the world frame; the world z axis points up, hence (0, 0, -g).
    Eigen::Vector3d gravity = Eigen::Vector3d(0.0, 0.0, -9.80665);

    // --- initialization -------------------------------------------------------
    int init_imu_samples = 100;  ///< samples used for gravity alignment
    /// Gravity alignment cannot observe yaw. When the IMU provides an absolute heading (KITTI
    /// OXTS), use it to align the world frame with ENU and share the GPS trajectory frame.
    bool use_external_initial_yaw = false;
    double initial_yaw = 0.0;  ///< [rad], about the world z axis

    // --- keyframes / local map ------------------------------------------------
    double keyframe_translation = 1.0;  ///< [m]
    double keyframe_rotation = 0.18;  ///< [rad]

    // --- ICP gating -----------------------------------------------------------
    /// Beyond this deviation from the IMU prediction the ICP result is dropped for the prediction.
    double max_icp_translation_deviation = 1.5;  ///< [m]
    double max_icp_rotation_deviation = 0.35;  ///< [rad]
    int min_icp_correspondences = 50;

    // --- ICP -> IMU feedback --------------------------------------------------
    double velocity_correction_gain = 0.9;  ///< [0,1]

    // --- deskew ---------------------------------------------------------------
    bool enable_deskew = true;
    double scan_period = 0.1;  ///< [s], one Velodyne revolution

    /// Whether the deskewed preprocessed scan is kept in the result. Off by default since it is
    /// only for eyeballing the channel selection in rviz and copies tens of thousands of points.
    bool keep_deskewed_scan = false;

    bool verbose = false;
};

/// Result of processing one scan -- only what the node publishes.
struct LioFrameResult
{
    bool valid = false;
    double timestamp = 0.0;
    imu_preint::NavState state;  ///< world <- imu
    Eigen::Isometry3d lidar_pose = Eigen::Isometry3d::Identity();  ///< world <- lidar
    Eigen::Isometry3d imu_prediction = Eigen::Isometry3d::Identity();
    bool icp_accepted = false;
    int icp_iterations = 0;
    int icp_correspondences = 0;
    double icp_error = 0.0;
    int num_features = 0;
    int num_map_points = 0;
    bool is_keyframe = false;
    /// Deskewed planar features moved into the world frame, for visualization.
    common::PointCloud feature_cloud_world;
    /// Deskewed preprocessed scan in the lidar frame, filled only when LioOptions::keep_deskewed_scan.
    /// `ring` is preserved so rviz can colour by channel.
    std::vector<RawLidarPoint> scan_lidar;
};
