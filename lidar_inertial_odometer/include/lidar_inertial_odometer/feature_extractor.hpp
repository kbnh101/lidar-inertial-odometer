#pragma once

#include <Eigen/Core>
#include <vector>

#include "lidar_inertial_odometer/kdtree_radius.hpp"
#include "lidar_inertial_odometer/util.hpp"

/**
 * @brief Scan preprocessing, planar feature selection and normal estimation
 *
 * There is no ROS dependency here; the node converts PointCloud2 into RawLidarPoint and passes it in.
 *
 * @see FeatureExtractorOptions / FeatureCloud / NormalMethod / RingSelection (util.hpp)
 */
class FeatureExtractor
{
public:
    FeatureExtractor() = default;
    explicit FeatureExtractor(const FeatureExtractorOptions& options) : options_(options)
    {
    }

    /**
     * @brief Replaces the options
     *
     * @param options new options
     */
    void set_options(const FeatureExtractorOptions& options)
    {
        options_ = options;
    }

    /**
     * @brief Current options, read only
     *
     * @return the options in use
     */
    const FeatureExtractorOptions& options() const
    {
        return options_;
    }

    /**
     * @brief Accessor for editing the options in place
     *
     * @return the options in use, mutable
     */
    FeatureExtractorOptions& mutable_options()
    {
        return options_;
    }

    /**
     * @brief Step 1: channel selection (64 -> 32ch), range/height filtering, ring and rel_time recovery
     *
     * A bag without a ring field gets it from the vertical angle, a missing rel_time from the azimuth.
     * The result is sorted by ring, as the curvature computation requires.
     *
     * @param raw the raw scan
     * @return the preprocessed points, with ring and rel_time both filled in
     */
    std::vector<RawLidarPoint> Preprocess(const std::vector<RawLidarPoint>& raw) const;

    /**
     * @brief Whether this channel survives preprocessing, also used by the node's startup log
     *
     * @param ring channel index
     * @return true when the channel is used
     */
    bool IsRingSelected(int ring) const;

    /**
     * @brief Step 2: planar feature selection and PCA normal estimation
     *
     * The input must have gone through Preprocess() (and deskewing, when it is enabled).
     *
     * @param points the preprocessed points
     * @return planar points with normals plus the count statistics
     */
    FeatureCloud Extract(const std::vector<RawLidarPoint>& points) const;

private:
    /**
     * @brief Selects planar candidates by LOAM curvature (NormalMethod::kLoamCurvature)
     *
     * @param points preprocessed points, sorted by ring
     * @return indices of the candidate points
     */
    std::vector<int> SelectPlanarByCurvature(const std::vector<RawLidarPoint>& points) const;

    /**
     * @brief Fits a plane by PCA over the k-NN of @p query to obtain a normal
     *
     * @param tree    k-NN tree built over the support cloud
     * @param support the cloud the neighbours come from (voxel downsampled)
     * @param query   the point whose normal is wanted
     * @param use_planarity_test whether to apply the min_planarity test (meaningful for method B only)
     * @param normal  output: unit normal, flipped towards the sensor
     * @return true when the normal is usable
     */
    bool EstimateNormal(const KdTreeRadius& tree, const std::vector<Eigen::Vector3d>& support, const Eigen::Vector3d& query, bool use_planarity_test,
                        Eigen::Vector3d* normal) const;

    FeatureExtractorOptions options_;
};
