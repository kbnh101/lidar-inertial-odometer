#include "lidar_inertial_odometer/feature_extractor.hpp"

#include <Eigen/Eigenvalues>
#include <algorithm>
#include <cmath>
#include <numeric>

#include "lidar_inertial_odometer/voxel_grid.hpp"

namespace
{
const double kTwoPi = 2.0 * M_PI;

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

/// LOAM curvature of one point together with its global index.
struct CurvatureEntry
{
    double curvature = 0.0;
    int point_index = -1;

    CurvatureEntry()
    {
    }

    CurvatureEntry(double curvature_value, int index) : curvature(curvature_value), point_index(index)
    {
    }

    /// Ascending curvature; planar candidates are taken from the low-curvature end.
    bool operator<(const CurvatureEntry& other) const
    {
        return curvature < other.curvature;
    }
};

/// Groups preprocessed points by ring while preserving capture order inside a ring, because the
/// curvature computation needs consecutive neighbours along the same scan line.
struct RingThenTimeLess
{
    bool operator()(const RawLidarPoint& lhs, const RawLidarPoint& rhs) const
    {
        if (lhs.ring != rhs.ring)
        {
            return lhs.ring < rhs.ring;
        }
        return lhs.rel_time < rhs.rel_time;
    }
};

/**
 * @brief Recovers the channel index from the vertical angle, for bags whose PointCloud2 has no ring
 *
 * @param point        point in the lidar frame
 * @param num_channels sensor channel count
 * @param fov_min_deg  lower bound of the vertical FOV [deg]
 * @param fov_max_deg  upper bound of the vertical FOV [deg]
 * @return the channel index, or -1 when it falls outside the range
 */
int RingFromVerticalAngle(const Eigen::Vector3d& point, int num_channels, double fov_min_deg, double fov_max_deg)
{
    const double horizontal = std::hypot(point.x(), point.y());
    if (horizontal < 1e-6)
    {
        return -1;
    }
    const double angle_deg = std::atan2(point.z(), horizontal) * 180.0 / M_PI;
    const double span = fov_max_deg - fov_min_deg;
    if (span <= 0.0)
    {
        return -1;
    }
    const int ring = static_cast<int>(std::lround((angle_deg - fov_min_deg) / span * (num_channels - 1)));
    if (ring < 0 || ring >= num_channels)
    {
        return -1;
    }
    return ring;
}

}  // namespace

bool FeatureExtractor::IsRingSelected(int ring) const
{
    // Drop everything outside the band first; this is the bulk of the 64 -> 32ch reduction.
    if (ring < options_.ring_min || ring > options_.ring_max)
    {
        return false;
    }
    if (options_.ring_selection == RingSelection::kStride)
    {
        return ((ring - options_.ring_min) % std::max(1, options_.ring_stride)) == 0;
    }
    return true;
}

std::vector<RawLidarPoint> FeatureExtractor::Preprocess(const std::vector<RawLidarPoint>& raw) const
{
    std::vector<RawLidarPoint> filtered;
    filtered.reserve(raw.size() / 2);

    const double min_squared = options_.min_range * options_.min_range;
    const double max_squared = options_.max_range * options_.max_range;

    // Azimuth unwrapping state, used when rel_time is missing.
    bool has_previous_azimuth = false;
    double previous_azimuth = 0.0;
    double azimuth_offset = 0.0;
    double start_azimuth = 0.0;

    for (const RawLidarPoint& point : raw)
    {
        const double squared_range = point.position.squaredNorm();
        if (!std::isfinite(squared_range) || squared_range < min_squared || squared_range > max_squared)
        {
            continue;
        }
        if (point.position.z() < options_.min_z)
        {
            continue;
        }

        RawLidarPoint item = point;

        // --- recover the ring, then thin 64 channels down to 32 --------------------
        if (item.ring < 0)
        {
            item.ring = RingFromVerticalAngle(item.position, options_.num_channels, options_.vertical_fov_min_deg, options_.vertical_fov_max_deg);
        }
        if (item.ring < 0)
        {
            continue;
        }
        // The default band [16, 47] keeps the middle 32 channels; see the RingSelection comment.
        if (!IsRingSelected(item.ring))
        {
            continue;
        }

        // --- recover rel_time, which deskewing absolutely requires -----------------
        if (item.rel_time < 0.0)
        {
            // A Velodyne spins clockwise seen from above, so the azimuth decreases; flip the sign
            // to make it monotonically increasing, then unwrap.
            double azimuth = -std::atan2(item.position.y(), item.position.x());
            if (!has_previous_azimuth)
            {
                has_previous_azimuth = true;
                start_azimuth = azimuth;
                previous_azimuth = azimuth;
            }
            else
            {
                // Correct the wrap-around where the angle jumps from -pi to +pi.
                if (azimuth + azimuth_offset < previous_azimuth - M_PI)
                {
                    azimuth_offset += kTwoPi;
                }
                previous_azimuth = azimuth + azimuth_offset;
            }
            azimuth += azimuth_offset;
            const double ratio = (azimuth - start_azimuth) / kTwoPi;
            item.rel_time = Clamp(ratio, 0.0, 1.0) * options_.scan_period;
        }

        filtered.push_back(item);
    }

    // Curvature looks at neighbours within one ring, so group the points by ring.
    std::stable_sort(filtered.begin(), filtered.end(), RingThenTimeLess());

    return filtered;
}

std::vector<int> FeatureExtractor::SelectPlanarByCurvature(const std::vector<RawLidarPoint>& points) const
{
    std::vector<int> selected;
    if (points.empty())
    {
        return selected;
    }

    const int window = std::max(1, options_.curvature_window);
    const int num_points = static_cast<int>(points.size());

    // Sweep ring by ring; Preprocess() already sorted the points that way.
    int begin = 0;
    while (begin < num_points)
    {
        int end = begin;
        while (end < num_points && points[end].ring == points[begin].ring)
        {
            ++end;
        }

        const int ring_size = end - begin;
        if (ring_size >= 2 * window + 1)
        {
            // --- LOAM curvature: c = |sum_{j in S} (p_i - p_j)| / (|S|*|p_i|) ------
            // On a plane the neighbours sit symmetrically and the sum cancels; on an edge it does
            // not. Dividing by |p_i| normalises the point spacing, which grows with range.
            std::vector<CurvatureEntry> curvature;
            curvature.reserve(ring_size);
            for (int i = begin + window; i < end - window; ++i)
            {
                Eigen::Vector3d difference = Eigen::Vector3d::Zero();
                for (int offset = -window; offset <= window; ++offset)
                {
                    if (offset == 0)
                    {
                        continue;
                    }
                    difference += points[i].position - points[i + offset].position;
                }
                const double range = points[i].position.norm();
                if (range < 1e-6)
                {
                    continue;
                }
                const double value = difference.norm() / (2.0 * window * range);
                curvature.push_back(CurvatureEntry(value, i));
            }

            // --- pick the lowest-curvature points sector by sector ------------------
            // Picking over a whole ring at once clusters the features on one side, the normals then
            // fail to span R^3 and observability suffers (the rank condition on sum J^T J).
            const int num_sectors = std::max(1, options_.num_sectors);
            const int sector_size = static_cast<int>(curvature.size()) / num_sectors;
            if (sector_size > 0)
            {
                for (int sector = 0; sector < num_sectors; ++sector)
                {
                    const int sector_begin = sector * sector_size;
                    // The last sector absorbs the remainder.
                    int sector_end = sector_begin + sector_size;
                    if (sector == num_sectors - 1)
                    {
                        sector_end = static_cast<int>(curvature.size());
                    }

                    std::vector<CurvatureEntry> sector_points(curvature.begin() + sector_begin, curvature.begin() + sector_end);
                    std::sort(sector_points.begin(), sector_points.end());

                    int picked = 0;
                    for (std::size_t i = 0; i < sector_points.size(); ++i)
                    {
                        const CurvatureEntry& entry = sector_points[i];

                        if (picked >= options_.max_planar_per_sector)
                        {
                            break;
                        }
                        if (entry.curvature > options_.max_planar_curvature)
                        {
                            break;  // sorted, so everything after this is larger
                        }

                        selected.push_back(entry.point_index);
                        ++picked;
                    }
                }
            }
        }
        begin = end;
    }

    return selected;
}

bool FeatureExtractor::EstimateNormal(const KdTreeRadius& tree, const std::vector<Eigen::Vector3d>& support, const Eigen::Vector3d& query, bool use_planarity_test,
                                      Eigen::Vector3d* normal) const
{
    // Point spacing grows with range, so the search radius grows with range too.
    const double radius = std::max(options_.normal_search_radius, options_.normal_search_radius_scale * query.norm());

    // Collect *every* point inside the radius. A fixed radius makes the neighbourhood patch the
    // same physical size everywhere, so the sqrt(l0) and l0/l1 thresholds below hold at any range.
    std::vector<int> neighbors;
    const int found = tree.radius_search(query, radius * radius, options_.normal_max_neighbors, &neighbors);
    if (found < std::max(3, options_.normal_min_neighbors))
    {
        return false;  // a plane needs at least 3 points
    }

    // --- neighbourhood covariance ---------------------------------------------
    Eigen::Vector3d centroid = Eigen::Vector3d::Zero();
    for (const int index : neighbors)
    {
        centroid += support[index];
    }
    centroid /= static_cast<double>(found);

    Eigen::Matrix3d covariance = Eigen::Matrix3d::Zero();
    for (const int index : neighbors)
    {
        const Eigen::Vector3d centered = support[index] - centroid;
        covariance += centered * centered.transpose();
    }
    covariance /= static_cast<double>(found);

    // --- PCA: the covariance is symmetric, so SelfAdjointEigenSolver (ascending eigenvalues) ---
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(covariance);
    if (solver.info() != Eigen::Success)
    {
        return false;
    }
    const Eigen::Vector3d eigenvalues = solver.eigenvalues();  // l0 <= l1 <= l2
    const double l0 = std::max(eigenvalues(0), 0.0);
    const double l1 = std::max(eigenvalues(1), 0.0);
    const double l2 = std::max(eigenvalues(2), 0.0);
    if (l2 <= 1e-12)
    {
        return false;
    }

    // The plane fit residual RMS is sqrt(smallest eigenvalue); it decides whether this is a plane.
    if (std::sqrt(l0) > options_.max_plane_rms)
    {
        return false;
    }
    // An l0/l1 near 1 means the smallest eigenvector is not unique (the neighbours are collinear or
    // isotropically scattered), so the normal direction itself is undefined and the point is dropped.
    if (l1 <= 1e-12 || l0 / l1 > options_.max_eigenvalue_ratio)
    {
        return false;
    }
    if (use_planarity_test && options_.min_planarity > 0.0)
    {
        // planarity = (l1 - l0)/l2, an isotropic-sampling metric that fits LiDAR data poorly.
        if ((l1 - l0) / l2 < options_.min_planarity)
        {
            return false;
        }
    }

    Eigen::Vector3d candidate = solver.eigenvectors().col(0);
    const double squared_norm = candidate.squaredNorm();
    if (!std::isfinite(squared_norm) || squared_norm <= 1e-12)
    {
        return false;
    }
    candidate.normalize();

    // Flip every normal towards the sensor origin so the signs agree. The point-to-plane residual is
    // signed, so mixed normal signs on one plane split the residual signs and the least squares cancels.
    if (candidate.dot(-query) < 0.0)
    {
        candidate = -candidate;
    }

    *normal = candidate;
    return true;
}

FeatureCloud FeatureExtractor::Extract(const std::vector<RawLidarPoint>& points) const
{
    FeatureCloud result;
    result.num_input = static_cast<int>(points.size());
    if (points.empty())
    {
        return result;
    }

    // --- support cloud used for the PCA neighbour search -----------------------
    // Both methods search the neighbours in a voxel downsampled cloud. The raw scan is extremely
    // dense near the sensor (about 3 cm spacing at 4 m), so k neighbours span a patch under 10 cm and
    // the measurement noise then rivals the patch size, leaving the normal direction undefined.
    // Voxelizing yields a roughly constant patch size regardless of range.
    std::vector<Eigen::Vector3d> raw_positions;
    raw_positions.reserve(points.size());
    for (const RawLidarPoint& point : points)
    {
        raw_positions.push_back(point.position);
    }
    const std::vector<Eigen::Vector3d> support = VoxelDownsample(raw_positions, options_.voxel_size);

    // --- decide which points get a normal --------------------------------------
    std::vector<Eigen::Vector3d> queries;
    bool use_planarity_test = false;

    if (options_.normal_method == NormalMethod::kLoamCurvature)
    {
        // (A) LOAM curvature candidates. Curvature needs the scan-line ordering, so it is computed
        //     on the original points, before downsampling.
        const std::vector<int> candidates = SelectPlanarByCurvature(points);
        queries.reserve(candidates.size());
        for (const int index : candidates)
        {
            queries.push_back(points[index].position);
        }
        // Curvature already filtered for planarity, so the planarity test is skipped; the eigenvalue
        // ratio and residual tests still apply.
        use_planarity_test = false;
    }
    else
    {
        // (B) No curvature pre-selection; every point is judged directly by its covariance.
        queries = support;
        use_planarity_test = true;
    }

    result.num_candidates = static_cast<int>(queries.size());
    if (support.size() < 3 || queries.empty())
    {
        return result;
    }

    KdTreeRadius tree;
    tree.build(support);

    result.planar.reserve(queries.size());
    for (const Eigen::Vector3d& query : queries)
    {
        Eigen::Vector3d normal;
        if (!EstimateNormal(tree, support, query, use_planarity_test, &normal))
        {
            continue;
        }
        p2p_icp::PointNormal item;
        item.point = query;
        item.normal = normal;
        result.planar.push_back(item);
    }
    return result;
}
