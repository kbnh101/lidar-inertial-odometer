#pragma once

#include <Eigen/Core>
#include <cstddef>
#include <vector>

namespace common
{
/**
 * @brief 3-D kd-tree for radius neighbour search
 *
 * Fitting a normal by PCA needs *every* point inside a fixed radius around the query, and no
 * external point cloud library is used here, so the tree is implemented directly.
 *
 * A fixed radius keeps the physical size of the neighbourhood patch constant. Unlike k-nearest
 * search, whose patch size swings with the point density, the plane fit residual sqrt(l0) then means
 * the same thing everywhere and a threshold such as `max_plane_rms` works independently of range.
 *
 * Splitting at the median of the widest axis makes the build O(n log n).
 */
class KdTreeRadius
{
public:
    /**
     * @brief Builds the tree from an array of points
     *
     * @param points the points to store
     */
    void build(const std::vector<Eigen::Vector3d>& points);

    /**
     * @brief Empties the tree
     */
    void clear();

    /**
     * @brief Finds every point within the radius around @p query
     *
     * @param query the query point
     * @param max_squared_distance squared radius [m^2]
     * @param max_count stops the search once this many are collected (not positive means no limit);
     *                  the near child is visited first, so a truncated result is mostly near points
     * @param indices output: indices of the points found, not sorted by distance
     *                (the covariance does not care about the order)
     * @return the number of points found
     */
    int radius_search(const Eigen::Vector3d& query, double max_squared_distance, int max_count, std::vector<int>* indices) const;

    bool empty() const
    {
        return points_.empty();
    }
    std::size_t size() const
    {
        return points_.size();
    }
    const std::vector<Eigen::Vector3d>& points() const
    {
        return points_;
    }

private:
    struct Node
    {
        int index = -1;  ///< index into points_
        int axis = 0;  ///< axis this node splits on (0=x, 1=y, 2=z)
        int left = -1;
        int right = -1;
    };

    int build_recursive(int begin, int end);
    void radius_recursive(int node, const Eigen::Vector3d& query, double max_squared_distance, int max_count, std::vector<int>* indices) const;

    std::vector<Eigen::Vector3d> points_;
    std::vector<int> indices_;
    std::vector<Node> nodes_;
    int root_ = -1;
};

}  // namespace common
