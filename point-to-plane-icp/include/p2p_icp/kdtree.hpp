#pragma once

#include <Eigen/Core>
#include <vector>

#include "p2p_icp/point_cloud.hpp"

namespace p2p_icp
{
/**
 * @brief Static 1-NN 3-D kd-tree, used for the correspondence search
 *
 * It is implemented directly because no external point cloud library such as PCL may be used.
 * Splitting at the median of the widest axis makes the build O(n log n) and a query O(log n) on average.
 */
class KdTree3d
{
public:
    /**
     * @brief Builds the tree from an array of points
     *
     * @param points the points to store
     */
    void build(const std::vector<Eigen::Vector3d>& points);

    /**
     * @brief Builds the tree from the positions of a cloud with normals
     *
     * @param cloud the cloud to store
     */
    void build(const PointCloud& cloud);

    /**
     * @brief Finds the stored point nearest to @p query
     *
     * @param query the query point
     * @param index output: index of the nearest point, nullptr allowed
     * @param squared_distance output: the squared distance, nullptr allowed
     * @return false when the tree is empty
     */
    bool nearest(const Eigen::Vector3d& query, int* index, double* squared_distance) const;

    /**
     * @brief Whether the tree is empty
     *
     * @return true when it holds no points
     */
    bool empty() const
    {
        return points_.empty();
    }

    /**
     * @brief Number of stored points
     *
     * @return the point count
     */
    std::size_t size() const
    {
        return points_.size();
    }

private:
    struct Node
    {
        int index = -1;  ///< index into points_
        int axis = 0;  ///< axis this node splits on (0=x, 1=y, 2=z)
        int left = -1;
        int right = -1;
    };

    /**
     * @brief Recursively builds the subtree over the range indices_[begin, end)
     *
     * @param begin start index
     * @param end   end index, exclusive
     * @return id of the node created, or -1 when the range is empty
     */
    int build_recursive(int begin, int end);

    /**
     * @brief Recursive body of the nearest neighbour search
     *
     * @param node current node id
     * @param query the query point
     * @param best_index best index so far, in/out
     * @param best_squared_distance best squared distance so far, in/out
     */
    void search_recursive(int node, const Eigen::Vector3d& query, int* best_index, double* best_squared_distance) const;

    std::vector<Eigen::Vector3d> points_;
    std::vector<int> indices_;
    std::vector<Node> nodes_;
    int root_ = -1;
};

}  // namespace p2p_icp
