#include "p2p_icp/kdtree.hpp"

#include <algorithm>
#include <limits>

namespace p2p_icp
{
namespace
{
/// Comparator for std::nth_element, ordering two point indices by the given axis coordinate.
struct AxisLess
{
    const std::vector<Eigen::Vector3d>* points = nullptr;
    int axis = 0;

    bool operator()(int lhs, int rhs) const
    {
        return (*points)[lhs][axis] < (*points)[rhs][axis];
    }
};

}  // namespace

void KdTree3d::build(const std::vector<Eigen::Vector3d>& points)
{
    points_ = points;

    indices_.resize(points_.size());
    for (std::size_t i = 0; i < indices_.size(); ++i)
    {
        indices_[i] = static_cast<int>(i);
    }

    nodes_.clear();
    nodes_.reserve(points_.size());

    root_ = -1;
    if (!points_.empty())
    {
        root_ = build_recursive(0, static_cast<int>(points_.size()));
    }
}

void KdTree3d::build(const PointCloud& cloud)
{
    std::vector<Eigen::Vector3d> points;
    points.reserve(cloud.size());
    for (const PointNormal& entry : cloud)
    {
        points.push_back(entry.point);
    }
    build(points);
}

int KdTree3d::build_recursive(int begin, int end)
{
    if (begin >= end)
    {
        return -1;
    }

    // Split on the axis with the widest spread; this balances the tree better than cycling axes.
    Eigen::Vector3d lower = points_[indices_[begin]];
    Eigen::Vector3d upper = lower;
    for (int i = begin + 1; i < end; ++i)
    {
        lower = lower.cwiseMin(points_[indices_[i]]);
        upper = upper.cwiseMax(points_[indices_[i]]);
    }
    int axis = 0;
    (upper - lower).maxCoeff(&axis);

    // Partially sort only, so the median of that axis lands on mid; a full sort is unnecessary.
    const int mid = begin + (end - begin) / 2;

    AxisLess axis_less;
    axis_less.points = &points_;
    axis_less.axis = axis;
    std::nth_element(indices_.begin() + begin, indices_.begin() + mid, indices_.begin() + end, axis_less);

    Node node;
    node.index = indices_[mid];
    node.axis = axis;
    node.left = -1;
    node.right = -1;

    const int node_id = static_cast<int>(nodes_.size());
    nodes_.push_back(node);
    // nodes_ may reallocate inside build_recursive(), so access it by index rather than reference.
    const int left = build_recursive(begin, mid);
    const int right = build_recursive(mid + 1, end);
    nodes_[node_id].left = left;
    nodes_[node_id].right = right;
    return node_id;
}

bool KdTree3d::nearest(const Eigen::Vector3d& query, int* index, double* squared_distance) const
{
    if (root_ < 0)
    {
        return false;
    }
    int best_index = -1;
    double best_squared_distance = std::numeric_limits<double>::infinity();
    search_recursive(root_, query, &best_index, &best_squared_distance);
    if (best_index < 0)
    {
        return false;
    }
    if (index != nullptr)
    {
        *index = best_index;
    }
    if (squared_distance != nullptr)
    {
        *squared_distance = best_squared_distance;
    }
    return true;
}

void KdTree3d::search_recursive(int node_id, const Eigen::Vector3d& query, int* best_index, double* best_squared_distance) const
{
    if (node_id < 0)
    {
        return;
    }
    const Node& node = nodes_[node_id];
    const double squared_distance = (points_[node.index] - query).squaredNorm();
    if (squared_distance < *best_squared_distance)
    {
        *best_squared_distance = squared_distance;
        *best_index = node.index;
    }

    const double diff = query[node.axis] - points_[node.index][node.axis];
    const int near_child = diff < 0.0 ? node.left : node.right;
    const int far_child = diff < 0.0 ? node.right : node.left;

    search_recursive(near_child, query, best_index, best_squared_distance);
    // Descend past the split plane only when a closer point could still hide there; this pruning is
    // what makes a kd-tree search faster than brute force.
    if (diff * diff < *best_squared_distance)
    {
        search_recursive(far_child, query, best_index, best_squared_distance);
    }
}

}  // namespace p2p_icp
