#include "common/kdtree_radius.hpp"

#include <algorithm>

namespace common
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

void KdTreeRadius::clear()
{
    points_.clear();
    indices_.clear();
    nodes_.clear();
    root_ = -1;
}

void KdTreeRadius::build(const std::vector<Eigen::Vector3d>& points)
{
    clear();
    points_ = points;
    if (points_.empty())
    {
        return;
    }
    indices_.resize(points_.size());
    for (std::size_t i = 0; i < indices_.size(); ++i)
    {
        indices_[i] = static_cast<int>(i);
    }

    nodes_.reserve(points_.size());
    root_ = build_recursive(0, static_cast<int>(indices_.size()));
}

int KdTreeRadius::build_recursive(int begin, int end)
{
    if (begin >= end)
    {
        return -1;
    }

    // Split on the axis with the widest spread.
    Eigen::Vector3d lower = points_[indices_[begin]];
    Eigen::Vector3d upper = lower;
    for (int i = begin + 1; i < end; ++i)
    {
        lower = lower.cwiseMin(points_[indices_[i]]);
        upper = upper.cwiseMax(points_[indices_[i]]);
    }
    int axis = 0;
    (upper - lower).maxCoeff(&axis);

    // Partially sort only, so that the median of that axis lands on mid.
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

    const int left = build_recursive(begin, mid);
    const int right = build_recursive(mid + 1, end);
    nodes_[node_id].left = left;
    nodes_[node_id].right = right;
    return node_id;
}

void KdTreeRadius::radius_recursive(int node, const Eigen::Vector3d& query, double max_squared_distance, int max_count, std::vector<int>* indices) const
{
    if (node < 0)
    {
        return;
    }
    if (max_count > 0 && static_cast<int>(indices->size()) >= max_count)
    {
        return;
    }

    const Node& current = nodes_[node];
    if ((points_[current.index] - query).squaredNorm() <= max_squared_distance)
    {
        indices->push_back(current.index);
    }

    // Signed distance to the split plane; negative means the query lies on the left.
    const double diff = query[current.axis] - points_[current.index][current.axis];
    const int near_child = diff < 0.0 ? current.left : current.right;
    const int far_child = diff < 0.0 ? current.right : current.left;

    // Descend into the near child first, so a max_count cut-off still keeps the nearest points.
    radius_recursive(near_child, query, max_squared_distance, max_count, indices);

    // If the split plane is farther than the radius, no point beyond it can be inside the radius.
    if (diff * diff <= max_squared_distance)
    {
        radius_recursive(far_child, query, max_squared_distance, max_count, indices);
    }
}

int KdTreeRadius::radius_search(const Eigen::Vector3d& query, double max_squared_distance, int max_count, std::vector<int>* indices) const
{
    indices->clear();
    if (points_.empty() || !(max_squared_distance > 0.0))
    {
        return 0;
    }
    radius_recursive(root_, query, max_squared_distance, max_count, indices);
    return static_cast<int>(indices->size());
}

}  // namespace common
