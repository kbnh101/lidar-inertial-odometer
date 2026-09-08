#pragma once

#include <Eigen/Core>
#include <cmath>
#include <deque>
#include <limits>
#include <optional>
#include <stdexcept>
#include <vector>

namespace gps_ground_truth
{
struct Fix
{
    double timestamp;
    double latitude;
    double longitude;
    double altitude;
};
struct Sample
{
    double timestamp;
    Eigen::Vector3d position;
};

// KITTI's scaled Mercator projection. This preserves the previous GT convention;
// it is a local ENU approximation, not an ellipsoidal ECEF-to-ENU conversion.
inline Eigen::Vector3d Project(const Fix& fix, double reference_latitude)
{
    constexpr double pi = 3.14159265358979323846;
    constexpr double radius = 6378137.0;
    const double scale = std::cos(reference_latitude * pi / 180.0);
    return {scale * fix.longitude * pi * radius / 180.0, scale * radius * std::log(std::tan((90.0 + fix.latitude) * pi / 360.0)), fix.altitude};
}

class Trajectory
{
public:
    explicit Trajectory(int max_buffer = 100000) : max_buffer_(max_buffer)
    {
        if (max_buffer < 2)
            throw std::invalid_argument("max_buffer must be at least 2");
    }
    bool AddFix(const Fix& fix)
    {
        if (!std::isfinite(fix.timestamp) || !std::isfinite(fix.latitude) || !std::isfinite(fix.longitude) || !std::isfinite(fix.altitude) ||
            std::abs(fix.latitude) >= 90.0 || std::abs(fix.longitude) > 180.0 || fix.timestamp <= last_time_)
            return false;
        last_time_ = fix.timestamp;
        fixes_.push_back(fix);
        if (fixes_.size() > max_buffer_)
        {
            discarded_before_ = fixes_.front().timestamp;
            fixes_.pop_front();
        }
        return true;
    }
    void SetStart(double timestamp)
    {
        if (!start_ && std::isfinite(timestamp))
            start_ = timestamp;
    }
    bool has_start() const
    {
        return start_.has_value();
    }
    bool has_origin() const
    {
        return origin_.has_value();
    }
    std::vector<Sample> Drain()
    {
        std::vector<Sample> output;
        if (!start_ || fixes_.empty())
            return output;
        if (!origin_)
        {
            if (fixes_.back().timestamp < *start_)
                return output;
            // Never silently use the wrong origin after the startup buffer overflowed.
            if (discarded_before_ && *start_ < fixes_.front().timestamp)
                throw std::runtime_error("GPS startup buffer no longer covers LIO start; increase max_buffer_fixes");
            Fix origin_fix = fixes_.front();
            // If GPS recording starts later than LIO, use the first available fix.
            for (std::size_t i = 1; i < fixes_.size() && *start_ > fixes_.front().timestamp; ++i)
            {
                const Fix& a = fixes_[i - 1];
                const Fix& b = fixes_[i];
                if (b.timestamp >= *start_)
                {
                    const double ratio = (*start_ - a.timestamp) / (b.timestamp - a.timestamp);
                    origin_fix = {*start_, a.latitude + ratio * (b.latitude - a.latitude), a.longitude + ratio * (b.longitude - a.longitude),
                                  a.altitude + ratio * (b.altitude - a.altitude)};
                    break;
                }
            }
            reference_latitude_ = origin_fix.latitude;
            origin_ = Project(origin_fix, reference_latitude_);
        }
        for (const Fix& fix : fixes_)
            if (fix.timestamp >= *start_)
                output.push_back({fix.timestamp, Project(fix, reference_latitude_) - *origin_});
        fixes_.clear();
        return output;
    }

private:
    std::size_t max_buffer_;
    std::deque<Fix> fixes_;
    std::optional<double> start_;
    std::optional<double> discarded_before_;
    std::optional<Eigen::Vector3d> origin_;
    double reference_latitude_ = 0.0;
    double last_time_ = -std::numeric_limits<double>::infinity();
};
}  // namespace gps_ground_truth
