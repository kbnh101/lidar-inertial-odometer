#include "lidar_inertial_odometer/imu_preintegrator.hpp"

#include <Eigen/Geometry>
#include <algorithm>

#include "lidar_inertial_odometer/so3.hpp"

void ImuPreintegrator::reset()
{
    // Initial values of the recursion: dR_ii = I, dv_ii = 0, dp_ii = 0.
    delta_ = PreintegratedDelta{};

    step_log_.clear();
    step_log_.push_back(delta_);  // (t_rel = 0, I, 0, 0)
    num_samples_ = 0;
}

void ImuPreintegrator::integrate(const Eigen::Vector3d& gyro, const Eigen::Vector3d& accel, double dt)
{
    if (dt <= 0.0)
    {
        return;
    }

    const Eigen::Matrix3d dR_ik = delta_.rotation;  // dR_ik *before* the update
    const Eigen::Matrix3d dR_step = so3::Exp(gyro * dt);  // Exp([w dt]x)

    // --- state update, strictly in the order dp -> dv -> dR ------------------
    // dp and dv both read the pre-update dR_ik, so this order must not change.
    // dp_{i,k+1} = dp_ik + dv_ik dt + 0.5 dR_ik a dt^2
    delta_.position += delta_.velocity * dt + 0.5 * dR_ik * accel * dt * dt;
    // dv_{i,k+1} = dv_ik + dR_ik a dt
    delta_.velocity += dR_ik * accel * dt;
    // dR_{i,k+1} = dR_ik Exp([w dt]x)
    delta_.rotation = dR_ik * dR_step;
    delta_.dt += dt;

    // Re-project onto SO(3) every step so numerical drift cannot break orthogonality.
    const Eigen::Quaterniond q(delta_.rotation);
    delta_.rotation = q.normalized().toRotationMatrix();

    step_log_.push_back(delta_);
    ++num_samples_;
}

PreintegratedDelta ImuPreintegrator::delta() const
{
    return delta_;
}

NavState ImuPreintegrator::predict(const NavState& state_i, const Eigen::Vector3d& gravity_w) const
{
    // The delta definitions solved for R_j, v_j, p_j -- absolute state and gravity reappear only
    // here, since the integration itself already finished without them.
    const double dt = delta_.dt;
    NavState state_j;
    state_j.timestamp = state_i.timestamp + dt;
    state_j.rotation = state_i.rotation * delta_.rotation;
    state_j.velocity = state_i.velocity + gravity_w * dt + state_i.rotation * delta_.velocity;
    state_j.position = state_i.position + state_i.velocity * dt + 0.5 * gravity_w * dt * dt + state_i.rotation * delta_.position;

    const Eigen::Quaterniond q(state_j.rotation);
    state_j.rotation = q.normalized().toRotationMatrix();
    return state_j;
}

PreintegratedDelta ImuPreintegrator::delta_at(double t_rel) const
{
    if (step_log_.empty())
    {
        return PreintegratedDelta();
    }
    if (t_rel <= step_log_.front().dt)
    {
        return step_log_.front();
    }
    if (t_rel >= step_log_.back().dt)
    {
        return step_log_.back();
    }

    // Find the two steps bracketing t_rel; step_log_ is sorted by dt.
    // Both ends are handled above, so 1 <= upper_index <= size-1 is guaranteed.
    std::size_t upper_index = 1;
    while (upper_index < step_log_.size() && step_log_[upper_index].dt < t_rel)
    {
        ++upper_index;
    }

    const PreintegratedDelta& lower = step_log_[upper_index - 1];
    const PreintegratedDelta& upper = step_log_[upper_index];

    const double span = upper.dt - lower.dt;
    const double ratio = span > 0.0 ? (t_rel - lower.dt) / span : 0.0;

    PreintegratedDelta interpolated;
    interpolated.dt = t_rel;
    // Rotation lives on a manifold, so it is slerped rather than linearly interpolated.
    const Eigen::Quaterniond q_lower(lower.rotation);
    const Eigen::Quaterniond q_upper(upper.rotation);
    interpolated.rotation = q_lower.slerp(ratio, q_upper).toRotationMatrix();
    interpolated.velocity = lower.velocity + ratio * (upper.velocity - lower.velocity);
    interpolated.position = lower.position + ratio * (upper.position - lower.position);
    return interpolated;
}
