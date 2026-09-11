#include "imu_preint/imu_preintegrator.hpp"

#include <Eigen/Geometry>
#include <stdexcept>

#include "imu_preint/so3.hpp"

namespace imu_preint
{
void ImuPreintegrator::reset()
{
    reset(Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());
}

void ImuPreintegrator::reset(const Eigen::Vector3d& gyro_bias, const Eigen::Vector3d& accel_bias)
{
    if (!gyro_bias.allFinite() || !accel_bias.allFinite())
        throw std::invalid_argument("IMU bias must be finite");
    // Initial values of the recursion: dR_ii = I, dv_ii = 0, dp_ii = 0.
    delta_ = PreintegratedDelta{};
    delta_.gyro_bias = gyro_bias;
    delta_.accel_bias = accel_bias;
    samples_.clear();

    step_log_.clear();
    step_log_.push_back(delta_);  // (t_rel = 0, I, 0, 0)
    num_samples_ = 0;
}

void ImuPreintegrator::set_noise(const ImuNoise& noise)
{
    for (double value : {noise.gyro, noise.accel, noise.gyro_bias, noise.accel_bias})
        if (!std::isfinite(value) || value <= 0.0)
            throw std::invalid_argument("IMU noise densities must be finite and positive");
    if (!samples_.empty())
        throw std::logic_error("Set IMU noise before integrating an interval");
    noise_ = noise;
}

void ImuPreintegrator::reintegrate(const Eigen::Vector3d& gyro_bias, const Eigen::Vector3d& accel_bias)
{
    const auto samples = samples_;
    reset(gyro_bias, accel_bias);
    for (const auto& sample : samples)
        integrate(sample.gyro, sample.accel, sample.dt);
}

PreintegratedDelta ImuPreintegrator::corrected_delta(const Eigen::Vector3d& gyro_bias, const Eigen::Vector3d& accel_bias) const
{
    Eigen::Matrix<double, 6, 1> db;
    db << gyro_bias - delta_.gyro_bias, accel_bias - delta_.accel_bias;
    const Eigen::Matrix<double, 9, 1> correction = delta_.bias_jacobian * db;
    PreintegratedDelta result = delta_;
    result.rotation *= so3::Exp(correction.head<3>());
    result.position += correction.segment<3>(3);
    result.velocity += correction.tail<3>();
    return result;
}

void ImuPreintegrator::integrate(const Eigen::Vector3d& gyro, const Eigen::Vector3d& accel, double dt)
{
    if (!std::isfinite(dt) || !gyro.allFinite() || !accel.allFinite())
        throw std::invalid_argument("IMU sample must be finite");
    if (dt <= 0.0)
    {
        return;
    }

    const Eigen::Matrix3d dR_ik = delta_.rotation;  // dR_ik *before* the update
    const Eigen::Vector3d w = gyro - delta_.gyro_bias;
    const Eigen::Vector3d a = accel - delta_.accel_bias;
    const Eigen::Matrix3d dR_step = so3::Exp(w * dt);
    const Eigen::Matrix3d jr = so3::RightJacobian(w * dt);

    // Discrete error transition for the same forward integration used below.
    Matrix15d f = Matrix15d::Identity();
    f.block<3, 3>(0, 0) = dR_step.transpose();
    f.block<3, 3>(0, 9) = -jr * dt;
    f.block<3, 3>(3, 0) = -0.5 * dR_ik * so3::Hat(a) * dt * dt;
    f.block<3, 3>(3, 6) = Eigen::Matrix3d::Identity() * dt;
    f.block<3, 3>(3, 12) = -0.5 * dR_ik * dt * dt;
    f.block<3, 3>(6, 0) = -dR_ik * so3::Hat(a) * dt;
    f.block<3, 3>(6, 12) = -dR_ik * dt;
    delta_.bias_jacobian = (f.topLeftCorner<9, 9>() * delta_.bias_jacobian + f.block<9, 6>(0, 9)).eval();

    Eigen::Matrix<double, 15, 12> g = Eigen::Matrix<double, 15, 12>::Zero();
    g.block<3, 3>(0, 0) = -jr * dt;
    g.block<3, 3>(3, 3) = -0.5 * dR_ik * dt * dt;
    g.block<3, 3>(6, 3) = -dR_ik * dt;
    g.block<6, 6>(9, 6) = Eigen::Matrix<double, 6, 6>::Identity() * dt;
    Eigen::Matrix<double, 12, 1> noise_variance;
    noise_variance << Eigen::Vector3d::Constant(noise_.gyro * noise_.gyro / dt),
         Eigen::Vector3d::Constant(noise_.accel * noise_.accel / dt),
         Eigen::Vector3d::Constant(noise_.gyro_bias * noise_.gyro_bias / dt),
         Eigen::Vector3d::Constant(noise_.accel_bias * noise_.accel_bias / dt);
    delta_.covariance = (f * delta_.covariance * f.transpose() + g * noise_variance.asDiagonal() * g.transpose()).eval();
    delta_.covariance = (0.5 * (delta_.covariance + delta_.covariance.transpose())).eval();

    // --- state update, strictly in the order dp -> dv -> dR ------------------
    // dp and dv both read the pre-update dR_ik, so this order must not change.
    // dp_{i,k+1} = dp_ik + dv_ik dt + 0.5 dR_ik a dt^2
    delta_.position += delta_.velocity * dt + 0.5 * dR_ik * a * dt * dt;
    // dv_{i,k+1} = dv_ik + dR_ik a dt
    delta_.velocity += dR_ik * a * dt;
    // dR_{i,k+1} = dR_ik Exp([w dt]x)
    delta_.rotation = dR_ik * dR_step;
    delta_.dt += dt;

    // Re-project onto SO(3) every step so numerical drift cannot break orthogonality.
    const Eigen::Quaterniond q(delta_.rotation);
    delta_.rotation = q.normalized().toRotationMatrix();

    step_log_.push_back(delta_);
    samples_.push_back({gyro, accel, dt});
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
    const auto corrected = corrected_delta(state_i.gyro_bias, state_i.accel_bias);
    NavState state_j;
    state_j.gyro_bias = state_i.gyro_bias;
    state_j.accel_bias = state_i.accel_bias;
    state_j.timestamp = state_i.timestamp + dt;
    state_j.rotation = state_i.rotation * corrected.rotation;
    state_j.velocity = state_i.velocity + gravity_w * dt + state_i.rotation * corrected.velocity;
    state_j.position = state_i.position + state_i.velocity * dt + 0.5 * gravity_w * dt * dt + state_i.rotation * corrected.position;

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

}  // namespace imu_preint
