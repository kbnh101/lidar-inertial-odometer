#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace imu_preint
{
using Matrix15d = Eigen::Matrix<double, 15, 15>;
using Vector15d = Eigen::Matrix<double, 15, 1>;

/// Continuous-time noise densities. Measurement units / sqrt(Hz), and bias units / sqrt(s).
struct ImuNoise
{
    double gyro = 0.002;
    double accel = 0.02;
    double gyro_bias = 0.0002;
    double accel_bias = 0.002;
};
/// Navigation state, matching (R_i, p_i, v_i) of the preintegration.
struct NavState
{
    double timestamp = 0.0;
    Eigen::Matrix3d rotation = Eigen::Matrix3d::Identity();  ///< R_wi (world <- imu)
    Eigen::Vector3d position = Eigen::Vector3d::Zero();  ///< p_i (world)
    Eigen::Vector3d velocity = Eigen::Vector3d::Zero();  ///< v_i (world)
    Eigen::Vector3d gyro_bias = Eigen::Vector3d::Zero();
    Eigen::Vector3d accel_bias = Eigen::Vector3d::Zero();

    /**
     * @brief Packs attitude and position into a single rigid transform
     *
     * @return T_wi (world <- imu), velocity not included
     */
    Eigen::Isometry3d isometry() const
    {
        Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
        T.linear() = rotation;
        T.translation() = position;
        return T;
    }
};

/// Preintegrated measurement -- (dR_ij, dv_ij, dp_ij, dt_ij).
struct PreintegratedDelta
{
    Eigen::Matrix3d rotation = Eigen::Matrix3d::Identity();  ///< dR_ij
    Eigen::Vector3d velocity = Eigen::Vector3d::Zero();  ///< dv_ij
    Eigen::Vector3d position = Eigen::Vector3d::Zero();  ///< dp_ij
    double dt = 0.0;  ///< dt_ij [s]
    Eigen::Vector3d gyro_bias = Eigen::Vector3d::Zero();  ///< integration linearization point
    Eigen::Vector3d accel_bias = Eigen::Vector3d::Zero();
    /// Right rotation error, position, velocity, gyro bias, accel bias (all delta errors in frame i).
    Matrix15d covariance = Matrix15d::Zero();
    /// Rows: right rotation, position, velocity. Columns: gyro bias, accel bias.
    Eigen::Matrix<double, 9, 6> bias_jacobian = Eigen::Matrix<double, 9, 6>::Zero();
};

}  // namespace imu_preint
