#pragma once

#include "imu_preint/imu_preintegrator.hpp"

namespace p2p_icp
{
/// Gaussian prior in [right rotation, world position, world velocity, bg, ba] coordinates.
struct NavStatePrior
{
    imu_preint::NavState state;
    imu_preint::Matrix15d covariance = InitialCovariance();

    static imu_preint::Matrix15d InitialCovariance()
    {
        imu_preint::Vector15d sigma;
        sigma << Eigen::Vector3d::Constant(0.02), Eigen::Vector3d::Constant(0.01),
                 Eigen::Vector3d::Constant(10.0), Eigen::Vector3d::Constant(0.05), Eigen::Vector3d::Constant(0.5);
        return sigma.array().square().matrix().asDiagonal();
    }
};

struct TightlyCoupledOptions
{
    double lidar_sigma = 0.1;  ///< point-to-plane standard deviation [m]
    int min_correspondences = 20;
    double max_translation_correction = 1.5;  ///< relative to IMU prediction [m]
    double max_rotation_correction = 0.35;  ///< [rad]
    double gyro_reintegration_threshold = 0.01;  ///< [rad/s]
    double accel_reintegration_threshold = 0.1;  ///< [m/s^2]
    bool use_lidar = true;  ///< false gives IMU-only propagation including covariance
};
}  // namespace p2p_icp
