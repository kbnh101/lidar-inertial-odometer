#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace imu_preint
{
/// Navigation state, matching (R_i, p_i, v_i) of the preintegration.
struct NavState
{
    double timestamp = 0.0;
    Eigen::Matrix3d rotation = Eigen::Matrix3d::Identity();  ///< R_wi (world <- imu)
    Eigen::Vector3d position = Eigen::Vector3d::Zero();  ///< p_i (world)
    Eigen::Vector3d velocity = Eigen::Vector3d::Zero();  ///< v_i (world)

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
};

}  // namespace imu_preint
