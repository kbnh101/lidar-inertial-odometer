#pragma once

#include <Eigen/Core>
#include <vector>

#include "lidar_inertial_odometer/util.hpp"

/**
 * @brief IMU preintegration
 *
 * Compresses the IMU measurements between keyframes i and j into a single relative quantity:
 *   dR_ij = R_i^T R_j,  dv_ij = R_i^T(v_j - v_i - g dt),
 *   dp_ij = R_i^T(p_j - p_i - v_i dt - 0.5 g dt^2)
 * Solved out, these depend on the measurements only:
 *   dR_ij = prod_k Exp([w_k dt]x)
 *   dv_ij = sum_k dR_ik a_k dt
 *   dp_ij = sum_k [ dv_ik dt + 0.5 dR_ik a_k dt^2 ]
 *
 * The key point is that no absolute state (R_i, v_i, p_i) and no gravity g survives on the right
 * hand side, so an update of R_i during optimization never forces a re-integration.
 *
 * The integration uses the recursion below in the order dp -> dv -> dR, which matters because dp
 * and dv must both see the pre-update dR.
 *
 * @see PreintegratedDelta (util.hpp) for the relative quantity this class produces
 */
class ImuPreintegrator
{
public:
    ImuPreintegrator()
    {
        reset();
    }

    /**
     * @brief Resets the interval to dR = I, dv = 0, dp = 0
     */
    void reset();

    /**
     * @brief Integrates one IMU sample through the recursion
     *
     * @param gyro  w_k [rad/s]
     * @param accel a_k [m/s^2]
     * @param dt    sample interval [s], ignored when not positive
     */
    void integrate(const Eigen::Vector3d& gyro, const Eigen::Vector3d& accel, double dt);

    /**
     * @brief Preintegrated measurement accumulated so far
     *
     * @return (dR_ij, dv_ij, dp_ij, dt_ij)
     */
    PreintegratedDelta delta() const;

    /**
     * @brief Predicts state j from state i by solving the delta definitions for R_j, v_j, p_j
     *
     * @param state_i   state at keyframe i
     * @param gravity_w gravity vector g in the world frame
     * @return predicted state j
     */
    NavState predict(const NavState& state_i, const Eigen::Vector3d& gravity_w) const;

    /**
     * @brief Preintegrated measurement at relative time @p t_rel inside the interval, for deskewing
     *
     * @param t_rel relative time from the start of the interval [s]
     * @return preintegrated measurement up to that time
     */
    PreintegratedDelta delta_at(double t_rel) const;

    /**
     * @brief Length of the integrated interval
     *
     * @return dt_ij [s]
     */
    double delta_t() const
    {
        return delta_.dt;
    }

    /**
     * @brief Number of IMU samples integrated into this interval
     *
     * @return sample count
     */
    int num_samples() const
    {
        return num_samples_;
    }

private:
    PreintegratedDelta delta_;

    // Per-step log used by delta_at(), in ascending t_rel.
    std::vector<PreintegratedDelta> step_log_;
    int num_samples_ = 0;
};
