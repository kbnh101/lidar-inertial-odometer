#include "lidar_inertial_odometer/lio_odometer.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <optional>
#include <stdexcept>

void LioOdometer::ProcessTightlyCoupledScan(const QueuedScan& scan)
{
    if (options_.deskew_iterations < 1)
        throw std::invalid_argument("deskew_iterations must be positive");
    if (scan.timestamp < state_.timestamp || (has_keyframe_ && scan.timestamp <= state_.timestamp))
        return;
    const p2p_icp::NavStatePrior prior{state_, state_covariance_};
    preintegrator_.reset(state_.gyro_bias, state_.accel_bias);
    preintegrator_.set_noise(options_.imu_noise);
    IntegrateInterval(&preintegrator_, state_.timestamp, scan.timestamp);
    // Do not fabricate an IMU factor when the queue does not cover the start boundary.
    if (std::abs(preintegrator_.delta_t() - (scan.timestamp - state_.timestamp)) > 1e-6)
        return;
    const auto predicted = preintegrator_.predict(state_, options_.gravity);
    auto estimate = predicted;
    auto covariance = state_covariance_;
    const auto preprocessed = feature_extractor_.Preprocess(scan.points);
    std::vector<RawLidarPoint> corrected;
    FeatureCloud features;
    const auto rebuild_scan = [&](const imu_preint::NavState& reference) {
        if (options_.enable_deskew)
        {
            imu_preint::ImuPreintegrator scan_integrator;
            scan_integrator.reset(reference.gyro_bias, reference.accel_bias);
            scan_integrator.set_noise(options_.imu_noise);
            IntegrateInterval(&scan_integrator, scan.timestamp, scan.timestamp + options_.scan_period);
            if (std::abs(scan_integrator.delta_t() - options_.scan_period) > 1e-6)
                return false;
            if (!has_keyframe_)
                bootstrap_integrator_ = scan_integrator;
            corrected = Deskew(preprocessed, reference, scan_integrator);
        }
        else
            corrected = preprocessed;
        features = feature_extractor_.Extract(corrected);
        return true;
    };
    if (icp_target_version_ != map_version_)
    {
        icp_.set_target(local_map_.cloud());
        icp_target_version_ = map_version_;
    }

    LioFrameResult result;
    result.timestamp = scan.timestamp;
    result.imu_prediction = predicted.isometry();
    result.num_map_points = static_cast<int>(local_map_.cloud().size());
    bool accepted = false;
    auto previous_estimate = prior.state;
    const bool refine_initial_map = options_.enable_deskew && !bootstrap_points_.empty() &&
                                    std::abs(prior.state.timestamp - bootstrap_state_.timestamp) < 1e-6;
    std::optional<LocalMap> refined_map;
    const auto restore_target = [&]() {
        if (refined_map)
            icp_.set_target(local_map_.cloud());
    };
    const auto begin = std::chrono::steady_clock::now();
    const int passes = options_.enable_deskew ? options_.deskew_iterations : 1;
    for (int pass = 0; pass < passes; ++pass)
    {
        if (refine_initial_map && pass > 0)
        {
            // The first scan was deskewed with unknown (zero) velocity. Correct that anchor
            // geometry once velocity is inferred; its world pose stays at the original gauge.
            auto reference = bootstrap_state_;
            reference.velocity = previous_estimate.velocity;
            reference.gyro_bias = previous_estimate.gyro_bias;
            reference.accel_bias = previous_estimate.accel_bias;
            bootstrap_integrator_.reintegrate(reference.gyro_bias, reference.accel_bias);
            const auto anchor_features = feature_extractor_.Extract(Deskew(bootstrap_points_, reference, bootstrap_integrator_));
            const auto anchor_pose = reference.isometry() * options_.T_imu_lidar;
            const auto anchor_world = common::transform_point_cloud(anchor_features.planar, anchor_pose.rotation(), anchor_pose.translation());
            if (!anchor_world.empty())
            {
                refined_map.emplace(local_map_.options());
                refined_map->AddKeyframe(anchor_world, reference.position);
                icp_.set_target(refined_map->cloud());
            }
        }
        if (!rebuild_scan(estimate))
        {
            restore_target();
            return;
        }
        if (preintegrator_.delta_t() <= 0.0)
            break;  // first scan establishes the map and initial prior, without a zero-time factor
        icp_.set_source(features.planar);
        auto tight_options = options_.tightly_coupled;
        tight_options.min_correspondences = options_.min_icp_correspondences;
        tight_options.max_translation_correction = options_.max_icp_translation_deviation;
        tight_options.max_rotation_correction = options_.max_icp_rotation_deviation;
        const auto solved = icp_.do_tightly_icp(prior, preintegrator_, options_.gravity, options_.T_imu_lidar, tight_options);
        if (!solved.usable)
        {
            restore_target();
            return;  // keep the old state/time and IMU queue so the next interval can be integrated
        }
        const double velocity_change = (estimate.velocity - solved.posterior.state.velocity).norm();
        const double bias_change = (estimate.accel_bias - solved.posterior.state.accel_bias).norm() +
                                   (estimate.gyro_bias - solved.posterior.state.gyro_bias).norm();
        const double rotation_change = Eigen::AngleAxisd(estimate.rotation.transpose() * solved.posterior.state.rotation).angle();
        estimate = solved.posterior.state;
        previous_estimate = solved.previous_state;
        covariance = solved.posterior.covariance;
        accepted = solved.lidar_accepted;
        result.icp_iterations += solved.icp.iterations;
        result.icp_correspondences = solved.icp.correspondences;
        result.icp_error = solved.icp.final_error;
        if (!accepted || (velocity_change < 1e-3 && bias_change < 1e-4 && rotation_change < 1e-5))
            break;
        // Rebuild against the new estimate, but never feed this scan's posterior back as its prior.
    }
    if (!rebuild_scan(estimate))
    {
        restore_target();
        return;
    }
    if (refined_map && accepted)
    {
        local_map_ = std::move(*refined_map);
        ++map_version_;
        icp_target_version_ = map_version_;
    }
    else
        restore_target();
    result.icp_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
    result.icp_accepted = accepted;
    result.num_features = static_cast<int>(features.planar.size());
    state_ = estimate;
    state_covariance_ = covariance;
    trajectory_.push_back(state_);
    const auto world_from_lidar = state_.isometry() * options_.T_imu_lidar;
    result.feature_cloud_world.reserve(features.planar.size());
    for (const auto& point : features.planar)
        result.feature_cloud_world.push_back({world_from_lidar * point.point, world_from_lidar.rotation() * point.normal});
    bool keyframe = !has_keyframe_;
    if (has_keyframe_ && accepted)
    {
        const auto relative = last_keyframe_pose_.inverse() * state_.isometry();
        keyframe = relative.translation().norm() > options_.keyframe_translation ||
                   Eigen::AngleAxisd(relative.rotation()).angle() > options_.keyframe_rotation;
    }
    // An IMU-only fallback must not contaminate an established map.
    if (keyframe && !features.planar.empty())
    {
        if (!has_keyframe_ && options_.enable_deskew)
        {
            bootstrap_points_ = preprocessed;
            bootstrap_state_ = state_;
        }
        local_map_.AddKeyframe(result.feature_cloud_world, state_.position);
        last_keyframe_pose_ = state_.isometry();
        has_keyframe_ = true;
        ++map_version_;
        result.is_keyframe = true;
    }
    if (preintegrator_.delta_t() > 0.0)
        bootstrap_points_.clear();
    if (options_.keep_deskewed_scan)
        result.scan_lidar = std::move(corrected);
    result.valid = true;
    result.state = state_;
    result.lidar_pose = world_from_lidar;
    if (options_.verbose)
        std::printf("[tight-lio] t=%.3f corr=%d lidar=%s rms=%.4f time=%.2f ms | v=%.3f bg=%.5f ba=%.5f\n",
                    scan.timestamp, result.icp_correspondences, accepted ? "ok" : "IMU", result.icp_error, result.icp_ms,
                    state_.velocity.norm(), state_.gyro_bias.norm(), state_.accel_bias.norm());
    results_.push_back(std::move(result));
}
