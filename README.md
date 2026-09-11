# LiDAR–Inertial Odometry: ROS 2 and Ceres ICP

ROS 2 Humble (Ubuntu 22.04), C++17, Eigen 3.4, Ceres >= 2.1.
The odometry algorithms remain ROS-free. ROS 2 supplies messages, parameters, TF,
launch, rosbag2 playback and RViz2. GPS reference generation is an independent package.

```text
common/                    shared geometry, point cloud, kd-trees, Euler derivatives
imu-preintegration/        ROS-free IMU preintegration
point-to-plane-icp/         ROS-free point-to-plane Ceres ICP
point-to-point-icp/         ROS-free point-to-point Ceres ICP
lidar_inertial_odometer/    ament_cmake ROS 2 interface + ROS-free lio_core
gps_ground_truth/         independent ament_cmake GPS reference node
docker/                    Humble development container
```

This is the **point-to-plane-icp** branch, based on the common ROS 2/GPS commit.
LIO links `IcpPointToPlane` and defaults to `do_tightly_icp()`, jointly minimizing
point-to-plane, IMU and bias residuals. The original `do_icp()` is unchanged and
can be selected with `use_tightly_coupled: false`. The matcher normalizes supplied normals, rejects targets
without a finite plane normal, and rejects unusable Ceres solutions.
The ROS 2 and GPS interfaces are shared with `main`.

## Build and run

The directories above must remain siblings inside this checkout. From the workspace
containing the checkout under `src/`:

```bash
source /opt/ros/humble/setup.bash
colcon build --packages-select gps_ground_truth lidar_inertial_odometer \
  --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
source install/setup.bash
ros2 launch lidar_inertial_odometer kitti_lio.launch.py play:=false
```

Ceres 2.1 is required; Ubuntu 22.04's default Ceres 2.0 package is too old.
The Dockerfile builds Ceres 2.1 from source and installs Eigen 3.4 from apt.

```bash
./docker/run.sh -d
./docker/exec.sh
# In the container, at /home/clobot_assignment/dev_ws:
colcon build --packages-select gps_ground_truth lidar_inertial_odometer \
  --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
source install/setup.bash
```

The Humble image/container use separate names and build/install volumes from the
old Noetic container. The entire checkout is mounted, so new packages and Git branch
changes appear in the container. Rebuild after switching branches. `--rebuild`
rebuilds the image; `--recreate` recreates its container.

The old ROS1 `.bag` must first be converted to rosbag2. `rosbags-convert` is installed
in the container and does not need ROS1:

```bash
rosbags-convert --src /path/to/2011_09_30_drive_0028.bag --dst /path/to/kitti_ros2
ros2 launch lidar_inertial_odometer kitti_lio.launch.py \
  play:=true bag:=/path/to/kitti_ros2 rate:=0.5 rviz:=true
```

`start:=52.0` skips the first 52 seconds. `play` defaults to false and enables
`use_sim_time` when true. You can set `use_sim_time:=true` for external playback.
The launch starts playback after a two-second discovery delay. For long startup
or slow hardware, launch nodes first and run `ros2 bag play ... --clock --rate 0.5`
separately. Humble's player does not provide the old ROS1 `duration` option.

Edit [kitti.yaml](lidar_inertial_odometer/config/kitti.yaml) or pass `config:=...`.
The YAML has the ROS 2 `lidar_inertial_odometer: ros__parameters:` structure.
`gps:=false` disables the independent reference node. `gps_config:=...`,
`trajectory_csv:=...` and `gt_csv:=...` select its configuration and TUM output files.

## Interfaces and GPS reference

| Node | Inputs | Outputs |
|---|---|---|
| `lidar_inertial_odometer` | `/points_raw`, `/imu_raw` | `~/odometry`, `~/path`, `~/features`, `~/submap`, `~/scan`, `~/trajectory_start`; TF `odom -> imu_link -> velodyne` |
| `gps_ground_truth` | `/gps/fix`, `/lidar_inertial_odometer/trajectory_start` | `~/path`, GPS TUM file |

Sensor subscriptions use best-effort volatile QoS and accept reliable publishers too.
Path, submap and trajectory-start publishers use reliable transient-local QoS;
the start timestamp remains available to late GT subscribers. See the
[ROS 2 QoS documentation](https://docs.ros.org/en/humble/Concepts/Intermediate/About-Quality-of-Service-Settings.html).

The LIO publishes its first valid scan timestamp and frame in a `std_msgs/msg/Header`.
The GPS package buffers fixes until that timestamp is known, interpolates the origin
between bracketing fixes and emits only fixes at or after the start. If recording
begins after the LIO start, the first available GPS fix is used. Invalid/no-fix,
nonfinite, duplicate and out-of-order fixes are rejected. Startup buffering is bounded
by `max_buffer_fixes`; losing the required origin through overflow is an explicit error.

The projection preserves the KITTI scaled Mercator convention of the previous node:
a local ENU approximation with altitude relative to the origin. Keep
`use_imu_orientation_for_yaw: true` with an ENU-referenced IMU orientation to overlay
LIO and GPS directly. GPS provides a **position reference**, not precise 6-DoF ground
truth; TUM quaternion values are identity placeholders. GPS noise and sensor lever
arms are not corrected.

To run GPS reference generation without LIO:

```bash
ros2 launch gps_ground_truth gps_ground_truth.launch.py origin_mode:=first_fix
```

The GT package has no dependency on `lio_core` or its matching method.

## Algorithms

For each scan: bias-corrected IMU prediction -> filtering and channel selection ->
deskew -> planar feature extraction with PCA normals -> joint scan-to-map/IMU
optimization -> keyframe map update. The tightly coupled path is in
[tightly_coupled_lio.cpp](lidar_inertial_odometer/src/tightly_coupled_lio.cpp);
the original independent ICP and velocity feedback path remains in
[lio_odometer.cpp](lidar_inertial_odometer/src/lio_odometer.cpp).

### Tightly coupled point-to-plane ICP

`IcpPointToPlane::do_tightly_icp(prior, preintegration, gravity, T_imu_lidar, options)`
uses the same nearest-neighbor/normal correspondence search as `do_icp()`. It
optimizes **both adjacent states** `(R, p, v, bg, ba)`, with a Gaussian prior on
the previous state, a 15-dimensional combined IMU/bias factor, and individual
point-to-plane residuals on the current state. It does not consume a completed
ICP pose as a measurement. Rotation uses right SO(3) increments, position and
velocity use world-frame increments. State/residual order is `rotation, position,
velocity, gyro bias, accel bias`.

The IMU library subtracts the integration bias, analytically propagates the
9-by-6 preintegration bias Jacobian and 15-by-15 covariance (including bias
random-walk cross correlations), and retains samples for reintegration. The
factor uses first-order bias correction and reintegrates between outer iterations
when its bias thresholds are exceeded. All three factors provide explicit analytic
Jacobians through `ceres::SizedCostFunction`, including the SO(3) Exp/Log and
bias-correction derivatives. Central differences are used only in tests.
Residuals are whitened by `L^-1` for covariance `L L^T`. A `1e-12` diagonal
integration floor handles the rank-deficient covariance of a single Euler step.

This is a **two-state recursive estimator**, not a multi-keyframe smoothing
backend. The marginal current-state covariance is extracted from the joint
linearized problem and transported to the new rotation tangent; it becomes the
next prior. Previous poses in the published trajectory and local map remain
fixed. The map is treated as deterministic, so its uncertainty and correlations
with the state are not modeled. No loop closure, extrinsic calibration or time
offset estimation is performed.

LIO rebuilds deskew/features with the updated velocity and bias for up to
`tight_deskew_iterations` passes. These passes always reuse the original prior
to avoid counting one scan repeatedly. Deskew is fixed within each Ceres solve;
its state derivatives are not included in the plane factor. The first scan
establishes the map without a zero-duration IMU factor. During the second scan,
the initial anchor cloud is re-deskewed using the inferred starting velocity and
bias, keeping its world pose fixed; this prevents a moving start from leaving
an incorrectly deskewed anchor in the map. Missing or gated-out
LiDAR produces an IMU-only posterior with propagated covariance and does not
insert a keyframe into an established map. Incomplete IMU intervals are skipped.

Configure noise densities, initial bias and state standard deviations in
`kitti.yaml`. Noise values are starting values, not a KITTI sensor calibration.
Initial gyro bias defaults to zero rather than assuming that the initialization
window is stationary; a calibrated bias can be supplied. Initial accel bias is
subtracted during gravity alignment. The initial pose prior anchors the world
frame, and a broad velocity prior allows a moving start. The legacy
`velocity_correction_gain` is used only with `use_tightly_coupled: false`.

The preintegration formulation follows
[Forster et al., On-Manifold Preintegration](https://arxiv.org/abs/1512.02363).

For a correspondence `(x, y)` and incremental pose `xi = [tx, ty, tz, alpha, beta, gamma]`:

```text
e = Rz(gamma) Ry(beta) Rx(alpha) x + t - y
point residual: r = e               (3 dimensions)
plane residual: r = n_y^T e         (1 dimension)
```

The standalone point-to-point and point-to-plane solvers use analytic Jacobians and Ceres, re-search correspondences each outer
iteration and compose small increments around a centroid pivot. Normals gate
correspondences in point-to-plane; point-to-point uses distance only.

IMU preintegration computes relative rotation, velocity and position independently
of the absolute starting state. `predict()` restores initial state and gravity,
and `delta_at(t)` provides within-scan motion for deskewing.

## Validation

```bash
colcon test --packages-select gps_ground_truth lidar_inertial_odometer
colcon test-result --verbose
# ROS 2 synthetic message / TF / late-subscriber / GPS integration check:
python3 src/lidar-inertial-odometer/lidar_inertial_odometer/test/test_ros2_interface.py
```

The core suite checks synthetic motion, deskew and normal estimation. GPS tests cover
projection, interpolation, delayed start, invalid fixes and buffer overflow. Standalone
library suites can also be built directly:

```bash
cmake -S point-to-plane-icp -B /tmp/p2plane -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/p2plane -j2
ctest --test-dir /tmp/p2plane --output-on-failure
# Substitute point-to-point-icp, imu-preintegration or common as needed.
```

Plot recorded positions with:

```bash
python3 src/lidar-inertial-odometer/lidar_inertial_odometer/scripts/plot_trajectory.py \
  --est /tmp/lio_trajectory.txt --gt /tmp/lio_gt.txt --out /tmp/trajectory.png
```

The ROS1 trajectory numbers are not a ROS2 benchmark; evaluate a full converted bag
before comparing matching accuracy on real data.
