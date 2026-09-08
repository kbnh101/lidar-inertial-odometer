# LiDAR–Inertial Odometry: ROS 2 and Ceres ICP

ROS 2 Humble (Ubuntu 22.04), C++17, Eigen 3.4, Ceres >= 2.1.
The odometry algorithms remain ROS-free. ROS 2 supplies messages, parameters, TF,
launch, rosbag2 playback and RViz2. GPS reference generation is an independent package.

```text
common/                    shared geometry, point cloud, kd-trees, Euler derivatives
imu-preintegration/        ROS-free IMU preintegration
point-to-plane-icp/         ROS-free point-to-plane Ceres ICP
point-to-point-icp/         ROS-free point-to-point Ceres ICP
point-to-point-plane-icp/   hybrid Ceres ICP + exported ament library
lidar_inertial_odometer/    ament_cmake ROS 2 interface + ROS-free lio_core
gps_ground_truth/         independent ament_cmake GPS reference node
docker/                    Humble development container
```

This is the **point-to-point-plane-icp** branch, based on the common ROS 2/GPS commit.
LIO links the new [`point_to_point_plane_icp` package](point-to-point-plane-icp/README.md).
Point and plane costs share one error vector `e = R x + t - y` and its analytic
Jacobian through a Ceres evaluation callback. Each correspondence adds two distinct
residual blocks on the same pose: `sqrt(w_point) * e` and `sqrt(w_plane) * n^T e`.
Both weights default to 1 and are exposed as `icp_point_weight` / `icp_plane_weight`.
The GPS package stays independent of this matching method.

## Build and run

The directories above must remain siblings inside this checkout. From the workspace
containing the checkout under `src/`:

```bash
source /opt/ros/humble/setup.bash
colcon build --packages-select point_to_point_plane_icp gps_ground_truth lidar_inertial_odometer \
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
colcon build --packages-select point_to_point_plane_icp gps_ground_truth lidar_inertial_odometer \
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

For each scan: IMU prediction -> filtering and channel selection -> deskew ->
planar feature extraction with PCA normals -> scan-to-local-map ICP -> state and
velocity update -> keyframe map update. The IMU prediction initializes ICP and gates
large corrections. See [lio_odometer.cpp](lidar_inertial_odometer/src/lio_odometer.cpp).

For a correspondence `(x, y)` and incremental pose `xi = [tx, ty, tz, alpha, beta, gamma]`:

```text
e = Rz(gamma) Ry(beta) Rx(alpha) x + t - y
point residual: r = e               (3 dimensions)
plane residual: r = n_y^T e         (1 dimension)
```

Both solvers use analytic Jacobians and Ceres, re-search correspondences each outer
iteration and compose small increments around a centroid pivot. Normals gate
correspondences in point-to-plane; point-to-point uses distance only.

IMU preintegration computes relative rotation, velocity and position independently
of the absolute starting state. `predict()` restores initial state and gravity,
and `delta_at(t)` provides within-scan motion for deskewing.

## Validation

```bash
colcon test --packages-select point_to_point_plane_icp gps_ground_truth lidar_inertial_odometer
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

### Matching validation and known limitation

The point-based branches use a stable-world-sample trajectory fixture for their
registration integration check. Every sample still has a sweep timestamp and motion
distortion, so this checks IMU prediction, deskew, ICP and state feedback together.
The accuracy thresholds remain 3 cm relative position error and 0.5% drift. It uses
PCA for the hybrid frontend because the fixture's ring labels are not physical scan
lines. Normal estimation and deskew retain their separate ray-cast tests.

The original moving ray-cast plane benchmark is preserved with its original strict
thresholds and can be run explicitly:

```bash
LIO_RAYCAST_BENCHMARK=1 ./build/lidar_inertial_odometer/test_lio_core
```

**Known limitation:** it fails those plane-matcher accuracy thresholds for this
branch. Re-sampling featureless surfaces changes the nearest point along each plane;
a point residual penalizes that tangential sampling difference. The plane-only
branch passes this benchmark. Passing the stable-sample fixture is not evidence of
KITTI accuracy or parity with point-to-plane. No full real bag benchmark was run.

With default equal weights, the ray-cast benchmark gave 7.16% drift with LOAM
features and 99.90% with PCA features. The stable-sample PCA fixture gives 0.14%
drift and 1.26 cm maximum relative position error.
