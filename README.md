# LiDAR–Inertial Odometry: ROS 2 and Ceres ICP

ROS 2 Humble (Ubuntu 22.04), C++17, Eigen 3.4, Ceres >= 2.1.
The odometry algorithms remain ROS-free. ROS 2 supplies messages, parameters, TF,
launch, rosbag2 playback and RViz2. GPS reference generation is an independent package.

```text
common/                    shared geometry, point cloud, kd-trees, Euler derivatives
imu-preintegration/        ROS-free IMU preintegration
fused-point-plane-icp/     ROS-free fused point-to-point + point-to-plane Ceres ICP (this branch)
point-to-plane-icp/         ROS-free point-to-plane Ceres ICP
point-to-point-icp/         ROS-free point-to-point Ceres ICP
lidar_inertial_odometer/    ament_cmake ROS 2 interface + ROS-free lio_core
gps_ground_truth/         independent ament_cmake GPS reference node
docker/                    Humble development container
```

`main` contains the common ROS 2 migration and GPS separation, retaining the
existing point-to-plane matcher. The matching variants branch from that common commit.
This branch (`fused-point-plane-icp`) adds the `fused-point-plane-icp` package and
switches `lio_core` to it; the two single-term packages stay as standalone references.

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

For each scan: IMU prediction -> filtering and channel selection -> deskew ->
planar feature extraction with PCA normals -> scan-to-local-map ICP -> state and
velocity update -> keyframe map update. The IMU prediction initializes ICP and gates
large corrections. See [lio_odometer.cpp](lidar_inertial_odometer/src/lio_odometer.cpp).

### Fused point-to-point + point-to-plane residual (this branch)

Implements the derivation note *Point-to-Point + Point-to-Plane Residual·Jacobian
상세 유도* (2026-09-10). For a correspondence `(p, q, n)` with `|n| = 1` and the SE(3)
right perturbation `T+ = T Exp(delta_xi)`, `delta_xi = [delta_phi; delta_rho]`:

```text
e     = R p + t - q                                              geometric error
G     = de/ddelta_xi = [-R [p]x   R]                             3x6 geometric Jacobian
Omega = alpha I + beta n n^T                                     2C = e^T Omega e
L     = sqrt(alpha) I + (sqrt(alpha+beta) - sqrt(alpha)) n n^T   L^T L = Omega
r_f   = L e                                                      3-D fused residual
J_f   = L G                                                      3x6 analytic Jacobian
```

`|r_f|^2 = alpha |e|^2 + beta (n^T e)^2`, so one 3-D residual per pair yields exactly the
Gauss-Newton normal equations of a 3-D point block plus a 1-D plane block; tangent-plane
errors are weighted by `alpha`, normal errors by `alpha + beta`. `alpha = 0` is pure
point-to-plane, `beta = 0` pure point-to-point (it reproduces `point-to-point-icp` to
1e-11 on the sample files). A Huber loss on the block is the joint loss
`rho(alpha |e|^2 + beta (n^T e)^2)`.

The pose block lives on a custom `ceres::Manifold` (`Se3RightManifold`) whose `Plus` is
the exact SE(3) exponential, so every Ceres iteration relinearizes at the current pose
with the analytic `J_f` -- no AutoDiff, no Euler angles. The cost function returns
`[J_f | 0]` as its 3x7 ambient Jacobian and the manifold's `PlusJacobian` is `[I_6; 0]`,
so Ceres' product is `J_f` exactly; the built-in `QuaternionManifold` would rescale it
(note, page 11). Before the solve the source is re-parametrized around its centroid
(`p' = p - c`, `T' = T [I c; 0 1]`) to remove the rotation/translation coupling of `H`.
See [fused_residual.hpp](fused-point-plane-icp/include/fused_icp/fused_residual.hpp),
[fused_cost.hpp](fused-point-plane-icp/include/fused_icp/fused_cost.hpp) and
[icp_fused_point_plane.cpp](fused-point-plane-icp/src/icp_fused_point_plane.cpp).

`icp_point_weight` (alpha) and `icp_plane_weight` (beta) in
[kitti.yaml](lidar_inertial_odometer/config/kitti.yaml) set the weights. Scan-to-map
never re-observes the same physical point, so the point term is biased along the
tangent plane and alpha must stay small: on the synthetic core test alpha = 0 matches
point-to-plane (drift 0.2 %), alpha <= 0.01 stays within its thresholds and
alpha >= 0.05 drifts more than 0.5 %. The default is alpha = 0.01, beta = 1.

### Single-term references

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
colcon test --packages-select gps_ground_truth lidar_inertial_odometer
colcon test-result --verbose
# ROS 2 synthetic message / TF / late-subscriber / GPS integration check:
python3 src/lidar-inertial-odometer/lidar_inertial_odometer/test/test_ros2_interface.py
```

The core suite checks synthetic motion, deskew and normal estimation. GPS tests cover
projection, interpolation, delayed start, invalid fixes and buffer overflow. Standalone
library suites can also be built directly:

```bash
cmake -S fused-point-plane-icp -B /tmp/fused -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/fused -j2
ctest --test-dir /tmp/fused --output-on-failure
/tmp/fused/icp_demo [data_dir] [alpha] [beta]
# Substitute point-to-plane-icp, point-to-point-icp, imu-preintegration or common as needed.
```

The fused suite follows the verification section of the note: the worked single-pair
example, central differences through the same SE(3) retraction (max error ~4e-9 for
G, J_pt, J_pl, J_f), equality of the 3-D fused / 4-D stacked / directly accumulated
`H, b`, the tangent Jacobian Ceres assembles through `Se3RightManifold`, the joint
Huber cost, the rank of `H` on a single plane with and without alpha, and end-to-end
registration on the sample files and a synthetic corner. Pure point-to-point
(`beta = 0`) started at the identity stops in a local minimum on the sample files, as
`point-to-point-icp` does, so that case runs from a nearby initial guess.

Plot recorded positions with:

```bash
python3 src/lidar-inertial-odometer/lidar_inertial_odometer/scripts/plot_trajectory.py \
  --est /tmp/lio_trajectory.txt --gt /tmp/lio_gt.txt --out /tmp/trajectory.png
```

### Comparing the matching methods on a bag

The matchers live on different branches, so [compare_matchers.py](lidar_inertial_odometer/scripts/compare_matchers.py)
makes a git worktree per branch, builds each with colcon, runs that build's `lio_node` +
`gps_ground_truth_node`, and replays the bag with
[paced_bag_player.py](lidar_inertial_odometer/scripts/paced_bag_player.py) -- a player that
republishes the serialized bag messages and holds the next scan until `~/odometry` answers
the previous one, so a slow matcher never drops scans and a fast one never waits.
[evaluate_trajectories.py](lidar_inertial_odometer/scripts/evaluate_trajectories.py) then
aligns each estimate to the GPS GT by **yaw only** on the first straight segment (the initial
IMU heading is not reliable), and prints/plots xy ATE, end-point drift, KITTI-style relative
error over 100-800 m segments (median; GPS multipath inflates the mean) and the per-scan
ICP time parsed from the node log.

```bash
python3 src/lidar-inertial-odometer/lidar_inertial_odometer/scripts/compare_matchers.py \
  --bag ~/data/kitti/lidar --work ~/lio_eval                    # p2plane tight/loose, p2point, fused, fused alpha=0
python3 .../compare_matchers.py --bag ~/data/kitti/lidar --work ~/lio_eval --common-frontend   # same front-end for all
python3 .../compare_matchers.py --work ~/lio_eval --evaluate-only                              # tables and plots only
```

Results on KITTI 2011_09_30_0028 are in
[results/matcher_comparison/README.md](lidar_inertial_odometer/results/matcher_comparison/README.md):
with a common front-end, tightly coupled point-to-plane has the lowest xy ATE (7.5 m over
4.2 km, 80 ms ICP), fused is second (8.6 m, 18 ms) and the best per millisecond, loose
point-to-plane / point-to-point / fused alpha=0 tie at 10.6-10.9 m (14 / 96 / 15 ms).

The ROS1 trajectory numbers are not a ROS2 benchmark; evaluate a full converted bag
before comparing matching accuracy on real data.
