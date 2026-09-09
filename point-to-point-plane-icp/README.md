# Point-to-point-plane ICP

A standalone C++17/Ceres library and ROS 2 `ament_cmake` package named
`point_to_point_plane_icp`. No ROS headers are used in the algorithm.

For each correspondence and the shared pose `xi = [t, alpha, beta, gamma]`:

```text
e = Rz(gamma) Ry(beta) Rx(alpha) x + t - y
J_e = [ I | (dR/dalpha)x | (dR/dbeta)x | (dR/dgamma)x ]
r_point = sqrt(w_point) * e
r_plane = sqrt(w_plane) * n_y^T e
J_point = sqrt(w_point) * J_e
J_plane = sqrt(w_plane) * n_y^T J_e
```

`SharedErrorEvaluation::PrepareForEvaluation()` computes **one error vector and one
Jacobian per correspondence** before residual evaluation. Both cost objects hold the
same `shared_ptr<const SharedError>`. The callback updates the cache for every new
Ceres evaluation point and supports a residual-only evaluation followed by a Jacobian
evaluation at the same point. The cost objects only read that cache, including when
Ceres evaluates blocks concurrently. There is no global cache or cross-solve state.

The two calls in `AddCorrespondenceResiduals()` are intentionally separate:

```cpp
problem.AddResidualBlock(new PointCostFunction(error, point_weight), loss, xi);
problem.AddResidualBlock(new PlaneCostFunction(error, plane_weight), loss, xi);
```

Every valid correspondence therefore adds two blocks (3D + 1D) on **one** six-parameter
pose block. Both weights must be finite and positive and default to 1. With no loss,
Ceres minimizes `0.5 * sum(w_point * ||e||^2 + w_plane * (n^T e)^2)`.
When enabled, Huber loss is applied to each block separately after residual scaling.
The normal component is present in both terms; the weights determine its extra emphasis.

The ICP loop searches target nearest neighbours, rejects distance/normal disagreement,
normalizes supplied normals, and rejects targets without a usable normal. It optimizes
centroid-pivoted increments, composes them, re-searches, and stops on small steps or
stalled error. `final_error` reports the RMS combined weighted residual norm before
robustification; no correspondences or unusable Ceres solutions yield infinity.

Do not call the cost objects with stale/uninitialized cache data: register their
`SharedErrorEvaluation` in `ceres::Problem::Options::evaluation_callback`. The callback
must outlive the problem, and its parameter array must remain alive throughout the solve.
`IcpPointToPointPlane` handles this lifetime internally.

```bash
# Standalone
cmake -S point-to-point-plane-icp -B /tmp/hybrid-icp -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/hybrid-icp -j2
ctest --test-dir /tmp/hybrid-icp --output-on-failure

# Matching demo on data/*.txt: timing plus point/plane residuals (--help lists the options)
/tmp/hybrid-icp/hybrid_icp_demo --repeat 10
/tmp/hybrid-icp/hybrid_icp_demo --initial-pose 0.3 -0.1 0.05 0.02 -0.03 0.05 --expected-pose -0.2 -0.2 0 0 0 0

# ROS 2 workspace (build this dependency before LIO)
colcon build --packages-select point_to_point_plane_icp gps_ground_truth lidar_inertial_odometer \
  --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
source install/setup.bash
ros2 launch lidar_inertial_odometer kitti_lio.launch.py
```

`hybrid_icp_demo` registers `data/source_point.txt` against `data/target_point.txt`
(ROS 2: `ros2 run point_to_point_plane_icp hybrid_icp_demo`). It reports the wall time of
the TXT load, the setup plus target kd-tree and the ICP loop separately over repeated runs
(mean/median/p95/min/max), then the correspondence count, the hybrid RMS, the point and
plane RMS and the recovered pose; `--expected-pose` adds the translation and rotation error.
Build Release before quoting any number. It exits 0 when every measured run converged,
1 on an input error and 2 otherwise.

Tests verify numerical Jacobians at nonzero poses, cache updates, 3+1 residual
layout, weights, known-transform recovery, tangential motion on a single plane,
invalid normals, missing correspondences and invalid weights.
