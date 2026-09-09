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

On the CPU backend, `SharedErrorEvaluation::PrepareForEvaluation()` computes **one
error vector and one Jacobian per correspondence** before residual evaluation. Both cost objects hold the
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
cmake -S point-to-point-plane-icp -B /tmp/hybrid-icp -DCMAKE_BUILD_TYPE=Release \
  -DP2PTPL_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build /tmp/hybrid-icp -j2
ctest --test-dir /tmp/hybrid-icp --output-on-failure

# Matching demo on data/*.txt: timing plus point/plane residuals (--help lists the options)
/tmp/hybrid-icp/hybrid_icp_demo --repeat 10
/tmp/hybrid-icp/hybrid_icp_demo --backend cpu --repeat 10
/tmp/hybrid-icp/hybrid_icp_demo --initial-pose 0.3 -0.1 0.05 0.02 -0.03 0.05 --expected-pose -0.2 -0.2 0 0 0 0

# ROS 2 workspace (build this dependency before LIO)
colcon build --packages-select point_to_point_plane_icp gps_ground_truth lidar_inertial_odometer \
  --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON \
  -DP2PTPL_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89
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

## CUDA backend

`P2PTPL_ENABLE_CUDA` defaults to `ON`. CUDA builds default `IcpOptions::use_cuda` to
`true`; set it to `false` to use the original CPU evaluator. Both `hybrid_icp_demo`
and `bag_icp_benchmark` accept `--backend cpu|cuda`; the ROS node exposes
`icp_use_cuda`. The startup output identifies the selected backend.

`CudaErrorEvaluation` uploads the fixed correspondences once per Ceres solve. One
CUDA thread computes one correspondence's shared error, weighted 3D point residual,
1D plane residual and both analytic Jacobians. The pose-wide rotation and derivative
matrices are prepared once on the CPU. The kernel uses doubles and no fast-math.
Two separate Ceres cost blocks copy from a pinned host cache, preserving independent
Huber losses after weighting. GPU completion is synchronized before Ceres reads the
cache, including multithreaded block evaluation. Device and pinned allocations are
reused across the outer iterations of a registration. Residual-only requests skip
Jacobian computation and transfer; a later same-pose Jacobian request refreshes them.

The stock Ceres >= 2.1 `EvaluationCallback` API is sufficient. No custom Ceres fork
or GPU rebuild is needed: see the official
[EvaluationCallback documentation](https://ceres-solver.org/nnls_modeling.html#evaluationcallback).
The six-parameter `DENSE_QR` solve, robustification, nearest-neighbour search and
diagnostic RMS remain on the CPU. GPU initialization or execution errors are
reported explicitly. There is no automatic CPU fallback after selecting CUDA.

Build without any CUDA dependency:

```bash
cmake -S point-to-point-plane-icp -B /tmp/hybrid-cpu \
  -DCMAKE_BUILD_TYPE=Release -DP2PTPL_ENABLE_CUDA=OFF
cmake --build /tmp/hybrid-cpu -j4
ctest --test-dir /tmp/hybrid-cpu --output-on-failure
```

The CPU build defaults to CPU evaluation and rejects an explicit CUDA request.
GPU tests require a working NVIDIA device and fail if it is unavailable.

Validated on RTX 4060 Laptop, CUDA 12.3, Ceres 2.1, Release, architecture `89`:

- CPU and CUDA standalone suites pass. CUDA tests compare weighted residuals and
  all 24 Jacobian entries with CPU results (absolute tolerance `5e-13`), and compare
  analytic Jacobians against central differences (`1e-8`). They cover partial CUDA
  blocks, buffer growth/reuse, stale caches, residual-to-Jacobian upgrades,
  multithreaded Ceres evaluation and separate Huber losses.
- Noisy/outlier registration agrees with CPU pose within `1e-7` matrix norm.
- ROS 2 package tests and the message/TF/GPS integration test pass.
- `compute-sanitizer --tool memcheck --error-exitcode 99 /tmp/hybrid-icp/test_cuda_icp`
  reports zero errors.

GPU evaluation does not guarantee a faster full ICP solve. With the bundled 300-point
dataset, 5 warmups and 50 measured registrations, CPU/GPU ICP medians were
6.94/10.93 ms on this machine; both converged 50/50 and recovered the reference pose
to below `2e-14` m translation error. This small workload does not amortize CUDA
launch/transfer costs. These times include correspondence search and Ceres, and
exclude process initialization warmup. Compare both backends on the intended scan
size before using these numbers to infer real-time performance.

A 15-pair smoke comparison on the local GLIM example bag (`/hesai/pandar`, starting
at 10 s, all rings, PCA normals, IMU disabled) also produced identical correspondence
counts, outer iteration counts and convergence flags. At the CSV's nine decimal
places, residual RMS and translation components matched; the largest translation
magnitude / rotation-angle differences were `1e-9` m / `2e-9` degrees. Both backends
converged on 7/15 pairs under the same 12-iteration limit (benchmark exit code 2).
This checks numerical parity on recorded data; it is not a trajectory accuracy
validation. With about 2,136 correspondences, single-pass median ICP times were
56.50 ms CPU / 62.76 ms CUDA. CUDA did not improve overall ICP time on either measured
workload; host block processing, QR and device transfers are still part of each solve.

```bash
# Run once with cpu, then cuda; use separate CSV outputs.
ros2 run lidar_inertial_odometer bag_icp_benchmark \
  --bag /path/to/glim_example --cloud-topic /hesai/pandar \
  --imu-topic '' --imu-time header --no-imu-rotation \
  --ring-all --normal-method neighborhood_pca --start 10 --frames 15 \
  --backend cuda --csv /tmp/lio-bag-cuda.csv
```
