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

`SharedErrorEvaluation::PrepareForEvaluation()` always computes on CPU **one
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
  -DP2PTPL_ENABLE_CUDA=OFF
cmake --build /tmp/hybrid-icp -j2
ctest --test-dir /tmp/hybrid-icp --output-on-failure

# Matching demo on data/*.txt: timing plus point/plane residuals (--help lists the options)
/tmp/hybrid-icp/hybrid_icp_demo --repeat 10
/tmp/hybrid-icp/hybrid_icp_demo --backend cpu --repeat 10
/tmp/hybrid-icp/hybrid_icp_demo --initial-pose 0.3 -0.1 0.05 0.02 -0.03 0.05 --expected-pose -0.2 -0.2 0 0 0 0

# ROS 2 workspace (build this dependency before LIO)
colcon build --packages-select point_to_point_plane_icp gps_ground_truth lidar_inertial_odometer \
  --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON \
  -DP2PTPL_ENABLE_CUDA=OFF
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

### Module timing

`IcpOptions::collect_timing` (both demos enable it) fills `IcpResult::timing` and every
`IcpIterationLog::timing` with per-module host wall time, reported at three levels so a slow
registration can be traced from the pipeline down to the outer iteration that caused it:

1. the host stages around `do_icp()`: TXT/PointCloud2 load, source copy, target kd-tree, the call;
2. the modules inside `do_icp()`: `correspondence`, `pivot`, `problem_setup`, `residual_block`,
   `ceres_solve`, `problem_cleanup`, `pose_update`, `diagnostic`, `other`. They are disjoint, add
   up to `total_ms`, and each row shows its share of it. The `ceres_*` and `callback_*` rows below
   them measure work **inside** `ceres_solve_ms` and overlap each other, so never sum those;
3. per outer iteration, with `--iterations` (demo) or for the last scan pair (bag benchmark) --
   this is where a one-off cost such as the first CUDA solve becomes visible.

`hybrid_icp_demo --csv` writes every module and call count per registration;
`bag_icp_benchmark --csv` adds the same columns to its per-frame file with an `icp_` prefix.
`test_icp_timing` prints the module table for 500/2000/8000-point registrations and, when Ceres
has CUDA, for CPU and GPU dense QR on the same 8000-point workload; `test_hybrid_icp` and
`test_ceres_cuda_icp` print the breakdown of the registrations they already run. No test asserts
on a duration -- the numbers are informational and only meaningful in a Release build.

## CUDA backend

Residuals and analytic Jacobians always run on the CPU through
`SharedErrorEvaluation`. `IcpOptions::use_cuda` / ROS `icp_use_cuda` / demo
`--backend cpu|cuda` select **only the Ceres dense linear solver**:

```text
CPU: correspondence search -> residual/Jacobian -> robust loss + system assembly
GPU: Ceres DENSE_QR (when use_cuda=true)
CPU: step acceptance + pose update
```

The default is CUDA when the linked Ceres library was built with CUDA support;
otherwise it is Eigen CPU QR. GPU mode sets
`dense_linear_algebra_library_type = ceres::CUDA` with `DENSE_QR` and requires
Ceres built with `USE_CUDA=ON`. The installed `/usr/local` Ceres 2.1 already supports
it. See the official
[CUDA DENSE_QR documentation](https://ceres-solver.org/nnls_solving.html#dense-qr).
The Ceres context is initialized lazily and reused across outer iterations and scans.
`IcpIterationLog::used_cuda_solver` records the backend reported by Ceres.
Selecting CUDA explicitly fails if unavailable; it does not silently switch to CPU.

`P2PTPL_ENABLE_CUDA=OFF` is the default and disables building the **legacy custom
CUDA residual evaluator**. It does not disable CUDA inside Ceres. This package
requires no CUDA compilation in that configuration; the linked CUDA-enabled Ceres
still requires its CUDA runtime libraries and an NVIDIA GPU. To run entirely on CPU
with the same installation, use `--backend cpu` or `icp_use_cuda: false`. A Ceres build
without CUDA support also defaults to CPU and rejects explicit GPU requests.

`test_ceres_cuda_icp` compares CPU residuals with CPU/GPU QR on noisy/outlier data,
checks Ceres' actual backend, repeated solves and switching back to CPU. It is built
independently of the legacy kernels, skips when Ceres lacks CUDA support, and
requires a working GPU when Ceres supports CUDA. The ordinary hybrid tests cover
residuals, Jacobians, shared cache updates and registration.

The previous `CudaErrorEvaluation` source and its direct tests remain available
through `-DP2PTPL_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89` for experimentation.
ICP does not call that evaluator even when it is compiled. The separate
`cuda-residual-tutorial` package continues to evaluate residuals on CUDA.

### CPU residual/Jacobian + GPU QR validation

On 2026-09-09, Release with `P2PTPL_ENABLE_CUDA=OFF` passed the hybrid residual,
Ceres CPU/GPU QR parity, and LIO core suites. A 20-second headless KITTI launch
reported `ICP residual/Jacobian backend: cpu; Ceres DENSE_QR: CUDA` and wrote
69 LIO / 70 GPS trajectory rows.

For the first 20 scan pairs of `/home/chanho/data/kitti/lidar`, using `/points_raw`,
`/imu_raw`, rings 16–47 (all), maximum range 150 m, identity rotation guesses,
constant-velocity translation and deskew off:

| Mean compute time | CPU residual/Jacobian + CPU QR | CPU residual/Jacobian + GPU QR |
|---|---:|---:|
| ICP search + Ceres | 103.15 ms | 148.55 ms |
| Per scan | 113.17 ms | 158.59 ms |

Correspondence counts, outer iterations and convergence flags matched. The CSV
residual RMS values matched at nine decimal places. Both runs reached the
12-iteration limit on all pairs (exit code 2), so this is a timing/parity smoke check,
not trajectory accuracy validation. Bag I/O, ROS and RViz are excluded. This short
comparison still favors CPU QR on this workload.

### Historical measurements

The following results predate the CPU residual/Jacobian + GPU QR configuration.

With CUDA QR enabled, a 20-pair KITTI smoke comparison on 2026-09-09 used
`/home/chanho/data/kitti/lidar`, `/points_raw`, `/imu_raw`, rings 16–47 (all),
maximum range 150 m, identity rotation initial guesses, constant-velocity translation,
and deskew off. Mean per-scan compute times were 117.84 ms CPU / 166.05 ms CUDA;
the ICP portions were 107.55 / 155.83 ms. Both runs reached the outer-iteration limit
on all 20 pairs (exit code 2) with mean hybrid RMS 0.274 m. These are short
scan-to-scan measurements excluding bag I/O and ROS/RViz, not full LIO timings or
trajectory accuracy validation. CUDA QR is active but did not improve this workload.
The installed ROS launch also produced LIO and GPS trajectories in a 20-second
headless playback check with `Ceres DENSE_QR: CUDA` in its startup log.

The timing measurements below are historical results from the residual/Jacobian-only
CUDA implementation, when Ceres QR still ran on the CPU. They do not measure the
current CUDA QR path. GPU evaluation does not guarantee a faster full ICP solve.
With the bundled 300-point
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
