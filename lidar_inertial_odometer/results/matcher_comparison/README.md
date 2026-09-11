# Matcher comparison on KITTI 2011_09_30_drive_0028 (2026-09-11)

Bag `~/data/kitti/lidar` (rosbag2, 538 s, 5177 scans, 4.2 km). Every variant was run offline
with `scripts/compare_matchers.py` (paced playback, no dropped scans; 5153 or 5162 frames
processed depending on `init_imu_samples`). GPS GT from `gps_ground_truth` pinned to the LIO
start; **rotation (yaw) only** aligned on the first straight GT segment (36-40 m). No
translation, no scale, no loop closure. Machine: 28 cores, another idle `lio_node` of the
user was running during the whole session.

Commits: `point-to-plane-icp` 8cee590, `point-to-point-icp` 57f48a9, `fused-point-plane-icp` 9b56882.

## Round A -- each branch with its own committed `kitti.yaml`

Front-ends differ (p2plane: ring 30-63, loam_curvature, 0.7 m voxels, init 1 sample /
p2point: ring 16-47, PCA, 0.4 m, ICP 50x50 iterations, 3.0 m pairs, Huber 0.1 /
fused: ring 31-63, PCA, 0.5 m, init 100 samples), so this round compares *configurations*,
not only matchers.

| method | xy ATE rmse / max [m] | xy final drift [%] | seg 100-800 m median / mean [%] | ICP ms mean / med / p95 | pipeline ms / scan |
|---|---|---|---|---|---|
| p2plane-tight | 19.16 / 33.62 | 0.66 | 1.31 / 2.57 | 85.8 / 87.1 / 107 | 96 |
| p2plane-loose | 33.58 / 97.64 | 2.33 | 1.91 / 5.61 | 18.1 / 15.9 / 40 | 42 |
| p2point | 132.0 / 292.8 | 3.82 | 10.16 / 14.76 | 90.8 / 65.5 / 255 | 104 |
| **fused** (alpha 0.01, beta 1) | **8.64 / 27.22** | **0.29** | **0.59** / 2.02 | 17.7 / 14.1 / 40 | 41 |
| fused alpha=0 (control) | 10.68 / 28.18 | 0.35 | 0.59 / 2.03 | 14.9 / 12.4 / 32 | 38 |

Full tables: [round_a_own_configs/summary.md](round_a_own_configs/summary.md), plots
`top_view.png`, `error_vs_time.png`, `timing_and_segments.png` in the same directory.

## Round B -- common front-end (fused branch settings), own ICP parameters

`ring_min 31, ring_max 63, normal_method neighborhood_pca, feature/map voxel 0.5 m, init_imu_samples 100`
overridden on every variant; `fused` and `fused alpha=0` are the same runs as above.

| method | xy ATE rmse / max [m] | xy final drift [%] | seg 100-800 m median / mean [%] | ICP ms mean / med / p95 | pipeline ms / scan |
|---|---|---|---|---|---|
| p2plane-tight | **7.53 / 26.77** | **0.18** | 0.67 / 2.07 | 79.8 / 73.5 / 127 | 91 |
| p2plane-loose | 10.59 / 28.26 | 0.37 | 0.60 / 2.03 | **14.2 / 11.6 / 31** | **38** |
| p2point | 10.85 / 27.55 | 0.11 | 0.62 / 2.05 | 96.4 / 84.8 / 188 | 116 |
| **fused** (alpha 0.01, beta 1) | 8.64 / 27.22 | 0.29 | **0.59** / 2.02 | 17.7 / 14.1 / 40 | 41 |
| fused alpha=0 (control) | 10.68 / 28.18 | 0.35 | 0.59 / 2.03 | 14.9 / 12.4 / 32 | 38 |

Full tables: [round_b_common_frontend/summary.md](round_b_common_frontend/summary.md).

## Reading the numbers

- **Segment mean vs median.** The mean relative error (~2-2.5 % for every method) is set by a
  GPS multipath stretch at GT path 680-810 m (the zigzag near (150, -130) m in the top view
  that no estimate follows); the worst segments are identical for all methods. The median
  (0.6-0.7 %) reflects the odometry.
- **Final drift** is a single end-point number; `p2point 0.11 %` in round B is luck of the end
  point (its ATE is the worst of the round). Use xy ATE and the segment median.
- **z** is not compared: every variant climbs 8-18 m rms in z against GPS altitude (gravity /
  pitch alignment, common to the pipeline).

## Conclusion

1. Most of the round-A gap between branches is front-end configuration, not the matcher:
   on the common front-end, loose point-to-plane, point-to-point and fused alpha=0 land within
   10.6-10.9 m ATE / 0.59-0.62 % segment median of each other (fused alpha=0 reproduces the
   point-to-plane branch as expected).
2. **Accuracy:** tightly coupled point-to-plane is the best globally (7.5 m ATE, 0.18 % end
   drift), fused second (8.6 m, 0.29 %); locally (segment median) fused is best or tied
   (0.59 %) and tight slightly worse (0.67 %).
3. **Cost:** loose point-to-plane 14 ms, fused 18 ms (+3.5 ms for the point term), tight 80 ms
   (5.6x), point-to-point 96 ms with a 188 ms p95 -- at 10 Hz only the first two leave
   headroom for the rest of the pipeline (features ~25 ms).
4. **Point-to-point** alone is not usable here: with its own config it diverges (132 m ATE);
   with the common front-end it matches loose point-to-plane accuracy at 7x the cost.
5. Best accuracy per millisecond: fused (18 % lower ATE than loose point-to-plane for 3.5 ms).
   Best absolute accuracy: tight, if 80 ms ICP per scan is acceptable. Combining the two
   (fused residual inside the tightly coupled problem) is the obvious next experiment.

## Reproduce

```bash
python3 lidar_inertial_odometer/scripts/compare_matchers.py --bag ~/data/kitti/lidar --work ~/lio_eval
python3 lidar_inertial_odometer/scripts/compare_matchers.py --bag ~/data/kitti/lidar --work ~/lio_eval --common-frontend
# tables / plots only, from runs/ in this directory:
python3 lidar_inertial_odometer/scripts/compare_matchers.py --work lidar_inertial_odometer/results/matcher_comparison --evaluate-only
```

`runs/<variant>/` holds `est.txt`, `gt.txt` (TUM), `lio.log` (per-frame ICP time), `player.log`, `wall.txt`.
