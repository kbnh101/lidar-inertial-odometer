#!/usr/bin/env python3
"""추정 궤적과 GT(GPS) 궤적을 top-view 로 겹쳐 그린다.

kitti_lio.launch 를 돌린 직후라면 인자 없이 그냥 실행하면 된다. 기본 입력은
config/kitti.yaml 의 trajectory_csv / gt_trajectory_csv 와 같은 경로다.

    python3 plot_trajectory.py
    python3 plot_trajectory.py --align
    python3 plot_trajectory.py --est /tmp/lio_trajectory.txt --gt /tmp/lio_gt.txt

두 궤적은 시각으로 짝지어 비교한다. --align 을 주면 SE(2)(yaw + 평행이동)
Umeyama 정합을 먼저 적용한다. odometry 의 world frame 은 첫 자세 기준이므로,
IMU 절대 방위로 초기 yaw 를 맞추지 않았다면 이 옵션이 필요하다.
"""

import argparse
import os

import numpy as np

# 노드가 쓰는 기본 출력 경로 (config/kitti.yaml 과 일치시킬 것).
DEFAULT_EST = '/home/chanho/git_ws/tmp_ws/src/lidar_inertial_odometer/results/lio_trajectory.txt'
DEFAULT_GT = '/home/chanho/git_ws/tmp_ws/src/lidar_inertial_odometer/results/lio_gt.txt'
# 그림은 cwd 가 아니라 패키지 안 results/ 로 떨어뜨린다. 어디서 실행하든
# 같은 자리에 쌓이게 하려는 것이다 (scripts/ 의 상위가 패키지 루트).
PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUT = os.path.join(PACKAGE_ROOT, 'results', 'trajectory_top_view.png')


def load_tum(path):
    """TUM 포맷(timestamp tx ty tz qx qy qz qw) 을 (N,4) 배열로 읽는다."""
    rows = []
    with open(path, 'r') as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            rows.append([float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])])
    if not rows:
        raise SystemExit(f'no poses in {path}')
    return np.array(rows)


def associate(estimate, ground_truth, max_difference=0.05):
    """가장 가까운 시각끼리 짝지어 (est_xyz, gt_xyz) 를 돌려준다."""
    gt_times = ground_truth[:, 0]
    pairs_est = []
    pairs_gt = []
    for row in estimate:
        index = int(np.argmin(np.abs(gt_times - row[0])))
        if abs(gt_times[index] - row[0]) <= max_difference:
            pairs_est.append(row[1:4])
            pairs_gt.append(ground_truth[index, 1:4])
    if not pairs_est:
        raise SystemExit('시각이 겹치는 pose 쌍이 없다 -- max_difference 를 늘려 볼 것')
    return np.array(pairs_est), np.array(pairs_gt)


def align_se2(source, target):
    """xy 평면에서 yaw + 평행이동만으로 source 를 target 에 맞춘다 (스케일 고정)."""
    source_xy = source[:, :2]
    target_xy = target[:, :2]
    source_mean = source_xy.mean(axis=0)
    target_mean = target_xy.mean(axis=0)
    centered_source = source_xy - source_mean
    centered_target = target_xy - target_mean

    covariance = centered_target.T @ centered_source
    u, _, vt = np.linalg.svd(covariance)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:          # 반사(reflection) 제거
        u[:, -1] *= -1
        rotation = u @ vt

    aligned = np.array(source)
    aligned[:, :2] = centered_source @ rotation.T + target_mean
    aligned[:, 2] = source[:, 2] - source[:, 2].mean() + target[:, 2].mean()
    return aligned


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--est', default=DEFAULT_EST,
                        help=f'추정 궤적 (TUM). 기본값 {DEFAULT_EST}')
    parser.add_argument('--gt', default=DEFAULT_GT,
                        help=f'GT / GPS 궤적 (TUM). 기본값 {DEFAULT_GT}')
    parser.add_argument('--out', default=DEFAULT_OUT,
                        help=f'저장할 png. 기본값 {DEFAULT_OUT}')
    parser.add_argument('--align', action='store_true',
                        help='SE(2) Umeyama 정합 후 비교')
    parser.add_argument('--max-difference', type=float, default=0.05,
                        help='시각 매칭 허용 오차 [s]')
    args = parser.parse_args()

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    estimate = load_tum(args.est)
    ground_truth = load_tum(args.gt)
    matched_estimate, matched_gt = associate(estimate, ground_truth, args.max_difference)

    if args.align:
        matched_estimate = align_se2(matched_estimate, matched_gt)

    errors = np.linalg.norm(matched_estimate - matched_gt, axis=1)
    ate_rmse = float(np.sqrt((errors ** 2).mean()))
    travelled = float(np.linalg.norm(np.diff(matched_gt, axis=0), axis=1).sum())

    print(f'matched pairs : {len(errors)}')
    print(f'GT 주행 거리  : {travelled:.1f} m')
    print(f'ATE RMSE      : {ate_rmse:.3f} m   (mean {errors.mean():.3f}, '
          f'max {errors.max():.3f})')
    if travelled > 0:
        print(f'상대 오차     : {100.0 * ate_rmse / travelled:.2f} % of path length')

    figure, axes = plt.subplots(1, 2, figsize=(14, 6),
                                gridspec_kw={'width_ratios': [3, 2]})

    axes[0].plot(matched_gt[:, 0], matched_gt[:, 1], linewidth=2.0,
                 label='GT (GPS/OXTS)', color='#444444')
    axes[0].plot(matched_estimate[:, 0], matched_estimate[:, 1], linewidth=1.6,
                 label='LiDAR-Inertial Odometry', color='#d1495b')
    axes[0].scatter(matched_gt[0, 0], matched_gt[0, 1], marker='o', s=60,
                    color='#2a9d8f', zorder=5, label='start')
    axes[0].set_xlabel('x [m] (East)')
    axes[0].set_ylabel('y [m] (North)')
    axes[0].set_title(f'Top view — ATE RMSE {ate_rmse:.2f} m / {travelled:.0f} m')
    axes[0].axis('equal')
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    time_axis = np.arange(len(errors))
    axes[1].plot(time_axis, errors, linewidth=1.2, color='#d1495b')
    axes[1].axhline(ate_rmse, linestyle='--', color='#444444',
                    label=f'RMSE {ate_rmse:.2f} m')
    axes[1].set_xlabel('frame')
    axes[1].set_ylabel('position error [m]')
    axes[1].set_title('Absolute trajectory error')
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    figure.tight_layout()
    out_directory = os.path.dirname(os.path.abspath(args.out))
    if out_directory:
        os.makedirs(out_directory, exist_ok=True)
    figure.savefig(args.out, dpi=150)
    print(f'saved {args.out}')


if __name__ == '__main__':
    main()
