#!/usr/bin/env python3
"""비교군 궤적을 GT 와 함께 subplot 으로 화면에 띄운다 (png 저장이 아니라 창).

compare_matchers.py 가 남긴 runs/<variant>/est.txt, gt.txt 를 읽어 방법마다 하나씩
subplot 을 만든다. 기본 4개(p2plane-tight, p2plane-loose, p2point, fused)면 2x2.
정렬은 evaluate_trajectories.py 와 같다: GT 첫 직진 구간으로 yaw 만 맞추고,
평행이동·스케일은 없다. 그리기 전에 방법별 ATE 와 계산 시간(프레임별 ICP ms, 전체
pipeline wall) 표를 stdout 에 출력하고, subplot 제목에도 같은 값을 적는다.

    python3 show_trajectories.py                                   # results/matcher_comparison/runs 의 4개
    python3 show_trajectories.py --common-frontend                 # *-cfe 런 (fused 는 그대로)
    python3 show_trajectories.py --runs ~/lio_eval/runs --variants fused,p2plane-tight
    python3 show_trajectories.py --save /tmp/trajectories.png      # 창도 띄우고 파일로도 저장
"""

import argparse
import math
import os
import sys

import numpy as np

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
PACKAGE_DIR = os.path.dirname(SCRIPTS_DIR)
sys.path.insert(0, SCRIPTS_DIR)
import evaluate_trajectories as ev  # noqa: E402

DEFAULT_RUNS = os.path.join(PACKAGE_DIR, 'results', 'matcher_comparison', 'runs')
DEFAULT_VARIANTS = ['p2plane-tight', 'p2plane-loose', 'p2point', 'fused']
LABELS = {
    'p2plane-tight': 'point-to-plane, tightly coupled',
    'p2plane-loose': 'point-to-plane, initial guess only',
    'p2point': 'point-to-point',
    'fused': 'fused point-to-point + point-to-plane',
    'fused-alpha0': 'fused, alpha = 0 (pure point-to-plane)',
}


def load_run(runs_dir, name, straight_length, straight_max_turn):
    """Time-associated, yaw-aligned estimate and GT positions for one run."""
    est = ev.load_tum(os.path.join(runs_dir, name, 'est.txt'))
    gt = ev.load_tum(os.path.join(runs_dir, name, 'gt.txt'))
    est_a, gt_a = ev.associate(est, gt)
    est_p, gt_p = est_a[:, 1:4], gt_a[:, 1:4]
    est_p = est_p - est_p[0] + gt_p[0]
    est_aligned, yaw_deg, _ = ev.align_yaw_on_straight(est_p, gt_p, straight_length, straight_max_turn)
    xy_error = np.linalg.norm((est_aligned - gt_p)[:, :2], axis=1)
    error_3d = np.linalg.norm(est_aligned - gt_p, axis=1)
    length = ev.path_length(gt_p)[-1]

    # Computation time: per-scan ICP time from the node log, whole-pipeline wall time from wall.txt.
    log_path = os.path.join(runs_dir, name, 'lio.log')
    timing = ev.load_timing(log_path) if os.path.exists(log_path) else np.zeros((0, 3))
    icp_ms = timing[:, 1] if len(timing) else np.array([np.nan])
    wall_path = os.path.join(runs_dir, name, 'wall.txt')
    wall_s = float(open(wall_path).read().split('=')[1]) if os.path.exists(wall_path) else np.nan

    return {
        'name': name, 'est': est_aligned, 'gt': gt_p, 'yaw_deg': yaw_deg,
        'xy_rmse': float(np.sqrt(np.mean(xy_error ** 2))), 'xy_max': float(np.max(xy_error)),
        'ate_rmse': float(np.sqrt(np.mean(error_3d ** 2))), 'z_rmse': float(np.sqrt(np.mean((est_aligned - gt_p)[:, 2] ** 2))),
        'drift_pct': float(xy_error[-1] / length * 100.0), 'length': length, 'frames': len(est),
        'icp_mean': float(np.nanmean(icp_ms)), 'icp_median': float(np.nanmedian(icp_ms)),
        'icp_p95': float(np.nanpercentile(icp_ms, 95)), 'icp_max': float(np.nanmax(icp_ms)), 'timed_frames': len(timing),
        'wall_s': wall_s, 'wall_ms_per_scan': wall_s / len(est) * 1000.0 if len(est) else np.nan,
    }


def print_table(runs):
    """ATE and computation time per variant, on stdout."""
    header = (f"{'method':<20} {'frames':>6} {'path[m]':>8} | {'xy ATE rmse':>11} {'xy max':>7} {'3D rmse':>8} {'z rmse':>7} {'end drift':>9} | "
              f"{'ICP mean':>8} {'median':>7} {'p95':>7} {'max':>6} | {'pipeline':>8} {'ms/scan':>7}")
    print(header)
    print('-' * len(header))
    for r in runs:
        print(f"{r['name']:<20} {r['frames']:>6} {r['length']:>8.0f} | {r['xy_rmse']:>9.2f} m {r['xy_max']:>5.1f} m {r['ate_rmse']:>6.2f} m "
              f"{r['z_rmse']:>5.2f} m {r['drift_pct']:>7.2f} % | {r['icp_mean']:>5.1f} ms {r['icp_median']:>4.1f} ms {r['icp_p95']:>4.1f} ms "
              f"{r['icp_max']:>4.0f} ms | {r['wall_s']:>6.0f} s {r['wall_ms_per_scan']:>5.1f} ms")
    print('ATE: yaw-only alignment on the first straight GT segment, no translation/scale. '
          'ICP: per-scan time from lio.log; pipeline: wall time of the whole run (features + ICP + ROS).')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--runs', default=DEFAULT_RUNS, help='directory holding <variant>/est.txt and gt.txt')
    parser.add_argument('--variants', default=','.join(DEFAULT_VARIANTS), help='comma separated run names, one subplot each')
    parser.add_argument('--common-frontend', action='store_true', help='use the <variant>-cfe runs where they exist')
    parser.add_argument('--straight-length', type=float, default=40.0)
    parser.add_argument('--straight-max-turn', type=float, default=5.0)
    parser.add_argument('--save', default='', help='also write the figure to this png')
    args = parser.parse_args()

    names = [n.strip() for n in args.variants.split(',') if n.strip()]
    if args.common_frontend:
        names = [n + '-cfe' if os.path.isdir(os.path.join(args.runs, n + '-cfe')) else n for n in names]

    runs = []
    for name in names:
        if not os.path.exists(os.path.join(args.runs, name, 'est.txt')):
            print(f'skipping {name}: no est.txt under {args.runs}', file=sys.stderr)
            continue
        runs.append(load_run(args.runs, name, args.straight_length, args.straight_max_turn))
    if not runs:
        raise SystemExit(f'no runs found under {args.runs}')
    print_table(runs)

    import matplotlib.pyplot as plt  # after argparse so --help works without a display

    columns = 2 if len(runs) > 1 else 1
    rows = math.ceil(len(runs) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(6.5 * columns, 6.0 * rows), squeeze=False)
    fig.suptitle('LIO trajectories vs GPS GT (yaw aligned on the first straight segment)', fontsize=13)

    # Same axis limits everywhere so the subplots are visually comparable.
    all_xy = np.vstack([np.vstack([r['est'][:, :2], r['gt'][:, :2]]) for r in runs])
    low, high = all_xy.min(axis=0) - 20.0, all_xy.max(axis=0) + 20.0

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for i, (ax, r) in enumerate(zip(axes.flat, runs)):
        gt, est = r['gt'], r['est']
        ax.plot(gt[:, 0], gt[:, 1], color='black', linewidth=2.0, label='GPS GT')
        ax.plot(est[:, 0], est[:, 1], color=colors[i % len(colors)], linewidth=1.3, label='estimate')
        ax.plot(gt[0, 0], gt[0, 1], 'ko', markersize=6, label='start')
        ax.plot(gt[-1, 0], gt[-1, 1], 'ks', markersize=5, markerfacecolor='none', label='GT end')
        ax.plot(est[-1, 0], est[-1, 1], 's', color=colors[i % len(colors)], markersize=5, label='estimate end')
        base = r['name'][:-4] if r['name'].endswith('-cfe') else r['name']
        ax.set_title(f"{r['name']}: {LABELS.get(base, base)}\n"
                     f"xy ATE {r['xy_rmse']:.2f} m (max {r['xy_max']:.1f}), end drift {r['drift_pct']:.2f} % of {r['length']:.0f} m, "
                     f"{r['frames']} frames, yaw fix {r['yaw_deg']:+.1f} deg\n"
                     f"ICP {r['icp_mean']:.1f} ms mean / {r['icp_median']:.1f} median / {r['icp_p95']:.1f} p95, "
                     f"pipeline {r['wall_ms_per_scan']:.1f} ms/scan", fontsize=9.5)
        ax.set_xlim(low[0], high[0]); ax.set_ylim(low[1], high[1])
        ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
        ax.set_xlabel('east [m]'); ax.set_ylabel('north [m]')
        ax.legend(loc='lower left', fontsize=8)
    for ax in list(axes.flat)[len(runs):]:
        ax.set_visible(False)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    if args.save:
        fig.savefig(args.save, dpi=130)
        print(f'saved {args.save}')
    plt.show()


if __name__ == '__main__':
    main()
