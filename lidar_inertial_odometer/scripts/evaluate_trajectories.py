#!/usr/bin/env python3
"""여러 매칭 방식의 LIO 궤적을 GPS GT 와 비교해 drift 와 ICP 계산 시간을 표로 만든다.

정렬은 **회전(yaw)만** 맞춘다. 두 궤적 모두 LIO 시작점을 원점으로 쓰므로 평행이동은
없고, 초기 방위(IMU 절대 방위)가 부정확하므로 GT 의 첫 직진 구간 방향에 추정 궤적의
같은 구간 방향을 맞춘다. 직진 구간은 GT 진행 방향의 누적 변화가 `--straight-max-turn`
(deg) 을 넘기 전까지, 최대 `--straight-length` m 로 잡는다.

    python3 evaluate_trajectories.py --out results/compare \
        --run tight:est.txt:gt.txt:log.txt --run fused:est.txt:gt.txt:log.txt

지표
  ATE      yaw 정렬 후 시각 짝지은 위치 오차의 RMSE / 중앙값 / 최대 [m]
  final    마지막 프레임 위치 오차 [m], drift = final / GT 경로 길이 [%]
  seg      GT 경로 길이 100/200/…/800 m 구간마다 구간 첫 20 m 로 회전만 국소 정렬한 뒤
           xy 구간 변위 차이 / 구간 길이 [%] (KITTI 방식, GT 방위 없이 위치만 사용).
           대표값은 중앙값 -- GPS 멀티패스 구간이 평균을 모든 방법에서 똑같이 부풀린다.
  icp ms   로그의 프레임별 ICP 시간 (loosely: `[lio] ... rms X T ms`, tightly: `time=T ms`)
"""

import argparse
import os
import re
import sys

import numpy as np


# --------------------------------------------------------------------------- I/O
def load_tum(path):
    rows = []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            rows.append([float(v) for v in parts[:4]])
    if not rows:
        raise RuntimeError(f'no poses in {path}')
    return np.array(rows)


LOOSE_RE = re.compile(r'\[lio\] t=([0-9.]+) \| .*\| icp (ok |REJ) iter\s*(\d+) corr\s*(\d+) rms\s*([0-9.a-z]+)\s+([0-9.]+) ms')
TIGHT_RE = re.compile(r'\[tight-lio\] t=([0-9.]+) corr=(\d+) lidar=(ok|IMU) rms=([0-9.a-z]+) time=([0-9.]+) ms')


def load_timing(path):
    """Per-frame (timestamp, icp_ms, accepted) parsed from a node log."""
    records = []
    with open(path, errors='replace') as handle:
        for line in handle:
            match = LOOSE_RE.search(line)
            if match:
                records.append((float(match.group(1)), float(match.group(6)), match.group(2).strip() == 'ok'))
                continue
            match = TIGHT_RE.search(line)
            if match:
                records.append((float(match.group(1)), float(match.group(5)), match.group(3) == 'ok'))
    return np.array(records) if records else np.zeros((0, 3))


# --------------------------------------------------------------------------- geometry
def yaw_rotation(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def associate(est, gt, max_dt=0.05):
    """Pairs every estimate with the GT sample nearest in time (GT interpolated linearly)."""
    t_est, t_gt = est[:, 0], gt[:, 0]
    keep = (t_est >= t_gt[0]) & (t_est <= t_gt[-1])
    est = est[keep]
    gt_interp = np.column_stack([np.interp(est[:, 0], t_gt, gt[:, k]) for k in range(1, 4)])
    index = np.searchsorted(t_gt, est[:, 0])
    index = np.clip(index, 1, len(t_gt) - 1)
    gap = np.minimum(np.abs(t_gt[index] - est[:, 0]), np.abs(t_gt[index - 1] - est[:, 0]))
    ok = gap <= max_dt
    return est[ok], np.column_stack([est[ok, 0], gt_interp[ok]])


def path_length(points):
    return np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))])


def straight_segment(gt_xy, max_length, max_turn_deg):
    """Index range [0, end) of the initial GT segment that stays within max_turn_deg of heading change."""
    cum = path_length(gt_xy)
    headings = np.arctan2(np.diff(gt_xy[:, 1]), np.diff(gt_xy[:, 0]))
    moving = np.linalg.norm(np.diff(gt_xy, axis=0), axis=1) > 0.05
    reference = None
    end = len(gt_xy)
    for i in range(len(headings)):
        if not moving[i]:
            continue
        if reference is None:
            reference = headings[i]
        turn = np.degrees(np.abs(np.angle(np.exp(1j * (headings[i] - reference)))))
        if turn > max_turn_deg or cum[i + 1] > max_length:
            end = i + 1
            break
    return max(end, 2)


def heading_of(points_xy):
    """Principal direction of a 2-D segment, oriented from its start to its end."""
    centered = points_xy - points_xy.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    direction = vt[0]
    if np.dot(direction, points_xy[-1] - points_xy[0]) < 0:
        direction = -direction
    return np.arctan2(direction[1], direction[0])


def align_yaw_on_straight(est_p, gt_p, max_length, max_turn_deg):
    end = straight_segment(gt_p[:, :2], max_length, max_turn_deg)
    yaw = heading_of(gt_p[:end, :2]) - heading_of(est_p[:end, :2])
    R = yaw_rotation(yaw)
    return est_p @ R.T, np.degrees(yaw), end


def relative_yaw(est_rel, gt_rel):
    """Rotation-only 2-D least squares: yaw minimizing sum |R est_k - gt_k|^2 (closed form)."""
    cross = np.sum(est_rel[:, 0] * gt_rel[:, 1] - est_rel[:, 1] * gt_rel[:, 0])
    dot = np.sum(est_rel[:, 0] * gt_rel[:, 0] + est_rel[:, 1] * gt_rel[:, 1])
    return np.arctan2(cross, dot)


def segment_errors(est_p, gt_p, lengths, heading_window=20.0, stride=10.0):
    """KITTI-style relative translation error over GT path-length segments, positions only.

    GPS gives no orientation, so the pose alignment at the segment start is replaced by a
    rotation-only fit of the first `heading_window` metres of the segment (positions relative
    to the start). That removes the heading accumulated before the segment; the fit window is
    long enough for 10 Hz GPS noise to stay below ~1 deg.
    """
    cum = path_length(gt_p)
    results = {}
    for length in lengths:
        errors = []
        start = 0
        while cum[start] + length <= cum[-1]:
            end = int(np.searchsorted(cum, cum[start] + length))
            head_end = int(np.searchsorted(cum, cum[start] + min(heading_window, length)))
            if head_end <= start + 2 or end >= len(cum):
                break
            yaw = relative_yaw(est_p[start:head_end + 1, :2] - est_p[start, :2], gt_p[start:head_end + 1, :2] - gt_p[start, :2])
            est_disp = yaw_rotation(yaw) @ (est_p[end] - est_p[start])
            gt_disp = gt_p[end] - gt_p[start]
            errors.append(np.linalg.norm((est_disp - gt_disp)[:2]) / length * 100.0)
            start = int(np.searchsorted(cum, cum[start] + stride))
        results[length] = np.array(errors)
    return results


# --------------------------------------------------------------------------- main
def evaluate(name, est_path, gt_path, log_path, args):
    est = load_tum(est_path)
    gt = load_tum(gt_path)
    est_a, gt_a = associate(est, gt)
    if len(est_a) < 10:
        raise RuntimeError(f'{name}: only {len(est_a)} time-associated poses')
    est_p, gt_p = est_a[:, 1:4], gt_a[:, 1:4]
    # Both trajectories start at the LIO origin; remove any residual offset of the first sample only.
    est_p = est_p - est_p[0] + gt_p[0]
    est_aligned, yaw_deg, straight_end = align_yaw_on_straight(est_p, gt_p, args.straight_length, args.straight_max_turn)

    error = np.linalg.norm(est_aligned - gt_p, axis=1)
    xy_error = np.linalg.norm((est_aligned - gt_p)[:, :2], axis=1)
    length = path_length(gt_p)[-1]
    segments = segment_errors(est_aligned, gt_p, args.segments)

    timing = load_timing(log_path) if log_path and os.path.exists(log_path) else np.zeros((0, 3))
    icp_ms = timing[:, 1] if len(timing) else np.array([np.nan])
    accepted = float(np.mean(timing[:, 2])) * 100.0 if len(timing) else np.nan

    return {
        'name': name, 'frames': len(est), 'associated': len(est_a), 'gt_length_m': length,
        'duration_s': est_a[-1, 0] - est_a[0, 0], 'yaw_deg': yaw_deg, 'straight_m': path_length(gt_p[:straight_end])[-1],
        'ate_rmse': float(np.sqrt(np.mean(error ** 2))), 'ate_median': float(np.median(error)), 'ate_max': float(np.max(error)),
        'xy_rmse': float(np.sqrt(np.mean(xy_error ** 2))), 'xy_max': float(np.max(xy_error)), 'z_rmse': float(np.sqrt(np.mean((est_aligned - gt_p)[:, 2] ** 2))),
        'final_err': float(error[-1]), 'drift_pct': float(error[-1] / length * 100.0),
        'final_xy_err': float(xy_error[-1]), 'xy_drift_pct': float(xy_error[-1] / length * 100.0),
        # (median, count, mean): the median is the headline -- a few GPS glitches (multipath) inflate
        # the mean identically for every method, the median stays with the odometry itself.
        'seg': {k: (float(np.median(v)) if len(v) else np.nan, len(v), float(np.mean(v)) if len(v) else np.nan) for k, v in segments.items()},
        'seg_median': float(np.nanmean([np.median(v) for v in segments.values() if len(v)])) if any(len(v) for v in segments.values()) else np.nan,
        'seg_mean': float(np.nanmean([np.mean(v) for v in segments.values() if len(v)])) if any(len(v) for v in segments.values()) else np.nan,
        'icp_ms_mean': float(np.nanmean(icp_ms)), 'icp_ms_median': float(np.nanmedian(icp_ms)),
        'icp_ms_p95': float(np.nanpercentile(icp_ms, 95)), 'icp_ms_max': float(np.nanmax(icp_ms)), 'icp_ms': icp_ms,
        'accepted_pct': accepted, 'timed_frames': len(timing),
        'est_xyz': est_aligned, 'gt_xyz': gt_p, 't': est_a[:, 0] - est_a[0, 0], 'error': error,
    }


def write_report(results, args):
    lines = []
    lines.append('| method | frames | path [m] | yaw fix [deg] | ATE 3D rmse / max [m] | ATE xy rmse / max [m] | z rmse [m] | '
                 'final err 3D / xy [m] | drift 3D / xy [%] | ' + ' / '.join(f'seg{l}' for l in args.segments)
                 + ' median [%] | seg median / mean [%] | ICP ms mean / med / p95 / max | accepted [%] |')
    lines.append('|' + '---|' * 13)
    for r in results:
        seg = ' / '.join(f'{r["seg"][l][0]:.2f}' if not np.isnan(r['seg'][l][0]) else '-' for l in args.segments)
        lines.append(f'| {r["name"]} | {r["frames"]} | {r["gt_length_m"]:.0f} | {r["yaw_deg"]:+.2f} | '
                     f'{r["ate_rmse"]:.2f} / {r["ate_max"]:.2f} | {r["xy_rmse"]:.2f} / {r["xy_max"]:.2f} | {r["z_rmse"]:.2f} | '
                     f'{r["final_err"]:.2f} / {r["final_xy_err"]:.2f} | {r["drift_pct"]:.2f} / {r["xy_drift_pct"]:.2f} | '
                     f'{seg} | {r["seg_median"]:.2f} / {r["seg_mean"]:.2f} | {r["icp_ms_mean"]:.1f} / {r["icp_ms_median"]:.1f} / {r["icp_ms_p95"]:.1f} / {r["icp_ms_max"]:.0f} | '
                     f'{r["accepted_pct"]:.1f} |')
    text = '\n'.join(lines)
    print(text)
    with open(os.path.join(args.out, 'summary.md'), 'w') as handle:
        handle.write(text + '\n')
        handle.write(f'\nyaw alignment on the first straight GT segment (<= {args.straight_length} m, turn <= {args.straight_max_turn} deg); '
                     f'straight lengths used: ' + ', '.join(f'{r["name"]} {r["straight_m"]:.0f} m' for r in results) + '\n')


def write_plots(results, args):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib not available, skipping plots', file=sys.stderr)
        return
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    fig, ax = plt.subplots(figsize=(9, 9))
    gt = results[0]['gt_xyz']
    ax.plot(gt[:, 0], gt[:, 1], 'k-', linewidth=2.0, label='GPS GT')
    for i, r in enumerate(results):
        ax.plot(r['est_xyz'][:, 0], r['est_xyz'][:, 1], '-', color=colors[i % len(colors)], linewidth=1.2,
                label=f'{r["name"]} (xy drift {r["xy_drift_pct"]:.2f} %, xy ATE {r["xy_rmse"]:.2f} m)')
    ax.plot(gt[0, 0], gt[0, 1], 'ko', markersize=6)
    ax.set_aspect('equal'); ax.grid(True, alpha=0.3); ax.legend(loc='best', fontsize=9)
    ax.set_xlabel('east [m]'); ax.set_ylabel('north [m]'); ax.set_title('Top view after yaw alignment on the first straight segment')
    fig.tight_layout(); fig.savefig(os.path.join(args.out, 'top_view.png'), dpi=130); plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for i, r in enumerate(results):
        axes[0].plot(r['t'], np.linalg.norm((r['est_xyz'] - r['gt_xyz'])[:, :2], axis=1), color=colors[i % len(colors)], linewidth=1.0, label=r['name'])
        axes[1].plot(r['t'], r['est_xyz'][:, 2] - r['gt_xyz'][:, 2], color=colors[i % len(colors)], linewidth=1.0, label=r['name'])
    axes[0].set_ylabel('xy position error [m]'); axes[0].grid(True, alpha=0.3); axes[0].legend(fontsize=9)
    axes[1].set_ylabel('z error [m]'); axes[1].set_xlabel('time [s]'); axes[1].grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(args.out, 'error_vs_time.png'), dpi=130); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].boxplot([r['icp_ms'][~np.isnan(r['icp_ms'])] for r in results], labels=[r['name'] for r in results], showfliers=False)
    axes[0].set_ylabel('ICP time per scan [ms]'); axes[0].grid(True, alpha=0.3); axes[0].set_title('ICP time (fliers hidden)')
    width = 0.8 / len(results)
    for i, r in enumerate(results):
        values = [r['seg'][l][0] for l in args.segments]
        axes[1].bar(np.arange(len(args.segments)) + i * width, values, width, label=r['name'], color=colors[i % len(colors)])
    axes[1].set_xticks(np.arange(len(args.segments)) + width * (len(results) - 1) / 2)
    axes[1].set_xticklabels([str(l) for l in args.segments]); axes[1].set_xlabel('segment length [m]')
    axes[1].set_ylabel('relative translation error, median [%]'); axes[1].grid(True, alpha=0.3, axis='y'); axes[1].legend(fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(args.out, 'timing_and_segments.png'), dpi=130); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run', action='append', required=True, help='name:est_tum:gt_tum[:node_log]')
    parser.add_argument('--out', required=True, help='output directory for summary.md and the plots')
    parser.add_argument('--straight-length', type=float, default=40.0, help='max GT path length of the alignment segment [m]')
    parser.add_argument('--straight-max-turn', type=float, default=5.0, help='heading change that ends the straight segment [deg]')
    parser.add_argument('--segments', type=int, nargs='+', default=[100, 200, 300, 400, 500, 600, 700, 800])
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    results = []
    for spec in args.run:
        parts = spec.split(':')
        if len(parts) < 3:
            parser.error(f'--run needs name:est:gt[:log], got {spec}')
        name, est, gt = parts[:3]
        log = parts[3] if len(parts) > 3 else ''
        results.append(evaluate(name, est, gt, log, args))
    write_report(results, args)
    write_plots(results, args)


if __name__ == '__main__':
    main()
