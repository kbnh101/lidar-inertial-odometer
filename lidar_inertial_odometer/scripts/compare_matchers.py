#!/usr/bin/env python3
"""매칭 방식별 LIO 를 같은 bag 에 돌려 궤적을 뽑고, drift 와 ICP 계산 시간을 출력·플롯한다.

각 방식은 서로 다른 git 브랜치에 있으므로 브랜치마다 worktree 를 만들어 colcon 으로
따로 빌드하고, 그 빌드의 lio_node + gps_ground_truth_node 를 띄운 뒤
paced_bag_player.py 로 bag 을 (드롭 없이) 재생한다. 결과는 run 디렉토리마다
est.txt / gt.txt (TUM) 와 lio.log (프레임별 ICP 시간) 로 남고, 마지막에
evaluate_trajectories.py 가 표를 출력하고 그림을 저장한다.

기본 variants (VARIANTS 참조):
  p2plane-tight   point-to-plane-icp 브랜치, use_tightly_coupled=true
  p2plane-loose   point-to-plane-icp 브랜치, use_tightly_coupled=false (initial guess 만)
  p2point         point-to-point-icp 브랜치
  fused           fused-point-plane-icp 브랜치 (alpha=0.01, beta=1)
  fused-alpha0    같은 브랜치, icp_point_weight=0 (같은 front-end 의 순수 point-to-plane 대조군)

    python3 compare_matchers.py --bag ~/data/kitti/lidar --work ~/lio_eval
    python3 compare_matchers.py --bag ~/data/kitti/lidar --work ~/lio_eval --only fused,p2plane-tight --duration 60
    python3 compare_matchers.py --bag ~/data/kitti/lidar --work ~/lio_eval --common-frontend   # front-end 통일
    python3 compare_matchers.py --work ~/lio_eval --evaluate-only                              # 표/그림만 다시

산출물: <work>/runs/<variant>/{est.txt,gt.txt,lio.log,gps.log,player.log,wall.txt},
        <work>/report/{summary.md,top_view.png,error_vs_time.png,timing_and_segments.png}
"""

import argparse
import os
import shlex
import signal
import subprocess
import sys
import time

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
PACKAGE_DIR = os.path.dirname(SCRIPTS_DIR)
REPO_DIR = os.path.dirname(PACKAGE_DIR)
sys.path.insert(0, SCRIPTS_DIR)
import evaluate_trajectories as ev  # noqa: E402

ROS_SETUP = '/opt/ros/humble/setup.bash'
LIO_YAML = 'lidar_inertial_odometer/config/kitti.yaml'
GPS_YAML = os.path.join(REPO_DIR, 'gps_ground_truth', 'config', 'gps.yaml')

# name -> (branch, extra ros parameters). The config is that branch's own kitti.yaml.
VARIANTS = {
    'p2plane-tight': ('point-to-plane-icp', ['use_tightly_coupled:=true']),
    'p2plane-loose': ('point-to-plane-icp', ['use_tightly_coupled:=false']),
    'p2point': ('point-to-point-icp', []),
    'fused': ('fused-point-plane-icp', []),
    'fused-alpha0': ('fused-point-plane-icp', ['icp_point_weight:=0.0']),
}
# --common-frontend: the fused branch's front-end on every variant (ICP solver parameters stay per branch).
COMMON_FRONTEND = ['ring_min:=31', 'ring_max:=63', 'normal_method:=neighborhood_pca', 'feature_voxel_size:=0.5',
                   'map_voxel_size:=0.5', 'init_imu_samples:=100']
# Publishing clouds costs time and is not part of the matcher; the log line needs verbose.
RUN_PARAMS = ['verbose:=true', 'icp_verbose:=false', 'publish_feature_cloud:=false', 'publish_submap:=false',
              'publish_scan_cloud:=false']


def log(message):
    print(f'[compare] {message}', flush=True)


def run(command, **kwargs):
    """Runs a shell command with ROS sourced; raises on failure."""
    log(command if len(command) < 200 else command[:200] + ' ...')
    return subprocess.run(['bash', '-c', f'source {ROS_SETUP} && {command}'], check=True, **kwargs)


def current_branch():
    return subprocess.check_output(['git', '-C', REPO_DIR, 'rev-parse', '--abbrev-ref', 'HEAD'], text=True).strip()


def existing_worktree(branch):
    """Path of a worktree that already has `branch` checked out, or None."""
    porcelain = subprocess.check_output(['git', '-C', REPO_DIR, 'worktree', 'list', '--porcelain'], text=True)
    path = None
    for line in porcelain.splitlines():
        if line.startswith('worktree '):
            path = line.split(' ', 1)[1]
        elif line == f'branch refs/heads/{branch}' and path and os.path.isdir(path):
            return path
    return None


def checkout_dir(branch, work):
    """The current checkout for its own branch, an existing worktree of it, or a new detached
    worktree under <work>/src/<branch> (a branch can be checked out in only one worktree)."""
    if branch == current_branch():
        return REPO_DIR
    path = os.path.join(work, 'src', branch)
    if os.path.isdir(path):
        return path
    existing = existing_worktree(branch)
    if existing is not None:
        log(f'{branch}: using existing worktree {existing}')
        return existing
    log(f'adding detached worktree of {branch} at {path}')
    subprocess.run(['git', '-C', REPO_DIR, 'worktree', 'add', '--detach', path, branch], check=True)
    return path


def build(branch, source_dir, work, force):
    ws = os.path.join(work, 'ws', branch)
    lio_bin = os.path.join(ws, 'install', 'lidar_inertial_odometer', 'lib', 'lidar_inertial_odometer', 'lio_node')
    if os.path.exists(lio_bin) and not force:
        log(f'{branch}: already built ({lio_bin})')
        return ws
    run(f'colcon build --base-paths {shlex.quote(source_dir)} --packages-select gps_ground_truth lidar_inertial_odometer '
        f'--build-base {shlex.quote(ws + "/build")} --install-base {shlex.quote(ws + "/install")} '
        f'--cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF', cwd=work)
    return ws


def spawn_node(ws, executable, args, log_path, domain):
    """Starts a ROS node from the given install space; the PID is the node itself (exec)."""
    env = f'source {ROS_SETUP} && source {shlex.quote(ws)}/install/setup.bash && export ROS_DOMAIN_ID={domain} && exec '
    command = env + ' '.join(shlex.quote(a) for a in [executable, '--ros-args', *args])
    handle = open(log_path, 'w')
    return subprocess.Popen(['bash', '-c', command], stdout=handle, stderr=subprocess.STDOUT), handle


def stop_node(process, handle, timeout=10.0):
    """SIGTERM (rclcpp treats it like SIGINT), then SIGKILL after the timeout."""
    if process.poll() is None:
        process.send_signal(signal.SIGTERM)
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
    handle.close()


def run_variant(name, branch, params, args):
    out = os.path.join(args.work, 'runs', name)
    est, gt = os.path.join(out, 'est.txt'), os.path.join(out, 'gt.txt')
    if os.path.exists(est) and os.path.exists(gt) and not args.force_run:
        log(f'{name}: reusing {out} (pass --force-run to rerun)')
        return out
    os.makedirs(out, exist_ok=True)

    source_dir = checkout_dir(branch, args.work)
    ws = build(branch, source_dir, args.work, args.force_build)
    config = os.path.join(source_dir, LIO_YAML)
    ros_params = [f'-p {p}' for p in [f'trajectory_csv:={est}', *RUN_PARAMS, *params]]
    ros_params = [token for pair in ros_params for token in pair.split(' ', 1)]

    lio_bin = os.path.join(ws, 'install', 'lidar_inertial_odometer', 'lib', 'lidar_inertial_odometer', 'lio_node')
    gps_bin = os.path.join(ws, 'install', 'gps_ground_truth', 'lib', 'gps_ground_truth', 'gps_ground_truth_node')
    log(f'{name}: branch {branch}, config {config}, params {params}')
    lio, lio_log = spawn_node(ws, lio_bin, ['--params-file', config, *ros_params], os.path.join(out, 'lio.log'), args.domain)
    gps, gps_log = spawn_node(ws, gps_bin, ['--params-file', GPS_YAML, '-p', f'trajectory_csv:={gt}'], os.path.join(out, 'gps.log'), args.domain)
    time.sleep(3.0)  # discovery
    try:
        begin = time.monotonic()
        player = os.path.join(SCRIPTS_DIR, 'paced_bag_player.py')
        with open(os.path.join(out, 'player.log'), 'w') as player_log:
            subprocess.run(['bash', '-c', f'source {ROS_SETUP} && export ROS_DOMAIN_ID={args.domain} && exec python3 '
                            + ' '.join(shlex.quote(a) for a in [player, '--bag', args.bag, '--duration', str(args.duration),
                                                                 '--start', str(args.start)])],
                           stdout=player_log, stderr=subprocess.STDOUT, check=True)
        wall = time.monotonic() - begin
    finally:
        stop_node(lio, lio_log)
        stop_node(gps, gps_log)
    with open(os.path.join(out, 'wall.txt'), 'w') as handle:
        handle.write(f'wall_s={wall:.1f}\n')
    frames = sum(1 for line in open(est) if not line.startswith('#'))
    log(f'{name}: done in {wall:.0f} s, {frames} poses -> {out}')
    return out


def print_summary(result, wall_s):
    r = result
    print(f'\n=== {r["name"]} ===')
    print(f'  frames             : {r["frames"]} ({r["associated"]} matched to GT), GT path {r["gt_length_m"]:.0f} m, {r["duration_s"]:.0f} s')
    print(f'  yaw alignment      : {r["yaw_deg"]:+.2f} deg on the first {r["straight_m"]:.0f} m straight segment')
    print(f'  ATE xy rmse / max  : {r["xy_rmse"]:.2f} / {r["xy_max"]:.2f} m   (3D rmse {r["ate_rmse"]:.2f} m, z rmse {r["z_rmse"]:.2f} m)')
    print(f'  final error xy/3D  : {r["final_xy_err"]:.2f} / {r["final_err"]:.2f} m  -> drift {r["xy_drift_pct"]:.2f} % / {r["drift_pct"]:.2f} %')
    seg = ', '.join(f'{k} m: {v[0]:.2f} %' for k, v in r['seg'].items() if v[1] > 0)
    print(f'  segment rel. error : {seg}  (median; mean over lengths {r["seg_median"]:.2f} % median / {r["seg_mean"]:.2f} % mean)')
    print(f'  ICP time [ms]      : mean {r["icp_ms_mean"]:.1f}, median {r["icp_ms_median"]:.1f}, p95 {r["icp_ms_p95"]:.1f}, max {r["icp_ms_max"]:.0f}'
          f'  ({r["timed_frames"]} frames, {r["accepted_pct"]:.1f} % accepted)')
    if wall_s is not None:
        print(f'  pipeline wall time : {wall_s:.0f} s total, {wall_s / max(r["frames"], 1) * 1000:.1f} ms / scan (features + ICP + ROS overhead)')


def evaluate(names, args):
    report = os.path.join(args.work, 'report')
    os.makedirs(report, exist_ok=True)
    eval_args = argparse.Namespace(out=report, straight_length=args.straight_length, straight_max_turn=args.straight_max_turn,
                                   segments=args.segments)
    results = []
    for name in names:
        out = os.path.join(args.work, 'runs', name)
        est, gt, lio_log = (os.path.join(out, f) for f in ('est.txt', 'gt.txt', 'lio.log'))
        if not (os.path.exists(est) and os.path.exists(gt)):
            log(f'{name}: no trajectory in {out}, skipping')
            continue
        result = ev.evaluate(name, est, gt, lio_log, eval_args)
        wall = None
        wall_file = os.path.join(out, 'wall.txt')
        if os.path.exists(wall_file):
            wall = float(open(wall_file).read().split('=')[1])
        print_summary(result, wall)
        results.append(result)
    if not results:
        raise SystemExit('nothing to evaluate')
    print()
    ev.write_report(results, eval_args)
    ev.write_plots(results, eval_args)
    log(f'report written to {report}')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--bag', default='', help='rosbag2 directory (required unless --evaluate-only)')
    parser.add_argument('--work', required=True, help='working directory for worktrees, builds, runs and the report')
    parser.add_argument('--only', default='', help='comma separated subset of ' + ', '.join(VARIANTS))
    parser.add_argument('--common-frontend', action='store_true', help='apply the fused branch front-end parameters to every variant')
    parser.add_argument('--start', type=float, default=0.0, help='skip this many seconds of the bag')
    parser.add_argument('--duration', type=float, default=0.0, help='bag seconds to play (0 = all)')
    parser.add_argument('--domain', type=int, default=77, help='ROS_DOMAIN_ID for the evaluation nodes')
    parser.add_argument('--force-build', action='store_true')
    parser.add_argument('--force-run', action='store_true', help='rerun variants whose trajectories already exist')
    parser.add_argument('--evaluate-only', action='store_true', help='skip running; print/plot from existing runs')
    parser.add_argument('--straight-length', type=float, default=40.0)
    parser.add_argument('--straight-max-turn', type=float, default=5.0)
    parser.add_argument('--segments', type=int, nargs='+', default=[100, 200, 300, 400, 500, 600, 700, 800])
    args = parser.parse_args()
    args.work = os.path.abspath(os.path.expanduser(args.work))
    args.bag = os.path.abspath(os.path.expanduser(args.bag)) if args.bag else ''

    names = [n.strip() for n in args.only.split(',') if n.strip()] or list(VARIANTS)
    unknown = [n for n in names if n not in VARIANTS]
    if unknown:
        parser.error(f'unknown variant(s) {unknown}; choose from {list(VARIANTS)}')
    if args.common_frontend:
        names = [f'{n}-cfe' for n in names]

    if not args.evaluate_only:
        if not args.bag:
            parser.error('--bag is required unless --evaluate-only')
        for name in names:
            base = name[:-4] if args.common_frontend else name
            branch, params = VARIANTS[base]
            if args.common_frontend:
                params = params + COMMON_FRONTEND
            run_variant(name, branch, params, args)
    evaluate(names, args)


if __name__ == '__main__':
    main()
