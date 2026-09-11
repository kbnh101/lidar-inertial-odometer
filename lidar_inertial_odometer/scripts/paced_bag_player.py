#!/usr/bin/env python3
"""rosbag2 를 LIO 처리 속도에 맞춰 재생한다 (평가용, 드롭 없는 결정적 재생).

`ros2 bag play` 는 벽시계 기준이라 매처가 느리면 best-effort 구독 큐에서 스캔이 빠지고,
빠르면 시간을 낭비한다. 이 플레이어는 bag 의 메시지를 직렬화된 그대로 다시 publish 하되,
PointCloud2 를 내보내기 전에 LIO 의 `~/odometry` 출력 개수를 보고 처리 중인 스캔 수를
`--max-in-flight` 이하로 유지한다. 초기화 구간(odometry 가 아직 없을 때)은 느린 고정
속도로 넘기고, 처리되지 않는 스캔은 timeout 으로 감지해 건너뛴다.

    python3 paced_bag_player.py --bag ~/data/kitti/lidar --duration 60
"""

import argparse
import sys
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
import rosbag2_py
from rosidl_runtime_py.utilities import get_message
from nav_msgs.msg import Odometry


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--bag', required=True, help='rosbag2 directory (holding metadata.yaml)')
    parser.add_argument('--storage', default='sqlite3')
    parser.add_argument('--lidar-topic', default='/points_raw')
    parser.add_argument('--imu-topic', default='/imu_raw')
    parser.add_argument('--gps-topic', default='/gps/fix')
    parser.add_argument('--odometry-topic', default='/lidar_inertial_odometer/odometry')
    parser.add_argument('--start', type=float, default=0.0, help='skip this many seconds from the bag start')
    parser.add_argument('--duration', type=float, default=0.0, help='stop after this many seconds of bag time (0 = all)')
    parser.add_argument('--max-in-flight', type=int, default=2, help='scans published but not yet answered by odometry')
    parser.add_argument('--timeout', type=float, default=8.0, help='seconds to wait for odometry before assuming a scan was skipped')
    parser.add_argument('--warmup-rate', type=float, default=3.0, help='scans per second before the first odometry arrives')
    parser.add_argument('--settle', type=float, default=5.0, help='seconds of odometry silence that ends the run')
    return parser.parse_args()


class PacedPlayer(Node):
    def __init__(self, args):
        super().__init__('paced_bag_player')
        self.args = args
        self.odometry_count = 0
        self.calibrated = False
        self.last_odometry_time = time.monotonic()
        # Reliable publishers are compatible with the nodes' best-effort SensorDataQoS subscriptions.
        sensor_qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=50,
                                durability=DurabilityPolicy.VOLATILE)
        self.publishers_by_topic = {}
        self.types_by_topic = {}
        self.create_subscription(Odometry, args.odometry_topic, self.on_odometry, 100)
        self.sensor_qos = sensor_qos

    def on_odometry(self, _msg):
        self.odometry_count += 1
        self.last_odometry_time = time.monotonic()

    def open_reader(self):
        reader = rosbag2_py.SequentialReader()
        reader.open(rosbag2_py.StorageOptions(uri=self.args.bag, storage_id=self.args.storage),
                    rosbag2_py.ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr'))
        wanted = {self.args.lidar_topic, self.args.imu_topic, self.args.gps_topic}
        for topic in reader.get_all_topics_and_types():
            if topic.name in wanted:
                self.types_by_topic[topic.name] = topic.type
                self.publishers_by_topic[topic.name] = self.create_publisher(get_message(topic.type), topic.name, self.sensor_qos)
        missing = wanted - set(self.publishers_by_topic)
        if missing:
            raise RuntimeError(f'topics not in bag: {sorted(missing)}')
        reader.set_filter(rosbag2_py.StorageFilter(topics=sorted(wanted)))
        return reader

    def spin_for(self, seconds):
        deadline = time.monotonic() + seconds
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=min(0.05, max(0.0, deadline - time.monotonic())))

    def wait_for_capacity(self, published_scans, skipped):
        """Blocks until the LIO has fewer than max_in_flight unanswered scans."""
        args = self.args
        if self.odometry_count == 0:
            # Initialization phase: nothing answers yet, so just go slowly.
            self.spin_for(1.0 / args.warmup_rate)
            return skipped
        if not self.calibrated:
            # First odometry: everything published before it that never answered was consumed by
            # initialization. Count those as skipped so that in_flight starts from the one-scan
            # latency of the odometer (a scan is processed once the IMU passes its sweep end).
            self.calibrated = True
            skipped = max(skipped, published_scans - self.odometry_count - 1)
            self.get_logger().info(f'first odometry after {published_scans} scans; {skipped} consumed by initialization')
        deadline = time.monotonic() + args.timeout
        while rclpy.ok():
            in_flight = published_scans - self.odometry_count - skipped
            if in_flight < args.max_in_flight:
                return skipped
            if time.monotonic() > deadline:
                # One scan produced no odometry (gated away or dropped); stop waiting for it.
                skipped += 1
                self.get_logger().warn(f'no odometry for {args.timeout:.0f}s at scan {published_scans}; assuming 1 skipped (total {skipped})')
                return skipped
            rclpy.spin_once(self, timeout_sec=0.02)
        return skipped

    def run(self):
        args = self.args
        reader = self.open_reader()
        # Give discovery a moment so the first messages are not lost before the subscriptions match.
        self.spin_for(2.0)

        first_stamp = None
        published = {topic: 0 for topic in self.publishers_by_topic}
        scans = 0
        skipped = 0
        wall_start = time.monotonic()
        while rclpy.ok() and reader.has_next():
            topic, data, stamp_ns = reader.read_next()
            if first_stamp is None:
                first_stamp = stamp_ns
            elapsed = (stamp_ns - first_stamp) * 1e-9
            if elapsed < args.start:
                continue
            if args.duration > 0.0 and elapsed > args.start + args.duration:
                break
            if topic == args.lidar_topic:
                skipped = self.wait_for_capacity(scans, skipped)
                scans += 1
                if scans % 100 == 0:
                    wall = time.monotonic() - wall_start
                    self.get_logger().info(f'scan {scans} | bag t={elapsed:7.1f}s | odometry {self.odometry_count} | '
                                           f'skipped {skipped} | wall {wall:6.0f}s ({scans / wall:.1f} scans/s)')
            self.publishers_by_topic[topic].publish(data)  # raw serialized bytes, no deserialization
            published[topic] += 1
            rclpy.spin_once(self, timeout_sec=0.0)

        # Let the pipeline drain: the last scan needs IMU past its sweep, which the bag no longer has.
        self.get_logger().info(f'bag done: published {published}; waiting for the odometer to settle')
        while rclpy.ok() and time.monotonic() - self.last_odometry_time < args.settle:
            rclpy.spin_once(self, timeout_sec=0.1)
        wall = time.monotonic() - wall_start
        self.get_logger().info(f'finished: {scans} scans published, {self.odometry_count} odometry received, '
                               f'{skipped} assumed skipped, wall {wall:.0f}s')
        return 0


def main():
    args = parse_args()
    rclpy.init()
    player = PacedPlayer(args)
    try:
        return player.run()
    except KeyboardInterrupt:
        return 130
    finally:
        player.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    sys.exit(main())
