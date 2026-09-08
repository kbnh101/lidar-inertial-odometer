#!/usr/bin/env python3
"""Integration test against the installed packages; no bag or ROS1 required.
Run after sourcing the ROS2 workspace. Uses an isolated ROS_DOMAIN_ID.
"""
import math
import os
import signal
import struct
import subprocess
import tempfile
import time

os.environ['ROS_DOMAIN_ID'] = '173'
os.environ['ROS_LOCALHOST_ONLY'] = '1'
os.environ['RMW_IMPLEMENTATION'] = 'rmw_fastrtps_cpp'
os.environ.pop('FASTRTPS_DEFAULT_PROFILES_FILE', None)
import rclpy
from rclpy.qos import QoSProfile, DurabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import Imu, NavSatFix, PointCloud2, PointField
from nav_msgs.msg import Odometry, Path
from std_msgs.msg import Header
from tf2_msgs.msg import TFMessage


def main():
    processes = []
    logs = []
    rclpy.init()
    node = rclpy.create_node('lio_interface_test')
    with tempfile.TemporaryDirectory(prefix='lio-ros2-test-') as tmp:
        def start(package, executable, args):
            log = open(os.path.join(tmp, executable + str(len(logs)) + '.log'), 'w+')
            logs.append(log)
            processes.append(subprocess.Popen(
                ['ros2', 'run', package, executable, '--ros-args', *args],
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True))

        def spin_until(predicate, seconds=15):
            deadline = time.monotonic() + seconds
            while not predicate() and time.monotonic() < deadline:
                assert all(p.poll() is None for p in processes), 'node exited unexpectedly'
                rclpy.spin_once(node, timeout_sec=0.02)
            assert predicate(), 'timed out waiting for ROS2 messages/discovery'

        def stamp(t):
            return rclpy.time.Time(nanoseconds=round(t * 1e9)).to_msg()

        try:
            imu_pub = node.create_publisher(Imu, '/imu_raw', qos_profile_sensor_data)
            cloud_pub = node.create_publisher(PointCloud2, '/points_raw', qos_profile_sensor_data)
            gps_pub = node.create_publisher(NavSatFix, '/gps/fix', qos_profile_sensor_data)
            odometry, gps_paths, transforms = [], [], []
            transient = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
            subscriptions = [
                node.create_subscription(Odometry, '/lidar_inertial_odometer/odometry', odometry.append, 100),
                node.create_subscription(Path, '/gps_ground_truth/path', gps_paths.append, transient),
                node.create_subscription(TFMessage, '/tf', transforms.append, 100)]
            start('lidar_inertial_odometer', 'lio_node', [
                '-p', 'init_imu_samples:=1', '-p', 'verbose:=false',
                '-p', 'trajectory_csv:=' + os.path.join(tmp, 'est.txt')])
            start('gps_ground_truth', 'gps_ground_truth_node', [
                '-p', 'trajectory_csv:=' + os.path.join(tmp, 'gt.txt')])
            spin_until(lambda: imu_pub.get_subscription_count() > 0 and cloud_pub.get_subscription_count() > 0
                       and gps_pub.get_subscription_count() > 0)
            discovery_deadline = time.monotonic() + 0.5
            while time.monotonic() < discovery_deadline:
                rclpy.spin_once(node, timeout_sec=0.02)
            cloud = PointCloud2()
            cloud.header.frame_id = 'velodyne'
            cloud.height = 1
            cloud.fields = [PointField(name=name, offset=i * 4, datatype=PointField.FLOAT32, count=1)
                            for i, name in enumerate(('x', 'y', 'z', 'ring'))]
            points = [(5 + i * .2, -4 + j * .2, -1.0, 20.0) for i in range(25) for j in range(40)]
            cloud.width = len(points)
            cloud.point_step = 16
            cloud.row_step = cloud.width * cloud.point_step
            cloud.data = b''.join(struct.pack('<ffff', *p) for p in points)
            cloud.is_dense = True
            # GPS brackets the first LIO timestamp regardless of callback ordering.
            for i in range(31):
                msg = NavSatFix()
                msg.header.stamp = stamp(999.9 + i * .1)
                msg.status.status = 0
                msg.latitude, msg.longitude, msg.altitude = 49.0, 8.0, 99.9 + i * .1
                gps_pub.publish(msg)
                rclpy.spin_once(node, timeout_sec=.005)
            for i in range(221):
                t = 1000 + i * .01
                msg = Imu()
                msg.header.stamp = stamp(t)
                msg.header.frame_id = 'imu_link'
                msg.orientation.w = 1.0
                msg.linear_acceleration.z = 9.80665
                imu_pub.publish(msg)
                if i % 10 == 0 and i < 200:
                    cloud.header.stamp = stamp(t)
                    cloud_pub.publish(cloud)
                rclpy.spin_once(node, timeout_sec=.005)
            spin_until(lambda: len(odometry) >= 3 and gps_paths and transforms)
            starts = []
            subscriptions.append(node.create_subscription(
                Header, '/lidar_inertial_odometer/trajectory_start', starts.append, transient))
            spin_until(lambda: bool(starts))  # receives a latched start after initialization
            start_time = starts[0].stamp.sec + starts[0].stamp.nanosec * 1e-9
            first_time = odometry[0].header.stamp.sec + odometry[0].header.stamp.nanosec * 1e-9
            assert abs(start_time - first_time) < 1e-8
            assert starts[0].frame_id == 'odom'
            for pose in gps_paths[-1].poses:
                t = pose.header.stamp.sec + pose.header.stamp.nanosec * 1e-9
                assert t >= start_time - 1e-8
                assert abs(pose.pose.position.z - (t - start_time)) < 1e-6, (t, start_time, pose.pose.position.z)
                assert pose.header.frame_id == 'odom'
            assert all(math.isfinite(o.pose.pose.position.x) for o in odometry)
            assert any(t.child_frame_id == 'imu_link' for m in transforms for t in m.transforms)
            assert os.path.getsize(os.path.join(tmp, 'est.txt')) > 70
            assert os.path.getsize(os.path.join(tmp, 'gt.txt')) > 70
            print(f'PASS: {len(odometry)} odometry messages, GPS interpolation, TUM files, TF, late start subscriber')
        except BaseException:
            for log in logs:
                log.flush()
                log.seek(0)
                print(log.read())
            raise
        finally:
            for process in processes:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGINT)
            for process in processes:
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            for log in logs:
                log.close()
            node.destroy_node()
            rclpy.shutdown()


if __name__ == '__main__':
    main()
