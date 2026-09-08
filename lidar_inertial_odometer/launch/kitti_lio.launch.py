"""ROS 2 LIO + independent GPS GT. Pass a converted rosbag2 directory to bag."""
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction, TimerAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def playback(context):
    if LaunchConfiguration('play').perform(context).lower() != 'true':
        return []
    bag = LaunchConfiguration('bag').perform(context)
    if not bag:
        raise RuntimeError('play:=true requires bag:=/path/to/rosbag2_directory')
    return [TimerAction(period=2.0, actions=[ExecuteProcess(cmd=[
        'ros2', 'bag', 'play', bag, '--clock', '--rate', LaunchConfiguration('rate'),
        '--start-offset', LaunchConfiguration('start')], output='screen')])]


def generate_launch_description():
    share = get_package_share_directory('lidar_inertial_odometer')
    gps_share = get_package_share_directory('gps_ground_truth')
    args = {'bag': '', 'play': 'false', 'rate': '0.5', 'start': '0.0', 'rviz': 'true',
            'gps': 'true', 'use_sim_time': LaunchConfiguration('play'),
            'config': os.path.join(share, 'config', 'kitti.yaml'),
            'gps_config': os.path.join(gps_share, 'config', 'gps.yaml'),
            'trajectory_csv': '/tmp/lio_trajectory.txt', 'gt_csv': '/tmp/lio_gt.txt'}
    sim_time = ParameterValue(LaunchConfiguration('use_sim_time'), value_type=bool)
    return LaunchDescription([
        *[DeclareLaunchArgument(name, default_value=value) for name, value in args.items()],
        Node(package='lidar_inertial_odometer', executable='lio_node', name='lidar_inertial_odometer',
             output='screen', parameters=[LaunchConfiguration('config'), {
                 'use_sim_time': sim_time, 'trajectory_csv': LaunchConfiguration('trajectory_csv')}]),
        Node(package='gps_ground_truth', executable='gps_ground_truth_node', name='gps_ground_truth',
             condition=IfCondition(LaunchConfiguration('gps')), output='screen',
             parameters=[LaunchConfiguration('gps_config'), {
                 'use_sim_time': sim_time, 'trajectory_csv': LaunchConfiguration('gt_csv')}]),
        Node(package='rviz2', executable='rviz2', arguments=['-d', os.path.join(share, 'rviz', 'kitti_lio.rviz')],
             parameters=[{'use_sim_time': sim_time}], condition=IfCondition(LaunchConfiguration('rviz'))),
        OpaqueFunction(function=playback),
    ])
