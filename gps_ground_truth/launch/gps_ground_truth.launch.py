from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
import os


def generate_launch_description():
    config = os.path.join(get_package_share_directory('gps_ground_truth'), 'config', 'gps.yaml')
    return LaunchDescription([
        DeclareLaunchArgument('config', default_value=config),
        DeclareLaunchArgument('origin_mode', default_value='lio_start'),
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        Node(package='gps_ground_truth', executable='gps_ground_truth_node', name='gps_ground_truth',
             output='screen', parameters=[LaunchConfiguration('config'), {
                 'origin_mode': LaunchConfiguration('origin_mode'),
                 'use_sim_time': ParameterValue(LaunchConfiguration('use_sim_time'), value_type=bool)}]),
    ])
