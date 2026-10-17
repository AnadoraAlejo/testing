"""Start integrated autonomy, dataset collector, and geometry node.

Start cave_explorer_startup.launch.py and cave_explorer_navigation.launch.py
separately first; do NOT launch cave_explorer_autonomy.launch.py as well.
"""
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    parameters = os.path.join(get_package_share_directory('cave_explorer'),
                              'config', 'mission_integrated.yaml')
    dataset = LaunchConfiguration('collect_dataset')
    return LaunchDescription([
        DeclareLaunchArgument('collect_dataset', default_value='true'),
        Node(package='cave_explorer', executable='cave_mission', name='cave_mission',
             output='screen', parameters=[parameters]),
        Node(package='cave_explorer', executable='cave_geometry', name='cave_geometry',
             output='screen', parameters=[parameters]),
        Node(package='cave_explorer', executable='cave_dataset_collector', name='dataset_collector',
             output='screen', parameters=[parameters], condition=IfCondition(dataset)),
    ])
