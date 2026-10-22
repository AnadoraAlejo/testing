import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():

    folder = os.path.join(
        get_package_share_directory('cave_explorer'),
        'launch'
    )

    def include(filename):
        return IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(folder, filename)
            )
        )

    return LaunchDescription([
        # Start Gazebo and RViz
        include('cave_explorer_startup.launch.py'),

        # Start SLAM and navigation after 12 seconds
        TimerAction(
            period=12.0,
            actions=[
                include('cave_explorer_navigation.launch.py')
            ]
        ),

        # Start Assignment 3 after 30 seconds
        TimerAction(
            period=30.0,
            actions=[
                include('full_assignment3.launch.py')
            ]
        ),
    ])
