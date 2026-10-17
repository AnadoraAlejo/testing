# === ASSIGNMENT 3 PROVENANCE (LOCAL ANNOTATION) ===
# Source branch / provenance: main
# GitHub commit account / updater: AnadoraAlejo (repository upload/commit, not necessarily original
# code author)
# Pinned commit SHA: c4b451e2f4b8740595f9b820ee6fef4a7b6fb6ee
# Integration notes: Starter SLAM and Nav2 launcher copied from local planning1 source.
# Attribution caveat: GitHub account != original author of every line; disclose AI assistance.
# === END ASSIGNMENT 3 PROVENANCE ===

import launch_ros.actions
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import (Command, LaunchConfiguration,
                                  PathJoinSubstitution)
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():

    ld = LaunchDescription()

    config_path = [FindPackageShare('cave_explorer'), 'config']
    
    # Additional command line arguments
    use_sim_time = LaunchConfiguration('use_sim_time')
    use_sim_time_launch_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='True',
        description='Flag to enable use_sim_time'
    )

    # Start Simultaneous Localisation and Mapping (SLaM)
    slam = IncludeLaunchDescription(
        PathJoinSubstitution([FindPackageShare('slam_toolbox'), 'launch', 'online_async_launch.py']),
        launch_arguments={
            'use_sim_time': use_sim_time,
            'slam_params_file': PathJoinSubstitution(config_path+['slam_params.yaml'])
        }.items()
    )
    
    # Start Navigation Stack
    navigation = IncludeLaunchDescription(
        PathJoinSubstitution([FindPackageShare('nav2_bringup'), 'launch', 'navigation_launch.py']),
        launch_arguments={
            'use_sim_time': use_sim_time,
            'params_file': PathJoinSubstitution(config_path+['nav2_params.yaml'])
        }.items()
    )

    ld.add_action(use_sim_time_launch_arg)
    ld.add_action(slam)
    ld.add_action(navigation)

    return ld