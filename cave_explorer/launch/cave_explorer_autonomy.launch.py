# === ASSIGNMENT 3 PROVENANCE (LOCAL ANNOTATION) ===
# Source branch / provenance: main
# GitHub commit account / updater: AnadoraAlejo (repository upload/commit, not necessarily original
# code author)
# Pinned commit SHA: c4b451e2f4b8740595f9b820ee6fef4a7b6fb6ee
# Integration notes: Legacy autonomous launcher inherited from starter; do NOT run with integrated
# mission.
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
    use_sim_time_launch_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='True',
        description='Flag to enable use_sim_time'
    )
    print_feedback_launch_arg = DeclareLaunchArgument(
        'print_feedback',
        default_value='False',
        description='Flag to enable print feedback from action server'
    )

    # Start Navigation Stack
    cave_explorer_node = Node(
        package='cave_explorer',
        executable='cave_explorer',
        output='screen',
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time'),
                     'print_feedback': LaunchConfiguration('print_feedback'),
                    'computer_vision_model_filename': PathJoinSubstitution(config_path+['stop_data.xml'])}]
    )

    ld.add_action(use_sim_time_launch_arg)
    ld.add_action(print_feedback_launch_arg)
    ld.add_action(cave_explorer_node)

    return ld