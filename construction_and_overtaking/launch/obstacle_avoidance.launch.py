from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():

    # Note: bird_view, detect_lanes, and follow_lanes are already launched by robot_bringup/autorace_2025.launch.py
    # We only launch the navigator here to avoid conflicts.

    obstacle_avoidance_node = Node(
        package='construction_and_overtaking',
        executable='occupancy_grid_navigator',
        name='obstacle_avoidance',
        output='screen',
        parameters=[{'use_sim_time': True}]
    )

    return LaunchDescription([
        obstacle_avoidance_node
    ])
