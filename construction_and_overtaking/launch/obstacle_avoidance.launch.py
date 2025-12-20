from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():

    bird_view_node = Node(
        package='lane_detection',
        executable='bird_view',
        name='bird_view',
        output='screen'
    )

    detect_lanes_node = Node(
        package='lane_detection',
        executable='detect_lanes',
        name='detect_lanes',
        output='screen'
    )

    follow_lanes_node = Node(
        package='lane_detection',
        executable='follow_lanes',
        name='follow_lanes',
        output='screen'
    )

    obstacle_avoidance_node = Node(
        package='construction_and_overtaking',
        executable='occupancy_grid_navigator',
        name='obstacle_avoidance',
        output='screen',
        parameters=[{'use_sim_time': True}]
    )

    return LaunchDescription([
        bird_view_node,
        detect_lanes_node,
        follow_lanes_node,
        obstacle_avoidance_node
    ])
