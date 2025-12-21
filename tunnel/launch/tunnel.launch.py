import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    config = os.path.join(
        get_package_share_directory('tunnel'),
        'config',
        'tunnel.yaml'
    )

    return LaunchDescription([
        # Vision nodes
        Node(
            package='lane_detection',
            executable='bird_view',
            name='bird_view_calibration'
        ),
        Node(
            package='lane_detection',
            executable='detect_lanes',
            name='detect_lanes'
        ),
        # Tunnel Navigator
        Node(
            package='tunnel',
            executable='tunnel_navigator',
            name='tunnel_navigator',
            output='screen',
            parameters=[config] if os.path.exists(config) else []
        )
    ])
