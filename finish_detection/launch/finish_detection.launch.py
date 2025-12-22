import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

def generate_launch_description():
    finish_detector_node = Node(
        package='finish_detection',
        executable='finish_detector',
        name='finish_detector',
        output='screen',
        parameters=[{
            # Можно настроить параметры здесь
        }]
    )

    return LaunchDescription([
        finish_detector_node,
    ])
