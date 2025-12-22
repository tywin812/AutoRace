from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='finish_detection',
            executable='finish_detector',
            name='finish_detector',
            output='screen',
            parameters=[{
                # Можно настроить параметры здесь
            }]
        ),
    ])
