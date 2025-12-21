from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        # Bird-eye view calibration
        Node(
            package='lane_detection',
            executable='bird_view_calibration',
            name='bird_view_calibration',
        ),
        
        # Lane detection
        Node(
            package='lane_detection',
            executable='detect_lanes',
            name='detect_lanes',
        ),
        
        # Sign detection (YOLO)
        Node(
            package='sign_detection',
            executable='detect_sign',
            name='detect_sign'
        ),
        
        # Race State Machine (ГЛАВНЫЙ КОНТРОЛЛЕР)
        Node(
            package='race_controller',
            executable='race_state_machine',
            name='race_state_machine',
            parameters=[{
                'base_speed': 0.6,
                'turn_speed': 0.3,
                'sign_confidence_threshold': 0.80,
                'sign_cooldown_time': 3.0
            }],
            output='screen'
        ),
    ])
