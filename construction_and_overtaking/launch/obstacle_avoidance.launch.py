#!/usr/bin/env python3
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        # Bird view calibration node
        Node(
            package='lane_detection',
            executable='bird_view',
            name='bird_view_calibration',
            output='screen'
        ),
        
        # Lane detection node
        Node(
            package='lane_detection',
            executable='detect_lanes',
            name='detect_lanes',
            output='screen'
        ),
        
        # Lane following control node
        Node(
            package='lane_detection',
            executable='follow_lanes',
            name='follow_lanes',
            output='screen'
        ),
        
        # Obstacle avoidance node
        Node(
            package='construction_and_overtaking',
            executable='obstacle_avoidance',
            name='obstacle_avoidance',
            output='screen'
        ),
    ])
