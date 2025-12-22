from launch import LaunchDescription
from launch_ros.actions import Node


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

    detect_signes_node = Node(
        package='sign_detection',
        executable='detect_sign',
        name='detect_sign'
    )

    detect_aruco_node = Node(
        package='aruco_detection',
        executable='detect_aruco',
        name='detect_aruco'
    )

    traffic_light_node = Node(
        package='traffic_light_recognition',
        executable='recognize_state',
        name='recognize_state'
    )

    race_controller_node = Node(
        package='autorace_core_erevan',
        executable='race_controller',
        name='race_controller',
        output='screen'
    )
    
    return LaunchDescription([
        bird_view_node,
        detect_lanes_node,
        follow_lanes_node,
        detect_signes_node,
        detect_aruco_node,
        traffic_light_node,
        race_controller_node,
    ])
