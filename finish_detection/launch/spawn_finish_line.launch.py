#!/usr/bin/env python3
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    # Параметры по умолчанию из Gazebo
    x_pos = LaunchConfiguration('x', default='0.1317')
    y_pos = LaunchConfiguration('y', default='-2.2171')
    z_pos = LaunchConfiguration('z', default='0.0338')
    
    roll = LaunchConfiguration('roll', default='0.0000')
    pitch = LaunchConfiguration('pitch', default='0.0084')
    yaw = LaunchConfiguration('yaw', default='-0.0918')
    
    model_name = LaunchConfiguration('model_name', default='finish_line')
    
    # Путь к SDF/URDF модели (нужно создать)
    # Предполагается что модель лежит в models/finish_line/
    pkg_share = get_package_share_directory('finish_detection')
    model_path = os.path.join(pkg_share, 'models', 'finish_line', 'model.sdf')
    
    # Spawn модели через Gazebo service
    spawn_entity = ExecuteProcess(
        cmd=[
            'ros2', 'run', 'gazebo_ros', 'spawn_entity.py',
            '-entity', model_name,
            '-file', model_path,
            '-x', x_pos,
            '-y', y_pos,
            '-z', z_pos,
            '-R', roll,
            '-P', pitch,
            '-Y', yaw
        ],
        output='screen'
    )
    
    return LaunchDescription([
        DeclareLaunchArgument('x', default_value='0.1317', description='X position'),
        DeclareLaunchArgument('y', default_value='-2.2171', description='Y position'),
        DeclareLaunchArgument('z', default_value='0.0338', description='Z position'),
        DeclareLaunchArgument('roll', default_value='0.0000', description='Roll angle'),
        DeclareLaunchArgument('pitch', default_value='0.0084', description='Pitch angle'),
        DeclareLaunchArgument('yaw', default_value='-0.0918', description='Yaw angle'),
        DeclareLaunchArgument('model_name', default_value='finish_line', description='Model name'),
        
        spawn_entity,
    ])
