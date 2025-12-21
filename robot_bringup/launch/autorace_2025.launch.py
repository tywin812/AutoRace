import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, Command
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.actions import Node
from launch.actions import TimerAction

def launch_setup(context, *args, **kwargs):
    spawn_mode = LaunchConfiguration('spawn_mode').perform(context)
    
    # Default coordinates
    x = '0.8603'
    y = '0.1800'
    z = '0.08'
    yaw = '0.0'
    
    if spawn_mode == 'tunnel':
        x = '-2.1248'
        y = '0.5010'
        z = '0.0338'
        yaw = '-1.5876'

    create = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=['-name', 'robot',
                   '-topic', 'robot_description',
                   '-x', x,
                   '-y', y,
                   '-z', z,
                   '-Y', yaw,
                ],
        output='screen',
    )
    
    return [create]

def generate_launch_description():
    # Configure ROS nodes for launch

    # Setup project paths
    pkg_project_bringup = get_package_share_directory('robot_bringup')
    pkg_project_description = get_package_share_directory('robot_description')
    pkg_ros_gz_sim = get_package_share_directory('ros_gz_sim')

    urdf_path  =  os.path.join(pkg_project_description, 'urdf', 'robot.urdf.xacro')
    robot_desc = ParameterValue(Command(['xacro ', urdf_path]), value_type=str)

    # Setup to launch the simulator and Gazebo world
    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_ros_gz_sim, 'launch', 'gz_sim.launch.py')),
        launch_arguments={'gz_args': "-r course.sdf"}.items(),
    )

    # Spawn robot
    spawn_mode_arg = DeclareLaunchArgument(
        'spawn_mode',
        default_value='default',
        description='Spawn location mode (default, tunnel)'
    )

    # Takes the description and joint angles as inputs and publishes the 3D poses of the robot links
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='both',
        parameters=[
            {'robot_description': robot_desc},
            {'frame_prefix': "robot/"},
            {"use_sim_time": True}
        ]
    )

    # Visualize in RViz
    rviz = Node(
       package='rviz2',
       executable='rviz2',
       arguments=['-d', os.path.join(pkg_project_bringup, 'config', 'autorace_2023.rviz')],
       condition=IfCondition(LaunchConfiguration('rviz')),
       parameters=[
            {"use_sim_time": True}
        ]
    )

    # Bridge ROS topics and Gazebo messages for establishing communication
    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        parameters=[{
            'config_file': os.path.join(pkg_project_bringup, 'config', 'robot_bridge.yaml'),
            'qos_overrides./tf_static.publisher.durability': 'transient_local',
            "use_sim_time": True,
        }],
        output='screen'
    )

    # Lane detection nodes
    bird_view_node = Node(
        package='lane_detection',
        executable='bird_view',
        name='bird_view',
        output='screen',
        parameters=[{"use_sim_time": True}]
    )

    detect_lanes_node = Node(
        package='lane_detection',
        executable='detect_lanes',
        name='detect_lanes',
        output='screen',
        parameters=[{"use_sim_time": True}]
    )

    follow_lanes_node = Node(
        package='lane_detection',
        executable='follow_lanes',
        name='follow_lanes',
        output='screen',
        parameters=[{"use_sim_time": True}]
    )

    return LaunchDescription([
        spawn_mode_arg,
        gz_sim,
        DeclareLaunchArgument('rviz', default_value='true',
                              description='Open RViz.'),
        bridge,
        robot_state_publisher,
        rviz,
        bird_view_node,
        detect_lanes_node,
        follow_lanes_node,
        TimerAction(
            period=0.0,
            actions=[OpaqueFunction(function=launch_setup)])
    ])
