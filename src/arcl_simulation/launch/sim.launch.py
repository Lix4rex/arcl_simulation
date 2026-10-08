import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import AppendEnvironmentVariable, DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_name = "arcl_simulation"
    pkg = get_package_share_directory(pkg_name)
    world = os.path.join(pkg, 'worlds', 'arene_omni.sdf')

    with open(os.path.join(pkg, 'urdf', 'robot_omni.urdf')) as f:
        robot_description = f.read()

    gz = IncludeLaunchDescription(PythonLaunchDescriptionSource(
        os.path.join(get_package_share_directory('ros_gz_sim'), 'launch', 'gz_sim.launch.py')),
        launch_arguments={'gz_args': f'-r {world}'}.items()
    )

    bridge = Node(
        package='ros_gz_bridge', 
        executable='parameter_bridge', 
        output='screen',
        parameters=[{'config_file': os.path.join(pkg, 'config', 'ros_bridge.yaml'), 'use_sim_time': True}]
    )

    rsp = Node(
        package='robot_state_publisher', 
        executable='robot_state_publisher', 
        output='screen',
        parameters=[{'robot_description': robot_description, 'use_sim_time': True}]
    )

    rviz = Node(
        package='rviz2', 
        executable='rviz2', 
        output='screen',
        arguments=['-d', os.path.join(pkg, 'rviz', 'arcl.rviz')],
        parameters=[{'use_sim_time': True, "--params-file": os.path.join(pkg, 'config', 'ekf_params.yaml')}],
        condition=IfCondition(LaunchConfiguration('rviz'))
    )

    robot_control = Node(
        package=pkg_name,
        executable="robot_control"
    )

    ekf_solo = Node(
        package=pkg_name,
        executable="ekf_solo"
    )

    table_markers = Node(
        package=pkg_name,
        executable="table_markers"
    )

    return LaunchDescription([
        DeclareLaunchArgument('rviz', default_value='true'),
        AppendEnvironmentVariable('GZ_SIM_RESOURCE_PATH', os.path.join(pkg, 'worlds')),
        AppendEnvironmentVariable('GZ_SIM_RESOURCE_PATH',os.path.join(pkg, 'models')),
        gz, bridge, rsp, rviz, robot_control, ekf_solo, table_markers
    ])
