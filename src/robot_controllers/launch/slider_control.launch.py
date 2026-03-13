import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():

    # joint_state_publisher_gui publishes to /joint_states by default.
    # Remap it to /joint_commands so it does NOT conflict with
    # joint_state_broadcaster which owns /joint_states.
    joint_state_publisher_gui_node = Node(
        package="joint_state_publisher_gui",
        executable="joint_state_publisher_gui",
        parameters=[{"use_sim_time": True}],
        remappings=[
            ("/joint_states", "/joint_commands"),
        ]
    )

    # slider_control node reads /joint_commands (remapped GUI output)
    # and publishes directly to /arm_controller/joint_trajectory
    slider_control_node = Node(
        package="robot_controllers",
        executable="slider_control",
        output="screen",
        parameters=[{"use_sim_time": True}],
    )

    return LaunchDescription([
        joint_state_publisher_gui_node,
        slider_control_node,
    ])