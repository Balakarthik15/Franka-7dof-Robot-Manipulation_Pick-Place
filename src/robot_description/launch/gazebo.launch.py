import os
from os import pathsep
from pathlib import Path
from ament_index_python.packages import get_package_share_directory
 
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    SetEnvironmentVariable,
    TimerAction,
    RegisterEventHandler,
)
from launch.event_handlers import OnProcessExit
from launch.substitutions import (
    Command, LaunchConfiguration, PathJoinSubstitution, PythonExpression
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
 
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    # Package name constant
    pkg_name = "robot_description"
    pkg_share = get_package_share_directory(pkg_name)

    # 1. Arguments
    model_arg = DeclareLaunchArgument(
        name="model", 
        default_value=os.path.join(pkg_share, "urdf", "robot.urdf.xacro"),
        description="Absolute path to robot urdf file"
    )

    world_name_arg = DeclareLaunchArgument(
        name="world_name", 
        default_value="scene"
    )

    # 2. Paths
    # Ignition uses .sdf for native worlds, but can load .world files
    world_path = PathJoinSubstitution([
            pkg_share,
            "worlds",
            PythonExpression(expression=["'", LaunchConfiguration("world_name"), "'", " + '.world'"])
        ]
    )

    # Resource Path: Points to the directory ABOVE the package so 'package://robot_description' resolves
    model_path = str(Path(pkg_share).parent.resolve())
    
    # Add the internal 'models' folder if it exists
    internal_models = os.path.join(pkg_share, 'models')
    if os.path.exists(internal_models):
        model_path += pathsep + internal_models

    # Environment Variable for Ignition
    gazebo_resource_path = SetEnvironmentVariable(
        "IGN_GAZEBO_RESOURCE_PATH",
        model_path
    )

    # 3. Robot Description Processing
    ros_distro = os.environ.get("ROS_DISTRO", "humble")
    is_ignition = "True" if ros_distro == "humble" else "False"

    robot_description = ParameterValue(Command([
            "xacro ",
            LaunchConfiguration("model"),
            " is_ignition:=",
            is_ignition
        ]),
        value_type=str
    )

    # 4. Nodes
    robot_state_publisher_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        parameters=[{"robot_description": robot_description,
                     "use_sim_time": True}]
    )

    # Ignition Gazebo Launch
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(get_package_share_directory("ros_gz_sim"), "launch", "gz_sim.launch.py")
        ]),
        launch_arguments={
            # -v 4: verbose, -r: run immediately
            "gz_args": PythonExpression(["'", world_path, " -v 4 -r'"])
        }.items()
    )

    # Spawn Entity in Ignition
    gz_spawn_entity = Node(
        package="ros_gz_sim",
        executable="create",
        output="screen",
        arguments=[
            "-topic", "robot_description",
            "-name", "Franka_panda",
            "-x", "0.40",  
            "-y", "0.0",  
            "-z", "0.55", # Spawning slightly above ground to prevent clipping
        ],
    )

    # ── 4. Controller spawners ─────────────────────────────────────────────
    # Connect to controller_manager that ign_ros2_control started inside Gazebo.
    # Never use ros2_control_node here — ign_ros2_control/IgnitionSystem
    # is NOT a standalone hardware plugin; it only works inside Ignition Gazebo.
 


    controller_manager = Node(
        package="controller_manager",
        executable="ros2_control_node",
        parameters=[
            {"robot_description": robot_description,
             "use_sim_time": True},
            os.path.join(
                get_package_share_directory("robot_controllers"),
                "config",
                "ros2_controllers.yaml",
            ),
        ],
    )

    joint_state_broadcaster_spawner = Node(
        package="controller_manager",
        executable="spawner",
        output="screen",
        arguments=["joint_state_broadcaster",
                   "--controller-manager", "/controller_manager"],
    )
 
    arm_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        output="screen",
        arguments=["arm_controller",
                   "--controller-manager", "/controller_manager"],
    )
 
    gripper_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        output="screen",
        arguments=["gripper_controller",
                   "--controller-manager", "/controller_manager"],
    )
 
    # Wait 10s for Ignition + ign_ros2_control to fully initialise,
    # then spawn joint_state_broadcaster first, then arm + gripper after it exits
    delayed_jsb = TimerAction(
        period=10.0,
        actions=[joint_state_broadcaster_spawner]
    )
 
    arm_after_jsb = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=joint_state_broadcaster_spawner,
            on_exit=[arm_controller_spawner],
        )
    )
 
    gripper_after_jsb = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=joint_state_broadcaster_spawner,
            on_exit=[gripper_controller_spawner],
        )
    )
 
    # ── 5. Bridges ─────────────────────────────────────────────────────────
    gz_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        output="screen",
        arguments=[
            "/clock@rosgraph_msgs/msg/Clock[ignition.msgs.Clock",
            "/panda/overhead_camera/camera_info"
            "@sensor_msgs/msg/CameraInfo[ignition.msgs.CameraInfo",
        ],
    )
 
    image_bridge = Node(
        package="ros_gz_image",
        executable="image_bridge",
        output="screen",
        arguments=["/panda/overhead_camera/image_raw"]
    )
 
    return LaunchDescription([
        model_arg,
        world_name_arg,
        gazebo_resource_path,
        robot_state_publisher_node,
        gazebo,
        gz_spawn_entity,
        delayed_jsb,
        arm_after_jsb,
        gripper_after_jsb,
        gz_bridge,
        image_bridge,
    ])
 