import os
from launch import LaunchDescription
from launch.actions import ExecuteProcess, SetEnvironmentVariable
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():

    pkg_name = 'robot_description' 

    pkg_share = get_package_share_directory(pkg_name)

    # Path to your local models folder
    local_models_path = os.path.join(pkg_share, 'models')

    # Path to your world file
    world_file = os.path.join(pkg_share, 'worlds', 'scene.world')

    return LaunchDescription([

        # ── Set local model path (uses local models, skips online DB) ──
        SetEnvironmentVariable(
            name='GAZEBO_MODEL_PATH',
            value=local_models_path + ':' + os.environ.get('GAZEBO_MODEL_PATH', '')
        ),

        # ── Disable online Gazebo model database completely ──
        SetEnvironmentVariable(
            name='GAZEBO_MODEL_DATABASE_URI',
            value=''
        ),

        # ── Launch Gazebo with your world ──
        ExecuteProcess(
            cmd=[
                'gazebo', '--verbose',
                '-s', 'libgazebo_ros_init.so',
                '-s', 'libgazebo_ros_factory.so',
                world_file
            ],
            output='screen'
        ),

    ])
