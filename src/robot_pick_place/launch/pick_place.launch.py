from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='robot_pick_place',
            executable='pick_place_node',
            name='pick_place_node',
            output='screen',
            parameters=[{
                'use_sim_time':    True,
                'planning_time':   5.0,    # seconds — increase if planning fails
                'velocity_scale':  0.3,    # 30% speed — increase to 0.5 once working
                'accel_scale':     0.3,
                'max_attempts':    3,      # OMPL retries per motion
            }],
        )
    ])