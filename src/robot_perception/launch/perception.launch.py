import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():

    return LaunchDescription([
        Node(
            package='robot_perception',
            executable='yolo_perception_node',
            name='yolo_perception',
            output='screen',
            parameters=[{
                'use_sim_time':       True,

                # Model path — change to OpenVINO folder for 2-3x speedup:
                # 'model_path': '/home/balakarthik/yolov8n_openvino_model/',
                # Or custom trained model:
                # 'model_path': '/home/balakarthik/panda_yolo_training/custom_v1/weights/best.pt',
                'model_path':         '/home/balakarthik/yolov8n_openvino_model',

                # Confidence threshold — detections below this are ignored.
                # Lower (0.3) = more detections but more false positives.
                # Higher (0.7) = fewer false positives but may miss objects.
                'confidence_thresh':  0.5,

                # Camera pose in world frame — must match scene.world exactly
                'camera_x':           1.55,
                'camera_y':           0.0,
                'camera_z':           2.40,

                # Camera resolution — must match scene.world sensor settings
                'image_width':        1280,
                'image_height':       720,

                # Horizontal field of view in radians — from scene.world
                # 1.0472 rad = 60 degrees
                'horizontal_fov':     1.0472,

                # Set False to disable debug image (saves CPU time)
                'publish_debug_img':  True,
            }],
        )
    ])