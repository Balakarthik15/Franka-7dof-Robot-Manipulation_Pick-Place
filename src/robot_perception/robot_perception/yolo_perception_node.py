#!/usr/bin/env python3
"""
yolo_perception_node.py
========================
Subscribes to the overhead camera image, runs YOLOv8 inference,
converts 2D detections to 3D world coordinates, and publishes
object poses for the pick-and-place node to consume.

Topics Published:
  /detected_objects/poses   → geometry_msgs/PoseArray   (3D positions)
  /detected_objects/labels  → std_msgs/String            (class:bin:confidence)
  /detected_objects/markers → visualization_msgs/MarkerArray (RViz2 spheres)
  /yolo/debug_image         → sensor_msgs/Image          (annotated camera feed)

Topics Subscribed:
  /panda/overhead_camera/image_raw → sensor_msgs/Image

Usage:
  ros2 launch panda_perception perception.launch.py
"""

import math
import rclpy
from rclpy.node import Node
import cv2
import numpy as np
from cv_bridge import CvBridge
from ultralytics import YOLO

from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseArray, Pose
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray


class YoloPerceptionNode(Node):

    # ── Known object heights above the floor (metres) ─────────────────────
    # These match the z values in scene.world exactly.
    # Used to compute depth without a depth sensor.
    OBJECT_HEIGHTS = {
        'bottle':      0.936,   # coke can centre (radius 0.033 + table 0.875 + small offset)
        'sports ball': 0.911,   # cricket ball centre (radius 0.036 + table 0.875)
        'cube':        0.900,   # wood cube centre (half-side 0.025 + table 0.875)
        'small_box':   0.910,   # small box centre (half-side 0.035 + table 0.875)
    }

    # ── Which bin each detected class goes into ────────────────────────────
    BIN_ASSIGNMENT = {
        'bottle':      'bin_red',
        'sports ball': 'bin_blue',
        'cube':        'bin_green',
        'small_box':   'bin_red',
    }

    # ── Bin world positions (x, y) — for pick-and-place node reference ────
    BIN_POSITIONS = {
        'bin_red':   (0.52,  0.25, 0.895),
        'bin_blue':  (0.52,  0.00, 0.895),
        'bin_green': (0.52, -0.25, 0.895),
    }

    def __init__(self):
        super().__init__('yolo_perception_node')

        # ── Declare ROS2 parameters ──────────────────────────────────────
        # All values can be overridden in the launch file or at runtime.
        self.declare_parameter('model_path',        'yolov8n.pt')
        self.declare_parameter('confidence_thresh',  0.5)
        self.declare_parameter('camera_x',           1.55)
        self.declare_parameter('camera_y',           0.0)
        self.declare_parameter('camera_z',           2.40)
        self.declare_parameter('image_width',        1280)
        self.declare_parameter('image_height',       720)
        self.declare_parameter('horizontal_fov',     1.0472)   # 60 degrees in radians
        self.declare_parameter('publish_debug_img',  True)

        # ── Read parameter values ────────────────────────────────────────
        model_path      = self.get_parameter('model_path').value
        self.conf_thresh = self.get_parameter('confidence_thresh').value
        self.cam_x       = self.get_parameter('camera_x').value
        self.cam_y       = self.get_parameter('camera_y').value
        self.cam_z       = self.get_parameter('camera_z').value
        img_w            = self.get_parameter('image_width').value
        img_h            = self.get_parameter('image_height').value
        fov_rad          = self.get_parameter('horizontal_fov').value
        self.debug       = self.get_parameter('publish_debug_img').value

        # ── Compute camera intrinsics from field of view ─────────────────
        # Formula: fx = (image_width / 2) / tan(horizontal_fov / 2)
        # This derives the focal length in pixels from the FoV angle.
        # Ignition Gazebo uses square pixels so fx == fy.
        # Principal point (cx, cy) is always the image centre.
        self.fx = (img_w / 2.0) / math.tan(fov_rad / 2.0)
        self.fy = self.fx
        self.cx = img_w / 2.0
        self.cy = img_h / 2.0

        # ── Load YOLOv8 model ────────────────────────────────────────────
        # First run: downloads weights (~6MB for yolov8n.pt) automatically.
        # Subsequent runs: loads from local cache instantly.
        # To use OpenVINO: set model_path to 'yolov8n_openvino_model/'
        self.model  = YOLO(model_path)
        self.bridge = CvBridge()

        # ── Publishers ───────────────────────────────────────────────────
        self.pose_pub   = self.create_publisher(
            PoseArray,   '/detected_objects/poses',   10)
        self.label_pub  = self.create_publisher(
            String,      '/detected_objects/labels',  10)
        self.marker_pub = self.create_publisher(
            MarkerArray, '/detected_objects/markers', 10)
        self.debug_pub  = self.create_publisher(
            Image,       '/yolo/debug_image',         10)

        # ── Subscriber ───────────────────────────────────────────────────
        self.sub = self.create_subscription(
            Image,
            '/panda/overhead_camera/image_raw',
            self.image_callback,
            10
        )

        self.get_logger().info(
            f'YOLOv8 perception node started\n'
            f'  Model      : {model_path}\n'
            f'  Confidence : {self.conf_thresh}\n'
            f'  Camera     : x={self.cam_x} y={self.cam_y} z={self.cam_z}\n'
            f'  Intrinsics : fx={self.fx:.1f}  cx={self.cx}  cy={self.cy}'
        )

    # ──────────────────────────────────────────────────────────────────────
    def image_callback(self, msg: Image):
        """
        Main callback — called every time a new camera frame arrives.

        Pipeline:
          ROS2 Image msg
            → OpenCV numpy array  (cv_bridge)
            → YOLOv8 inference    (neural network)
            → filter by class + confidence
            → pixel_to_world()    (2D → 3D)
            → publish poses, labels, markers
            → publish debug image
        """

        # Step 1: Convert ROS2 Image message to OpenCV array
        # imgmsg_to_cv2 decodes the compressed bytes into a (H, W, 3) numpy array.
        # 'bgr8' = Blue-Green-Red byte order, 8 bits per channel (OpenCV default).
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # Step 2: Run YOLOv8 inference
        # Pass the numpy array directly — Ultralytics handles all preprocessing.
        # verbose=False suppresses the per-frame speed printout in the terminal.
        # Returns a list of Results objects (one per image; we pass one image).
        results = self.model(cv_image, verbose=False)

        # Step 3: Parse all detections from results
        detections = []

        # results[0].boxes = all bounding boxes detected in our single image
        for box in results[0].boxes:

            # box.conf is a 1-element tensor → float() extracts the Python float
            conf = float(box.conf[0])

            # Skip detections below the confidence threshold (default 0.5)
            if conf < self.conf_thresh:
                continue

            # box.cls is a 1-element tensor containing the integer class index
            class_id   = int(box.cls[0])
            class_name = self.model.names[class_id]   # e.g. 'bottle'

            # Only process classes that we know how to handle
            if class_name not in self.OBJECT_HEIGHTS:
                continue

            # box.xyxy[0] = [x1, y1, x2, y2] bounding box corners in pixels
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            # Pixel centre of the bounding box
            u = (x1 + x2) / 2.0   # horizontal pixel position
            v = (y1 + y2) / 2.0   # vertical pixel position

            # Step 4: Convert pixel centre to 3D world coordinate
            world_x, world_y, world_z = self.pixel_to_world(u, v, class_name)

            detections.append({
                'class_name': class_name,
                'confidence': conf,
                'pixel_u':    u,
                'pixel_v':    v,
                'world_x':    world_x,
                'world_y':    world_y,
                'world_z':    world_z,
                'bin':        self.BIN_ASSIGNMENT.get(class_name, 'bin_blue'),
                'bin_pos':    self.BIN_POSITIONS.get(
                                  self.BIN_ASSIGNMENT.get(class_name, 'bin_blue'),
                                  (0.52, 0.0, 0.895)),
                'bbox':       (x1, y1, x2, y2),
            })

        # Step 5: Publish detection results
        if detections:
            self.publish_poses(detections, msg.header)
            self.publish_labels(detections)
            self.publish_markers(detections)

            # Log summary to terminal
            summary = ', '.join(
                f"{d['class_name']}({d['confidence']:.2f})" for d in detections
            )
            self.get_logger().info(f'Detected: {summary}')

        # Step 6: Draw bounding boxes on image and publish for debugging
        if self.debug:
            debug_img = self.draw_detections(cv_image.copy(), detections)
            debug_msg = self.bridge.cv2_to_imgmsg(debug_img, encoding='bgr8')
            debug_msg.header = msg.header
            self.debug_pub.publish(debug_msg)

    # ──────────────────────────────────────────────────────────────────────
    def pixel_to_world(self, u: float, v: float, class_name: str):
        """
        Convert a 2D pixel coordinate (u, v) to a 3D world coordinate.

        Uses the pinhole camera model (inverse projection):
            X_camera = (u - cx) * Z / fx
            Y_camera = (v - cy) * Z / fy

        The camera is mounted overhead at (cam_x=1.55, cam_y=0, cam_z=2.40)
        pointing straight down (pitch = PI/2).

        With a straight-down camera:
            camera +X  ≡  world +X  (forward)
            camera +Y  ≡  world +Y  (left)
            camera +Z  points DOWN  (optical axis = -world_z)

        Args:
            u          : horizontal pixel (0 = left image edge)
            v          : vertical pixel   (0 = top image edge)
            class_name : detected class — used to look up known object height

        Returns:
            (world_x, world_y, world_z) in metres in the world frame
        """
        # Known z-height of this object type (from scene.world)
        object_z = self.OBJECT_HEIGHTS[class_name]

        # Depth along camera optical axis = camera height - object height
        # Camera points straight down, so depth = how far below the camera
        # the object surface is.
        depth = self.cam_z - object_z

        # Inverse pinhole projection:
        # Pixel offset from image centre × (depth / focal_length) = real distance
        # (u - cx) positive means object is to the RIGHT of camera centre
        # (v - cy) positive means object is BELOW camera centre in image
        local_x = (u - self.cx) * depth / self.fx
        local_y = (v - self.cy) * depth / self.fy

        # Translate from camera frame to world frame.
        # Camera is located at (cam_x, cam_y) in world frame.
        world_x = self.cam_x + local_x
        world_y = self.cam_y + local_y
        world_z = object_z   # use known table height — no depth sensor needed

        return world_x, world_y, world_z

    # ──────────────────────────────────────────────────────────────────────
    def publish_poses(self, detections, header):
        """
        Publish all detected object positions as a PoseArray.

        Each Pose contains:
          position    : 3D world coordinate (x, y, z) in metres
          orientation : gripper pointing straight down
                        quaternion (x=0, y=1, z=0, w=0) = 180° rotation around Y
                        This means the gripper approaches from above, fingers down.

        The pick-and-place node subscribes to /detected_objects/poses
        and uses each pose directly as a MoveIt2 setPoseTarget() goal.
        """
        pose_array = PoseArray()
        pose_array.header        = header
        pose_array.header.frame_id = 'world'

        for det in detections:
            pose = Pose()

            # 3D object position
            pose.position.x = det['world_x']
            pose.position.y = det['world_y']
            pose.position.z = det['world_z']

            # Gripper orientation: pointing straight down
            # Quaternion for 180° rotation around Y axis = (x=0, y=1, z=0, w=0)
            pose.orientation.x = 0.0
            pose.orientation.y = 1.0
            pose.orientation.z = 0.0
            pose.orientation.w = 0.0

            pose_array.poses.append(pose)

        self.pose_pub.publish(pose_array)

    # ──────────────────────────────────────────────────────────────────────
    def publish_labels(self, detections):
        """
        Publish detection labels as a comma-separated string.

        Format: 'class_name:bin_name:confidence,class_name:bin_name:confidence,...'
        Example: 'bottle:bin_red:0.87,sports ball:bin_blue:0.92'

        The pick-and-place node splits on ',' then ':' to get:
          [0] class_name  — what the object is
          [1] bin_name    — where to sort it
          [2] confidence  — how sure YOLO is

        Using a plain String is simpler than a custom message for this data.
        """
        entries = [
            f"{d['class_name']}:{d['bin']}:{d['confidence']:.2f}"
            for d in detections
        ]
        msg      = String()
        msg.data = ','.join(entries)
        self.label_pub.publish(msg)

    # ──────────────────────────────────────────────────────────────────────
    def publish_markers(self, detections):
        """
        Publish RViz2 visualisation markers (coloured spheres) at object positions.

        These are ONLY for visual debugging in RViz2.
        The robot does NOT use these — it uses /detected_objects/poses.

        Each sphere is coloured to match its target bin:
          Red sphere   → goes to bin_red
          Blue sphere  → goes to bin_blue
          Green sphere → goes to bin_green
        """
        marker_array = MarkerArray()

        BIN_COLORS = {
            'bin_red':   (1.0, 0.1, 0.1),
            'bin_blue':  (0.1, 0.1, 1.0),
            'bin_green': (0.1, 0.8, 0.1),
        }

        for i, det in enumerate(detections):
            marker = Marker()
            marker.header.frame_id = 'world'
            marker.ns              = 'detected_objects'
            marker.id              = i
            marker.type            = Marker.SPHERE
            marker.action          = Marker.ADD

            # Position: slightly above the actual object so sphere is visible
            marker.pose.position.x  = det['world_x']
            marker.pose.position.y  = det['world_y']
            marker.pose.position.z  = det['world_z'] + 0.06
            marker.pose.orientation.w = 1.0

            # Sphere diameter: 6cm — visible but not too large
            marker.scale.x = 0.06
            marker.scale.y = 0.06
            marker.scale.z = 0.06

            # Colour based on which bin this object goes to
            r, g, b = BIN_COLORS.get(det['bin'], (1.0, 1.0, 0.0))
            marker.color.r = r
            marker.color.g = g
            marker.color.b = b
            marker.color.a = 0.85   # slightly transparent

            marker_array.markers.append(marker)

        self.marker_pub.publish(marker_array)

    # ──────────────────────────────────────────────────────────────────────
    def draw_detections(self, img: np.ndarray, detections: list) -> np.ndarray:
        """
        Draw bounding boxes, labels, and 3D coordinates on the camera image.

        Returns the annotated image as a numpy array.
        Subscribe to /yolo/debug_image in rqt_image_view to see this live.

        Colour coding (BGR for OpenCV):
          Blue box   → bin_blue
          Red box    → bin_red
          Green box  → bin_green
        """
        # OpenCV uses BGR not RGB
        BOX_COLORS = {
            'bin_red':   (0, 0, 220),
            'bin_blue':  (220, 0, 0),
            'bin_green': (0, 180, 0),
        }

        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            name  = det['class_name']
            conf  = det['confidence']
            wx    = det['world_x']
            wy    = det['world_y']
            bin_  = det['bin']
            color = BOX_COLORS.get(bin_, (0, 220, 220))

            # Draw bounding box rectangle
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            # Draw class name + confidence above the box
            top_label = f'{name}  {conf:.2f}'
            cv2.putText(img, top_label,
                        (x1, max(y1 - 8, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

            # Draw 3D world coordinates below the box
            coord_label = f'world({wx:.3f}, {wy:.3f})'
            cv2.putText(img, coord_label,
                        (x1, y2 + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

            # Draw bin assignment below coordinates
            bin_label = f'→ {bin_}'
            cv2.putText(img, bin_label,
                        (x1, y2 + 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)

            # Draw a filled dot at the pixel centre of the detection
            cx_px = int((x1 + x2) / 2)
            cy_px = int((y1 + y2) / 2)
            cv2.circle(img, (cx_px, cy_px), 5, color, -1)

        # Draw detection count in top-left corner
        count_label = f'Detections: {len(detections)}'
        cv2.putText(img, count_label, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        return img


# ──────────────────────────────────────────────────────────────────────────
def main(args=None):
    rclpy.init(args=args)
    node = YoloPerceptionNode()
    try:
        rclpy.spin(node)       # blocks here, calling callbacks as messages arrive
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
