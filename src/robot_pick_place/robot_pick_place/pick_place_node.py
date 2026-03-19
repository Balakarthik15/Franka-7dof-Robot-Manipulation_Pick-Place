#!/usr/bin/env python3

from threading import Thread, Lock
import time

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.time import Time
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor

from std_msgs.msg import String
from geometry_msgs.msg import Pose, PoseArray, PoseStamped

from tf2_ros import Buffer, TransformListener
from tf2_geometry_msgs import do_transform_pose

from pymoveit2 import MoveIt2


class PickPlaceNode(Node):
    def __init__(self):
        super().__init__("pick_place_node")

        self.callback_group = ReentrantCallbackGroup()
        self.lock = Lock()
        self.is_busy = False

        # --------------------------------------------------
        # Parameters
        # --------------------------------------------------
        self.declare_parameter("approach_height", 0.15)
        self.declare_parameter("grasp_z_offset", 0.00)
        self.declare_parameter("lift_height", 0.18)
        self.declare_parameter("bin_approach_offset", 0.12)
        self.declare_parameter("arm_velocity", 0.20)
        self.declare_parameter("arm_acceleration", 0.20)
        self.declare_parameter("hand_velocity", 0.50)
        self.declare_parameter("hand_acceleration", 0.50)
        self.declare_parameter("pick_loop_period", 2.0)
        self.declare_parameter("move_to_start_on_boot", True)

        self.approach_height = float(self.get_parameter("approach_height").value)
        self.grasp_z_offset = float(self.get_parameter("grasp_z_offset").value)
        self.lift_height = float(self.get_parameter("lift_height").value)
        self.bin_approach_offset = float(self.get_parameter("bin_approach_offset").value)

        self.arm_velocity = float(self.get_parameter("arm_velocity").value)
        self.arm_acceleration = float(self.get_parameter("arm_acceleration").value)
        self.hand_velocity = float(self.get_parameter("hand_velocity").value)
        self.hand_acceleration = float(self.get_parameter("hand_acceleration").value)

        self.pick_loop_period = float(self.get_parameter("pick_loop_period").value)
        self.move_to_start_on_boot = bool(self.get_parameter("move_to_start_on_boot").value)

        # --------------------------------------------------
        # TF
        # --------------------------------------------------
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # --------------------------------------------------
        # MoveIt2 interfaces
        # --------------------------------------------------
        self.arm = MoveIt2(
            node=self,
            joint_names=[
                "panda_joint1",
                "panda_joint2",
                "panda_joint3",
                "panda_joint4",
                "panda_joint5",
                "panda_joint6",
                "panda_joint7",
            ],
            base_link_name="panda_link0",
            end_effector_name="panda_hand",
            group_name="arm",
            callback_group=self.callback_group,
        )
        self.arm.max_velocity = self.arm_velocity
        self.arm.max_acceleration = self.arm_acceleration

        self.hand = MoveIt2(
            node=self,
            joint_names=[
                "panda_finger_joint1",
                "panda_finger_joint2",
            ],
            base_link_name="panda_link0",
            end_effector_name="panda_hand",
            group_name="hand",
            callback_group=self.callback_group,
        )
        self.hand.max_velocity = self.hand_velocity
        self.hand.max_acceleration = self.hand_acceleration

        # --------------------------------------------------
        # State storage
        # --------------------------------------------------
        self.latest_pose_frame = "world"
        self.latest_poses = []
        self.latest_detections = []

        # --------------------------------------------------
        # Predefined poses
        # --------------------------------------------------
        self.start_joints = [
            0.0,
            -0.785398,
            0.0,
            -2.356194,
            0.0,
            1.570796,
            0.785398,
        ]

        self.home_joints = [
            0.0,
            -0.785398,
            0.0,
            -2.356194,
            0.0,
            1.570796,
            0.785398,
        ]

        # Gripper targets for 2-finger trajectory control
        self.hand_open = [0.04, 0.04]
        self.hand_closed = [0.01, 0.01]

        # End-effector orientation: straight down
        # This matches what your perception node already uses in published poses. :contentReference[oaicite:3]{index=3}
        self.grasp_quat = [0.0, 1.0, 0.0, 0.0]

        # Bin positions in WORLD frame
        self.bin_positions_world = {
            "bin_red":   [0.52,  0.25, 1.05],
            "bin_blue":  [0.52,  0.00, 1.05],
            "bin_green": [0.52, -0.25, 1.05],
        }

        # --------------------------------------------------
        # Subscribers
        # --------------------------------------------------
        self.create_subscription(
            PoseArray,
            "/detected_objects/poses",
            self.poses_callback,
            10,
        )

        self.create_subscription(
            String,
            "/detected_objects/labels",
            self.labels_callback,
            10,
        )

        self.create_timer(self.pick_loop_period, self.pick_loop)

        self.get_logger().info("Pick & Place node started with pymoveit2")
        self.get_logger().info("Arm group: arm")
        self.get_logger().info("Hand group: hand")
        self.get_logger().info("Waiting for detections...")

        if self.move_to_start_on_boot:
            # Wait a moment for joint states / controllers
            time.sleep(1.0)
            try:
                self.move_arm_joints(self.start_joints, "Moving to start joint configuration")
            except Exception as e:
                self.get_logger().warn(f"Initial move_to_start skipped: {e}")

    # --------------------------------------------------
    # Subscribers
    # --------------------------------------------------
    def poses_callback(self, msg: PoseArray):
        with self.lock:
            self.latest_pose_frame = msg.header.frame_id if msg.header.frame_id else "world"
            self.latest_poses = list(msg.poses)

    def labels_callback(self, msg: String):
        parsed = []
        if msg.data:
            for entry in msg.data.split(","):
                parts = entry.strip().split(":")
                if len(parts) != 3:
                    continue
                try:
                    parsed.append({
                        "class_name": parts[0],
                        "bin": parts[1],
                        "confidence": float(parts[2]),
                    })
                except ValueError:
                    continue

        with self.lock:
            self.latest_detections = parsed

    # --------------------------------------------------
    # Helpers
    # --------------------------------------------------
    def transform_pose(self, pose, source_frame, target_frame):
        try:
            transform = self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                Time(),
                timeout=Duration(seconds=1.0),
            )

            # Pass Pose, not PoseStamped
            transformed_pose = do_transform_pose(pose, transform)
            return transformed_pose

        except Exception as e:
            self.get_logger().error(
                f"Failed to transform pose from {source_frame} to {target_frame}: {e}"
            )
            return None

    def transform_xyz(self, xyz, source_frame, target_frame):
        try:
            pose = PoseStamped()
            pose.header.frame_id = source_frame
            pose.header.stamp = self.get_clock().now().to_msg()
            pose.pose.position.x = float(xyz[0])
            pose.pose.position.y = float(xyz[1])
            pose.pose.position.z = float(xyz[2])
            pose.pose.orientation.w = 1.0

            transform = self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                Time(),
                timeout=Duration(seconds=1.0),
            )

            # Again: pass only pose.pose
            transformed_pose = do_transform_pose(pose.pose, transform)

            return [
                transformed_pose.position.x,
                transformed_pose.position.y,
                transformed_pose.position.z,
            ]

        except Exception as e:
            self.get_logger().error(
                f"Failed to transform xyz from {source_frame} to {target_frame}: {e}"
            )
            return None

    def move_arm_joints(self, joints, desc="Arm joint move"):
        self.get_logger().info(desc)
        self.arm.move_to_configuration(joints)
        self.arm.wait_until_executed()
        time.sleep(0.2)

    def move_hand_joints(self, joints, desc="Hand move"):
        self.get_logger().info(desc)
        self.hand.move_to_configuration(joints)
        self.hand.wait_until_executed()
        time.sleep(0.2)

    def move_arm_pose(self, position, quat_xyzw, cartesian=False, desc="Arm pose move"):
        self.get_logger().info(desc)
        self.arm.move_to_pose(
            position=position,
            quat_xyzw=quat_xyzw,
            cartesian=cartesian,
        )
        self.arm.wait_until_executed()
        time.sleep(0.2)

    # --------------------------------------------------
    # Pick loop
    # --------------------------------------------------
    def pick_loop(self):
        if self.is_busy:
            return

        with self.lock:
            if not self.latest_poses or not self.latest_detections:
                return

            count = min(len(self.latest_poses), len(self.latest_detections))
            if count == 0:
                return

            objects = []
            for i in range(count):
                objects.append({
                    "pose": self.latest_poses[i],
                    "frame_id": self.latest_pose_frame,
                    "class_name": self.latest_detections[i]["class_name"],
                    "bin": self.latest_detections[i]["bin"],
                    "confidence": self.latest_detections[i]["confidence"],
                })

        objects.sort(key=lambda x: x["confidence"], reverse=True)
        target = objects[0]

        self.is_busy = True
        Thread(target=self.execute_pick_place, args=(target,), daemon=True).start()

    # --------------------------------------------------
    # Main sequence
    # --------------------------------------------------
    def execute_pick_place(self, target):
        try:
            self.get_logger().info(
                f"Starting pick: {target['class_name']} -> {target['bin']} "
                f"(conf={target['confidence']:.2f}, frame={target['frame_id']})"
            )

            # Detection poses arrive in world frame from perception. 
            # Convert them to panda_link0 for arm motion targets.
            obj_pose_base = self.transform_pose(
                target["pose"],
                target["frame_id"],
                "panda_link0",
            )
            if obj_pose_base is None:
                self.get_logger().error("Object pose transform failed")
                return

            pick_x = obj_pose_base.position.x
            pick_y = obj_pose_base.position.y
            pick_z = obj_pose_base.position.z + self.grasp_z_offset

            pre_grasp = [pick_x, pick_y, pick_z + self.approach_height]
            grasp_pose = [pick_x, pick_y, pick_z]
            lift_pose = [pick_x, pick_y, pick_z + self.lift_height]

            if target["bin"] not in self.bin_positions_world:
                self.get_logger().error(f"Unknown bin: {target['bin']}")
                return

            bin_base = self.transform_xyz(
                self.bin_positions_world[target["bin"]],
                "world",
                "panda_link0",
            )
            if bin_base is None:
                self.get_logger().error("Bin pose transform failed")
                return

            bin_approach = [
                bin_base[0],
                bin_base[1],
                bin_base[2] + self.bin_approach_offset,
            ]
            bin_drop = bin_base

            # Sequence
            self.move_arm_joints(self.home_joints, "Step 1/8: Move home")
            self.move_hand_joints(self.hand_open, "Step 2/8: Open hand")

            self.move_arm_pose(
                position=pre_grasp,
                quat_xyzw=self.grasp_quat,
                cartesian=False,
                desc="Step 3/8: Move to pre-grasp",
            )

            self.move_arm_pose(
                position=grasp_pose,
                quat_xyzw=self.grasp_quat,
                cartesian=True,
                desc="Step 4/8: Move to grasp",
            )

            self.move_hand_joints(self.hand_closed, "Step 5/8: Close hand")

            self.move_arm_pose(
                position=lift_pose,
                quat_xyzw=self.grasp_quat,
                cartesian=True,
                desc="Step 6/8: Lift object",
            )

            self.move_arm_pose(
                position=bin_approach,
                quat_xyzw=self.grasp_quat,
                cartesian=False,
                desc=f"Step 7/8: Move above {target['bin']}",
            )

            self.move_arm_pose(
                position=bin_drop,
                quat_xyzw=self.grasp_quat,
                cartesian=True,
                desc=f"Step 7b/8: Move down to {target['bin']}",
            )

            self.move_hand_joints(self.hand_open, "Step 8/8: Release object")

            self.move_arm_pose(
                position=bin_approach,
                quat_xyzw=self.grasp_quat,
                cartesian=True,
                desc="Step 8b/8: Retreat upward",
            )

            self.move_arm_joints(self.home_joints, "Return home")

            self.get_logger().info(
                f"Pick-and-place complete: {target['class_name']} -> {target['bin']}"
            )

        except Exception as e:
            self.get_logger().error(f"Pick-and-place failed: {e}")
            try:
                self.move_arm_joints(self.home_joints, "Recovery: Return home")
            except Exception:
                pass
        finally:
            self.is_busy = False


def main(args=None):
    rclpy.init(args=args)
    node = PickPlaceNode()

    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    executor_thread = Thread(target=executor.spin, daemon=True)
    executor_thread.start()

    try:
        executor_thread.join()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()