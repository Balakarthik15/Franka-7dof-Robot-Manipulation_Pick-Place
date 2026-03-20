#!/usr/bin/env python3


import math
from threading import Thread

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup

from geometry_msgs.msg import PoseArray
from std_msgs.msg import String

from pymoveit2 import MoveIt2, GripperInterface
from pymoveit2.robots import panda


class PickPlaceNode(Node):

    # ── Bin positions (x, y, z) — matches scene.world ─────────────────────
    BIN_POSITIONS = {
        'bin_red':   (0.52,  0.25, 1.05),
        'bin_blue':  (0.52,  0.00, 1.05),
        'bin_green': (0.52, -0.25, 1.05),
    }

    # ── Gripper pointing straight down ─────────────────────────────────────
    # Quaternion (x=0, y=1, z=0, w=0) = 180° rotation around Y = facing down
    GRASP_QUAT = [0.0, 1.0, 0.0, 0.0]

    # ── Approach / retreat heights ─────────────────────────────────────────
    PRE_GRASP_OFFSET = 0.15   # 15cm above object
    LIFT_OFFSET      = 0.15   # 15cm lift after grasp

    # ── Home joint configuration — matches SRDF 'ready' pose ──────────────
    HOME_JOINTS = [
        0.0,           # panda_joint1
        -0.785398,     # panda_joint2  (-45°)
        0.0,           # panda_joint3
        -2.356194,     # panda_joint4  (-135°)
        0.0,           # panda_joint5
        1.570796,      # panda_joint6  (+90°)
        0.785398,      # panda_joint7  (+45°)
    ]

    def __init__(self):
        super().__init__('pick_place_node')

        # ── Parameters ────────────────────────────────────────────────────
        self.declare_parameter('velocity_scale', 0.3)
        self.declare_parameter('accel_scale',    0.3)
        vel = self.get_parameter('velocity_scale').value
        acc = self.get_parameter('accel_scale').value

        self.callback_group = ReentrantCallbackGroup()

        # ── pymoveit2 — Arm interface ──────────────────────────────────────
        # Uses your SRDF group name 'arm' (not panda.MOVE_GROUP_ARM)
        self.moveit2 = MoveIt2(
            node=self,
            joint_names=panda.joint_names(),
            base_link_name=panda.base_link_name(),
            end_effector_name=panda.end_effector_name(),
            group_name='arm',                       # your SRDF group name
            callback_group=self.callback_group,
        )
        self.moveit2.max_velocity     = vel
        self.moveit2.max_acceleration = acc

        # ── pymoveit2 — Gripper interface ──────────────────────────────────
        # Uses your SRDF group name 'hand'
        self.gripper = GripperInterface(
            node=self,
            gripper_joint_names=panda.gripper_joint_names(),
            open_gripper_joint_positions=panda.OPEN_GRIPPER_JOINT_POSITIONS,
            closed_gripper_joint_positions=panda.CLOSED_GRIPPER_JOINT_POSITIONS,
            gripper_group_name='hand',              # your SRDF group name
            callback_group=self.callback_group,
            gripper_command_action_name='/gripper_controller/gripper_cmd',
        )

        # ── State ──────────────────────────────────────────────────────────
        self.latest_poses   = []
        self.detections     = []
        self.is_busy        = False

        # ── Subscribers ────────────────────────────────────────────────────
        self.create_subscription(
            PoseArray, '/detected_objects/poses',
            self.poses_callback, 10,
            callback_group=self.callback_group)

        self.create_subscription(
            String, '/detected_objects/labels',
            self.labels_callback, 10,
            callback_group=self.callback_group)

        # ── Pick loop timer ────────────────────────────────────────────────
        self.create_timer(2.0, self.pick_loop,
                          callback_group=self.callback_group)

        # ── Move to home on startup ────────────────────────────────────────
        self.get_logger().info('Moving to home pose...')
        self.moveit2.move_to_configuration(self.HOME_JOINTS)
        self.moveit2.wait_until_executed()
        self.get_logger().info('Pick & Place node ready — waiting for detections')

    # ──────────────────────────────────────────────────────────────────────
    # SUBSCRIBERS
    # ──────────────────────────────────────────────────────────────────────

    def poses_callback(self, msg: PoseArray):
        self.latest_poses = list(msg.poses)

    def labels_callback(self, msg: String):
        if not msg.data:
            return
        self.detections = []
        for entry in msg.data.split(','):
            parts = entry.strip().split(':')
            if len(parts) == 3:
                self.detections.append({
                    'class_name': parts[0],
                    'bin':        parts[1],
                    'confidence': float(parts[2]),
                })

    # ──────────────────────────────────────────────────────────────────────
    # PICK LOOP — called every 2 seconds
    # ──────────────────────────────────────────────────────────────────────

    def pick_loop(self):
        """Check for detected objects and start a pick if robot is idle."""
        if self.is_busy:
            return
        if not self.detections or not self.latest_poses:
            return

        # Match poses to labels (same order from perception node)
        objects = []
        for i, det in enumerate(self.detections):
            if i < len(self.latest_poses):
                objects.append({
                    'class_name': det['class_name'],
                    'bin':        det['bin'],
                    'confidence': det['confidence'],
                    'pose':       self.latest_poses[i],
                })

        if not objects:
            return

        # Pick highest confidence detection first
        objects.sort(key=lambda x: x['confidence'], reverse=True)
        target = objects[0]

        self.get_logger().info(
            f"Picking: {target['class_name']} → {target['bin']} "
            f"(conf={target['confidence']:.2f})"
        )

        self.is_busy = True
        Thread(target=self._execute_pick, args=(target,), daemon=True).start()

    # ──────────────────────────────────────────────────────────────────────
    # PICK EXECUTION
    # ──────────────────────────────────────────────────────────────────────

    def _execute_pick(self, target: dict):
        """Execute 8-step pick-and-place using pymoveit2."""
        try:
            ox = target['pose'].position.x
            oy = target['pose'].position.y
            oz = target['pose'].position.z
            bn = target['bin']

            # ── Step 1: Home ───────────────────────────────────────────────
            self.get_logger().info('Step 1/8: Home...')
            self.moveit2.move_to_configuration(self.HOME_JOINTS)
            self.moveit2.wait_until_executed()

            # ── Step 2: Open gripper ───────────────────────────────────────
            self.get_logger().info('Step 2/8: Open gripper...')
            self.gripper.open()
            self.gripper.wait_until_executed()

            # ── Step 3: Pre-grasp (15cm above object) ─────────────────────
            self.get_logger().info('Step 3/8: Pre-grasp...')
            self.moveit2.move_to_pose(
                position=[ox, oy, oz + self.PRE_GRASP_OFFSET],
                quat_xyzw=self.GRASP_QUAT
            )
            self.moveit2.wait_until_executed()

            # ── Step 4: Descend to object (Cartesian straight down) ────────
            self.get_logger().info('Step 4/8: Descend to grasp...')
            self.moveit2.move_to_pose(
                position=[ox, oy, oz],
                quat_xyzw=self.GRASP_QUAT,
                cartesian=True       # straight line motion — safe for descent
            )
            self.moveit2.wait_until_executed()

            # ── Step 5: Close gripper ──────────────────────────────────────
            self.get_logger().info('Step 5/8: Close gripper...')
            self.gripper.close()
            self.gripper.wait_until_executed()

            # ── Step 6: Lift (Cartesian straight up) ──────────────────────
            self.get_logger().info('Step 6/8: Lift...')
            self.moveit2.move_to_pose(
                position=[ox, oy, oz + self.LIFT_OFFSET],
                quat_xyzw=self.GRASP_QUAT,
                cartesian=True
            )
            self.moveit2.wait_until_executed()

            # ── Step 7: Move to bin ────────────────────────────────────────
            self.get_logger().info(f'Step 7/8: Move to {bn}...')
            bx, by, bz = self.BIN_POSITIONS[bn]
            self.moveit2.move_to_pose(
                position=[bx, by, bz],
                quat_xyzw=self.GRASP_QUAT
            )
            self.moveit2.wait_until_executed()

            # ── Step 8: Release and return home ───────────────────────────
            self.get_logger().info('Step 8/8: Release...')
            self.gripper.open()
            self.gripper.wait_until_executed()

            self.moveit2.move_to_configuration(self.HOME_JOINTS)
            self.moveit2.wait_until_executed()

            self.get_logger().info(
                f"✅ {target['class_name']} sorted into {bn}")

        except Exception as e:
            self.get_logger().error(f'Pick failed: {e}')
            # Return home on failure
            self.moveit2.move_to_configuration(self.HOME_JOINTS)
            self.moveit2.wait_until_executed()

        finally:
            self.is_busy = False


# ──────────────────────────────────────────────────────────────────────────
def main():
    rclpy.init()
    node = PickPlaceNode()

   
    executor = rclpy.executors.MultiThreadedExecutor(2)
    executor.add_node(node)

    executor_thread = Thread(target=executor.spin, daemon=True)
    executor_thread.start()

    try:
        executor_thread.join()
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()