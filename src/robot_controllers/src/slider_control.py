#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from builtin_interfaces.msg import Duration
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

ARM_JOINTS = [
    "panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4",
    "panda_joint5", "panda_joint6", "panda_joint7"
]

class SliderControl(Node):
    def __init__(self):
        super().__init__("slider_control")
        self.arm_pub_ = self.create_publisher(
            JointTrajectory, "arm_controller/joint_trajectory", 10)

        # /joint_commands = GUI output remapped in launch file
        # avoids conflict with joint_state_broadcaster on /joint_states
        self.sub_ = self.create_subscription(
            JointState, "joint_commands", self.sliderCallback, 10)

        self.get_logger().info("Slider Control Node started")

    def sliderCallback(self, msg):
        # Look up by name — GUI order is not guaranteed
        name_to_pos = dict(zip(msg.name, msg.position))

        if not all(j in name_to_pos for j in ARM_JOINTS):
            return

        arm_controller = JointTrajectory()
        arm_controller.joint_names = ARM_JOINTS

        arm_goal = JointTrajectoryPoint()
        arm_goal.positions = [name_to_pos[j] for j in ARM_JOINTS]
        # time_from_start must be non-zero — controller rejects zero duration
        arm_goal.time_from_start = Duration(sec=0, nanosec=500_000_000)  # 0.5s

        arm_controller.points.append(arm_goal)
        self.arm_pub_.publish(arm_controller)

def main():
    rclpy.init()
    simple_publisher = SliderControl()
    rclpy.spin(simple_publisher)
    simple_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()