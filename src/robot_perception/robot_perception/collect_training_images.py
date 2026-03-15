#!/usr/bin/env python3
"""

Saves overhead camera frames for YOLOv8 custom training.


TWO MODES:
  Mode 1 — AUTO: saves one frame every N seconds automatically
  Mode 2 — MANUAL: press ENTER in terminal to save each frame

Usage:
  # Auto mode (saves every 3 seconds):
  ros2 run panda_perception collect_training_images

  # Manual mode (press ENTER to save):
  ros2 run panda_perception collect_training_images --ros-args -p auto_mode:=false

HOW TO GET VARIETY:
  In Ignition Gazebo, select an object with your mouse and drag it
  to a new position. Then save a frame. Repeat for each object.
  The camera is fixed — you vary positions by moving objects.

Images saved to: ~/training_data/images/
"""

import os
import threading
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2


class ImageCollector(Node):

    def __init__(self):
        super().__init__('image_collector')

        # ── Parameters ────────────────────────────────────────────────
        self.declare_parameter('save_dir',       '~/training_data/images/')
        self.declare_parameter('auto_mode',      True)
        self.declare_parameter('auto_interval',  3.0)   # seconds between auto-saves

        save_dir      = os.path.expanduser(
            self.get_parameter('save_dir').value)
        self.auto     = self.get_parameter('auto_mode').value
        interval      = self.get_parameter('auto_interval').value

        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        self.save_dir = save_dir

        # Count existing images so we don't overwrite old ones
        existing      = [f for f in os.listdir(save_dir)
                         if f.endswith('.jpg') or f.endswith('.png')]
        self.count    = len(existing)

        self.bridge   = CvBridge()
        self.latest   = None          # most recent frame
        self.lock     = threading.Lock()

        # ── Subscriber ────────────────────────────────────────────────
        self.sub = self.create_subscription(
            Image,
            '/panda/overhead_camera/image_raw',
            self._image_cb,
            10
        )

        # ── Auto-save timer ───────────────────────────────────────────
        if self.auto:
            self.create_timer(interval, self._auto_save)
            self.get_logger().info(
                f'\n{"="*55}\n'
                f'IMAGE COLLECTOR — AUTO MODE\n'
                f'  Save directory : {self.save_dir}\n'
                f'  Images so far  : {self.count}\n'
                f'  Saving every   : {interval} seconds\n'
                f'\n'
                f'  HOW TO USE:\n'
                f'  1. Gazebo is running with your objects on the table\n'
                f'  2. Every {interval}s a frame is saved automatically\n'
                f'  3. Between saves, MOVE OBJECTS in Gazebo to new positions\n'
                f'     (click and drag objects in the Gazebo window)\n'
                f'  4. Press Ctrl+C when done\n'
                f'{"="*55}'
            )
        else:
            # Manual mode — run keyboard listener in background thread
            self._manual_thread = threading.Thread(
                target=self._manual_listener, daemon=True)
            self._manual_thread.start()
            self.get_logger().info(
                f'\n{"="*55}\n'
                f'IMAGE COLLECTOR — MANUAL MODE\n'
                f'  Save directory : {self.save_dir}\n'
                f'  Images so far  : {self.count}\n'
                f'\n'
                f'  HOW TO USE:\n'
                f'  1. Arrange objects in Gazebo\n'
                f'  2. Switch to THIS terminal\n'
                f'  3. Press ENTER to save a frame\n'
                f'  4. Switch back to Gazebo, move objects, repeat\n'
                f'  5. Type q + ENTER to quit\n'
                f'{"="*55}'
            )

    # ──────────────────────────────────────────────────────────────────
    def _image_cb(self, msg: Image):
        """Store the latest camera frame."""
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        with self.lock:
            self.latest = frame

    # ──────────────────────────────────────────────────────────────────
    def _save_frame(self):
        """Save the current latest frame to disk."""
        with self.lock:
            if self.latest is None:
                self.get_logger().warn('No frame received yet — is Gazebo running?')
                return False
            frame = self.latest.copy()

        filename = os.path.join(self.save_dir, f'frame_{self.count:04d}.jpg')
        cv2.imwrite(filename, frame)
        self.count += 1
        self.get_logger().info(
            f'Saved: frame_{self.count-1:04d}.jpg  '
            f'(total: {self.count})  →  {self.save_dir}'
        )
        return True

    # ──────────────────────────────────────────────────────────────────
    def _auto_save(self):
        """Timer callback for auto mode — saves one frame."""
        self._save_frame()

    # ──────────────────────────────────────────────────────────────────
    def _manual_listener(self):
        """
        Runs in background thread for manual mode.
        Listens for ENTER key press in the terminal.
        Threading is needed because rclpy.spin() blocks the main thread.
        """
        print('\nManual mode active. Press ENTER to save. Type q+ENTER to quit.\n')
        while rclpy.ok():
            try:
                user_input = input()   # blocks until ENTER pressed
                if user_input.strip().lower() == 'q':
                    self.get_logger().info(
                        f'Done. Total images saved: {self.count}')
                    rclpy.shutdown()
                    break
                else:
                    self._save_frame()
            except (EOFError, KeyboardInterrupt):
                break


# ──────────────────────────────────────────────────────────────────────
def main(args=None):
    rclpy.init(args=args)
    node = ImageCollector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print(f'\nCollection stopped. Total images saved: {node.count}')
        print(f'Images location: {node.save_dir}')
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()