"""Perception 1: manually triggered dataset captures while autonomy continues."""

from pathlib import Path
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import Empty, String
from cave_explorer.perception1_dataset import (KNOWN_LABELS, clean_label,
                                                prepare_dataset, save_frame)


class DatasetCollector(Node):
    def __init__(self):
        super().__init__('dataset_collector')
        self.declare_parameter('dataset_directory', '~/cave_artefact_dataset')
        self.declare_parameter('dataset_label', 'negative')
        self.root, self.counts = prepare_dataset(self.get_parameter('dataset_directory').value)
        self.label = clean_label(self.get_parameter('dataset_label').value)
        self.pending = 0
        self.bridge = CvBridge()
        self.create_subscription(Image, 'camera/image', self.on_image, qos_profile_sensor_data)
        self.create_subscription(String, 'dataset_label', self.on_label, 10)
        self.create_subscription(Empty, 'dataset_save', self.on_save, 10)
        self.get_logger().info(f'Dataset ready at {self.root}; capture via /dataset_save')

    def on_label(self, message):
        label = clean_label(message.data)
        if label not in KNOWN_LABELS:
            self.get_logger().warn(f'Unexpected class {label}; permitted: {", ".join(KNOWN_LABELS)}')
            return
        self.label = label
        self.get_logger().info(f'Capture class set to {label}')

    def on_save(self, _message):
        
        self.pending = 1
        self.get_logger().info(f'Will capture next image as {self.label}')

    def on_image(self, message):
        if not self.pending:
            return
        try:
            bgr = self.bridge.imgmsg_to_cv2(message, desired_encoding='bgr8')
            path = save_frame(self.root, self.counts, bgr, self.label)
        except Exception as error:
            self.get_logger().error('Capture failed: ' + str(error))
            return
        self.pending = 0
        self.get_logger().info('Saved ' + str(path))


def main(args=None):
    rclpy.init(args=args)
    node = DatasetCollector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
