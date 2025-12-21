import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ament_index_python.packages import get_package_share_directory

from ultralytics import YOLO
import cv2
import os

from detection_interfaces.msg import DetectionMessage


class YoloSignDetector(Node):

    def __init__(self):
        super().__init__('yolo_sign_detector')

        self.bridge = CvBridge()

        self.subscription = self.create_subscription(
            Image,
            '/color/image',
            self.image_callback,
            50
        )

        pkg_share = get_package_share_directory('sign_detection')
        weights_path = os.path.join(
            pkg_share,
            'model_weights',
            'best.pt'
        )

        self.get_logger().info(f'Loading YOLO model from: {weights_path}')
        self.model = YOLO(weights_path)

        self.class_names = {
            0: 'left',
            1: 'right',
            2: 'construction'
        }

        self.conf_threshold = 0.80
        self.pub = self.create_publisher(
            DetectionMessage,
            '/detections',
            50
        )
        self.get_logger().info('YOLO sign detector started')

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        results = self.model(
            frame,
            conf=self.conf_threshold,
            verbose=False
        )
        if not results:
            return

        detections = results[0].boxes
        if detections is None:
            return

        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x_center = float(box.xywh[0][0]) / frame.shape[1]
            y_center = float(box.xywh[0][1]) / frame.shape[0]
            width = float(box.xywh[0][2]) / frame.shape[1]
            height = float(box.xywh[0][3]) / frame.shape[0]

            msg_out = DetectionMessage()
            msg_out.class_name = self.class_names.get(cls_id, 'unknown')
            msg_out.confidence = conf
            msg_out.x_center = x_center
            msg_out.y_center = y_center
            msg_out.width = width
            msg_out.height = height
            msg_out.stamp = self.get_clock().now().to_msg()

            self.pub.publish(msg_out)
            self.get_logger().info(
                f'Published: {msg_out.class_name} ({conf:.2f})'
            )

def main(args=None):
    rclpy.init(args=args)
    node = YoloSignDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
